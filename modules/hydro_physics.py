# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os
from modules.interpolation import interpolador_maestro

# --- A. BASE DE CONOCIMIENTO (Sin Cambios) ---
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbanos
    211: 0.60, 231: 0.50, 241: 0.45, # Agrícolas
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05,            # Herbazales/Páramo
    'default': 0.50
}

# --- B. INTERPOLACIÓN AVANZADA (IDW) ---
def idw_interpolation(x, y, z, grid_x, grid_y, power=2):
    """
    Interpolación Inverse Distance Weighting (IDW).
    Elimina el efecto de 'líneas rectas' y 'triángulos'.
    """
    # Aplanamos las coordenadas de la grilla
    xi = grid_x.flatten()
    yi = grid_y.flatten()
    
    # Coordenadas de las estaciones
    xi_st = x
    yi_st = y
    zi_st = z
    
    # Calculamos distancias (Broadcasting)
    # Distancia entre cada pixel y cada estación
    dist = np.sqrt((xi[:, None] - xi_st[None, :])**2 + (yi[:, None] - yi_st[None, :])**2)
    
    # Evitar división por cero (si un pixel cae exacto en una estación)
    dist = np.where(dist == 0, 1e-10, dist)
    
    # Pesos
    weights = 1.0 / dist**power
    
    # Suma ponderada
    z_interp = np.sum(weights * zi_st, axis=1) / np.sum(weights, axis=1)
    
    return z_interp.reshape(grid_x.shape)

def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y, method='kriging', dem_array=None):
    """
    Función puente que llama al interpolador maestro de modules/interpolation.py
    """
    # Llamamos a la función que acabamos de crear/potenciar
    Z_Interp, Z_Error = interpolador_maestro(
        df_puntos=gdf_puntos,
        col_val=columna_valor,
        grid_x=grid_x,
        grid_y=grid_y,
        metodo=method,
        dem_grid=dem_array # Pasamos el DEM por si se usa KED
    )
    
    # Saneamiento final para física
    Z_Interp = np.nan_to_num(Z_Interp, nan=0)
    return np.maximum(Z_Interp, 0), Z_Error


# --- C. WARPING (Sin Cambios) ---
def warper_raster_to_grid(raster_path, bounds, shape):
    """
    Lee un raster (sin importar su proyección original) y lo fuerza 
    a encajar en la malla WGS84 (Lat/Lon) de la aplicación.
    
    Args:
        raster_path: Ruta al archivo .tif (puede estar en Magna Sirgas EPSG:3116)
        bounds: (minx, miny, maxx, maxy) en WGS84 (Grados)
        shape: (height, width) dimensiones de la malla de destino
    """
    if not raster_path: return None
    
    try:
        dst_crs = 'EPSG:4326' # La App SIEMPRE trabaja en WGS84
        minx, miny, maxx, maxy = bounds
        
        with rasterio.open(raster_path) as src:
            # Calculamos la transformación matemática para ir de 
            # las coordenadas del archivo original -> a la caja de la App.
            dst_transform = rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, shape[1], shape[0]
            )
            
            # Preparamos la matriz vacía
            destination = np.zeros(shape, dtype=np.float32)

            # --- AQUÍ OCURRE LA MAGIA ---
            # Rasterio se encarga de torcer y estirar la imagen (Warping)
            # desde metros (src.crs) hasta grados (dst_crs)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                # Nodata handling: rellena vacíos con NaN, no con ceros
                dst_nodata=np.nan 
            )
            
            # Saneamiento final: Si todo es NaN, algo falló en la intersección
            if np.isnan(destination).all():
                # print("Advertencia: El raster no se intersecta con la zona seleccionada.")
                return None
                
            return destination

    except Exception as e:
        # print(f"Error crítico en warper: {e}")
        return None


# --- D. MOTOR FÍSICO ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    """
    Motor Físico Blindado Aleph: 
    Elimina Infinitos y NaNs en el balance hídrico para prevenir el colapso de la interfaz.
    """
    import numpy as np
    shape = grid_x.shape
    
    # --- 1. ELEVACIÓN Y TEMPERATURA ---
    # Escudo: Altitud por defecto si el DEM falla
    Z_Alt = np.full_like(Z_P, 1500.0) 
    if paths.get('dem'):
        try:
            # Saneamiento del DEM para evitar NaNs en la temperatura
            Z_dem_raw = physics.warper_raster_to_grid(paths['dem'], bounds, shape)
            if np.any(~np.isnan(Z_dem_raw)):
                Z_Alt = np.nan_to_num(Z_dem_raw, nan=np.nanmean(Z_dem_raw))
        except: pass
    
    # Escudo Térmico: Temperatura mínima de 1.0°C para evitar división por cero en Turc
    Z_T = np.maximum(28.0 - (0.006 * Z_Alt), 1.0)

    # --- 2. ETR (TURC BLINDADO CONTRA INFINITOS) ---
    # L(T) = 300 + 25T + 0.05T^3 -> Nunca será 0 por el escudo de Z_T anterior
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Limitamos el ratio P/L a 500 para evitar que el cuadrado desborde a 'inf'
        ratio_seguro = np.clip(Z_P / L, 0, 500) 
        denom = np.sqrt(0.9 + (ratio_seguro)**2)
        # Saneamiento inmediato: ETR no puede ser mayor que la lluvia (P)
        Z_ETR = np.nan_to_num(np.minimum(Z_P / denom, Z_P), nan=0.0)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # --- 3. PENDIENTE (Saneamiento de celdas contra división por cero) ---
    scale_factor = 111000 
    dy, dx = np.gradient(Z_Alt)
    
    # Evitamos división por cero si el grid es demasiado pequeño o plano
    cell_size_y = np.maximum(np.abs(grid_y[1,0] - grid_y[0,0]) * scale_factor, 1.0)
    cell_size_x = np.maximum(np.abs(grid_x[0,1] - grid_x[0,0]) * scale_factor, 1.0)
    
    Z_Slope_Pct = np.nan_to_num(np.sqrt((dy/cell_size_y)**2 + (dx/cell_size_x)**2), nan=0.0)

    # --- 4. COBERTURA Y ESCORRENTÍA (C) ---
    Z_C = np.full_like(Z_P, 0.45) 
    if paths.get('cobertura'):
        try:
            Z_Cob = physics.warper_raster_to_grid(paths['cobertura'], bounds, shape)
            def map_c(code): return physics.CLC_C_BASE.get(int(code), 0.50)
            vfunc = np.vectorize(map_c)
            Z_C = vfunc(np.nan_to_num(Z_Cob, nan=0))
        except: pass

    # Escudo: El coeficiente C_Mod debe estar entre 0.05 y 0.95 (límites físicos reales)
    Z_C_Mod = np.clip(Z_C + (Z_Slope_Pct * 0.2), 0.05, 0.95)
    Z_Q_Sup = Z_Exc * Z_C_Mod

    # --- 5. INFILTRACIÓN Y RECARGA ---
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    Z_Rec_Real = Z_Inf * 0.3 
    Z_Rendimiento = (Z_Q_Sup + Z_Rec_Real) * 10 

    # --- 6. EROSIÓN (Blindaje de Inversa de Cobertura) ---
    # Z_C_Inv nunca será 0 gracias al np.clip de Z_C_Mod anterior
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.05)
    Z_Erosion = np.nan_to_num((Z_P * 0.5) * 0.3 * (1 + Z_Slope_Pct * 5) * (1.0 / Z_C_Inv), nan=0.0)
    Z_Rec_Potencial = Z_Inf  # Infiltración total disponible
    Z_Rec_Real = Z_Inf * 0.3 # Estimación de recarga neta al acuífero

    # --- 7. RETORNO DE MATRICES LIMPIAS Y LIGERAS ---
 
    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T, 'ETR': Z_ETR, 
        'Q': Z_Q_Sup, 'Infiltracion': Z_Inf, 
        'Recarga_Potencial': Z_Rec_Potencial, # <-- Nueva
        'Recarga_Real': Z_Rec_Real,          # <-- Nueva
        'Rendimiento': Z_Rendimiento, 
        'Erosion': Z_Erosion, 
        'C_Escorrentia': Z_C_Mod
    }