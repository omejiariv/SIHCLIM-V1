# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os
from modules.interpolation import interpolador_maestro

# --- A. BASE DE CONOCIMIENTO ---
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbanos
    211: 0.60, 231: 0.50, 241: 0.45, # Agrícolas
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05,            # Herbazales/Páramo
    'default': 0.50
}

# --- B. INTERPOLACIÓN PUENTE ---
def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y, method='kriging', dem_array=None):
    """
    Función puente que llama al interpolador maestro.
    """
    Z_Interp, Z_Error = interpolador_maestro(
        df_puntos=gdf_puntos,
        col_val=columna_valor,
        grid_x=grid_x,
        grid_y=grid_y,
        metodo=method,
        dem_grid=dem_array
    )
    
    Z_Interp = np.nan_to_num(Z_Interp, nan=0)
    return np.maximum(Z_Interp, 0), Z_Error

# --- C. WARPING (LECTURA DE RASTERS) ---
def warper_raster_to_grid(raster_path, bounds, shape):
    """
    Lee un raster y lo fuerza a encajar en la malla WGS84 (Lat/Lon).
    """
    if not raster_path: return None
    
    try:
        dst_crs = 'EPSG:4326'
        minx, miny, maxx, maxy = bounds
        
        with rasterio.open(raster_path) as src:
            dst_transform = rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, shape[1], shape[0]
            )
            
            destination = np.zeros(shape, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan 
            )
            
            if np.isnan(destination).all():
                return None
                
            return destination

    except Exception as e:
        return None


# --- D. MOTOR FÍSICO "EL ALEPH" ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    """
    Calcula el balance hídrico distribuido.
    RETORNA: Diccionario con CLAVES LARGAS Y NUMERADAS para visualización directa.
    """
    import numpy as np
    shape = grid_x.shape
    
    # --- 1. ELEVACIÓN Y TEMPERATURA ---
    Z_Alt = np.full_like(Z_P, 1500.0) 
    if paths.get('dem'):
        try:
            Z_dem_raw = warper_raster_to_grid(paths['dem'], bounds, shape)
            if Z_dem_raw is not None and np.any(~np.isnan(Z_dem_raw)):
                Z_Alt = np.nan_to_num(Z_dem_raw, nan=np.nanmean(Z_dem_raw))
        except: pass
    
    # Gradiente Térmico
    Z_T = np.maximum(28.0 - (0.006 * Z_Alt), 1.0)

    # --- 2. ETR (TURC) ---
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_seguro = np.clip(Z_P / L, 0, 500) 
        denom = np.sqrt(0.9 + (ratio_seguro)**2)
        Z_ETR = np.nan_to_num(np.minimum(Z_P / denom, Z_P), nan=0.0)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # --- 3. PENDIENTE ---
    scale_factor = 111000 
    dy, dx = np.gradient(Z_Alt)
    cell_size_y = np.maximum(np.abs(grid_y[1,0] - grid_y[0,0]) * scale_factor, 1.0)
    cell_size_x = np.maximum(np.abs(grid_x[0,1] - grid_x[0,0]) * scale_factor, 1.0)
    Z_Slope_Pct = np.nan_to_num(np.sqrt((dy/cell_size_y)**2 + (dx/cell_size_x)**2), nan=0.0)

    # --- 4. COBERTURA Y ESCORRENTÍA ---
    Z_C = np.full_like(Z_P, 0.45) 
    Z_Cob_Viz = None # Para visualización

    if paths.get('cobertura'):
        try:
            Z_Cob = warper_raster_to_grid(paths['cobertura'], bounds, shape)
            if Z_Cob is not None:
                Z_Cob_Viz = Z_Cob # Guardamos para mostrar
                def map_c(code): return CLC_C_BASE.get(int(code), 0.50)
                vfunc = np.vectorize(map_c)
                Z_C = vfunc(np.nan_to_num(Z_Cob, nan=0))
        except: pass

    Z_C_Mod = np.clip(Z_C + (Z_Slope_Pct * 0.2), 0.05, 0.95)
    Z_Q_Sup = Z_Exc * Z_C_Mod

    # --- 5. INFILTRACIÓN Y RECARGA ---
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    Z_Rec_Real = Z_Inf * 0.3 
    Z_Rendimiento = (Z_Q_Sup + Z_Rec_Real) * 10 

    # --- 6. EROSIÓN ---
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.05)
    Z_Erosion = np.nan_to_num((Z_P * 0.5) * 0.3 * (1 + Z_Slope_Pct * 5) * (1.0 / Z_C_Inv), nan=0.0)

    # --- 7. DICCIONARIO FINAL (CLAVES AMIGABLES) ---
    resultados = {
        '1. Precipitación (mm/año)': Z_P,
        '2. Temperatura Media (°C)': Z_T,
        '3. Evapotranspiración Real (mm/año)': Z_ETR,
        '4. Elevación (msnm)': Z_Alt,
        '5. Escorrentía Superficial (mm/año)': Z_Q_Sup,
        '6. Infiltración (mm/año)': Z_Inf,
        '7. Recarga Potencial (mm/año)': Z_Inf,  # Asumimos Inf = Recarga Potencial
        '8. Recarga Real (mm/año)': Z_Rec_Real,
        '9. Rendimiento Hídrico (L/s/km²)': Z_Rendimiento,
        '10. Susceptibilidad Erosión (Adim)': Z_Erosion
    }

    # Agregamos Cobertura solo si se pudo procesar
    if Z_Cob_Viz is not None:
        resultados['11. Cobertura de Suelo (Clase)'] = Z_Cob_Viz

    return resultados
