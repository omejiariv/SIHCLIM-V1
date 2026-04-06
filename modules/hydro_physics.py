# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
import os
import io
import urllib.request
import ssl
import streamlit as st

# Importamos el interpolador maestro blindado
from modules.interpolation import interpolador_maestro

# --- 🚀 CLOUD SMART CACHE BLINDADO ---
@st.cache_data(show_spinner="📥 Descargando Raster desde Supabase...")
def download_raster_secure(url):
    """
    Descarga el raster asegurando que no haya bloqueos de SSL.
    """
    if not url or not isinstance(url, str) or not url.startswith("http"): 
        return url
        
    cache_dir = os.path.join(os.getcwd(), "data", "cloud_cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    local_path = os.path.join(cache_dir, filename)
    
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Error descargando {url}: {e}")
            return None
    return local_path

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
    Z_Interp, Z_Error = interpolador_maestro(
        df_puntos=gdf_puntos,
        col_val=columna_valor,
        grid_x=grid_x,
        grid_y=grid_y,
        metodo=method.lower(),
        dem_grid=dem_array
    )
    Z_Interp = np.nan_to_num(Z_Interp, nan=0)
    return np.maximum(Z_Interp, 0), Z_Error

# --- C. WARPING (NUBE Y LOCAL) ---
def _ejecutar_reproject(src, bounds, shape):
    """Función interna modularizada para re-proyectar cualquier fuente raster."""
    dst_crs = 'EPSG:4326'
    minx, miny, maxx, maxy = bounds
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, shape[1], shape[0])
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
    if np.isnan(destination).all(): return None
    return destination

def warper_raster_to_grid(raster_path, bounds, shape):
    """Lee un raster (ya sea desde URL o desde la RAM) y lo ajusta a la malla."""
    # 1. Si el archivo viene desde la RAM (Supabase BytesIO)
    if isinstance(raster_path, (io.BytesIO, bytes)) or hasattr(raster_path, 'read'):
        if hasattr(raster_path, 'seek'): raster_path.seek(0) # Rebobinar
        try:
            with MemoryFile(raster_path) as memfile:
                with memfile.open() as src:
                    return _ejecutar_reproject(src, bounds, shape)
        except Exception as e:
            print(f"Error warper_raster_to_grid (Memoria): {e}")
            return None
            
    # 2. Si el archivo es un string (Ruta local o URL)
    safe_path = download_raster_secure(raster_path)
    if not safe_path: return None
    try:
        with rasterio.open(safe_path) as src:
            return _ejecutar_reproject(src, bounds, shape)
    except Exception as e:
        print(f"Error warper_raster_to_grid (Local): {e}")
        return None

# --- D. MOTOR FÍSICO "EL ALEPH" ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    shape = grid_x.shape
    
    # --- 1. ELEVACIÓN Y TEMPERATURA ---
    Z_Alt = np.full_like(Z_P, 1500.0) 
    if paths.get('dem'):
        try:
            Z_dem_raw = warper_raster_to_grid(paths['dem'], bounds, shape)
            if Z_dem_raw is not None and np.any(~np.isnan(Z_dem_raw)):
                Z_Alt = np.nan_to_num(Z_dem_raw, nan=np.nanmean(Z_dem_raw))
        except: pass
    
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
    Z_Cob_Viz = None
    if paths.get('cobertura'):
        try:
            Z_Cob = warper_raster_to_grid(paths['cobertura'], bounds, shape)
            if Z_Cob is not None:
                Z_Cob_Viz = Z_Cob 
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

    # --- 7. DICCIONARIO FINAL ---
    resultados = {
        '1. Precipitación (mm/año)': Z_P,
        '2. Temperatura Media (°C)': Z_T,
        '3. Evapotranspiración Real (mm/año)': Z_ETR,
        '4. Elevación (msnm)': Z_Alt,
        '5. Escorrentía Superficial (mm/año)': Z_Q_Sup,
        '6. Infiltración (mm/año)': Z_Inf,
        '7. Recarga Potencial (mm/año)': Z_Inf,
        '8. Recarga Real (mm/año)': Z_Rec_Real,
        '9. Rendimiento Hídrico (L/s/km²)': Z_Rendimiento,
        '10. Susceptibilidad Erosión (Adim)': Z_Erosion
    }
    if Z_Cob_Viz is not None:
        resultados['11. Cobertura de Suelo (Clase)'] = Z_Cob_Viz

    return resultados
