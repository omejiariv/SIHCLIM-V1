# modules/dem_extractor.py

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import streamlit as st
from modules.config import Config
from modules.hydro_physics import download_raster_secure

def completar_altitudes_con_dem(df_estaciones, col_lat='latitud', col_lon='longitud', col_alt='elevacion'):
    """
    Toma un DataFrame de estaciones, identifica las que NO tienen altitud (NaN o None),
    y extrae el valor exacto cruzando sus coordenadas con el DEM del proyecto.
    """
    # 1. Asegurar que la columna de altitud sea numérica para poder buscar los nulos
    df_estaciones[col_alt] = pd.to_numeric(df_estaciones[col_alt], errors='coerce')
    mascara_nulos = df_estaciones[col_alt].isna()
    
    # Si todas las estaciones ya tienen altitud, devolvemos el dataframe intacto
    if not mascara_nulos.any():
        return df_estaciones 
        
    cantidad_faltantes = mascara_nulos.sum()
    print(f"🏔️ Detectadas {cantidad_faltantes} estaciones sin altitud. Iniciando rescate espacial con el DEM...")
    
    # 2. Obtener la ruta segura del DEM (funciona tanto en local como en Supabase)
    ruta_dem_segura = download_raster_secure(Config.DEM_FILE_PATH)
    
    if not ruta_dem_segura:
        print("⚠️ Advertencia: No se pudo acceder al DEM. Las altitudes quedarán nulas.")
        return df_estaciones
        
    # 3. Crear un GeoDataFrame SOLO con las estaciones que necesitan rescate (para optimizar memoria)
    df_faltantes = df_estaciones[mascara_nulos].copy()
    gdf_faltantes = gpd.GeoDataFrame(
        df_faltantes, 
        geometry=gpd.points_from_xy(df_faltantes[col_lon], df_faltantes[col_lat]),
        crs="EPSG:4326"  # Asumimos que las coordenadas base vienen en Grados (GPS)
    )
    
    # 4. Abrir el DEM y realizar el cruce espacial
    try:
        with rasterio.open(ruta_dem_segura) as src:
            crs_dem = src.crs
            print(f"   -> Reproyectando puntos al vuelo al CRS del DEM: {crs_dem}")
            
            # ¡EL TRUCO VITAL! Alinear los puntos al sistema de coordenadas del mapa
            gdf_proyectado = gdf_faltantes.to_crs(crs_dem)
            
            # Extraer las tuplas de coordenadas (x, y)
            coordenadas = [(geom.x, geom.y) for geom in gdf_proyectado.geometry]
            
            # Muestrear el raster con esas coordenadas
            altitudes_rescatadas = []
            for val in src.sample(coordenadas):
                alt = val[0]
                
                # Filtrar valores NoData (ej. -32768) o caídas fuera del mapa
                if alt == src.nodata or alt < -500:
                    altitudes_rescatadas.append(np.nan)
                else:
                    altitudes_rescatadas.append(round(float(alt), 1))
                    
            # 5. Inyectar las altitudes rescatadas de vuelta al DataFrame original
            df_estaciones.loc[mascara_nulos, col_alt] = altitudes_rescatadas
            
        vacios_restantes = df_estaciones[col_alt].isna().sum()
        print(f"✅ Rescate completado con éxito. Estaciones que cayeron fuera de la cobertura del DEM: {vacios_restantes}")
        
    except Exception as e:
        print(f"❌ Error crítico durante el cruce espacial: {e}")
        
    return df_estaciones
