import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import plotly.express as px
import os
from modules import selectors

# Configuración de Página
st.set_page_config(page_title="Geomorfología Avanzada", page_icon="🏔️", layout="wide")

st.title("🏔️ Análisis Geomorfológico y Terreno 3D")
st.markdown("""
Esta herramienta utiliza el **Modelo Digital de Elevación (DEM)** para modelar el terreno, 
calcular pendientes y definir redes de drenaje.
""")

# --- 1. BARRA LATERAL (SELECTOR) ---
# Reutilizamos tu selector robusto que ya filtra por Región/Cuenca/Municipio
ids, nombre_zona, alt_ref, gdf_zona_seleccionada = selectors.render_selector_espacial()

# --- 2. CARGA DEL DEM (RASTER) ---
# Ruta del archivo (Ajusta la ruta si está en una subcarpeta 'data' o 'rasters')
DEM_PATH = os.path.join("data", "DemAntioquia_EPSG3116.tif")

@st.cache_data(show_spinner="Cortando DEM...")
def cargar_y_cortar_dem(ruta_dem, gdf_corte):
    """
    Corta el DEM grande usando la geometría seleccionada.
    Maneja la reproyección de coordenadas automáticamente.
    """
    if gdf_corte is None or gdf_corte.empty:
        return None, None, None

    try:
        with rasterio.open(ruta_dem) as src:
            # 1. Verificar CRS del DEM y del Vector
            crs_dem = src.crs
            
            # 2. Reproyectar el Vector al sistema del DEM (EPSG:3116 - Metros)
            # Esto es vital para que el corte coincida
            gdf_proyectado = gdf_corte.to_crs(crs_dem)
            
            # 3. Obtener geometría para la máscara
            geoms = gdf_proyectado.geometry.values
            
            # 4. Cortar (Masking)
            # crop=True elimina los bordes negros sobrantes
            out_image, out_transform = mask(src, geoms, crop=True)
            
            # 5. Metadatos del nuevo recorte
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # El array viene como (bandas, alto, ancho). Usamos la banda 1.
            dem_array = out_image[0]
            
            # Filtrar valores "No Data" (a veces vienen como -9999)
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            # Filtrar valores absurdos (ej: elevación < 0 en montaña)
            dem_array = np.where(dem_array < 0, np.nan, dem_array)

            return dem_array, out_meta, out_transform

    except Exception as e:
        st.error(f"Error procesando el DEM: {e}")
        return None, None, None

# --- 3. LÓGICA PRINCIPAL ---

if gdf_zona_seleccionada is not None:
    # Verificación de archivo
    if not os.path.exists(DEM_PATH):
        st.error(f"⚠️ No encuentro el archivo DEM en: {DEM_PATH}")
        st.info("Por favor verifica que el archivo 'DemAntioquia_EPSG3116.tif' esté en la carpeta 'data'.")
    else:
        # Procesar DEM
        arr_elevacion, meta, transform = cargar_y_cortar_dem(DEM_PATH, gdf_zona_seleccionada)
        
        if arr_elevacion is not None:
            # Estadísticas Básicas
            min_el = np.nanmin(arr_elevacion)
            max_el = np.nanmax(arr_elevacion)
            mean_el = np.nanmean(arr_elevacion)
            
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Elevación Mínima", f"{min_el:.0f} m.s.n.m")
            c2.metric("Elevación Máxima", f"{max_el:.0f} m.s.n.m")
            c3.metric("Elevación Media", f"{mean_el:.0f} m.s.n.m")
            c4.metric("Rango Altitudinal", f"{max_el - min_el:.0f} m")

            # --- PESTAÑAS DE ANÁLISIS ---
            tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Elevación", "📈 Hipsometría", "🌊 Red de Drenaje (Beta)"])
            
            with tab1:
                st.subheader(f"Modelo Digital de Elevación: {nombre_zona}")
                # Visualización rápida 2D con Plotly (Heatmap)
                # Submuestreo para velocidad (toma 1 de cada 5 píxeles) si es muy grande
                factor = 1 if arr_elevacion.shape[0] < 1000 else 5
                fig_dem = px.imshow(
                    arr_elevacion[::factor, ::factor], 
                    color_continuous_scale='Earth',
                    title="Mapa de Altitudes (Vista Plana)"
                )
                st.plotly_chart(fig_dem, use_container_width=True)
                
            with tab2:
                st.info("Aquí irá la Curva Hipsométrica Integrada.")
                
            with tab3:
                st.info("Aquí procesaremos el DEM con PySheds para obtener ríos.")

else:
    st.info("👈 Por favor selecciona una Cuenca o Municipio en la barra lateral para iniciar el análisis.")
