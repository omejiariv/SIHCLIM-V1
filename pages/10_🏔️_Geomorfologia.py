import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import plotly.express as px
import os
import pydeck as pdk
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
def cargar_y_cortar_dem(ruta_dem, _gdf_corte):
    """
    Corta el DEM grande usando la geometría seleccionada.
    """
    if _gdf_corte is None or _gdf_corte.empty:
        return None, None, None

    try:
        if not os.path.exists(ruta_dem):
            return None, None, None

        with rasterio.open(ruta_dem) as src:
            crs_dem = src.crs
            # Usamos el argumento con guion bajo
            gdf_proyectado = _gdf_corte.to_crs(crs_dem)
            geoms = gdf_proyectado.geometry.values
            
            out_image, out_transform = mask(src, geoms, crop=True)
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            dem_array = out_image[0]
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            dem_array = np.where(dem_array < -100, np.nan, dem_array)

            return dem_array, out_meta, out_transform

    except Exception as e:
        st.error(f"Error técnico procesando el DEM: {e}")
        return None, None, None
        
# --- 3. LÓGICA PRINCIPAL ---

@st.cache_data
def generar_mapa_3d(arr_elev, transform):
    """
    Genera una nube de puntos simplificada para visualización 3D en PyDeck.
    """
    # Submuestreo para rendimiento (max 100x100 puntos para fluidez)
    h, w = arr_elev.shape
    factor = max(1, int(max(h, w) / 100))
    
    # Crear malla de coordenadas
    rows, cols = np.indices(arr_elev.shape)
    elevs = arr_elev
    
    # Aplicar submuestreo
    rows = rows[::factor, ::factor].flatten()
    cols = cols[::factor, ::factor].flatten()
    elevs = elevs[::factor, ::factor].flatten()
    
    # Filtrar NaNs
    mask_valid = ~np.isnan(elevs)
    rows = rows[mask_valid]
    cols = cols[mask_valid]
    elevs = elevs[mask_valid]
    
    # Convertir índices pixel a coordenadas reales (EPSG:3116 -> Lat/Lon)
    # Nota: PyDeck necesita Lat/Lon. Aquí haremos una aproximación o reproyección.
    # Para simplificar hoy, usaremos un mapa de calor 3D sobre el mapa base.
    
    # ESTRATEGIA: Usar 'TerrainLayer' de PyDeck es complejo sin un servidor de teselas.
    # Usaremos 'ColumnLayer' (Hexágonos) que es más robusto para datos locales.
    
    # Transformación afín para obtener X, Y (Metros)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    df_3d = pd.DataFrame({
        "x": xs,
        "y": ys,
        "elev": elevs
    })
    
    # Convertir a Lat/Lon (Necesitamos pyproj)
    # Si no tienes pyproj instalado, esto fallará. 
    # ¿Tienes pyproj en requirements.txt? Si no, usaremos Plotly 3D que es más fácil.
    return df_3d

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
                st.subheader(f"Modelo Digital de Elevación 3D: {nombre_zona}")
                
                # --- VISUALIZACIÓN 3D INTERACTIVA (PLOTLY) ---
                import plotly.graph_objects as go

                # 1. Submuestreo inteligente (Downsampling)
                # Esto es vital: si intentamos graficar 1 millón de puntos, el navegador colapsa.
                # Calculamos un factor para tener aprox. una malla de 150x150 puntos, que se ve HD y es rápida.
                h, w = arr_elevacion.shape
                factor = max(1, int(max(h, w) / 150))
                
                # Creamos la versión ligera para el gráfico
                arr_3d = arr_elevacion[::factor, ::factor]
                
                # 2. Crear el Gráfico de Superficie (Surface Plot)
                fig_surf = go.Figure(data=[go.Surface(z=arr_3d, colorscale='Earth')])
                
                fig_surf.update_layout(
                    title=f"Topografía 3D - {nombre_zona}",
                    autosize=True,
                    width=800, 
                    height=600,
                    scene=dict(
                        xaxis_title='Oeste - Este',
                        yaxis_title='Sur - Norte',
                        zaxis_title='Altitud (m)',
                        # aspectmode='auto' ajusta la caja para que se vea bien visualmente
                        aspectmode='auto' 
                    ),
                    margin=dict(l=65, r=50, b=65, t=90)
                )
                
                st.plotly_chart(fig_surf, use_container_width=True)
                
                st.info(f"💡 Usa el mouse para rotar, acercar y explorar el relieve. (Factor de optimización: 1 píxel de cada {factor})")
                
            with tab2:
                st.info("Aquí irá la Curva Hipsométrica Integrada.")
                
            with tab3:
                st.info("Aquí procesaremos el DEM con PySheds para obtener ríos.")

else:
    st.info("👈 Por favor selecciona una Cuenca o Municipio en la barra lateral para iniciar el análisis.")
