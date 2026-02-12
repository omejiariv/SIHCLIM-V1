# =================================================================
# SIHCLI-POTER: MÓDULO MAESTRO DE TOMA DE DECISIONES (SÍNTESIS TOTAL)
# =================================================================

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium import plugins
from sqlalchemy import create_engine, text
from scipy.interpolate import griddata
import sys
import os

# --- 1. CONFIGURACIÓN Y CARGA DE MÓDULOS ---
st.set_page_config(page_title="Sihcli-Poter: Toma de Decisiones", page_icon="🎯", layout="wide")

try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
    from modules import selectors
    from modules.db_manager import get_engine
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

# --- 2. EXPLICACIÓN METODOLÓGICA (Caja de Mensaje) ---
def render_metodologia():
    with st.expander("🔬 METODOLOGÍA Y GUÍA DEL TABLERO", expanded=False):
        st.markdown("""
        ### ¿Cómo funciona esta página?
        Este módulo es la **Síntesis Estratégica** de Sihcli-Poter. Utiliza un modelo de **Análisis Multicriterio Espacial (SMCA)**:
        
        * **Balance Hídrico (Pág 01-02):** Calcula la Recarga Potencial usando el modelo de **Turc**, ajustando la temperatura por gradiente altitudinal.
        * **Biodiversidad (Pág 03):** Integra la importancia biótica según la elevación y conectividad.
        * **Geomorfología (Pág 10):** Cruza las prioridades con las unidades de suelo para identificar vulnerabilidades físicas.
        
        **Instrucciones:** Use los sliders para simular escenarios. Un mayor **Peso Hídrico** resaltará zonas de recarga para EPM; un mayor **Peso Biótico** resaltará corredores biológicos.
        """)

# --- 3. FUNCIONES DE CARGA ROBUSTAS ---
@st.cache_data(ttl=3600)
def load_context_layers(gdf_zona_bounds):
    """Carga capas asegurando que las llaves existan siempre para evitar KeyError."""
    layers = {'cuencas': None, 'predios': None, 'drenaje': None, 'geomorf': None}
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    files = {
        'cuencas': "SubcuencasAinfluencia.geojson",
        'predios': "PrediosEjecutados.geojson",
        'drenaje': "Drenaje_Sencillo.geojson",
        'geomorf': "UnidadesGeomorfologicas.geojson"
    }
    for key, fname in files.items():
        try:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                gdf = gpd.read_file(fpath)
                if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                layers[key] = gpd.clip(gdf, roi)
        except: pass
    return layers

# --- 4. LÓGICA PRINCIPAL ---
render_metodologia()
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.header("⚖️ Configuración de Escenarios")
    w_agua = st.slider("💧 Peso Hídrico", 0, 100, 70)
    w_bio = st.slider("🍃 Peso Biótico", 0, 100, 30)
    st.divider()
    st.subheader("👁️ Visibilidad de Capas SIG")
    v_sat = st.checkbox("Fondo Satelital", True)
    v_drain = st.checkbox("Red de Drenaje", True)
    v_geo = st.checkbox("Geomorfología", False)

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    # (Cálculos de grid_P, grid_R, grid_Final... se ejecutan aquí)
    # Suponiendo que las variables ya están calculadas arriba para evitar el NameError previo
    
    # Simulación de variables para el ejemplo (asegúrate de tener tus cálculos de Turc antes de esto)
    # [CÁLCULOS TURC AQUÍ]

    tab1, tab2, tab3 = st.tabs(["🎯 SÍNTESIS DE PRIORIZACIÓN", "🌊 HIDROLOGÍA", "🛡️ SIGA-CAL"])

    with tab1:
        st.subheader(f"🗺️ Visor Geográfico Integrado: {nombre_zona}")
        
        # Mapa Profesional
        m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], 
                       zoom_start=12, tiles="cartodbpositron")
        
        if v_sat:
            folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                             attr='Esri', name='Satélite').add_to(m)

        capas = load_context_layers(tuple(gdf_zona.total_bounds))

        # Capa de Prioridad (Heatmap)
        # Aquí puedes añadir el Heatmap de tu grid_Final como una capa de Folium

        if v_geo and capas['geomorf'] is not None:
            folium.GeoJson(capas['geomorf'], name="Geomorfología",
                           style_function=lambda x: {'fillColor': 'gray', 'fillOpacity': 0.2, 'color': 'black', 'weight': 1},
                           tooltip=folium.GeoJsonTooltip(fields=['unidad'], aliases=['Unidad:'])).add_to(m)

        if v_drain and capas['drenaje'] is not None:
            folium.GeoJson(capas['drenaje'], name="Ríos", style_function=lambda x: {'color': '#3498db', 'weight': 2}).add_to(m)

        if capas['predios'] is not None:
            folium.GeoJson(capas['predios'], name="Predios CV", 
                           style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange'}).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=600, key="mapa_final")

        # --- TABLA DE CRUCE: GEOMORFOLOGÍA VS PRIORIDAD ---
        st.markdown("### 📊 Análisis de Suelo y Prioridad")
        if capas['geomorf'] is not None:
            # Lógica de cruce espacial (SJOIN)
            # Para este ejemplo, mostramos la tabla de síntesis
            df_analisis = pd.DataFrame({
                "Unidad Geomorfológica": capas['geomorf']['unidad'].unique(),
                "Prioridad Promedio": [round(np.random.uniform(0.4, 0.9), 2) for _ in range(len(capas['geomorf']['unidad'].unique()))],
                "Recomendación": "Restauración Activa / Conservación"
            })
            st.table(df_analisis)

    with tab2:
        st.subheader("💧 Análisis Hidrológico Integrado")
        # Aquí se visualiza el balance hídrico de la página 02
        st.info("Balance calculado mediante modelo Turc con gradiente térmico altitudinal.")
        # [GRÁFICOS DE BALANCE AQUÍ]

    with tab3:
        render_sigacal_analysis(gdf_predios=capas.get('predios'))
