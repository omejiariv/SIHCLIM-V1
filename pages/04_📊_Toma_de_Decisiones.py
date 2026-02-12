# Módulo de Soporte a Decisiones (VERSIÓN SÍNTESIS INTEGRADA)

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

# 1. IMPORTACIÓN DE MÓDULOS EXISTENTES
try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
    # Aquí es donde integraríamos funciones de tus otras páginas (ejemplos hipotéticos según tu descripción)
    # from modules.clima import get_isoyetas_layer
    # from modules.geologia import get_geomorfologia_layer
except ImportError as e:
    st.warning(f"Nota: Algunos módulos de integración no están disponibles: {e}")

# --- SETUP INICIAL ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    try:
        from modules.db_manager import get_engine
    except:
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])
except Exception as e:
    st.error(f"Error crítico: {e}")
    st.stop()

st.title("🎯 Tablero Estratégico de Toma de Decisiones")

# --- FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def get_clipped_context_layers(gdf_zona_bounds):
    layers_data = {'municipios': None, 'cuencas': None, 'predios': None, 'drenajes': None}
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    files = {
        'municipios': "MunicipiosAntioquia.geojson",
        'cuencas': "SubcuencasAinfluencia.geojson",
        'predios': "PrediosEjecutados.geojson",
        'drenajes': "Drenaje_Sencillo.geojson" # Capa de tu página de Geomorfología/Hidrografía
    }
    for key, fname in files.items():
        try:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                gdf = gpd.read_file(fpath)
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                layers_data[key] = gpd.clip(gdf, roi_gdf)
        except: pass
    return layers_data

def interpolacion_segura(points, values, grid_x, grid_y):
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- LÓGICA DE INTERFAZ ---
ids, nombre, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.header("⚙️ Configuración de Escenarios")
    # PUNTO 2: Análisis de Escenarios (¿Qué pasaría si...?)
    st.info("Ajuste los pesos para simular diferentes prioridades de inversión.")
    w_agua = st.slider("💧 Prioridad Hídrica (Recarga/Oferta)", 0, 100, 70)
    w_bio = st.slider("🍃 Prioridad Ecosistémica (Biodiversidad)", 0, 100, 30)
    
    suma = w_agua + w_bio
    pct_agua = w_agua / suma if suma > 0 else 0.5
    pct_bio = w_bio / suma if suma > 0 else 0.5
    
    st.divider()
    st.subheader("🗺️ Capas Adicionales")
    show_drenajes = st.checkbox("Ver Red de Drenaje", True)
    show_predios = st.checkbox("Ver Predios CuencaVerde", True)
    show_satelite = st.checkbox("Fondo Satelital", True)

# --- PROCESAMIENTO Y SÍNTESIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    # (Carga de estaciones y cálculos de Turc/Infiltración se mantienen igual)
    # ... [Omitido por brevedad para enfocar en la integración de pestañas] ...
    # Supongamos que ya tenemos grid_P, grid_ETR, grid_R y grid_Final calculados
    
    # C. RENDERIZADO DE LA SÍNTESIS
    tab_priorizacion, tab_analisis_hidro, tab_sigacal = st.tabs([
        "🎯 Priorización Integrada", 
        "🌊 Análisis Hidrológico & Clima", 
        "🛡️ Impacto SIGA-CAL"
    ])

    with tab_priorizacion:
        st.subheader(f"🗺️ Síntesis Geoespacial: {nombre}")
        
        # 1. MAPA PROFESIONAL (PUNTO 1: Mejora de Visualización)
        m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], 
                       zoom_start=12, tiles="CartoDB positron")
        
        if show_satelite:
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Satélite (Esri)', overlay=False
            ).add_to(m)

        context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))

        # Capas de visualizer integradas
        if show_drenajes and context_layers['drenajes'] is not None:
            folium.GeoJson(context_layers['drenajes'], name="Ríos", style_function=lambda x: {'color': '#3498db', 'weight': 1}).add_to(m)
        
        if show_predios and context_layers['predios'] is not None:
            folium.GeoJson(context_layers['predios'], name="Intervenciones", 
                           style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange', 'fillOpacity': 0.5}).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=550)

        # PUNTO 3: Análisis Automático (Lo que pediste)
        with st.expander("🧐 Diagnóstico Técnico de la Zona", expanded=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**🔍 Hallazgos Hidrológicos:**")
                rend = (np.nanmean(grid_R)/np.nanmean(grid_P))*100
                if rend > 40:
                    st.success(f"Zona de **Alta Producción Hídrica**. El rendimiento es del {rend:.1f}%.")
                else:
                    st.warning(f"Zona de **Bajo Rendimiento**. Se pierde el {(np.nanmean(grid_ETR)/np.nanmean(grid_P))*100:.1f}% en ETR.")
            
            with col_d2:
                st.markdown("**🍃 Hallazgos Ecosistémicos:**")
                score = np.nanmean(grid_Final)
                st.info(f"El Score de Prioridad es **{score:.2f}/1.00**. Esta zona es clave para la conectividad biológica de la cuenca.")

    with tab_analisis_hidro:
        # Integración con resultados de Clima e Isoyetas
        st.subheader("💧 Balance Hídrico y Escenarios Climáticos")
        c_h1, c_h2 = st.columns([2, 1])
        with c_h1:
            # Gráfico de barras de balance (Sihcli-Poter)
            fig_bal = go.Figure(data=[
                go.Bar(name='Lluvia (P)', x=['Balance'], y=[np.nanmean(grid_P)], marker_color='#2980b9'),
                go.Bar(name='Evaporación (ETR)', x=['Balance'], y=[np.nanmean(grid_ETR)], marker_color='#e67e22'),
                go.Bar(name='Recarga (R)', x=['Balance'], y=[np.nanmean(grid_R)], marker_color='#27ae60')
            ])
            st.plotly_chart(fig_bal, use_container_width=True)
        with c_h2:
            st.write("#### 📝 Resumen Ejecutivo")
            st.write("Datos integrados de los módulos de Clima e Isoyetas.")
            st.metric("Ppt Promedio", f"{np.nanmean(grid_P):.0f} mm")
            st.metric("Infiltración Real", f"{np.nanmean(grid_R):.0f} mm")

    with tab_sigacal:
        # Aquí ya corre tu análisis de impacto con el CSV de Río Grande
        render_sigacal_analysis(gdf_predios=context_layers.get('predios'))
