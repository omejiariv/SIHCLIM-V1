# Módulo de Soporte a Decisiones (INTEGRADO CON SIGA-CAL)

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from scipy.interpolate import griddata
import sys
import os

# 1. IMPORTACIÓN DEL NUEVO MÓDULO
try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
except ImportError:
    st.error("No se encontró el módulo impacto_Serv_Ecosist en la carpeta modules.")

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
    st.error(f"Error crítico de importación: {e}")
    st.stop()

st.title("🎯 Priorización y Análisis de Impacto")

# --- FUNCIONES AUXILIARES ---
@st.cache_data(ttl=3600)
def get_clipped_context_layers(gdf_zona_bounds):
    layers_data = {'municipios': None, 'cuencas': None, 'predios': None}
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi_poly = box(minx, miny, maxx, maxy)
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:4326")
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    files = {'municipios': "MunicipiosAntioquia.geojson", 'cuencas': "SubcuencasAinfluencia.geojson", 'predios': "PrediosEjecutados.geojson"}
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

# --- INTERFAZ DE FILTROS ---
ids, nombre, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.header("⚖️ Criterios de Priorización")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    suma = w_agua + w_bio
    pct_agua = w_agua / suma if suma > 0 else 0.5
    pct_bio = w_bio / suma if suma > 0 else 0.5
    st.divider()
    umbral = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0)

# --- LÓGICA PRINCIPAL ---
if gdf_zona is not None and not gdf_zona.empty:
    # 2. CREACIÓN DE PESTAÑAS (Se crean aquí para que estén disponibles)
# Definimos las pestañas para organizar el análisis
tab_priorizacion, tab_analisis_hidro, tab_sigacal = st.tabs([
    "🎯 Priorización de Áreas", 
    "🗺️ Análisis Hidrológico", 
    "💧 Impacto SIGA-CAL (Río Grande)"
])

with tab_priorizacion:
    st.subheader("Análisis Multicriterio de Priorización")
    c1, c2 = st.columns([3, 1])
    
    with c1:
        # Aquí va tu mapa Plotly (go.Contour) que ya tienes
        # Asegúrate de incluir una leyenda o colorbar
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.write("#### ⚖️ Leyenda y Score")
        st.write("**Rojo:** Prioridad Crítica")
        st.write("**Verde:** Prioridad Baja (Conservada)")
        st.metric("Score de la Zona", f"{np.nanmean(grid_Final):.2f}")
        st.write(f"Basado en {pct_agua*100:.0f}% agua y {pct_bio*100:.0f}% bio.")

with tab_analisis_hidro:
    st.subheader("🌊 Balance Hídrico Local (Sihcli-Poter)")
    ch1, ch2 = st.columns(2)
    
    with ch1:
        # Gráfico de barras de balance
        fig_bal = go.Figure(data=[
            go.Bar(name='Oferta (Ppt)', x=['Balance'], y=[np.nanmean(grid_P)]),
            go.Bar(name='Pérdida (ETR)', x=['Balance'], y=[np.nanmean(grid_ETR)]),
            go.Bar(name='Recarga (R)', x=['Balance'], y=[np.nanmean(grid_R)])
        ])
        st.plotly_chart(fig_bal, use_container_width=True)
        
    with ch2:
        st.write("#### 📝 Análisis de Rendimiento")
        rendimiento = (np.nanmean(grid_R) / np.nanmean(grid_P)) * 100
        st.write(f"El rendimiento hídrico de esta zona es del **{rendimiento:.1f}%**.")
        st.info("Zonas con rendimiento > 40% son fábricas de agua críticas para EPM.")

with tab_sigacal:
    # Llamamos al módulo que ya importaste arriba
    render_sigacal_analysis(gdf_predios=context_layers.get('predios'))
