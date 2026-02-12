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

# 1. IMPORTACIONES DE MÓDULOS DEL SISTEMA
try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
    from modules import selectors
    try:
        from modules.db_manager import get_engine
    except:
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

# --- SETUP ---
st.set_page_config(page_title="Sihcli-Poter: Toma de Decisiones", page_icon="🎯", layout="wide")

# --- FUNCIONES AUXILIARES DE INTEGRACIÓN ---
@st.cache_data(ttl=3600)
def get_integrated_layers(gdf_zona_bounds):
    """Carga capas de contexto desde el repositorio de datos compartido."""
    layers = {'municipios': None, 'cuencas': None, 'predios': None, 'drenajes': None, 'geomorf': None}
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Mapeo de archivos (usando los nombres de tus otros módulos)
    files = {
        'municipios': "MunicipiosAntioquia.geojson",
        'cuencas': "SubcuencasAinfluencia.geojson",
        'predios': "PrediosEjecutados.geojson",
        'drenajes': "Drenaje_Sencillo.geojson" # Vinculado a Geomorfología
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

def interpolacion_segura(points, values, grid_x, grid_y):
    z = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(z)
    if np.any(mask):
        z_near = griddata(points, values, (grid_x, grid_y), method='nearest')
        z[mask] = z_near[mask]
    return z

# --- INTERFAZ ---
st.title("🎯 Tablero Estratégico de Síntesis")
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.header("⚖️ Escenarios de Simulación")
    w_agua = st.slider("💧 Peso Hídrico (Oferta/Recarga)", 0, 100, 70)
    w_bio = st.slider("🍃 Peso Biótico (Conectividad)", 0, 100, 30)
    st.divider()
    st.subheader("🛠️ Capas Maestras")
    show_sat = st.checkbox("Mapa Satelital (Esri)", True)
    show_hydro = st.checkbox("Red de Drenaje (Geomorfología)", True)
    show_inter = st.checkbox("Predios CuencaVerde", True)

# --- NÚCLEO DE CÁLCULO E INTEGRACIÓN ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    
    # 1. Carga de datos base (Clima + Estaciones)
    q = text("SELECT id_estacion, nombre, latitud as latitude, longitud as longitude, altitud as alt_est FROM estaciones")
    df_est = pd.read_sql(q, engine)
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    df_filt = df_est[df_est['longitude'].between(minx-0.1, maxx+0.1) & df_est['latitude'].between(miny-0.1, maxy+0.1)].copy()

    if len(df_filt) >= 3:
        # Cruce con lluvia mensualizada (Página Clima)
        ids_s = ",".join([f"'{x}'" for x in df_filt['id_estacion'].unique()])
        q_p = text(f"SELECT id_estacion, AVG(valor)*12 as p_anual FROM precipitacion WHERE id_estacion IN ({ids_s}) GROUP BY id_estacion")
        df_ppt = pd.read_sql(q_p, engine)
        df_data = pd.merge(df_filt, df_ppt, on='id_estacion')

        # 2. MODELACIÓN HIDROLÓGICA (Sihcli-Poter Core)
        gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
        pts = df_data[['longitude', 'latitude']].values
        grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
        grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
        
        # Modelo Turc (Página 02 - Aguas Subterráneas)
        grid_T = np.maximum(5, 30 - (0.0065 * grid_Alt))
        L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
        grid_R = (grid_P - grid_ETR).clip(min=0) # Recarga Potencial
        
        # SMCA - Priorización
        norm_R = grid_R / np.nanmax(grid_R) if np.nanmax(grid_R) > 0 else grid_R
        norm_Bio = grid_Alt / np.nanmax(grid_Alt) # Proxy conectividad altitudinal
        grid_Final = (norm_R * (w_agua/100)) + (norm_Bio * (w_bio/100))

        # 3. RENDERIZADO DE PESTAÑAS INTEGRADAS
        tab_prior, tab_hydro, tab_sigacal = st.tabs(["🎯 Priorización Social y Técnica", "🌊 Hidrología & Clima", "🛡️ Impacto SIGA-CAL"])

        with tab_prior:
            st.subheader("🗺️ Síntesis Espacial del Territorio")
            
            # Mapa Folium (Página 10 - Geomorfología Style)
            m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], zoom_start=12, tiles="CartoDB positron")
            if show_sat:
                folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                                 attr='Esri', name='Satélite').add_to(m)

            layers = get_integrated_layers(tuple(gdf_zona.total_bounds))
            
            if show_hydro and layers['drenajes'] is not None:
                folium.GeoJson(layers['drenajes'], name="Drenajes", style_function=lambda x: {'color': '#3498db', 'weight': 1}).add_to(m)
            
            if show_inter and layers['predios'] is not None:
                folium.GeoJson(layers['predios'], name="Predios", style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange'}).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=500, key="mapa_prior_final")

            # --- DIAGNÓSTICO INTEGRADO ---
            st.markdown("### 🧐 Diagnóstico Estratégico")
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🌊 Factores Hidrológicos:**")
                    rend = (np.nanmean(grid_R)/np.nanmean(grid_P))*100
                    st.write(f"• Rendimiento Hídrico: {rend:.1f}%")
                    if rend > 40: st.success("✅ Zona identificada como 'Fábrica de Agua' crítica.")
                    else: st.warning("⚠️ Zona con alta pérdida por Evapotranspiración.")
                
                with col2:
                    st.markdown("**🍃 Factores Ecosistémicos:**")
                    score_p = np.nanmean(grid_Final)
                    st.write(f"• Score de Prioridad: {score_p:.2f} / 1.0")
                    if w_bio > 50: st.info("ℹ️ Escenario enfocado en conectividad biológica.")

        with tab_hydro:
            # Integración de Balance (Página 01 y 02)
            st.subheader("💧 Balance Hídrico (Sihcli-Poter)")
            fig_b = go.Figure(data=[
                go.Bar(name='Lluvia', x=['Balance'], y=[np.nanmean(grid_P)], marker_color='#2980b9'),
                go.Bar(name='Evaporación', x=['Balance'], y=[np.nanmean(grid_ETR)], marker_color='#e67e22'),
                go.Bar(name='Recarga', x=['Balance'], y=[np.nanmean(grid_R)], marker_color='#27ae60')
            ])
            fig_b.update_layout(height=400, barmode='group', title="Distribución de Masa de Agua (mm/año)")
            st.plotly_chart(fig_b, use_container_width=True)

        with tab_sigacal:
            # Impacto en Calidad de Agua (SIGA-CAL)
            render_sigacal_analysis(gdf_predios=layers.get('predios'))

    else:
        st.warning("⚠️ Se requieren al menos 3 estaciones climáticas en la zona para generar la síntesis.")
else:
    st.info("👈 Seleccione una cuenca o municipio en el panel izquierdo para iniciar el análisis.")
