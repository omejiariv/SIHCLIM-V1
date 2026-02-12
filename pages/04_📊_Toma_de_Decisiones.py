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

# 1. IMPORTACIÓN DEL NUEVO MÓDULO
try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
except ImportError:
    st.error("No se encontró el módulo impacto_serv_ecosist en la carpeta modules.")

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
    files = {
        'municipios': "MunicipiosAntioquia.geojson", 
        'cuencas': "SubcuencasAinfluencia.geojson", 
        'predios': "PrediosEjecutados.geojson"
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
    engine = get_engine()
    
    # A. Carga de datos
    q_est = text("SELECT id_estacion, nombre, latitud, longitud, altitud FROM estaciones")
    df_est = pd.read_sql(q_est, engine)
    df_est = df_est.rename(columns={'latitud': 'latitude', 'longitud': 'longitude', 'altitud': 'alt_est'})
    for c in ['latitude', 'longitude', 'alt_est']: df_est[c] = pd.to_numeric(df_est[c], errors='coerce')
    df_est = df_est.dropna(subset=['latitude', 'longitude']).copy()
    
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    mask = (df_est['longitude'].between(minx-0.1, maxx+0.1)) & (df_est['latitude'].between(miny-0.1, maxy+0.1))
    df_filt = df_est[mask].copy()

    if not df_filt.empty:
        ids_v = ",".join([f"'{x}'" for x in df_filt['id_estacion'].unique()])
        q_ppt = text(f"SELECT id_estacion, AVG(valor)*12 as p_anual FROM precipitacion WHERE id_estacion IN ({ids_v}) GROUP BY id_estacion")
        df_ppt = pd.read_sql(q_ppt, engine)
        df_data = pd.merge(df_filt, df_ppt, on='id_estacion')

        if len(df_data) >= 3:
            # B. Cálculos científicos (Sihcli-Poter)
            gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
            pts = df_data[['longitude', 'latitude']].values
            grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
            grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
            
            grid_T = np.maximum(5, 30 - (0.0065 * grid_Alt))
            L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
            grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
            grid_R = (grid_P - grid_ETR).clip(min=0)
            
            norm_R = grid_R / np.nanmax(grid_R) if np.nanmax(grid_R) > 0 else grid_R
            norm_Bio = grid_Alt / np.nanmax(grid_Alt)
            grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)

            # C. RENDERIZADO DE PESTAÑAS
            tab_priorizacion, tab_analisis_hidro, tab_sigacal = st.tabs([
                "🎯 Priorización de Áreas", 
                "🗺️ Análisis Hidrológico", 
                "💧 Impacto SIGA-CAL (Río Grande)"
            ])

            with tab_priorizacion:
                st.subheader("🗺️ Síntesis Geoespacial de Priorización")
                
                # Mapa Base Integrado (Lógica de visualizer)
                m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=11, tiles="CartoDB positron")
                folium.TileLayer(
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri', name='Satélite (Esri)'
                ).add_to(m)

                # Capas de Contexto
                context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))
                
                if context_layers['cuencas'] is not None:
                    folium.GeoJson(context_layers['cuencas'], name="🌊 Red de Drenaje", 
                                   style_function=lambda x: {'color': '#2980b9', 'weight': 1.5}).add_to(m)
                
                if context_layers['predios'] is not None:
                    folium.GeoJson(context_layers['predios'], name="🏡 Predios CuencaVerde",
                                   style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange', 'fillOpacity': 0.6}).add_to(m)

                folium.LayerControl().add_to(m)
                st_folium(m, width="100%", height=500, key="mapa_sintesis")

                # Diagnóstico Integrado
                with st.expander("🧐 Diagnóstico Integrado de la Zona", expanded=True):
                    col_inf1, col_inf2 = st.columns(2)
                    with col_inf1:
                        st.markdown(f"""
                        **Análisis Hidro-Climático:**
                        * **Precipitación Media:** {np.nanmean(grid_P):.0f} mm/año.
                        * **Rendimiento Estimado:** {(np.nanmean(grid_R)/np.nanmean(grid_P))*100:.1f}% de la lluvia.
                        """)
                    with col_inf2:
                        st.markdown(f"""
                        **Evaluación SMCA:**
                        * **Score Prioridad:** {np.nanmean(grid_Final):.2f}/1.00.
                        * **Factor Dominante:** {"Hídrico" if pct_agua > pct_bio else "Biodiversidad"}.
                        """)

            with tab_analisis_hidro:
                st.subheader("🌊 Balance Hídrico Local (Sihcli-Poter)")
                ch1, ch2 = st.columns([2, 1])
                with ch1:
                    fig_bal = go.Figure(data=[
                        go.Bar(name='Oferta (Ppt)', x=['Balance'], y=[np.nanmean(grid_P)], marker_color='#3498db'),
                        go.Bar(name='Pérdida (ETR)', x=['Balance'], y=[np.nanmean(grid_ETR)], marker_color='#e67e22'),
                        go.Bar(name='Recarga (R)', x=['Balance'], y=[np.nanmean(grid_R)], marker_color='#2ecc71')
                    ])
                    fig_bal.update_layout(height=400, title="Distribución de Masa de Agua (mm/año)")
                    st.plotly_chart(fig_bal, use_container_width=True)
                with ch2:
                    st.write("#### 📝 Resumen del Balance")
                    st.write(f"La evapotranspiración real (ETR) consume el **{(np.nanmean(grid_ETR)/np.nanmean(grid_P))*100:.1f}%** de la lluvia recibida.")
                    st.metric("Rendimiento Hídrico", f"{np.nanmean(grid_R):.1f} mm")

            with tab_sigacal:
                # Llamada al módulo externo
                render_sigacal_analysis(gdf_predios=context_layers.get('predios'))

        else:
            st.warning("⚠️ Datos insuficientes para generar el análisis (Mínimo 3 estaciones).")
else:
    st.info("👈 Seleccione una zona en el menú lateral para iniciar el análisis.")

