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
    engine = get_engine()
    
    # A. Carga de Estaciones y Precipitación
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
            # B. Cálculos Hidrológicos (Turc)
            gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
            pts = df_data[['longitude', 'latitude']].values
            grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
            grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
            
            grid_T = np.maximum(5, 30 - (0.0065 * grid_Alt))
            L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
            grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
            grid_R = (grid_P - grid_ETR).clip(min=0)
            
            # Matriz de Priorización
            norm_R = grid_R / np.nanmax(grid_R) if np.nanmax(grid_R) > 0 else grid_R
            norm_Bio = grid_Alt / np.nanmax(grid_Alt) # Simplificado para bio
            grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)

            # C. RENDERIZADO DE PESTAÑAS (Indentado correctamente)
            tab_priorizacion, tab_analisis_hidro, tab_sigacal = st.tabs([
                "🎯 Priorización de Áreas", 
                "🗺️ Análisis Hidrológico", 
                "💧 Impacto SIGA-CAL (Río Grande)"
            ])

            with tab_priorizacion:
                st.subheader("Análisis Multicriterio de Priorización")
                col_m1, col_m2 = st.columns([3, 1])
                with col_m1:
                    fig = go.Figure(data=go.Contour(
                        z=grid_Final.T, 
                        x=np.linspace(minx, maxx, 100), 
                        y=np.linspace(miny, maxy, 100),
                        colorscale="RdYlGn",
                        colorbar=dict(title="Prioridad")
                    ))
                    fig.update_layout(height=500, margin=dict(l=0,r=0,b=0,t=0))
                    st.plotly_chart(fig, use_container_width=True)
                with col_m2:
                    st.metric("Prioridad Media", f"{np.nanmean(grid_Final):.2f}")
                    st.write("**Leyenda:**")
                    st.write("- 🟢 Baja (Estable)")
                    st.write("- 🟡 Media (Vigilancia)")
                    st.write("- 🔴 Alta (Intervención)")

            with tab_analisis_hidro:
                st.subheader("🌊 Balance Hídrico Local (Sihcli-Poter)")
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_bal = go.Figure(data=[
                        go.Bar(name='Oferta (Ppt)', x=['Balance'], y=[np.nanmean(grid_P)], marker_color='#3498db'),
                        go.Bar(name='Pérdida (ETR)', x=['Balance'], y=[np.nanmean(grid_ETR)], marker_color='#e67e22'),
                        go.Bar(name='Recarga (R)', x=['Balance'], y=[np.nanmean(grid_R)], marker_color='#2ecc71')
                    ])
                    st.plotly_chart(fig_bal, use_container_width=True)
                with ch2:
                    st.write("#### 📝 Análisis de Rendimiento")
                    rendimiento = (np.nanmean(grid_R) / np.nanmean(grid_P)) * 100
                    st.metric("Rendimiento Hídrico", f"{rendimiento:.1f}%")
                    st.info("Este porcentaje representa la fracción de lluvia que se convierte en agua disponible.")

            with tab_sigacal:
                context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))
                render_sigacal_analysis(gdf_predios=context_layers.get('predios'))

        else:
            st.warning("Datos insuficientes para interpolar.")
else:
    st.info("👈 Seleccione una zona en el menú lateral.")

