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
    tab_priorizacion, tab_analisis_hidro, tab_sigacal = st.tabs([
        "🎯 Priorización de Áreas", 
        "🗺️ Análisis Hidrológico", 
        "💧 Impacto SIGA-CAL (Río Grande)"
    ])

    engine = get_engine()
    try:
        q_est = text("SELECT id_estacion, nombre, latitud, longitud, altitud FROM estaciones")
        df_est = pd.read_sql(q_est, engine)
        
        if not df_est.empty:
            df_est = df_est.rename(columns={'latitud': 'latitude', 'longitud': 'longitude', 'altitud': 'alt_est'})
            for c in ['latitude', 'longitude', 'alt_est']: df_est[c] = pd.to_numeric(df_est[c], errors='coerce')
            df_est = df_est.dropna(subset=['latitude', 'longitude'])
            df_est['alt_est'] = df_est['alt_est'].fillna(alt_ref)
            
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            margin = 0.05
            mask = ((df_est['longitude'] >= minx-margin) & (df_est['longitude'] <= maxx+margin) & 
                    (df_est['latitude'] >= miny-margin) & (df_est['latitude'] <= maxy+margin))
            df_filt = df_est[mask].copy()
            
            if not df_filt.empty:
                ids_v = df_filt['id_estacion'].astype(str).str.strip().unique()
                ids_s = ",".join([f"'{x}'" for x in ids_v])
                q_ppt = text(f"SELECT id_estacion, AVG(valor)*12 as p_anual FROM precipitacion WHERE id_estacion IN ({ids_s}) GROUP BY id_estacion")
                df_ppt = pd.read_sql(q_ppt, engine)
                df_data = pd.merge(df_filt, df_ppt, on='id_estacion', how='inner')
            else:
                df_data = pd.DataFrame()

            # --- RENDERIZADO DE PESTAÑAS ---
            if len(df_data) >= 3:
                with tab_priorizacion:
                    # Lógica de cálculo (Turc y SMCA)
                    gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                    pts = df_data[['longitude', 'latitude']].values
                    grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
                    grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
                    grid_T = np.maximum(5, 30 - (0.0065 * grid_Alt))
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    grid_R = (grid_P - grid_ETR).clip(min=0)
                    norm_R = grid_R / np.nanmax(grid_R) if np.nanmax(grid_R) > 0 else grid_R
                    norm_Bio = ((grid_Alt * 0.7) + (grid_P * 0.3)) / np.nanmax((grid_Alt * 0.7) + (grid_P * 0.3))
                    
                    grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    grid_Final = np.where(grid_Final >= (umbral/100.0), grid_Final, np.nan)

                    # Dibujar Mapa Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Contour(z=grid_Final.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100), colorscale="RdYlGn", opacity=0.7))
                    
                    context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))
                    # (Aquí iría la función add_poly_layer que ya tienes para pintar municipios/predios)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab_analisis_hidro:
                    st.info("Aquí puedes añadir mapas específicos de Escurrimiento o Evapotranspiración.")
                    st.write(f"Precipitación Promedio en la zona: {df_data['p_anual'].mean():.1f} mm/año")

                with tab_sigacal:
                    # 3. LLAMADA AL MÓDULO EXTERNO
                    # Le pasamos los predios recortados para que el mapa sea coherente
                    render_sigacal_analysis(gdf_predios=context_layers.get('predios'))

            else:
                st.warning("⚠️ Datos insuficientes para el análisis espacial.")

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👈 Seleccione una zona en el menú lateral.")

