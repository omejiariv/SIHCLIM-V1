# pages/03_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import Rbf
import os
import sys

# --- IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
    from modules import db_manager, selectors
    from modules.utils import encender_gemelo_digital
    try:
        from modules.data_processor import complete_series
    except ImportError:
        complete_series = None
except:
    from modules import db_manager, selectors
    from modules.config import Config
    complete_series = None

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")
encender_gemelo_digital()
st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# --- 2. FICHA TÉCNICA ---
with st.expander("📘 Ficha Técnica: Metodología, Utilidad y Fuentes", expanded=False):
    st.markdown("""
    ### 1. Metodología: Interpolación RBF
    Se utiliza el algoritmo **RBF (Radial Basis Function)** con núcleo *Thin-Plate Spline*. Este método genera una superficie suave que respeta los valores exactos de las estaciones.
    ### 2. Capas Dinámicas
    El sistema permite superponer el mapa de **Cuencas** y **Municipios** sobre la mancha de lluvia para identificar zonas de estrés hídrico.
    """)

# --- 3. SELECTOR ESPACIAL GLOBAL ---
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

if not ids_sel or gdf_zona is None or gdf_zona.empty:
    st.info("👈 Seleccione un Territorio en el menú lateral para iniciar.")
    st.stop()

# --- 4. FUNCIONES DE SOPORTE GIS ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [os.path.join("data", filename), os.path.join("..", "data", filename)]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                return gdf.to_crs("EPSG:4326")
            except: continue
    return None

def detectar_columna(df, keywords):
    if df is None or df.empty: return None
    cols = df.columns.tolist()
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower(): return c
    return None

def add_mapbox_context_layers(fig, show_cuencas, show_municipios):
    """Agrega capas vectoriales sobre el mapa de Mapbox."""
    if show_municipios:
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            for _, r in gdf_m.iterrows():
                if r.geometry.geom_type == 'Polygon': polys = [r.geometry]
                else: polys = list(r.geometry.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=1, color='gray'), name="Municipios", hoverinfo='skip', showlegend=False))

    if show_cuencas:
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cu is not None:
            for _, r in gdf_cu.iterrows():
                if r.geometry.geom_type == 'Polygon': polys = [r.geometry]
                else: polys = list(r.geometry.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=2, color='blue'), name="Cuencas", hoverinfo='skip', showlegend=False))

# --- 5. SIDEBAR: CONFIGURACIÓN ---
st.sidebar.header("⚙️ Configuración del Mapa")

# A. Modos de Análisis
tipo_analisis = st.sidebar.selectbox("📊 Modo de Análisis:", ["Año Específico", "Promedio Multianual", "Variabilidad Temporal", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"])

# B. Estética y Capas (LO NUEVO)
st.sidebar.subheader("🗺️ Capas y Estilo")
map_style = st.sidebar.selectbox("Mapa de Fondo:", ["Street Map (Claro)", "Satelital", "Dark Matter (Oscuro)", "Open Street Map"], index=0)
ver_cuencas = st.sidebar.checkbox("✅ Ver Capa de Cuencas", value=True)
ver_municipios = st.sidebar.checkbox("🏙️ Ver Capa de Municipios", value=False)

style_dict = {
    "Street Map (Claro)": "carto-positron",
    "Satelital": "stamen-terrain", # O usar tiles de ESRI abajo
    "Dark Matter (Oscuro)": "carto-darkmatter",
    "Open Street Map": "open-street-map"
}

# Parámetros adicionales
buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 100, 25)
suavidad = st.sidebar.slider("🖌️ Suavizado (RBF):", 0.0, 2.0, 0.1)

# --- 6. LÓGICA DE DATOS ---
bounds = gdf_zona.total_bounds
buffer_deg = buffer_km / 111.0
q_minx, q_miny, q_maxx, q_maxy = bounds[0]-buffer_deg, bounds[1]-buffer_deg, bounds[2]+buffer_deg, bounds[3]+buffer_deg

tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])

with tab_mapa:
    try:
        engine = db_manager.get_engine()
        q_raw = text("SELECT p.id_estacion, p.fecha, p.valor, e.latitud, e.longitud, e.nombre FROM precipitacion p JOIN estaciones e ON p.id_estacion = e.id_estacion WHERE e.longitud BETWEEN :mx AND :Mx AND e.latitud BETWEEN :my AND :My")
        df_raw = pd.read_sql(q_raw, engine, params={"mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
        
        if not df_raw.empty:
            df_raw['year'] = pd.to_datetime(df_raw['fecha']).dt.year
            # Simplificación: Promedio por estación (según modo)
            if tipo_analisis == "Año Específico":
                y_sel = st.sidebar.selectbox("📅 Año:", range(2025, 1990, -1))
                df_final = df_raw[df_raw['year'] == y_sel].groupby(['id_estacion','nombre','latitud','longitud'])['valor'].sum().reset_index()
            else:
                df_final = df_raw.groupby(['id_estacion','nombre','latitud','longitud'])['valor'].mean().reset_index()

            if len(df_final) >= 3:
                # 📐 INTERPOLACIÓN
                grid_res = 150
                x, y, z = df_final['longitud'].values, df_final['latitud'].values, df_final['valor'].values
                xi = np.linspace(q_minx, q_maxx, grid_res)
                yi = np.linspace(q_miny, q_maxy, grid_res)
                XI, YI = np.meshgrid(xi, yi)
                
                rbf = Rbf(x, y, z, function='thin_plate', smooth=suavidad)
                ZI = rbf(XI, YI)
                ZI = np.maximum(ZI, 0) # No lluvias negativas

                # --- 7. RENDERIZADO MAPBOX ---
                fig = go.Figure()

                # A. Mancha de Isoyetas (Contourmapbox)
                fig.add_trace(go.Contourmapbox(
                    lon=xi, lat=yi, z=ZI,
                    colorscale="YlGnBu", opacity=0.6,
                    contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                    colorbar=dict(title="mm")
                ))

                # B. Capas de Contexto (Cuencas/Municipios)
                add_mapbox_context_layers(fig, ver_cuencas, ver_municipios)

                # C. Puntos de Estaciones
                fig.add_trace(go.Scattermapbox(
                    lon=df_final['longitud'], lat=df_final['latitud'],
                    mode='markers', marker=dict(size=8, color='black', opacity=0.8),
                    text=df_final['nombre'] + ": " + df_final['valor'].astype(int).astype(str) + " mm",
                    name="Estaciones"
                ))

                # D. Configuración de Layout
                center_lat = gdf_zona.centroid.y.mean()
                center_lon = gdf_zona.centroid.x.mean()

                fig.update_layout(
                    title=f"Isoyetas: {tipo_analisis} | {nombre_zona}",
                    mapbox=dict(
                        style=style_dict[map_style],
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=10
                    ),
                    margin=dict(l=0, r=0, t=40, b=0), height=700
                )

                # Si eligió Satélite, inyectamos la capa de ESRI (opcional si stamen-terrain no carga)
                if map_style == "Satelital":
                    fig.update_layout(mapbox_layers=[{
                        "below": 'traces', "sourcetype": 'raster',
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                    }])

                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            else:
                st.warning("⚠️ Datos insuficientes. Aumente el buffer de búsqueda.")
    except Exception as e:
        st.error(f"Error en visualización: {e}")

# --- 8. PESTAÑA DESCARGAS (Blindada) ---
with tab_datos:
    if 'df_final' in locals() and not df_final.empty:
        st.subheader("💾 Exportar Resultados")
        st.dataframe(df_final.head(50), use_container_width=True)
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Descargar Datos (CSV)", csv, "isoyetas_datos.csv", "text/csv")
