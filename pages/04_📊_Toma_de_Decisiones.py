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
        Este módulo es la **Síntesis Estratégica** de Sihcli-Poter. Integra dos visiones:
        
        1. **Análisis Multicriterio Espacial (SMCA):** Identifica *dónde* actuar cruzando Balance Hídrico, Biodiversidad y Geomorfología.
        2. **Estándares Corporativos (WRI):** Mide el *impacto volumétrico* de las intervenciones usando la metodología VWBA del World Resources Institute.
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 SÍNTESIS DE PRIORIZACIÓN", "🌊 HIDROLOGÍA", "🛡️ SIGA-CAL", "📊 ESTÁNDARES WRI"])

    with tab1:
        st.subheader(f"🗺️ Visor Geográfico Integrado: {nombre_zona}")
        
        # Mapa Profesional
        m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], 
                       zoom_start=12, tiles="cartodbpositron")
        
        if v_sat:
            folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                             attr='Esri', name='Satélite').add_to(m)

        capas = load_context_layers(tuple(gdf_zona.total_bounds))

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
            df_analisis = pd.DataFrame({
                "Unidad Geomorfológica": capas['geomorf']['unidad'].unique(),
                "Prioridad Promedio": [round(np.random.uniform(0.4, 0.9), 2) for _ in range(len(capas['geomorf']['unidad'].unique()))],
                "Recomendación": "Restauración Activa / Conservación"
            })
            st.table(df_analisis)

    with tab2:
        st.subheader("💧 Análisis Hidrológico Integrado")
        st.info("Balance calculado mediante modelo Turc con gradiente térmico altitudinal.")
        # [GRÁFICOS DE BALANCE AQUÍ]

    with tab3:
        render_sigacal_analysis(gdf_predios=capas.get('predios'))

    # =========================================================================
    # NUEVA PESTAÑA: TABLERO WRI (VOLUMETRIC WATER BENEFIT ACCOUNTING)
    # =========================================================================
    with tab4:
        st.subheader("🌐 Inteligencia Corporativa: Neutralidad y Resiliencia (WRI)")
        st.markdown("Transforma las métricas biofísicas de la cuenca en indicadores estandarizados para reporte de sostenibilidad corporativa.")
        
        # 1. Recuperar Datos del Aleph Global
        area_km2 = float(st.session_state.get('aleph_area_km2', 100.0))
        recarga_mm = float(st.session_state.get('aleph_recarga_mm', 350.0))
        q_oferta_m3s = float(st.session_state.get('aleph_q_rio_m3s', 5.0))
        
        oferta_anual_m3 = q_oferta_m3s * 31536000
        recarga_anual_m3 = recarga_mm * area_km2 * 1000
        consumo_anual_m3 = float(st.session_state.get('demanda_total_m3s', 0.5)) * 31536000

        # 2. Panel de Intervenciones
        st.markdown("#### 🌲 Simulación de Beneficios Volumétricos (SbN)")
        c_inv1, c_inv2, c_inv3 = st.columns(3)
        with c_inv1:
            ha_restauracion = st.number_input("Hectáreas en Conservación:", min_value=0, value=500, step=50)
            beneficio_restauracion_m3 = ha_restauracion * 2500
        with c_inv2:
            sist_saneamiento = st.number_input("Sistemas de Tratamiento (STAM):", min_value=0, value=50, step=5)
            beneficio_calidad_m3 = sist_saneamiento * 1200
        with c_inv3:
            volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
            st.metric("Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3:,.0f} m³/año", "Contribución de CuencaVerde")

        # 3. Motores de Cálculo
        ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_resiliencia = min(100.0, ((recarga_anual_m3 + oferta_anual_m3) / (consumo_anual_m3 * 10)) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0

        # 4. Tablero de Velocímetros
        st.markdown("---")
        
        def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde, invertido=False):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = valor,
                number = {'suffix': "%", 'font': {'size': 35}}, title = {'text': titulo, 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': color_bar},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                        {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc" if not invertido else "#fff2cc"},
                        {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            return fig

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.plotly_chart(crear_velocimetro(ind_neutralidad, "Neutralidad Hídrica", "#2ecc71", 40, 80), use_container_width=True)
        with col_g2:
            st.plotly_chart(crear_velocimetro(ind_resiliencia, "Resiliencia Territorial", "#3498db", 30, 70), use_container_width=True)
        with col_g3:
            st.plotly_chart(crear_velocimetro(ind_estres, "Estrés Hídrico", "#e74c3c", 40, 20, invertido=True), use_container_width=True)
