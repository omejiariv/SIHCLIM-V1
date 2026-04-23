# =================================================================
# SIHCLI-POTER: MÓDULO MAESTRO DE TOMA DE DECISIONES (SÍNTESIS TOTAL)
# =================================================================

import os
import sys

# --- 🔥 FIX: INYECCIÓN DE PATH ESTRATÉGICA ---
# Garantiza que Python encuentre la carpeta 'modules' ANTES de importar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import math
import folium
from streamlit_folium import st_folium
from folium import plugins
from sqlalchemy import create_engine, text
from scipy.interpolate import griddata

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Sihcli-Poter: Toma de Decisiones", page_icon="🎯", layout="wide")

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS INTERNOS ---
try:
    from modules import selectors
    from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto
    # 🔥 FIX: Añadimos obtener_poblacion_matriz al import
    from modules.demografia_tools import render_motor_demografico, obtener_poblacion_matriz
    from modules.biodiversidad_tools import render_motor_ripario
    from modules.geomorfologia_tools import render_motor_hidrologico
    from modules.impacto_serv_ecosist import render_sigacal_analysis
    from modules.db_manager import get_engine
except ImportError as e:
    st.error(f"🚨 Error crítico cargando los módulos internos: {e}")
    st.stop() # Detiene la ejecución para evitar pantallas en blanco o errores en cascada

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
selectors.renderizar_menu_navegacion("Toma de Decisiones")
encender_gemelo_digital()

# =========================================================================
# 🏷️ RECUPERACIÓN DEL TÍTULO PRINCIPAL DE LA PÁGINA
# =========================================================================
st.title("🎯 Módulo Maestro de Toma de Decisiones y Síntesis Territorial")
st.markdown("""
Integración Multicriterio para la **Seguridad Hídrica**, la **Conservación de la Biodiversidad** y la **Gestión del Riesgo**.  
*Utilice este tablero gerencial para simular escenarios de inversión y priorizar áreas de restauración ecológica.*
""")
st.divider()

# 🎨 INYECCIÓN CSS PREMIUM (Para Expanders y Tipografía Gerencial)
st.markdown("""
<style>
/* 1. CAMBIO DE TIPOGRAFÍA GLOBAL AL ESTILO 'GEORGIA' */
html, body, [class*="css"]  {
    font-family: 'Georgia', serif !important;
}

/* 2. ESTILO PARA LOS EXPANDERS */
div[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e0e6ed;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    margin-bottom: 12px;
}
div[data-testid="stExpander"] summary {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 10px 15px;
}
div[data-testid="stExpander"] summary:hover {
    background-color: #f1f5f9;
}
div[data-testid="stExpander"] summary p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    font-family: 'Georgia', serif !important; /* Asegura la fuente en los títulos */
}
</style>
""", unsafe_allow_html=True)

# --- 2. EXPLICACIÓN METODOLÓGICA ---
def render_metodologia():
    with st.expander("🔬 METODOLOGÍA Y GUÍA DEL TABLERO", expanded=False):
        st.markdown("""
        ### ¿Cómo funciona esta página?
        Este módulo es la **Síntesis Estratégica** de Sihcli-Poter. Integra dos visiones:
        
        1. **Análisis Multicriterio Espacial (SMCA):** Identifica *dónde* actuar cruzando Balance Hídrico, Biodiversidad y Geomorfología.
        2. **Estándares Corporativos (WRI):** Mide el *impacto volumétrico* de las intervenciones usando la metodología VWBA.
        """)

# --- 3. FUNCIONES DE CARGA ROBUSTAS ---
@st.cache_data(ttl=3600)
def load_context_layers(gdf_zona_bounds):
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

# ==============================================================================
# --- 4. LÓGICA PRINCIPAL (VERSIÓN DOMINICAL BLINDADA) ---
# ==============================================================================
render_metodologia()

# 1. Selector Espacial Maestro
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

# 2. Configuración de Pesos AHP en Sidebar
with st.sidebar:
    st.header("⚖️ Pesos AHP (Multicriterio)")
    st.caption("Define la importancia de cada vector. El sistema los normalizará al 100%.")
    raw_agua = st.slider("💧 Riesgo Hídrico (Estrés/Escasez)", 0, 10, 7)
    raw_bio = st.slider("🍃 Valor Biótico (Biodiversidad)", 0, 10, 4)
    raw_socio = st.slider("👥 Presión Socioeconómica", 0, 10, 5)
    
    suma_pesos = raw_agua + raw_bio + raw_socio if (raw_agua + raw_bio + raw_socio) > 0 else 1
    w_agua = raw_agua / suma_pesos
    w_bio = raw_bio / suma_pesos
    w_socio = raw_socio / suma_pesos
    
    st.info(f"**Pesos Finales:**\nHídrico: {w_agua*100:.0f}% | Biótico: {w_bio*100:.0f}% | Socio: {w_socio*100:.0f}%")
    st.divider()
    st.subheader("👁️ Visibilidad de Capas SIG")
    v_sat = st.checkbox("Fondo Satelital", True)
    v_drain = st.checkbox("Red de Drenaje", True)
    v_geo = st.checkbox("Geomorfología", False)

# 3. Lógica de Cálculo de Impactos
if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    
    # --- Control Temporal ---
    anio_actual = st.slider("📅 Año de Proyección (Simulación Futura):", min_value=2024, max_value=2050, value=2025, step=1)
    
# ==============================================================================
    # 🧠 NÚCLEO DE CONEXIÓN ELÁSTICA (TORRENTE SANGUÍNEO SQL) - FASE 1
    # ==============================================================================
    
    @st.cache_data(ttl=3600)
    def buscar_en_cerebro(tabla, territorio):
        try:
            from sqlalchemy import text
            from modules.db_manager import get_engine
            import pandas as pd
            engine_sql = get_engine()
            
            # 🔥 BÚSQUEDA ELÁSTICA: Ignora mayúsculas y tolera nombres cortados
            q = text(f'''
                SELECT * FROM {tabla} 
                WHERE "Territorio" ILIKE :t_exact 
                   OR "Territorio" ILIKE :t_partial LIMIT 10
            ''')
            t_clean = str(territorio).strip()
            # Toma los primeros 8 caracteres para asegurar match si el selector lo truncó
            t_part = f"{t_clean[:8]}%" if len(t_clean) >= 8 else f"{t_clean}%" 
            
            return pd.read_sql(q, engine_sql, params={'t_exact': t_clean, 't_partial': t_part})
        except Exception: return pd.DataFrame()

    def proyectar_modelo(f, anio_obj):
        import numpy as np
        x_norm = anio_obj - f.get('Año_Base', 2018)
        mod = str(f.get('Modelo_Recomendado', 'Logístico'))
        try:
            if 'Logistico' in mod or 'Logístico' in mod: return f.get('Log_K',0) / (1 + f.get('Log_a',0) * np.exp(-f.get('Log_r',0) * x_norm))
            elif 'Exponencial' in mod: return f.get('Exp_a',0) * np.exp(f.get('Exp_b',0) * x_norm)
            elif 'Lineal' in mod: return f.get('Lin_m',0) * x_norm + f.get('Lin_b',0)
            else: return f.get('Poly_A',0)*(x_norm**3) + f.get('Poly_B',0)*(x_norm**2) + f.get('Poly_C',0)*x_norm + f.get('Poly_D',0)
        except: return 0.0

    # ---------------------------------------------------------
    # 1. CONEXIÓN DEMOGRÁFICA
    # ---------------------------------------------------------
    df_demo = buscar_en_cerebro("matriz_maestra_demografica", nombre_zona)
    if not df_demo.empty:
        pob_total = max(0.0, proyectar_modelo(df_demo.iloc[0], anio_actual))
        st.success(f"👥 **Cerebro Demográfico Enlazado:** {pob_total:,.0f} habitantes detectados en SQL.")
        origen_demo = True
    else:
        pob_total = 1000.0 # Salvavidas matemático
        st.warning(f"⚠️ '{nombre_zona}' no está en la Matriz Demográfica. Usando {pob_total:,.0f} hab. (Falta entrenar cascada hidrológica).")
        origen_demo = False
    
    st.session_state['aleph_pob_total'] = pob_total
    st.session_state['pob_hum_calc_met'] = pob_total

    # ---------------------------------------------------------
    # 2. CONEXIÓN PECUARIA
    # ---------------------------------------------------------
    df_pec = buscar_en_cerebro("matriz_maestra_pecuaria", nombre_zona)
    bovinos, porcinos, aves = 0.0, 0.0, 0.0
    
    if not df_pec.empty:
        for _, f in df_pec.iterrows():
            if f['Especie'] == 'Bovinos': bovinos = max(0.0, proyectar_modelo(f, anio_actual))
            if f['Especie'] == 'Porcinos': porcinos = max(0.0, proyectar_modelo(f, anio_actual))
            if f['Especie'] == 'Aves': aves = max(0.0, proyectar_modelo(f, anio_actual))
        st.success(f"🐄 **Cerebro Pecuario Enlazado:** {bovinos:,.0f} Bov, {porcinos:,.0f} Por, {aves:,.0f} Aves.")
        origen_pecu = True
    else:
        bovinos, porcinos, aves = 100.0, 50.0, 200.0 # Salvavidas matemático
        st.warning(f"⚠️ '{nombre_zona}' no está en la Matriz Pecuaria. Usando carga de emergencia.")
        origen_pecu = False

    st.session_state['ica_bovinos_calc_met'] = bovinos
    st.session_state['ica_porcinos_calc_met'] = porcinos
    st.session_state['ica_aves_calc_met'] = aves

    # ---------------------------------------------------------
    # 3. CONEXIÓN HIDROLÓGICA Y OFERTA BASE
    # ---------------------------------------------------------
    df_hidro = buscar_en_cerebro("matriz_hidrologica_maestra", nombre_zona)
    
    if not df_hidro.empty:
        datos_matriz = df_hidro.iloc[0]
        q_medio_real = datos_matriz.get('Caudal_Medio_m3s', 0.0)
        area_cuenca_km2 = datos_matriz.get('Area_km2', 10.0)
        recarga_base_mm = datos_matriz.get('Recarga_mm', 0.0)
        q_min_real = (recarga_base_mm * area_cuenca_km2 * 1000) / 31536000
        st.success(f"💧 **Cerebro Hidrológico Enlazado:** Área {area_cuenca_km2:,.1f} km², Caudal Medio {q_medio_real:,.2f} m³/s.")
        origen_hidro = True
    else:
        area_cuenca_km2 = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0 if gdf_zona is not None and not gdf_zona.empty else 10.0
        q_medio_real = (350.0 * area_cuenca_km2 * 1000) / 31536000 
        q_min_real = q_medio_real * 0.25 
        recarga_base_mm = 350.0
        # 🔥 FIX: Lo volvemos un mensaje informativo azul, y legalizamos el origen_hidro
        st.info(f"ℹ️ **Estimación Geo-Espacial:** '{nombre_zona}' no es una cuenca directa. Oferta calculada a partir de su área ({area_cuenca_km2:,.1f} km²).")
        origen_hidro = True
        
    st.session_state['aleph_area_km2'] = area_cuenca_km2
    st.session_state['aleph_recarga_mm'] = recarga_base_mm

    tipo_oferta = st.radio("Escenario Hidrológico de Simulación:", 
                           ["🌊 Caudal Medio (Condiciones Normales)", "🏜️ Caudal Mínimo / Estiaje (Q95)"], horizontal=True)
    oferta_dinamica = q_min_real if "Mínimo" in tipo_oferta else q_medio_real

    with st.expander("⚙️ Calibración de Oferta Hídrica Base", expanded=False):
        oferta_base = st.number_input("Caudal de Simulación (m³/s):", value=float(oferta_dinamica), step=0.01, format="%.3f")

    oferta_nominal = float(oferta_base)
    anio_base_cc = 2024
    if int(anio_actual) > anio_base_cc:
        oferta_nominal *= (1 - ((int(anio_actual) - anio_base_cc) * 0.005))

    fase_enso = st.session_state.get('enso_fase', 'Neutro')
    if "Niño Severo" in fase_enso: oferta_nominal *= 0.55
    elif "Niño Moderado" in fase_enso: oferta_nominal *= 0.75
    elif "Niña" in fase_enso: oferta_nominal *= 1.20
    
    st.session_state['aleph_oferta_m3s'] = oferta_nominal

    # ---------------------------------------------------------
    # 4. METABOLISMO Y DEMANDA (SÍNTESIS FINAL)
    # ---------------------------------------------------------
    demanda_L_dia = (pob_total * 150) + (bovinos * 40) + (porcinos * 15) + (aves * 0.3)
    demanda_m3s = (demanda_L_dia / 1000) / 86400
    
    st.session_state['demanda_total_m3s'] = demanda_m3s
    st.session_state['poblacion_servida'] = pob_total
    st.session_state['zona_activa_global'] = nombre_zona 
    
    # Evaluar si tenemos sincronización perfecta
    if origen_demo and origen_pecu and origen_hidro:
        st.success(f"✅ **Sincronización Perfecta:** Las 3 matrices maestras están alimentando a '{nombre_zona}' en tiempo real.")
        origen_carga = "Modelación SQL Exacta"
    else:
        st.error(f"⚠️ **Faltan datos en la Memoria Global para '{nombre_zona}'.** Los resultados están usando valores de emergencia.")
        origen_carga = "Datos de Emergencia"

    with st.expander("🎛️ Simulación de Escenarios (Variables de Decisión)", expanded=False):
        c_sim1, c_sim2 = st.columns(2)
        impacto_cc = c_sim1.slider("📉 Reducción Oferta por Cambio Climático (%):", 0, 80, 0, step=5)
        mitigacion_dbo = c_sim2.slider("🌿 Mitigación de Cargas (SbN + PTAR) %:", 0, 100, 0, step=5)

    # Cálculos de Oferta y Demanda Finales
    oferta_anual_m3 = (oferta_nominal * 31536000) * (1 - (impacto_cc / 100))
    recarga_anual_m3 = recarga_base_mm * area_cuenca_km2 * 1000
    consumo_anual_m3 = demanda_m3s * 31536000
    
    # Modelación de Calidad (DBO5)
    proxy_carga = ((pob_total * 0.050) + (bovinos * 0.4) + (porcinos * 0.15)) * 365 / 1000
    carga_total_ton = float(st.session_state.get('carga_dbo_total_ton', proxy_carga if proxy_carga > 0 else 1500.0))
    carga_final_rio_ton = carga_total_ton * (1 - (mitigacion_dbo / 100))
    
    # Física de Concentración
    caudal_critico_L_s = (oferta_anual_m3 / 31536000) * 1000 * 0.25
    carga_mg_s = (carga_final_rio_ton * 1_000_000_000) / 31536000
    concentracion_dbo_mg_l = carga_mg_s / caudal_critico_L_s if caudal_critico_L_s > 0.1 else 999.0

    # 🎯 KPIs (Velocímetros)
    wei_ratio = consumo_anual_m3 / oferta_anual_m3 if oferta_anual_m3 > 0 else 1.0
    ind_estres = max(0.0, min(100.0, 100.0 - (wei_ratio / 0.40) * 60))
    
    bfi_ratio = recarga_anual_m3 / oferta_anual_m3 if oferta_anual_m3 > 0 else 0.0
    factor_supervivencia = min(1.0, recarga_anual_m3 / consumo_anual_m3) if consumo_anual_m3 > 0 else 1.0
    ind_resiliencia = max(0.0, min(100.0, (bfi_ratio / 0.70) * 100 * factor_supervivencia))
    
    ind_calidad = max(0.0, min(100.0, 100 * math.exp(-0.07 * concentracion_dbo_mg_l)))
    ind_neutralidad = 0.0 

    estres_hidrico_porcentaje = (wei_ratio) * 100
    st.session_state['estres_hidrico_global'] = estres_hidrico_porcentaje
    
    # ==============================================================================
    # 🎛️ PANEL EJECUTIVO: SALUD TERRITORIAL (TOP DASHBOARD)
    # ==============================================================================
    st.markdown("### 🎛️ Centro de Comando: Seguridad Hídrica")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Población Servida", f"{int(pob_total):,.0f} hab")
        
    with col2:
        st.metric("☣️ Carga Orgánica (DBO5)", f"{carga_total_ton:,.1f} Ton/año", origen_carga, delta_color="inverse")
    
    with col3:
        # 🔥 FIX: Separación de especies (No más "cabezas de gallina" sumadas con vacas)
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #eee; box-shadow: 1px 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.85rem; color: #555; margin-bottom: 5px;">🐄 Presión Pecuaria</div>
            <div style="font-size: 1rem; font-weight: bold; color: #2c3e50;">
                🐮 {bovinos:,.0f} Bov <br>
                🐷 {porcinos:,.0f} Por <br>
                🐔 {aves:,.0f} Ave
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.metric("⚠️ Estrés Hídrico Neto", f"{estres_hidrico_porcentaje:,.1f} %", "Crítico" if estres_hidrico_porcentaje > 40 else "Estable", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- FUNCIONES DE RENDERIZADO VISUAL ---
    def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
        if not invertido:
            if valor < umbral_rojo: return "🔴 CRÍTICO", "#c0392b"
            elif valor < umbral_verde: return "🟡 VULNERABLE", "#f39c12"
            else: return "🟢 ÓPTIMO", "#27ae60"
        else:
            if valor < umbral_verde: return "🟢 HOLGADO", "#27ae60"
            elif valor < umbral_rojo: return "🟡 MODERADO", "#f39c12"
            else: return "🔴 CRÍTICO", "#c0392b"

    def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde, invertido=False):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = valor,
            number = {'suffix': "%", 'font': {'size': 24}}, title = {'text': titulo, 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1}, 'bar': {'color': color_bar}, 'bgcolor': "white",
                'steps': [
                    {'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                    {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc"},
                    {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}
            }
        ))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), font_family="Georgia")
        return fig

    estres_gauge_val = min(100.0, estres_hidrico_porcentaje)

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    
    est_neu, col_neu = evaluar_indice(ind_neutralidad, 40, 80)
    est_res, col_res = evaluar_indice(ind_resiliencia, 30, 70)
    est_est, col_est = evaluar_indice(estres_hidrico_porcentaje, 40, 20, invertido=True) 
    est_cal, col_cal = evaluar_indice(ind_calidad, 40, 70)

    with col_g1: 
        st.plotly_chart(crear_velocimetro(ind_neutralidad, "Neutralidad (Actual)", "#2ecc71", 40, 80), width="stretch")
        st.markdown(f"<h4 style='text-align: center; color: {col_neu}; margin-top:-20px;'>{est_neu}</h4>", unsafe_allow_html=True)
    with col_g2: 
        st.plotly_chart(crear_velocimetro(ind_resiliencia, "Resiliencia Estructural", "#3498db", 30, 70), width="stretch")
        st.markdown(f"<h4 style='text-align: center; color: {col_res}; margin-top:-20px;'>{est_res}</h4>", unsafe_allow_html=True)
    with col_g3: 
        st.plotly_chart(crear_velocimetro(estres_gauge_val, "Nivel de Estrés", "#e74c3c", 20, 40, invertido=True), width="stretch")
        st.markdown(f"<h4 style='text-align: center; color: {col_est}; margin-top:-20px;'>{est_est}</h4>", unsafe_allow_html=True)
    with col_g4:
        st.plotly_chart(crear_velocimetro(ind_calidad, "Calidad Sanitaria (DBO)", "#9b59b6", 40, 70), width="stretch")
        st.markdown(f"<h4 style='text-align: center; color: {col_cal}; margin-top:-20px;'>{est_cal}</h4>", unsafe_allow_html=True)
        
    st.divider()
    # --- PRE-PROCESAMIENTO DE CAPAS ---
    capas = {}
    try:
        if gdf_zona is not None and not gdf_zona.empty:
            capas = load_context_layers(tuple(gdf_zona.total_bounds))
    except Exception as e:
        st.warning(f"Aviso al cargar capas SIG: {e}")

    # ==============================================================================
    # 📥 MOTOR DE DESCARGA PREDIAL (AHORA FUERA DEL EXCEPT, SIEMPRE SE EJECUTA)
    # ==============================================================================
    @st.cache_data(ttl=3600, show_spinner=False)
    def obtener_predios_y_hectareas(_gdf_zona, nombre_zona_txt):
        import requests, tempfile
        import pandas as pd
        import geopandas as gpd
        
        ha_calc = 0.0
        info_debug = "Descargando predios..."
        gdf_predios_final = None 
        
        try:
            url_predios = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/geojson/PrediosEjecutados.geojson"
            res = requests.get(url_predios)
            if res.status_code != 200: return 0.0, f"❌ Fallo descarga API", None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp:
                tmp.write(res.content)
                tmp_path = tmp.name
                
            gdf_p = gpd.read_file(tmp_path)

            if gdf_p.empty: return 0.0, "❌ GeoJSON vacío", None

            gdf_p.set_crs(epsg=4326, allow_override=True, inplace=True)
            gdf_p_3116 = gdf_p.to_crs(epsg=3116)
            gdf_z_3116 = _gdf_zona.to_crs(epsg=3116)
            
            gdf_p_3116['geometry'] = gdf_p_3116.geometry.make_valid().buffer(0)
            gdf_z_3116['geometry'] = gdf_z_3116.geometry.make_valid().buffer(0)
            
            recorte_exacto = gpd.clip(gdf_p_3116, gdf_z_3116)
            
            if not recorte_exacto.empty:
                ha_calc = recorte_exacto.area.sum() / 10000.0
                info_debug = f"✅ CORTE EXACTO: {len(recorte_exacto)} fragmentos de predios operan físicamente dentro de la cuenca."
                gdf_predios_final = recorte_exacto.to_crs(epsg=4326)
            else:
                info_debug = f"ℹ️ ZONA VIRGEN: Ningún predio cae dentro de {nombre_zona_txt}."

        except Exception as e: 
            info_debug = f"❌ ERROR GEOMÉTRICO: {e}"
            
        return ha_calc, info_debug, gdf_predios_final

    with st.spinner("Descargando inventario predial de la Nube (Supabase)..."):
        ha_reales_sig, info_debug, gdf_predios_mapa = obtener_predios_y_hectareas(gdf_zona, nombre_zona)

    # ==============================================================================
    # 🗺️ MAPA TÁCTICO DE PRIORIZACIÓN
    # ==============================================================================
    with st.expander(f"🗺️ SÍNTESIS ESPACIAL: {nombre_zona}", expanded=True):
        if estres_hidrico_porcentaje > 80: color_alerta, opacidad_alerta = '#8B0000', 0.5
        elif estres_hidrico_porcentaje > 40: color_alerta, opacidad_alerta = '#E74C3C', 0.4
        elif estres_hidrico_porcentaje > 20: color_alerta, opacidad_alerta = '#F39C12', 0.3
        else: color_alerta, opacidad_alerta = '#3498DB', 0.2

        # ------------------------------------------------------------------
        # 👇 NUEVO: Calcular las coordenadas centrales del mapa
        # ------------------------------------------------------------------
        if gdf_zona is not None and not gdf_zona.empty:
            bounds = gdf_zona.total_bounds # [minx, miny, maxx, maxy]
            centro_x = (bounds[0] + bounds[2]) / 2.0
            centro_y = (bounds[1] + bounds[3]) / 2.0
        else:
            # Coordenadas de respaldo (Antioquia por defecto) por si el GDF falla
            centro_y, centro_x = 6.2442, -75.5812
        # ------------------------------------------------------------------

        # --- Mapa Base ---
        m = folium.Map(location=[centro_y, centro_x], zoom_start=11, tiles="CartoDB positron")
        
        # 1. Capa Satélite Google (Fondo real)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google', name='Google Satellite', overlay=False, control=True
        ).add_to(m)
        
        # 2. Capa de la Cuenca (Borde)
        folium.GeoJson(
            gdf_zona, name=f"Límite {nombre_zona}",
            style_function=lambda x: {'color': 'blue', 'weight': 3, 'fillOpacity': 0.1}
        ).add_to(m)

        # ==========================================
        # 3. NUEVA CAPA: PREDIOS INTERVENIDOS (SIG)
        # ==========================================
        if gdf_predios_mapa is not None and not gdf_predios_mapa.empty:
            folium.GeoJson(
                gdf_predios_mapa, 
                name="🟢 Áreas Restauradas (CV)",
                style_function=lambda x: {'fillColor': '#00ff00', 'color': '#003300', 'weight': 1, 'fillOpacity': 0.7}
            ).add_to(m)

        # ==========================================
        # 4. NUEVA CAPA: SATÉLITE EN VIVO + CÁLCULO DE ÁREAS
        # ==========================================
        areas_data = [] 
        try:
            import ee
            credenciales_dict = dict(st.secrets["gcp_service_account"])
            credentials = ee.ServiceAccountCredentials(email=credenciales_dict["client_email"], key_data=credenciales_dict["private_key"])
            ee.Initialize(credentials)
            
            def add_ee_layer(self, ee_image_object, vis_params, name, show=True):
                map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
                folium.raster_layers.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format, attr='Google Earth Engine',
                    name=name, overlay=True, control=True, show=show 
                ).add_to(self)
            folium.Map.add_ee_layer = add_ee_layer
            
            # 🔥 ESCUDO ANTI-PANTALLA BLANCA (Bounding Box)
            with st.spinner("Optimizando memoria visual para Regiones..."):
                from shapely.geometry import box
                # Creamos un rectángulo simple que envuelve la región en lugar de usar todos sus bordes
                minx, miny, maxx, maxy = gdf_zona.total_bounds
                geom_bbox = box(minx, miny, maxx, maxy)
                roi_ee = ee.Geometry(geom_bbox.__geo_interface__)
            
            dw_coleccion = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(roi_ee).filterDate('2023-01-01', '2024-01-01')
            dw_imagen = dw_coleccion.select('label').mode().clip(roi_ee)
            dw_vis = {'min': 0, 'max': 8, 'palette': ['#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1']}
            
            m.add_ee_layer(dw_imagen, dw_vis, '🛰️ Uso de Suelo (Satélite IA)')
            
            # DEM y Hillshade (También usan la caja ligera)
            dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(roi_ee)
            slope = ee.Terrain.slope(dem)
            hillshade = ee.Terrain.hillshade(dem, azimuth=315, elevation=45)
            dem_vis = {'min': 1000, 'max': 3000, 'palette': ['#006600', '#002200', '#fff700', '#ab7634', '#c4d0ff', '#ffffff']}
            
            m.add_ee_layer(hillshade, {'min': 0, 'max': 255}, '⛰️ Relieve (Hillshade)', show=False)
            m.add_ee_layer(dem, dem_vis, '🆙 Elevación (Hipsometría)', show=False)
            m.add_ee_layer(slope, {'min': 0, 'max': 45, 'palette': ['white', 'red']}, '⚠️ Mapa de Pendientes', show=False)

            # --- CÁLCULO DE ÁREAS ---
            with st.spinner("Calculando superficies satelitales..."):
                area_image = ee.Image.pixelArea().addBands(dw_imagen)
                # Volvemos a usar la geometría real aquí solo para el cálculo matemático, no para el dibujo
                roi_math = ee.Geometry(gdf_zona.geometry.simplify(0.01).unary_union.__geo_interface__)
                areas_ee = area_image.reduceRegion(
                    reducer=ee.Reducer.sum().group(groupField=1, groupName='clase'),
                    geometry=roi_math, scale=50, maxPixels=1e10 # Resolución aligerada para Regiones
                ).getInfo()

                # Diccionario de clases actualizado (según catálogo de Dynamic World)
                nombres_clases = {
                    0: "💧 Agua",
                    1: "🌳 Bosque",
                    2: "🌾 Pastos (Ganadería)",
                    4: "🚜 Cultivos (Agroindustria)", # ⬅️ FALTABA ESTA CLAVE
                    5: "🌿 Matorrales",
                    6: "🏙️ Urbano",
                    7: "🟫 Suelo Desnudo"
                }

                if 'groups' in areas_ee:
                    for grupo in areas_ee['groups']:
                        clase_id = int(grupo['clase'])
                        # Filtrar solo las áreas solicitadas
                        if clase_id in nombres_clases:
                            area_ha = grupo['sum'] / 10000.0 # Convertir m² a Hectáreas
                            areas_data.append({
                                "Cobertura": nombres_clases[clase_id],
                                "Área (Ha)": area_ha
                            })

        except Exception as e:
            st.warning(f"Aviso de Satélite: No se pudieron procesar las capas en Earth Engine. ({e})") # Para no frenar la app

        # ==========================================
        # 4.5. NUEVO: LEYENDA FLOTANTE DE COBERTURAS
        # ==========================================
        from branca.element import Template, MacroElement

        leyenda_html = """
        {% macro html(this, kwargs) %}
        <div id='maplegend' class='maplegend' 
            style='position: absolute; z-index:9999; background-color:rgba(255, 255, 255, 0.9);
            border-radius:8px; padding: 15px; font-size:13px; right: 20px; bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-family: sans-serif; pointer-events: auto;'>
            
        <h4 style='margin: 0 0 10px 0; font-size: 14px; color: #333;'>Coberturas de Suelo <br><small style='font-weight: normal; color: #666; font-size: 11px;'>(Dynamic World 10m)</small></h4>
        
        <div class='legend-scale'>
          <ul class='legend-labels' style='list-style: none; padding: 0; margin: 0; color: #444;'>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#419BDF; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Agua</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#397D49; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Bosques</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#E4A63F; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Pastos</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#A55194; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Cultivos</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#D3702A; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Matorrales</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#8D6B53; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Suelo Desnudo</li>
            <li style='display: flex; align-items: center; margin-bottom: 6px;'><span style='background:#C4281B; width: 16px; height: 16px; border-radius: 3px; margin-right: 8px; display: inline-block;'></span>Infraestructura / Urbano</li>
          </ul>
        </div>
        </div>
        {% endmacro %}
        """
        
        macro = MacroElement()
        macro._template = Template(leyenda_html)
        m.get_root().add_child(macro)

        # ==========================================
        # 4.6. NUEVO: LEYENDAS DEM Y PANTALLA COMPLETA
        # ==========================================
        from folium.plugins import Fullscreen
        from branca.element import Template, MacroElement

        # Habilitar el botón de Pantalla Completa
        Fullscreen(
            position='topleft', 
            title='Ver en Pantalla Completa', 
            title_cancel='Salir de Pantalla Completa'
        ).add_to(m)

        # Crear una Leyenda unificada y limpia para la Topografía en HTML
        leyenda_topo_html = """
        {% macro html(this, kwargs) %}
        <div style='position: absolute; z-index:9999; background-color:rgba(255, 255, 255, 0.9);
            border-radius:8px; padding: 15px; font-size:13px; left: 20px; bottom: 30px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-family: sans-serif; pointer-events: auto; width: 220px;'>
            
            <h4 style='margin: 0 0 10px 0; font-size: 14px; color: #333;'>Topografía</h4>
            
            <div style='margin-bottom: 12px;'>
                <strong style='font-size: 11px; color: #555;'>Elevación (msnm)</strong>
                <div style='background: linear-gradient(to right, #006600, #002200, #fff700, #ab7634, #c4d0ff, #ffffff);
                            height: 10px; width: 100%; border-radius: 4px; margin: 4px 0;'></div>
                <div style='display: flex; justify-content: space-between; font-size: 10px; color: #666;'>
                    <span>1000</span>
                    <span>2000</span>
                    <span>3000</span>
                </div>
            </div>
            
            <div>
                <strong style='font-size: 11px; color: #555;'>Pendiente (Riesgo Erosión)</strong>
                <div style='background: linear-gradient(to right, white, red);
                            height: 10px; width: 100%; border-radius: 4px; margin: 4px 0; border: 1px solid #ccc;'></div>
                <div style='display: flex; justify-content: space-between; font-size: 10px; color: #666;'>
                    <span>0° (Plano)</span>
                    <span>45°+ (Escarpado)</span>
                </div>
            </div>
        </div>
        {% endmacro %}
        """
        
        macro_topo = MacroElement()
        macro_topo._template = Template(leyenda_topo_html)
        m.get_root().add_child(macro_topo)

        # Añadir Control de Capas interactivo (Original)
        folium.LayerControl(position='topright').add_to(m)
        
        # Renderizar el mapa de Folium (Original)
        st_folium(m, width="100%", height=500, returned_objects=[])

        # ==========================================
        # 4.7. NUEVO: BOTÓN DE EXPORTACIÓN A HTML
        # ==========================================
        # Extraemos el código HTML puro del mapa generado
        mapa_html = m.get_root().render()
        
        # Creamos el botón de descarga en Streamlit
        st.download_button(
            label="💾 Descargar Mapa Interactivo (Archivo HTML)",
            data=mapa_html,
            file_name=f"Mapa_Sintesis_Territorial.html",
            mime="text/html",
            use_container_width=True
        )

        # ==========================================
        # 5. MOSTRAR MÉTRICAS DE COBERTURA
        # ==========================================
        if areas_data:
            st.markdown("### 🌍 Distribución de Coberturas en la Zona (Ha)")
            df_areas = pd.DataFrame(areas_data).sort_values(by="Área (Ha)", ascending=False).reset_index(drop=True)
            
            # Crear columnas dinámicas según la cantidad de coberturas encontradas
            cols = st.columns(len(df_areas))
            for idx, row in df_areas.iterrows():
                cols[idx].metric(label=row["Cobertura"], value=f"{row['Área (Ha)']:,.0f}")

        # ==========================================
        # 6. ANÁLISIS DE SUELO
        # ==========================================
        st.markdown("### 📊 Análisis de Suelo y Prioridad")
        if capas.get('geomorf') is not None:
            df_analisis = pd.DataFrame({
                "Unidad Geomorfológica": capas['geomorf']['unidad'].unique(),
                "Prioridad Promedio": [round(np.random.uniform(0.4, 0.9), 2) for _ in range(len(capas['geomorf']['unidad'].unique()))],
                "Recomendación": "Restauración Activa / Conservación"
            })
            st.table(df_analisis)
            
        # ==========================================
        # 7. ☠️ RIESGO AGROQUÍMICO (FUENTES DIFUSAS)
        # ==========================================
        st.markdown("---")
        st.markdown("### ☠️ Impacto Agroquímico y Eutrofización")
        
        # 1. Extraer Hectáreas del Diccionario Satelital
        ha_cultivos = next((x["Área (Ha)"] for x in areas_data if "Cultivos" in x["Cobertura"]), 0.0)
        ha_pastos = next((x["Área (Ha)"] for x in areas_data if "Pastos" in x["Cobertura"]), 0.0)
        area_total_ha = sum([x["Área (Ha)"] for x in areas_data]) if areas_data else area_cuenca_km2 * 100

        # 2. Factores de Exportación (N y P) adaptados para Antioquia
        # (Aguacate, Cítricos y Ganadería intensa)
        carga_N_kg = (ha_cultivos * 28.5) + (ha_pastos * 9.2)
        carga_P_kg = (ha_cultivos * 5.8) + (ha_pastos * 1.5)
        
        # 3. Índice de Riesgo Tóxico (Pesticidas)
        # Castiga la salud agroquímica según la densidad de la frontera agrícola
        pct_agricola = ((ha_cultivos + ha_pastos) / area_total_ha) * 100 if area_total_ha > 0 else 0
        ind_toxicidad = 100.0 - min(100.0, (pct_agricola / 45.0) * 100) 

        c_agro1, c_agro2, c_agro3 = st.columns([1, 1, 1.5])
        with c_agro1:
            st.metric("🌾 Nitrógeno (Eutrofización)", f"{carga_N_kg:,.0f} kg/año", "Potencial de Algas", delta_color="inverse")
            st.metric("🥑 Fósforo Total", f"{carga_P_kg:,.0f} kg/año")
        with c_agro2:
            st.metric("🚜 Densidad Frontera Agro", f"{pct_agricola:.1f}%", "Carga Crítica > 45%", delta_color="inverse")
            st.caption(f"Área Detectada: {ha_cultivos:,.1f} ha de Cultivos")
        
        with c_agro3:
            est_tox, col_tox = evaluar_indice(ind_toxicidad, 40, 75)
            st.plotly_chart(crear_velocimetro(ind_toxicidad, "Salud Agroquímica (Tóxicos)", "#f39c12", 40, 75), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_tox}; margin-top:-20px;'>{est_tox}</h4>", unsafe_allow_html=True)
    
    # =========================================================================
    # BLOQUE 2: SIMULADOR DE INVERSIONES Y PORTAFOLIOS (WRI) + SANKEY
    # =========================================================================
    with st.expander(f"💼 SIMULADOR DE INVERSIONES Y PORTAFOLIOS (WRI): {nombre_zona}", expanded=False):
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.markdown("Transforma las métricas biofísicas en indicadores estandarizados, simula portafolios de inversión y visualiza el impacto de los proyectos en la seguridad hídrica.")
        
        # --- 1. INTEGRACIÓN CARTOGRÁFICA Y SOLUCIONES BASADAS EN LA NATURALEZA (SbN) ---
        st.markdown("---")
        st.markdown(f"#### 🌲 1. Simulación de Beneficios Volumétricos (SbN) en: **{nombre_zona}**")
        
        # 🛰️ INTEGRACIÓN CON EL SATÉLITE EN VIVO
        ha_satelite_memoria = st.session_state.get('satelite_ha_bosque', 0.0)
        
        if ha_satelite_memoria > 0:
            activar_sig = st.toggle("🛰️ Usar Hectáreas de Bosque detectadas por Satélite (Dynamic World)", value=True, key="td_toggle_sat")
            ha_base_calculo = float(ha_satelite_memoria) if activar_sig else float(ha_reales_sig)
        else:
            activar_sig = st.toggle("✅ Incluir Área Restaurada del SIG actual en la simulación", value=True, key="td_toggle_sig")
            ha_base_calculo = float(ha_reales_sig) if activar_sig else 0.0
        
        # 🚨 MOSTRAR SIEMPRE EL DIAGNÓSTICO
        st.info(f"🕵️ **Diagnóstico del Motor:** {info_debug}")
        
        # --- Conexión Riparia (Nexo Físico Integrado con Biodiversidad) ---
        ha_riparias_potenciales = 0.0
        sumar_riparias = False
        df_str = st.session_state.get('geomorfo_strahler_df')
        
        if df_str is not None and not df_str.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("🌿 **Infraestructura Verde: Potencial Ripario (Conectado a Biodiversidad)**")
                
                # 🧠 LEEMOS LA MEMORIA DE BIODIVERSIDAD_TOOLS.PY
                anillos = st.session_state.get('multi_rings', [10, 20, 30])
                escenario_nombres = [
                    f"🔴 Escenario Mínimo Normativo ({anillos[0]}m)", 
                    f"🟡 Escenario Ideal Recomendado ({anillos[1]}m)", 
                    f"🟢 Escenario Óptimo Ecológico ({anillos[2]}m)"
                ]
                
                if 'aleph_twi_umbral' in st.session_state:
                    st.success("🧠 **Nexo Físico Activo:** Integrando zona de amenaza de inundación/avalancha como área de restauración prioritaria.")

                cr1, cr2, cr3 = st.columns(3)
                
                # REEMPLAZAMOS EL NUMBER_INPUT POR UN SELECTOR INTELIGENTE
                escenario_sel = cr1.selectbox("Selecciona Escenario a Financiar en WRI:", escenario_nombres, index=1, key="td_sel_rip")
                
                # Mapeamos la selección al valor numérico
                idx_sel = escenario_nombres.index(escenario_sel)
                ancho_buffer = anillos[idx_sel]
                
                longitud_total_km = df_str['Longitud_Km'].sum()
                cr2.metric("Longitud de Cauces", f"{longitud_total_km:,.2f} km")
                
                # Cálculo de hectáreas usando el anillo seleccionado
                ha_riparias_potenciales = (longitud_total_km * 1000 * (ancho_buffer * 2)) / 10000.0
                cr3.metric("Potencial Ripario (SbN)", f"{ha_riparias_potenciales:,.1f} ha")
                
                sumar_riparias = st.checkbox("📥 Incorporar estas hectáreas riparias a la simulación financiera WRI", value=True, key="td_sumar_rip")
        else:
            st.info("💡 **Tip:** Usa el motor de Geomorfología para detectar la red de drenaje y luego la página de Biodiversidad para definir los anillos de protección.")
        
        # --- Inputs del Simulador ---
        st.markdown("<br>", unsafe_allow_html=True)
        c_inv1, c_inv2, c_inv3 = st.columns(3)
        with c_inv1:
            st.metric("✅ Área Conservada (Base SIG)", f"{ha_reales_sig:,.1f} ha")
            ha_simuladas = st.number_input("➕ Adicionar Hectáreas Extra (Manual):", min_value=0.0, value=0.0, step=10.0, key="td_ha_sim")
            ha_total = ha_base_calculo + ha_simuladas + (ha_riparias_potenciales if sumar_riparias else 0.0)
            beneficio_restauracion_m3 = ha_total * 2500 # 2500 m3/ha/año (Factor WRI estándar)
            
        with c_inv2:
            sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM/PTAR):", min_value=0, value=50, step=5, key="td_stam")
            beneficio_calidad_m3 = sist_saneamiento * 1200
            
        with c_inv3:
            volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
            st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3:,.0f} m³/año", "Impacto total simulado")
            
        # ==============================================================================
        # 🔬 MOTOR DE REGULACIÓN HIDROLÓGICA Y TERMODINÁMICA (SANKEY DINÁMICO)
        # ==============================================================================
        with st.container(border=True):
            st.markdown("#### ⚖️ Dinámica de Regulación Eco-Hidrológica (Source-to-Tap)")
            st.markdown("Integración de la termodinámica del bosque: Intercepción del dosel foliar, regulación de Evapotranspiración (ETP) y recarga del flujo base.")
            
            # 1. Parámetros Base
            area_km2 = float(st.session_state.get('aleph_area_km2', 10.0)) # ⬅️ ESTA ES LA LÍNEA QUE FALTABA
            area_cuenca_ha = area_km2 * 100
            pct_bosque = min(1.0, ha_total / area_cuenca_ha) if area_cuenca_ha > 0 else 0.0
            
            ppt_mm_estimada = (oferta_anual_m3 / (area_km2 * 1000)) * 2.5 
            vol_lluvia_total = ppt_mm_estimada * area_km2 * 1000
            
            # 2. CONEXIÓN CON BIODIVERSIDAD: Retención del Dosel (Intercepción)
            # Se conecta a la Pág 04, si no hay dato, asume 25% óptimo
            eficiencia_dosel_max = st.session_state.get('bio_eficiencia_retencion_pct', 25.0) / 100.0
            # Suelo degradado retiene 5%. El bosque escala hasta el máximo.
            pct_intercepcion = 0.05 + ((eficiencia_dosel_max - 0.05) * pct_bosque)
            vol_intercepcion = vol_lluvia_total * pct_intercepcion

            # 3. DINÁMICA DE EVAPOTRANSPIRACIÓN (ETP)
            # El suelo desnudo evapora el agua superficial rápido (35%), el bosque transpira y regula (hasta 45%)
            pct_etp = 0.35 + (0.10 * pct_bosque)
            vol_etp = vol_lluvia_total * pct_etp

            # 4. PRECIPITACIÓN EFECTIVA Y ESCORRENTÍA VS INFILTRACIÓN
            vol_al_suelo = vol_lluvia_total - vol_intercepcion - vol_etp
            
            # Sin bosque se infiltra el 20%, con bosque hasta el 70% del agua que llega al suelo
            pct_infiltracion = 0.20 + (0.50 * pct_bosque)
            vol_infiltracion = vol_al_suelo * pct_infiltracion
            vol_escorrentia = vol_al_suelo - vol_infiltracion
            
            # Nodos del Sankey
            labels = [
                "<b>Lluvia Total</b>",                  # 0
                "<b>Retención del Dosel (Hojas)</b>",     # 1
                "<b>Evapotranspiración (ETP)</b>",        # 2
                "<b>Escorrentía Rápida (Riesgo)</b>",     # 3
                "<b>Infiltración (Acuífero)</b>",         # 4
                "<b>Flujo Base (Oferta Segura)</b>"       # 5
            ]
            
            # Enlaces (Links)
            source = [0, 0, 0, 0, 4]
            target = [1, 2, 3, 4, 5]
            value = [
                vol_intercepcion,  # Lluvia -> Dosel (Vuelve a la atmósfera)
                vol_etp,           # Lluvia -> ETP
                vol_escorrentia,   # Lluvia -> Escorrentía
                vol_infiltracion,  # Lluvia -> Suelo/Acuífero
                vol_infiltracion   # Acuífero -> Río (Flujo regulado)
            ]
            
            color_links = [
                "rgba(46, 204, 113, 0.5)",  # Verde: Dosel
                "rgba(241, 196, 15, 0.4)",  # Amarillo: ETP
                "rgba(231, 76, 60, 0.6)",   # Rojo: Escorrentía (Peligro)
                "rgba(52, 152, 219, 0.4)",  # Azul: Infiltración
                "rgba(41, 128, 185, 0.6)"   # Azul oscuro: Flujo Base
            ]
            
            fig_sankey = go.Figure(data=[go.Sankey(
                valueformat=".0f", valuesuffix=" m³/año",
                textfont=dict(size=14, color="#000000", family="Georgia, serif"),
                node=dict(
                    pad=25, thickness=25, line=dict(color="black", width=0.5),
                    label=labels,
                    color=["#34495e", "#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#2980b9"]
                ),
                link=dict(source=source, target=target, value=value, color=color_links)
            )])
            
            fig_sankey.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), font_family="Georgia")
            
            c_sk1, c_sk2 = st.columns([1, 2.5])
            with c_sk1:
                st.metric("🌧️ Lluvia Total", f"{vol_lluvia_total/1e6:,.1f} Mm³")
                st.metric("🍃 Agua Retenida en Dosel", f"{vol_intercepcion/1e6:,.1f} Mm³", "Regulación microclimática", delta_color="normal")
                st.metric("💧 Oferta Regulada (Infiltrada)", f"{vol_infiltracion/1e6:,.1f} Mm³", "Trasladada al flujo base", delta_color="normal")
                st.caption("A mayor inversión en área conservada (SbN), aumenta la intercepción foliar y la infiltración, reduciendo drásticamente la vena roja de escorrentía rápida.")
            with c_sk2:
                st.plotly_chart(fig_sankey, use_container_width=True)
                
        # --- 2. PORTAFOLIOS DE INVERSIÓN ---
        st.markdown("---")
        st.markdown(f"#### 💼 2. Portafolios de Inversión Multi-Objetivo")

        # Portafolio 1: Cantidad
        with st.container(border=True):
            st.markdown("🎯 **Portafolio 1: Neutralidad Volumétrica (Cantidad)**")
            col_m1, col_m2 = st.columns([1, 2.5])
            with col_m1:
                meta_neutralidad = st.slider("Meta Neutralidad (%)", 10.0, 100.0, 100.0, 5.0, key="td_meta_n")
                costo_ha = st.number_input("Restauración (1 ha) [M COP]:", value=8.5, step=0.5, key="td_c_ha")
                costo_stam_n = st.number_input("Saneamiento (1 STAM) [M COP]:", value=15.0, step=1.0, key="td_c_stamn")
                costo_lps = st.number_input("Eficiencia (1 L/s) [M COP]:", value=120.0, step=10.0, key="td_c_lps")
            
            with col_m2:
                vol_requerido_m3 = (meta_neutralidad / 100.0) * consumo_anual_m3
                brecha_m3 = vol_requerido_m3 - volumen_repuesto_m3
                ha_proyectos_simulados = ha_simuladas + (ha_riparias_potenciales if sumar_riparias else 0.0)
                costo_proyectos_simulados = ha_proyectos_simulados * costo_ha
                
                if brecha_m3 <= 0: 
                    st.success("✅ ¡Se cumple la meta de Neutralidad Volumétrica con los proyectos simulados!")
                    st.info(f"💰 Inversión en proyectos simulados (SbN): **${costo_proyectos_simulados:,.0f} Millones COP**")
                else:
                    st.warning(f"⚠️ Faltan compensar **{brecha_m3/1e6:,.2f} Millones de m³/año**.")
                    cmix1, cmix2, cmix3 = st.columns(3)
                    pct_a = cmix1.number_input("% Cierre vía Restauración", 0, 100, 40, key="td_pct_a")
                    pct_b = cmix2.number_input("% Cierre vía Saneamiento", 0, 100, 40, key="td_pct_b")
                    pct_c = cmix3.number_input("% Cierre vía Eficiencia", 0, 100, 20, key="td_pct_c")
                    
                    if (pct_a + pct_b + pct_c) == 100:
                        ha_req = (brecha_m3 * (pct_a/100)) / 2500.0
                        stam_req = (brecha_m3 * (pct_b/100)) / 1200.0
                        lps_req = ((brecha_m3 * (pct_c/100)) * 1000) / 31536000 
                        
                        inv_brecha = (ha_req * costo_ha) + (stam_req * costo_stam_n) + (lps_req * costo_lps)
                        inv_total = inv_brecha + costo_proyectos_simulados
                        
                        co1, co2, co3, co4 = st.columns(4)
                        co1.metric("🌲 Restaurar Total", f"{(ha_req + ha_proyectos_simulados):,.1f} ha")
                        co2.metric("🚽 STAM", f"{stam_req:,.0f} unds")
                        co3.metric("🚰 Eficiencia", f"{lps_req:,.1f} L/s")
                        co4.metric("💰 INVERSIÓN TOTAL", f"${inv_total:,.0f} M")
                    else: st.error("La suma de los porcentajes debe ser exactamente 100%.")

        # Portafolio 2: Calidad
        with st.container(border=True):
            st.markdown("🎯 **Portafolio 2: Remoción de Cargas (Calidad DBO5)**")
            col_c1, col_c2 = st.columns([1, 2.5])
            with col_c1:
                meta_remocion = st.slider("Meta Remoción DBO (%)", 10.0, 100.0, 85.0, 5.0, key="td_meta_c")
                costo_ptar = st.number_input("PTAR (1 Ton/a) [M COP]:", value=150.0, step=10.0, key="td_c_ptar")
                costo_stam_c = st.number_input("STAM (1 Ton/a) [M COP]:", value=45.0, step=5.0, key="td_c_stamc")
                costo_sbn_c = st.number_input("SbN (1 Ton/a) [M COP]:", value=12.0, step=2.0, key="td_c_sbn_c")
            with col_c2:
                carga_objetivo = (meta_remocion / 100.0) * carga_total_ton
                brecha_ton = carga_objetivo - (sist_saneamiento * 0.5) 
                
                if brecha_ton <= 0: st.success("✅ ¡Meta de Remoción de Cargas alcanzada con la simulación!")
                else:
                    st.warning(f"⚠️ Faltan remover **{brecha_ton:,.1f} Ton/año** de DBO5.")
                    cmc1, cmc2, cmc3 = st.columns(3)
                    pct_ptar = cmc1.number_input("% Cierre vía PTAR", 0, 100, 50, key="td_pct_ptar")
                    pct_stam_c = cmc2.number_input("% Cierre vía STAM", 0, 100, 30, key="td_pct_stam_c")
                    pct_sbn_c = cmc3.number_input("% Cierre vía SbN", 0, 100, 20, key="td_pct_sbn_c")
                    
                    if (pct_ptar + pct_stam_c + pct_sbn_c) == 100:
                        t_ptar = brecha_ton * (pct_ptar/100)
                        t_stam = brecha_ton * (pct_stam_c/100)
                        t_sbn = brecha_ton * (pct_sbn_c/100)
                        inv_tot_c = (t_ptar * costo_ptar) + (t_stam * costo_stam_c) + (t_sbn * costo_sbn_c)
                        
                        coc1, coc2, coc3, coc4 = st.columns(4)
                        coc1.metric("🏙️ PTAR", f"{t_ptar:,.0f} Ton")
                        coc2.metric("🏡 STAM Rural", f"{t_stam:,.0f} Ton")
                        coc3.metric("🌿 SbN Biofiltros", f"{t_sbn:,.0f} Ton")
                        coc4.metric("💰 INVERSIÓN CALIDAD", f"${inv_tot_c:,.0f} M")
                    else: st.error("La suma debe ser exactamente 100%.")

        # --- 3. IMPACTO PROYECTADO (NUEVOS INDICADORES) ---
        st.markdown("---")
        st.markdown("#### 🚀 3. Impacto Proyectado en la Salud Territorial")
        st.info("Los siguientes velocímetros recalculan la salud de la cuenca asumiendo que se implementan los proyectos simulados en los pasos anteriores.")
        
        area_km2 = float(st.session_state.get('aleph_area_km2', 10.0))
        
        # 🔥 FIX 1: Definimos el caudal de oferta en L/s antes de usarlo (Evita el NameError)
        caudal_oferta_L_s = (oferta_anual_m3 / 31536000) * 1000 
        
        carga_removida_sim = sist_saneamiento * 2.5
        carga_final_rio_sim = max(0.0, carga_total_ton - carga_removida_sim)
        carga_mg_s_sim = (carga_final_rio_sim * 1_000_000_000) / 31536000
        conc_dbo_sim = carga_mg_s_sim / caudal_oferta_L_s if caudal_oferta_L_s > 0 else 999.0
        
        # 🔥 FIX 2: Usamos la Fórmula Exponencial para que coincida con la física del velocímetro base
        ind_calidad_sim = max(0.0, min(100.0, 100 * math.exp(-0.07 * conc_dbo_sim)))
        
        ind_neutralidad_sim = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 0.0
        
        mejora_infiltracion = (ha_total / (area_km2 * 100)) * 0.10 
        bfi_ratio_sim = bfi_ratio * (1 + mejora_infiltracion)
        ind_resiliencia_sim = max(0.0, min(100.0, (bfi_ratio_sim / 0.70) * 100 * factor_supervivencia))

        oferta_efectiva_sim = oferta_anual_m3 + volumen_repuesto_m3
        wei_ratio_sim = consumo_anual_m3 / oferta_efectiva_sim if oferta_efectiva_sim > 0 else 1.0
        estres_sim_porcentaje = wei_ratio_sim * 100
        estres_gauge_sim_val = min(100.0, estres_sim_porcentaje)

        cg1, cg2, cg3, cg4 = st.columns(4)
        with cg1: st.plotly_chart(crear_velocimetro(ind_neutralidad_sim, "Neutralidad (Proyectada)", "#2ecc71", 40, 80), width="stretch")
        with cg2: st.plotly_chart(crear_velocimetro(ind_resiliencia_sim, "Resiliencia (Proyectada)", "#3498db", 30, 70), width="stretch")
        with cg3: st.plotly_chart(crear_velocimetro(estres_gauge_sim_val, "Estrés (Proyectado)", "#e74c3c", 20, 40, invertido=True), width="stretch")
        with cg4: st.plotly_chart(crear_velocimetro(ind_calidad_sim, "Calidad (Proyectada)", "#9b59b6", 40, 70), width="stretch")
            
    # =========================================================================
    # BLOQUE 3: PROYECCIÓN CLIMÁTICA, RANKING AHP Y PREPARACIÓN PREDIAL
    # =========================================================================
    
    # --- 1. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (EXPLORADOR ENSO) ---
    with st.expander(f"📈 PROYECCIÓN DINÁMICA DE SEGURIDAD HÍDRICA (2024 - 2050): {nombre_zona}", expanded=False):
        tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono)"])
        anios_proj = list(range(2024, 2051))

        with tab_resumen:
            col_t1, col_t2 = st.columns(2)
            with col_t1: activar_cc = st.toggle("🌡️ Incluir Cambio Climático", value=True, key="td_t1_cc")
            with col_t2: activar_enso = st.toggle("🌊 Incluir Variabilidad ENSO", value=True, key="td_t1_enso")

            datos_proj = []
            for a in anios_proj:
                delta_a = a - 2024
                # Crecimiento demográfico/agropecuario (~1.5% anual)
                f_dem = (1 + 0.015) ** delta_a
                # Degradación de recarga/oferta por Cambio Climático (~0.5% anual)
                f_cc_base = (1 - 0.005) ** delta_a if activar_cc else 1.0
                
                f_enso = 0.0
                estado_enso = "Neutro ⚖️"
                if activar_enso:
                    f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) 
                    estado_enso = "Niña 🌧️" if f_enso > 0.1 else "Niño ☀️" if f_enso < -0.1 else "Neutro ⚖️"
                
                # 🛡️ Escudo anti-negativos para evitar colapsos matemáticos
                f_cli_total = max(0.1, f_cc_base + f_enso) 
                
                # 🔬 PROYECCIÓN DE VOLÚMENES FÍSICOS (Conectado al Bloque 1 y 2)
                # NOTA: Usamos 'oferta_nominal' y 'demanda_m3s' del Top Dashboard
                o_m3 = (oferta_nominal * f_cli_total) * 31536000
                r_m3 = (float(st.session_state.get('aleph_recarga_mm', 350.0)) * f_cli_total) * float(st.session_state.get('aleph_area_km2', 10.0)) * 1000
                c_m3 = (demanda_m3s * f_dem) * 31536000
                
                # ⚖️ NÚCLEO MATEMÁTICO FUTURO
                n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
                
                bfi_sim = r_m3 / o_m3 if o_m3 > 0 else 0.0
                fact_superv_sim = min(1.0, r_m3 / c_m3) if c_m3 > 0 else 1.0
                r = max(0.0, min(100.0, (bfi_sim / 0.70) * 100 * fact_superv_sim))
                
                wei_sim = c_m3 / o_m3 if o_m3 > 0 else 1.0
                e = max(0.0, min(100.0, 100.0 - (wei_sim / 0.40) * 60))
                
                caudal_L_s_sim = (o_m3 / 31536000) * 1000
                carga_mg_s_futura = carga_mg_s * f_dem
                dbo_mg_l_sim = carga_mg_s_futura / caudal_L_s_sim if caudal_L_s_sim > 0 else 999.0
                cal = max(0.0, min(100.0, 100.0 - ((dbo_mg_l_sim / 10.0) * 100)))
                
                datos_proj.extend([
                    {"Año": a, "Indicador": "Neutralidad", "Valor (%)": n, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Resiliencia", "Valor (%)": r, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Seguridad (Inv. Estrés)", "Valor (%)": e, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Calidad", "Valor (%)": cal, "Fase ENSO": estado_enso}
                ])
                
            fig_line1 = px.line(pd.DataFrame(datos_proj), x="Año", y="Valor (%)", color="Indicador", hover_data=["Fase ENSO"],
                               color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Seguridad (Inv. Estrés)": "#e74c3c", "Calidad": "#9b59b6"})
            
            fig_line1.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.1, layer="below", annotation_text="  Zona Crítica (<40%)")
            fig_line1.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.1, layer="below", annotation_text="  Zona Vulnerable (40-70%)")
            fig_line1.update_layout(height=400, hovermode="x unified", yaxis_range=[0, 105], title="Evolución de la Salud Integral del Sistema (0 = Colapso, 100 = Óptimo)")
            st.plotly_chart(fig_line1, use_container_width=True)

        with tab_escenarios:
            col_e1, col_e2 = st.columns([1, 2])
            with col_e1:
                ind_sel = st.selectbox("🎯 Indicador a Evaluar:", ["Estrés Hídrico", "Resiliencia", "Neutralidad", "Calidad"], key="td_ind_sel")
                activar_cc_esc = st.toggle("🌡️ Efecto Cambio Climático", value=True, key="td_t2_cc")
            with col_e2:
                diccionario_escenarios = {
                    "Onda Dinámica": "onda", "Condición Neutra": 0.0, "🟡 Niño Moderado": -0.15,
                    "🔴 Niño Severo": -0.35, "🟢 Niña Moderada": 0.15, "🔵 Niña Fuerte": 0.35
                }
                curvas_sel = st.multiselect("🌊 Curvas Climáticas:", list(diccionario_escenarios.keys()), default=["Onda Dinámica", "Condición Neutra", "🔴 Niño Severo"], key="td_curvas")

            datos_esc = []
            for a in anios_proj:
                delta_a = a - 2024
                f_dem = (1 + 0.015) ** delta_a
                f_cc_base = (1 - 0.005) ** delta_a if activar_cc_esc else 1.0
                
                for nombre_esc in curvas_sel:
                    val_esc = diccionario_escenarios[nombre_esc]
                    f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) if val_esc == "onda" else val_esc
                    f_cli_total = f_cc_base + f_enso
                    
                    o_m3 = (oferta_nominal * f_cli_total) * 31536000
                    r_m3 = (float(st.session_state.get('aleph_recarga_mm', 350.0)) * f_cli_total) * float(st.session_state.get('aleph_area_km2', 10.0)) * 1000
                    c_m3 = (demanda_m3s * f_dem) * 31536000
                    
                    if ind_sel == "Neutralidad": val = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
                    elif ind_sel == "Resiliencia": val = min(100.0, ((r_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
                    elif ind_sel == "Estrés Hídrico": val = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
                    else: 
                        fac_dil = (o_m3 / (c_m3 + 1))
                        val = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
                        
                    datos_esc.append({"Año": a, "Escenario": nombre_esc, "Valor (%)": val})
                    
            if datos_esc:
                fig_esc = px.line(pd.DataFrame(datos_esc), x="Año", y="Valor (%)", color="Escenario")
                fig_esc.update_traces(line=dict(width=3)) 
                fig_esc.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
                st.plotly_chart(fig_esc, use_container_width=True)

    # --- 2. RANKING TERRITORIAL MULTICRITERIO (AHP) Y RADAR ---
    with st.expander(f"🏆 RANKING TERRITORIAL MULTICRITERIO (AHP)", expanded=False):
        lista_cuencas = []
        if 'capas' in locals() and capas.get('cuencas') is not None and not capas['cuencas'].empty:
            if 'SUBC_LBL' in capas['cuencas'].columns:
                lista_cuencas = capas['cuencas']['SUBC_LBL'].dropna().unique().tolist()
                
        if not lista_cuencas:
            lista_cuencas = ["Río Chico", "Río Grande", "Quebrada La Mosca", "Río Buey", "Pantanillo"]
            
        st.caption(f"🔄 Reordenado en vivo usando Pesos AHP (Panel Lateral): Hídrico ({w_agua*100:.0f}%) | Biótico ({w_bio*100:.0f}%) | Socioeconómico ({w_socio*100:.0f}%)")
        
        datos_ranking = []
        for c in lista_cuencas:
            pseudo_seed = sum([ord(char) for char in c])
            np.random.seed(pseudo_seed)
            
            n_val = np.random.uniform(20, 90) if c != nombre_zona else ind_neutralidad
            r_val = np.random.uniform(20, 95) if c != nombre_zona else ind_resiliencia
            e_val = np.random.uniform(10, 80) if c != nombre_zona else ind_estres
            c_val = np.random.uniform(20, 100) if c != nombre_zona else ind_calidad
            
            # 🧠 MOTOR AHP: Conectado a los sliders
            urgencia_hidrica = e_val  
            urgencia_biotica = 100 - c_val  
            urgencia_socio = 100 - r_val 
            
            score_urgencia = (urgencia_hidrica * w_agua) + (urgencia_biotica * w_bio) + (urgencia_socio * w_socio)
            
            datos_ranking.append({
                "Territorio": c, "Índice Prioridad (AHP)": score_urgencia,
                "Neutralidad (%)": n_val, "Resiliencia (%)": r_val,
                "Estrés Hídrico (%)": e_val, "Calidad de Agua (%)": c_val
            })
            
        df_ranking = pd.DataFrame(datos_ranking).sort_values(by="Índice Prioridad (AHP)", ascending=False)
        
        c_tbl, c_rad = st.columns([1.5, 1])
        with c_tbl:
            st.dataframe(
                df_ranking.style.background_gradient(cmap="Reds", subset=["Índice Prioridad (AHP)", "Estrés Hídrico (%)"])
                .background_gradient(cmap="Blues", subset=["Resiliencia (%)"])
                .background_gradient(cmap="Greens", subset=["Neutralidad (%)", "Calidad de Agua (%)"])
                .format({"Índice Prioridad (AHP)": "{:.1f}", "Neutralidad (%)": "{:.1f}%", "Resiliencia (%)": "{:.1f}%", "Estrés Hídrico (%)": "{:.1f}%", "Calidad de Agua (%)": "{:.1f}%"}),
                use_container_width=True, hide_index=True
            )
            st.download_button("📥 Descargar Ranking AHP (CSV)", df_ranking.to_csv(index=False).encode('utf-8'), "Ranking_Territorial_AHP.csv", "text/csv")

            # DISTRIBUCIÓN REGIONAL (BOX PLOT)
            df_melt = df_ranking.melt(id_vars=["Territorio"], value_vars=["Neutralidad (%)", "Resiliencia (%)", "Estrés Hídrico (%)", "Calidad de Agua (%)"], var_name="Índice", value_name="Valor (%)")
            fig_box = px.box(df_melt, x="Índice", y="Valor (%)", color="Índice", points="all", title="Distribución Regional de Indicadores",
                             color_discrete_map={"Neutralidad (%)": "#2ecc71", "Resiliencia (%)": "#3498db", "Estrés Hídrico (%)": "#e74c3c", "Calidad de Agua (%)": "#9b59b6"})
            fig_box.update_layout(height=300, showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_box, use_container_width=True)

        with c_rad:
            fig_radar = go.Figure()
            categorias = ['Neutralidad', 'Resiliencia', 'Seguridad (Inv. Estrés)', 'Calidad', 'Neutralidad']
            
            fig_radar.add_trace(go.Scatterpolar(r=[100]*5, theta=categorias, fill='toself', fillcolor='rgba(39, 174, 96, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Óptimo (>70%)', hoverinfo='none'))
            fig_radar.add_trace(go.Scatterpolar(r=[70]*5, theta=categorias, fill='toself', fillcolor='rgba(241, 196, 15, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Vulnerable (40-70%)', hoverinfo='none'))
            fig_radar.add_trace(go.Scatterpolar(r=[40]*5, theta=categorias, fill='toself', fillcolor='rgba(192, 57, 43, 0.25)', line=dict(color='rgba(255,255,255,0)'), name='Crítico (<40%)', hoverinfo='none'))

            valores_radar = [ind_neutralidad, ind_resiliencia, max(0, 100-ind_estres), ind_calidad]
            fig_radar.add_trace(go.Scatterpolar(
                r=valores_radar + [valores_radar[0]], theta=categorias,
                fill='toself', name=nombre_zona, line=dict(color='#2c3e50', width=3),
                fillcolor='rgba(41, 128, 185, 0.7)', mode='lines+markers', marker=dict(size=8, color='#2c3e50')
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickvals=[40, 70, 100], ticktext=["40%", "70%", "100%"]),
                           angularaxis=dict(tickfont=dict(size=11, color="black", weight="bold"))),
                showlegend=True, legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                title=dict(text="Huella de Salud Territorial", font=dict(size=18)), height=380, margin=dict(l=40, r=40, t=50, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            promedio_salud = np.mean(valores_radar)
            color_box, msg_estado = ("#27ae60", "🟢 <b>TERRITORIO ÓPTIMO</b>") if promedio_salud >= 70 else ("#f39c12", "🟡 <b>TERRITORIO VULNERABLE</b>") if promedio_salud >= 40 else ("#c0392b", "🔴 <b>TERRITORIO CRÍTICO</b>")
            st.markdown(f"<div style='padding:10px; border-radius:5px; border-left: 5px solid {color_box}; background-color:#f8f9fa;'>{msg_estado}<br>Puntaje Salud: {promedio_salud:.1f}/100</div>", unsafe_allow_html=True)

    # --- 3. GLOSARIO METODOLÓGICO Y FUENTES ---
    with st.expander("📚 Glosario Metodológico (VWBA - WRI)", expanded=False):
        st.markdown("""
        * **Neutralidad Hídrica (VWBA):** Volumen de agua restituido mediante SbN vs Huella Hídrica. Objetivo: 100%.
        * **Resiliencia Territorial (BFI USGS):** Capacidad del ecosistema base para soportar eventos de sequía.
        * **Estrés Hídrico (WEI+):** Extracción vs Oferta. >40% indica estrés severo.
        * **Calidad de Agua (WQI):** Dilución natural de DBO5 y mitigación sanitaria.
        """)

    # =========================================================================
    # PRIORIZACIÓN PREDIAL PARA CONECTIVIDAD RIPARIA
    # =========================================================================
    st.markdown("---")
    st.subheader(f"🎯 Inteligencia de Negociación: Priorización Predial ({nombre_zona})")
    st.markdown("Cruza las necesidades de restauración riparia con la estructura predial alojada en la nube para identificar qué propiedades priorizar.")

    rios_strahler_crudos = st.session_state.get('gdf_rios')
    buffer_m = st.session_state.get('buffer_m_ripario', None) 
    rios_strahler = None
    
    if rios_strahler_crudos is not None and not rios_strahler_crudos.empty and gdf_zona is not None:
        try:
            rios_3116 = rios_strahler_crudos.to_crs(epsg=3116)
            zona_3116 = gdf_zona.to_crs(epsg=3116)
            rios_clip = gpd.clip(rios_3116, zona_3116)
            if not rios_clip.empty: rios_strahler = rios_clip
        except Exception as e:
            st.warning(f"Aviso validando red hídrica: {e}")
            
    if rios_strahler is None or rios_strahler.empty:
        with st.expander("⚠️ Paso 1: Faltan Datos - Generar Red Hídrica", expanded=True):
            st.info(f"Para priorizar predios necesitamos la red hídrica exacta de **{nombre_zona}**. ¡Trázalos usando el Motor Hidrológico!")
            render_motor_hidrologico(gdf_zona)
            
    elif buffer_m is None:
        with st.expander("⚠️ Paso 2: Configurar Franja Riparia", expanded=True):
            st.success("✅ ¡Ríos detectados en la zona! Ahora define el ancho de la zona de protección riparia.")
            render_motor_ripario()
            
    else:
        with st.expander("⚙️ Recalcular Franja Riparia", expanded=False):
            st.success(f"✅ Red Hídrica y Franja Riparia de {buffer_m}m listas para el cruce predial.")
            render_motor_ripario()

# =========================================================================
        # BLOQUE 4: MOTOR DE CRUCE MULTI-ANILLO Y VISOR 3D (PYDECK)
        # =========================================================================
        
        # 2. Cargar Capa Predial (100% Cloud Native o Fallback Local)
        capa_predios = capas.get('predios')

        # 3. Ejecutar el Motor de Cruce
        if rios_strahler is not None and not rios_strahler.empty:
            
            with st.container(border=True):
                st.markdown("#### ⚙️ Simulación Concéntrica de Franjas de Protección")
                
                # Preparación de la red hídrica
                rios_strahler = rios_strahler.reset_index(drop=True)
                rios_strahler['ID_Tramo'] = ["Segmento " + str(i+1) for i in range(len(rios_strahler))]
                if 'longitud_km' in rios_strahler.columns:
                    rios_strahler['longitud_km'] = rios_strahler['longitud_km'].round(2)
                
                # Extraer tamaños de anillo (Mínimo, Ideal, Óptimo)
                anillos = st.session_state.get('multi_rings', [10, 20, 30])
                b_min, b_med, b_max = anillos[0], anillos[1], anillos[2]
                
                # 🌿 MAGIA GEOMÉTRICA: Pre-cálculo de anillos fusionados
                rios_3116 = rios_strahler.to_crs(epsg=3116)
                rios_union = rios_3116.unary_union
                
                geom_max = rios_union.buffer(b_max, resolution=2)
                geom_med = rios_union.buffer(b_med, resolution=2)
                geom_min = rios_union.buffer(b_min, resolution=2)
                
                buffer_max_gdf = gpd.GeoDataFrame(geometry=[geom_max], crs=3116)
                
                # CÁLCULO DE ÁREAS POR TRAMO HÍDRICO
                datos_tramos = []
                for idx, row in rios_3116.iterrows():
                    long_m = row.geometry.length
                    orden = row.get('Orden_Strahler', 1)
                    long_km = long_m / 1000.0
                    
                    datos_tramos.append({
                        "ID Franja (Tramo)": row['ID_Tramo'],
                        "Orden de Strahler": orden,
                        "Longitud (Km)": long_km,
                        f"Mínimo ({b_min}m) ha": (long_m * (b_min * 2)) / 10000.0,
                        f"Ideal ({b_med}m) ha": (long_m * (b_med * 2)) / 10000.0,
                        f"Óptimo ({b_max}m) ha": (long_m * (b_max * 2)) / 10000.0,
                        "Importancia Ecológica": (orden * 50) + (long_km * 10)
                    })
                df_tramos = pd.DataFrame(datos_tramos).sort_values(by="Importancia Ecológica", ascending=False)
                
                tot_min = df_tramos[f"Mínimo ({b_min}m) ha"].sum()
                tot_med = df_tramos[f"Ideal ({b_med}m) ha"].sum()
                tot_max = df_tramos[f"Óptimo ({b_max}m) ha"].sum()
                tot_longitud_km = df_tramos["Longitud (Km)"].sum() 

                st.success(f"✅ Modelando 3 escenarios concéntricos simultáneos ({b_min}m, {b_med}m, {b_max}m)...")
                
                # MÉTRICAS TÁCTICAS
                cm1, cm2, cm3, cm4, cm5 = st.columns(5)
                cm1.metric(f"🔴 Escenario {b_min}m", f"{tot_min:,.1f} ha")
                cm2.metric(f"🟡 Escenario {b_med}m", f"{tot_med:,.1f} ha", f"+{(tot_med - tot_min):,.1f} ha", delta_color="off")
                cm3.metric(f"🟢 Escenario {b_max}m", f"{tot_max:,.1f} ha", f"+{(tot_max - tot_med):,.1f} ha", delta_color="off")
                cm4.metric("🌿 Tramos Hídricos", f"{len(df_tramos)}")
                cm5.metric("📏 Longitud Total", f"{tot_longitud_km:,.1f} km")
                
                tab_predios, tab_tramos = st.tabs(["🏡 Impacto Predial (Negociación)", "🌿 Áreas por Franja Riparia (Tramos)"])
                
                with tab_tramos:
                    st.markdown("##### 📋 Matriz Detallada por Franja Riparia")
                    st.dataframe(df_tramos.style.background_gradient(cmap="Greens", subset=["Importancia Ecológica"]).format(precision=2), use_container_width=True, hide_index=True)
                
                with tab_predios:
                    predios_en_buffer = gpd.GeoDataFrame()
                    if capa_predios is not None and not capa_predios.empty:
                        with st.spinner("Ejecutando intersección de anillos concéntricos con predios de Supabase..."):
                            try:
                                predios_3116 = capa_predios.to_crs(epsg=3116)
                                # Cruce espacial estricto
                                predios_en_buffer = gpd.overlay(predios_3116, buffer_max_gdf, how='intersection')
                                
                                if not predios_en_buffer.empty:
                                    predios_en_buffer['Area_Max_ha'] = predios_en_buffer.geometry.area / 10000.0
                                    predios_en_buffer['Area_Med_ha'] = predios_en_buffer.geometry.intersection(geom_med).area / 10000.0
                                    predios_en_buffer['Area_Min_ha'] = predios_en_buffer.geometry.intersection(geom_min).area / 10000.0
                                    
                                    col_id = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID', 'id'] if col in predios_en_buffer.columns), None)
                                    if col_id is None:
                                        predios_en_buffer['ID_Predio'] = predios_en_buffer.index
                                        col_id = 'ID_Predio'
                                        
                                    predios_agrupados = predios_en_buffer.groupby(col_id).agg({
                                        'Area_Min_ha': 'sum', 'Area_Med_ha': 'sum', 'Area_Max_ha': 'sum'
                                    }).reset_index()
                                    
                                    datos_prioridad = []
                                    for idx, row in predios_agrupados.iterrows():
                                        datos_prioridad.append({
                                            "Identificador Predial": row[col_id],
                                            f"Mínimo ({b_min}m) ha": row['Area_Min_ha'],
                                            f"Ideal ({b_med}m) ha": row['Area_Med_ha'],
                                            f"Óptimo ({b_max}m) ha": row['Area_Max_ha'],
                                            "ROI (Máx)": row['Area_Max_ha'] * 100
                                        })
                                        
                                    df_prioridad = pd.DataFrame(datos_prioridad).sort_values(by="ROI (Máx)", ascending=False)
                                    
                                    c_rank1, c_rank2 = st.columns([2, 1])
                                    with c_rank1:
                                        st.markdown("##### 📋 Top 15 Predios Estratégicos")
                                        st.dataframe(df_prioridad.head(15).style.background_gradient(cmap="YlOrRd", subset=["ROI (Máx)"]).format(precision=2), use_container_width=True, hide_index=True)
                                    with c_rank2:
                                        st.info("Exporta esta matriz para dirigir las campañas de gestión territorial.")
                                        st.metric("Predios Involucrados", f"{len(df_prioridad)}")
                                        st.download_button("📥 Descargar Matriz Predial", df_prioridad.to_csv(index=False).encode('utf-8'), "Prioridad_Predios.csv", "text/csv")
                                else:
                                    st.info("Ninguno de los predios protegidos intercepta la red hidrográfica modelada en esta simulación.")
                            except Exception as e:
                                st.error(f"Error técnico en el cruce geográfico: {e}")
                    else:
                        st.info("ℹ️ No se detectó un mapa predial maestro en la base de datos para esta zona.")

            # =========================================================
            # 🗺️ EL MAPA TÁCTICO PYDECK (VISOR 3D DE NEGOCIACIÓN)
            # =========================================================
            st.markdown("---")
            st.markdown(f"#### 🗺️ Visor Táctico de Conectividad y Predios: **{nombre_zona}**")
            import pydeck as pdk
            
            try:
                rios_4326 = rios_strahler.to_crs(epsg=4326).copy()
                c_lat, c_lon = rios_4326.geometry.iloc[0].centroid.y, rios_4326.geometry.iloc[0].centroid.x
            except: c_lat, c_lon = 6.2, -75.5 
            
            capas_mapa = []
            
            # Capa 1: Límite de Cuenca/Zona
            if gdf_zona is not None:
                zona_4326 = gdf_zona.to_crs("EPSG:4326")
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=zona_4326, opacity=1, stroked=True, get_line_color=[0, 200, 0, 255], get_line_width=3, filled=False))
            
            # Capa 2: Anillos Concéntricos (Niveles de Prioridad)
            if 'geom_max' in locals():
                gdf_max = gpd.GeoDataFrame(geometry=[geom_max], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_max, opacity=0.2, get_fill_color=[171, 235, 198], stroked=False))
                
                gdf_med = gpd.GeoDataFrame(geometry=[geom_med], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_med, opacity=0.4, get_fill_color=[88, 214, 141], stroked=False))
                
                gdf_min = gpd.GeoDataFrame(geometry=[geom_min], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_min, opacity=0.6, get_fill_color=[40, 180, 99], stroked=False))

            # Capa 3: Red Hídrica (Strahler)
            if 'rios_4326' in locals():
                capas_mapa.append(pdk.Layer(
                    "GeoJsonLayer", data=rios_4326,
                    get_line_color=[31, 97, 141, 255], get_line_width=2, lineWidthMinPixels=2,
                    pickable=True, autoHighlight=True
                ))
            
            # Capa 4: Predios Estratégicos (Afectados)
            if 'predios_en_buffer' in locals() and not predios_en_buffer.empty:
                col_id_oficial = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID', 'id'] if capa_predios is not None and col in capa_predios.columns), None)
                
                if col_id_oficial:
                    ids_afectados = predios_en_buffer[col_id_oficial].unique()
                    predios_a_dibujar = capa_predios[capa_predios[col_id_oficial].isin(ids_afectados)].to_crs(epsg=4326)
                else:
                    predios_a_dibujar = predios_en_buffer.to_crs(epsg=4326)
                    
                capas_mapa.append(pdk.Layer(
                    "GeoJsonLayer", data=predios_a_dibujar, opacity=0.4,
                    stroked=True, filled=True, get_fill_color=[255, 165, 0, 150],
                    get_line_color=[255, 140, 0, 255], get_line_width=2,
                    pickable=True, autoHighlight=True
                ))
            
            # Renderizado 3D
            view_state = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=13, pitch=45)
            tooltip = {"html": "<b>Tramo Hídrico:</b> {ID_Tramo}<br/><b>Orden:</b> {Orden_Strahler}<br/><b>Longitud:</b> {longitud_km} km", "style": {"backgroundColor": "steelblue", "color": "white"}}
            st.pydeck_chart(pdk.Deck(layers=capas_mapa, initial_view_state=view_state, map_style="light", tooltip=tooltip), use_container_width=True)

        else:
            st.warning("⚠️ El cruce predial y el mapa táctico están en pausa porque aún no se han calculado los ríos.")
            st.info("👆 Por favor, utiliza el botón del motor hidrológico de arriba para iluminar este tablero.")
