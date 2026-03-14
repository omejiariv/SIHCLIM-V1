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

    # ==============================================================================
    # 🧠 CEREBRO MAESTRO: LECTURA DE LA MEMORIA GLOBAL Y KPIs METROPOLITANOS
    # ==============================================================================
    lugar_actual = st.session_state.get('aleph_lugar', nombre_zona)
    anio_actual = st.session_state.get('aleph_anio', 2024)
    pob_total = st.session_state.get('aleph_pob_total', 4150000)
    demanda_m3s = st.session_state.get('demanda_total_m3s', 6.5) 
    fase_enso = st.session_state.get('enso_fase', 'Neutro')

    st.markdown("### 🎛️ Panel Global de Control y Monitoreo")
    
    # 1. Recuperamos la Oferta desde el Aleph (O ponemos 14.5 como respaldo)
    oferta_memoria = st.session_state.get('aleph_oferta_hidrica', 14.5)
    
    # 2. Cajón Manual Híbrido: Lee la memoria, pero permite sobreescritura manual
    with st.expander("⚙️ Calibración de Oferta Hídrica Base", expanded=False):
        oferta_base = st.number_input(
            "Oferta Nominal del Sistema (m³/s):", 
            value=float(oferta_memoria), 
            step=0.5,
            help="Dato heredado de Sistemas Hídricos. Puedes modificarlo aquí para simular sequías extremas o nuevas represas."
        )

    # 3. Motor de Estrés Hídrico (El ENSO castiga la oferta elegida)
    oferta_nominal = oferta_base 
    if "Niño Severo" in fase_enso: oferta_nominal *= 0.55
    elif "Niño Moderado" in fase_enso: oferta_nominal *= 0.75
    elif "Niña" in fase_enso: oferta_nominal *= 1.20

    estres_hidrico = (demanda_m3s / oferta_nominal) * 100 if oferta_nominal > 0 else 100
    st.session_state['estres_hidrico_global'] = estres_hidrico

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"👥 Población ({lugar_actual} - {anio_actual})", f"{int(pob_total):,.0f} hab")
    with col2:
        st.metric("💧 Demanda Continua", f"{demanda_m3s:,.2f} m³/s", "Humana + Pecuaria", delta_color="off")
    with col3:
        st.metric("🌍 Fase ENSO Actual", fase_enso)
    with col4:
        alerta_estres = "inverse" if estres_hidrico > 80 else "normal"
        if estres_hidrico > 100: alerta_estres = "inverse" 
        st.metric("⚠️ Estrés Hídrico", f"{estres_hidrico:,.1f} %", "Crítico" if estres_hidrico > 80 else "Estable", delta_color=alerta_estres)
    
    st.divider()

    # --- AHORA SÍ DIBUJAMOS LAS PESTAÑAS ---
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
    # TABLERO WRI, CALIDAD Y PROYECCIONES (INTEGRADO CON METABOLISMO Y ENSO)
    # =========================================================================
    with tab4:
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        st.subheader(f"🌐 Inteligencia Territorial (WRI): {nombre_zona}")
        st.markdown("Transforma las métricas biofísicas de la cuenca/municipio en indicadores estandarizados, evalúa portafolios de inversión y simula escenarios climáticos (ENSO).")
        
        # --- 1. RECUPERACIÓN DE DATOS BASE Y METABOLISMO ---
        # Recuperar Datos Base del Aleph (Oferta y Geometría)
        area_km2 = float(st.session_state.get('aleph_area_km2', 100.0))
        recarga_mm_base = float(st.session_state.get('aleph_recarga_mm', 350.0))
        q_oferta_m3s_base = float(st.session_state.get('aleph_q_rio_m3s', 5.0))
        
        # Recuperar Metabolismo de la Memoria (Demanda y Cargas inyectadas desde otras páginas)
        demanda_m3s_base = float(st.session_state.get('demanda_total_m3s', 0.5))
        carga_total_ton = float(st.session_state.get('carga_total_ton', 500.0))
        
        oferta_anual_m3 = q_oferta_m3s_base * 31536000
        recarga_anual_m3 = recarga_mm_base * area_km2 * 1000
        consumo_anual_m3 = demanda_m3s_base * 31536000

        # --- 2. INTEGRACIÓN CARTOGRÁFICA (PREDIOS EJECUTADOS) ---
        st.markdown("---")
        st.markdown(f"#### 🌲 Beneficios Volumétricos (SbN) en: **{nombre_zona}**")
        st.info("El sistema realiza un geoprocesamiento en vivo (clip espacial) para calcular las hectáreas exactas de los predios que caen dentro de los límites de la selección.")
        
        # Cálculo de Hectáreas Reales desde el SIG
        ha_reales_sig = 0.0
        if capas.get('predios') is not None and not capas['predios'].empty and gdf_zona is not None and not gdf_zona.empty:
            try:
                gdf_zona_4326 = gdf_zona.to_crs("EPSG:4326") if gdf_zona.crs != "EPSG:4326" else gdf_zona
                predios_4326 = capas['predios'].to_crs("EPSG:4326") if capas['predios'].crs != "EPSG:4326" else capas['predios']
                predios_en_cuenca = gpd.clip(predios_4326, gdf_zona_4326)
                if not predios_en_cuenca.empty:
                    ha_reales_sig = predios_en_cuenca.to_crs(epsg=3116).area.sum() / 10000.0
            except Exception as e:
                pass # Fallback silencioso
                
        # 🎚️ EL BOTÓN DE "REALIDAD ALTERNATIVA"
        activar_sig = st.toggle("✅ Incluir Área Restaurada del SIG en la línea base", value=True, key="td_toggle_sig",
                                help="Apaga este interruptor para simular el escenario contrafactual.")
        
        ha_base_calculo = ha_reales_sig if activar_sig else 0.0
        
        # --- 🌟 CONEXIÓN CON BOSQUES RIPARIOS (GEOMORFOLOGÍA) ---
        ha_riparias_potenciales = 0.0
        sumar_riparias = False
        df_str = st.session_state.get('geomorfo_strahler_df')
        
        if df_str is not None and not df_str.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🌿 Infraestructura Verde: Potencial de Reforestación Riparia", expanded=True):
                st.markdown("El modelo detectó una red hidrográfica calculada por la Ley de Horton. Dimensiona las franjas forestales protectoras (buffer) para integrarlas como **Soluciones Basadas en la Naturaleza (SbN)** al cálculo WRI.")
                
                c_rip1, c_rip2, c_rip3 = st.columns(3)
                ancho_buffer = c_rip1.number_input("Ancho de Aislamiento por lado (m):", min_value=5, max_value=100, value=30, step=5, key="td_buffer_rip")
                
                longitud_total_km = df_str['Longitud_Km'].sum()
                c_rip2.metric("Longitud Total de Cauces", f"{longitud_total_km:,.2f} km")
                
                # Fórmula Hectáreas: (Longitud (m) * Ancho_Total (m)) / 10,000
                ha_riparias_potenciales = (longitud_total_km * 1000 * (ancho_buffer * 2)) / 10000.0
                c_rip3.metric("Potencial Ripario (SbN)", f"{ha_riparias_potenciales:,.1f} ha", "Área disponible para restauración", delta_color="normal")
                
                sumar_riparias = st.checkbox("📥 Incorporar estas hectáreas riparias a la simulación financiera WRI", value=True, key="td_sumar_rip")
        else:
            st.info("💡 **Tip:** Visita el módulo de **Geomorfología** y calcula la red de drenaje para desbloquear el diseño automático de corredores riparios.")
        
        # --- RESUMEN DE BENEFICIOS VOLUMÉTRICOS ---
        st.markdown("<br>", unsafe_allow_html=True)
        c_inv1, c_inv2, c_inv3 = st.columns(3)
        with c_inv1:
            st.metric("✅ Área Conservada (Base SIG)", f"{ha_reales_sig:,.1f} ha")
            ha_simuladas = st.number_input("➕ Adicionar Hectáreas Extra (Manual):", min_value=0.0, value=0.0, step=10.0, key="td_ha_sim")
            
            # El cálculo final suma el SIG, las manuales y las riparias detectadas por el algoritmo
            ha_total = ha_base_calculo + ha_simuladas + (ha_riparias_potenciales if sumar_riparias else 0.0)
            beneficio_restauracion_m3 = ha_total * 2500
            
            if sumar_riparias and ha_riparias_potenciales > 0:
                st.caption(f"*(Incluye {ha_riparias_potenciales:,.1f} ha riparias)*")
            
        with c_inv2:
            sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM/PTAR):", min_value=0, value=50, step=5, key="td_stam")
            beneficio_calidad_m3 = sist_saneamiento * 1200
            
        with c_inv3:
            volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
            st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3:,.0f} m³/año", "Total compensado")

        # --- 3. PORTAFOLIOS DE INVERSIÓN (CANTIDAD Y CALIDAD) ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💼 Portafolios de Inversión Multi-Objetivo")

        # Portafolio 1: Neutralidad
        with st.expander("🎯 1. Optimización de Brechas: Oferta y Demanda (Neutralidad)", expanded=False):
            col_m1, col_m2 = st.columns([1, 2.5])
            with col_m1:
                meta_neutralidad = st.slider("Objetivo Neutralidad (%)", 10.0, 100.0, 100.0, 5.0, key="td_meta_n")
                costo_ha = st.number_input("Restauración (1 ha) [M COP]:", value=8.5, step=0.5, key="td_c_ha")
                costo_stam_n = st.number_input("Saneamiento (1 STAM) [M COP]:", value=15.0, step=1.0, key="td_c_stamn")
                costo_lps = st.number_input("Eficiencia (1 L/s) [M COP]:", value=120.0, step=10.0, key="td_c_lps")
            
            with col_m2:
                vol_requerido_m3 = (meta_neutralidad / 100.0) * consumo_anual_m3
                brecha_m3 = vol_requerido_m3 - volumen_repuesto_m3
                
                # --- ⚖️ CONTABILIDAD ESTRICTA: Cobrar los Proyectos Simulados ---
                ha_proyectos_simulados = ha_simuladas + (ha_riparias_potenciales if sumar_riparias else 0.0)
                costo_proyectos_simulados = ha_proyectos_simulados * costo_ha
                
                if brecha_m3 <= 0: 
                    st.success("✅ ¡Se cumple la meta de Neutralidad!")
                    # Aún si se cumple la meta, mostramos cuánto costó la simulación para llegar allí
                    if costo_proyectos_simulados > 0:
                        st.info(f"💰 Inversión en proyectos simulados (Riparios/Manuales): **${costo_proyectos_simulados:,.0f} Millones**")
                else:
                    st.warning(f"⚠️ Faltan compensar **{brecha_m3/1e6:,.2f} Mm³/año**.")
                    c_mix1, c_mix2, c_mix3 = st.columns(3)
                    pct_a = c_mix1.number_input("% Restauración", 0, 100, 40, key="td_pct_a")
                    pct_b = c_mix2.number_input("% Saneamiento", 0, 100, 40, key="td_pct_b")
                    pct_c = c_mix3.number_input("% Eficiencia", 0, 100, 20, key="td_pct_c")
                    
                    if (pct_a + pct_b + pct_c) == 100:
                        ha_req = (brecha_m3 * (pct_a/100)) / 2500.0
                        stam_req = (brecha_m3 * (pct_b/100)) / 1200.0
                        lps_req = ((brecha_m3 * (pct_c/100)) * 1000) / 31536000 
                        
                        # INVERSIÓN TOTAL = Lo que falta para la brecha + Lo que ya pusimos en la simulación
                        inv_brecha = (ha_req*costo_ha) + (stam_req*costo_stam_n) + (lps_req*costo_lps)
                        inv_total = inv_brecha + costo_proyectos_simulados
                        
                        st.markdown("📊 **Requerimientos Físicos y Presupuesto Final:**")
                        c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                        
                        # Sumamos las ha simuladas a las requeridas para mostrar el esfuerzo territorial total
                        c_op1.metric("🌲 Restaurar Total", f"{(ha_req + ha_proyectos_simulados):,.1f} ha", help="Incluye Riparias/Manuales + Brecha")
                        c_op2.metric("🚽 Saneamiento", f"{stam_req:,.0f} STAM")
                        c_op3.metric("🚰 Eficiencia", f"{lps_req:,.1f} L/s")
                        c_op4.metric("💰 INVERSIÓN TOTAL", f"${inv_total:,.0f} M", help="Suma el costo de la brecha más los proyectos riparios/manuales simulados.")
                    else: 
                        st.error("La suma debe ser 100%.")

        # Portafolio 2: Calidad
        with st.expander("🎯 2. Optimización de Cargas Contaminantes (Saneamiento DBO5)", expanded=False):
            col_c1, col_c2 = st.columns([1, 2.5])
            with col_c1:
                meta_remocion = st.slider("Meta Remoción DBO (%)", 10.0, 100.0, 85.0, 5.0, key="td_meta_c")
                costo_ptar = st.number_input("PTAR (1 Ton/a) [M COP]:", value=150.0, step=10.0, key="td_c_ptar")
                costo_stam_c = st.number_input("STAM (1 Ton/a) [M COP]:", value=45.0, step=5.0, key="td_c_stamc")
                costo_sbn = st.number_input("SbN (1 Ton/a) [M COP]:", value=12.0, step=2.0, key="td_c_sbn")
            with col_c2:
                carga_objetivo = (meta_remocion / 100.0) * carga_total_ton
                brecha_ton = carga_objetivo - (sist_saneamiento * 0.5) # Aprox 0.5 ton removidas por STAM existente
                
                if brecha_ton <= 0: st.success("✅ ¡Se cumple la meta de Remoción!")
                else:
                    st.warning(f"⚠️ Faltan remover **{brecha_ton:,.1f} Ton/año** de DBO5. (Base: {carga_total_ton:,.0f} Ton)")
                    c_mc1, c_mc2, c_mc3 = st.columns(3)
                    pct_ptar = c_mc1.number_input("% PTAR", 0, 100, 50, key="td_pct_ptar")
                    pct_stam_c = c_mc2.number_input("% STAM Rural", 0, 100, 30, key="td_pct_stam_c")
                    pct_sbn_c = c_mc3.number_input("% SbN Filtros", 0, 100, 20, key="td_pct_sbn_c")
                    
                    if (pct_ptar + pct_stam_c + pct_sbn_c) == 100:
                        t_ptar = brecha_ton * (pct_ptar/100)
                        t_stam = brecha_ton * (pct_stam_c/100)
                        t_sbn = brecha_ton * (pct_sbn_c/100)
                        inv_tot_c = (t_ptar*costo_ptar) + (t_stam*costo_stam_c) + (t_sbn*costo_sbn)
                        
                        st.markdown("📊 **Requerimientos de Remoción y Presupuesto:**")
                        c_oc1, c_oc2, c_oc3, c_oc4 = st.columns(4)
                        c_oc1.metric("🏙️ PTAR", f"{t_ptar:,.0f} Ton")
                        c_oc2.metric("🏡 STAM", f"{t_stam:,.0f} Ton")
                        c_oc3.metric("🌿 SbN", f"{t_sbn:,.0f} Ton")
                        c_oc4.metric("💰 INVERSIÓN", f"${inv_tot_c:,.0f} M")
                    else: st.error("La suma debe ser 100%.")

        # --- 4. MOTORES DE CÁLCULO ACTUALES Y VELOCÍMETROS ---
        ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_resiliencia = min(100.0, ((recarga_anual_m3 + oferta_anual_m3) / ((consumo_anual_m3+1) * 10)) * 100)
        ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0
        factor_dilucion = (oferta_anual_m3 / (consumo_anual_m3 + 1)) 
        ind_calidad = min(100.0, max(0.0, 50.0 + (factor_dilucion * 0.5) + (sist_saneamiento * 0.05)))
        
        st.markdown("---")
        st.subheader(f"🧭 Tablero de Seguridad Hídrica Integral: {nombre_zona}")
        
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
                number = {'suffix': "%", 'font': {'size': 26}}, title = {'text': titulo, 'font': {'size': 14}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': color_bar},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                        {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc"},
                        {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}
                }
            ))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            return fig

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        est_neu, col_neu = evaluar_indice(ind_neutralidad, 40, 80)
        est_res, col_res = evaluar_indice(ind_resiliencia, 30, 70)
        est_est, col_est = evaluar_indice(ind_estres, 40, 20, invertido=True)
        est_cal, col_cal = evaluar_indice(ind_calidad, 40, 70)

        with col_g1: 
            st.plotly_chart(crear_velocimetro(ind_neutralidad, "Neutralidad", "#2ecc71", 40, 80), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_neu}; margin-top:-20px;'>{est_neu}</h4>", unsafe_allow_html=True)
        with col_g2: 
            st.plotly_chart(crear_velocimetro(ind_resiliencia, "Resiliencia", "#3498db", 30, 70), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_res}; margin-top:-20px;'>{est_res}</h4>", unsafe_allow_html=True)
        with col_g3: 
            st.plotly_chart(crear_velocimetro(ind_estres, "Estrés Hídrico", "#e74c3c", 40, 20, invertido=True), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_est}; margin-top:-20px;'>{est_est}</h4>", unsafe_allow_html=True)
        with col_g4:
            st.plotly_chart(crear_velocimetro(ind_calidad, "Calidad del Agua", "#9b59b6", 40, 70), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_cal}; margin-top:-20px;'>{est_cal}</h4>", unsafe_allow_html=True)

        # --- 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (EXPLORADOR ENSO) ---
        st.markdown("---")
        st.subheader(f"📈 Proyección Dinámica de Seguridad Hídrica (2024 - 2050) - {nombre_zona}")
        
        tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono)"])
        anios_proj = list(range(2024, 2051))

        with tab_resumen:
            col_t1, col_t2 = st.columns(2)
            with col_t1: activar_cc = st.toggle("🌡️ Incluir Cambio Climático", value=True, key="td_t1_cc")
            with col_t2: activar_enso = st.toggle("🌊 Incluir Variabilidad ENSO", value=True, key="td_t1_enso")

            datos_proj = []
            for a in anios_proj:
                delta_a = a - 2024
                f_dem = (1 + 0.015) ** delta_a
                f_cc_base = (1 - 0.005) ** delta_a if activar_cc else 1.0
                
                f_enso = 0.0
                estado_enso = "Neutro ⚖️"
                if activar_enso:
                    f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) 
                    estado_enso = "Niña 🌧️" if f_enso > 0.1 else "Niño ☀️" if f_enso < -0.1 else "Neutro ⚖️"
                
                f_cli_total = f_cc_base + f_enso 
                o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
                r_m3 = (recarga_mm_base * f_cli_total) * area_km2 * 1000
                c_m3 = (demanda_m3s_base * f_dem) * 31536000
                
                n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
                r = min(100.0, ((r_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
                e = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
                fac_dil = (o_m3 / (c_m3 + 1))
                cal = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
                
                datos_proj.extend([
                    {"Año": a, "Indicador": "Neutralidad", "Valor (%)": n, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Resiliencia", "Valor (%)": r, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Estrés Hídrico", "Valor (%)": e, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Calidad", "Valor (%)": cal, "Fase ENSO": estado_enso}
                ])
                
            fig_line1 = px.line(pd.DataFrame(datos_proj), x="Año", y="Valor (%)", color="Indicador", hover_data=["Fase ENSO"],
                               color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Estrés Hídrico": "#e74c3c", "Calidad": "#9b59b6"})
            fig_line1.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.05, layer="below")
            fig_line1.update_layout(height=400, hovermode="x unified")
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
                    
                    o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
                    r_m3 = (recarga_mm_base * f_cli_total) * area_km2 * 1000
                    c_m3 = (demanda_m3s_base * f_dem) * 31536000
                    
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

        # --- 6. RANKING TERRITORIAL Y BOXPLOTS (INTACTO) ---
        st.markdown("---")
        st.subheader(f"🏆 Ranking Territorial y Dispersión de Índices - {nombre_zona}")
        
        lista_cuencas = []
        if capas['cuencas'] is not None and not capas['cuencas'].empty:
            if 'SUBC_LBL' in capas['cuencas'].columns:
                lista_cuencas = capas['cuencas']['SUBC_LBL'].dropna().unique().tolist()
                
        if not lista_cuencas:
            lista_cuencas = ["Río Chico", "Río Grande", "Quebrada La Mosca", "Río Buey", "Pantaniíllo"]
            
        np.random.seed(42) 
        datos_ranking = []
        for c in lista_cuencas:
            # Algoritmo de ranking simulado (en prod se conecta a los calculos base reales por cuenca)
            n_val = np.random.uniform(10, 90) if c != nombre_zona else ind_neutralidad
            r_val = np.random.uniform(20, 95) if c != nombre_zona else ind_resiliencia
            e_val = np.random.uniform(5, 60) if c != nombre_zona else ind_estres
            c_val = np.random.uniform(30, 100) if c != nombre_zona else ind_calidad
            score_urgencia = (e_val * 0.5) + ((100 - r_val) * 0.3) + ((100 - c_val) * 0.2)
            
            datos_ranking.append({
                "Territorio": c,
                "Urgencia Intervención": score_urgencia,
                "Neutralidad (%)": n_val,
                "Resiliencia (%)": r_val,
                "Estrés Hídrico (%)": e_val,
                "Calidad de Agua (%)": c_val
            })
            
        df_ranking = pd.DataFrame(datos_ranking).sort_values(by="Urgencia Intervención", ascending=False)
        
        c_tbl, c_box = st.columns([1.2, 1])
        with c_tbl:
            st.dataframe(
                df_ranking.style.background_gradient(cmap="Reds", subset=["Urgencia Intervención", "Estrés Hídrico (%)"])
                .background_gradient(cmap="Blues", subset=["Resiliencia (%)"])
                .background_gradient(cmap="Greens", subset=["Neutralidad (%)", "Calidad de Agua (%)"])
                .format({"Urgencia Intervención": "{:.1f}", "Neutralidad (%)": "{:.1f}%", "Resiliencia (%)": "{:.1f}%", "Estrés Hídrico (%)": "{:.1f}%", "Calidad de Agua (%)": "{:.1f}%"}),
                use_container_width=True, hide_index=True
            )
            csv_ranking = df_ranking.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Ranking (CSV)", csv_ranking, "Ranking_Territorial_WRI.csv", "text/csv")

        with c_box:
            df_melt = df_ranking.melt(id_vars=["Territorio"], value_vars=["Neutralidad (%)", "Resiliencia (%)", "Estrés Hídrico (%)", "Calidad de Agua (%)"], var_name="Índice", value_name="Valor (%)")
            fig_box = px.box(df_melt, x="Índice", y="Valor (%)", color="Índice", points="all",
                             title="Distribución Regional de Indicadores",
                             color_discrete_map={"Neutralidad (%)": "#2ecc71", "Resiliencia (%)": "#3498db", "Estrés Hídrico (%)": "#e74c3c", "Calidad de Agua (%)": "#9b59b6"})
            fig_box.update_layout(height=350, showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_box, use_container_width=True)

        # 7. GLOSARIO METODOLÓGICO Y FUENTES
        st.markdown("---")
        with st.expander("📚 Conceptos, Metodología y Fuentes (VWBA - WRI)", expanded=False):
            st.markdown("""
            ### 📖 Glosario de Indicadores
            
            * **Neutralidad Hídrica (Volumetric Water Benefit VWBA):**
              * **Concepto:** Mide si el volumen de agua restituido a la cuenca mediante Soluciones Basadas en la Naturaleza (SbN) compensa la Huella Hídrica del consumo humano/industrial.
              * **Interpretación:** Un 100% indica que se está reponiendo cada gota extraída. Valores $<40\%$ son críticos e implican deuda ecológica.
              * **Fórmula:** $\\frac{\\sum Beneficios\\ Volumétricos\\ (m^3/a)}{Consumo\\ Total\\ (m^3/a)} \\times 100$
              
            * **Resiliencia Territorial:**
              * **Concepto:** Capacidad del ecosistema (aguas subterráneas + escorrentía) para soportar eventos de sequía (El Niño) sin colapsar el suministro.
              * **Interpretación:** Zonas con alta recarga de acuíferos ($>70\%$) son buffers climáticos naturales. 
              
            * **Estrés Hídrico (Indicador Falkenmark / ODS 6.4.2):**
              * **Concepto:** Porcentaje de la oferta total anual que está siendo extraída por los diversos sectores económicos.
              * **Interpretación:** Valores $>40\%$ denotan estrés severo (competencia intensa por el recurso). Valores $<20\%$ indican un sistema holgado.

            * **Calidad de Agua (WQI):** Índice modificado basado en la capacidad de dilución natural (Oferta vs Extracción) y mitigación sanitaria (STAM).
              
            ### 🌐 Fuentes y Estándares de Referencia
            * **WRI (World Resources Institute):** [Volumetric Water Benefit Accounting (VWBA) - Metodología Oficial](https://www.wri.org/research/volumetric-water-benefit-accounting-vwba-implementing-guidelines)
            * **CEO Water Mandate:** Iniciativa del Pacto Global de Naciones Unidas para la resiliencia hídrica corporativa.
            * **Naciones Unidas:** Objetivo de Desarrollo Sostenible (ODS) 6.4.2 (Nivel de estrés hídrico).
            """)

        # =========================================================================
        # PRIORIZACIÓN PREDIAL PARA CONECTIVIDAD RIPARIA
        # =========================================================================
        st.markdown("---")
        
        # 🏷️ CORRECCIÓN 1: Inyectamos el nombre dinámico directamente desde tu selector
        nombre_zona_td = nombre_seleccion if 'nombre_seleccion' in locals() else "el Territorio"
        st.subheader(f"🎯 Priorización Predial: Inteligencia de Negociación en {nombre_zona}")
        st.markdown("Cruza las necesidades de restauración riparia con la estructura predial para identificar qué propiedades deben ser priorizadas.")

        # 1. Recuperar datos
        rios_strahler = st.session_state.get('gdf_rios')
        buffer_m = st.session_state.get('buffer_m_ripario', 30) 
        
        # 2. CARGAR LA CAPA DE PREDIOS
        capa_predios = None
        ruta_geojson_adquiridos = "data/PREDIOS_ADQUIRIDOS_ANTIOQUIA.geojson"
        ruta_shp_ant = "data/Predios_Ant.shp"
        
        try:
            if gdf_zona is not None:
                bbox = tuple(gdf_zona.to_crs(epsg=4326).total_bounds)
                if os.path.exists(ruta_geojson_adquiridos):
                    capa_predios = gpd.read_file(ruta_geojson_adquiridos, bbox=bbox)
                elif os.path.exists(ruta_shp_ant):
                    capa_predios = gpd.read_file(ruta_shp_ant, bbox=bbox)
                else:
                    capa_predios = capas.get('predios') if 'capas' in locals() else None
        except Exception as e:
            st.warning(f"No se pudo cargar la capa de predios local: {e}")

        # 3. EJECUTAR EL MOTOR DE CRUCE
        if rios_strahler is not None and not rios_strahler.empty:
            
            # 🛠️ CORRECCIÓN 2: EL "ANCLA" DE SINCRONIZACIÓN
            # Reseteamos el índice de la tabla para que la matemática y el mapa hablen el mismo idioma
            rios_strahler = rios_strahler.reset_index(drop=True)
            # Creamos un ID ÚNICO y permanente para cada tramo
            rios_strahler['ID_Tramo'] = ["Segmento " + str(i+1) for i in range(len(rios_strahler))]
            if 'longitud_km' in rios_strahler.columns:
                rios_strahler['longitud_km'] = rios_strahler['longitud_km'].round(2)
            
            if capa_predios is not None and not capa_predios.empty:
                # ---------------------------------------------------------
                # RUTA A: CON MAPA DE PREDIOS 
                # ---------------------------------------------------------
                st.success(f"✅ Estructura predial detectada. Calculando impacto ripario ({buffer_m}m por lado)...")
                with st.spinner("Ejecutando cruce espacial avanzado..."):
                    try:
                        predios_3116 = capa_predios.to_crs(epsg=3116)
                        rios_3116 = rios_strahler.to_crs(epsg=3116)
                        
                        buffer_tramos = gpd.GeoDataFrame(geometry=rios_3116.buffer(buffer_m, resolution=1), crs="EPSG:3116")
                        predios_en_buffer = gpd.overlay(predios_3116, buffer_tramos, how='intersection')
                        
                        if not predios_en_buffer.empty:
                            predios_en_buffer['Area_Riparia_ha'] = predios_en_buffer.geometry.area / 10000.0
                            col_id = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID'] if col in predios_en_buffer.columns), None)
                            if col_id is None:
                                predios_en_buffer['ID_Predio'] = predios_en_buffer.index
                                col_id = 'ID_Predio'
                                
                            predios_agrupados = predios_en_buffer.groupby(col_id).agg({'Area_Riparia_ha': 'sum'}).reset_index()
                            
                            datos_ranking = []
                            for idx, row in predios_agrupados.iterrows():
                                area_ha = row['Area_Riparia_ha']
                                datos_ranking.append({
                                    "Identificador Predial": row[col_id],
                                    "Área a Negociar (Buffer ha)": area_ha,
                                    "Índice de Prioridad (ROI)": area_ha * 100,
                                    "Estado": "Sin Contactar"
                                })
                                
                            df_prioridad = pd.DataFrame(datos_ranking).sort_values(by="Índice de Prioridad (ROI)", ascending=False)
                            
                            c_rank1, c_rank2 = st.columns([2, 1])
                            with c_rank1:
                                st.markdown("##### 📋 Top 10 Predios Estratégicos")
                                st.dataframe(df_prioridad.head(10).style.background_gradient(cmap="YlOrRd", subset=["Índice de Prioridad (ROI)"]).format({"Área a Negociar (Buffer ha)": "{:.2f}", "Índice de Prioridad (ROI)": "{:.0f}"}), use_container_width=True, hide_index=True)
                            with c_rank2:
                                st.markdown("##### 💡 Resumen Operativo")
                                st.metric("Predios Involucrados", f"{len(df_prioridad)}")
                                csv_predios = df_prioridad.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Descargar Plan de Negociación (CSV)", csv_predios, "Priorizacion_Predial.csv", "text/csv")
                        else:
                            st.info("Ninguno de los predios cargados intercepta la red hidrográfica calculada.")
                    except Exception as e:
                        st.error(f"Error en el cruce geográfico: {e}")
            else:
                # ---------------------------------------------------------
                # RUTA B: SIN PREDIOS (PRIORIZACIÓN ECOLÓGICA POR TRAMOS)
                # ---------------------------------------------------------
                st.info("ℹ️ No se detectó mapa predial. Activando **Priorización Ecológica por Tramos Hídricos**.")
                with st.spinner("Analizando jerarquía de conectividad del paisaje..."):
                    datos_tramos = []
                    for idx, row in rios_strahler.iterrows():
                        orden = row.get('Orden_Strahler', 1)
                        longitud = row.get('longitud_km', 0)
                        score_ecologico = (orden * 50) + (longitud * 10)
                        
                        datos_tramos.append({
                            "ID Tramo": row['ID_Tramo'], # <- Usamos el ID Anclado
                            "Orden de Strahler": orden,
                            "Longitud (Km)": longitud,
                            "Importancia Ecológica": score_ecologico,
                            "Rol en Conectividad": "Corredor Principal" if orden >= 3 else "Conector Secundario" if orden == 2 else "Área de Nacimiento"
                        })
                        
                    df_tramos = pd.DataFrame(datos_tramos).sort_values(by="Importancia Ecológica", ascending=False)
                    
                    c_tram1, c_tram2 = st.columns([2, 1])
                    with c_tram1:
                        st.markdown(f"##### 🌿 Tramos Críticos para Restauración en: **{nombre_zona}** (Top 10)")
                        st.markdown(f"##### 🗺️ Visor Táctico de Conectividad y Predios en: **{nombre_zona}**")
                        st.dataframe(df_tramos.head(10).style.background_gradient(cmap="Greens", subset=["Importancia Ecológica"]).format({"Longitud (Km)": "{:.2f}", "Importancia Ecológica": "{:.0f}"}), use_container_width=True, hide_index=True)
                    with c_tram2:
                        st.markdown("##### 💡 Resumen Ecológico")
                        tramos_clave = df_tramos[df_tramos['Orden de Strahler'] >= 2]
                        st.metric("Tramos Estratégicos (> Orden 1)", f"{len(tramos_clave)} segmentos")
                        st.caption("Estrategia: Al no contar con información predial, la prioridad de siembra se dirige a los tramos de mayor Orden de Strahler.")

            # =========================================================
            # 🗺️ EL MAPA TÁCTICO DE NEGOCIACIÓN 
            # =========================================================
            st.markdown("---")
            # 🏷️ CORRECCIÓN TÍTULO 3
            st.markdown(f"##### 🗺️ Visor Táctico de Conectividad y Predios en: **{nombre_zona}**")
            
            import pydeck as pdk
            
            # El mapa ahora lee la tabla maestra que ya tiene el 'ID_Tramo' correcto
            rios_4326 = rios_strahler.to_crs(epsg=4326).copy()
            
            try: c_lat, c_lon = rios_4326.geometry.iloc[0].centroid.y, rios_4326.geometry.iloc[0].centroid.x
            except: c_lat, c_lon = 6.2, -75.5
            
            capas_mapa = []
            
            if gdf_zona is not None:
                zona_4326 = gdf_zona.to_crs("EPSG:4326")
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=zona_4326, opacity=1, stroked=True, get_line_color=[0, 200, 0, 255], get_line_width=3, filled=False))
            
            # Capa de Ríos
            capas_mapa.append(pdk.Layer(
                "GeoJsonLayer",
                data=rios_4326,
                opacity=0.6,
                stroked=True,
                get_line_color=[39, 174, 96, 255], 
                get_line_width=buffer_m * 2,
                lineWidthUnits='"meters"',
                lineWidthMinPixels=2,
                pickable=True,
                autoHighlight=True
            ))
            
            # Capa de Predios
            if 'predios_en_buffer' in locals() and not predios_en_buffer.empty:
                col_id_pred = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID', 'ID_Predio'] if col in predios_en_buffer.columns), None)
                if col_id_pred and capa_predios is not None:
                    ids_afectados = predios_en_buffer[col_id_pred].unique()
                    predios_a_dibujar = capa_predios[capa_predios[col_id_pred].isin(ids_afectados)].to_crs(epsg=4326)
                    
                    capas_mapa.append(pdk.Layer(
                        "GeoJsonLayer",
                        data=predios_a_dibujar,
                        opacity=0.4,
                        stroked=True,
                        filled=True,
                        get_fill_color=[255, 165, 0, 150],
                        get_line_color=[255, 140, 0, 255],
                        get_line_width=2,
                        pickable=True,
                        autoHighlight=True
                    ))
            
            view_state = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=12)
            
            # Tooltip unificado que soporta información tanto de ríos como de predios
            tooltip = {
                "html": "<b>Detalle de Selección:</b><br/>{ID_Tramo} {MATRICULA} {COD_CATAST}<br/>Orden de Strahler: {Orden_Strahler}<br/>Longitud: {longitud_km} km",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
            st.pydeck_chart(pdk.Deck(layers=capas_mapa, initial_view_state=view_state, map_style="light", tooltip=tooltip), use_container_width=True)

        else:
            st.info("💡 Asegúrate de haber calculado la franja riparia de esta cuenca en la página de **Biodiversidad** primero.")




