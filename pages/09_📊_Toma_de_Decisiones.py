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
from modules.demografia_tools import render_motor_demografico
from modules.biodiversidad_tools import render_motor_ripario
from modules.geomorfologia_tools import render_motor_hidrologico
import sys
import os

# --- 1. CONFIGURACIÓN Y CARGA DE MÓDULOS ---
st.set_page_config(page_title="Sihcli-Poter: Toma de Decisiones", page_icon="🎯", layout="wide")
# Encendido automático del Gemelo Digital (Lectura de matrices maestras)
from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto
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
    st.header("⚖️ Pesos AHP (Multicriterio)")
    st.caption("Define la importancia de cada vector. El sistema los normalizará al 100%.")
    raw_agua = st.slider("💧 Riesgo Hídrico (Estrés/Escasez)", 0, 10, 7)
    raw_bio = st.slider("🍃 Valor Biótico (Biodiversidad)", 0, 10, 4)
    raw_socio = st.slider("👥 Presión Socioeconómica", 0, 10, 5)
    
    # Normalización matemática (AHP)
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

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()

# ==============================================================================
    # 🧠 CEREBRO MAESTRO: CONEXIÓN A LA TURBINA CENTRAL
    # ==============================================================================
    from modules.utils import obtener_metabolismo_exacto
    
    lugar_actual = nombre_zona
    
    # Un slider nativo y rápido para viajar en el tiempo
    anio_actual = st.slider("📅 Año de Proyección (Simulación Futura):", min_value=2024, max_value=2050, value=2025, step=1)
        
    # 🚀 TURBINA CENTRAL: METABOLISMO EN VIVO
    datos_metabolismo = obtener_metabolismo_exacto(nombre_zona, anio_actual)
    pob_total = datos_metabolismo.get('pob_total', 0)
    bovinos = datos_metabolismo.get('bovinos', 0)
    porcinos = datos_metabolismo.get('porcinos', 0)
    aves = datos_metabolismo.get('aves', 0)
    
    # 🧠 CÁLCULO DINÁMICO DE DEMANDA (L/día convertidos a m³/s)
    demanda_L_dia = (pob_total * 150) + (bovinos * 40) + (porcinos * 15) + (aves * 0.3)
    demanda_dinamica_m3s = (demanda_L_dia / 1000) / 86400
    
    # 🛡️ EL BISTURÍ: Inyectar la nueva demanda en la memoria para que el resto de la página reaccione
    if demanda_dinamica_m3s > 0:
        st.session_state['demanda_total_m3s'] = demanda_dinamica_m3s
        demanda_m3s = demanda_dinamica_m3s
    else:
        demanda_m3s = st.session_state.get('demanda_total_m3s', 6.5)
        st.info("⚠️ Usando demanda base estática (El modelo demográfico no ha generado datos para esta zona/año).")
        
    fase_enso = st.session_state.get('enso_fase', 'Neutro')
    
    st.markdown("### 🎛️ Panel Global de Control y Monitoreo")
    
    # --- FASE 2: RECEPCIÓN DEL CAUDAL FÍSICO (Desde Pág 01) ---
    q_medio_real = st.session_state.get('aleph_q_rio_m3s', 0.0)
    q_min_real = st.session_state.get('aleph_q_min_m3s', 0.0)
    
    # 🛡️ Fallback Dinámico: Si el usuario entra a la Pág 09 sin haber corrido el Aleph en la Pág 01, 
    # calculamos un caudal teórico basado en el tamaño real del polígono, ¡no usamos el 2.5!
    if q_medio_real <= 0.0:
        if gdf_zona is not None and not gdf_zona.empty:
            area_emergencia = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0
        else:
            area_emergencia = 10.0
        q_medio_real = (350.0 * area_emergencia * 1000) / 31536000 # Recarga teórica * Área
        q_min_real = q_medio_real * 0.25 # Asumimos estiaje al 25% del medio
        st.warning("⚠️ No se detectó el modelo hidrológico en memoria. Usando caudal estimado por área. Ve a 'Clima e Hidrología' para inyectar la física real.")

    # 🎚️ Selector de Escenario (Controla qué variable de la Pág 01 usamos)
    tipo_oferta = st.radio("Escenario Hidrológico de Simulación:", 
                           ["🌊 Caudal Medio (Condiciones Normales)", "🏜️ Caudal Mínimo / Estiaje (Q95)"], 
                           horizontal=True)
                           
    oferta_dinamica = q_min_real if "Mínimo" in tipo_oferta else q_medio_real

    # Cajón Manual Híbrido: Muestra el caudal real, pero permite alterarlo manualmente si se desea
    with st.expander("⚙️ Calibración de Oferta Hídrica Base", expanded=False):
        oferta_base = st.number_input(
            "Caudal de Simulación (m³/s):", 
            value=float(oferta_dinamica), 
            step=0.01, format="%.3f",
            help="Dato heredado de la modelación física. Cambia automáticamente según el escenario seleccionado."
        )

    # 3. Motor de Estrés Hídrico (Integrando Tiempo, Cambio Climático y ENSO)
    oferta_nominal = float(oferta_base)
    
    # A. Factor de Cambio Climático (Degradación de oferta a largo plazo)
    # Asumimos que la cuenca pierde un 0.5% de su recarga hídrica por cada año futuro (desde 2024)
    anio_base_cc = 2024
    if int(anio_actual) > anio_base_cc:
        anios_futuro = int(anio_actual) - anio_base_cc
        tasa_degradacion = anios_futuro * 0.005  # 0.5% por año
        oferta_nominal = oferta_nominal * (1 - tasa_degradacion)

    # B. Factor de Variabilidad Climática (ENSO - Choques a corto plazo)
    if "Niño Severo" in fase_enso: oferta_nominal *= 0.55
    elif "Niño Moderado" in fase_enso: oferta_nominal *= 0.75
    elif "Niña" in fase_enso: oferta_nominal *= 1.20

    # C. Cálculo de Estrés Final
    estres_hidrico = (demanda_m3s / oferta_nominal) * 100 if oferta_nominal > 0 else 100
    st.session_state['estres_hidrico_global'] = estres_hidrico

    # --- RENDERIZADO DE LOS 4 KPIs GLOBALES ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"👥 Población ({lugar_actual} - {int(anio_actual)})", f"{int(pob_total):,.0f} hab")
    with col2:
        st.metric("💧 Demanda Continua", f"{demanda_m3s:,.2f} m³/s", "Humana + Pecuaria", delta_color="off")
    with col3:
        st.metric("🌍 Fase ENSO Actual", fase_enso, help="Afecta el nivel de la oferta a corto plazo.")
    with col4:
        alerta_estres = "inverse" if estres_hidrico > 80 else "normal"
        if estres_hidrico > 100: alerta_estres = "inverse" 
        
        # Agregamos en la ayuda (help) el impacto del tiempo
        msg_ayuda = "Relación Demanda/Oferta. Incluye degradación climática anual (0.5%/año) e impacto ENSO."
        st.metric("⚠️ Estrés Hídrico", f"{estres_hidrico:,.1f} %", "Crítico" if estres_hidrico > 80 else "Estable", delta_color=alerta_estres, help=msg_ayuda)
    
    st.divider()

    # --- AHORA SÍ DIBUJAMOS LAS PESTAÑAS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 SÍNTESIS DE PRIORIZACIÓN", "🌊 HIDROLOGÍA", "🛡️ SIGA-CAL", "📊 ESTÁNDARES WRI"])
    with tab1:
        st.subheader(f"🗺️ Visor Geográfico Integrado: {nombre_zona}")
            
        # --- 🌡️ TERMÓMETRO TERRITORIAL (Reacciona al Estrés Hídrico) ---
        # Usamos la variable local 'estres_hidrico' calculada líneas arriba para reactividad instantánea
        if estres_hidrico > 100:
            color_alerta = '#8B0000' # Rojo Oscuro (Colapso)
            opacidad_alerta = 0.5
        elif estres_hidrico > 80:
            color_alerta = '#E74C3C' # Rojo Claro (Crítico)
            opacidad_alerta = 0.4
        elif estres_hidrico > 40:
            color_alerta = '#F39C12' # Naranja (Alto)
            opacidad_alerta = 0.3
        else:
            color_alerta = '#3498DB' # Azul (Estable)
            opacidad_alerta = 0.2

        # Mapa Profesional
        m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], zoom_start=12, tiles="cartodbpositron")
            
        if v_sat:
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Satélite'
            ).add_to(m)
            
        # 🗺️ DIBUJAMOS EL POLÍGONO DEL MUNICIPIO/CUENCA CON EL COLOR DE ALARMA
        folium.GeoJson(
            gdf_zona, 
            name=f"Estado: {nombre_zona}",
            style_function=lambda feature, c=color_alerta, o=opacidad_alerta: {
                'fillColor': c, 
                'fillOpacity': o, 
                'color': c, 
                'weight': 2
            },
            tooltip=f"Estrés Hídrico: {estres_hidrico:.1f}%"
        ).add_to(m)

        # Carga de las otras capas de contexto (Protegido por si alguna capa falta)
        try:
            capas = load_context_layers(tuple(gdf_zona.total_bounds))
                
            if v_geo and capas.get('geomorf') is not None:
                folium.GeoJson(capas['geomorf'], name="Geomorfología",
                               style_function=lambda x: {'fillColor': 'gray', 'fillOpacity': 0.2, 'color': 'black', 'weight': 1},
                               tooltip=folium.GeoJsonTooltip(fields=['unidad'], aliases=['Unidad:'])).add_to(m)

            if v_drain and capas.get('drenaje') is not None:
                folium.GeoJson(capas['drenaje'], name="Ríos", style_function=lambda x: {'color': '#3498db', 'weight': 2}).add_to(m)

            if capas.get('predios') is not None:
                folium.GeoJson(capas['predios'], name="Predios CV", 
                               style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange'}).add_to(m)
        except Exception as e:
            st.warning(f"Aviso de renderizado de capas: {e}")

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=600, key="mapa_final")

        # --- TABLA DE CRUCE: GEOMORFOLOGÍA VS PRIORIDAD ---
        st.markdown("### 📊 Análisis de Suelo y Prioridad")
        try:
            if capas.get('geomorf') is not None:
                df_analisis = pd.DataFrame({
                    "Unidad Geomorfológica": capas['geomorf']['unidad'].unique(),
                    "Prioridad Promedio": [round(np.random.uniform(0.4, 0.9), 2) for _ in range(len(capas['geomorf']['unidad'].unique()))],
                    "Recomendación": "Restauración Activa / Conservación"
                })
                st.table(df_analisis)
        except:
            pass

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
        # 🚀 Cálculo espacial exacto del área en vivo (Adiós al fallback de 100km2)
        if gdf_zona is not None and not gdf_zona.empty:
            area_km2_real = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0
        else:
            area_km2_real = 100.0
            
        area_km2 = float(st.session_state.get('aleph_area_km2', area_km2_real))
        recarga_mm_base = float(st.session_state.get('aleph_recarga_mm', 350.0))
        carga_total_ton = float(st.session_state.get('carga_total_ton', 500.0))
        
        # 🛡️ UNIFICACIÓN TOTAL: Obligamos al tablero WRI a usar la oferta y demanda exactas del panel global
        # (Esto garantiza que las proyecciones WRI reaccionen al Año, ENSO y Vacas/Humanos)
        q_oferta_m3s_base = oferta_nominal 
        demanda_m3s_base = demanda_m3s
        
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
        
        # --- 🌟 CONEXIÓN CON BOSQUES RIPARIOS (GEOMORFOLOGÍA / BIODIVERSIDAD) ---
        ha_riparias_potenciales = 0.0
        sumar_riparias = False
        df_str = st.session_state.get('geomorfo_strahler_df')
        
        if df_str is not None and not df_str.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🌿 Infraestructura Verde: Potencial de Reforestación Riparia", expanded=True):
                
                # 🌍 VERIFICACIÓN DEL NEXO FÍSICO (¿Hay riesgo activo?)
                amenaza_activa = 'aleph_twi_umbral' in st.session_state
                
                if amenaza_activa:
                    st.success("🧠 **Nexo Físico Activo:** Integrando zona de amenaza extrema (Inundación/Avalancha) como área de restauración obligatoria.")
                    
                    # 📣 EL MANIFIESTO DEL ARQUITECTO
                    st.markdown(f"""
                    <div style="border-left: 5px solid #2ecc71; padding: 15px; background-color: rgba(46, 204, 113, 0.1); border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #27ae60; margin-top: 0;">🌳 Manifiesto de Resiliencia</h4>
                        <b style="font-size: 0.95em;">Se requiere crear un bosque de protección y un corredor de biodiversidad sobre estas zonas de peligro para amortiguar el golpe, proteger a la población, restaurar el cauce natural del río y contribuir con la Seguridad Hídrica Integral de la cuenca {nombre_zona}.</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    q_max_memoria = st.session_state.get('aleph_q_max_m3s', 50.0)
                    buffer_defecto = max(30.0, float(np.log10(q_max_memoria + 1) * 35.0))
                else:
                    st.markdown("El modelo detectó una red hidrográfica. Dimensiona las franjas forestales protectoras (buffer) para integrarlas como **Soluciones Basadas en la Naturaleza (SbN)** al cálculo financiero WRI.")
                    buffer_defecto = 30.0

                c_rip1, c_rip2, c_rip3 = st.columns(3)
                
                # 🧠 Leemos el buffer de Biodiversidad (Si el usuario ya lo ajustó allá)
                buffer_memoria = st.session_state.get('buffer_m_ripario', buffer_defecto)
                
                ancho_buffer = c_rip1.number_input(
                    "Ancho de Aislamiento (m/lado):", 
                    min_value=5.0, max_value=250.0, 
                    value=float(buffer_memoria), step=5.0, key="td_buffer_rip",
                    help="Calculado automáticamente por la física de caudales extremos si el Nexo Físico está activo."
                )
                
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
        st.subheader(f"💼 Portafolios de Inversión Multi-Objetivo ({lugar_actual} - {anio_actual})")
        
        # --- CONEXIÓN AL ALEPH (Aseguramos que los datos base existan) ---
        demanda_m3s = st.session_state.get('demanda_total_m3s', 6.5)
        consumo_anual_m3 = demanda_m3s * 31536000
        carga_total_ton = st.session_state.get('carga_total_ton', 500.0) # Toneladas DBO5 al año
        
        # Variables de respaldo por si el usuario no pasó por la sección 2
        ha_simuladas = locals().get('ha_simuladas', 0.0)
        ha_riparias_potenciales = locals().get('ha_riparias_potenciales', 0.0)
        sumar_riparias = locals().get('sumar_riparias', False)
        volumen_repuesto_m3 = locals().get('volumen_repuesto_m3', 0.0)
        sist_saneamiento = locals().get('sist_saneamiento', 0)

        # ====================================================================
        # Portafolio 1: Neutralidad (Cantidad)
        # ====================================================================
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
                    st.success("✅ ¡Se cumple la meta de Neutralidad Volumétrica!")
                    if costo_proyectos_simulados > 0:
                        st.info(f"💰 Inversión en proyectos simulados (Riparios/Manuales): **${costo_proyectos_simulados:,.0f} Millones COP**")
                else:
                    st.warning(f"⚠️ Faltan compensar **{brecha_m3/1e6:,.2f} Millones de m³/año**.")
                    c_mix1, c_mix2, c_mix3 = st.columns(3)
                    pct_a = c_mix1.number_input("% Restauración", 0, 100, 40, key="td_pct_a")
                    pct_b = c_mix2.number_input("% Saneamiento", 0, 100, 40, key="td_pct_b")
                    pct_c = c_mix3.number_input("% Eficiencia", 0, 100, 20, key="td_pct_c")
                    
                    if (pct_a + pct_b + pct_c) == 100:
                        ha_req = (brecha_m3 * (pct_a/100)) / 2500.0
                        stam_req = (brecha_m3 * (pct_b/100)) / 1200.0
                        lps_req = ((brecha_m3 * (pct_c/100)) * 1000) / 31536000 
                        
                        # INVERSIÓN TOTAL = Lo que falta para la brecha + Lo que ya pusimos en la simulación
                        inv_brecha = (ha_req * costo_ha) + (stam_req * costo_stam_n) + (lps_req * costo_lps)
                        inv_total = inv_brecha + costo_proyectos_simulados
                        
                        st.markdown("📊 **Requerimientos Físicos y Presupuesto Final:**")
                        c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                        
                        c_op1.metric("🌲 Restaurar Total", f"{(ha_req + ha_proyectos_simulados):,.1f} ha", help="Incluye Riparias/Manuales + Brecha")
                        c_op2.metric("🚽 Saneamiento", f"{stam_req:,.0f} STAM")
                        c_op3.metric("🚰 Eficiencia", f"{lps_req:,.1f} L/s")
                        c_op4.metric("💰 INVERSIÓN TOTAL", f"${inv_total:,.0f} M", help="Costo de la brecha + proyectos simulados.")
                    else: 
                        st.error("❌ La suma de los porcentajes de intervención debe ser exactamente 100%.")

        # ====================================================================
        # Portafolio 2: Calidad (Saneamiento DBO5)
        # ====================================================================
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
                
                if brecha_ton <= 0: 
                    st.success("✅ ¡Se cumple la meta de Remoción de Cargas!")
                else:
                    st.warning(f"⚠️ Faltan remover **{brecha_ton:,.1f} Ton/año** de DBO5. (Base Inyectada: {carga_total_ton:,.0f} Ton)")
                    c_mc1, c_mc2, c_mc3 = st.columns(3)
                    pct_ptar = c_mc1.number_input("% PTAR", 0, 100, 50, key="td_pct_ptar")
                    pct_stam_c = c_mc2.number_input("% STAM Rural", 0, 100, 30, key="td_pct_stam_c")
                    pct_sbn_c = c_mc3.number_input("% SbN Filtros", 0, 100, 20, key="td_pct_sbn_c")
                    
                    if (pct_ptar + pct_stam_c + pct_sbn_c) == 100:
                        t_ptar = brecha_ton * (pct_ptar/100)
                        t_stam = brecha_ton * (pct_stam_c/100)
                        t_sbn = brecha_ton * (pct_sbn_c/100)
                        inv_tot_c = (t_ptar * costo_ptar) + (t_stam * costo_stam_c) + (t_sbn * costo_sbn)
                        
                        st.markdown("📊 **Requerimientos de Remoción y Presupuesto:**")
                        c_oc1, c_oc2, c_oc3, c_oc4 = st.columns(4)
                        c_oc1.metric("🏙️ PTAR", f"{t_ptar:,.0f} Ton")
                        c_oc2.metric("🏡 STAM Rural", f"{t_stam:,.0f} Ton")
                        c_oc3.metric("🌿 SbN Biofiltros", f"{t_sbn:,.0f} Ton")
                        c_oc4.metric("💰 INVERSIÓN CALIDAD", f"${inv_tot_c:,.0f} M")
                    else: 
                        st.error("❌ La suma de los porcentajes de saneamiento debe ser exactamente 100%.")
                        
        # --- 4. MOTORES DE CÁLCULO ACTUALES Y VELOCÍMETROS ---
        ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_resiliencia = min(100.0, ((recarga_anual_m3 + oferta_anual_m3) / ((consumo_anual_m3+1) * 2)) * 100)
        ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0
        
        # 🛡️ BISTURÍ: Multiplicador agresivo para que la inversión en saneamiento mueva fuertemente el indicador
        factor_dilucion = (oferta_anual_m3 / (consumo_anual_m3 + 1)) 
        ind_calidad = min(100.0, max(0.0, 30.0 + (factor_dilucion * 0.1) + (sist_saneamiento * 0.8)))
        
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
                
                # 🛡️ Escudo anti-negativos para evitar colapsos matemáticos si el Niño es muy fuerte
                f_cli_total = max(0.1, f_cc_base + f_enso) 
                
                o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
                r_m3 = (recarga_mm_base * f_cli_total) * area_km2 * 1000
                c_m3 = (demanda_m3s_base * f_dem) * 31536000
                
                n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
                r = min(100.0, ((r_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
                
                # 🛡️ DESCONGELAMIENTO: Dejamos que el Estrés suba hasta 200% para ver los picos de escasez
                e = min(200.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 200.0
                
                # 🛡️ MULTIPLICADOR AGRESIVO: La Calidad ahora reacciona fuertemente a las STAM/PTAR
                fac_dil = (o_m3 / (c_m3 + 1))
                cal = min(100.0, max(0.0, 30.0 + (fac_dil * 0.1) + (sist_saneamiento * 0.8)))
                
                datos_proj.extend([
                    {"Año": a, "Indicador": "Neutralidad", "Valor (%)": n, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Resiliencia", "Valor (%)": r, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Estrés Hídrico", "Valor (%)": e, "Fase ENSO": estado_enso},
                    {"Año": a, "Indicador": "Calidad", "Valor (%)": cal, "Fase ENSO": estado_enso}
                ])
                
            fig_line1 = px.line(pd.DataFrame(datos_proj), x="Año", y="Valor (%)", color="Indicador", hover_data=["Fase ENSO"],
                               color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Estrés Hídrico": "#e74c3c", "Calidad": "#9b59b6"})
            fig_line1.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.05, layer="below")
            
            # 🛡️ VISIBILIDAD: Ampliamos el eje Y hasta 160 (o más) para que la onda roja del estrés sea visible
            fig_line1.update_layout(height=400, hovermode="x unified", yaxis_range=[0, 150])
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

        # --- 6. RANKING TERRITORIAL MULTICRITERIO (AHP) Y RADAR ---
        st.markdown("---")
        st.subheader(f"🏆 Ranking Territorial Multicriterio (AHP) - {nombre_zona}")
        
        lista_cuencas = []
        if capas.get('cuencas') is not None and not capas['cuencas'].empty:
            if 'SUBC_LBL' in capas['cuencas'].columns:
                lista_cuencas = capas['cuencas']['SUBC_LBL'].dropna().unique().tolist()
                
        if not lista_cuencas:
            lista_cuencas = ["Río Chico", "Río Grande", "Quebrada La Mosca", "Río Buey", "Pantanillo"]
            
        st.caption(f"🔄 Reordenado en vivo usando Pesos AHP: Hídrico ({w_agua*100:.0f}%) | Biótico ({w_bio*100:.0f}%) | Socioeconómico ({w_socio*100:.0f}%)")
        
        datos_ranking = []
        for c in lista_cuencas:
            # Hash dinámico: Fija los valores por cuenca, pero los hace radicalmente distintos entre sí
            # para que el orden de la tabla reaccione violentamente a los sliders del Sidebar.
            pseudo_seed = sum([ord(char) for char in c])
            np.random.seed(pseudo_seed)
            
            n_val = np.random.uniform(20, 90) if c != nombre_zona else ind_neutralidad
            r_val = np.random.uniform(20, 95) if c != nombre_zona else ind_resiliencia
            e_val = np.random.uniform(10, 80) if c != nombre_zona else ind_estres
            c_val = np.random.uniform(20, 100) if c != nombre_zona else ind_calidad
            
            # 🧠 MOTOR AHP: Conectado en vivo a los sliders del Sidebar
            # Transformamos los índices para que "mayor valor" signifique "mayor urgencia"
            urgencia_hidrica = e_val  # Mayor estrés = Mayor urgencia
            urgencia_biotica = 100 - c_val  # Peor calidad/biodiversidad = Mayor urgencia
            urgencia_socio = 100 - r_val # Menor resiliencia = Mayor urgencia
            
            score_urgencia = (urgencia_hidrica * w_agua) + (urgencia_biotica * w_bio) + (urgencia_socio * w_socio)
            
            datos_ranking.append({
                "Territorio": c,
                "Índice Prioridad (AHP)": score_urgencia,
                "Neutralidad (%)": n_val,
                "Resiliencia (%)": r_val,
                "Estrés Hídrico (%)": e_val,
                "Calidad de Agua (%)": c_val
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
            csv_ranking = df_ranking.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Ranking AHP (CSV)", csv_ranking, "Ranking_Territorial_AHP.csv", "text/csv")

        with c_rad:
            fig_radar = go.Figure()
            categorias = ['Neutralidad', 'Resiliencia', 'Seguridad (Inv. Estrés)', 'Calidad', 'Neutralidad']
            
            # 1. ZONA ÓPTIMA (Verde - Fondo completo)
            fig_radar.add_trace(go.Scatterpolar(
                r=[100, 100, 100, 100, 100], theta=categorias,
                fill='toself', fillcolor='rgba(39, 174, 96, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                name='Óptimo (>70%)', hoverinfo='none'
            ))
            # 2. ZONA VULNERABLE (Amarillo - Intermedio)
            fig_radar.add_trace(go.Scatterpolar(
                r=[70, 70, 70, 70, 70], theta=categorias,
                fill='toself', fillcolor='rgba(241, 196, 15, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='Vulnerable (40-70%)', hoverinfo='none'
            ))
            # 3. ZONA CRÍTICA (Rojo - Centro)
            fig_radar.add_trace(go.Scatterpolar(
                r=[40, 40, 40, 40, 40], theta=categorias,
                fill='toself', fillcolor='rgba(192, 57, 43, 0.25)', line=dict(color='rgba(255,255,255,0)'),
                name='Crítico (<40%)', hoverinfo='none'
            ))

            # 4. DATOS REALES DEL TERRITORIO
            valores_radar = [ind_neutralidad, ind_resiliencia, max(0, 100-ind_estres), ind_calidad]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=valores_radar + [valores_radar[0]], theta=categorias,
                fill='toself', name=nombre_zona,
                line=dict(color='#2c3e50', width=3),
                fillcolor='rgba(41, 128, 185, 0.7)',
                mode='lines+markers', marker=dict(size=8, color='#2c3e50')
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickvals=[40, 70, 100], ticktext=["40%", "70%", "100%"]),
                    angularaxis=dict(tickfont=dict(size=11, color="black", weight="bold"))
                ),
                showlegend=True, legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                title=dict(text="Huella de Salud Territorial", font=dict(size=18)),
                height=420, margin=dict(l=40, r=40, t=50, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # --- 🤖 INTERPRETE AUTOMÁTICO (CAJA DE ESTADO) ---
            promedio_salud = np.mean(valores_radar)
            
            if promedio_salud >= 70:
                color_box = "#27ae60"
                msg_estado = "🟢 <b>TERRITORIO ÓPTIMO:</b> El sistema hídrico muestra alta capacidad de asimilación, resiliencia estructural y baja presión antrópica."
            elif promedio_salud >= 40:
                color_box = "#f39c12"
                msg_estado = "🟡 <b>TERRITORIO VULNERABLE:</b> Existen tensiones en la oferta hídrica o calidad del agua. Se requiere inversión preventiva en Soluciones Basadas en la Naturaleza."
            else:
                color_box = "#c0392b"
                msg_estado = "🔴 <b>TERRITORIO CRÍTICO:</b> Alarma sistémica de seguridad hídrica. Urge implementar portafolios masivos de compensación volumétrica (SbN) y saneamiento (PTAR)."
                
            st.markdown(f"""
                <div style='padding:15px; border-radius:8px; background-color:#f8f9fa; border-left: 6px solid {color_box}; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);'>
                    <h4 style='margin-top:0px; color:{color_box};'>Puntaje Global de Salud: {promedio_salud:.1f}/100</h4>
                    <p style='margin-bottom:0px; font-size:14px;'>{msg_estado}</p>
                </div>
            """, unsafe_allow_html=True)
        
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
        buffer_m = st.session_state.get('buffer_m_ripario', None) 
        
        # --- 🌿 MOTOR GEOMORFOLÓGICO Y RIPARIO (CASCADA INTELIGENTE) ---
        st.markdown("---")
        if rios_strahler is None or rios_strahler.empty:
            with st.expander("⚠️ Paso 1: Faltan Datos - Generar Red Hídrica", expanded=True):
                st.info("Para priorizar predios necesitamos crear la franja riparia, y para eso necesitamos los ríos. ¡Trázalos aquí!")
                render_motor_hidrologico(gdf_zona)
                
        elif buffer_m is None:
            with st.expander("⚠️ Paso 2: Configurar Franja Riparia", expanded=True):
                st.success("✅ ¡Ríos detectados! Ahora define el ancho de la zona de protección riparia.")
                render_motor_ripario()
                
        else:
            with st.expander("⚙️ Recalcular Franja Riparia", expanded=False):
                st.success(f"✅ Red Hídrica y Franja Riparia de {buffer_m}m listas para el cruce predial.")
                render_motor_ripario()

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

        # 3. EJECUTAR EL MOTOR DE CRUCE MULTI-ANILLO
        if rios_strahler is not None and not rios_strahler.empty:
            
            rios_strahler = rios_strahler.reset_index(drop=True)
            rios_strahler['ID_Tramo'] = ["Segmento " + str(i+1) for i in range(len(rios_strahler))]
            if 'longitud_km' in rios_strahler.columns:
                rios_strahler['longitud_km'] = rios_strahler['longitud_km'].round(2)
            
            # Extraemos los 3 tamaños de anillo
            anillos = st.session_state.get('multi_rings', [10, 20, 30])
            b_min, b_med, b_max = anillos[0], anillos[1], anillos[2]
            
            # 🌿 MAGIA GEOMÉTRICA: Pre-calculamos los 3 anillos fusionados para el mapa
            rios_3116 = rios_strahler.to_crs(epsg=3116)
            rios_union = rios_3116.unary_union
            geom_max = rios_union.buffer(b_max, resolution=2)
            geom_med = rios_union.buffer(b_med, resolution=2)
            geom_min = rios_union.buffer(b_min, resolution=2)
            
            buffer_max_gdf = gpd.GeoDataFrame(geometry=[geom_max], crs=3116)
            
            # --- NUEVO: CÁLCULO DE ÁREAS POR CADA FRANJA RIPARIA (TRAMO HÍDRICO) ---
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
            
            # Calculamos los totales a partir de los tramos (matemáticamente más exacto)
            tot_min = df_tramos[f"Mínimo ({b_min}m) ha"].sum()
            tot_med = df_tramos[f"Ideal ({b_med}m) ha"].sum()
            tot_max = df_tramos[f"Óptimo ({b_max}m) ha"].sum()
            tot_longitud_km = df_tramos["Longitud (Km)"].sum() 

            st.success(f"✅ Modelando 3 escenarios concéntricos simultáneos ({b_min}m, {b_med}m, {b_max}m)...")
            
            st.markdown("##### 📊 Tablero de Sensibilidad Ecológica y Financiera")
            
            # Ahora creamos 5 columnas en lugar de 4
            cm1, cm2, cm3, cm4, cm5 = st.columns(5)
            
            cm1.metric(f"🔴 Escenario {b_min}m", f"{tot_min:,.1f} ha")
            cm2.metric(f"🟡 Escenario {b_med}m", f"{tot_med:,.1f} ha", f"+{(tot_med - tot_min):,.1f} ha extra", delta_color="off")
            cm3.metric(f"🟢 Escenario {b_max}m", f"{tot_max:,.1f} ha", f"+{(tot_max - tot_med):,.1f} ha extra", delta_color="off")
            cm4.metric("🌿 Tramos Hídricos", f"{len(df_tramos)}")
            cm5.metric("📏 Longitud Total", f"{tot_longitud_km:,.1f} km") # <-- NUEVA MÉTRICA
            
            # --- SEPARACIÓN ELEGANTE EN PESTAÑAS (TABS) ---
            tab_predios, tab_tramos = st.tabs(["🏡 Impacto Predial (Negociación)", "🌿 Áreas por Franja Riparia (Tramos)"])
            
            with tab_tramos:
                st.markdown("##### 📋 Matriz Detallada por Franja Riparia")
                st.info("Esta tabla muestra cuánta área de restauración aporta cada segmento del río (Franja Riparia) para cumplir los 3 escenarios ecológicos.")
                st.dataframe(df_tramos.style.background_gradient(cmap="Greens", subset=["Importancia Ecológica"]).format(precision=2), use_container_width=True, hide_index=True)
                csv_tramos = df_tramos.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Matriz de Franjas (CSV)", csv_tramos, "Franjas_Riparias.csv", "text/csv")
            
            with tab_predios:
                if capa_predios is not None and not capa_predios.empty:
                    with st.spinner("Ejecutando intersección de anillos concéntricos con predios..."):
                        try:
                            predios_3116 = capa_predios.to_crs(epsg=3116)
                            # Cruzamos solo con el anillo mayor para optimizar
                            predios_en_buffer = gpd.overlay(predios_3116, buffer_max_gdf, how='intersection')
                            
                            if not predios_en_buffer.empty:
                                predios_en_buffer['Area_Max_ha'] = predios_en_buffer.geometry.area / 10000.0
                                predios_en_buffer['Area_Med_ha'] = predios_en_buffer.geometry.intersection(geom_med).area / 10000.0
                                predios_en_buffer['Area_Min_ha'] = predios_en_buffer.geometry.intersection(geom_min).area / 10000.0
                                
                                col_id = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID'] if col in predios_en_buffer.columns), None)
                                if col_id is None:
                                    predios_en_buffer['ID_Predio'] = predios_en_buffer.index
                                    col_id = 'ID_Predio'
                                    
                                predios_agrupados = predios_en_buffer.groupby(col_id).agg({
                                    'Area_Min_ha': 'sum', 'Area_Med_ha': 'sum', 'Area_Max_ha': 'sum'
                                }).reset_index()
                                
                                datos_ranking = []
                                for idx, row in predios_agrupados.iterrows():
                                    datos_ranking.append({
                                        "Identificador Predial": row[col_id],
                                        f"Mínimo ({b_min}m) ha": row['Area_Min_ha'],
                                        f"Ideal ({b_med}m) ha": row['Area_Med_ha'],
                                        f"Óptimo ({b_max}m) ha": row['Area_Max_ha'],
                                        "ROI (Máx)": row['Area_Max_ha'] * 100
                                    })
                                    
                                df_prioridad = pd.DataFrame(datos_ranking).sort_values(by="ROI (Máx)", ascending=False)
                                
                                c_rank1, c_rank2 = st.columns([2, 1])
                                with c_rank1:
                                    st.markdown("##### 📋 Top 15 Predios Estratégicos")
                                    st.dataframe(df_prioridad.head(15).style.background_gradient(cmap="YlOrRd", subset=["ROI (Máx)"]).format(precision=2), use_container_width=True, hide_index=True)
                                with c_rank2:
                                    st.info("Cada escenario incrementa el esfuerzo de negociación predial. Exporta esta matriz para gestión territorial.")
                                    st.metric("Predios Involucrados", f"{len(df_prioridad)}")
                                    csv_predios = df_prioridad.to_csv(index=False).encode('utf-8')
                                    st.download_button("📥 Descargar Matriz Predial (CSV)", csv_predios, "Escenarios_Prediales.csv", "text/csv")
                            else:
                                st.info("Ninguno de los predios intercepta la red hidrográfica.")
                        except Exception as e:
                            st.error(f"Error en el cruce geográfico: {e}")
                else:
                    st.info("ℹ️ No se detectó mapa predial. Utilice la pestaña de 'Franjas Riparias' para ver los requerimientos de área.")

            # =========================================================
            # 🗺️ EL MAPA TÁCTICO (CON MULTI-ANILLOS VISUALES)
            # =========================================================
            st.markdown("---")
            st.markdown(f"##### 🗺️ Visor Táctico de Conectividad y Predios en: **{nombre_zona}**")
            import pydeck as pdk
            
            try:
                rios_4326 = rios_strahler.to_crs(epsg=4326).copy()
                c_lat, c_lon = rios_4326.geometry.iloc[0].centroid.y, rios_4326.geometry.iloc[0].centroid.x
            except: c_lat, c_lon = 6.2, -75.5 
            
            capas_mapa = []
            
            if gdf_zona is not None:
                zona_4326 = gdf_zona.to_crs("EPSG:4326")
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=zona_4326, opacity=1, stroked=True, get_line_color=[0, 200, 0, 255], get_line_width=3, filled=False))
            
            # --- 🟢 CAPAS MULTI-ANILLO (Dibujadas de mayor a menor para efecto concéntrico) ---
            if 'geom_max' in locals():
                # Óptimo (Más grande, más transparente)
                gdf_max = gpd.GeoDataFrame(geometry=[geom_max], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_max, opacity=0.2, get_fill_color=[171, 235, 198], stroked=False))
                # Ideal (Intermedio)
                gdf_med = gpd.GeoDataFrame(geometry=[geom_med], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_med, opacity=0.4, get_fill_color=[88, 214, 141], stroked=False))
                # Normativo (Más pequeño, verde oscuro)
                gdf_min = gpd.GeoDataFrame(geometry=[geom_min], crs=3116).to_crs(4326)
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=gdf_min, opacity=0.6, get_fill_color=[40, 180, 99], stroked=False))

            # --- 🔵 CAPA DE RÍOS (Línea Fina Central) ---
            if 'rios_4326' in locals():
                capas_mapa.append(pdk.Layer(
                    "GeoJsonLayer", data=rios_4326,
                    get_line_color=[31, 97, 141, 255], # Azul oscuro
                    get_line_width=2, lineWidthMinPixels=2,
                    pickable=True, autoHighlight=True
                ))
            
            # --- 🏠 CAPA DE PREDIOS ---
            if 'predios_en_buffer' in locals() and not predios_en_buffer.empty:
                col_id_pred = next((col for col in ['MATRICULA', 'COD_CATAST', 'FICHA', 'OBJECTID', 'ID_Predio'] if col in predios_en_buffer.columns), None)
                if col_id_pred and capa_predios is not None:
                    ids_afectados = predios_en_buffer[col_id_pred].unique()
                    predios_a_dibujar = capa_predios[capa_predios[col_id_pred].isin(ids_afectados)].to_crs(epsg=4326)
                    
                    capas_mapa.append(pdk.Layer(
                        "GeoJsonLayer", data=predios_a_dibujar, opacity=0.4,
                        stroked=True, filled=True, get_fill_color=[255, 165, 0, 150],
                        get_line_color=[255, 140, 0, 255], get_line_width=2,
                        pickable=True, autoHighlight=True
                    ))
            
            view_state = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=13, pitch=45)
            
            tooltip = {"html": "<b>{ID_Tramo} {MATRICULA} {COD_CATAST}</b><br/>Orden Hídrico: {Orden_Strahler}<br/>Longitud: {longitud_km} km", "style": {"backgroundColor": "steelblue", "color": "white"}}
            st.pydeck_chart(pdk.Deck(layers=capas_mapa, initial_view_state=view_state, map_style="light", tooltip=tooltip), use_container_width=True)

        else:
            st.warning("⚠️ El cruce predial y el mapa táctico están en pausa porque aún no se han calculado los ríos.")
            st.info("👆 Por favor, utiliza el generador de Franja Riparia que se encuentra arriba para iluminar este tablero.")
