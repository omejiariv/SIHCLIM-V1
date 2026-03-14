# pages/Sistemas_Hídricos_Territoriales.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import geopandas as gpd

st.set_page_config(page_title="Metabolismo Complejo", page_icon="🔗", layout="wide")

st.title("🔗 Metabolismo Territorial Complejo: Nodos y Trasvases")
st.markdown("""
Modelo de topología de redes para el **Sistema de Abastecimiento del Valle de Aburrá**. 
Evalúa cómo los embalses integran las cuencas propias con los trasvases artificiales para sostener la demanda urbana, alterando el flujo natural de los ecosistemas aportantes.
""")
st.divider()

# Creamos un "espacio reservado" en la parte superior de la página
contenedor_sankey = st.empty()

# =========================================================================
# 0. CARGA DE CARTOGRAFÍA (Desde Supabase en la Nube)
# =========================================================================
# 1. Buscamos la URL base de tu Supabase en los secretos
url_supabase = None
if "SUPABASE_URL" in st.secrets:
    url_supabase = st.secrets["SUPABASE_URL"]
elif "supabase" in st.secrets:
    url_supabase = st.secrets["supabase"].get("url") or st.secrets["supabase"].get("SUPABASE_URL")
elif "iri" in st.secrets and "SUPABASE_URL" in st.secrets["iri"]:
    url_supabase = st.secrets["iri"]["SUPABASE_URL"]
elif "connections" in st.secrets and "supabase" in st.secrets["connections"]:
    url_supabase = st.secrets["connections"]["supabase"]["SUPABASE_URL"]

gdf_embalses = None 

if url_supabase:
    # 2. Construimos el enlace web directo y público al GeoJSON
    nombre_bucket = "sihcli_maestros"
    # IMPORTANTE: Verifica que el nombre del archivo sea exactamente este en Supabase
    nombre_archivo = "embalses_CV_9377.geojson"
    
    ruta_embalses_nube = f"{url_supabase}/storage/v1/object/public/{nombre_bucket}/Puntos_de_interes/{nombre_archivo}"
    
    try:
        # Leemos el archivo directo desde la URL de Supabase
        gdf_embalses = gpd.read_file(ruta_embalses_nube)
        st.sidebar.success(f"✅ Embalses conectados desde la Nube ({len(gdf_embalses)} registros)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ No se pudo cargar la capa desde la nube. Revisa si el nombre del archivo es correcto. Detalle: {e}")
else:
    st.sidebar.error("❌ No se encontró la URL de Supabase en los secretos.")

# =========================================================================
# 1. BASE DE DATOS ESTRUCTURAL: NEXO AGUA-ENERGÍA Y TAMAÑO
# =========================================================================
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# A. Datos paramétricos (Estructura Ampliada)
sistemas_embalses = {
    "La Fe": {
        "capacidad_util_Mm3": 11.5, 
        "afluentes_naturales": {"Quebrada Espíritu Santo": 1.2},
        "trasvases": {"Pantanillo": 1.5, "Río Buey": 3.0, "Piedras": 0.8},
        "demanda_acueducto_m3s": 5.0, 
        "generacion_energia_m3s": 0.0,
        "evaporacion_m3s": 0.1,
        "caudal_ecologico_m3s": 0.3,
        "factor_energia_kwh_m3": 0.0, 
        "costo_bombeo_kwh_m3": 0.85,
        "ha_conservadas_base": 3600.0
    },
    "Río Grande II": {
        "capacidad_util_Mm3": 220.0, 
        "afluentes_naturales": {"Río Grande": 10.0, "Río Chico": 3.0, "Quebrada Las Ánimas": 2.0},
        "trasvases": {}, 
        "demanda_acueducto_m3s": 6.5, 
        "generacion_energia_m3s": 12.0, 
        "evaporacion_m3s": 0.5,
        "caudal_ecologico_m3s": 1.0,
        "factor_energia_kwh_m3": 0.65,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 4500.0 # Ajustado según indicación
    },
    "El Peñol (Guatapé)": {
        "capacidad_util_Mm3": 1070.0, 
        "afluentes_naturales": {"Río Nare": 35.0},
        "trasvases": {}, 
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 30.0, 
        "evaporacion_m3s": 1.5,
        "caudal_ecologico_m3s": 2.0,
        "factor_energia_kwh_m3": 1.2,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0 # Corrección: Sin intervenciones registradas
    },
    "Punchiná (San Carlos)": {
        "capacidad_util_Mm3": 68.0, 
        "afluentes_naturales": {"Río Guatapé": 40.0, "Río San Carlos": 15.0},
        "trasvases": {}, 
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 45.0, 
        "evaporacion_m3s": 0.3,
        "caudal_ecologico_m3s": 4.0,
        "factor_energia_kwh_m3": 2.5,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0 # Corrección: Sin intervenciones registradas
    },
    "Hidroituango": {
        "capacidad_util_Mm3": 2720.0, 
        "afluentes_naturales": {"Río Cauca": 1100.0},
        "trasvases": {}, 
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 900.0, 
        "evaporacion_m3s": 5.0,
        "caudal_ecologico_m3s": 200.0, 
        "factor_energia_kwh_m3": 0.9,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0 # Corrección: Sin intervenciones registradas
    }
}

# B. INYECCIÓN DE DATOS ESPACIALES AL MODELO MATEMÁTICO
if gdf_embalses is not None and not gdf_embalses.empty:
    col_nombre = next((c for c in gdf_embalses.columns if 'nom' in c.lower() or 'proyect' in c.lower() or 'embalse' in c.lower()), None)
    col_vol = next((c for c in gdf_embalses.columns if 'vol' in c.lower() or 'cap' in c.lower()), None)
    
    if col_nombre and col_vol:
        def inyectar_capacidad_real(nombre_nodo, texto_busqueda):
            try:
                match = gdf_embalses[gdf_embalses[col_nombre].astype(str).str.contains(texto_busqueda, case=False, na=False)]
                if not match.empty:
                    vol_real = match.iloc[0][col_vol]
                    if pd.notnull(vol_real) and float(vol_real) > 0:
                        if float(vol_real) > 10000: vol_real = float(vol_real) / 1000000
                        sistemas_embalses[nombre_nodo]["capacidad_util_Mm3"] = round(float(vol_real), 2)
            except: pass

        inyectar_capacidad_real("La Fe", "fe")
        inyectar_capacidad_real("Río Grande II", "grande")
        inyectar_capacidad_real("El Peñol (Guatapé)", "peñol|guatap")
        inyectar_capacidad_real("Punchiná (San Carlos)", "punchin|carlos")
        inyectar_capacidad_real("Hidroituango", "ituango")

        # -----------------------------------------------------------------
        # C. INYECCIÓN DE PREDIOS CONSERVADOS (SbN) DESDE SUPABASE
        # -----------------------------------------------------------------
        try:
            # 1. Aseguramos el patrimonio histórico base de La Fe
            sistemas_embalses["La Fe"]["ha_conservadas_base"] = 3600.0
            
            # Construimos la ruta directa a tu archivo de Predios
            ruta_predios = f"{url_limpia}/storage/v1/object/public/{nombre_bucket}/Puntos_de_interes/PrediosEjecutados.geojson"
            import requests
            import tempfile
            
            res_predios = requests.get(ruta_predios)
            if res_predios.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp_p:
                    tmp_p.write(res_predios.content)
                    tmp_path_p = tmp_p.name
                
                import geopandas as gpd
                gdf_predios = gpd.read_file(tmp_path_p)
                
                # Sumar por embalse leyendo tus columnas reales 'EMBALSE' y 'AREA_HA'
                if 'EMBALSE' in gdf_predios.columns and 'AREA_HA' in gdf_predios.columns:
                    resumen_predios = gdf_predios.groupby('EMBALSE')['AREA_HA'].sum().to_dict()
                    
                    # Mapear el resultado al diccionario del modelo
                    for embalse_mapa, ha_total in resumen_predios.items():
                        nombre_mapa_limpio = str(embalse_mapa).lower().strip()
                        for nombre_nodo in sistemas_embalses.keys():
                            if nombre_mapa_limpio in nombre_nodo.lower() or (nombre_mapa_limpio == "riogrande ii" and "grande" in nombre_nodo.lower()):
                                # SUMAMOS (+), no sobrescribimos (=), para proteger las 3600 ha de La Fe
                                sistemas_embalses[nombre_nodo]["ha_conservadas_base"] += float(ha_total)
                                
                    st.sidebar.success("🌲 Áreas de conservación inyectadas (Se respetó el histórico de La Fe).")
        except Exception as e:
            st.sidebar.warning(f"No se pudo cargar la capa de predios: {e}")

# ==============================================================================
# 🧠 EL ALEPH HÍDRICO (RECEPTOR INTELIGENTE DE DEMANDA Y ROUTER)
# ==============================================================================
conectado_aleph = False
pob_amva_aleph = None
pob_local_aleph = None

if 'aleph_lugar' in st.session_state and 'aleph_pob_total' in st.session_state:
    aleph_lugar = st.session_state['aleph_lugar']
    aleph_pob = float(st.session_state['aleph_pob_total'])
    aleph_anio = st.session_state.get('aleph_anio', 2035)
    
    if aleph_pob > 0:
        conectado_aleph = True
        st.markdown("---")
        with st.expander("🧠 Conexión Activa con el Modelo Demográfico (El Aleph)", expanded=True):
            st.success(f"Recibiendo proyección para **{aleph_lugar}** (Año **{aleph_anio}**): **{aleph_pob:,.0f} habitantes**.")
            
            # Limpiamos el texto al máximo para que "R. Chico" coincida con "chico"
            import unicodedata
            lugar_limpio = unicodedata.normalize('NFKD', str(aleph_lugar).lower()).encode('ascii', 'ignore').decode('utf-8')
            
            # Listas de palabras clave precisas
            claves_rg2 = ["belmira", "donmatias", "san pedro", "entrerrios", "santa rosa", "chico", "grande", "animas"]
            claves_lafe = ["retiro", "ceja", "rionegro", "negro", "espiritu santo", "pantanillo", "buey", "piedras", "arma"]
            claves_amva = ["medellin", "bello", "itagui", "envigado", "sabaneta", "copacabana", "estrella", "girardota", "caldas", "barbosa", "aburra", "amva", "total"]
            
            if any(x in lugar_limpio for x in claves_amva):
                st.info("🏙️ **Rol detectado: Consumidor Metropolitano.** Población asignada a la Demanda Urbana del Sistema.")
                pob_amva_aleph = aleph_pob
                st.session_state['nodo_sugerido'] = "La Fe" 
            elif any(x in lugar_limpio for x in claves_rg2):
                st.info("🌲 **Rol detectado: Productor (Sistema Río Grande II).** Población asignada a la Demanda Local aguas arriba.")
                pob_local_aleph = aleph_pob
                st.session_state['nodo_sugerido'] = "Río Grande II"
            elif any(x in lugar_limpio for x in claves_lafe):
                st.info("🌲 **Rol detectado: Productor (Sistema La Fe).** Población asignada a la Demanda Local aguas arriba.")
                pob_local_aleph = aleph_pob
                st.session_state['nodo_sugerido'] = "La Fe"
            else:
                st.warning(f"⚠️ El territorio '{aleph_lugar}' no se autodetectó. Usando datos manuales.")

# =========================================================================
# 2. PANEL DE OPERACIONES Y CONTROLES (CON AUTO-SELECCIÓN)
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")

# Determinamos si el Aleph sugirió un embalse
nodos_lista = list(sistemas_embalses.keys())
idx_defecto = 0
if 'nodo_sugerido' in st.session_state and st.session_state['nodo_sugerido'] in nodos_lista:
    idx_defecto = nodos_lista.index(st.session_state['nodo_sugerido'])

nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", nodos_lista, index=idx_defecto)
datos_nodo = sistemas_embalses[nodo_seleccionado]

# --- ☁️ MOTOR CLIMÁTICO LOCAL (ENSO) ---
st.sidebar.markdown("### 🌍 Motor Climático (ENSO)")
escenario_enso = st.sidebar.select_slider(
    "Fase actual del Pacífico:",
    options=["Niño Severo", "Niño Moderado", "Neutro", "Niña Moderada", "Niña Fuerte"],
    value="Neutro"
)

# Acoplamiento Termodinámico (Precipitación vs Evaporación)
if escenario_enso == "Niño Severo": factor_p, factor_et = 0.65, 1.40
elif escenario_enso == "Niño Moderado": factor_p, factor_et = 0.85, 1.20
elif escenario_enso == "Neutro": factor_p, factor_et = 1.0, 1.0
elif escenario_enso == "Niña Moderada": factor_p, factor_et = 1.15, 0.85
else: factor_p, factor_et = 1.35, 0.70  # Niña Fuerte

st.markdown(f"### 💧 Balance de Masa en Tiempo Real: {nodo_seleccionado}")
if factor_p < 1.0:
    st.error(f"⚠️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) **{(factor_et-1)*100:+.0f}%** debido a anomalía térmica.")
elif factor_p > 1.0:
    st.info(f"🌧️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) reducida al **{(factor_et)*100:.0f}%**.")
else:
    st.info(f"**Capacidad Útil Máxima:** {datos_nodo['capacidad_util_Mm3']:,.1f} Mm³ (Condiciones Climáticas Históricas)")

col_in, col_out = st.columns(2)

# --- ENTRADAS DINÁMICAS (AFECTADAS POR EL CLIMA) ---
afluentes_inputs = {}
trasvases_inputs = {}

with col_in:
    st.markdown("#### 📥 ENTRADAS (Inflows)")
    st.caption("Aportes Naturales de la Cuenca (Afectados por ENSO):")
    for nombre, caudal in datos_nodo["afluentes_naturales"].items():
        caudal_afectado = float(caudal * factor_p)
        max_val = float(caudal * 3) if caudal > 0 else 10.0
        afluentes_inputs[nombre] = st.slider(f"{nombre} [m³/s]:", 0.0, max_val, caudal_afectado, 0.1, key=f"in_{nombre}")
    
    st.caption("Bombas y Túneles (Trasvases Externos):")
    if datos_nodo["trasvases"]:
        for nombre, caudal in datos_nodo["trasvases"].items():
            trasvases_inputs[nombre] = st.slider(f"Bombeo {nombre} [m³/s]:", 0.0, float(caudal * 2), float(caudal), 0.1, key=f"tr_{nombre}")
    else:
        st.write("*(Sistema impulsado 100% por gravedad)*")

# --- SALIDAS DINÁMICAS ---
with col_out:
    st.markdown("#### 📤 SALIDAS (Outflows)")
    
    # Acoplamiento Evaporativo (El Niño evapora más rápido el embalse)
    evaporacion_dinamica = datos_nodo["evaporacion_m3s"] * factor_et
    st.metric("Evaporación Directa (Afectada por T°)", f"{evaporacion_dinamica:.2f} m³/s", f"{(evaporacion_dinamica - datos_nodo['evaporacion_m3s']):+.2f} m³/s", delta_color="inverse")
    
    # Extracción Consuntiva conectada a la memoria
    demanda_memoria = st.session_state.get('demanda_total_m3s', datos_nodo["demanda_acueducto_m3s"])
    max_acueducto = float(max(demanda_memoria, datos_nodo["demanda_acueducto_m3s"]) * 2)
    
    val_acueducto = 0.0
    if max_acueducto > 0:
        val_acueducto = st.slider("Extracción Consuntiva (Multi-Sector) [m³/s]:", 0.0, max_acueducto, float(demanda_memoria), 0.1)
    
    val_turbinado = 0.0
    if datos_nodo["generacion_energia_m3s"] > 0:
        max_turb = float(datos_nodo["generacion_energia_m3s"] * 1.5)
        val_turbinado = st.slider("Caudal Turbinado (Energía) [m³/s]:", 0.0, max_turb, float(datos_nodo["generacion_energia_m3s"]), 1.0)
        
    val_ecologico = st.number_input("Caudal Ecológico / Vertimiento [m³/s]:", min_value=0.0, value=float(datos_nodo["caudal_ecologico_m3s"]), step=1.0)

# Corrección vital para el cálculo de balance (Sección 3)
# Asegúrate de que la variable sum_salidas use la nueva evaporación
sum_salidas = val_acueducto + val_turbinado + val_ecologico + evaporacion_dinamica

# =========================================================================
# 3. CÁLCULO DE BALANCE Y ENERGÍA
# =========================================================================
sum_entradas = sum(afluentes_inputs.values()) + sum(trasvases_inputs.values())
sum_salidas = val_acueducto + val_turbinado + val_ecologico + datos_nodo["evaporacion_m3s"]
balance = sum_entradas - sum_salidas

# --- HUELLA ENERGÉTICA ---
m3_hora_turbinados = val_turbinado * 3600
m3_hora_bombeados = sum(trasvases_inputs.values()) * 3600
potencia_generada_kw = m3_hora_turbinados * datos_nodo["factor_energia_kwh_m3"]
potencia_consumida_kw = m3_hora_bombeados * datos_nodo["costo_bombeo_kwh_m3"]
balance_energetico_MW = (potencia_generada_kw - potencia_consumida_kw) / 1000

ingreso_hora_cop = potencia_generada_kw * 350
costo_hora_cop = potencia_consumida_kw * 350

c1, c2, c3, c4 = st.columns(4)
c1.metric("Balance Hídrico (ΔS/Δt)", f"{balance:+.1f} m³/s", "Llenándose 📈" if balance > 0 else "Vaciándose 📉" if balance < 0 else "Estable ⚖️")
c2.metric("Energía Generada", f"{potencia_generada_kw/1000:,.1f} MW", f"${ingreso_hora_cop/1e6:,.1f} M/hora", delta_color="normal")
c3.metric("Energía Consumida", f"{potencia_consumida_kw/1000:,.1f} MW", f"-${costo_hora_cop/1e6:,.1f} M/hora", delta_color="inverse")
c4.metric("Balance Neto", f"{balance_energetico_MW:,.1f} MW", "Superávit ⚡" if balance_energetico_MW > 0 else "Déficit 🔴" if balance_energetico_MW < 0 else "Neutro")

# =========================================================================
# 4. TABLERO WRI: NEUTRALIDAD, RESILIENCIA Y CALIDAD
# =========================================================================
st.markdown("---")
st.subheader(f"🌐 Inteligencia Territorial WRI: {nodo_seleccionado}")

anio_analisis = st.slider("Seleccione el Año de Evaluación (Actual o Futuro):", min_value=2024, max_value=2050, value=2025, step=1)

delta_anios = anio_analisis - 2025
factor_demanda = (1 + 0.015) ** delta_anios
factor_clima = (1 - 0.005) ** delta_anios

q_oferta_m3s_base = sum_entradas
demanda_m3s_base = val_acueducto + (val_turbinado * 0.1) 
capacidad_embalse_m3 = datos_nodo["capacidad_util_Mm3"] * 1000000

oferta_anual_m3 = (q_oferta_m3s_base * factor_clima) * 31536000
consumo_anual_m3 = (demanda_m3s_base * factor_demanda) * 31536000

# --- 2. INTEGRACIÓN CARTOGRÁFICA (PREDIOS EJECUTADOS SbN) ---
st.markdown("---")
st.markdown(f"#### 🌲 Beneficios Volumétricos (SbN) en el Sistema: **{nodo_seleccionado}**")

# Recuperamos las hectáreas reales que la Cirugía 1 inyectó desde Supabase
ha_reales_sig = float(datos_nodo.get("ha_conservadas_base", 0.0))

st.markdown("##### ⚙️ Escenario Base vs. Proyectado")
activar_sig = st.toggle("✅ Incluir Área Restaurada/Conservada del SIG en el cálculo WRI", value=True, 
                        help="Apaga este interruptor para simular el escenario contrafactual: ¿Cómo estarían los índices si no se hubieran realizado estas intervenciones?")

# Si el usuario apaga el interruptor, la base para el cálculo se vuelve 0
ha_base_calculo = ha_reales_sig if activar_sig else 0.0

c_inv1, c_inv2, c_inv3 = st.columns(3)
with c_inv1:
    st.metric("✅ Área Restaurada/Conservada (SIG)", f"{ha_reales_sig:,.1f} ha", "Línea base actual")
    ha_simuladas = st.number_input("➕ Adicionar Hectáreas (Simulación):", min_value=0.0, value=0.0, step=50.0)
    
    ha_total = ha_base_calculo + ha_simuladas
    beneficio_restauracion_m3 = ha_total * 2500 
    
with c_inv2:
    sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM):", min_value=0, value=120, step=5)
    beneficio_calidad_m3 = (sist_saneamiento * 1200) if activar_sig else 0
with c_inv3:
    volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
    st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3/1e6:,.2f} Mm³/año", "Total compensado")

# --- 3. MOTORES DE CÁLCULO WRI ---
ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
ind_resiliencia = min(100.0, ((capacidad_embalse_m3 + oferta_anual_m3) / ((consumo_anual_m3+1) * 2)) * 100)
ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0
ind_calidad = min(100.0, max(0.0, 50.0 + ((oferta_anual_m3 / (consumo_anual_m3 + 1)) * 0.5) + (sist_saneamiento * 0.05)))

def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
    if not invertido:
        return ("🔴 CRÍTICO", "#c0392b") if valor < umbral_rojo else ("🟡 VULNERABLE", "#f39c12") if valor < umbral_verde else ("🟢 ÓPTIMO", "#27ae60")
    else:
        return ("🟢 HOLGADO", "#27ae60") if valor < umbral_verde else ("🟡 MODERADO", "#f39c12") if valor < umbral_rojo else ("🔴 CRÍTICO", "#c0392b")

# Generador de Leyendas Interpretativas (CON NOMBRES)
def generar_leyenda(u_r, u_v, inv):
    if not inv:
        return f"🔴 <b>Crítico</b> &lt; {u_r}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 <b>Vulnerable</b> {u_r}-{u_v}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟢 <b>Óptimo</b> &gt; {u_v}%"
    else:
        return f"🟢 <b>Holgado</b> &lt; {u_v}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 <b>Moderado</b> {u_v}-{u_r}% &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 <b>Severo</b> &gt; {u_r}%"

st.markdown("---")
def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde, invertido=False):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = valor, number = {'suffix': "%", 'font': {'size': 26}}, title = {'text': titulo, 'font': {'size': 14}},
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1}, 'bar': {'color': color_bar}, 'bgcolor': "white",
                 'steps': [{'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                           {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc"},
                           {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}],
                 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}}
    ))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
    return fig

col_g1, col_g2, col_g3, col_g4 = st.columns(4)
for col, ind, tit, col_h, u_r, u_v, inv in zip(
    [col_g1, col_g2, col_g3, col_g4], [ind_neutralidad, ind_resiliencia, ind_estres, ind_calidad],
    ["Neutralidad", "Resiliencia", "Estrés Hídrico", "Calidad"], ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"],
    [20, 30, 40, 40], [50, 70, 20, 70], [False, False, True, False]
):
    with col:
        est, color_txt = evaluar_indice(ind, u_r, u_v, inv)
        st.plotly_chart(crear_velocimetro(ind, tit, col_h, u_r, u_v, inv), use_container_width=True)
        st.markdown(f"<h4 style='text-align: center; color: {color_txt}; margin-top:-20px;'>{est}</h4>", unsafe_allow_html=True)
        # Inyectar leyenda interpretativa
        leyenda = generar_leyenda(u_r, u_v, inv)
        st.markdown(f"<div style='text-align: center; font-size: 13px; color: #7F8C8D; margin-top: -5px;'>{leyenda}</div>", unsafe_allow_html=True)

# Caja desplegable con el glosario
with st.expander("📚 Conceptos, Metodología y Fuentes (VWBA - WRI)", expanded=False):
    st.markdown("""
    ### 📖 Glosario de Indicadores
    
    * **Neutralidad Hídrica (Volumetric Water Benefit VWBA):**
      * **Concepto:** Mide si el volumen de agua restituido a la cuenca mediante Soluciones Basadas en la Naturaleza (SbN) compensa la Huella Hídrica del consumo humano/industrial.
      * **Interpretación:** Un 100% indica que se está reponiendo cada gota extraída. Valores $<40\%$ son críticos e implican deuda ecológica.
      * **Fórmula:** $\\frac{\\sum Beneficios\\ Volumétricos\\ (m^3/a)}{Consumo\\ Total\\ (m^3/a)} \\times 100$
      
    * **Resiliencia Territorial:**
      * **Concepto:** Capacidad del ecosistema (aguas subterráneas + escorrentía + buffer del embalse) para soportar eventos de sequía (El Niño) sin colapsar el suministro.
      * **Interpretación:** Zonas con alta resiliencia ($>70\%$) son buffers climáticos naturales y estructurales. 
      
    * **Estrés Hídrico (Indicador Falkenmark / ODS 6.4.2):**
      * **Concepto:** Porcentaje de la oferta total anual que está siendo extraída por los diversos sectores económicos.
      * **Interpretación:** Valores $>40\%$ denotan estrés severo (competencia intensa por el recurso). Valores $<20\%$ indican un sistema holgado.

    * **Calidad de Agua (WQI):** * Índice modificado basado en la capacidad de dilución natural (Oferta vs Extracción) y mitigación sanitaria (STAM).
      
    ### 🌐 Fuentes y Estándares de Referencia
    * **WRI (World Resources Institute):** [Volumetric Water Benefit Accounting (VWBA) - Metodología Oficial](https://www.wri.org/research/volumetric-water-benefit-accounting-vwba-implementing-guidelines)
    * **CEO Water Mandate:** Iniciativa del Pacto Global de Naciones Unidas para la resiliencia hídrica corporativa.
    * **Naciones Unidas:** Objetivo de Desarrollo Sostenible (ODS) 6.4.2 (Nivel de estrés hídrico).
    """)

# --- OPTIMIZADOR DE INVERSIONES Y METAS (PORTAFOLIO INTEGRAL) ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("💼 Portafolios de Inversión Multi-Objetivo")

# -------------------------------------------------------------------------
# PORTAFOLIO 1: CANTIDAD (NEUTRALIDAD HÍDRICA)
# -------------------------------------------------------------------------
with st.expander("🎯 1. Optimización de Brechas: Oferta y Demanda (Neutralidad)", expanded=False):
    st.markdown("Combina estrategias (SbN, Saneamiento, Eficiencia) y estima el presupuesto necesario para compensar la Huella Hídrica.")
    
    col_m1, col_m2 = st.columns([1, 2.5])
    with col_m1:
        meta_neutralidad = st.slider("🎯 Objetivo de Neutralidad (%)", 10.0, 100.0, 100.0, 5.0)
        st.markdown("**Costos Unitarios (Millones COP):**")
        costo_ha = st.number_input("Restauración (1 ha):", value=8.5, step=0.5)
        costo_stam_n = st.number_input("Saneamiento (1 STAM):", value=15.0, step=1.0)
        costo_lps = st.number_input("Eficiencia (1 L/s ahorrado):", value=120.0, step=10.0)
    
    with col_m2:
        vol_requerido_m3 = (meta_neutralidad / 100.0) * consumo_anual_m3
        brecha_m3 = vol_requerido_m3 - volumen_repuesto_m3
        
        if brecha_m3 <= 0:
            st.success(f"✅ ¡El sistema cumple o supera la meta del {meta_neutralidad}% con las acciones actuales!")
        else:
            st.warning(f"⚠️ **Déficit para la meta:** Faltan compensar **{brecha_m3/1e6:,.2f} Mm³/año**.")
            
            st.markdown("🎚️ **Diseña tu Mix de Intervención (% de aporte):**")
            c_mix1, c_mix2, c_mix3 = st.columns(3)
            pct_a = c_mix1.number_input("% Restauración", 0, 100, 40)
            pct_b = c_mix2.number_input("% Saneamiento", 0, 100, 40, key="pct_b_neut")
            pct_c = c_mix3.number_input("% Eficiencia", 0, 100, 20)
            
            total_pct = pct_a + pct_b + pct_c
            if total_pct != 100:
                st.error(f"La suma de las estrategias debe ser 100%. Actual: {total_pct}%")
            else:
                vol_a = brecha_m3 * (pct_a / 100.0)
                vol_b = brecha_m3 * (pct_b / 100.0)
                vol_c = brecha_m3 * (pct_c / 100.0)
                
                ha_req = vol_a / 2500.0
                stam_req = vol_b / 1200.0
                lps_req = (vol_c * 1000) / 31536000 
                
                inv_a = ha_req * costo_ha
                inv_b = stam_req * costo_stam_n
                inv_c = lps_req * costo_lps
                inv_total = inv_a + inv_b + inv_c
                
                st.markdown("📊 **Requerimientos Físicos y Presupuesto:**")
                c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                
                costo_L_a = (inv_a * 1000000) / (vol_a * 1000) if vol_a > 0 else 0
                costo_L_b = (inv_b * 1000000) / (vol_b * 1000) if vol_b > 0 else 0
                costo_L_c = (inv_c * 1000000) / (vol_c * 1000) if vol_c > 0 else 0
                
                c_op1.metric("🌲 Restaurar (ha)", f"{ha_req:,.1f}", f"${inv_a:,.0f} Millones")
                c_op1.caption(f"**Costo Eficiencia:** ${costo_L_a:,.1f} COP por Litro")
                c_op2.metric("🚽 Saneamiento (STAM)", f"{stam_req:,.0f}", f"${inv_b:,.0f} Millones")
                c_op2.caption(f"**Costo Eficiencia:** ${costo_L_b:,.1f} COP por Litro")
                c_op3.metric("🚰 Eficiencia (L/s)", f"{lps_req:,.1f}", f"${inv_c:,.0f} Millones")
                c_op3.caption(f"**Costo Eficiencia:** ${costo_L_c:,.1f} COP por Litro")
                c_op4.metric("💰 INVERSIÓN (CANTIDAD)", f"${inv_total:,.0f} M", "Millones COP", delta_color="off")

# -------------------------------------------------------------------------
# PORTAFOLIO 2: CALIDAD (SANEAMIENTO Y REMOCIÓN DBO)
# -------------------------------------------------------------------------
with st.expander("🎯 2. Optimización de Cargas Contaminantes (Saneamiento DBO5)", expanded=False):
    st.markdown("Combina infraestructura gris y verde para estimar el presupuesto necesario para limpiar la cuenca.")
    
    # Lee la carga del metabolismo (Sección 6) a través de la memoria global
    carga_total_ton = st.session_state.get('carga_total_ton', 1000.0)
    carga_removida_actual = 0.0 
    
    col_c1, col_c2 = st.columns([1, 2.5])
    with col_c1:
        meta_remocion = st.slider("🎯 Meta de Remoción de DBO (%)", 10.0, 100.0, 85.0, 5.0)
        st.markdown("**Costos Unitarios (Millones COP por Ton/año):**")
        costo_ptar = st.number_input("PTAR Urbana (1 Ton DBO/a):", value=150.0, step=10.0)
        costo_stam_c = st.number_input("STAM Rural (1 Ton DBO/a):", value=45.0, step=5.0)
        costo_sbn = st.number_input("SbN/Filtros (1 Ton DBO/a):", value=12.0, step=2.0)
    
    with col_c2:
        carga_objetivo_remover = (meta_remocion / 100.0) * carga_total_ton
        brecha_ton = carga_objetivo_remover - carga_removida_actual
        
        if brecha_ton <= 0:
            st.success(f"✅ ¡El sistema cumple o supera la meta del {meta_remocion}% de remoción!")
        else:
            st.warning(f"⚠️ **Déficit de Saneamiento:** Faltan remover **{brecha_ton:,.1f} Toneladas/año** de DBO5. *(Base: {carga_total_ton:,.0f} Ton)*")
            
            st.markdown("🎚️ **Diseña tu Mix de Mitigación (% de aporte):**")
            c_mix_c1, c_mix_c2, c_mix_c3 = st.columns(3)
            pct_ptar = c_mix_c1.number_input("% PTAR Urbana", 0, 100, 50)
            pct_stam = c_mix_c2.number_input("% STAM Rural", 0, 100, 30, key="pct_stam_cal")
            pct_sbn = c_mix_c3.number_input("% SbN / Humedales", 0, 100, 20)
            
            total_pct_c = pct_ptar + pct_stam + pct_sbn
            if total_pct_c != 100:
                st.error(f"La suma debe ser 100%. Actual: {total_pct_c}%")
            else:
                ton_ptar = brecha_ton * (pct_ptar / 100.0)
                ton_stam = brecha_ton * (pct_stam / 100.0)
                ton_sbn = brecha_ton * (pct_sbn / 100.0)
                
                inv_ptar = ton_ptar * costo_ptar
                inv_stam = ton_stam * costo_stam_c
                inv_sbn = ton_sbn * costo_sbn
                inv_total_c = inv_ptar + inv_stam + inv_sbn
                
                st.markdown("📊 **Requerimientos Físicos y Presupuesto:**")
                c_op_c1, c_op_c2, c_op_c3, c_op_c4 = st.columns(4)
                
                costo_kg_ptar = (inv_ptar * 1000000) / (ton_ptar * 1000) if ton_ptar > 0 else 0
                costo_kg_stam = (inv_stam * 1000000) / (ton_stam * 1000) if ton_stam > 0 else 0
                costo_kg_sbn = (inv_sbn * 1000000) / (ton_sbn * 1000) if ton_sbn > 0 else 0
                
                c_op_c1.metric("🏙️ PTAR Urbana", f"{ton_ptar:,.0f} Ton", f"${inv_ptar:,.0f} Millones")
                c_op_c1.caption(f"**Eficiencia:** ${costo_kg_ptar:,.0f} COP/Kg DBO")
                c_op_c2.metric("🏡 STAM Rural", f"{ton_stam:,.0f} Ton", f"${inv_stam:,.0f} Millones")
                c_op_c2.caption(f"**Eficiencia:** ${costo_kg_stam:,.0f} COP/Kg DBO")
                c_op_c3.metric("🌿 SbN / Humedales", f"{ton_sbn:,.0f} Ton", f"${inv_sbn:,.0f} Millones")
                c_op_c3.caption(f"**Eficiencia:** ${costo_kg_sbn:,.0f} COP/Kg DBO")
                c_op_c4.metric("💰 INVERSIÓN (CALIDAD)", f"${inv_total_c:,.0f} M", "Millones COP", delta_color="off")

# =========================================================================
# 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (EXPLORADOR DE ESCENARIOS)
# =========================================================================
st.markdown("---")
st.subheader(f"📈 Proyección Dinámica de Seguridad Hídrica {nodo_seleccionado} (2024 - 2050)")
st.caption("Simulación a largo plazo que integra crecimiento poblacional, pérdida base por Cambio Climático y la variabilidad cíclica/extrema del fenómeno ENSO.")

# --- CREACIÓN DE LAS DOS PESTAÑAS INTERNAS ---
tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono de Incertidumbre)"])

anios_proj = list(range(2024, 2051))

# -------------------------------------------------------------------------
# PESTAÑA 1: TU GRÁFICA ACTUAL (ONDA DINÁMICA)
# -------------------------------------------------------------------------
with tab_resumen:
    col_t1, col_t2 = st.columns(2)
    with col_t1: activar_cc = st.toggle("🌡️ Incluir Cambio Climático", value=True, key="t1_cc")
    with col_t2: activar_enso = st.toggle("🌊 Incluir Variabilidad ENSO", value=True, key="t1_enso")

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
        c_m3 = (demanda_m3s_base * f_dem) * 31536000
        
        n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
        r = min(100.0, ((capacidad_embalse_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
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
    fig_line1.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.05, layer="below", annotation_text="Zona Crítica (>40%)", annotation_position="top left")
    fig_line1.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_line1, use_container_width=True)

# -------------------------------------------------------------------------
# PESTAÑA 2: EL EXPLORADOR DE ESCENARIOS (CONO DE INCERTIDUMBRE)
# -------------------------------------------------------------------------
with tab_escenarios:
    st.markdown("Analiza la dispersión de futuros posibles para un solo indicador bajo intensidades climáticas sostenidas.")
    
    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        ind_sel = st.selectbox("🎯 Seleccione el Indicador a Evaluar:", ["Estrés Hídrico", "Resiliencia", "Neutralidad", "Calidad"])
        activar_cc_esc = st.toggle("🌡️ Sumar Efecto Cambio Climático", value=True, key="t2_cc")
    
    with col_e2:
        diccionario_escenarios = {
            "Onda Dinámica (Ciclo de 4.5 años)": "onda",
            "Condición Neutra (Línea Base)": 0.0,
            "🟡 El Niño Moderado Constante (-15%)": -0.15,
            "🔴 El Niño Severo Constante (-35%)": -0.35,
            "🟢 La Niña Moderada Constante (+15%)": 0.15,
            "🔵 La Niña Fuerte Constante (+35%)": 0.35
        }
        
        curvas_sel = st.multiselect(
            "🌊 Curvas Climáticas a Superponer:", 
            list(diccionario_escenarios.keys()), 
            default=["Onda Dinámica (Ciclo de 4.5 años)", "Condición Neutra (Línea Base)", "🔴 El Niño Severo Constante (-35%)"]
        )

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
            c_m3 = (demanda_m3s_base * f_dem) * 31536000
            
            if ind_sel == "Neutralidad": val = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
            elif ind_sel == "Resiliencia": val = min(100.0, ((capacidad_embalse_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
            elif ind_sel == "Estrés Hídrico": val = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
            else: 
                fac_dil = (o_m3 / (c_m3 + 1))
                val = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
                
            datos_esc.append({"Año": a, "Escenario Climático": nombre_esc, "Valor (%)": val})
            
    if datos_esc:
        df_esc = pd.DataFrame(datos_esc)
        
        color_map = {
            "Onda Dinámica (Ciclo de 4.5 años)": "#9b59b6",  
            "Condición Neutra (Línea Base)": "#34495e",       
            "🟡 El Niño Moderado Constante (-15%)": "#f1c40f",
            "🔴 El Niño Severo Constante (-35%)": "#e74c3c",
            "🟢 La Niña Moderada Constante (+15%)": "#2ecc71",
            "🔵 La Niña Fuerte Constante (+35%)": "#3498db"
        }
        
        fig_esc = px.line(df_esc, x="Año", y="Valor (%)", color="Escenario Climático", color_discrete_map=color_map)
        fig_esc.update_traces(line=dict(width=3)) 
        
        if ind_sel == "Estrés Hídrico":
            fig_esc.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.05, layer="below", annotation_text="Estrés Crítico (>40%)")
        elif ind_sel == "Resiliencia":
            fig_esc.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.05, layer="below", annotation_text="Colapso Hídrico (<50%)")
            
        fig_esc.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig_esc, use_container_width=True)
    else:
        st.info("👈 Seleccione al menos una curva climática para visualizar.")

# =========================================================================
# 6. METABOLISMO TERRITORIAL: DEMANDA, VERTIMIENTOS Y RESIDUOS SÓLIDOS
# =========================================================================
st.markdown("---")
st.header(f"💧 Metabolismo Hídrico y Material: {nodo_seleccionado}")
st.info("Cálculo integrado de extracción hídrica, cargas orgánicas vertidas (DBO5) y generación de residuos sólidos (Lixiviados/Emisiones).")

# --- CONEXIÓN DINÁMICA CON EL NODO SELECCIONADO (CON LÓGICA DE TRASVASE) ---
# def_pob_res = Habitantes en la cuenca (Generan DBO y RS locales)
# def_pob_ext = Población de la ciudad abastecida (Solo extraen agua)
if nodo_seleccionado == "La Fe": def_pob_res, def_pob_ext, def_bov, def_por, def_ave = 15000, 450000, 5000, 2000, 150000
elif "Grande" in nodo_seleccionado: def_pob_res, def_pob_ext, def_bov, def_por, def_ave = 45000, 1200000, 85000, 45000, 800000
elif "Peñol" in nodo_seleccionado: def_pob_res, def_pob_ext, def_bov, def_por, def_ave = 25000, 0, 40000, 15000, 120000
elif "Ituango" in nodo_seleccionado: def_pob_res, def_pob_ext, def_bov, def_por, def_ave = 35000, 0, 250000, 60000, 300000
else: def_pob_res, def_pob_ext, def_bov, def_por, def_ave = 20000, 0, 25000, 10000, 50000

st.markdown("### 1. Inventario Poblacional (Diferenciado) y Módulos")

# --- CONEXIÓN CON MEMORIA GLOBAL (CENSOS ICA Y DEMOGRAFÍA) ---
# Intentamos leer la memoria. Si no hay nada (el usuario no ha pasado por las otras páginas), usamos el valor por defecto de la cuenca.
mem_pob_res = st.session_state.get('pob_residente_proy', def_pob_res)
mem_bovinos = st.session_state.get('ica_bovinos_proy', def_bov)
mem_porcinos = st.session_state.get('ica_porcinos_proy', def_por)
mem_aves = st.session_state.get('ica_aves_proy', def_ave)

c_p1, c_p2, c_p3, c_p4 = st.columns(4)
pob_residente = c_p1.number_input("🏘️ Pob. Residente (Cuenca):", value=int(mem_pob_res), step=1000, key="sh_pob_residente")
pob_externa = c_p2.number_input("🏙️ Pob. Externa (Trasvase):", value=int(def_pob_ext), step=50000, key="sh_pob_externa")
cabezas_bovinas = c_p3.number_input("🐄 Bovinos (Censo ICA):", value=int(mem_bovinos), step=1000, key="sh_bovinos_ica")
cabezas_porcinas = c_p4.number_input("🐖 Porcinos (Censo ICA):", value=int(mem_porcinos), step=1000, key="sh_porcinos_ica")

with st.expander("⚙️ Ajustar Módulos de Consumo y Generación (Agua y Residuos)"):
    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    dot_hum = c_d1.number_input("Dotación Humana (L/d):", value=150, key="sh_dot_hum")
    dot_bov = c_d2.number_input("Dotación Bovina (L/d):", value=40, key="sh_dot_bov")
    dot_por = c_d3.number_input("Dotación Porcina (L/d):", value=15, key="sh_dot_por")
    cabezas_aves = c_d4.number_input("🐔 Aves (Censo ICA):", value=int(mem_aves), step=5000, key="sh_aves_ica")
    
    st.markdown("**Residuos Sólidos (Aplicable solo a Población Residente):**")
    c_rs1, c_rs2 = st.columns(2)
    kg_rs_hab = c_rs1.number_input("Generación RS (kg/hab/día):", value=0.8, step=0.1, key="sh_rs_generacion")
    pct_organico = c_rs2.number_input("Fracción Orgánica (%):", value=55.0, step=5.0, key="sh_rs_fraccion")

with st.expander("⚙️ Ajustar Módulos de Consumo y Generación (Agua y Residuos)"):
    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    dot_hum = c_d1.number_input("Dotación Humana (L/d):", value=150)
    dot_bov = c_d2.number_input("Dotación Bovina (L/d):", value=40)
    dot_por = c_d3.number_input("Dotación Porcina (L/d):", value=15)
    cabezas_aves = c_d4.number_input("🐔 Aves (Unidades):", value=def_ave, step=5000)
    
    st.markdown("**Residuos Sólidos (Aplicable solo a Población Residente):**")
    c_rs1, c_rs2 = st.columns(2)
    kg_rs_hab = c_rs1.number_input("Generación RS (kg/hab/día):", value=0.8, step=0.1)
    pct_organico = c_rs2.number_input("Fracción Orgánica (%):", value=55.0, step=5.0)

st.markdown("### 2. Balance Metabólico Integral")

# A. EXTRACCIÓN de Agua - Demandas Hídricas- (Suma a residentes y ciudad externa)
pob_total_agua = pob_residente + pob_externa
dem_hum_m3_dia = (pob_total_agua * dot_hum) / 1000
dem_bov_m3_dia = (cabezas_bovinas * dot_bov) / 1000
dem_por_m3_dia = (cabezas_porcinas * dot_por) / 1000
dem_ave_m3_dia = (cabezas_aves * 0.3) / 1000
dem_total_m3_s = (dem_hum_m3_dia + dem_bov_m3_dia + dem_por_m3_dia + dem_ave_m3_dia) / 86400

# B. CARGA ORGÁNICA (Solo penaliza la Población Residente en la cuenca)
carga_hum_ton = (pob_residente * 18.25) / 1000
carga_bov_ton = (cabezas_bovinas * 292.0) / 1000
carga_por_ton = (cabezas_porcinas * 91.25) / 1000
carga_ave_ton = (cabezas_aves * 5.47) / 1000
carga_total_ton = carga_hum_ton + carga_bov_ton + carga_por_ton + carga_ave_ton

# C. RESIDUOS SÓLIDOS LOCALES
rs_total_ton_ano = (pob_residente * kg_rs_hab * 365) / 1000
rs_org_ton_ano = rs_total_ton_ano * (pct_organico / 100.0)
rs_inorg_ton_ano = rs_total_ton_ano - rs_org_ton_ano

# --- VISUALIZACIÓN ---
c_res1, c_res2 = st.columns([1, 1.5])

with c_res1:
    st.metric("💧 Extracción Continua (Agua)", f"{dem_total_m3_s:,.2f} m³/s")
    st.metric("☣️ Carga Orgánica al Río (DBO5)", f"{carga_total_ton:,.0f} Ton/año")
    st.metric("🗑️ Residuos Sólidos Totales", f"{rs_total_ton_ano:,.0f} Ton/año", f"{rs_org_ton_ano:,.0f} Ton Orgánicas", delta_color="off")
    
    if st.button("🚀 Actualizar Dinámica del Embalse", use_container_width=True):
        st.session_state['demanda_total_m3s'] = dem_total_m3_s
        st.success("✅ Extracción guardada. Sube al panel superior para ver el impacto en la Resiliencia.")

with c_res2:
    # Gráfico Sunburst Combinado de Metabolismo
    df_metab = pd.DataFrame([
        {"Macro", "Agua (Extracción)", "Sector", "Humano", "Valor", dem_hum_m3_dia * 365},
        {"Macro", "Agua (Extracción)", "Sector", "Agropecuario", "Valor", (dem_bov_m3_dia+dem_por_m3_dia+dem_ave_m3_dia) * 365},
        {"Macro", "Vertimientos (DBO)", "Sector", "Humano", "Valor", carga_hum_ton},
        {"Macro", "Vertimientos (DBO)", "Sector", "Agropecuario", "Valor", carga_bov_ton+carga_por_ton+carga_ave_ton},
        {"Macro", "Residuos Sólidos", "Sector", "Orgánicos", "Valor", rs_org_ton_ano},
        {"Macro", "Residuos Sólidos", "Sector", "Inorgánicos", "Valor", rs_inorg_ton_ano}
    ], columns=["Macro", "Sub", "Sector", "Tipo", "dummy", "Valor"]) # Formato seguro para Plotly
    
    # Reestructuramos simple para px.sunburst
    df_plot = pd.DataFrame({
        "Categoria": ["Agua (Extracción Mm³/a)", "Agua (Extracción Mm³/a)", "Vertimientos (Ton DBO)", "Vertimientos (Ton DBO)", "Residuos Sólidos (Ton)", "Residuos Sólidos (Ton)"],
        "Subcategoria": ["Urbana", "Agropecuaria", "Urbana", "Agropecuaria", "Orgánicos (Lixiviables)", "Inorgánicos (Aprovechables/Rechazo)"],
        "Valor": [(dem_hum_m3_dia*365)/1e6, ((dem_bov_m3_dia+dem_por_m3_dia+dem_ave_m3_dia)*365)/1e6, carga_hum_ton, carga_bov_ton+carga_por_ton+carga_ave_ton, rs_org_ton_ano, rs_inorg_ton_ano]
    })
    
    import plotly.express as px
    fig_sun = px.sunburst(df_plot, path=['Categoria', 'Subcategoria'], values='Valor', 
                          title="Distribución del Metabolismo Material (Proporcional por Categoría)",
                          color='Categoria', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sun.update_layout(height=400, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_sun, use_container_width=True)

# =========================================================================
# 8. HUELLA HÍDRICA TERRITORIAL (CONEXIÓN MAESTROS ICA + DANE + MEMORIA GLOBAL)
# =========================================================================
st.markdown("---")
st.header("💧 Metabolismo Hídrico: Presión Demográfica y Agropecuaria")
st.info("Cálculo de la demanda hídrica real integrando la población humana (DANE) y los inventarios pecuarios alojados en la nube (ICA).")

col_h1, col_h2 = st.columns([1, 1.5])

with col_h1:
    st.subheader("1. Conexión a Censos ICA (Supabase)")
    
    # 🧠 MOTOR DE CACHÉ ESTRICTO
    @st.cache_data(show_spinner=False, ttl=3600)
    def descargar_maestro_ica(url):
        import pandas as pd
        import requests, io
        try:
            res = requests.get(url)
            if res.status_code == 200:
                return pd.read_csv(io.BytesIO(res.content), encoding='utf-8-sig', sep=None, engine='python')
        except: return pd.DataFrame()
    
    # 🚀 SLIDER DEL TIEMPO (Al moverlo, o al cambiar el menú lateral, recalcula solo)
    anio_censo = st.slider("📅 Año del Censo Pecuario:", 2018, 2025, 2024)
    
    # --- CÁLCULO AUTOMÁTICO (Sin botón) ---
    bovinos_tot, porcinos_tot, aves_tot = 0, 0, 0
    import unicodedata
    
    def limpiar_mpio(m):
        if pd.isna(m): return ""
        return unicodedata.normalize('NFKD', str(m).upper()).encode('ascii', 'ignore').decode('utf-8')
    
    # Determinamos los municipios aportantes según el nodo seleccionado en el sidebar
    mpios_cuenca = []
    if "Grande" in nodo_seleccionado:
        mpios_cuenca = ["BELMIRA", "DONMATIAS", "DON MATIAS", "SAN PEDRO", "ENTRERRIOS", "SANTA ROSA"]
    elif "Fe" in nodo_seleccionado:
        mpios_cuenca = ["RETIRO", "CEJA", "RIONEGRO"]
        
    mpios_cuenca_limpios = [limpiar_mpio(m) for m in mpios_cuenca]

    archivos_maestros = [
        ("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Bovinos.csv", "TOTAL_BOVINOS", "Bovinos"),
        ("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Porcinos.csv", "TOTAL_CERDOS", "Porcinos"),
        ("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Aves.csv", "TOTAL_AVES_CAP_OCUPADA_MAS_AVES_TRASPATIO", "Aves")
    ]
    
    for url_censo, col_total, nombre_animal in archivos_maestros:
        df = descargar_maestro_ica(url_censo)
        if not df.empty:
            df_anio = df[df['AÑO'] == anio_censo].copy()
            if not df_anio.empty:
                col_mpio = next((c for c in df_anio.columns if 'MUNICIPIO' in c), None)
                col_tot_real = next((c for c in df_anio.columns if col_total in c), None)
                if col_mpio and col_tot_real:
                    df_anio['MPIO_NORM'] = df_anio[col_mpio].apply(limpiar_mpio)
                    filtro = df_anio['MPIO_NORM'].apply(lambda x: any(m in x for m in mpios_cuenca_limpios))
                    df_final = df_anio[filtro]
                    suma_animales = pd.to_numeric(df_final[col_tot_real], errors='coerce').sum()
                    if "Bovinos" in nombre_animal: bovinos_tot = suma_animales
                    elif "Porcinos" in nombre_animal: porcinos_tot = suma_animales
                    elif "Aves" in nombre_animal: aves_tot = suma_animales
            
    st.success(f"✅ Metabolismo Pecuario de **{nodo_seleccionado}** actualizado.")
    st.info(f"🐄 Bovinos: **{bovinos_tot:,.0f}** | 🐖 Porcinos: **{porcinos_tot:,.0f}** | 🐔 Aves: **{aves_tot:,.0f}**")
    
    # Guardamos en la memoria global en tiempo real
    st.session_state['ica_bovinos_calc'] = bovinos_tot
    st.session_state['ica_porcinos_calc'] = porcinos_tot
    st.session_state['ica_aves_calc'] = aves_tot
    
with col_h2:
    st.subheader("2. Demanda Total (Urbana + Rural)")
    
    # Traemos del Aleph la población humana y los animales
    pob_humana_memoria = st.session_state.get('aleph_pob_total', 4000000) 
    cabezas_bovinas = st.session_state.get('ica_bovinos_calc', 0)
    cabezas_porcinas = st.session_state.get('ica_porcinos_calc', 0)
    cabezas_aves = st.session_state.get('ica_aves_calc', 0)
        
    c_i1, c_i2, c_i3, c_i4 = st.columns(4)
    pob_humana = c_i1.number_input("👥 Pob (Hab):", value=int(pob_humana_memoria))
    cabezas_bovinas = c_i2.number_input("🐄 Bovinos:", value=int(cabezas_bovinas))
    cabezas_porcinas = c_i3.number_input("🐖 Porcinos:", value=int(cabezas_porcinas))
    cabezas_aves_in = c_i4.number_input("🐔 Aves:", value=int(cabezas_aves)) 
    
    st.markdown("### 📊 Demanda Metabólica Equivalente")
    
    # Parámetros estándar de consumo
    consumo_humano_ld = 150 # Litros/hab/día
    consumo_bovino_ld = 40  # Litros/bovino/día
    consumo_porcino_ld = 15 # Litros/cerdo/día
    consumo_ave_ld = 0.3    # Litros/ave/día 
    
    demanda_humana_m3_dia = (pob_humana * consumo_humano_ld) / 1000
    demanda_agro_m3_dia = ((cabezas_bovinas * consumo_bovino_ld) + (cabezas_porcinas * consumo_porcino_ld) + (cabezas_aves_in * consumo_ave_ld)) / 1000
    demanda_total_m3_dia = demanda_humana_m3_dia + demanda_agro_m3_dia
    
    # Convertir a m3/s
    demanda_total_m3_s = demanda_total_m3_dia / 86400  
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Demanda Humana (m³/día)", f"{demanda_humana_m3_dia:,.1f}")
    c_m2.metric("Demanda Agro (m³/día)", f"{demanda_agro_m3_dia:,.1f}")
    c_m3.metric("Extracción Continua", f"{demanda_total_m3_s:,.3f} m³/s", delta_color="inverse")
    
    if st.button("💾 Enviar Datos al Modelo (Memoria Global)", help="..."):
        # 1. Inyectar Demanda
        st.session_state['demanda_total_m3s'] = demanda_total_m3_s
        
        # 2. Inyectar Oferta REAL (Suma tus variables de entrada aquí)
        # Reemplaza 'variable_q_natural', 'variable_bombeo1', etc., por los nombres reales de tus inputs
        oferta_calculada = variable_q_natural + variable_bombeo1 + variable_bombeo2 
        
        # Nota: Si ya tienes una variable que sumaba el total de entradas, ponla directo:
        # oferta_calculada = total_inflows 
            
        st.session_state['aleph_oferta_hidrica'] = oferta_calculada
        
        st.success(f"✅ ¡Memoria actualizada! Demanda: {demanda_total_m3_s:.2f} m³/s | Oferta: {oferta_calculada:.2f} m³/s")

# ==============================================================================
# 🕸️ DIBUJO DEL MAPA CONCEPTUAL (Se inyecta en la parte superior)
# ==============================================================================
with contenedor_sankey.container():
    with st.expander("🕸️ Mapa Conceptual: Topología del Metabolismo Hídrico", expanded=True):
        st.markdown(f"Visualización en tiempo real de las transferencias de caudal (m³/s) para el **Embalse {nodo_seleccionado}**.")
        
        labels = [f"Embalse {nodo_seleccionado}"]
        source, target, value, color = [], [], [], []
        idx = 1

        for nombre, q in afluentes_inputs.items():
            if q > 0:
                labels.append(nombre)
                source.append(idx)
                target.append(0)
                value.append(q)
                color.append("rgba(46, 204, 113, 0.6)")
                idx += 1

        for nombre, q in trasvases_inputs.items():
            if q > 0:
                labels.append(f"Bombeo {nombre} ⚡(-)")
                source.append(idx)
                target.append(0)
                value.append(q)
                color.append("rgba(231, 76, 60, 0.8)")
                idx += 1

        if val_acueducto > 0:
            labels.append("Acueducto (Aburrá)")
            source.append(0)
            target.append(idx)
            value.append(val_acueducto)
            color.append("rgba(52, 152, 219, 0.6)")
            idx += 1
            
        if val_turbinado > 0:
            labels.append("Generación ⚡(+)")
            source.append(0)
            target.append(idx)
            value.append(val_turbinado)
            color.append("rgba(241, 196, 15, 0.8)")
            idx += 1
            
        if val_ecologico > 0:
            labels.append("Río Abajo (Eco)")
            source.append(0)
            target.append(idx)
            value.append(val_ecologico)
            color.append("rgba(149, 165, 166, 0.6)")
            idx += 1
            
        if datos_nodo["evaporacion_m3s"] > 0:
            labels.append("Evaporación")
            source.append(0)
            target.append(idx)
            value.append(datos_nodo["evaporacion_m3s"])
            color.append("rgba(189, 195, 199, 0.3)")

        fig_sankey = go.Figure(data=[go.Sankey(
            textfont=dict(size=15, color="black", family="Arial Black"), 
            node=dict(pad=20, thickness=30, line=dict(color="black", width=0.5), label=labels, color="#2C3E50"),
            link=dict(source=source, target=target, value=value, color=color)
        )])
        fig_sankey.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=50))
        st.plotly_chart(fig_sankey, use_container_width=True)
        
# =========================================================================
# 8. MATEMÁTICA Y CIENCIA
# =========================================================================
with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)"):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad:")
    st.markdown("$$\\frac{\\Delta S}{\\Delta t} = I_{nat} + \\sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}$$")
    st.markdown("Si $\\frac{\\Delta S}{\\Delta t}$ es negativo de forma sostenida (ej. durante un fenómeno de El Niño donde $I_{nat} \\approx 0$), el volumen útil del embalse se agota, generando racionamiento en la metrópolis externa.")
