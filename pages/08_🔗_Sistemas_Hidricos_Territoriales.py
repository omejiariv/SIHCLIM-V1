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

# =========================================================================
# 2. PANEL DE OPERACIONES Y CONTROLES (RECEPTOR CLIMÁTICO Y METABÓLICO)
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")
nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", list(sistemas_embalses.keys()))
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

st.markdown("---")
st.subheader(f"🕸️ Topología del Metabolismo: {nodo_seleccionado} (Agua y Energía)")

labels = [f"Embalse {nodo_seleccionado}"]
source, target, value, color = [], [], [], []
idx = 1

for nombre, q in afluentes_inputs.items():
    if q > 0:
        labels.append(nombre); source.append(idx); target.append(0); value.append(q); color.append("rgba(46, 204, 113, 0.6)")
        idx += 1

for nombre, q in trasvases_inputs.items():
    if q > 0:
        labels.append(f"Bombeo {nombre} ⚡(-)"); source.append(idx); target.append(0); value.append(q); color.append("rgba(231, 76, 60, 0.8)")
        idx += 1

if val_acueducto > 0:
    labels.append("Acueducto (Aburrá)"); source.append(0); target.append(idx); value.append(val_acueducto); color.append("rgba(52, 152, 219, 0.6)")
    idx += 1
if val_turbinado > 0:
    labels.append("Generación ⚡(+)"); source.append(0); target.append(idx); value.append(val_turbinado); color.append("rgba(241, 196, 15, 0.8)")
    idx += 1
if val_ecologico > 0:
    labels.append("Río Abajo (Eco)"); source.append(0); target.append(idx); value.append(val_ecologico); color.append("rgba(149, 165, 166, 0.6)")
    idx += 1
if datos_nodo["evaporacion_m3s"] > 0:
    labels.append("Evaporación"); source.append(0); target.append(idx); value.append(datos_nodo["evaporacion_m3s"]); color.append("rgba(189, 195, 199, 0.3)")

# Ajuste 2: Se añaden márgenes (b=50) para que Evaporación no quede cortada en pantalla completa
fig_sankey = go.Figure(data=[go.Sankey(
    textfont=dict(size=15, color="black", family="Arial Black"), 
    node=dict(pad=20, thickness=30, line=dict(color="black", width=0.5), label=labels, color="#2C3E50"),
    link=dict(source=source, target=target, value=value, color=color)
)])
fig_sankey.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=50))
st.plotly_chart(fig_sankey, use_container_width=True)

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

# =========================================================================
# 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (EXPLORADOR DE ESCENARIOS)
# =========================================================================
st.markdown("---")
st.subheader(f"📈 Proyección Dinámica de Seguridad Hídrica {nodo_seleccionado} (2024 - 2050)")
st.caption("Simulación a largo plazo que integra crecimiento poblacional, pérdida base por Cambio Climático y la variabilidad cíclica/extrema del fenómeno ENSO.")

# Creamos dos pestañas para no perder el resumen general y ganar el análisis profundo
tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono de Incertidumbre)"])

anios_proj = list(range(2024, 2051))

# -------------------------------------------------------------------------
# PESTAÑA 1: TU GRÁFICA ACTUAL (INTACTA)
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
# PESTAÑA 2: EL NUEVO LABORATORIO DE ESCENARIOS (CURVAS SELECCIONABLES)
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
            
            # Asignar la anomalía según si es onda o línea constante
            f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) if val_esc == "onda" else val_esc
            f_cli_total = f_cc_base + f_enso
            
            o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
            c_m3 = (demanda_m3s_base * f_dem) * 31536000
            
            # Calcular solo el indicador seleccionado
            if ind_sel == "Neutralidad": val = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
            elif ind_sel == "Resiliencia": val = min(100.0, ((capacidad_embalse_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
            elif ind_sel == "Estrés Hídrico": val = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
            else: 
                fac_dil = (o_m3 / (c_m3 + 1))
                val = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
                
            datos_esc.append({"Año": a, "Escenario Climático": nombre_esc, "Valor (%)": val})
            
    if datos_esc:
        df_esc = pd.DataFrame(datos_esc)
        
        # Paleta de colores fija para que cada escenario tenga sentido (Rojo = Niño Severo, Azul = Niña Fuerte)
        color_map = {
            "Onda Dinámica (Ciclo de 4.5 años)": "#9b59b6",  # Morado
            "Condición Neutra (Línea Base)": "#34495e",       # Gris Oscuro
            "🟡 El Niño Moderado Constante (-15%)": "#f1c40f",
            "🔴 El Niño Severo Constante (-35%)": "#e74c3c",
            "🟢 La Niña Moderada Constante (+15%)": "#2ecc71",
            "🔵 La Niña Fuerte Constante (+35%)": "#3498db"
        }
        
        fig_esc = px.line(df_esc, x="Año", y="Valor (%)", color="Escenario Climático", color_discrete_map=color_map, markers=False)
        fig_esc.update_traces(line=dict(width=3)) # Líneas más gruesas
        
        # Marcadores visuales de riesgo
        if ind_sel == "Estrés Hídrico":
            fig_esc.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.05, layer="below", annotation_text="Estrés Crítico (>40%)")
        elif ind_sel == "Resiliencia":
            fig_esc.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.05, layer="below", annotation_text="Colapso Hídrico (<50%)")
            
        fig_esc.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig_esc, use_container_width=True)
    else:
        st.info("👈 Seleccione al menos una curva climática para visualizar.")

# =========================================================================
# 6. RANKING TERRITORIAL Y DISPERSIÓN
# =========================================================================
# Ajuste 3: Incorporación del Ranking de Nodos
st.markdown("---")
st.subheader("🏆 Ranking Territorial y Dispersión de Índices (Todos los Nodos)")
st.info("El sistema evalúa de forma cruzada el estado base de todos los embalses para identificar qué territorio requiere intervención prioritaria.")

datos_ranking = []
for nombre, param in sistemas_embalses.items():
    # Cálculo base para el ranking, asumiendo su oferta y demanda estándar
    o_base = sum(param["afluentes_naturales"].values()) + sum(param["trasvases"].values())
    d_base = param["demanda_acueducto_m3s"] + (param["generacion_energia_m3s"] * 0.1)
    
    o_m3 = o_base * 31536000
    c_m3 = d_base * 31536000
    cap_m3 = param["capacidad_util_Mm3"] * 1000000
    
    # Si es el nodo actual, usamos los valores dinámicos calculados arriba
    if nombre == nodo_seleccionado:
        n_val = ind_neutralidad
        r_val = ind_resiliencia
        e_val = ind_estres
        c_val = ind_calidad
    else:
        # Valores por defecto para los demás
        vol_rep = param["ha_conservadas_base"] * 2500 + (120 * 1200) 
        n_val = min(100.0, (vol_rep / c_m3) * 100) if c_m3 > 0 else 100.0
        r_val = min(100.0, ((cap_m3 + o_m3) / ((c_m3+1) * 2)) * 100)
        e_val = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
        fac_dil = (o_m3 / (c_m3 + 1))
        c_val = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (120 * 0.05)))
        
    urgencia = (e_val * 0.5) + ((100 - r_val) * 0.3) + ((100 - c_val) * 0.2)
    
    datos_ranking.append({
        "Territorio": nombre,
        "Urgencia Intervención": urgencia,
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

with c_box:
    df_melt = df_ranking.melt(id_vars=["Territorio"], value_vars=["Neutralidad (%)", "Resiliencia (%)", "Estrés Hídrico (%)", "Calidad de Agua (%)"], var_name="Índice", value_name="Valor (%)")
    fig_box = px.box(df_melt, x="Índice", y="Valor (%)", color="Índice", points="all",
                     title="Distribución Regional de Indicadores",
                     color_discrete_map={"Neutralidad (%)": "#2ecc71", "Resiliencia (%)": "#3498db", "Estrés Hídrico (%)": "#e74c3c", "Calidad de Agua (%)": "#9b59b6"})
    fig_box.update_layout(height=350, showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================================
# 8. HUELLA HÍDRICA TERRITORIAL (CONEXIÓN CENSOS ICA + DANE + MEMORIA GLOBAL)
# =========================================================================
st.markdown("---")
st.header("💧 Metabolismo Hídrico: Presión Demográfica y Agropecuaria")
st.info("Cálculo de la demanda hídrica real integrando la población humana (DANE) y los inventarios pecuarios alojados en la nube (ICA).")

col_h1, col_h2 = st.columns([1, 1.5])

with col_h1:
    st.subheader("1. Conexión a Censos ICA (Supabase)")
    
    url_base = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url") or st.secrets.get("supabase", {}).get("SUPABASE_URL")
    key_supabase = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key") or st.secrets.get("supabase", {}).get("SUPABASE_KEY")

    if url_base and key_supabase:
        from supabase import create_client
        cliente_supabase = create_client(url_base, key_supabase)
        
        try:
            archivos_ica = cliente_supabase.storage.from_("sihcli_maestros").list("censos_ICA")
            lista_archivos = [a['name'] for a in archivos_ica if a['name'] != '.emptyFolderPlaceholder']
            
            if lista_archivos:
                archivo_seleccionado = st.selectbox("Seleccione el Censo ICA a evaluar:", lista_archivos)
                
                if st.button("📥 Analizar Censo y Calcular Huella"):
                    with st.spinner("Descargando y procesando matriz ICA..."):
                        url_limpia = url_base.strip().strip("'").strip('"').rstrip('/')
                        ruta_censo = f"{url_limpia}/storage/v1/object/public/sihcli_maestros/censos_ICA/{archivo_seleccionado}"
                        
                        import requests
                        import io
                        res_censo = requests.get(ruta_censo)
                        
                        if res_censo.status_code == 200:
                            if archivo_seleccionado.endswith(('.xlsx', '.xls')):
                                df_ica = pd.read_excel(io.BytesIO(res_censo.content))
                            else:
                                df_ica = pd.read_csv(io.BytesIO(res_censo.content))
                            
                            st.success(f"✅ Matriz ICA cargada: {len(df_ica)} registros.")
                            st.session_state['df_ica_cargado'] = df_ica
                        else:
                            st.error("Error al descargar el archivo del bucket.")
            else:
                st.warning("No hay censos ICA subidos en la carpeta de Supabase.")
        except Exception as e:
            st.error(f"Error conectando al bucket: {e}")
    else:
        st.warning("Faltan credenciales de Supabase en secrets para leer los censos.")

with col_h2:
    st.subheader("2. Demanda Total (Urbana + Rural)")
    
    # 1. Recuperar población humana de la MEMORIA GLOBAL (st.session_state) o usar un valor por defecto (Valle de Aburrá aprox)
    pob_humana_memoria = st.session_state.get('poblacion_total', 4000000) 
    
    # 2. Recuperar inventario animal del Censo ICA
    cabezas_bovinas = 0
    cabezas_porcinas = 0
    if 'df_ica_cargado' in st.session_state:
        df_ica = st.session_state['df_ica_cargado']
        cols_bov = [c for c in df_ica.columns if 'bovin' in str(c).lower() or 'vaca' in str(c).lower()]
        cols_por = [c for c in df_ica.columns if 'porcin' in str(c).lower() or 'cerdo' in str(c).lower()]
        if cols_bov: cabezas_bovinas = df_ica[cols_bov[0]].sum()
        if cols_por: cabezas_porcinas = df_ica[cols_por[0]].sum()
        
    c_i1, c_i2, c_i3 = st.columns(3)
    pob_humana = c_i1.number_input("👥 Población Humana (Hab):", value=int(pob_humana_memoria), help="Puedes modificarlo, o se auto-llenará si vienes de la página Demografía.")
    cabezas_bovinas = c_i2.number_input("🐄 Inventario Bovino:", value=int(cabezas_bovinas))
    cabezas_porcinas = c_i3.number_input("🐖 Inventario Porcino:", value=int(cabezas_porcinas))
    
    st.markdown("### 📊 Demanda Metabólica Equivalente")
    
    # Parámetros estándar de consumo
    consumo_humano_ld = 150 # Litros por habitante al día
    consumo_bovino_ld = 40
    consumo_porcino_ld = 15
    
    demanda_humana_m3_dia = (pob_humana * consumo_humano_ld) / 1000
    demanda_agro_m3_dia = ((cabezas_bovinas * consumo_bovino_ld) + (cabezas_porcinas * consumo_porcino_ld)) / 1000
    demanda_total_m3_dia = demanda_humana_m3_dia + demanda_agro_m3_dia
    
    # Convertir a m3/s para cruzarlo con el WRI
    demanda_total_m3_s = demanda_total_m3_dia / 86400  
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Demanda Humana (m³/día)", f"{demanda_humana_m3_dia:,.1f}")
    c_m2.metric("Demanda Agro (m³/día)", f"{demanda_agro_m3_dia:,.1f}")
    c_m3.metric("Extracción Continua", f"{demanda_total_m3_s:,.3f} m³/s", delta_color="inverse")
    
    if st.button("💾 Enviar Demanda al Modelo (Memoria Global)"):
        st.session_state['demanda_total_m3s'] = demanda_total_m3_s
        st.success("✅ Dato inyectado en la memoria global. Si ajustaste la población, esto afectará a Calidad y Vertimientos.")
        
# =========================================================================
# 8. MATEMÁTICA Y CIENCIA
# =========================================================================
with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)"):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad:")
    st.markdown("$$\\frac{\\Delta S}{\\Delta t} = I_{nat} + \\sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}$$")
    st.markdown("Si $\\frac{\\Delta S}{\\Delta t}$ es negativo de forma sostenida (ej. durante un fenómeno de El Niño donde $I_{nat} \\approx 0$), el volumen útil del embalse se agota, generando racionamiento en la metrópolis externa.")
