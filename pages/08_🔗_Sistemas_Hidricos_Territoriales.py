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
# 1. BASE DE DATOS ESTRUCTURAL DE EMBALSES (Topología Corregida)
# =========================================================================
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# A. Datos paramétricos (Estructura corregida hidrológicamente)
sistemas_embalses = {
    "La Fe": {
        "capacidad_util_Mm3": 11.5, 
        "afluentes_naturales": {"Quebrada Espíritu Santo": 1.2},
        "trasvases": {"Pantanillo": 1.5, "Río Buey": 3.0, "Piedras": 0.8}, # Bombeos reales
        "demanda_acueducto_m3s": 5.0, 
        "evaporacion_m3s": 0.1,
        "caudal_ecologico_m3s": 0.3,
        "generacion_energia_m3s": 0.0
    },
    "Río Grande II": {
        "capacidad_util_Mm3": 220.0, 
        "afluentes_naturales": {
            "Río Grande": 10.0, 
            "Río Chico": 3.0, 
            "Quebrada Las Ánimas": 2.0
        },
        "trasvases": {}, # Cero bombeos externos
        "demanda_acueducto_m3s": 6.5, 
        "generacion_energia_m3s": 12.0, 
        "evaporacion_m3s": 0.5,
        "caudal_ecologico_m3s": 1.0
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

# =========================================================================
# 2. PANEL DE OPERACIONES Y CONTROLES
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")
nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", list(sistemas_embalses.keys()))

datos_nodo = sistemas_embalses[nodo_seleccionado]

st.markdown(f"### 💧 Balance de Masa en Tiempo Real: Embalse {nodo_seleccionado}")
st.info(f"**Capacidad Útil Máxima:** {datos_nodo['capacidad_util_Mm3']} Millones de m³")

col_in, col_out = st.columns(2)

# --- ENTRADAS DINÁMICAS ---
afluentes_inputs = {}
trasvases_inputs = {}

with col_in:
    st.markdown("#### 📥 ENTRADAS (Inflows)")
    
    st.caption("Aportes Naturales de la Cuenca (Gravedad):")
    for nombre, caudal in datos_nodo["afluentes_naturales"].items():
        afluentes_inputs[nombre] = st.slider(f"{nombre} [m³/s]:", 0.0, float(caudal * 3), float(caudal), 0.1, key=f"in_{nombre}")
    
    st.caption("Bombas y Túneles (Trasvases Externos):")
    if datos_nodo["trasvases"]:
        for nombre, caudal in datos_nodo["trasvases"].items():
            trasvases_inputs[nombre] = st.slider(f"Bombeo {nombre} [m³/s]:", 0.0, float(caudal * 2), float(caudal), 0.1, key=f"tr_{nombre}")
    else:
        st.write("*(Sistema sin dependencia de trasvases artificiales)*")

# --- SALIDAS DINÁMICAS ---
with col_out:
    st.markdown("#### 📤 SALIDAS (Outflows)")
    val_acueducto = st.slider("Extracción Acueducto [m³/s]:", 0.0, float(datos_nodo["demanda_acueducto_m3s"] * 2), float(datos_nodo["demanda_acueducto_m3s"]), 0.1)
    
    val_turbinado = 0.0
    if datos_nodo["generacion_energia_m3s"] > 0:
        val_turbinado = st.slider("Caudal Turbinado (Energía) [m³/s]:", 0.0, float(datos_nodo["generacion_energia_m3s"] * 2), float(datos_nodo["generacion_energia_m3s"]), 0.1)
        
    val_ecologico = st.number_input("Caudal Ecológico / Vertimiento [m³/s]:", min_value=0.0, value=float(datos_nodo["caudal_ecologico_m3s"]), step=0.1)

# =========================================================================
# 3. CÁLCULO DE BALANCE Y DIAGRAMA DE SANKEY
# =========================================================================
sum_entradas = sum(afluentes_inputs.values()) + sum(trasvases_inputs.values())
sum_salidas = val_acueducto + val_turbinado + val_ecologico + datos_nodo["evaporacion_m3s"]
balance = sum_entradas - sum_salidas

c1, c2, c3 = st.columns(3)
c1.metric("Sumatoria Entradas", f"{sum_entradas:.2f} m³/s")
c2.metric("Sumatoria Salidas", f"{sum_salidas:.2f} m³/s")
c3.metric("Balance (ΔS/Δt)", f"{balance:+.2f} m³/s", "Llenándose 📈" if balance > 0 else "Vaciándose 📉" if balance < 0 else "Estable ⚖️")

st.markdown("---")
st.subheader("🕸️ Topología del Metabolismo Hídrico (Sankey)")

# Construcción Dinámica del Grafo Sankey
labels = [f"Embalse {nodo_seleccionado}"]
source, target, value, color = [], [], [], []
idx = 1

# Nodos de Entrada Natural (Verdes)
for nombre, q in afluentes_inputs.items():
    if q > 0:
        labels.append(nombre); source.append(idx); target.append(0); value.append(q); color.append("rgba(46, 204, 113, 0.6)")
        idx += 1

# Nodos de Entrada Trasvases (Rojos)
for nombre, q in trasvases_inputs.items():
    if q > 0:
        labels.append(f"Bombeo {nombre}"); source.append(idx); target.append(0); value.append(q); color.append("rgba(231, 76, 60, 0.6)")
        idx += 1

# Nodos de Salida (Azules y Grises)
if val_acueducto > 0:
    labels.append("Acueducto (Valle de Aburrá)"); source.append(0); target.append(idx); value.append(val_acueducto); color.append("rgba(52, 152, 219, 0.6)")
    idx += 1
if val_turbinado > 0:
    labels.append("Generación Energía"); source.append(0); target.append(idx); value.append(val_turbinado); color.append("rgba(241, 196, 15, 0.6)")
    idx += 1
if val_ecologico > 0:
    labels.append("Río Abajo (Ecológico/Rebose)"); source.append(0); target.append(idx); value.append(val_ecologico); color.append("rgba(149, 165, 166, 0.6)")
    idx += 1
if datos_nodo["evaporacion_m3s"] > 0:
    labels.append("Evaporación"); source.append(0); target.append(idx); value.append(datos_nodo["evaporacion_m3s"]); color.append("rgba(189, 195, 199, 0.4)")

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(pad=20, thickness=30, line=dict(color="black", width=0.5), label=labels, color="blue"),
    link=dict(source=source, target=target, value=value, color=color)
)])
fig_sankey.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_sankey, use_container_width=True)


# =========================================================================
# 4. TABLERO WRI: NEUTRALIDAD, RESILIENCIA Y CALIDAD (INTEGRADO)
# =========================================================================
st.markdown("---")
st.subheader("🌐 Inteligencia Territorial: Neutralidad, Resiliencia y Calidad (WRI)")
st.markdown(f"Transforma el balance en tiempo real del **Embalse {nodo_seleccionado}** en indicadores estandarizados y evalúa su viabilidad futura.")

# --- 1. MÁQUINA DEL TIEMPO (PROYECCIONES) ---
st.markdown("#### ⏳ Máquina del Tiempo (Análisis de Tendencias)")
anio_analisis = st.slider("Seleccione el Año de Evaluación (Actual o Futuro):", min_value=2024, max_value=2050, value=2025, step=1)

delta_anios = anio_analisis - 2025
factor_demanda = (1 + 0.015) ** delta_anios
factor_clima = (1 - 0.005) ** delta_anios

# Integración Directa con los Sliders Superiores
q_oferta_m3s_base = sum_entradas
demanda_m3s_base = val_acueducto + val_turbinado # Demanda humana/industrial
capacidad_embalse_m3 = datos_nodo["capacidad_util_Mm3"] * 1000000

oferta_anual_m3 = (q_oferta_m3s_base * factor_clima) * 31536000
consumo_anual_m3 = (demanda_m3s_base * factor_demanda) * 31536000

# --- 2. INTEGRACIÓN DE SOLUCIONES BASADAS EN LA NATURALEZA ---
st.markdown("---")
st.markdown(f"#### 🌲 Beneficios Volumétricos (SbN) en la cuenca de: **{nodo_seleccionado}**")
st.info("Simula el impacto de predios conservados, restauración ecológica y saneamiento sobre la seguridad del embalse.")

activar_sig = st.toggle("✅ Incluir Área Restaurada/Conservada en el cálculo WRI", value=True)

c_inv1, c_inv2, c_inv3 = st.columns(3)
with c_inv1:
    ha_simuladas = st.number_input("➕ Hectáreas Conservadas / BPAs:", min_value=0.0, value=1500.0, step=50.0)
    ha_total = ha_simuladas if activar_sig else 0.0
    beneficio_restauracion_m3 = ha_total * 2500 # 2500 m3 recuperados por hectárea/año
    
with c_inv2:
    sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM):", min_value=0, value=120, step=5)
    beneficio_calidad_m3 = (sist_saneamiento * 1200) if activar_sig else 0
    
with c_inv3:
    volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
    st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3/1e6:,.2f} Mm³/año", "Compensación Total")

# --- 3. MOTORES DE CÁLCULO WRI ---
ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
# La Resiliencia en un embalse suma la oferta anual MÁS su capacidad de almacenamiento (Buffer)
ind_resiliencia = min(100.0, ((capacidad_embalse_m3 + oferta_anual_m3) / (consumo_anual_m3 * 2)) * 100) if consumo_anual_m3 > 0 else 100.0
ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0

factor_dilucion = (oferta_anual_m3 / (consumo_anual_m3 + 1)) 
ind_calidad = min(100.0, max(0.0, 50.0 + (factor_dilucion * 0.5) + (sist_saneamiento * 0.05)))

def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
    if not invertido:
        if valor < umbral_rojo: return "🔴 CRÍTICO", "#c0392b"
        elif valor < umbral_verde: return "🟡 VULNERABLE", "#f39c12"
        else: return "🟢 ÓPTIMO", "#27ae60"
    else:
        if valor < umbral_verde: return "🟢 HOLGADO", "#27ae60"
        elif valor < umbral_rojo: return "🟡 MODERADO", "#f39c12"
        else: return "🔴 CRÍTICO", "#c0392b"

# --- 4. TABLERO DE VELOCÍMETROS ---
st.markdown("---")
st.subheader(f"🧭 Tablero de Seguridad Hídrica Integral ({anio_analisis})")

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
                {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc" if not invertido else "#fff2cc"},
                {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}
        }
    ))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
    return fig

col_g1, col_g2, col_g3, col_g4 = st.columns(4)
est_neu, col_neu = evaluar_indice(ind_neutralidad, 20, 50)
est_res, col_res = evaluar_indice(ind_resiliencia, 30, 70)
est_est, col_est = evaluar_indice(ind_estres, 40, 20, invertido=True)
est_cal, col_cal = evaluar_indice(ind_calidad, 40, 70)

with col_g1: 
    st.plotly_chart(crear_velocimetro(ind_neutralidad, "Neutralidad", "#2ecc71", 20, 50), use_container_width=True)
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

# --- 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA ---
st.markdown("---")
st.subheader(f"📈 Proyección de Seguridad Hídrica del Sistema {nodo_seleccionado} (2024 - 2050)")
st.caption("Evolución de los indicadores asumiendo un crecimiento poblacional (+1.5%/año) y pérdida de escorrentía por Cambio Climático (-0.5%/año).")

anios_proj = list(range(2024, 2051, 2))
datos_proj = []
for a in anios_proj:
    f_dem = (1 + 0.015) ** (a - 2025)
    f_cli = (1 - 0.005) ** (a - 2025)
    
    o_m3 = (q_oferta_m3s_base * f_cli) * 31536000
    c_m3 = (demanda_m3s_base * f_dem) * 31536000
    
    n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
    r = min(100.0, ((capacidad_embalse_m3 + o_m3) / (c_m3 * 2)) * 100) if c_m3 > 0 else 100.0
    e = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
    
    fac_dil = (o_m3 / (c_m3 + 1))
    cal = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
    
    datos_proj.extend([
        {"Año": a, "Indicador": "Neutralidad", "Valor": n},
        {"Año": a, "Indicador": "Resiliencia", "Valor": r},
        {"Año": a, "Indicador": "Estrés Hídrico", "Valor": e},
        {"Año": a, "Indicador": "Calidad", "Valor": cal}
    ])
    
df_tendencias = pd.DataFrame(datos_proj)
fig_line = px.line(df_tendencias, x="Año", y="Valor", color="Indicador", markers=True,
                   color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Estrés Hídrico": "#e74c3c", "Calidad": "#9b59b6"})
fig_line.add_vline(x=anio_analisis, line_dash="dash", line_color="black", annotation_text=f"Año ({anio_analisis})")
fig_line.update_layout(height=350, yaxis_title="Índice (%)", xaxis_title="Año Proyectado", legend_title="Indicador WRI")
st.plotly_chart(fig_line, use_container_width=True)

# =========================================================================
# 6. MATEMÁTICA Y CIENCIA
# =========================================================================
with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)"):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad:")
    st.latex(r"\frac{\Delta S}{\Delta t} = I_{nat} + \sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}")
    st.markdown("Si $\\frac{\\Delta S}{\\Delta t}$ es negativo de forma sostenida (ej. durante un fenómeno de El Niño donde $I_{nat} \\approx 0$), el volumen útil del embalse se agota, generando racionamiento en la metrópolis externa.")
