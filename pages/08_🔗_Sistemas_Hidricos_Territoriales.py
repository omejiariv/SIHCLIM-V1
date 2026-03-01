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
        "ha_conservadas_base": 300.0  # <--- Base dinámica
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
        "ha_conservadas_base": 1500.0 # <--- Base dinámica
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
        "ha_conservadas_base": 5000.0 # <--- Base dinámica
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
        "ha_conservadas_base": 1000.0 # <--- Base dinámica
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
        "ha_conservadas_base": 10000.0 # <--- Base dinámica
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
            # Construimos la ruta directa a tu archivo de Predios
            ruta_predios = f"{url_limpia}/storage/v1/object/public/{nombre_bucket}/Puntos_de_interes/PrediosEjecutados.geojson"
            import requests
            import tempfile
            
            res_predios = requests.get(ruta_predios)
            if res_predios.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp_p:
                    tmp_p.write(res_predios.content)
                    tmp_path_p = tmp_p.name
                
                gdf_predios = gpd.read_file(tmp_path_p)
                
                # Sumar por embalse leyendo tus columnas reales 'EMBALSE' y 'AREA_HA'
                if 'EMBALSE' in gdf_predios.columns and 'AREA_HA' in gdf_predios.columns:
                    resumen_predios = gdf_predios.groupby('EMBALSE')['AREA_HA'].sum().to_dict()
                    
                    # Mapear el resultado al diccionario del modelo
                    for embalse_mapa, ha_total in resumen_predios.items():
                        nombre_mapa_limpio = str(embalse_mapa).lower().strip()
                        for nombre_nodo in sistemas_embalses.keys():
                            # Coincidencia inteligente (ej. "La Fe" con "La Fe")
                            if nombre_mapa_limpio in nombre_nodo.lower() or (nombre_mapa_limpio == "riogrande ii" and "grande" in nombre_nodo.lower()):
                                sistemas_embalses[nombre_nodo]["ha_conservadas_base"] = float(ha_total)
                                
                    st.sidebar.success("🌲 Áreas de conservación (PrediosEjecutados) inyectadas desde la nube.")
        except Exception as e:
            st.sidebar.warning(f"No se pudo cargar la capa de predios: {e}")

# =========================================================================
# 2. PANEL DE OPERACIONES Y CONTROLES
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")
nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", list(sistemas_embalses.keys()))

datos_nodo = sistemas_embalses[nodo_seleccionado]

st.markdown(f"### 💧 Balance de Masa en Tiempo Real: {nodo_seleccionado}")
st.info(f"**Capacidad Útil Máxima:** {datos_nodo['capacidad_util_Mm3']:,.1f} Millones de m³")

col_in, col_out = st.columns(2)

# --- ENTRADAS DINÁMICAS ---
afluentes_inputs = {}
trasvases_inputs = {}

with col_in:
    st.markdown("#### 📥 ENTRADAS (Inflows)")
    st.caption("Aportes Naturales de la Cuenca (Gravedad):")
    for nombre, caudal in datos_nodo["afluentes_naturales"].items():
        max_val = float(caudal * 3) if caudal > 0 else 10.0
        afluentes_inputs[nombre] = st.slider(f"{nombre} [m³/s]:", 0.0, max_val, float(caudal), 0.1, key=f"in_{nombre}")
    
    st.caption("Bombas y Túneles (Trasvases Externos):")
    if datos_nodo["trasvases"]:
        for nombre, caudal in datos_nodo["trasvases"].items():
            trasvases_inputs[nombre] = st.slider(f"Bombeo {nombre} [m³/s]:", 0.0, float(caudal * 2), float(caudal), 0.1, key=f"tr_{nombre}")
    else:
        st.write("*(Sistema impulsado 100% por gravedad)*")

# --- SALIDAS DINÁMICAS ---
with col_out:
    st.markdown("#### 📤 SALIDAS (Outflows)")
    val_acueducto = 0.0
    if datos_nodo["demanda_acueducto_m3s"] > 0:
        val_acueducto = st.slider("Extracción Acueducto [m³/s]:", 0.0, float(datos_nodo["demanda_acueducto_m3s"] * 2), float(datos_nodo["demanda_acueducto_m3s"]), 0.1)
    
    val_turbinado = 0.0
    if datos_nodo["generacion_energia_m3s"] > 0:
        max_turb = float(datos_nodo["generacion_energia_m3s"] * 1.5)
        val_turbinado = st.slider("Caudal Turbinado (Energía) [m³/s]:", 0.0, max_turb, float(datos_nodo["generacion_energia_m3s"]), 1.0)
        
    val_ecologico = st.number_input("Caudal Ecológico / Vertimiento [m³/s]:", min_value=0.0, value=float(datos_nodo["caudal_ecologico_m3s"]), step=1.0)

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
c2.metric("Energía Generada", f"{potencia_generada_kw/1000:,.1f} MW", f"${ingreso_hora_cop/1e6:,.1f} Millones/hora", delta_color="normal")
c3.metric("Energía Consumida", f"{potencia_consumida_kw/1000:,.1f} MW", f"-${costo_hora_cop/1e6:,.1f} Millones/hora", delta_color="inverse")
c4.metric("Balance Neto", f"{balance_energetico_MW:,.1f} MW", "Superávit ⚡" if balance_energetico_MW > 0 else "Déficit 🔴" if balance_energetico_MW < 0 else "Neutro")

st.markdown("---")
# MEJORA 4: TÍTULO DINÁMICO
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
    labels.append("Acueducto (Valle Aburrá)"); source.append(0); target.append(idx); value.append(val_acueducto); color.append("rgba(52, 152, 219, 0.6)")
    idx += 1
if val_turbinado > 0:
    labels.append("Generación ⚡(+)"); source.append(0); target.append(idx); value.append(val_turbinado); color.append("rgba(241, 196, 15, 0.8)")
    idx += 1
if val_ecologico > 0:
    labels.append("Río Abajo (Ecológico)"); source.append(0); target.append(idx); value.append(val_ecologico); color.append("rgba(149, 165, 166, 0.6)")
    idx += 1
if datos_nodo["evaporacion_m3s"] > 0:
    labels.append("Evaporación"); source.append(0); target.append(idx); value.append(datos_nodo["evaporacion_m3s"]); color.append("rgba(189, 195, 199, 0.3)")

# MEJORA 3: LETRAS GRANDES Y LEGIBLES EN EL SANKEY
fig_sankey = go.Figure(data=[go.Sankey(
    textfont=dict(size=15, color="black", family="Arial Black"), # Letra nítida, gruesa y oscura
    node=dict(pad=20, thickness=30, line=dict(color="black", width=0.5), label=labels, color="#2C3E50"),
    link=dict(source=source, target=target, value=value, color=color)
)])
fig_sankey.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_sankey, use_container_width=True)

# =========================================================================
# 4. TABLERO WRI: NEUTRALIDAD, RESILIENCIA Y CALIDAD
# =========================================================================
st.markdown("---")
# MEJORA 4: TÍTULO DINÁMICO
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

activar_sig = st.toggle("✅ Incluir Área Restaurada/Conservada en el cálculo WRI", value=True)

c_inv1, c_inv2, c_inv3 = st.columns(3)
with c_inv1:
    # MEJORA 2: VALOR POR DEFECTO DINÁMICO
    ha_simuladas = st.number_input("➕ Hectáreas Conservadas / BPAs:", min_value=0.0, value=float(datos_nodo["ha_conservadas_base"]), step=50.0)
    beneficio_restauracion_m3 = (ha_simuladas if activar_sig else 0.0) * 2500 
with c_inv2:
    sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM):", min_value=0, value=120, step=5)
    beneficio_calidad_m3 = (sist_saneamiento * 1200) if activar_sig else 0
with c_inv3:
    volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
    st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3/1e6:,.2f} Mm³/año")

ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
ind_resiliencia = min(100.0, ((capacidad_embalse_m3 + oferta_anual_m3) / ((consumo_anual_m3+1) * 2)) * 100)
ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0
ind_calidad = min(100.0, max(0.0, 50.0 + ((oferta_anual_m3 / (consumo_anual_m3 + 1)) * 0.5) + (sist_saneamiento * 0.05)))

def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
    if not invertido:
        return ("🔴 CRÍTICO", "#c0392b") if valor < umbral_rojo else ("🟡 VULNERABLE", "#f39c12") if valor < umbral_verde else ("🟢 ÓPTIMO", "#27ae60")
    else:
        return ("🟢 HOLGADO", "#27ae60") if valor < umbral_verde else ("🟡 MODERADO", "#f39c12") if valor < umbral_rojo else ("🔴 CRÍTICO", "#c0392b")

# MEJORA 1: GENERADOR DE LEYENDAS INTERPRETATIVAS
def generar_leyenda(u_r, u_v, inv):
    if not inv:
        return f"🔴 &lt; {u_r}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 {u_r}-{u_v}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟢 &gt; {u_v}%"
    else:
        return f"🟢 &lt; {u_v}% &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 {u_v}-{u_r}% &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 &gt; {u_r}%"

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
        # INYECCIÓN DE LA LEYENDA BAJO EL VELOCÍMETRO
        leyenda = generar_leyenda(u_r, u_v, inv)
        st.markdown(f"<div style='text-align: center; font-size: 13px; color: #7F8C8D; margin-top: -5px;'>{leyenda}</div>", unsafe_allow_html=True)

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

# =========================================================================
# NUEVA SECCIÓN: HUELLA HÍDRICA TERRITORIAL (CENSOS ICA + DANE)
# =========================================================================
st.markdown("---")
st.header("💧 Metabolismo Hídrico: Presión Demográfica y Agropecuaria")
st.info("Cálculo de la demanda hídrica real integrando la población humana y los inventarios pecuarios alojados en la nube.")

col_h1, col_h2 = st.columns([1, 1.5])

with col_h1:
    st.subheader("1. Conexión a Censos ICA (Supabase)")
    
    # Buscamos la URL y Llave base de Supabase
    url_base = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url") or st.secrets.get("supabase", {}).get("SUPABASE_URL")
    key_supabase = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key") or st.secrets.get("supabase", {}).get("SUPABASE_KEY")

    if url_base and key_supabase:
        from supabase import create_client
        cliente_supabase = create_client(url_base, key_supabase)
        
        try:
            # Listar archivos Excel/CSV en la carpeta censos_ICA en vivo
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
    st.subheader("2. Calculadora de Huella Hídrica Total")
    
    # Parámetros estándar de consumo
    c_p1, c_p2, c_p3 = st.columns(3)
    consumo_humano_ld = c_p1.number_input("Humano (L/hab/día):", value=150)
    consumo_bovino_ld = c_p2.number_input("Bovino (L/cabeza/día):", value=40)
    consumo_porcino_ld = c_p3.number_input("Porcino (L/cabeza/día):", value=15)
    
    # 1. Recuperar población humana de la página actual
    pob_humana = 0
    if 'df_plot' in locals() and not df_plot.empty and 'Poblacion_hab' in df_plot.columns:
        pob_humana = df_plot['Poblacion_hab'].sum()
    elif 'df_veredas' in locals() and not df_veredas.empty and 'Poblacion_hab' in df_veredas.columns:
        pob_humana = df_veredas['Poblacion_hab'].sum()
        
    # 2. Recuperar inventario animal del Censo ICA
    cabezas_bovinas = 0
    cabezas_porcinas = 0
    if 'df_ica_cargado' in st.session_state:
        df_ica = st.session_state['df_ica_cargado']
        # Buscar columnas clave heurísticamente
        cols_bov = [c for c in df_ica.columns if 'bovin' in str(c).lower() or 'vaca' in str(c).lower()]
        cols_por = [c for c in df_ica.columns if 'porcin' in str(c).lower() or 'cerdo' in str(c).lower()]
        if cols_bov: cabezas_bovinas = df_ica[cols_bov[0]].sum()
        if cols_por: cabezas_porcinas = df_ica[cols_por[0]].sum()
        
    c_i1, c_i2, c_i3 = st.columns(3)
    pob_humana = c_i1.number_input("Población Humana:", value=int(pob_humana))
    cabezas_bovinas = c_i2.number_input("Inventario Bovino:", value=int(cabezas_bovinas))
    cabezas_porcinas = c_i3.number_input("Inventario Porcino:", value=int(cabezas_porcinas))
    
    st.markdown("### 📊 Demanda Metabólica Equivalente")
    
    demanda_humana_m3_dia = (pob_humana * consumo_humano_ld) / 1000
    demanda_agro_m3_dia = ((cabezas_bovinas * consumo_bovino_ld) + (cabezas_porcinas * consumo_porcino_ld)) / 1000
    demanda_total_m3_dia = demanda_humana_m3_dia + demanda_agro_m3_dia
    
    # Convertir a m3/s para cruzarlo con el WRI
    demanda_total_m3_s = demanda_total_m3_dia / 86400  
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Urbana/Humana (m³/día)", f"{demanda_humana_m3_dia:,.1f}")
    c_m2.metric("Agropecuaria (m³/día)", f"{demanda_agro_m3_dia:,.1f}")
    c_m3.metric("Demanda Continua", f"{demanda_total_m3_s:,.3f} m³/s", delta_color="inverse")
    
    if st.button("💾 Enviar Demanda al WRI (Sistemas Hídricos)"):
        st.session_state['demanda_total_m3s'] = demanda_total_m3_s
        st.success("Dato inyectado en la memoria global. ¡Ve a la página de Sistemas Hídricos Territoriales para ver el impacto!")
