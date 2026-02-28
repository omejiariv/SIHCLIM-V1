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
# 1. BASE DE DATOS ESTRUCTURAL DE EMBALSES (Memoria del Sistema)
# =========================================================================
# Datos paramétricos basados en las características reales de los sistemas de EPM
sistemas_embalses = {
    "La Fe": {
        "capacidad_util_Mm3": 11.5,
        "cuenca_propia": "Quebrada Espíritu Santo",
        "oferta_natural_m3s": 1.2,
        "trasvases": {"Pantanillo": 1.5, "Río Buey": 3.0, "Piedras": 0.8},
        "demanda_acueducto_m3s": 5.0, # Planta Manantiales / Ayurá
        "evaporacion_m3s": 0.1,
        "caudal_ecologico_m3s": 0.3
    },
    "Río Grande II": {
        "capacidad_util_Mm3": 220.0,
        "cuenca_propia": "Río Grande + Quebrada Las Ánimas",
        "oferta_natural_m3s": 15.0,
        "trasvases": {"Río Chico": 5.0},
        "demanda_acueducto_m3s": 6.5, # Planta Manantiales
        "generacion_energia_m3s": 12.0, # Turbinado
        "evaporacion_m3s": 0.5,
        "caudal_ecologico_m3s": 1.0
    }
}

# =========================================================================
# 2. PANEL DE OPERACIÓN (Centro de Control)
# =========================================================================
st.sidebar.header("🎛️ Centro de Operaciones")
sistema_sel = st.sidebar.selectbox("Seleccione el Nodo Principal:", list(sistemas_embalses.keys()))

datos = sistemas_embalses[sistema_sel]

st.subheader(f"💧 Balance de Masa en Tiempo Real: Embalse {sistema_sel}")
st.info(f"**Capacidad Útil Máxima:** {datos['capacidad_util_Mm3']} Millones de m³")

# Controles interactivos simulando la operación de válvulas y clima
c_in, c_out = st.columns(2)

with c_in:
    st.markdown("#### 📥 ENTRADAS (Inflows)")
    oferta_natural = st.slider(
        f"Oferta Natural ({datos['cuenca_propia']}) [m³/s]:", 
        0.0, datos['oferta_natural_m3s'] * 2, datos['oferta_natural_m3s'], step=0.1
    )
    
    caudales_trasvases = {}
    st.caption("Bombas y Túneles (Trasvases Externos):")
    for nombre_t, caudal_t in datos['trasvases'].items():
        caudales_trasvases[nombre_t] = st.slider(f"Bombeo desde {nombre_t} [m³/s]:", 0.0, caudal_t * 1.5, caudal_t, step=0.1)
        
    total_entradas = oferta_natural + sum(caudales_trasvases.values())

with c_out:
    st.markdown("#### 📤 SALIDAS (Outflows)")
    demanda_urbana = st.slider("Extracción Acueducto (Valle de Aburrá) [m³/s]:", 0.0, datos['demanda_acueducto_m3s'] * 1.5, datos['demanda_acueducto_m3s'], step=0.1)
    
    # Manejo específico si tiene turbinado de energía (como Río Grande II)
    energia = 0.0
    if "generacion_energia_m3s" in datos:
        energia = st.slider("Caudal Turbinado (Energía) [m³/s]:", 0.0, datos['generacion_energia_m3s'] * 1.5, datos['generacion_energia_m3s'], step=0.5)
        
    evaporacion = datos['evaporacion_m3s']
    caudal_ecologico = st.number_input("Caudal Ecológico / Vertimiento [m³/s]:", value=datos['caudal_ecologico_m3s'])
    
    total_salidas = demanda_urbana + energia + evaporacion + caudal_ecologico

# Ecuación de Balance: dS/dt = In - Out
balance_neto = total_entradas - total_salidas

# =========================================================================
# 3. METRICA DE VARIACIÓN DE ALMACENAMIENTO (dS/dt)
# =========================================================================
st.markdown("---")
cm1, cm2, cm3 = st.columns(3)
cm1.metric("Sumatoria Entradas", f"{total_entradas:.2f} m³/s")
cm2.metric("Sumatoria Salidas", f"{total_salidas:.2f} m³/s")

color_delta = "normal" if balance_neto >= 0 else "inverse"
estado_embalse = "Llenándose 📈" if balance_neto > 0 else ("Vaciándose 📉" if balance_neto < 0 else "Estable ⚖️")

cm3.metric("Balance (ΔS/Δt)", f"{balance_neto:+.2f} m³/s", estado_embalse, delta_color=color_delta)

# =========================================================================
# 4. DIAGRAMA DE SANKEY (Topología del Flujo Hídrico)
# =========================================================================
st.subheader("🕸️ Topología del Metabolismo Hídrico")
st.caption("Grosor de las líneas proporcional al caudal (m³/s). Permite visualizar la altísima dependencia de cuencas externas.")

# Construcción dinámica de Nodos y Enlaces para el Sankey
nodos = [f"Embalse {sistema_sel}", datos['cuenca_propia']] + list(datos['trasvases'].keys()) + ["Valle de Aburrá (Acueducto)", "Evaporación / Pérdidas", "Río Abajo (Ecológico)"]
if energia > 0: nodos.append("Generación Energía")

# Índices fijos
idx_embalse = 0
idx_propia = 1
idx_aburra = len(nodos) - (3 if energia == 0 else 4)
idx_evap = len(nodos) - (2 if energia == 0 else 3)
idx_eco = len(nodos) - (1 if energia == 0 else 2)
idx_energia = len(nodos) - 1 if energia > 0 else -1

sources = []
targets = []
values = []
colors = []

# Enlace: Cuenca Propia -> Embalse
sources.append(idx_propia)
targets.append(idx_embalse)
values.append(oferta_natural)
colors.append("rgba(46, 204, 113, 0.6)") # Verde (Natural)

# Enlaces: Trasvases -> Embalse
current_idx = 2
for nombre_t, caudal_t in caudales_trasvases.items():
    sources.append(current_idx)
    targets.append(idx_embalse)
    values.append(caudal_t)
    colors.append("rgba(231, 76, 60, 0.5)") # Rojo (Artificial / Bombeo)
    current_idx += 1

# Enlace: Embalse -> Ciudad
sources.append(idx_embalse)
targets.append(idx_aburra)
values.append(demanda_urbana)
colors.append("rgba(52, 152, 219, 0.7)") # Azul (Consumo)

# Enlace: Embalse -> Evaporación
sources.append(idx_embalse)
targets.append(idx_evap)
values.append(evaporacion)
colors.append("rgba(149, 165, 166, 0.4)") # Gris (Pérdidas)

# Enlace: Embalse -> Río Abajo
sources.append(idx_embalse)
targets.append(idx_eco)
values.append(caudal_ecologico)
colors.append("rgba(46, 204, 113, 0.6)") # Verde (Ecológico)

# Enlace: Embalse -> Turbinas
if energia > 0:
    sources.append(idx_embalse)
    targets.append(idx_energia)
    values.append(energia)
    colors.append("rgba(241, 196, 15, 0.6)") # Amarillo (Energía)

fig_sankey = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 25,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = nodos,
      color = "blue"
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = colors
    )
)])

fig_sankey.update_layout(title_text=f"Grafo de Flujos: Sistema {sistema_sel}", font_size=12, height=450)
st.plotly_chart(fig_sankey, use_container_width=True)

# =========================================================================
# 5. MATEMÁTICA Y CIENCIA
# =========================================================================
with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)"):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad:")
    st.latex(r"\frac{\Delta S}{\Delta t} = I_{nat} + \sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}")
    st.markdown("Si $\\frac{\\Delta S}{\\Delta t}$ es negativo de forma sostenida (ej. durante un fenómeno de El Niño donde $I_{nat} \\approx 0$), el volumen útil del embalse se agota, generando racionamiento en la metrópolis externa.")
