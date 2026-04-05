# app.py

import streamlit as st
from modules.utils import inicializar_torrente_sanguineo

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLI-POTER | Centro de Comando",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 💉 Inyección de Sistema Inmunológico (Datos Maestros)
inicializar_torrente_sanguineo()

# --- 2. DISEÑO DE BIENVENIDA GERENCIAL ---
st.title("🎯 SIHCLI-POTER: Gemelo Digital de Seguridad Hídrica")
st.markdown("""
### Plataforma de Inteligencia Territorial para la Gestión del Agua.
Integración de datos **satelitales, demográficos, pecuarios y climatológicos** en un solo núcleo de toma de decisiones.
""")

st.divider()

# --- 3. KPIs GLOBALES (RESUMEN EJECUTIVO) ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("🌍 Fase ENSO", st.session_state.get('enso_fase', 'Neutro'))
with c2:
    st.metric("📡 Conexión Nube", "Activa", delta="Supabase Online")
with c3:
    st.metric("🧠 Motor Físico", "Listo", delta="Smart Cache OK")

st.divider()

# --- 4. ACCESO DIRECTO Y FILOSOFÍA ---
t1, t2 = st.tabs(["🚀 Inicio Rápido", "🔬 La Visión"])

with t1:
    st.info("Utilice el menú lateral izquierdo para navegar entre los módulos de análisis.")
    st.markdown("""
    1. **Clima e Hidrología:** Monitoreo en tiempo real y Gumbel.
    2. **Geomorfología:** Análisis de terreno y red hídrica.
    3. **Toma de Decisiones:** El tablero maestro de inversión SbN.
    """)

with t2:
    st.markdown("""
    **La Complejidad de los Andes**
    SIHCLI-POTER nace para leer el latido de la montaña. Es un esfuerzo por unificar datos dispersos en una sola fuente de verdad, permitiendo a los planificadores ver el impacto de una gota de lluvia desde su caída en el páramo hasta su llegada al embalse.
    """)
    st.caption("> *'Vi en el Aleph la tierra, y en la tierra otra vez el Aleph...'* - Jorge Luis Borges")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("SIHCLI-POTER v2.0 | 2026")
