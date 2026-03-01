# app.py

import streamlit as st
import plotly.express as px
import pandas as pd

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLI-POTER | Centro de Comando",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3498db;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A5276;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ENCABEZADO Y KPIS ---
st.markdown('<p class="main-header">🌊 SIHCLI-POTER</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Información Hidroclimática Integrada | Gemelo Digital Territorial</p>', unsafe_allow_html=True)

# Panel de Métricas Globales (Dashboard)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Módulos Analíticos", "9 Especializados", "Operativos")
col2.metric("Resolución Temporal", "1950 - 2070", "Datos Históricos + Proyecciones")
col3.metric("Cobertura Geográfica", "Región Andina", "Topología de Cuencas")
col4.metric("Motores de Decisión", "WRI / AHP / Turc", "Estándares Globales")

st.divider()

# --- 3. PESTAÑAS PRINCIPALES ---
tab_dashboard, tab_arquitectura, tab_aleph = st.tabs([
    "🎛️ Centro de Comando (Módulos)", 
    "🏗️ Arquitectura del Sistema", 
    "📖 Filosofía (El Aleph)"
])

# =====================================================================
# PESTAÑA 1: CENTRO DE COMANDO (GRID DE NAVEGACIÓN)
# =====================================================================
with tab_dashboard:
    st.markdown("### 🌍 EJE 1: Soporte Biofísico (Condiciones Base)")
    st.caption("Módulos dedicados a la lectura y modelación del entorno natural, el clima y el subsuelo.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.page_link("pages/01_🌦️_Clima_e_Hidrologia.py", label="**Clima e Hidrología**", icon="🌦️")
        st.markdown("<small>Tablero de control telemétrico, análisis de variabilidad y pronósticos ENSO (IRI).</small>", unsafe_allow_html=True)
    with c2:
        st.page_link("pages/10_🏔️_Geomorfologia.py", label="**Geomorfología**", icon="🏔️")
        st.markdown("<small>Análisis de Modelos Digitales de Elevación (DEM), redes de drenaje y morfometría.</small>", unsafe_allow_html=True)
    with c3:
        st.page_link("pages/02_💧_Aguas_Subterraneas.py", label="**Aguas Subterráneas**", icon="💧")
        st.markdown("<small>Modelación de recarga potencial de acuíferos mediante balance hídrico (Turc).</small>", unsafe_allow_html=True)
    with c4:
        st.page_link("pages/03_🍃_Biodiversidad.py", label="**Biodiversidad**", icon="🍃")
        st.markdown("<small>Monitor de especies (GBIF), endemismos y vulnerabilidad de flora/fauna.</small>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ EJE 2: Metabolismo Territorial (Presiones Antrópicas)")
    st.caption("Módulos enfocados en la interacción humana: cómo poblamos, demandamos e impactamos la cuenca.")
    
    c5, c6, c7 = st.columns([1, 1, 1])
    with c5:
        st.page_link("pages/08_🔗_Sistemas_Hidricos_Territoriales.py", label="**Sistemas Hídricos (Metabolismo)**", icon="🔗")
        st.markdown("<small>Topología de redes, diagrama de Sankey de trasvases y Estándares de Seguridad Hídrica (WRI).</small>", unsafe_allow_html=True)
    with c6:
        st.page_link("pages/07_👥_Demografia_y_Poblacion.py", label="**Demografía y Población**", icon="👥")
        st.markdown("<small>Proyecciones poblacionales (DANE) y censos agropecuarios (ICA) para cálculo de demanda real.</small>", unsafe_allow_html=True)
    with c7:
        st.page_link("pages/06_💧_Calidad_y_Vertimientos.py", label="**Calidad y Vertimientos**", icon="🧪")
        st.markdown("<small>Mapeo de usuarios del recurso, modelación de concesiones y cargas contaminantes.</small>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🧠 EJE 3: Síntesis y Estrategia (DSS)")
    st.caption("Sistemas de Soporte de Decisiones para planeación, ordenamiento y priorización de inversiones.")
    
    c8, c9, c10 = st.columns([1, 1, 1])
    with c8:
        st.page_link("pages/04_📊_Toma_de_Decisiones.py", label="**Toma de Decisiones (AHP)**", icon="📊")
        st.markdown("<small>Matrices multicriterio (AHP) para priorizar proyectos ambientales y de compensación (SbN).</small>", unsafe_allow_html=True)
    with c9:
        st.page_link("pages/05_🗺️_Isoyetas_HD.py", label="**Isoyetas HD y Espacialización**", icon="🗺️")
        st.markdown("<small>Generador climático: Interpolación espacial (RBF) de lluvia y escenarios predictivos.</small>", unsafe_allow_html=True)
    with c10:
        # Espacio reservado para el panel de administración
        st.page_link("pages/09_👑_Panel_Administracion.py", label="**Panel de Administración**", icon="👑")
        st.markdown("<small>Aduana SIG, carga de datos a la nube (Supabase) y gestión integral del sistema.</small>", unsafe_allow_html=True)

# =====================================================================
# PESTAÑA 2: ARQUITECTURA DEL SISTEMA (SUNBURST)
# =====================================================================
with tab_arquitectura:
    st.markdown("### Mapa Topológico del Sistema")
    st.info("Visualización jerárquica de la arquitectura de la plataforma y sus submódulos lógicos.")
    
    ids = ['SIHCLI-POTER', 'Metabolismo Territorial', 'Dinámica Poblacional', 'Calidad y Saneamiento', 'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Geomorfología', 'Embalses y Trasvases', 'Estándares WRI', 'Topología de Redes', 'Proyecciones DANE', 'Pirámides Edades', 'Censos ICA', 'Vertimientos', 'Concesiones', 'Cargas Contaminantes', 'Isoyetas HD', 'Monitoreo Clima', 'Índices ENSO', 'Balance Turc', 'Acuíferos', 'Monitor GBIF', 'AHP']
    parents = ['', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial', 'Dinámica Poblacional', 'Dinámica Poblacional', 'Dinámica Poblacional', 'Calidad y Saneamiento', 'Calidad y Saneamiento', 'Calidad y Saneamiento', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Aguas Subterráneas', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones']
    values = [100, 20, 15, 15, 15, 10, 10, 10, 5, 7, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10]

    if len(ids) == len(parents) == len(values):
        df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
        fig = px.sunburst(df, names='ids', parents='parents', values='values', color='parents', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=20, l=0, r=0, b=0), height=600, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PESTAÑA 3: EL ALEPH (FILOSOFÍA)
# =====================================================================
with tab_aleph:
    st.markdown("### La Visión Detrás de SIHCLI-POTER")
    c_aleph1, c_aleph2 = st.columns([1, 1])
    
    with c_aleph1:
        st.markdown("""
        **La Complejidad de los Andes**
        La región Andina presenta uno de los sistemas climáticos más complejos del mundo. Viajar hacia arriba es como viajar hacia los polos. Esta verticalidad y la interacción con dos océanos crean un mosaico de biodiversidad y clima, sometido al constante latido del ENSO (El Niño / La Niña).
        
        SIHCLI-POTER nace para leer este latido. Es un esfuerzo por unificar datos dispersos en una sola fuente de verdad, permitiendo a los planificadores ver el impacto de una gota de lluvia desde su caída en el páramo hasta su llegada al embalse y su distribución en la ciudad.
        """)
        
    with c_aleph2:
        st.info("""
        **Jorge Luis Borges - El Aleph (1945)**
        > *"En la parte inferior del escalón, hacia la derecha, vi una pequeña esfera tornasolada, de casi intolerable fulgor... El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí, sin disminución de tamaño. Cada cosa era infinitas cosas, porque yo la veía claramente desde todos los puntos del universo... vi en el Aleph la tierra, y en la tierra otra vez el Aleph... y sentí vértigo"*
        """)

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v3.0 | Un Aleph Hidroclimático: Plataforma de Inteligencia Territorial")

