# app.py

import streamlit as st
import plotly.express as px
import pandas as pd

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE DEBE IR PRIMERO) ---
st.set_page_config(
    page_title="SIHCLI-POTER | Centro de Comando",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "Agua2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "Agua2026"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo
muro_de_acceso_beta()
# ==============================================================================


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
col1.metric("Módulos Analíticos", "15 Especializados", "Operativos") # Actualizado a 15
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
    # --- EJE 1 ---
    st.markdown("### 🌍 EJE 1: Soporte Biofísico (Condiciones Base)")
    st.caption("Módulos dedicados a la lectura y modelación del entorno natural, el clima y el subsuelo.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("pages/01_🌦️_Clima_e_Hidrologia.py", label="**Clima e Hidrología**", icon="🌦️")
        st.markdown("<small>Tablero de control telemétrico, análisis de variabilidad y pronósticos ENSO (IRI).</small>", unsafe_allow_html=True)
    with c2:
        st.page_link("pages/02_💧_Aguas_Subterraneas.py", label="**Aguas Subterráneas**", icon="💧")
        st.markdown("<small>Modelación de recarga potencial de acuíferos mediante balance hídrico (Turc).</small>", unsafe_allow_html=True)
    with c3:
        st.page_link("pages/03_🗺️_Isoyetas_HD.py", label="**Isoyetas HD y Espacialización**", icon="🗺️")
        st.markdown("<small>Generador climático: Interpolación espacial (RBF) de lluvia y escenarios predictivos.</small>", unsafe_allow_html=True)
        
    c4, c5, _ = st.columns(3)
    with c4:
        st.page_link("pages/04_🍃_Biodiversidad.py", label="**Biodiversidad**", icon="🍃")
        st.markdown("<small>Monitor de especies (GBIF), endemismos y valoración económica de servicios ecosistémicos.</small>", unsafe_allow_html=True)
    with c5:
        st.page_link("pages/05_🏔️_Geomorfologia.py", label="**Geomorfología**", icon="🏔️")
        st.markdown("<small>Análisis de Modelos Digitales de Elevación (DEM), redes de drenaje y morfometría.</small>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)
    
    # --- EJE 2 ---
    st.markdown("### ⚙️ EJE 2: Metabolismo Territorial (Presiones Antrópicas)")
    st.caption("Módulos enfocados en la dinámica poblacional y el impacto sobre la red hídrica.")
    
    c6, c7 = st.columns(2)
    with c6:
        st.page_link("pages/06_📈_Modelo_Demografico.py", label="**Modelo Demográfico (Humanos)**", icon="📈")
        st.markdown("<small>Proyecciones poblacionales (DANE) multimodelo con inyección a la Memoria Global.</small>", unsafe_allow_html=True)
        st.write("") # Espaciador
        
        # 🚀 RUTA ACTUALIZADA AQUÍ
        st.page_link("pages/06_🐄_Modelo_Pecuario.py", label="**Modelo Pecuario (Animales)**", icon="🐄")
        st.markdown("<small>Proyecciones de crecimiento (ICA) para Bovinos, Porcinos y Aves en escalas hidrográficas.</small>", unsafe_allow_html=True)
        
    with c7:
        st.page_link("pages/07_💧_Calidad_y_Vertimientos.py", label="**Calidad y Vertimientos**", icon="🧪")
        st.markdown("<small>Mapeo de usuarios del recurso, modelación de concesiones y cargas contaminantes DBO.</small>", unsafe_allow_html=True)
        st.write("") # Espaciador
        st.page_link("pages/08_🔗_Sistemas_Hidricos_Territoriales.py", label="**Sistemas Hídricos Territoriales**", icon="🔗")
        st.markdown("<small>Topología de redes, diagramas de Sankey y huella hídrica consolidada en la nube.</small>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)

    # --- EJE 3 ---
    st.markdown("### 🧠 EJE 3: Síntesis y Estrategia (DSS)")
    st.caption("Sistemas de Soporte de Decisiones para planeación, ordenamiento y priorización de inversiones.")
    
    c9, c10, _ = st.columns(3)
    with c9:
        st.page_link("pages/09_📊_Toma_de_Decisiones.py", label="**Toma de Decisiones**", icon="📊")
        st.markdown("<small>Dashboard Maestro: Estrés hídrico, Portafolio WRI y análisis multicriterio (AHP).</small>", unsafe_allow_html=True)
    with c10:
        st.page_link("pages/10_👑_Panel_Administracion.py", label="**Panel de Administración**", icon="👑")
        st.markdown("<small>Aduana SIG, carga de datos maestros a la nube (Supabase) y gestión del sistema.</small>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)

    # --- EJE 4 ---
    st.markdown("### 🛠️ EJE 4: Soporte Técnico y Herramientas")
    st.caption("Utilidades del sistema para mantenimiento, documentación y depuración del gemelo digital.")
    
    c11, c12, c13, c14 = st.columns(4)
    with c11:
        st.page_link("pages/11_⚙️_Generador.py", label="**Generador**", icon="⚙️")
    with c12:
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="**Ayuda y Docs**", icon="📚")
    with c13:
        st.page_link("pages/13_🚑_Diagnostico.py", label="**Diagnóstico**", icon="🚑")
    with c14:
        st.page_link("pages/14_🕵️_Detective.py", label="**Detective**", icon="🕵️")


# =====================================================================
# PESTAÑA 2: ARQUITECTURA DEL SISTEMA (SUNBURST)
# =====================================================================
with tab_arquitectura:
    st.markdown("### Mapa Topológico del Sistema")
    st.info("Visualización jerárquica de la arquitectura de la plataforma y sus submódulos lógicos.")
    
    # Actualizado con la nueva estructura pecuaria
    ids = ['SIHCLI-POTER', 'Soporte Biofísico', 'Metabolismo Territorial', 'Síntesis Estratégica', 'Herramientas', 
           'Clima e Hidrología', 'Aguas Subterráneas', 'Isoyetas HD', 'Biodiversidad', 'Geomorfología',
           'Modelo Demográfico', 'Modelo Pecuario', 'Calidad y Vertimientos', 'Sistemas Hídricos', 
           'Toma de Decisiones', 'Panel Administración',
           'Generador', 'Ayuda y Docs', 'Diagnóstico', 'Detective']
            
    parents = ['', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
               'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico',
               'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial',
               'Síntesis Estratégica', 'Síntesis Estratégica',
               'Herramientas', 'Herramientas', 'Herramientas', 'Herramientas']
                
    values = [100, 30, 40, 20, 20, 
              6, 6, 6, 6, 6, 
              10, 10, 10, 10, 
              10, 10, 
              5, 5, 5, 5]

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
