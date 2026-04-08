# app.py

import os
import sys
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

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo
except ImportError:
    # Fallback de rutas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo

# ==========================================
# 📂 MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible para mantener coherencia en todo el sistema
selectors.renderizar_menu_navegacion("Inicio")

# ==============================================================================
# 💉 INYECTAR TORRENTE SANGUÍNEO AL INICIO (Sistema Inmunológico)
# ==============================================================================
try:
    inicializar_torrente_sanguineo()
except Exception as e:
    pass # Fallback silencioso para no romper la UI en caso de error de importación

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
                # 💡 La contraseña por defecto es "Aleph2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "Aleph2026"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo
muro_de_acceso_beta()
# ==============================================================================

# --- ESTILOS CSS PERSONALIZADOS (MEJORADOS) ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3498db;
        border-top: 1px solid #e0e6ed;
        border-right: 1px solid #e0e6ed;
        border-bottom: 1px solid #e0e6ed;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 25px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ENCABEZADO Y KPIS ---
st.markdown('<p class="main-header">🌊 SIHCLI-POTER</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Información Hidroclimática Integrada | Gemelo Digital Territorial</p>', unsafe_allow_html=True)

# Panel de Métricas Globales (Dashboard)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Módulos Analíticos", "15 Especializados", "Operativos") # Actualizado a 15 con el satélite
col2.metric("Resolución Temporal", "1950 - 2070", "Datos Históricos + Proyecciones")
col3.metric("Cobertura Geográfica", "Región Andina", "Topología de Cuencas")
col4.metric("Motores de Decisión", "WRI / AHP / Earth Engine", "Estándares Globales")

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
        
    c4, c5, c6 = st.columns(3)
    with c4:
        st.page_link("pages/04_🍃_Biodiversidad.py", label="**Biodiversidad**", icon="🍃")
        st.markdown("<small>Monitor de especies (GBIF), endemismos y valoración económica de servicios ecosistémicos.</small>", unsafe_allow_html=True)
    with c5:
        st.page_link("pages/05_🏔️_Geomorfologia.py", label="**Geomorfología**", icon="🏔️")
        st.markdown("<small>Análisis de Modelos Digitales de Elevación (DEM), redes de drenaje y morfometría.</small>", unsafe_allow_html=True)
    with c6:
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="**Satélite en Vivo**", icon="🛰️")
        st.markdown("<small>Conexión directa a Google Earth Engine para monitoreo de coberturas Dynamic World a 10m.</small>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)
    
    # --- EJE 2 ---
    st.markdown("### ⚙️ EJE 2: Metabolismo Territorial (Presiones Antrópicas)")
    st.caption("Módulos enfocados en la dinámica poblacional y el impacto sobre la red hídrica.")
    
    c7, c8 = st.columns(2)
    with c7:
        st.page_link("pages/06_📈_Modelo_Demografico.py", label="**Modelo Demográfico (Humanos)**", icon="📈")
        st.markdown("<small>Proyecciones poblacionales (DANE) multimodelo con inyección a la Memoria Global.</small>", unsafe_allow_html=True)
        st.write("") # Espaciador
        
        st.page_link("pages/06_🐄_Modelo_Pecuario.py", label="**Modelo Pecuario (Animales)**", icon="🐄")
        st.markdown("<small>Proyecciones de crecimiento (ICA) para Bovinos, Porcinos y Aves en escalas hidrográficas.</small>", unsafe_allow_html=True)
        
    with c8:
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
    
    c11, c12, c13 = st.columns(3)
    with c11:
        st.page_link("pages/11_⚙️_Generador.py", label="**Generador**", icon="⚙️")
        st.markdown("<small>Motor de geoprocesamiento.</small>", unsafe_allow_html=True)
    with c12:
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="**Ayuda y Docs**", icon="📚")
        st.markdown("<small>Manuales y documentación técnica.</small>", unsafe_allow_html=True)
    with c13:
        st.page_link("pages/13_🕵️_Detective.py", label="**Detective Espacial**", icon="🕵️")
        st.markdown("<small>Auditoría BD y diagnóstico de proyecciones (CRS).</small>", unsafe_allow_html=True)


# =====================================================================
# PESTAÑA 2: ARQUITECTURA DEL SISTEMA (SUNBURST)
# =====================================================================
with tab_arquitectura:
    st.markdown("### Mapa Topológico del Sistema")
    st.info("Visualización jerárquica de la arquitectura de la plataforma y sus submódulos lógicos.")
    
    ids = ['SIHCLI-POTER', 'Soporte Biofísico', 'Metabolismo Territorial', 'Síntesis Estratégica', 'Herramientas', 
           'Clima e Hidrología', 'Aguas Subterráneas', 'Isoyetas HD', 'Biodiversidad', 'Geomorfología', 'Satélite en Vivo',
           'Modelo Demográfico', 'Modelo Pecuario', 'Calidad y Vertimientos', 'Sistemas Hídricos', 
           'Toma de Decisiones', 'Panel Administración',
           'Generador', 'Ayuda y Docs', 'Detective']
            
    parents = ['', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
               'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico', 'Soporte Biofísico',
               'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial',
               'Síntesis Estratégica', 'Síntesis Estratégica',
               'Herramientas', 'Herramientas', 'Herramientas']
                
    values = [100, 35, 35, 20, 10, 
              5.8, 5.8, 5.8, 5.8, 5.8, 6.0, 
              8.75, 8.75, 8.75, 8.75, 
              10, 10, 
              3.3, 3.3, 3.4]

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
