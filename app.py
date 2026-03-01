# app.py

import streamlit as st
import plotly.express as px
import pandas as pd
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLI-POTER",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. TÍTULO Y BIENVENIDA ---
st.title("🌊 Sistema de Información Hidroclimática (SIHCLI-POTER)")
st.markdown("**Sistema de Información Hidroclimática Integrada para la Gestión Integral del Agua y la Biodiversidad en el Norte de la Región Andina.**")

# --- 3. PESTAÑAS DE INICIO (NUEVA ESTRUCTURA) ---
tab_pres, tab_modulos, tab_clima, tab_aleph = st.tabs([
    "📘 Presentación del Sistema", 
    "🛠️ Módulos y Capacidades", 
    "🏔️ Climatología Andina", 
    "📖 El Aleph"
])

# --- PESTAÑA 1: PRESENTACIÓN ---
with tab_pres:
    with st.expander("Origen y Visión", expanded=True):
        st.write("""
        **SIHCLI-POTER** nace de la necesidad imperativa de integrar datos, ciencia y tecnología para la toma de decisiones informadas en el territorio. En un contexto de variabilidad climática creciente, la gestión del recurso hídrico y el ordenamiento territorial requieren herramientas que transformen datos dispersos en conocimiento accionable.

        Este sistema no es solo un repositorio de datos; es un **cerebro analítico** diseñado para procesar, modelar y visualizar la complejidad hidrometeorológica de la región Andina. Su arquitectura modular permite desde el monitoreo en tiempo real hasta la proyección de escenarios de cambio climático a largo plazo.
        """)

# --- PESTAÑA 2: MÓDULOS Y CAPACIDADES ---
with tab_modulos:
    # Sección A: Aplicaciones Clave
    with st.expander("🎯 Aplicaciones Clave", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Gestión del Riesgo:** Alertas tempranas y mapas de vulnerabilidad ante eventos extremos (sequías e inundaciones).")
            st.info("**Planeación Territorial (POT):** Insumos técnicos para la zonificación ambiental y la gestión de cuencas.")
        with c2:
            st.success("**Agricultura de Precisión:** Calendarios de siembra basados en pronósticos estacionales y zonas de vida.")
            st.warning("**Investigación:** Base de datos depurada y herramientas estadísticas para estudios académicos.")

    # Sección B: Arquitectura del Sistema
    with st.expander("🏗️ Arquitectura del Sistema: Módulos Especializados", expanded=True):
        st.markdown("""
        **SIHCLI-POTER está estructurado en módulos especializados interconectados:**

        * **🚨 Monitoreo (Tiempo Real):**
            * Tablero de control con las últimas lecturas de estaciones telemétricas.
            * Alertas inmediatas de umbrales críticos.
        
        * **🗺️ Distribución Espacial:**
            * Mapas interactivos para visualizar la red de monitoreo.
            * Análisis de cobertura espacial y densidad de datos.
        
        * **🔮 Pronóstico Climático & ENSO:**
            * Integración directa con el **IRI (Columbia University)** para pronósticos oficiales de El Niño/La Niña.
            * Modelos de predicción local (Prophet, SARIMA) y análisis de probabilidades.
        
        * **📉 Tendencias y Riesgo:**
            * Análisis estadístico de largo plazo (Mann-Kendall) para detectar si llueve más o menos que antes.
            * Mapas de vulnerabilidad hídrica interpolados.
        
        * **🛰️ Satélite y Sesgo:**
            * Comparación de datos de tierra vs. reanálisis satelital (ERA5-Land).
            * Herramientas para corregir y rellenar series históricas.
        
        * **🌱 Zonas de Vida y Cobertura:**
            * Cálculo automático de la clasificación de Holdridge.
            * Análisis de uso del suelo y cobertura vegetal.
        """)

# --- PESTAÑA 3: CLIMATOLOGÍA ---
with tab_clima:
    with st.expander("🏔️ La Complejidad de los Andes", expanded=False):
        st.write("""
        La región Andina presenta uno de los sistemas climáticos más complejos del mundo. La interacción entre la Zona de Convergencia Intertropical (ZCIT), los vientos alisios y la topografía escarpada genera microclimas que cambian en distancias cortas. La región Andina es un mosaico climático de una complejidad fascinante. Aquí, la geografía no es solo un escenario, sino un actor protagonista que esculpe el clima kilómetro a kilómetro.

        **La Verticalidad como Destino:** En los Andes, viajar hacia arriba es como viajar hacia los polos. En pocos kilómetros lineales, pasamos del calor húmedo de los valles interandinos (bosque seco tropical) a la neblina perpetua de los bosques de niebla, y finalmente al gélido silencio de los páramos y las nieves perpetuas. Esta zonificación altitudinal (bien descrita por Holdridge) define la vocación del suelo y la biodiversidad.

        **El Pulso de Dos Océanos:** Colombia es un país anfibio, respirando la humedad que llega tanto del Pacífico (Chocó Biogeográfico) como de la Amazonía. Los vientos alisios chocan contra El Sistema Cordillerano de los Andes, descargando su humedad en las vertientes orientales y creando "remolinos de agua" que dan lugar a una Hidrodiversidad magica representada en los grandes ríos, quebradas, arroyos, caños, cañadas, acuiferos, lagunas, embalses y todo tipo de humedales.

        **La Variabilidad (ENSO):** Este sistema complejo no es estático. Está sometido al latido irregular del Pacífico Ecuatorial:
        * **El Niño (Fase Cálida):** Cuando el océano se calienta, la atmósfera sobre nosotros se estabiliza, las nubes se disipan y la sequía amenaza, trayendo consigo el riesgo de incendios y desabastecimiento.
        * **La Niña (Fase Fría):** Cuando el océano se enfría, los vientos se aceleran y la humedad se condensa con furia, desbordando ríos y saturando laderas.
        
        Entender esta climatología no es solo leer termómetros; es comprender la interacción dinámica entre la montaña, el viento y el océano.
        
        **SIHCLI-POTER** está diseñado específicamente para capturar esta variabilidad, integrando estaciones en tierra con modelos satelitales para llenar los vacíos de información en zonas de alta montaña.
        """)

# --- PESTAÑA 4: EL ALEPH ---
with tab_aleph:
    with st.expander("📖 Fragmento de 'El Aleph' - Jorge Luis Borges (1945)", expanded=True):
        st.markdown("""
        > *"... Todo lenguaje es un alfabeto de símbolos cuyo ejercicio presupone un pasado que los interlocutores comparten; ¿cómo transmitir a los otros el infinito Aleph, que mi temerosa memoria apenas abarca? (...)*
        >
        > *En la parte inferior del escalón, hacia la derecha, vi una pequeña esfera tornasolada, de casi intolerable fulgor. Al principio la creí giratoria; luego comprendí que ese movimiento era una ilusión producida por los vertiginosos espectáculos que encerraba. El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí, sin disminución de tamaño. Cada cosa (la luna del espejo, digamos) era infinitas cosas, porque yo la veía claramente desde todos los puntos del universo.*
        >
        > *Vi el populoso mar, vi el alba y la tarde, vi las muchedumbres de América, vi una plateada telaraña en el centro de una negra pirámide, vi un laberinto roto (era Londres), vi interminables ojos inmediatos escrutándose en mí como en un espejo, vi todos los espejos del planeta y ninguno me reflejó...*
        >
        > *Vi el engranaje del amor y la modificación de la muerte, vi el Aleph, desde todos los puntos, vi en el Aleph la tierra, y en la tierra otra vez el Aleph y en el Aleph la tierra, vi mi cara y mis vísceras, vi tu cara, y sentí vértigo y lloré, porque mis ojos habían visto ese objeto secreto y conjetural, cuyo nombre usurpan los hombres, pero que ningún hombre ha mirado: el inconcebible universo."*
        """)

st.divider()

# --- 4. DATOS DEL GRÁFICO SUNBURST ---
# IDs: 28 Elementos
ids = [
    'SIHCLI-POTER', 
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Isoyetas HD', 'Herramientas',
    'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
    'Escenarios', 'Pronósticos', 'Variabilidad',
    'Modelo Turc', 'Recarga', 'Balance',
    'GBIF', 'Taxonomía', 'Amenazas',
    'Priorización', 'Multicriterio',
    'Calidad', 'Auditoría',
    'Geomorfología', 'Morfometría', 'Drenaje', 'Elevación'
]

# Parents: 28 Elementos
parents = [
    '', 
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
    'Isoyetas HD', 'Isoyetas HD', 'Isoyetas HD',
    'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
    'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
    'Toma de Decisiones', 'Toma de Decisiones',
    'Herramientas', 'Herramientas',
    'SIHCLI-POTER', 'Geomorfología', 'Geomorfología', 'Geomorfología'
]

# Values: 28 Elementos
values = [100, 20, 15, 15, 15, 20, 15, 5, 5, 5, 5, 7, 7, 6, 5, 5, 5, 5, 5, 5, 7, 8, 7, 8, 15, 5, 5, 5]

def create_system_map():
    if not (len(ids) == len(parents) == len(values)):
        st.error(f"Error interno: Ids={len(ids)}, Parents={len(parents)}")
        return None
        
    df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
    
    # IMPORTANTE: Quitamos branchvalues='total' para evitar el error silencioso de Plotly
    # cuando los valores no suman exactamente el 100%
    fig = px.sunburst(
        df, names='ids', parents='parents', values='values', 
        color='parents', color_discrete_sequence=px.colors.qualitative.Pastel1
    )
    fig.update_layout(
        title={'text': "🗺️ Mapa de Navegación", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        margin=dict(t=60, l=0, r=0, b=0), height=550
    )
    return fig

# --- 5. LAYOUT PRINCIPAL (MAPA Y CAJAS LATERALES) ---
c1, c2 = st.columns([1.8, 1.2])

with c1:
    fig_map = create_system_map()
    if fig_map: 
        st.plotly_chart(fig_map, use_container_width=True, key="unique_sunburst_chart")

with c2:
    st.subheader("🛠️ Módulos")
    st.info("Seleccione una página en el menú lateral para comenzar.")
    
    with st.expander("🗺️ Isoyetas HD", expanded=True):
        st.write("""
        **Generador Climático:**
        * ✅ Interpolación RBF.
        * ✅ Pronóstico Lineal.
        """)
        st.caption("✅ Operativo")

    with st.expander("🌦️ Clima e Hidrología"):
        st.write("""
        **Tablero de Control:**
        * ✅ Monitoreo Lluvia/Caudal.
        * ✅ Índices ENSO.
        """)
        st.caption("✅ Operativo")

    with st.expander("🏔️ Geomorfología & Amenazas", expanded=True):
        st.write("""
        **Análisis del Terreno:**
        * ✅ DEM 3D y Drenaje.
        * ✅ Amenazas Torrenciales.
        """)
        st.caption("✅ Operativo")    

    with st.expander("💧 Aguas Subterráneas"):
        st.write("""
        **Hidrogeología:**
        * ✅ Balance Hídrico (Turc).
        """)
        st.caption("✅ Operativo")

    with st.expander("🍃 Biodiversidad"):
        st.write("""
        **Biología:**
        * ✅ Monitor GBIF.
        """)
        st.caption("✅ Operativo")

    with st.expander("🎯 Toma de Decisiones"):
        st.write("""
        **Estrategia:**
        * ✅ Priorización AHP.
        """)
        st.caption("✅ Operativo")

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v3.0 | Un Aleph Hidroclimático: Plataforma de Inteligencia Territorial")



