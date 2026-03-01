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

        Este sistema no es solo un repositorio de datos; es un **cerebro analítico** diseñado para procesar, modelar y visualizar la complejidad hidrometeorológica de la región Andina. Su arquitectura modular permite desde el monitoreo en tiempo real hasta la proyección de escenarios de cambio climático y metabolismo territorial a largo plazo.
        """)

# --- PESTAÑA 2: MÓDULOS Y CAPACIDADES ---
with tab_modulos:
    # Sección A: Aplicaciones Clave
    with st.expander("🎯 Aplicaciones Clave", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Metabolismo Territorial:** Modelación de topología de redes, balances de masa y Estándares de Seguridad Hídrica (WRI) en tiempo real.")
            st.info("**Gestión del Riesgo y Calidad:** Alertas tempranas, mapas de vulnerabilidad torrencial y análisis espacial de presiones contaminantes.")
        with c2:
            st.success("**Planeación y Demografía:** Cruce de proyecciones poblacionales con oferta hídrica para el ordenamiento territorial (POT).")
            st.warning("**Investigación Climática:** Pronósticos ENSO, isoyetas HD e interpolación espacial para estudios académicos.")

    # Sección B: Arquitectura del Sistema
    with st.expander("🏗️ Arquitectura del Sistema: Módulos Especializados", expanded=True):
        st.markdown("""
        **SIHCLI-POTER está estructurado en módulos especializados interconectados:**

        * **🔗 Sistemas Hídricos Territoriales:**
            * Topología de nodos, trasvases y embalses.
            * Cálculo dinámico de Neutralidad, Resiliencia y Estrés Hídrico.
        
        * **👥 Demografía y Población:**
            * Proyecciones DANE y estructura poblacional (1950-2070).
            * Censos agropecuarios (ICA) y su presión sobre el territorio.
            
        * **💧 Calidad y Vertimientos:**
            * Mapeo de usuarios del recurso, concesiones y permisos de vertimiento.
            * Análisis de carga contaminante en la red de drenaje.

        * **🚨 Clima, Hidrología e Isoyetas HD:**
            * Tablero de control de estaciones, lluvia y caudales.
            * Generador climático (Interpolación RBF y Pronósticos).
            * Integración directa con el **IRI (Columbia University)** para alertas ENSO.
        
        * **🏔️ Geomorfología y Aguas Subterráneas:**
            * DEM 3D, morfometría de cuencas y balance hídrico (Turc).
        
        * **🍃 Biodiversidad y Toma de Decisiones:**
            * Monitor GBIF y Priorización Multicriterio (AHP).
        """)

# --- PESTAÑA 3: CLIMATOLOGÍA ---
with tab_clima:
    with st.expander("🏔️ La Complejidad de los Andes", expanded=False):
        st.write("""
        La región Andina presenta uno de los sistemas climáticos más complejos del mundo. La interacción entre la Zona de Convergencia Intertropical (ZCIT), los vientos alisios y la topografía escarpada genera microclimas que cambian en distancias cortas. La región Andina es un mosaico climático de una complejidad fascinante. Aquí, la geografía no es solo un escenario, sino un actor protagonista que esculpe el clima kilómetro a kilómetro.

        **La Verticalidad como Destino:** En los Andes, viajar hacia arriba es como viajar hacia los polos. En pocos kilómetros lineales, pasamos del calor húmedo de los valles interandinos a la neblina perpetua de los bosques de niebla, y finalmente al silencio de los páramos. 

        **El Pulso de Dos Océanos:** Colombia respira la humedad del Pacífico y la Amazonía. Los vientos alisios chocan contra los Andes, creando una hidrodiversidad mágica de ríos, acuíferos y embalses.

        **La Variabilidad (ENSO):** Este sistema está sometido al latido del Pacífico:
        * **El Niño (Fase Cálida):** Sequía, incendios y desabastecimiento.
        * **La Niña (Fase Fría):** Inundaciones y saturación de laderas.
        
        **SIHCLI-POTER** captura esta variabilidad, integrando datos en tierra, modelos satelitales y metabolismo humano para gestionar la incertidumbre.
        """)

# --- PESTAÑA 4: EL ALEPH ---
with tab_aleph:
    with st.expander("📖 Fragmento de 'El Aleph' - Jorge Luis Borges (1945)", expanded=True):
        st.markdown("""
        > *"... Todo lenguaje es un alfabeto de símbolos cuyo ejercicio presupone un pasado que los interlocutores comparten; ¿cómo transmitir a los otros el infinito Aleph, que mi temerosa memoria apenas abarca? (...)*
        >
        > *En la parte inferior del escalón, hacia la derecha, vi una pequeña esfera tornasolada, de casi intolerable fulgor... El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí, sin disminución de tamaño. Cada cosa era infinitas cosas, porque yo la veía claramente desde todos los puntos del universo.*
        >
        > *Vi el populoso mar, vi el alba y la tarde, vi las muchedumbres de América... vi en el Aleph la tierra, y en la tierra otra vez el Aleph... y sentí vértigo y lloré, porque mis ojos habían visto ese objeto secreto y conjetural, cuyo nombre usurpan los hombres, pero que ningún hombre ha mirado: el inconcebible universo."*
        """)

st.divider()

# --- 4. DATOS DEL GRÁFICO SUNBURST (ACTUALIZADO) ---
# Hemos ampliado y reestructurado el mapa para incluir los nuevos módulos.
ids = [
    'SIHCLI-POTER', 
    'Metabolismo Territorial', 'Dinámica Poblacional', 'Calidad y Saneamiento',
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Geomorfología',
    # Hijos de Metabolismo
    'Embalses y Trasvases', 'Estándares WRI', 'Topología de Redes',
    # Hijos de Poblacional
    'Proyecciones DANE', 'Pirámides Edades', 'Censos ICA',
    # Hijos de Calidad
    'Vertimientos', 'Concesiones', 'Cargas Contaminantes',
    # Hijos Históricos
    'Isoyetas HD', 'Monitoreo Clima', 'Índices ENSO',
    'Balance Turc', 'Acuíferos',
    'Monitor GBIF', 'AHP'
]

parents = [
    '', 
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    
    'Metabolismo Territorial', 'Metabolismo Territorial', 'Metabolismo Territorial',
    'Dinámica Poblacional', 'Dinámica Poblacional', 'Dinámica Poblacional',
    'Calidad y Saneamiento', 'Calidad y Saneamiento', 'Calidad y Saneamiento',
    
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
    'Aguas Subterráneas', 'Aguas Subterráneas',
    'Biodiversidad', 'Toma de Decisiones'
]

values = [
    100, 
    20, 15, 15, 
    15, 10, 10, 10, 5,
    
    7, 7, 6,
    5, 5, 5,
    5, 5, 5,
    
    5, 5, 5,
    5, 5,
    10, 10
]

def create_system_map():
    if not (len(ids) == len(parents) == len(values)):
        st.error(f"Error interno: Ids={len(ids)}, Parents={len(parents)}, Values={len(values)}")
        return None
        
    df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
    
    fig = px.sunburst(
        df, names='ids', parents='parents', values='values', 
        color='parents', color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        title={'text': "🗺️ Arquitectura del Sistema", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
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
    st.subheader("🛠️ Estado de los Módulos")
    st.info("Seleccione una página en el menú lateral para comenzar.")
    
    with st.expander("🔗 Sistemas Hídricos Territoriales", expanded=True):
        st.write("""
        **Metabolismo y Topología:**
        * ✅ Modelación de Embalses (Nube).
        * ✅ Diagramas de Sankey Dinámicos.
        * ✅ Integración Estándares WRI.
        """)
        st.caption("✅ Operativo (Cloud-Native)")
        
    with st.expander("👥 Demografía y Población", expanded=True):
        st.write("""
        **Dinámica Social y Productiva:**
        * ✅ Proyecciones DANE (1950-2070).
        * ✅ Inventarios Pecuarios (ICA).
        """)
        st.caption("✅ Operativo")

    with st.expander("💧 Calidad y Vertimientos", expanded=True):
        st.write("""
        **Presiones sobre el Recurso:**
        * ✅ Red de Saneamiento.
        * ⚙️ Modelación de Cargas (En Desarrollo).
        """)
        st.caption("⚙️ Fase de Integración")

    with st.expander("🌦️ Clima e Isoyetas HD"):
        st.write("""
        **Atmósfera:**
        * ✅ Monitoreo Lluvia/Caudal y ENSO.
        * ✅ Interpolación RBF y Pronósticos.
        """)
        st.caption("✅ Operativo")

    with st.expander("🏔️ Ciencias de la Tierra"):
        st.write("""
        **Soporte Físico:**
        * ✅ DEM 3D, Drenaje y Morfometría.
        * ✅ Aguas Subterráneas (Turc).
        * ✅ Biodiversidad (GBIF) y AHP.
        """)
        st.caption("✅ Operativo") 

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v3.0 | Un Aleph Hidroclimático: Plataforma de Inteligencia Territorial")
