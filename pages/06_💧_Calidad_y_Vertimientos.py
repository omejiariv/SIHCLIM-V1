# pages/06_💧_Calidad_y_Vertimientos.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página (Streamlit permite configurar cada página individualmente)
st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

# ==============================================================================
# ENCABEZADO Y CONTEXTO
# ==============================================================================
st.title("💧 Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo de simulación de cargas contaminantes (DBO, SST, Nutrientes), capacidad de asimilación 
y dilución en la red hídrica. Integra descargas puntuales (urbanas, industriales) y difusas (agrícolas).
""")
st.divider()

# ==============================================================================
# ESTRUCTURA DE PESTAÑAS
# ==============================================================================
tab_fuentes, tab_dilucion, tab_mitigacion = st.tabs([
    "🏭 Inventario de Cargas", 
    "🌊 Asimilación y Dilución", 
    "🛡️ Escenarios de Mitigación (PTAR/BPA)"
])

# ------------------------------------------------------------------------------
# TAB 1: INVENTARIO DE FUENTES CONTAMINANTES
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header("Inventario Territorial de Fuentes Contaminantes")
    st.info("Configura las actividades humanas, industriales y agropecuarias presentes en la subcuenca de análisis.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏘️ Población y Urbanismo")
        st.caption("Aportes de DBO, SST y Coliformes.")
        pob_urbana = st.number_input("Población Urbana (Con alcantarillado):", min_value=0, value=15000, step=1000)
        pob_rural = st.number_input("Población Rural (Sistemas in situ/Directo):", min_value=0, value=5000, step=500)
        cobertura_ptar = st.slider("Cobertura actual de Tratamiento (PTAR) %:", 0, 100, 15)

    with col2:
        st.subheader("🐄 Agroindustria (Ej. Norte)")
        st.caption("Sueros lácteos, lavado de porquerizas (Alta DBO/DQO).")
        vol_suero = st.number_input("Descarga de Sueros Lácteos (L/día):", min_value=0, value=2000, step=500)
        cerdos_agua = st.number_input("Porcinos (Cabezas en confinamiento):", min_value=0, value=1500, step=100)
        vacas_ordeno = st.number_input("Vacas en Ordeño (Lavado de salas):", min_value=0, value=300, step=50)

    with col3:
        st.subheader("🍓 Agricultura (Ej. Oriente)")
        st.caption("Cargas difusas: Agroquímicos, Fertilizantes (N, P).")
        ha_papa = st.number_input("Cultivos Limpios (Papa, Hortalizas) [Ha]:", min_value=0.0, value=50.0, step=5.0)
        ha_frutales = st.number_input("Frutales (Mora, Fresa, Tomate) [Ha]:", min_value=0.0, value=80.0, step=5.0)
        ha_pastos_fert = st.number_input("Pastos Fertilizados [Ha]:", min_value=0.0, value=200.0, step=10.0)

    st.markdown("---")
    
    # --- GRÁFICO PRELIMINAR (Demostrativo de la UI) ---
    st.subheader("📊 Estimación Preliminar de Cargas Orgánicas (DBO5)")
    st.caption("*Nota: Gráfico demostrativo. El motor matemático se conectará en el próximo paso.*")
    
    # Datos simulados reactivos a la UI para dar la sensación de vida
    dbo_pob = (pob_urbana * (1 - cobertura_ptar/100) + pob_rural) * 0.04  # 40g DBO/hab/día aprox
    dbo_suero = vol_suero * 0.035 # 35,000 mg/L = 35g/L aprox
    dbo_cerdos = cerdos_agua * 0.15 # 150g DBO/cerdo/día aprox
    dbo_agricola = (ha_papa + ha_frutales) * 1.2 # Escorrentía base simulada
    
    df_cargas = pd.DataFrame({
        "Fuente": ["Población Urbana/Rural", "Industria Láctea (Sueros)", "Porcicultura", "Escorrentía Agrícola"],
        "DBO_kg_dia": [dbo_pob, dbo_suero, dbo_cerdos, dbo_agricola]
    })
    
    fig_cargas = px.bar(
        df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', 
        title="Aporte Diario Estimado de Materia Orgánica (kg DBO5/día)",
        color="Fuente", color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_cargas, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: ASIMILACIÓN Y DILUCIÓN (Próxima fase)
# ------------------------------------------------------------------------------
with tab_dilucion:
    st.header("🌊 Modelo de Dilución en Río")
    st.info("Aquí cruzaremos las cargas del Tab 1 con los datos del Módulo de Hidrología (Caudales Q95, Q70) para modelar la concentración de contaminantes en el cauce.")
    st.image("https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f?auto=format&fit=crop&w=1200&q=80", caption="El módulo hídrico calculará la capacidad de autodepuración del río.", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ESCENARIOS DE MITIGACIÓN (Próxima fase)
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Intervenciones (CuencaVerde)")
    st.info("¿Qué pasa si instalamos pozos sépticos? ¿Si implementamos Buenas Prácticas Agrícolas (BPA)? ¿Si construimos una PTAR? Aquí simularemos la reducción de las curvas de contaminación.")
