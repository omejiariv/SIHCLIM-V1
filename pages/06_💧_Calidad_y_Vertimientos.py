import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Simulación de demanda sectorial, cargas contaminantes (DBO, SST, Nutrientes), 
capacidad de asimilación y dilución en la red hídrica. Integra descargas puntuales y difusas.
""")
st.divider()

# ==============================================================================
# 🔌 CONECTOR A LA BASE DE DATOS DEMOGRÁFICA (PÁGINA 07)
# ==============================================================================
def leer_csv_robusto(ruta):
    try:
        df = pd.read_csv(ruta, sep=None, engine='python')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty and 'municipio' in df.columns:
            df.dropna(subset=['municipio'], inplace=True)
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_veredas():
    ruta = "data/veredas_Antioquia.xlsx"
    return pd.read_excel(ruta) if os.path.exists(ruta) else pd.DataFrame()

df_mpios = cargar_municipios()
df_veredas = cargar_veredas()

# Función para extraer población actual
def obtener_poblacion_actual(lugar_sel, nivel_sel):
    pob_urbana, pob_rural = 0, 0
    if nivel_sel == "Municipal" and not df_mpios.empty:
        df_f = df_mpios[(df_mpios['municipio'] == lugar_sel) & (df_mpios['año'] == df_mpios['año'].max())]
        if not df_f.empty:
            areas_str = df_f['area_geografica'].astype(str).str.lower()
            pob_urbana = df_f[areas_str.str.contains('urbano|cabecera', na=False)]['Poblacion'].sum()
            pob_rural = df_f[areas_str.str.contains('rural|resto|centro', na=False)]['Poblacion'].sum()
    elif nivel_sel == "Veredal" and not df_veredas.empty:
        df_v = df_veredas[df_veredas['Vereda'] == lugar_sel]
        if not df_v.empty:
            pob_rural = df_v['Poblacion_hab'].values[0]
    return float(pob_urbana), float(pob_rural)

# ==============================================================================
# ESTRUCTURA DE PESTAÑAS
# ==============================================================================
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion = st.tabs([
    "🚰 Demanda Hídrica",
    "🏭 Inventario de Cargas", 
    "🌊 Asimilación y Dilución", 
    "🛡️ Escenarios de Mitigación"
])

# ------------------------------------------------------------------------------
# TAB 0: DEMANDA HÍDRICA
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header("🚰 Demanda Hídrica Sectorial")
    st.info("Módulo en construcción: Aquí cruzaremos las proyecciones demográficas con las concesiones agrícolas e industriales.")

# ------------------------------------------------------------------------------
# TAB 1: INVENTARIO DE FUENTES CONTAMINANTES
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header("Inventario Territorial de Fuentes Contaminantes")
    st.markdown("Cálculo automático de cargas basándose en la demografía real extraída del DANE y Veredas.")

    # --- SELECTOR DE TERRITORIO (CONECTADO A DEMOGRAFÍA) ---
    st.subheader("📍 1. Selección de la Unidad de Análisis")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        nivel_sel = st.selectbox("Nivel Territorial a evaluar:", ["Municipal", "Veredal"])
    with col_s2:
        lugar_sel = "N/A"
        if nivel_sel == "Municipal" and not df_mpios.empty:
            opciones = sorted([str(x) for x in df_mpios['municipio'].unique() if pd.notna(x)])
            idx = opciones.index('Rionegro') if 'Rionegro' in opciones else 0
            lugar_sel = st.selectbox("Municipio:", opciones, index=idx)
        elif nivel_sel == "Veredal" and not df_veredas.empty:
            opciones = sorted([str(x) for x in df_veredas['Vereda'].dropna().unique()])
            lugar_sel = st.selectbox("Vereda:", opciones)
            
    # Extraemos la población automáticamente
    pob_u_auto, pob_r_auto = obtener_poblacion_actual(lugar_sel, nivel_sel)
    
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏘️ Población y Urbanismo")
        st.caption("Conectado a la Base Maestra Demográfica.")
        # Mostramos los datos extraídos pero permitimos editarlos manualmente si el usuario quiere simular
        pob_urbana = st.number_input("Pob. Urbana (Con alcantarillado):", min_value=0.0, value=pob_u_auto, step=100.0)
        pob_rural = st.number_input("Pob. Rural (Sistemas in situ/Directo):", min_value=0.0, value=pob_r_auto, step=100.0)
        cobertura_ptar = st.slider("Cobertura de Tratamiento (PTAR) %:", 0, 100, 15)

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
    
    # --- GRÁFICOS RESULTANTES ---
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("📊 Aportes de Materia Orgánica (DBO5)")
        # Ecuaciones reales de ingeniería sanitaria (RAS-2017)
        dbo_urbana = pob_urbana * 0.050 * (1 - cobertura_ptar/100) # 50g DBO/hab/día (Aporte neto al río)
        dbo_rural = pob_rural * 0.040 # 40g DBO/hab/día sin tratamiento
        dbo_suero = vol_suero * 0.035 # 35g/L
        dbo_cerdos = cerdos_agua * 0.150 # 150g DBO/cerdo/día
        dbo_agricola = (ha_papa + ha_frutales) * 1.2 # Estimación de escorrentía
        
        df_cargas = pd.DataFrame({
            "Fuente": ["Pob. Urbana (PTAR)", "Pob. Rural (Difusa)", "Lácteos (Sueros)", "Porcicultura", "Agrícola"],
            "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_cerdos, dbo_agricola]
        })
        
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', 
                            title=f"Cargas Contaminantes en {lugar_sel} (kg/día)",
                            color="Fuente", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with col_g2:
        st.subheader("📈 Proyección de Caudal Residual (Saturación PTAR)")
        st.caption("Proyección basada en un crecimiento poblacional del 1.5% anual.")
        # Ecuación Q = (P * Dot * Retorno) / 86400
        anios_proy = np.arange(2024, 2051)
        pob_proyectada = pob_urbana * (1 + 0.015)**(anios_proy - 2024)
        dotacion_l_hab_dia = 120 
        coef_retorno = 0.85
        
        caudal_lps = (pob_proyectada * dotacion_l_hab_dia * coef_retorno) / 86400
        
        fig_caudal = go.Figure()
        fig_caudal.add_trace(go.Scatter(x=anios_proy, y=caudal_lps, mode='lines', fill='tozeroy', 
                                        name='Caudal Afluente (L/s)', line=dict(color='#3498db', width=3)))
        
        # Línea de saturación hipotética
        capacidad_actual = caudal_lps[0] * 1.2 # Asumimos que la PTAR actual está al 80% de capacidad
        fig_caudal.add_hline(y=capacidad_actual, line_dash="dash", line_color="red", 
                             annotation_text="Capacidad Máx PTAR Actual", annotation_position="top left")
        
        fig_caudal.update_layout(title=f"Evolución del Caudal de Aguas Residuales", xaxis_title="Año", yaxis_title="Caudal (L/s)")
        st.plotly_chart(fig_caudal, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: ASIMILACIÓN Y DILUCIÓN
# ------------------------------------------------------------------------------
with tab_dilucion:
    st.header("🌊 Modelo de Dilución en Río")
    st.info("Aquí cruzaremos las cargas del Tab 1 con los datos del Módulo de Hidrología (Caudales Q95, Q70) para modelar la concentración de contaminantes en el cauce.")

# ------------------------------------------------------------------------------
# TAB 3: ESCENARIOS DE MITIGACIÓN
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Intervenciones (CuencaVerde)")
    st.info("¿Qué pasa si instalamos pozos sépticos? ¿Si implementamos Buenas Prácticas Agrícolas (BPA)? ¿Si construimos una PTAR? Aquí simularemos la reducción de las curvas de contaminación.")
