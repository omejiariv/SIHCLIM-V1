import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Simulación de demanda sectorial, cargas contaminantes (DBO, SST), 
capacidad de asimilación y dilución en la red hídrica mediante balance de masas.
""")
st.divider()

# ==============================================================================
# 🔌 CONECTOR A LA BASE DE DATOS DEMOGRÁFICA
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

def obtener_poblacion_actual(lugar_sel, nivel_sel):
    pob_u, pob_r = 0, 0
    if nivel_sel == "Municipal" and not df_mpios.empty:
        df_f = df_mpios[(df_mpios['municipio'] == lugar_sel) & (df_mpios['año'] == df_mpios['año'].max())]
        if not df_f.empty:
            areas_str = df_f['area_geografica'].astype(str).str.lower()
            pob_u = df_f[areas_str.str.contains('urbano|cabecera', na=False)]['Poblacion'].sum()
            pob_r = df_f[areas_str.str.contains('rural|resto|centro', na=False)]['Poblacion'].sum()
    elif nivel_sel == "Veredal" and not df_veredas.empty:
        df_v = df_veredas[df_veredas['Vereda'] == lugar_sel]
        if not df_v.empty: pob_r = df_v['Poblacion_hab'].values[0]
    return float(pob_u), float(pob_r)

# ==============================================================================
# 🎛️ PANEL MAESTRO DE VARIABLES (Aplica a todas las pestañas)
# ==============================================================================
st.subheader("📍 1. Configuración de la Unidad Territorial")
col_m1, col_m2, col_m3 = st.columns([1, 1, 2])

with col_m1:
    nivel_sel = st.selectbox("Nivel Territorial:", ["Municipal", "Veredal"])
with col_m2:
    lugar_sel = "N/A"
    if nivel_sel == "Municipal" and not df_mpios.empty:
        opciones = sorted([str(x) for x in df_mpios['municipio'].unique() if pd.notna(x)])
        idx = opciones.index('Rionegro') if 'Rionegro' in opciones else 0
        lugar_sel = st.selectbox("Unidad:", opciones, index=idx)
    elif nivel_sel == "Veredal" and not df_veredas.empty:
        opciones = sorted([str(x) for x in df_veredas['Vereda'].dropna().unique()])
        lugar_sel = st.selectbox("Unidad:", opciones)

pob_u_auto, pob_r_auto = obtener_poblacion_actual(lugar_sel, nivel_sel)

with col_m3:
    st.caption("Población base extraída automáticamente (Editable para simulaciones):")
    col_p1, col_p2 = st.columns(2)
    with col_p1: pob_urbana = st.number_input("Pob. Urbana:", min_value=0.0, value=pob_u_auto, step=100.0)
    with col_p2: pob_rural = st.number_input("Pob. Rural:", min_value=0.0, value=pob_r_auto, step=100.0)

st.divider()

# ==============================================================================
# ESTRUCTURA DE PESTAÑAS
# ==============================================================================
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion = st.tabs([
    "🚰 2. Demanda Hídrica",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios (Próximamente)"
])

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA (NUEVO)
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header("🚰 Demanda Hídrica Sectorial")
    st.markdown("Estimación del caudal requerido por los diferentes sectores de la unidad territorial.")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.subheader("Parámetros de Demanda")
        dotacion = st.number_input("Dotación Doméstica (L/hab/día):", value=120.0, step=5.0)
        q_domestico = ((pob_urbana + pob_rural) * dotacion) / 86400  # Convertimos a L/s
        
        st.metric("Caudal Doméstico Requerido", f"{q_domestico:.2f} L/s")
        
        q_agricola = st.number_input("Concesiones Agrícolas / Riego (L/s):", value=45.0, step=5.0)
        q_industrial = st.number_input("Concesiones Industriales (L/s):", value=20.0, step=2.0)
        
        q_total_demanda = q_domestico + q_agricola + q_industrial
        
    with col_d2:
        df_demanda = pd.DataFrame({
            "Sector": ["Doméstico (Poblacional)", "Agrícola", "Industrial"],
            "Caudal (L/s)": [q_domestico, q_agricola, q_industrial]
        })
        fig_pie = px.pie(df_demanda, values='Caudal (L/s)', names='Sector', title=f"Distribución de la Demanda Hídrica ({q_total_demanda:.1f} L/s)", hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE FUENTES CONTAMINANTES (CALCULOS MAESTROS)
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header("Inventario de Presiones y Cargas Contaminantes")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏘️ Saneamiento Básico")
        cobertura_ptar = st.slider("Cobertura de Tratamiento (PTAR) %:", 0, 100, 15)
        eficiencia_ptar = st.slider("Remoción DBO en PTAR %:", 0, 100, 80)

    with col2:
        st.subheader("🐄 Agroindustria")
        vol_suero = st.number_input("Sueros Lácteos (L/día):", min_value=0, value=2000, step=500)
        cerdos_agua = st.number_input("Porcinos (Cabezas):", min_value=0, value=1500, step=100)

    with col3:
        st.subheader("🍓 Agricultura")
        ha_papa = st.number_input("Cultivos Limpios [Ha]:", min_value=0.0, value=50.0, step=5.0)
        ha_pastos = st.number_input("Pastos Fertilizados [Ha]:", min_value=0.0, value=200.0, step=10.0)

    st.markdown("---")
    
    # CÁLCULO DE CARGAS DBO5 (Aporte neto al cauce)
    dbo_urbana = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) 
    dbo_rural = pob_rural * 0.040 # Carga difusa / In situ
    dbo_suero = vol_suero * 0.035 
    dbo_cerdos = cerdos_agua * 0.150 
    dbo_agricola = (ha_papa + ha_pastos) * 0.8 
    
    carga_total_dbo = dbo_urbana + dbo_rural + dbo_suero + dbo_cerdos + dbo_agricola
    
    # CÁLCULO DE CAUDAL RESIDUAL (Qe)
    coef_retorno = 0.85
    # Asumimos que la industria y cerdos aportan al caudal residual. (Suero ya está en L/dia).
    q_efluente_lps = (q_domestico * coef_retorno) + (q_industrial * 0.8) + (vol_suero / 86400)
    
    # CONCENTRACIÓN DEL EFLUENTE (Ce) en mg/L -> (kg/dia * 1e6) / (L/s * 86400)
    conc_efluente_mg_l = (carga_total_dbo * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_cargas = pd.DataFrame({
            "Fuente": ["Pob. Urbana", "Pob. Rural", "Lácteos", "Porcicultura", "Agrícola"],
            "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_cerdos, dbo_agricola]
        })
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo:.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with col_g2:
        anios_proy = np.arange(2024, 2051)
        pob_proyectada = pob_urbana * (1 + 0.015)**(anios_proy - 2024)
        caudal_futuro_lps = (pob_proyectada * dotacion * coef_retorno) / 86400
        
        fig_caudal = go.Figure()
        fig_caudal.add_trace(go.Scatter(x=anios_proy, y=caudal_futuro_lps, mode='lines', fill='tozeroy', name='Caudal (L/s)', line=dict(color='#e74c3c', width=3)))
        fig_caudal.update_layout(title=f"Saturación de Redes (Crecimiento Poblacional)", xaxis_title="Año", yaxis_title="Caudal Residual (L/s)")
        st.plotly_chart(fig_caudal, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ASIMILACIÓN Y DILUCIÓN (NUEVO)
# ------------------------------------------------------------------------------
with tab_dilucion:
    st.header("🌊 Modelo de Dilución y Balance de Masas")
    st.markdown("Evalúa la capacidad del cuerpo receptor para asimilar las cargas contaminantes calculadas en el paso anterior.")
    
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("Datos del Río Receptor")
        st.info("Estos datos se conectarán automáticamente con el Módulo de Hidrología en el futuro.")
        q_rio = st.number_input("Caudal del Río aguas arriba (L/s):", value=1500.0, step=100.0)
        c_rio = st.number_input("Concentración DBO aguas arriba (mg/L):", value=2.0, step=0.5)
        
        st.markdown("---")
        st.subheader("Datos del Vertimiento (Automático)")
        st.metric("Caudal del Efluente (Qe)", f"{q_efluente_lps:.1f} L/s")
        st.metric("Concentración DBO (Ce)", f"{conc_efluente_mg_l:.1f} mg/L")
        
        # ECUACIÓN DE BALANCE DE MASAS
        # C_mix = (Qr*Cr + Qe*Ce) / (Qr + Qe)
        c_mix = ((q_rio * c_rio) + (q_efluente_lps * conc_efluente_mg_l)) / (q_rio + q_efluente_lps)
        
    with col_a2:
        st.subheader("Impacto Aguas Abajo (Concentración Final)")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = c_mix,
            title = {'text': "DBO5 en el Río tras la mezcla (mg/L)", 'font': {'size': 20}},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max(20, c_mix + 5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 3], 'color': "#2ecc71", 'name': 'Excelente'},
                    {'range': [3, 5], 'color': "#f1c40f", 'name': 'Aceptable'},
                    {'range': [5, 10], 'color': "#e67e22", 'name': 'Contaminado'},
                    {'range': [10, 100], 'color': "#e74c3c", 'name': 'Pésimo'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5.0} # Límite normativo hipotético
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if c_mix <= 5: st.success("✅ **Capacidad de asimilación positiva:** El río tiene el caudal suficiente para diluir la carga sin superar el límite de 5 mg/L.")
        else: st.error("⚠️ **Alerta de Contaminación:** El vertimiento supera la capacidad de dilución del río. Se requiere aumentar la cobertura de la PTAR o reducir aportes difusos.")

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Intervenciones")
    st.info("Próximamente: Regresa a la pestaña 2, modifica la 'Cobertura de PTAR' al 90% y observa cómo el velocímetro en la pestaña de Dilución pasa de Rojo a Verde. ¡Esa es la potencia de este simulador!")
