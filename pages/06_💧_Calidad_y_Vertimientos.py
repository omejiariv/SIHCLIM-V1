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

# MAGIA TEMPORAL: Extrae datos históricos o proyecta al futuro
def obtener_poblacion_dinamica(lugar_sel, nivel_sel, anio_obj, tasa_crec):
    pob_u, pob_r = 0.0, 0.0
    
    if nivel_sel == "Veredal" and not df_veredas.empty:
        df_v = df_veredas[df_veredas['Vereda'] == lugar_sel]
        if not df_v.empty: 
            pob_r_base = df_v['Poblacion_hab'].values[0]
            # Simulamos que el dato veredal es de 2020 para proyectar
            anios_dif = anio_obj - 2020
            pob_r = pob_r_base * ((1 + tasa_crec)**anios_dif) if anios_dif > 0 else pob_r_base
        return 0.0, float(pob_r)

    if not df_mpios.empty and nivel_sel in ["Nacional (Colombia)", "Departamental", "Regional", "Municipal"]:
        max_hist = df_mpios['año'].max()
        anio_base = min(anio_obj, max_hist) # Año a buscar en el histórico
        
        # Filtro jerárquico
        if nivel_sel == "Nacional (Colombia)":
            df_f = df_mpios[df_mpios['año'] == anio_base]
        elif nivel_sel == "Departamental":
            df_f = df_mpios[(df_mpios['depto_nom'] == lugar_sel) & (df_mpios['año'] == anio_base)]
        elif nivel_sel == "Regional":
            df_f = df_mpios[(df_mpios['region'] == lugar_sel) & (df_mpios['año'] == anio_base)]
        elif nivel_sel == "Municipal":
            df_f = df_mpios[(df_mpios['municipio'] == lugar_sel) & (df_mpios['año'] == anio_base)]
            
        if not df_f.empty:
            areas_str = df_f['area_geografica'].astype(str).str.lower()
            pob_u = df_f[areas_str.str.contains('urbano|cabecera', na=False)]['Poblacion'].sum()
            pob_r = df_f[areas_str.str.contains('rural|resto|centro', na=False)]['Poblacion'].sum()
            
        # Si el año pedido es mayor al histórico, proyectamos matemáticamente
        if anio_obj > max_hist:
            factor_crecimiento = (1 + tasa_crec) ** (anio_obj - max_hist)
            pob_u *= factor_crecimiento
            pob_r *= factor_crecimiento
            
    return float(pob_u), float(pob_r)

# ==============================================================================
# 🎛️ PANEL MAESTRO DE VARIABLES (TEMPORALIDAD Y CASCADA)
# ==============================================================================
st.subheader("📍 1. Configuración de la Unidad Territorial y Temporalidad")

# --- CONTROL DE TEMPORALIDAD ---
col_t1, col_t2 = st.columns([1.5, 1])
with col_t1:
    anio_analisis = st.slider("📅 Año de Análisis (Máquina del Tiempo):", min_value=2005, max_value=2050, value=2024, step=1)
with col_t2:
    st.caption("Proyección Matemática (Si superas el censo base):")
    tasa_crecimiento = st.number_input("Tasa de Crecimiento Anual (%):", value=1.50, step=0.1) / 100.0

# --- CASCADA TERRITORIAL ---
nivel_sel = st.selectbox("🎯 Nivel de Análisis Objetivo:", ["Nacional (Colombia)", "Departamental", "Regional", "Municipal", "Veredal"])
lugar_sel = "N/A"

if nivel_sel == "Nacional (Colombia)":
    lugar_sel = "Colombia"
elif nivel_sel == "Departamental" and not df_mpios.empty:
    deptos = sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)])
    lugar_sel = st.selectbox("1. Departamento:", deptos, index=deptos.index("Antioquia") if "Antioquia" in deptos else 0)
elif nivel_sel == "Regional" and not df_mpios.empty:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        deptos = sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)])
        depto_sel = st.selectbox("1. Departamento:", deptos, index=deptos.index("Antioquia") if "Antioquia" in deptos else 0)
    with col_f2:
        df_filtro = df_mpios[df_mpios['depto_nom'] == depto_sel]
        regiones = sorted([str(x) for x in df_filtro['region'].unique() if pd.notna(x)]) if 'region' in df_filtro.columns else []
        lugar_sel = st.selectbox("2. Región:", regiones) if regiones else "N/A"
elif nivel_sel == "Municipal" and not df_mpios.empty:
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        deptos = sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)])
        depto_sel = st.selectbox("1. Departamento:", deptos, index=deptos.index("Antioquia") if "Antioquia" in deptos else 0)
    with col_f2:
        df_filtro1 = df_mpios[df_mpios['depto_nom'] == depto_sel]
        regiones = sorted([str(x) for x in df_filtro1['region'].unique() if pd.notna(x)]) if 'region' in df_filtro1.columns else []
        region_sel = st.selectbox("2. Región (Opcional):", ["Todas"] + regiones)
    with col_f3:
        df_filtro2 = df_filtro1 if region_sel == "Todas" else df_filtro1[df_filtro1['region'] == region_sel]
        mpios = sorted([str(x) for x in df_filtro2['municipio'].unique() if pd.notna(x)])
        lugar_sel = st.selectbox("3. Municipio:", mpios)
elif nivel_sel == "Veredal" and not df_veredas.empty:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mpios_v = sorted([str(x) for x in df_veredas['Municipio'].dropna().unique()])
        mpio_sel = st.selectbox("1. Municipio Anfitrión:", mpios_v)
    with col_f2:
        veredas = sorted([str(x) for x in df_veredas[df_veredas['Municipio'] == mpio_sel]['Vereda'].dropna().unique()])
        lugar_sel = st.selectbox("2. Vereda:", veredas)

# --- PANEL DE POBLACIÓN (URBANO, RURAL Y TOTAL) ---
pob_u_auto, pob_r_auto = obtener_poblacion_dinamica(lugar_sel, nivel_sel, anio_analisis, tasa_crecimiento)

st.info(f"👥 Demografía dinámica para **{lugar_sel}** en el año **{anio_analisis}**:")
col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
with col_p1: 
    pob_urbana = st.number_input("Pob. Urbana (Editable):", min_value=0.0, value=pob_u_auto, step=100.0)
with col_p2: 
    pob_rural = st.number_input("Pob. Rural (Editable):", min_value=0.0, value=pob_r_auto, step=100.0)
with col_p3:
    pob_total = pob_urbana + pob_rural
    st.metric(label="Población Total", value=f"{pob_total:,.0f} Hab.")

st.divider()

# ==============================================================================
# PESTAÑAS
# ==============================================================================
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion = st.tabs([
    "🚰 2. Demanda Hídrica",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios de Mitigación"
])

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA Y EVOLUCIÓN
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header(f"🚰 Demanda Hídrica Sectorial ({anio_analisis})")
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.subheader("Parámetros de Demanda")
        dotacion = st.number_input("Dotación Doméstica (L/hab/día):", value=120.0, step=5.0)
        q_domestico = (pob_total * dotacion) / 86400
        
        st.metric(f"Caudal Doméstico Requerido ({anio_analisis})", f"{q_domestico:.2f} L/s")
        
        q_agricola = st.number_input("Concesiones Agrícolas / Riego (L/s):", value=45.0, step=5.0)
        q_industrial = st.number_input("Concesiones Industriales (L/s):", value=20.0, step=2.0)
        q_total_demanda = q_domestico + q_agricola + q_industrial
        
        df_demanda = pd.DataFrame({
            "Sector": ["Doméstico", "Agrícola", "Industrial"],
            "Caudal (L/s)": [q_domestico, q_agricola, q_industrial]
        })
        fig_pie = px.pie(df_demanda, values='Caudal (L/s)', names='Sector', hole=0.4, title=f"Distribución Actual")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_d2:
        st.subheader("📈 Análisis Evolutivo de la Demanda Doméstica")
        st.caption(f"Proyección desde {anio_analisis} hasta 30 años en el futuro.")
        
        anios_evo = np.arange(anio_analisis, anio_analisis + 31)
        pob_evo = pob_total * (1 + tasa_crecimiento)**(anios_evo - anio_analisis)
        demanda_evo = (pob_evo * dotacion) / 86400
        
        fig_evo_dem = go.Figure()
        fig_evo_dem.add_trace(go.Scatter(x=anios_evo, y=demanda_evo, mode='lines', fill='tozeroy', name='Demanda (L/s)', line=dict(color='#2980b9', width=3)))
        fig_evo_dem.update_layout(title="Evolución de Requerimientos Poblacionales (L/s)", xaxis_title="Año", yaxis_title="Caudal Doméstico (L/s)")
        st.plotly_chart(fig_evo_dem, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE CARGAS Y EVOLUCIÓN
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header(f"Inventario de Presiones y Cargas Contaminantes ({anio_analisis})")
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
    
    dbo_urbana = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) 
    dbo_rural = pob_rural * 0.040 
    dbo_suero = vol_suero * 0.035 
    dbo_cerdos = cerdos_agua * 0.150 
    dbo_agricola = (ha_papa + ha_pastos) * 0.8 
    carga_total_dbo = dbo_urbana + dbo_rural + dbo_suero + dbo_cerdos + dbo_agricola
    
    coef_retorno = 0.85
    q_efluente_lps = (q_domestico * coef_retorno) + (q_industrial * 0.8) + (vol_suero / 86400)
    conc_efluente_mg_l = (carga_total_dbo * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_cargas = pd.DataFrame({"Fuente": ["Pob. Urbana", "Pob. Rural", "Lácteos", "Porcicultura", "Agrícola"], "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_cerdos, dbo_agricola]})
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with col_g2:
        st.subheader("📈 Evolución de Carga Orgánica (Impacto Poblacional)")
        anios_evo = np.arange(anio_analisis, anio_analisis + 31)
        pob_u_evo = pob_urbana * (1 + tasa_crecimiento)**(anios_evo - anio_analisis)
        dbo_evo = (pob_u_evo * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))) + dbo_rural + dbo_suero + dbo_cerdos + dbo_agricola
        
        fig_dbo_evo = go.Figure()
        fig_dbo_evo.add_trace(go.Scatter(x=anios_evo, y=dbo_evo, mode='lines', fill='tozeroy', name='Carga DBO (kg/d)', line=dict(color='#e74c3c', width=3)))
        fig_dbo_evo.update_layout(title="Proyección de Vertimientos (DBO5 Total)", xaxis_title="Año", yaxis_title="Carga Contaminante (kg/día)")
        st.plotly_chart(fig_dbo_evo, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ASIMILACIÓN Y DILUCIÓN
# ------------------------------------------------------------------------------
with tab_dilucion:
    st.header(f"🌊 Modelo de Dilución y Balance de Masas ({anio_analisis})")
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("Datos del Río Receptor")
        q_rio = st.number_input("Caudal del Río aguas arriba (L/s):", value=1500.0, step=100.0)
        c_rio = st.number_input("Concentración DBO aguas arriba (mg/L):", value=2.0, step=0.5)
        
        st.markdown("---")
        st.subheader("Datos del Vertimiento Consolidado")
        st.metric("Caudal del Efluente (Qe)", f"{q_efluente_lps:.1f} L/s")
        st.metric("Concentración DBO (Ce)", f"{conc_efluente_mg_l:,.1f} mg/L")
        
        c_mix = ((q_rio * c_rio) + (q_efluente_lps * conc_efluente_mg_l)) / (q_rio + q_efluente_lps)
        
    with col_a2:
        st.subheader("Impacto Aguas Abajo (Concentración Final)")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = c_mix,
            title = {'text': "DBO5 en el Río tras la mezcla (mg/L)", 'font': {'size': 20}},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max(20, c_mix + 5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'steps': [{'range': [0, 3], 'color': "#2ecc71", 'name': 'Excelente'}, {'range': [3, 5], 'color': "#f1c40f", 'name': 'Aceptable'}, {'range': [5, 10], 'color': "#e67e22", 'name': 'Contaminado'}, {'range': [10, 100], 'color': "#e74c3c", 'name': 'Pésimo'}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5.0}
            }))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if c_mix <= 5: st.success(f"✅ **Asimilación positiva en {anio_analisis}:** El río tiene el caudal suficiente para diluir la carga.")
        else: st.error(f"⚠️ **Alerta de Contaminación en {anio_analisis}:** El vertimiento supera la capacidad de dilución del río.")

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Intervenciones")
    st.info("El simulador de mitgación te permitirá crear diferentes escenarios cruzados en el tiempo. ¡Próxima fase de desarrollo!")
