import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Simulación de demanda sectorial, cargas contaminantes (DBO, SST), 
capacidad de asimilación y dilución en la red hídrica mediante balance de masas.
""")
st.divider()

# ==============================================================================
# 🔌 CONECTOR A BASES DE DATOS (LECTURA INTELIGENTE Y DIFUSA)
# ==============================================================================
def leer_csv_robusto(ruta):
    """Lee el CSV forzando la detección del separador correcto de Excel Colombia"""
    try:
        df = pd.read_csv(ruta, sep=';', low_memory=False)
        if len(df.columns) < 2:
            df = pd.read_csv(ruta, sep=',', low_memory=False)
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

@st.cache_data
def cargar_concesiones():
    ruta = "data/Concesiones_Corantioquia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty:
            # Estandarizar columnas
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
            
            # CAZADOR DE COLUMNAS (Búsqueda difusa)
            col_caudal = next((c for c in df.columns if 'caudal' in c), None)
            col_uso = next((c for c in df.columns if 'uso' in c), None)
            col_mpio = next((c for c in df.columns if 'municipio' in c), None)
            col_vereda = next((c for c in df.columns if 'vereda' in c), None)
            
            if col_caudal and col_uso and col_mpio:
                df = df.dropna(subset=[col_uso, col_mpio, col_caudal]).copy()
                
                # Exorcismo de la coma decimal europea de Excel
                if df[col_caudal].dtype == object:
                    df[col_caudal] = df[col_caudal].astype(str).str.replace(',', '.')
                df['caudal_lps'] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
                
                # Normalización de textos
                df['municipio'] = df[col_mpio].astype(str).str.strip().str.title()
                if col_vereda: df['vereda'] = df[col_vereda].astype(str).str.strip().str.title()
                
                # Agrupación Inteligente de Sectores Sihcli
                def clasificar_uso(u):
                    u = str(u).title().strip()
                    if any(x in u for x in ['Domestico', 'Consumo Humano', 'Abastecimiento', 'Acueducto']): return 'Doméstico'
                    elif any(x in u for x in ['Agricola', 'Pecuario', 'Acuicultura', 'Agroindustrial', 'Riego']): return 'Agrícola/Pecuario'
                    elif any(x in u for x in ['Industrial', 'Mineria', 'Minero']): return 'Industrial'
                    else: return 'Otros'
                    
                df['Sector_Sihcli'] = df[col_uso].apply(clasificar_uso)
                return df
    return pd.DataFrame()

df_mpios = cargar_municipios()
df_veredas = cargar_veredas()
df_concesiones = cargar_concesiones()

# EXTRAER POBLACIÓN BASE
def obtener_poblacion_base(lugar_sel, nivel_sel):
    pob_u, pob_r, anio_base = 0.0, 0.0, 2020
    if nivel_sel == "Veredal" and not df_veredas.empty:
        df_v = df_veredas[df_veredas['Vereda'] == lugar_sel]
        if not df_v.empty: pob_r = df_v['Poblacion_hab'].values[0]
    elif not df_mpios.empty and nivel_sel in ["Nacional (Colombia)", "Departamental", "Regional", "Municipal"]:
        anio_base = df_mpios['año'].max()
        if nivel_sel == "Nacional (Colombia)": df_f = df_mpios[df_mpios['año'] == anio_base]
        elif nivel_sel == "Departamental": df_f = df_mpios[(df_mpios['depto_nom'] == lugar_sel) & (df_mpios['año'] == anio_base)]
        elif nivel_sel == "Regional": df_f = df_mpios[(df_mpios['region'] == lugar_sel) & (df_mpios['año'] == anio_base)]
        elif nivel_sel == "Municipal": df_f = df_mpios[(df_mpios['municipio'] == lugar_sel) & (df_mpios['año'] == anio_base)]
            
        if not df_f.empty:
            areas_str = df_f['area_geografica'].astype(str).str.lower()
            pob_u = df_f[areas_str.str.contains('urbano|cabecera', na=False)]['Poblacion'].sum()
            pob_r = df_f[areas_str.str.contains('rural|resto|centro', na=False)]['Poblacion'].sum()
            
    return float(pob_u), float(pob_r), anio_base

def proyectar_curva(p_base, anios_array, anio_base, modelo, r, k):
    t = np.maximum(0, anios_array - anio_base) 
    if modelo == "Logístico":
        k_val = max(k, p_base * 1.05) 
        return k_val / (1 + ((k_val - p_base) / p_base) * np.exp(-r * t))
    elif modelo == "Exponencial": return p_base * np.exp(r * t)
    elif modelo == "Lineal (Tendencial)": return p_base * (1 + r * t)
    else: return p_base * ((1 + r) ** t)

# ==============================================================================
# 🎛️ PANEL MAESTRO DE VARIABLES
# ==============================================================================
st.subheader("📍 1. Configuración Territorial y Máquina del Tiempo")

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

st.markdown("⚙️ **Parámetros de Proyección Demográfica**")
pob_u_base, pob_r_base, anio_base = obtener_poblacion_base(lugar_sel, nivel_sel)
pob_t_base = pob_u_base + pob_r_base

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1: anio_analisis = st.slider("📅 Año a Simular:", min_value=anio_base, max_value=2060, value=2024, step=1)
with col_t2: modelo_sel = st.selectbox("Ecuación Evolutiva:", ["Logístico", "Geométrico", "Exponencial", "Lineal (Tendencial)"])
with col_t3: tasa_r = st.number_input("Tasa de Crecimiento (r) %:", value=1.50, step=0.1) / 100.0
with col_t4:
    k_man = st.number_input("Capacidad de Carga (K):", value=float(pob_t_base * 2.0), step=1000.0, disabled=(modelo_sel != "Logístico"))

factor_proy = proyectar_curva(pob_t_base, np.array([anio_analisis]), anio_base, modelo_sel, tasa_r, k_man)[0] / pob_t_base if pob_t_base > 0 else 1.0
pob_u_auto = pob_u_base * factor_proy
pob_r_auto = pob_r_base * factor_proy

st.info(f"👥 Demografía dinámica proyectada para **{lugar_sel}** en el año **{anio_analisis}**:")
col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
with col_p1: pob_urbana = st.number_input("Pob. Urbana (Editable):", min_value=0.0, value=pob_u_auto, step=100.0)
with col_p2: pob_rural = st.number_input("Pob. Rural (Editable):", min_value=0.0, value=pob_r_auto, step=100.0)
with col_p3:
    pob_total = pob_urbana + pob_rural
    st.metric(label="Población Total Estimada", value=f"{pob_total:,.0f} Hab.", delta=f"+ {pob_total - pob_t_base:,.0f} desde {anio_base}" if pob_total > pob_t_base else None)

st.divider()

# ==============================================================================
# PESTAÑAS
# ==============================================================================
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion = st.tabs([
    "🚰 2. Demanda y Subregistro",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios de Mitigación"
])

anios_evo = np.arange(anio_analisis, anio_analisis + 31)
factor_evo = proyectar_curva(pob_t_base, anios_evo, anio_base, modelo_sel, tasa_r, k_man) / pob_t_base if pob_t_base > 0 else np.ones_like(anios_evo)
pob_evo = pob_total * (factor_evo / factor_proy)

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA (AHORA CON ANÁLISIS DE SUBREGISTRO SIRENA)
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header(f"🚰 Metabolismo Hídrico y Nivel de Formalización")
    col_d1, col_d2 = st.columns([1, 1.2])
    
    with col_d1:
        st.subheader("1. Demanda Teórica (Requerimiento Real)")
        dotacion = st.number_input("Dotación Doméstica (L/hab/día):", value=120.0, step=5.0)
        q_teorico_dom = (pob_total * dotacion) / 86400
        st.metric(f"Caudal Doméstico Necesario ({anio_analisis})", f"{q_teorico_dom:.2f} L/s")
        
        st.markdown("---")
        st.subheader("2. Demanda Legal (Autorizada por Corantioquia)")
        
        # CRUZAMOS LA BASE DE DATOS DE CORANTIOQUIA
        q_legal_dom, q_legal_agr, q_legal_ind = 0.0, 0.0, 0.0
        
        if not df_concesiones.empty and lugar_sel != "N/A":
            if nivel_sel == "Municipal":
                df_filtro_c = df_concesiones[df_concesiones['municipio'].str.lower() == lugar_sel.lower()]
            elif nivel_sel == "Veredal" and 'vereda' in df_concesiones.columns:
                df_filtro_c = df_concesiones[df_concesiones['vereda'].str.lower() == lugar_sel.lower()]
            else:
                df_filtro_c = df_concesiones
                
            if not df_filtro_c.empty:
                q_legal_dom = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Doméstico']['caudal_lps'].sum()
                q_legal_agr = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Agrícola/Pecuario']['caudal_lps'].sum()
                q_legal_ind = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Industrial']['caudal_lps'].sum()
                
        st.caption(f"Caudales formales extraídos de SIRENA para **{lugar_sel}**:")
        q_concesionado_dom = st.number_input("Caudal Doméstico Autorizado (L/s):", value=float(q_legal_dom), step=1.0)
        q_agricola = st.number_input("Caudal Agrícola/Pecuario (L/s):", value=float(q_legal_agr), step=1.0)
        q_industrial = st.number_input("Caudal Industrial Autorizado (L/s):", value=float(q_legal_ind), step=1.0)
        
    with col_d2:
        st.subheader("📊 Análisis de Ilegalidad o Subregistro")
        st.markdown("Comparativa entre lo que la población consume realmente frente a lo que tienen registrado ante la Corporación.")
        
        df_sub = pd.DataFrame({
            "Naturaleza": ["Demanda Teórica (Real)", "Caudal Autorizado (Corantioquia)"],
            "Caudal (L/s)": [q_teorico_dom, q_concesionado_dom]
        })
        
        brecha = max(0, q_teorico_dom - q_concesionado_dom)
        porc_legal = (q_concesionado_dom / q_teorico_dom * 100) if q_teorico_dom > 0 else 100
        
        fig_sub = px.bar(df_sub, x="Naturaleza", y="Caudal (L/s)", color="Naturaleza",
                         color_discrete_sequence=["#e74c3c", "#2ecc71"],
                         title=f"Brecha Hídrica: {brecha:,.1f} L/s extraídos sin registro formal")
        fig_sub.add_hline(y=q_teorico_dom, line_dash="dash", line_color="black", annotation_text="Límite Real de Consumo")
        st.plotly_chart(fig_sub, use_container_width=True)
        
        st.metric("Índice de Formalización (Sector Doméstico)", f"{min(porc_legal, 100):.1f}%", 
                  delta="- Alerta de Ilegalidad Alta" if porc_legal < 50 else "+ Nivel de Formalización Óptimo")

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE CARGAS
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
    
    # Usamos los datos reales extraídos de la base de concesiones para calcular el caudal efluente
    coef_retorno = 0.85
    q_efluente_lps = (q_teorico_dom * coef_retorno) + (q_industrial * 0.8) + (vol_suero / 86400)
    conc_efluente_mg_l = (carga_total_dbo * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_cargas = pd.DataFrame({"Fuente": ["Pob. Urbana", "Pob. Rural", "Lácteos", "Porcicultura", "Agrícola"], "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_cerdos, dbo_agricola]})
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with col_g2:
        st.subheader(f"📈 Evolución de Carga Orgánica ({modelo_sel})")
        pob_u_evo = pob_urbana * (factor_evo / factor_proy)
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
        st.metric("Caudal del Efluente (Qe)", f"{q_efluente_lps:,.1f} L/s")
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
