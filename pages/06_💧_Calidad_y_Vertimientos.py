import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Simulación de demanda, cargas contaminantes, 
capacidad de asimilación, formalización y visor espacial de calor (Concesiones y Vertimientos).
""")
st.divider()

# ==============================================================================
# 🧽 FUNCIÓN NORMALIZADORA (MATA-TILDES Y ESPACIOS)
# ==============================================================================
def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

# ==============================================================================
# 🔌 CONECTORES A BASES DE DATOS
# ==============================================================================
def leer_csv_robusto(ruta):
    try:
        df = pd.read_csv(ruta, sep=';', low_memory=False)
        if len(df.columns) < 5: df = pd.read_csv(ruta, sep=',', low_memory=False)
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception: return pd.DataFrame()

@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if 'departamento' in df.columns: df.rename(columns={'departamento': 'depto_nom'}, inplace=True)
        if not df.empty and 'municipio' in df.columns:
            df.dropna(subset=['municipio'], inplace=True)
            df['municipio'] = df['municipio'].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_veredas():
    ruta = "data/veredas_Antioquia.xlsx"
    return pd.read_excel(ruta) if os.path.exists(ruta) else pd.DataFrame()

@st.cache_data
def cargar_concesiones():
    ruta_xlsx = "data/Concesiones_Corantioquia.xlsx"
    ruta_csv = "data/Concesiones_Corantioquia.csv"
    
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
        
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        col_caudal = 'caudal_por_uso' if 'caudal_por_uso' in df.columns else ('caudal_usuario' if 'caudal_usuario' in df.columns else next((c for c in df.columns if 'caudal' in c and 'acumulado' not in c), None))
        col_uso, col_mpio, col_vereda = ('uso' if 'uso' in df.columns else None), ('municipio' if 'municipio' in df.columns else None), ('vereda' if 'vereda' in df.columns else None)
        col_depto, col_region = ('departamento' if 'departamento' in df.columns else None), ('region' if 'region' in df.columns else None)
        col_asunto, col_cota = ('asunto' if 'asunto' in df.columns else None), ('cota' if 'cota' in df.columns else None)
        col_estado, col_car = ('estado' if 'estado' in df.columns else None), ('car' if 'car' in df.columns else None)
        col_x, col_y = ('coordenada_x' if 'coordenada_x' in df.columns else None), ('coordenada_y' if 'coordenada_y' in df.columns else None)

        if col_caudal and col_mpio:
            df = df.dropna(subset=[col_mpio]).copy() 
            if df[col_caudal].dtype == object: df[col_caudal] = df[col_caudal].astype(str).str.replace(',', '.')
            df['caudal_lps'] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
            
            if col_cota: df['cota_num'] = pd.to_numeric(df[col_cota], errors='coerce').fillna(-1)
            else: df['cota_num'] = -1
            
            if col_x: df['coordenada_x'] = pd.to_numeric(df[col_x], errors='coerce').fillna(0)
            if col_y: df['coordenada_y'] = pd.to_numeric(df[col_y], errors='coerce').fillna(0)

            df['municipio'] = df[col_mpio].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            if col_vereda: df['vereda_norm'] = df[col_vereda].apply(normalizar_texto)
            else: df['vereda_norm'] = ""
            if col_depto: df['departamento_norm'] = df[col_depto].apply(normalizar_texto)
            if col_region: df['region_norm'] = df[col_region].apply(normalizar_texto)
            if col_car: df['car_norm'] = df[col_car].astype(str).str.strip().apply(normalizar_texto)
            else: df['car_norm'] = "sin_car"
            
            if col_asunto: df['tipo_agua'] = np.where(df[col_asunto].str.lower().str.contains('subterran|subterrán|pozo|aljibe', regex=True, na=False), 'Subterránea', np.where(df[col_asunto].str.lower().str.contains('superficial|corriente', regex=True, na=False), 'Superficial', 'No Especificado'))
            else: df['tipo_agua'] = 'No Especificado'

            if col_uso: df['uso_detalle'] = df[col_uso].fillna('Sin Información').astype(str).str.title().str.strip()
            else: df['uso_detalle'] = 'Sin Información'

            def clasificar_uso(u):
                u = normalizar_texto(u)
                if any(x in u for x in ['domestico', 'consumo humano', 'abastecimiento', 'acueducto']): return 'Doméstico'
                elif any(x in u for x in ['agricola', 'pecuario', 'acuicultura', 'agroindustrial', 'riego', 'piscicola', 'silvicultura']): return 'Agrícola/Pecuario'
                elif any(x in u for x in ['industrial', 'mineria', 'minero', 'generacion de energia']): return 'Industrial'
                else: return 'Otros'
                
            df['Sector_Sihcli'] = df['uso_detalle'].apply(clasificar_uso)
            if col_estado: df['estado'] = df[col_estado].fillna('Desconocido').astype(str).str.title().str.strip()
            else: df['estado'] = 'Desconocido'
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_vertimientos():
    ruta_xlsx = "data/Vertimientos_Cornare.xlsx"
    ruta_csv = "data/Vertimientos_Cornare.csv"
    
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
        
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        col_caudal = next((c for c in df.columns if 'caudal' in c), None)
        col_mpio = 'municipio' if 'municipio' in df.columns else None
        col_tipo = next((c for c in df.columns if 'tipo' in c and 've' in c), None) 
        col_car = 'car' if 'car' in df.columns else None
        col_x, col_y = ('coordenada_x' if 'coordenada_x' in df.columns else None), ('coordenada_y' if 'coordenada_y' in df.columns else None)
        
        if col_caudal and col_mpio:
            df = df.dropna(subset=[col_mpio]).copy() 
            if df[col_caudal].dtype == object: df[col_caudal] = df[col_caudal].astype(str).str.replace(',', '.')
            df['caudal_vert_lps'] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
            
            if col_x: df['coordenada_x'] = pd.to_numeric(df[col_x], errors='coerce').fillna(0)
            if col_y: df['coordenada_y'] = pd.to_numeric(df[col_y], errors='coerce').fillna(0)

            df['municipio'] = df[col_mpio].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            
            if col_car: df['car_norm'] = df[col_car].astype(str).str.strip().apply(normalizar_texto)
            else: df['car_norm'] = "sin_car"
                
            if col_tipo: df['tipo_vertimiento'] = df[col_tipo].fillna('No Especificado').astype(str).str.title().str.strip()
            else: df['tipo_vertimiento'] = 'No Especificado'
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_censo_bovino():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Bovinos_2023.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Bovinos_2023.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_censo_porcino():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Porcinos_2023.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Porcinos_2023.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_censo_aviar():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Aves_2025.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Aves_2025.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_territorio_maestro():
    ruta_xlsx = "data/territorio_maestro.xlsx"
    ruta_csv = "data/territorio_maestro.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
        
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        if 'municipio' in df.columns:
            df['municipio_norm'] = df['municipio'].astype(str).apply(normalizar_texto)
        if 'car' in df.columns:
            df['car'] = df['car'].astype(str).str.upper()
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.title()
        return df
    return pd.DataFrame()

df_mpios = cargar_municipios()
df_veredas = cargar_veredas()
df_concesiones = cargar_concesiones()
df_vertimientos = cargar_vertimientos()
df_territorio = cargar_territorio_maestro()
df_bovinos = cargar_censo_bovino()
df_porcinos = cargar_censo_porcino()
df_aves = cargar_censo_aviar() # 👈 ¡Nuevo integrante!

# ==============================================================================
# MOTOR MATEMÁTICO POBLACIONAL (MEJORADO CON CRUCE TERRITORIAL)
# ==============================================================================
def obtener_poblacion_base(lugar_sel, nivel_sel):
    pob_u, pob_r, anio_base = 0.0, 0.0, 2020
    if nivel_sel == "Veredal" and not df_veredas.empty:
        df_v = df_veredas[df_veredas['Vereda'] == lugar_sel]
        if not df_v.empty: pob_r = df_v['Poblacion_hab'].values[0]
        
    elif not df_mpios.empty:
        anio_base = df_mpios['año'].max()
        df_f = pd.DataFrame()
        
        if nivel_sel == "Nacional (Colombia)": 
            df_f = df_mpios[df_mpios['año'] == anio_base]
        elif nivel_sel == "Municipal": 
            lugar_n = normalizar_texto(lugar_sel)
            df_f = df_mpios[(df_mpios['municipio_norm'] == lugar_n) & (df_mpios['año'] == anio_base)]
        elif nivel_sel in ["Jurisdicción Ambiental (CAR)", "Regional", "Departamental"]:
            mpios_activos = []
            if not df_territorio.empty:
                if nivel_sel == "Jurisdicción Ambiental (CAR)":
                    car_name = lugar_sel.replace("CAR: ", "")
                    mpios_activos = df_territorio[df_territorio['car'] == car_name]['municipio_norm'].tolist()
                elif nivel_sel == "Regional":
                    mpios_activos = df_territorio[df_territorio['region'] == lugar_sel]['municipio_norm'].tolist()
                elif nivel_sel == "Departamental":
                    mpios_activos = df_territorio[df_territorio['depto_nom'].astype(str).str.title() == lugar_sel]['municipio_norm'].tolist()
            
            # Cruce mágico: Suma solo los municipios que encontró en la tabla maestra
            if mpios_activos:
                df_f = df_mpios[(df_mpios['municipio_norm'].isin(mpios_activos)) & (df_mpios['año'] == anio_base)]
            
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
# 🎛️ PANEL MAESTRO DE VARIABLES (DESPLEGABLE Y DINÁMICO)
# ==============================================================================
with st.expander("📍 1. Configuración Territorial y Máquina del Tiempo", expanded=True):
    nivel_sel_visual = st.selectbox("🎯 Nivel de Análisis Objetivo:", ["Nacional (Colombia)", "Jurisdicción Ambiental (CAR)", "Departamental", "Regional", "Municipal", "Cuenca Hidrográfica", "Veredal"], key="sel_nivel_maestro")
    nivel_sel_interno = nivel_sel_visual

    if nivel_sel_visual == "Nacional (Colombia)": lugar_sel = "Colombia"
    
    elif nivel_sel_visual == "Jurisdicción Ambiental (CAR)":
        if not df_territorio.empty and 'car' in df_territorio.columns:
            # Escudo protector contra celdas vacías en Excel
            cars = sorted([str(x) for x in df_territorio['car'].dropna().unique() if str(x).strip() != ''])
            col_f1, col_f2 = st.columns(2)
            with col_f1: car_sel = st.selectbox("1. Autoridad Ambiental (CAR):", cars, key="sel_car")
            with col_f2:
                mpios_car = sorted([str(x) for x in df_territorio[df_territorio['car'] == car_sel]['municipio'].dropna().unique()])
                sub_sel = st.selectbox("2. Municipio (Opcional):", ["Toda la Jurisdicción"] + mpios_car, key="sel_car_mpio")
            
            if sub_sel == "Toda la Jurisdicción": lugar_sel = f"CAR: {car_sel}"
            else: 
                lugar_sel = sub_sel
                nivel_sel_interno = "Municipal"
        else: st.warning("No se detectó la tabla territorial maestra."); lugar_sel = "N/A"

    elif nivel_sel_visual == "Departamental" and not df_mpios.empty:
        deptos = sorted([str(x) for x in df_mpios['depto_nom'].dropna().unique() if str(x).strip() != ''])
        lugar_sel = st.selectbox("1. Departamento:", deptos, index=deptos.index("Antioquia") if "Antioquia" in deptos else 0, key="sel_depto")
        
    elif nivel_sel_visual == "Regional":
        if not df_territorio.empty and 'region' in df_territorio.columns:
            regiones = sorted([str(x) for x in df_territorio['region'].dropna().unique() if str(x).strip() != ''])
            lugar_sel = st.selectbox("Región (Antioquia):", regiones, key="sel_region")
        else: st.warning("No se detectó la tabla territorial."); lugar_sel = "N/A"
            
    elif nivel_sel_visual == "Municipal":
        if not df_territorio.empty and 'municipio' in df_territorio.columns:
            mpios = sorted([str(x) for x in df_territorio['municipio'].dropna().unique() if str(x).strip() != ''])
            lugar_sel = st.selectbox("Municipio (Antioquia):", mpios, key="sel_mpio")
        else: st.warning("No se detectó la tabla territorial."); lugar_sel = "N/A"

    elif nivel_sel_visual == "Cuenca Hidrográfica":
        st.warning("🚧 Módulo en construcción: Requiere archivo 'cuencas_mpios_proporcion.csv' con el % de área municipal en cada cuenca.")
        lugar_sel = "N/A"    
            
    elif nivel_sel_visual == "Veredal" and not df_veredas.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            mpios_v = sorted([str(x) for x in df_veredas['Municipio'].dropna().unique()])
            mpio_sel = st.selectbox("1. Municipio Anfitrión:", mpios_v, key="sel_ver_mpio")
        with col_f2:
            veredas = sorted([str(x) for x in df_veredas[df_veredas['Municipio'] == mpio_sel]['Vereda'].dropna().unique()])
            lugar_sel = st.selectbox("2. Vereda:", veredas, key="sel_vereda")

    st.markdown("⚙️ **Parámetros de Proyección Demográfica**")
    pob_u_base, pob_r_base, anio_base = obtener_poblacion_base(lugar_sel, nivel_sel_interno)
    pob_t_base = pob_u_base + pob_r_base

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1: anio_analisis = st.slider("📅 Año a Simular:", min_value=anio_base, max_value=2060, value=2024, step=1)
    with col_t2: modelo_sel = st.selectbox("Ecuación Evolutiva:", ["Logístico", "Geométrico", "Exponencial", "Lineal (Tendencial)"])
    with col_t3: tasa_r = st.number_input("Tasa de Crecimiento (r) %:", value=1.50, step=0.1) / 100.0
    with col_t4: k_man = st.number_input("Capacidad de Carga (K):", value=float(max(pob_t_base * 2.0, 1000)), step=1000.0, disabled=(modelo_sel != "Logístico"))

    factor_proy = proyectar_curva(pob_t_base, np.array([anio_analisis]), anio_base, modelo_sel, tasa_r, k_man)[0] / pob_t_base if pob_t_base > 0 else 1.0
    pob_u_auto = pob_u_base * factor_proy
    pob_r_auto = pob_r_base * factor_proy

    st.info(f"👥 Demografía dinámica para **{lugar_sel}** en el año **{anio_analisis}**:")
    col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
    with col_p1: pob_urbana = st.number_input("Pob. Urbana (Editable):", min_value=0.0, value=pob_u_auto, step=100.0)
    with col_p2: pob_rural = st.number_input("Pob. Rural (Editable):", min_value=0.0, value=pob_r_auto, step=100.0)
    with col_p3:
        pob_total = pob_urbana + pob_rural
        st.metric(label="Población Total Estimada", value=f"{pob_total:,.0f} Hab.", delta=f"+ {pob_total - pob_t_base:,.0f} desde {anio_base}" if pob_total > pob_t_base else None)

st.success(f"📌 **SÍNTESIS ACTIVA |** 📍 Territorio: **{lugar_sel} ({nivel_sel_visual})** | 📅 Año: **{anio_analisis}** | 👥 Población: **{pob_total:,.0f} Hab.**")

tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion, tab_mapa, tab_sirena, tab_extern, tab_lactosuero = st.tabs([
    "🚰 2. Demanda y Eficiencia",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios de Mitigación",
    "🗺️ 6. Mapa de Calor (Visor)",
    "📊 7. Explorador Ambiental",
    "⚠️ 8. Externalidades",
    "🥛 9. Economía Circular" # 👈 La nueva pestaña
])

anios_evo = np.arange(anio_analisis, anio_analisis + 31)
factor_evo = proyectar_curva(pob_t_base, anios_evo, anio_base, modelo_sel, tasa_r, k_man) / pob_t_base if pob_t_base > 0 else np.ones_like(anios_evo)
pob_evo = pob_total * (factor_evo / factor_proy)

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA Y EFICIENCIA
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header(f"🚰 Demanda, Eficiencia de Sistemas y Formalización")
    col_d1, col_d2 = st.columns([1, 1.5])
    
    with col_d1:
        st.subheader("1. Demanda Teórica (Neto vs Bruto)")
        
        st.markdown("**A. Uso Doméstico**")
        col_d_dom1, col_d_dom2 = st.columns(2)
        with col_d_dom1: dotacion = st.number_input("Dotación Neta (L/hab/d):", value=120.0, step=5.0)
        with col_d_dom2: perd_dom = st.slider("Pérdidas del Acueducto (%):", 0.0, 100.0, 25.0, step=1.0)
        q_necesario_dom = (pob_total * dotacion) / 86400
        q_efectivo_dom = q_necesario_dom / (1 - (perd_dom/100)) if perd_dom < 100 else q_necesario_dom
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Neto (Necesario)", f"{q_necesario_dom:.2f} L/s")
        col_res2.metric("Bruto (Efectivo)", f"{q_efectivo_dom:.2f} L/s", delta=f"Pérdida: {(q_efectivo_dom - q_necesario_dom):.2f} L/s", delta_color="inverse")
        
        st.markdown("**B. Uso Agrícola / Pecuario**")
        col_d_agr1, col_d_agr2 = st.columns(2)
        with col_d_agr1: q_necesario_agr = st.number_input("Demanda Neta Agrícola (L/s):", value=45.0, step=5.0)
        with col_d_agr2: perd_agr = st.slider("Pérdidas Sist. de Riego (%):", 0.0, 100.0, 30.0, step=1.0)
        q_efectivo_agr = q_necesario_agr / (1 - (perd_agr/100)) if perd_agr < 100 else q_necesario_agr
        st.caption(f"Caudal Bruto Agrícola a captar: **{q_efectivo_agr:.2f} L/s**")
        
        st.markdown("**C. Uso Industrial**")
        col_d_ind1, col_d_ind2 = st.columns(2)
        with col_d_ind1: q_necesario_ind = st.number_input("Demanda Neta Industrial (L/s):", value=20.0, step=2.0)
        with col_d_ind2: perd_ind = st.slider("Pérdidas de Industria (%):", 0.0, 100.0, 10.0, step=1.0)
        q_efectivo_ind = q_necesario_ind / (1 - (perd_ind/100)) if perd_ind < 100 else q_necesario_ind
        st.caption(f"Caudal Bruto Industrial a captar: **{q_efectivo_ind:.2f} L/s**")
        
        st.markdown("---")
        st.subheader("2. Demanda Legal (Autorizada)")
        q_sup, q_sub, q_legal_agr, q_legal_ind = 0.0, 0.0, 0.0, 0.0
        df_usos_detalle = pd.DataFrame()
        
        if not df_concesiones.empty and lugar_sel != "N/A":
            lugar_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            
            if nivel_sel_interno == "Nacional (Colombia)": df_filtro_c = df_concesiones.copy()
            elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)": df_filtro_c = df_concesiones[df_concesiones['car_norm'] == lugar_norm] if 'car_norm' in df_concesiones.columns else pd.DataFrame()
            elif nivel_sel_interno == "Departamental": df_filtro_c = df_concesiones[df_concesiones['departamento_norm'] == lugar_norm] if 'departamento_norm' in df_concesiones.columns else df_concesiones.copy()
            elif nivel_sel_interno == "Regional": df_filtro_c = df_concesiones[df_concesiones['region_norm'] == lugar_norm] if 'region_norm' in df_concesiones.columns else pd.DataFrame()
            elif nivel_sel_interno == "Municipal": df_filtro_c = df_concesiones[df_concesiones['municipio_norm'] == lugar_norm]
            elif nivel_sel_interno == "Veredal" and 'vereda_norm' in df_concesiones.columns: df_filtro_c = df_concesiones[df_concesiones['vereda_norm'] == lugar_norm]
            else: df_filtro_c = pd.DataFrame()
                
            if not df_filtro_c.empty:
                df_dom = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Doméstico']
                q_sup = df_dom[df_dom['tipo_agua'] == 'Superficial']['caudal_lps'].sum()
                q_sub = df_dom[df_dom['tipo_agua'] == 'Subterránea']['caudal_lps'].sum()
                q_legal_agr = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Agrícola/Pecuario']['caudal_lps'].sum()
                q_legal_ind = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Industrial']['caudal_lps'].sum()
                
                df_usos_detalle = df_filtro_c.groupby(['uso_detalle', 'tipo_agua'])['caudal_lps'].sum().reset_index()
                df_usos_detalle.rename(columns={'uso_detalle':'Uso Específico', 'tipo_agua':'Fuente', 'caudal_lps':'Caudal (L/s)'}, inplace=True)
                df_usos_detalle = df_usos_detalle.sort_values(by='Caudal (L/s)', ascending=False)
                
        q_concesionado_dom = q_sup + q_sub
        st.write(f"- **Superficial Doméstico:** {q_sup:,.2f} L/s")
        st.write(f"- **Subterráneo Doméstico:** {q_sub:,.2f} L/s")
        st.write(f"- **Total Legal Doméstico:** {q_concesionado_dom:,.2f} L/s")
        
    with col_d2:
        st.subheader("📊 Análisis de Formalización (Uso Doméstico)")
        margen = 0.05 
        if q_concesionado_dom > q_efectivo_dom * (1 + margen): st.error(f"🔴 **Sobreconcesión:** Otorgado {q_concesionado_dom - q_efectivo_dom:,.1f} L/s por encima de la extracción bruta requerida.")
        elif q_concesionado_dom < q_efectivo_dom * (1 - margen): st.warning(f"⚠️ **Riesgo de Subregistro:** Se requiere extraer {q_efectivo_dom - q_concesionado_dom:,.1f} L/s adicionales que no aparecen formalizados en la corporación.")
        else: st.success(f"✅ **Equilibrio Hídrico:** La concesión ({q_concesionado_dom:,.1f} L/s) cubre perfectamente la demanda y las pérdidas del sistema.")

        df_chart = pd.DataFrame([
            {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Consumo Neto", "Caudal (L/s)": q_necesario_dom},
            {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Pérdidas de Acueducto", "Caudal (L/s)": (q_efectivo_dom - q_necesario_dom)},
            {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Superficial", "Caudal (L/s)": q_sup},
            {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Subterránea", "Caudal (L/s)": q_sub}
        ])
        fig_sub = px.bar(df_chart, x="Categoría", y="Caudal (L/s)", color="Componente", color_discrete_map={"Consumo Neto": "#2980b9", "Pérdidas de Acueducto": "#e67e22", "Concesión Superficial": "#3498db", "Concesión Subterránea": "#2ecc71"}, title="Demanda Bruta vs Permisos Otorgados")
        fig_sub.add_hline(y=q_efectivo_dom, line_dash="dash", line_color="red", annotation_text="Límite Extracción Bruta")
        st.plotly_chart(fig_sub, use_container_width=True)
        
    st.divider()
    st.subheader("📋 Consolidado de Todos los Usos Registrados")
    if not df_usos_detalle.empty:
        c1, c2 = st.columns([2,1])
        with c1: st.dataframe(df_usos_detalle.astype(str), width="stretch")
        with c2:
            csv = df_usos_detalle.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Desglose (CSV)", data=csv, file_name=f'Usos_{lugar_sel}.csv', mime='text/csv')
    else: st.warning(f"⚠️ El territorio **{lugar_sel}** no registra datos formales.")

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE CARGAS (CONECTADO AL ALEPH)
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header(f"Inventario de Cargas Contaminantes ({anio_analisis})")
    
    # Lógica de recepción del Aleph (Blindada contra mayúsculas/minúsculas)
    aleph_activo = False
    if 'aleph_ha_pastos' in st.session_state and 'aleph_territorio_origen' in st.session_state:
        origen_aleph = normalizar_texto(st.session_state['aleph_territorio_origen'])
        destino_actual = normalizar_texto(lugar_sel)
        
        # Si el usuario está analizando el mismo municipio en ambas páginas
        if origen_aleph == destino_actual:
            aleph_activo = True

    if aleph_activo:
        st.success(f"🌐 **Conexión Aleph:** Las áreas agrícolas y de pastos para **{lugar_sel}** han sido extraídas automáticamente del modelo satelital de la página de Biodiversidad.")
    
    # Extraer valores del satélite, o usar manuales por defecto
    val_def_papa = float(st.session_state.get('aleph_ha_agricola', 50.0)) if aleph_activo else 50.0
    val_def_pastos = float(st.session_state.get('aleph_ha_pastos', 200.0)) if aleph_activo else 200.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏘️ Saneamiento")
        cobertura_ptar = st.slider("Cobertura de Tratamiento PTAR Urbana %:", 0, 100, 15)
        eficiencia_ptar = st.slider("Remoción DBO en PTAR %:", 0, 100, 80)
    with col2:
        st.markdown("### 🏭 Agroindustria")
        vol_suero = st.number_input("Sueros Lácteos Vertidos (L/día):", min_value=0, value=2000, step=500)
    with col3:
        st.markdown("### 🍓 Agricultura")
        ha_papa = st.number_input("Cultivos / Mosaico [Ha]:", min_value=0.0, value=val_def_papa, step=5.0, disabled=aleph_activo)
        ha_pastos = st.number_input("Pastos [Ha]:", min_value=0.0, value=val_def_pastos, step=10.0, disabled=aleph_activo)

    st.markdown("---")
    st.markdown(f"### 🐄🚜 Censo Pecuario ICA (2023) para: **{lugar_sel}**")
    
    # MOTOR DE CRUCE TERRITORIAL
    mpios_activos = []
    lugar_n = normalizar_texto(lugar_sel.replace("CAR: ", ""))
    if not df_territorio.empty:
        if nivel_sel_interno == "Jurisdicción Ambiental (CAR)": mpios_activos = df_territorio[df_territorio['car'].str.upper() == lugar_sel.replace("CAR: ", "").upper()]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Departamental": mpios_activos = df_territorio[df_territorio['depto_nom'].apply(normalizar_texto) == lugar_n]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Regional": mpios_activos = df_territorio[df_territorio['region'].apply(normalizar_texto) == lugar_n]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Municipal": mpios_activos = [lugar_n]
    if not mpios_activos: mpios_activos = [lugar_n]

    # EXTRACCIÓN DE BASES DE DATOS ICA
# EXTRACCIÓN DE BASES DE DATOS ICA
    total_bovinos, total_porcinos, total_aves, default_trat_porc = 0, 0, 0, 20
    if not df_bovinos.empty: total_bovinos = int(df_bovinos[df_bovinos['MUNICIPIO_NORM'].isin(mpios_activos)]['TOTALBOVINOS'].sum())
    
    if not df_porcinos.empty:
        df_p_f = df_porcinos[df_porcinos['MUNICIPIO_NORM'].isin(mpios_activos)]
        if not df_p_f.empty:
            total_porcinos = int(df_p_f['TOTAL_CERDOS'].sum())
            tecnificados = df_p_f['TOTAL_PORCINOS__TECNIFICADA'].sum() + df_p_f['TOTAL_PORCINOS__COMERCIAL_INDUSTRIAL_-_2021'].sum() if 'TOTAL_PORCINOS__TECNIFICADA' in df_p_f.columns else 0
            if total_porcinos > 0: default_trat_porc = int((tecnificados / total_porcinos) * 85)
            
    if not df_aves.empty:
        col_aves = 'TOTAL_AVES_CAPACIDAD_OCUPADA_MAS_AVES_TRASPATIO' if 'TOTAL_AVES_CAPACIDAD_OCUPADA_MAS_AVES_TRASPATIO' in df_aves.columns else 'TOTAL_AVES_CAPACIDAD_OCUPADA'
        if col_aves in df_aves.columns:
            total_aves = int(df_aves[df_aves['MUNICIPIO_NORM'].isin(mpios_activos)][col_aves].sum())

    if total_bovinos == 0: total_bovinos = int(pob_rural * 1.5)
    if total_porcinos == 0: total_porcinos = int(pob_rural * 0.8)

    col_pec1, col_pec2, col_pec3 = st.columns(3)
    with col_pec1:
        st.subheader("Sector Bovino")
        cabezas_bovinos = st.number_input("Bovinos (Cabezas):", min_value=0, value=total_bovinos, step=100)
        sistema_bovino = st.radio("Sistema Bovino:", ["Extensivo", "Estabulado"])
        factor_dbo_bov = 0.8 if "Estabulado" in sistema_bovino else 0.15 
        
    with col_pec2:
        st.subheader("Sector Porcícola")
        cabezas_porcinos = st.number_input("Porcinos (Cabezas):", min_value=0, value=total_porcinos, step=100)
        tratamiento_porc = st.slider("Eficiencia Biodigestor %:", 0, 100, default_trat_porc)
        factor_dbo_porc = 0.150 * (1 - (tratamiento_porc/100))
        
    with col_pec3:
        st.subheader("Sector Avícola")
        cabezas_aves = st.number_input("Aves (Galpones):", min_value=0, value=total_aves, step=1000)
        tratamiento_aves = st.slider("Manejo Gallinaza %:", 0, 100, 75)
        factor_dbo_aves = 0.015 * (1 - (tratamiento_aves/100)) # Factor aprox 15g DBO/ave

    dbo_urbana = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) 
    dbo_rural = pob_rural * 0.040 
    dbo_suero = vol_suero * 0.035 
    dbo_agricola = (ha_papa + ha_pastos) * 0.8 
    dbo_bovinos = cabezas_bovinos * factor_dbo_bov
    dbo_porcinos = cabezas_porcinos * factor_dbo_porc
    dbo_aves = cabezas_aves * factor_dbo_aves
    
    carga_total_dbo = dbo_urbana + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_aves + dbo_agricola
    
    coef_retorno = 0.85
    q_efluente_lps = (q_necesario_dom * coef_retorno) + (q_necesario_ind * 0.8) + (vol_suero / 86400) + ((cabezas_porcinos * 40)/86400)
    conc_efluente_mg_l = (carga_total_dbo * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0

    df_cargas = pd.DataFrame({
        "Fuente": ["Urbana", "Rural", "Agroindustria", "Agricultura", "Bovinos", "Porcinos", "Avicultura"], 
        "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_agricola, dbo_bovinos, dbo_porcinos, dbo_aves]
    })

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_cargas, use_container_width=True)
        csv_cargas = df_cargas.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Datos de Cargas (CSV)", data=csv_cargas, file_name=f"Inventario_Cargas_{lugar_sel}.csv", mime='text/csv')

    with col_g2:
        st.subheader(f"📈 Evolución de Carga Orgánica ({modelo_sel})")
        pob_u_evo = pob_urbana * (factor_evo / factor_proy)
        dbo_evo = (pob_u_evo * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))) + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_agricola
        fig_dbo_evo = go.Figure()
        fig_dbo_evo.add_trace(go.Scatter(x=anios_evo, y=dbo_evo, mode='lines', fill='tozeroy', name='Carga DBO (kg/d)', line=dict(color='#e74c3c', width=3)))
        st.plotly_chart(fig_dbo_evo, use_container_width=True)
        
# ------------------------------------------------------------------------------
# TAB 3: ASIMILACIÓN Y DILUCIÓN
# ------------------------------------------------------------------------------
with tab_dilucion:
    st.header(f"🌊 Modelo de Dilución y Capacidad de Asimilación ({anio_analisis})")
    col_a1, col_a2 = st.columns([1, 1.5])
    
    with col_a1:
        st.subheader("1. Cuerpo Receptor (Río)")
        q_rio = st.number_input("Caudal del Río aguas arriba (L/s):", value=1500.0, step=100.0)
        c_rio = st.number_input("Concentración DBO aguas arriba (mg/L):", value=2.0, step=0.5)
        
        st.markdown("---")
        st.subheader("2. Vertimientos al Cauce")
        
        q_vert_dom, q_vert_ind = 0.0, 0.0
        if not df_vertimientos.empty and lugar_sel != "N/A":
            lugar_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            if nivel_sel_interno == "Jurisdicción Ambiental (CAR)": df_filtro_v = df_vertimientos[df_vertimientos['car_norm'] == lugar_norm] if 'car_norm' in df_vertimientos.columns else pd.DataFrame()
            elif nivel_sel_interno == "Departamental": df_filtro_v = df_vertimientos[df_vertimientos['departamento_norm'] == lugar_norm] if 'departamento_norm' in df_vertimientos.columns else df_vertimientos.copy()
            elif nivel_sel_interno == "Municipal": df_filtro_v = df_vertimientos[df_vertimientos['municipio_norm'] == lugar_norm]
            else: df_filtro_v = pd.DataFrame()
                
            if not df_filtro_v.empty:
                q_vert_dom = df_filtro_v[df_filtro_v['tipo_vertimiento'].str.contains('Domestico|Municipal', case=False, na=False)]['caudal_vert_lps'].sum()
                q_vert_ind = df_filtro_v[df_filtro_v['tipo_vertimiento'].str.contains('Industrial|Agroindustrial', case=False, na=False)]['caudal_vert_lps'].sum()
                
        q_vert_total_legal = q_vert_dom + q_vert_ind
        
        st.caption("Comparativa de Descargas:")
        st.write(f"- **Efluente Teórico Generado:** {q_efluente_lps:,.1f} L/s")
        st.write(f"- **Vertimiento Legal (CAR):** {q_vert_total_legal:,.1f} L/s")
        
        tipo_modelo = st.radio("Caudal efluente a utilizar en la mezcla:", ["Usar Caudal Teórico (Peor Escenario)", "Usar Caudal Formalizado (Legal)"])
        q_modelo = q_efluente_lps if "Teórico" in tipo_modelo else q_vert_total_legal
        
        c_mix = ((q_rio * c_rio) + (q_modelo * conc_efluente_mg_l)) / (q_rio + q_modelo) if (q_rio + q_modelo) > 0 else 0
        
    with col_a2:
        st.subheader("Impacto Aguas Abajo (Balance de Masas)")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = c_mix, title = {'text': "DBO5 Final en el Río (mg/L)", 'font': {'size': 20}},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {'axis': {'range': [None, max(20, c_mix + 5)]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 3], 'color': "#2ecc71"}, {'range': [3, 5], 'color': "#f1c40f"}, {'range': [5, 10], 'color': "#e67e22"}, {'range': [10, 100], 'color': "#e74c3c"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5.0}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN (HOLÍSTICOS)
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Escenarios de Intervención (CuencaVerde)")
    st.markdown("Combina infraestructura gris (PTAR) con soluciones basadas en la naturaleza y agroecología.")
    
    st.subheader("A. Saneamiento Urbano")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1: esc_perdidas = st.slider("Fugas Acueducto (%):", 0.0, 100.0, float(max(0, perd_dom - 10)))
    with col_e2: esc_cobertura = st.slider("Cobertura PTAR Urbana (%):", 0.0, 100.0, float(min(100, cobertura_ptar + 30)))
    with col_e3: esc_eficiencia = st.slider("Remoción DBO PTAR (%):", 0.0, 100.0, float(min(100, eficiencia_ptar + 10)))
        
    st.subheader("B. Intervención Rural y Agroindustrial")
    col_e4, col_e5, col_e6 = st.columns(3)
    with col_e4: esc_biodigestores = st.slider("Cerdos con Biodigestor (%):", 0.0, 100.0, float(min(100, tratamiento_porc + 40)))
    with col_e5: esc_gallinaza = st.slider("Manejo de Gallinaza (%):", 0.0, 100.0, float(min(100, tratamiento_aves + 20)))
    with col_e6: esc_suero = st.slider("Suero Recuperado (Economía Circular) %:", 0.0, 100.0, 50.0)

    st.divider()
    
    # Cálculos del nuevo escenario
    q_efectivo_esc = q_necesario_dom / (1 - (esc_perdidas/100)) if esc_perdidas < 100 else q_necesario_dom
    
    dbo_urbana_esc = pob_urbana * 0.050 * (1 - (esc_cobertura/100 * esc_eficiencia/100))
    dbo_porcinos_esc = cabezas_porcinos * (0.150 * (1 - (esc_biodigestores/100)))
    dbo_aves_esc = cabezas_aves * (0.015 * (1 - (esc_gallinaza/100)))
    dbo_suero_esc = vol_suero * (1 - (esc_suero/100)) * 0.035
    
    carga_total_esc = dbo_urbana_esc + dbo_rural + dbo_suero_esc + dbo_bovinos + dbo_porcinos_esc + dbo_aves_esc + dbo_agricola
    
    col_er1, col_er2 = st.columns([1, 1.5])
    with col_er1:
        st.metric("Agua extraída de la fuente", f"{q_efectivo_esc:,.1f} L/s", delta=f"{q_efectivo_esc - q_efectivo_dom:,.1f} L/s (Agua conservada)", delta_color="inverse")
        st.metric("Materia Orgánica Total al Río", f"{carga_total_esc:,.1f} kg/día", delta=f"{carga_total_esc - carga_total_dbo:,.1f} kg/día (Contaminación evitada)", delta_color="inverse")
    
    with col_er2:
        df_esc = pd.DataFrame({
            "Escenario": ["1. Situación Actual", "1. Situación Actual", "2. Con Proyecto CuencaVerde", "2. Con Proyecto CuencaVerde"],
            "Variable": ["Extracción de Agua (L/s)", "Carga DBO (kg/día)", "Extracción de Agua (L/s)", "Carga DBO (kg/día)"],
            "Valor": [q_efectivo_dom, carga_total_dbo, q_efectivo_esc, carga_total_esc]
        })
        fig_esc = px.bar(df_esc, x="Variable", y="Valor", color="Escenario", barmode="group", title="Impacto Integral del Proyecto", color_discrete_sequence=["#e74c3c", "#2ecc71"])
        st.plotly_chart(fig_esc, use_container_width=True)
        
# ------------------------------------------------------------------------------
# TAB 5: MAPA DE CALOR (VISOR ESPACIAL CON FONDO MAPBOX)
# ------------------------------------------------------------------------------
with tab_mapa:
    st.header("🗺️ Mapa de Calor y Análisis Espacial")
    st.info(f"📍 **Enfoque Espacial:** Mostrando datos para **{nivel_sel_visual} - {lugar_sel}**.")
    
    var_mapa = st.selectbox("Variable a cartografiar:", [
        "1. Densidad de Puntos de Concesión (Coordenadas)", 
        "2. Densidad de Puntos de Vertimiento (Coordenadas)",
        "3. Cargas Contaminantes DBO Teóricas (Topología por Municipio)",
        "4. Caudal Requerido Teórico (Topología por Municipio)"
    ])
    
    if "Teórica" in var_mapa:
        st.caption("Mapa de calor jerárquico (Treemap) basado en cálculos poblacionales.")
        df_agg = pd.DataFrame()
        
        if nivel_sel_interno == "Nacional (Colombia)": df_m = df_mpios[df_mpios['año'] == anio_base].copy()
        elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)":
            car_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            mpios_car = set()
            if not df_concesiones.empty: mpios_car.update(df_concesiones[df_concesiones['car_norm'] == car_norm]['municipio_norm'].unique())
            if not df_vertimientos.empty: mpios_car.update(df_vertimientos[df_vertimientos['car_norm'] == car_norm]['municipio_norm'].unique())
            df_m = df_mpios[(df_mpios['municipio_norm'].isin(mpios_car)) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Departamental": df_m = df_mpios[(df_mpios['depto_nom'] == lugar_sel) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Regional": df_m = df_mpios[(df_mpios['region'] == lugar_sel) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Municipal": df_m = df_mpios[(df_mpios['municipio_norm'] == normalizar_texto(lugar_sel)) & (df_mpios['año'] == anio_base)].copy()
        else: df_m = df_mpios[df_mpios['año'] == anio_base].copy() 
            
        if not df_m.empty:
            df_agg = df_m.groupby('municipio')['Poblacion'].sum().reset_index()
            df_agg['Poblacion_Proy'] = df_agg['Poblacion'] * factor_proy
            
            if "Caudal" in var_mapa:
                df_agg['Valor'] = (df_agg['Poblacion_Proy'] * dotacion) / 86400
                fig_tree = px.treemap(df_agg, path=[px.Constant(lugar_sel), 'municipio'], values='Valor', color='Valor', color_continuous_scale='Blues', title="Caudal Doméstico Requerido (L/s)")
            else:
                df_agg['Valor'] = df_agg['Poblacion_Proy'] * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))
                fig_tree = px.treemap(df_agg, path=[px.Constant(lugar_sel), 'municipio'], values='Valor', color='Valor', color_continuous_scale='Reds', title="Carga Orgánica DBO (kg/día) aportada por Municipio")
                
            fig_tree.update_traces(textinfo="label+value")
            st.plotly_chart(fig_tree, use_container_width=True)
            
    else:
        st.caption("Mapa de densidad sobre cartografía base (Convierte MAGNA-SIRGAS a WGS84 en tiempo real).")
        df_map = df_concesiones.copy() if "Concesión" in var_mapa else df_vertimientos.copy()
        
        if not df_map.empty:
            lugar_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            if nivel_sel_interno == "Jurisdicción Ambiental (CAR)" and 'car_norm' in df_map.columns: df_map = df_map[df_map['car_norm'] == lugar_norm]
            elif nivel_sel_interno == "Departamental" and 'departamento_norm' in df_map.columns: df_map = df_map[df_map['departamento_norm'] == normalizar_texto(lugar_sel)]
            elif nivel_sel_interno == "Regional" and 'region_norm' in df_map.columns: df_map = df_map[df_map['region_norm'] == normalizar_texto(lugar_sel)]
            elif nivel_sel_interno == "Municipal": df_map = df_map[df_map['municipio_norm'] == lugar_norm]
            elif nivel_sel_interno == "Veredal" and 'vereda_norm' in df_map.columns: df_map = df_map[df_map['vereda_norm'] == lugar_norm]
            
            col_z = 'caudal_lps' if "Concesión" in var_mapa else 'caudal_vert_lps'
            df_map['coordenada_x'] = pd.to_numeric(df_map['coordenada_x'], errors='coerce')
            df_map['coordenada_y'] = pd.to_numeric(df_map['coordenada_y'], errors='coerce')
            df_map[col_z] = pd.to_numeric(df_map[col_z], errors='coerce')
            df_map = df_map.dropna(subset=['coordenada_x', 'coordenada_y', col_z])
            
            mask_magna = (df_map['coordenada_x'] > 100000) & (df_map['coordenada_y'] > 100000)
            mask_wgs84 = (df_map['coordenada_x'] < 0) & (df_map['coordenada_x'] > -85) & (df_map['coordenada_y'] > -5)
            
            df_plot = df_map[mask_magna | mask_wgs84].copy()
            
            if not df_plot.empty and df_plot[col_z].sum() > 0:
                try:
                    import pyproj
                    # EPSG:3116 es el origen Magna-Sirgas central (muy común en Antioquia)
                    transformer = pyproj.Transformer.from_crs("epsg:3116", "epsg:4326", always_xy=True)
                    
                    def to_wgs84(row):
                        if row['coordenada_x'] > 100000: # Es plana
                            lon, lat = transformer.transform(row['coordenada_x'], row['coordenada_y'])
                            return pd.Series({'lon': lon, 'lat': lat})
                        else: # Ya es WGS84
                            return pd.Series({'lon': row['coordenada_x'], 'lat': row['coordenada_y']})
                            
                    with st.spinner("Proyectando coordenadas a satélite..."):
                        coords = df_plot.apply(to_wgs84, axis=1)
                        df_plot['lon'] = coords['lon']
                        df_plot['lat'] = coords['lat']
                        # Limpiar errores de proyección
                        df_plot = df_plot[(df_plot['lon'] >= -85) & (df_plot['lon'] <= -60) & (df_plot['lat'] >= -5) & (df_plot['lat'] <= 15)]
                    
                    fig_dens = px.density_mapbox(df_plot, lat='lat', lon='lon', z=col_z, radius=12,
                                                 center=dict(lat=df_plot['lat'].mean(), lon=df_plot['lon'].mean()),
                                                 zoom=8, mapbox_style="carto-positron", 
                                                 title=f"Densidad Espacial de Caudales (L/s)",
                                                 color_continuous_scale="Viridis")
                    st.plotly_chart(fig_dens, use_container_width=True)
                    
                except ImportError:
                    st.error("💡 Para habilitar el mapa de fondo real, debes instalar 'pyproj'. Usando mapa base temporal.")
                    fig_dens = px.density_contour(df_plot, x="coordenada_x", y="coordenada_y", z=col_z, histfunc="sum", title="Densidad (Sin mapa de fondo)")
                    fig_dens.update_traces(contours_coloring="fill", colorscale="Viridis")
                    st.plotly_chart(fig_dens, use_container_width=True)
            else:
                st.warning("No hay suficientes coordenadas válidas para generar un mapa en esta jurisdicción.")
        else:
            st.warning("No hay base de datos disponible para esta variable.")

# ------------------------------------------------------------------------------
# TAB 6: EXPLORADOR SIRENA Y VERTIMIENTOS
# ------------------------------------------------------------------------------
with tab_sirena:
    st.header("📊 Explorador Ambiental Avanzado")
    st.info(f"📍 **Contexto Global Activo:** Estás navegando la base de datos bajo la lupa de: **{nivel_sel_visual} - {lugar_sel}**.")
    
    if not df_concesiones.empty:
        df_exp = df_concesiones.copy()
        lugar_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
        
        if nivel_sel_interno == "Jurisdicción Ambiental (CAR)" and 'car_norm' in df_exp.columns: df_exp = df_exp[df_exp['car_norm'] == lugar_norm]
        elif nivel_sel_interno == "Departamental" and 'departamento_norm' in df_exp.columns: df_exp = df_exp[df_exp['departamento_norm'] == normalizar_texto(lugar_sel)]
        elif nivel_sel_interno == "Regional" and 'region_norm' in df_exp.columns: df_exp = df_exp[df_exp['region_norm'] == normalizar_texto(lugar_sel)]
        elif nivel_sel_interno == "Municipal": df_exp = df_exp[df_exp['municipio_norm'] == lugar_norm]
        
        c_exp1, c_exp2 = st.columns([2, 1.5])
        with c_exp1:
            st.subheader(f"Registros Encontrados: {len(df_exp)}")
            st.dataframe(df_exp.astype(str), width="stretch")
        with c_exp2:
            if not df_exp.empty and df_exp['caudal_lps'].sum() > 0:
                agrupador = st.selectbox("Agrupar Concesiones por:", ["tipo_agua", "Sector_Sihcli", "uso_detalle", "estado"], index=0)
                df_agg = df_exp.groupby(agrupador)['caudal_lps'].sum().reset_index()
                fig_exp = px.pie(df_agg[df_agg['caudal_lps']>0], values='caudal_lps', names=agrupador, hole=0.4, title=f"Total: {df_agg['caudal_lps'].sum():,.1f} L/s")
                fig_exp.update_traces(textposition='inside', textinfo='value+label')
                st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.warning("No se ha cargado correctamente la base de datos oficial.")

# ------------------------------------------------------------------------------
# TAB 7: ANÁLISIS SECTORIAL DE EXTERNALIDADES (NIVEL AVANZADO)
# ------------------------------------------------------------------------------
with tab_extern:
    st.header("⚠️ Análisis Sectorial de Externalidades Ambientales")
    st.markdown(f"Evaluación avanzada de Huella de Carbono y Balance de Nutrientes (NPK) en **{lugar_sel}**.")
    
    # 1. RE-CÁLCULO INTERNO (Para evitar errores de variables huérfanas)
    dbo_domestica = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) + (pob_rural * 0.040)
    dbo_bovinos_ext = cabezas_bovinos * factor_dbo_bov
    dbo_porcinos_ext = cabezas_porcinos * factor_dbo_porc
    dbo_agricola_ext = (ha_papa + ha_pastos) * 0.8
    dbo_agroind_ext = vol_suero * 0.035
    
    # 2. MODELO DE NUTRIENTES (Nitrógeno, Fósforo, Potasio)
    # Factores de excreción y escorrentía aproximados (kg/día)
    n_dom = pob_urbana * 0.012 * (1 - (cobertura_ptar/100 * 0.3)) + (pob_rural * 0.012)
    p_dom = pob_urbana * 0.003 * (1 - (cobertura_ptar/100 * 0.2)) + (pob_rural * 0.003)
    k_dom = pob_urbana * 0.005 + (pob_rural * 0.005)
    
    n_bov = cabezas_bovinos * 0.15 * 0.2 # Solo el 20% llega a cuerpos de agua (escorrentía)
    p_bov = cabezas_bovinos * 0.04 * 0.2
    k_bov = cabezas_bovinos * 0.12 * 0.2
    
    n_porc = cabezas_porcinos * 0.08 * (1 - (tratamiento_porc/100 * 0.5))
    p_porc = cabezas_porcinos * 0.02 * (1 - (tratamiento_porc/100 * 0.4))
    k_porc = cabezas_porcinos * 0.05 * (1 - (tratamiento_porc/100 * 0.2))

    n_aves = cabezas_aves * 0.0015 * (1 - (tratamiento_aves/100 * 0.6))
    p_aves = cabezas_aves * 0.0005 * (1 - (tratamiento_aves/100 * 0.6))
    k_aves = cabezas_aves * 0.0006 * (1 - (tratamiento_aves/100 * 0.6))    
    
    n_agr = (ha_papa * 1.5) + (ha_pastos * 0.5)
    p_agr = (ha_papa * 0.3) + (ha_pastos * 0.1)
    k_agr = (ha_papa * 1.2) + (ha_pastos * 0.4)

    col_ext1, col_ext2 = st.columns([1, 1.2])
    
    with col_ext1:
        st.subheader("Huella de Carbono (Gases Efecto Invernadero)")
        st.caption("Conversión de Metano (CH4) y Óxido Nitroso (N2O) a Toneladas de CO2 equivalente (tCO2e/año).")
        
        # Factores IPCC: CH4 a CO2e (x 28), N2O a CO2e (x 265)
        # Doméstico: Principalmente CH4 de fosas sépticas y PTAR sin captura.
        co2e_dom = (dbo_domestica * 0.25 * 28 * 365) / 1000 
        # Bovinos: Fermentación entérica (muy alta) + estiércol
        co2e_bov = (cabezas_bovinos * 0.16 * 28 * 365) / 1000 # ~60kg CH4/vaca/año
        # Porcinos: Manejo de estiércol (Lagunas anaerobias generan mucho CH4)
        co2e_porc = (dbo_porcinos_ext * 0.25 * 28 * 365) / 1000
        # Aves: Manejo de Gallinaza (Compost)
        co2e_aves = (dbo_aves * 0.25 * 28 * 365) / 1000
        
        # Agrícola: Emisiones de N2O por fertilización nitrogenada
        co2e_agr = (n_agr * 0.01 * 265 * 365) / 1000 
        
        df_co2e = pd.DataFrame({
        "Sector": ["Saneamiento Doméstico", "Ganadería Bovina", "Porcicultura", "Avicultura", "Agricultura"],
        "tCO2e_año": [co2e_dom, co2e_bov, co2e_porc, co2e_aves, co2e_agr]
        })
        
        st.metric("Emisiones Totales del Territorio", f"{df_co2e['tCO2e_año'].sum():,.0f} Ton CO2e/año", help="Toneladas de Dióxido de Carbono equivalente al año.")
        
        fig_co2 = px.pie(df_co2e, values="tCO2e_año", names="Sector", hole=0.4, title="Distribución de la Huella de Carbono")
        fig_co2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_co2, use_container_width=True)
        
    with col_ext2:
        st.subheader("Balance de Nutrientes (Eutrofización)")
        st.caption("Cargas diarias de N-P-K aportadas por cada sector a la cuenca.") 
        
        df_nutrientes = pd.DataFrame([
            {"Sector": "Doméstico", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_dom},
            {"Sector": "Doméstico", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_dom},
            {"Sector": "Doméstico", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_dom},
            {"Sector": "Bovinos", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_bov},
            {"Sector": "Bovinos", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_bov},
            {"Sector": "Bovinos", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_bov},
            {"Sector": "Porcinos", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_porc},
            {"Sector": "Porcinos", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_porc},
            {"Sector": "Porcinos", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_porc},
            {"Sector": "Agrícola", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_agr},
            {"Sector": "Agrícola", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_agr},
            {"Sector": "Agrícola", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_agr},
            {"Sector": "Avicultura", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_aves},
            {"Sector": "Avicultura", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_aves},
            {"Sector": "Avicultura", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_aves}
        ])
    
        n_total = n_dom + n_bov + n_porc + n_aves + n_agr
        if n_total > 1000:
            st.error(f"🔴 **Riesgo Severo de Eutrofización:** Carga de Nitrógeno crítica ({n_total:,.0f} kg/día). Alta probabilidad de blooms algales e hipoxia en el cuerpo receptor.")
        elif n_total > 300:
            st.warning(f"⚠️ **Riesgo Moderado:** Carga de Nitrógeno de {n_total:,.0f} kg/día. Se requiere monitoreo de calidad del agua.")
        else:
            st.success(f"✅ **Carga Estable:** El aporte de nutrientes ({n_total:,.0f} kg N/día) está dentro de límites asimilables.")

    st.divider()
    st.subheader("📋 Consolidado de Externalidades")
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        # Pivot table para una visualización elegante
        df_pivot = df_nutrientes.pivot(index='Sector', columns='Nutriente', values='Carga_kg_dia').reset_index()
        df_pivot = pd.merge(df_pivot, df_co2e, on='Sector', how='left')
        df_pivot.rename(columns={'tCO2e_año': 'Huella_Carbono (tCO2e/año)'}, inplace=True)
        df_pivot.fillna(0, inplace=True)
        
        st.dataframe(df_pivot.style.format({
            "Nitrógeno (N)": "{:,.1f}", "Fósforo (P)": "{:,.1f}", "Potasio (K)": "{:,.1f}", "Huella_Carbono (tCO2e/año)": "{:,.0f}"
        }), use_container_width=True)
        
    with col_t2:
        csv_ext = df_pivot.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Reporte de Externalidades (CSV)", data=csv_ext, file_name=f"Externalidades_NPK_CO2_{lugar_sel}.csv", mime='text/csv')

# ------------------------------------------------------------------------------
# TAB 9: ECONOMÍA CIRCULAR (VALORIZACIÓN DE LACTOSUERO)
# ------------------------------------------------------------------------------
with tab_lactosuero:
    st.header("🥛 Economía Circular: Valorización Industrial de Lactosueros")
    st.markdown(f"Evaluación del potencial tecnológico y mitigación ambiental para la cuenca lechera de **{lugar_sel}**.")
    
    # Extraer específicamente las vacas en edad de producción del censo ICA
    vacas_adultas = 0
    if not df_bovinos.empty and 'HEMBRAS>3AÑOS' in df_bovinos.columns:
        vacas_adultas = int(df_bovinos[df_bovinos['MUNICIPIO_NORM'].isin(mpios_activos)]['HEMBRAS>3AÑOS'].sum())
    if vacas_adultas == 0: vacas_adultas = int(cabezas_bovinos * 0.45) # Estimado si no hay datos
    
    col_l1, col_l2 = st.columns([1, 1.3])
    with col_l1:
        st.subheader("⚙️ Parámetros de la Industria Quesera")
        vacas_ordeno = st.number_input("Vacas en Ordeño (Hembras > 3 años - ICA):", min_value=0, value=int(vacas_adultas * 0.6), step=100, help="Asume un 60% de hembras adultas en lactancia activa.")
        litros_vaca = st.slider("Producción Media (L/vaca/día):", 5.0, 30.0, 12.0, step=0.5)
        pct_queso = st.slider("% de Leche destinada a Quesería Local:", 0.0, 100.0, 30.0, step=1.0)
        
        leche_total = vacas_ordeno * litros_vaca
        leche_queso = leche_total * (pct_queso / 100.0)
        # La regla de oro: 100L leche -> 10kg queso + 90L suero
        suero_generado = leche_queso * 0.90 
        
        st.markdown("---")
        st.metric("🥛 Leche Acopiada Estimada", f"{leche_total:,.0f} L/día")
        st.metric("🧀 Leche hacia Queserías", f"{leche_queso:,.0f} L/día")
        st.metric("⚠️ Lactosuero Generado", f"{suero_generado:,.0f} L/día", delta="Residuo Altamente Contaminante", delta_color="inverse")
        
        df_suero = pd.DataFrame({
            "Destino": ["Queso (Producto Útil)", "Lactosuero (Residuo/Subproducto)"],
            "Volumen (L/día)": [leche_queso * 0.10, suero_generado]
        })
        fig_suero = px.pie(df_suero, values="Volumen (L/día)", names="Destino", hole=0.5, color_discrete_sequence=["#f1c40f", "#e67e22"])
        fig_suero.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_suero, use_container_width=True)

    with col_l2:
        st.subheader("🔄 Potencial Tecnológico (Transformación a WPC)")
        st.info("El suero contiene proteínas y lactosa de altísimo valor. Mediante plantas de **Ultrafiltración (UF)** y secado, se obtiene Proteína de Suero Concentrada (WPC) comercial.")
        
        # Rendimiento tecnológico: 1000 L de suero generan aprox. 6.5 a 7 kg de WPC al 80%
        kg_proteina = (suero_generado / 1000) * 6.8 
        precio_kg_wpc = 8.5 # Precio internacional USD/kg aprox
        ingresos_anuales = (kg_proteina * precio_kg_wpc) * 365
        
        # Impacto Ambiental Evitado (El suero tiene aprox. 35,000 mg/L de DBO)
        dbo_suero_evitada = suero_generado * 0.035 # kg/día
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Proteína Extraíble (WPC 80%)", f"{kg_proteina:,.1f} kg/día")
        c_res2.metric("Nuevos Ingresos Potenciales", f"${ingresos_anuales:,.0f} USD/año")
        
        st.success(f"🌱 **Impacto Hídrico Evitado:** Si este suero se procesa en lugar de arrojarse al campo o alcantarillado, el territorio evita la contaminación equivalente a **{dbo_suero_evitada:,.0f} kg de DBO/día**. ¡Esto equivale a las aguas residuales de una ciudad de {(dbo_suero_evitada/0.050):,.0f} habitantes!")
        
        # Integración con el Aleph: Sincroniza el dato si quieren usarlo en la pestaña de inventario
        if st.button("🔌 Sincronizar suero con el Inventario General (Aleph)"):
            st.session_state['aleph_vol_suero'] = float(suero_generado)
            st.toast("✅ ¡Volumen de suero inyectado en la memoria global de Sihcli-Poter!")

# =====================================================================
        # 🌊 MÓDULO AVANZADO: ASIMILACIÓN Y CURVA DE OXÍGENO (STREETER-PHELPS)
        # =====================================================================
        st.markdown("---")
        st.header("🌊 4. Capacidad de Asimilación del Río Receptor")
        st.info("Modelo de Streeter-Phelps: Simula la caída y recuperación del Oxígeno Disuelto (OD) aguas abajo del vertimiento principal de la zona seleccionada.")
        
        from modules.water_quality import calcular_streeter_phelps

        # 1. Parámetros Físicos del Río (Interactivos)
        with st.expander("⚙️ Características Físicas y Climáticas del Río", expanded=True):
            cr1, cr2, cr3 = st.columns(3)
            with cr1:
                q_rio = st.number_input("Caudal del Río (m³/s):", min_value=0.1, value=5.0, step=0.5, help="Caudal medio en el punto de vertimiento.")
                t_agua = st.slider("Temperatura del Agua (°C):", min_value=10.0, max_value=35.0, value=22.0, step=0.5)
            with cr2:
                v_rio = st.slider("Velocidad del Flujo (m/s):", min_value=0.1, max_value=3.0, value=0.5, step=0.1, help="Ríos rápidos reairean mejor.")
                h_rio = st.slider("Profundidad Media (m):", min_value=0.2, max_value=5.0, value=1.0, step=0.2, help="Ríos pandas (poco profundos) capturan más oxígeno.")
            with cr3:
                od_rio_arriba = st.slider("Oxígeno Disuelto Aguas Arriba (mg/L):", min_value=0.0, max_value=12.0, value=7.5, step=0.5)
                dist_sim = st.slider("Distancia a Simular (km):", min_value=5, max_value=150, value=50, step=5)

        # 2. Balance de Masas (Mezcla Río + Vertimiento)
        # Nota: Aquí usamos la variable de carga total que ya calculaste en las pestañas anteriores.
        # Asumo que tienes una variable que suma toda la DBO (ej. carga_dbo_total). 
        # Si tu variable se llama diferente, cámbiala en la línea de abajo.
        
        try:
            # Intentamos leer la carga calculada en el flujo normal de tu script
            carga_vertimiento_kg_dia = carga_dbo_total # <--- Ajusta este nombre si es necesario
        except NameError:
            # Fallback seguro por si la variable tiene otro nombre temporalmente
            carga_vertimiento_kg_dia = 5000.0 
            
        # Asumimos un caudal de vertimiento de las aguas residuales (ej. 150 L/hab/dia convertido a m3/s)
        # Para simplificar el balance inicial, usamos un caudal de vertimiento base (0.2 m3/s)
        q_vertimiento = 0.2 
        q_mezcla = q_rio + q_vertimiento
        
        # Concentración de DBO del vertimiento (mg/L) = (kg/dia * 1000) / (m3/s * 86400)
        dbo_vert_mgL = (carga_vertimiento_kg_dia * 1000) / (q_vertimiento * 86400)
        
        # DBO del río limpio aguas arriba (asumimos río sano = 2 mg/L)
        dbo_rio_arriba = 2.0 
        
        # BALANCE DE DBO (Ecuación de Mezcla)
        L0_mezcla = ((q_rio * dbo_rio_arriba) + (q_vertimiento * dbo_vert_mgL)) / q_mezcla
        
        # BALANCE DE OXÍGENO
        # Asumimos que el vertimiento residual no tiene oxígeno (0 mg/L)
        od_mezcla = ((q_rio * od_rio_arriba) + (q_vertimiento * 0.0)) / q_mezcla
        
        # Oxígeno de saturación (para calcular el déficit inicial D0)
        od_sat = 14.652 - 0.41022 * t_agua + 0.007991 * (t_agua ** 2) - 0.000077774 * (t_agua ** 3)
        D0_mezcla = max(0, od_sat - od_mezcla)

        # 3. Ejecutar el Motor Matemático
        df_sag = calcular_streeter_phelps(
            L0=L0_mezcla, 
            D0=D0_mezcla, 
            T_agua=t_agua, 
            v_ms=v_rio, 
            H_m=h_rio, 
            dist_max_km=dist_sim, 
            paso_km=0.5
        )

        # 4. Encontrar el Punto Crítico (Donde el oxígeno es mínimo)
        punto_critico = df_sag.loc[df_sag['Oxigeno_Disuelto_mgL'].idxmin()]
        od_minimo = punto_critico['Oxigeno_Disuelto_mgL']
        km_critico = punto_critico['Distancia_km']
        
        # 5. Dibujar la Curva de Oxígeno
        import plotly.graph_objects as go
        
        fig_sag = go.Figure()
        
        # Curva principal del Oxígeno
        fig_sag.add_trace(go.Scatter(
            x=df_sag['Distancia_km'], 
            y=df_sag['Oxigeno_Disuelto_mgL'], 
            mode='lines', 
            name='Oxígeno Disuelto (OD)', 
            line=dict(color='#3498db', width=4)
        ))
        
        # Línea de Saturación (El máximo posible según la temperatura)
        fig_sag.add_trace(go.Scatter(
            x=df_sag['Distancia_km'], 
            y=df_sag['OD_Saturacion'], 
            mode='lines', 
            name='Saturación Ideal', 
            line=dict(color='rgba(52, 152, 219, 0.3)', width=2, dash='dash')
        ))
        
        # Línea de Límite Ecológico (4 mg/L)
        fig_sag.add_trace(go.Scatter(
            x=df_sag['Distancia_km'], 
            y=df_sag['Limite_Critico'], 
            mode='lines', 
            name='Límite Fauna Acuática (4 mg/L)', 
            line=dict(color='#e74c3c', width=2, dash='dot')
        ))
        
        # Marcador del Punto Crítico
        fig_sag.add_trace(go.Scatter(
            x=[km_critico], 
            y=[od_minimo], 
            mode='markers+text', 
            name='Punto Crítico',
            marker=dict(color='red', size=12, symbol='x'),
            text=[f"{od_minimo:.1f} mg/L"],
            textposition="bottom center"
        ))

        fig_sag.update_layout(
            title=f"Curva de Oxígeno Disuelto - Río Receptor ({t_agua}°C)",
            xaxis_title="Distancia Aguas Abajo (km)",
            yaxis_title="Concentración (mg/L)",
            hovermode="x unified",
            height=450,
            yaxis=dict(range=[0, od_sat + 1])
        )

        # Mostrar métricas y gráfica
        m_r1, m_r2, m_r3 = st.columns(3)
        m_r1.metric("DBO Total Mezcla (L0)", f"{L0_mezcla:.1f} mg/L")
        estado_rio = "⚠️ Zona Anóxica Muerte Peces" if od_minimo < 4.0 else "✅ Saludable"
        m_r2.metric("OD Mínimo Crítico", f"{od_minimo:.1f} mg/L", delta=estado_rio, delta_color="normal" if od_minimo >= 4.0 else "inverse")
        m_r3.metric("Ubicación del Impacto Crítico", f"Km {km_critico:.1f}")

        st.plotly_chart(fig_sag, use_container_width=True)
