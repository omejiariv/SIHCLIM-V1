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
Modelo integral del ciclo hidrosocial: Simulación de demanda sectorial, cargas contaminantes (DBO, SST), 
capacidad de asimilación, análisis de formalización y minería de datos de concesiones (SIRENA).
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
# 🔌 CONECTOR A BASES DE DATOS (Soporte nativo para XLSX)
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
        
        # Autodetectar columnas 
        col_caudal = 'caudal_por_uso' if 'caudal_por_uso' in df.columns else ('caudal_usuario' if 'caudal_usuario' in df.columns else None)
        if not col_caudal: 
            cands = [c for c in df.columns if 'caudal' in c and 'acumulado' not in c]
            col_caudal = cands[0] if cands else None
            
        col_uso = 'uso' if 'uso' in df.columns else None
        col_mpio = 'municipio' if 'municipio' in df.columns else None
        col_vereda = 'vereda' if 'vereda' in df.columns else None
        col_depto = 'departamento' if 'departamento' in df.columns else None
        col_region = 'region' if 'region' in df.columns else None
        col_asunto = 'asunto' if 'asunto' in df.columns else None
        col_cota = 'cota' if 'cota' in df.columns else None
        col_estado = 'estado' if 'estado' in df.columns else None
        
        if col_caudal and col_mpio:
            df = df.dropna(subset=[col_mpio]).copy() 
            
            # Conversión de caudales
            if df[col_caudal].dtype == object:
                df[col_caudal] = df[col_caudal].astype(str).str.replace(',', '.')
            df['caudal_lps'] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
            
            if col_cota: df['cota_num'] = pd.to_numeric(df[col_cota], errors='coerce').fillna(-1)
            else: df['cota_num'] = -1
            
            # Normalización de variables territoriales
            df['municipio'] = df[col_mpio].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            
            if col_vereda: 
                df['vereda'] = df[col_vereda].astype(str).str.strip().str.title()
                df['vereda_norm'] = df['vereda'].apply(normalizar_texto)
            else: df['vereda_norm'] = ""

            if col_depto: df['departamento_norm'] = df[col_depto].apply(normalizar_texto)
            if col_region: df['region_norm'] = df[col_region].apply(normalizar_texto)
            
            if col_asunto:
                df['tipo_agua'] = np.where(df[col_asunto].str.lower().str.contains('subterran|subterrán|pozo|aljibe', regex=True, na=False), 'Subterránea',
                                  np.where(df[col_asunto].str.lower().str.contains('superficial|corriente', regex=True, na=False), 'Superficial', 'No Especificado'))
            else:
                df['tipo_agua'] = 'No Especificado'

            if col_uso: df['uso_detalle'] = df[col_uso].fillna('Sin Información').astype(str).str.title().str.strip()
            else: df['uso_detalle'] = 'Sin Información'

            def clasificar_uso_base(u):
                u = normalizar_texto(u)
                if any(x in u for x in ['domestico', 'consumo humano', 'abastecimiento', 'acueducto']): return 'Doméstico'
                elif any(x in u for x in ['agricola', 'pecuario', 'acuicultura', 'agroindustrial', 'riego', 'piscicola', 'silvicultura']): return 'Agrícola/Pecuario'
                elif any(x in u for x in ['industrial', 'mineria', 'minero', 'generacion de energia']): return 'Industrial'
                else: return 'Otros'
                
            df['Sector_Sihcli'] = df['uso_detalle'].apply(clasificar_uso_base)
            
            if col_estado: df['estado'] = df[col_estado].fillna('Desconocido').astype(str).str.title().str.strip()
            else: df['estado'] = 'Desconocido'
                
            return df
    return pd.DataFrame()

# ... (aquí arriba termina la función cargar_concesiones) ...

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


df_mpios = cargar_municipios()
df_veredas = cargar_veredas()
df_concesiones = cargar_concesiones()
df_vertimientos = cargar_vertimientos()

# FUNCIONES MATEMÁTICAS
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
nivel_sel_interno = nivel_sel

if nivel_sel == "Nacional (Colombia)": lugar_sel = "Colombia"
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
with col_t4: k_man = st.number_input("Capacidad de Carga (K):", value=float(max(pob_t_base * 2.0, 1000)), step=1000.0, disabled=(modelo_sel != "Logístico"))

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
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion, tab_mapa, tab_sirena = st.tabs([
    "🚰 2. Demanda y Eficiencia",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios de Mitigación",
    "🗺️ 6. Mapa de Calor (Visor)",
    "📊 7. Explorador SIRENA"
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
        st.subheader("2. Demanda Legal (SIRENA)")
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
            {"Categoría": "Registro SIRENA (Legal)", "Componente": "Concesión Superficial", "Caudal (L/s)": q_sup},
            {"Categoría": "Registro SIRENA (Legal)", "Componente": "Concesión Subterránea", "Caudal (L/s)": q_sub}
        ])
        fig_sub = px.bar(df_chart, x="Categoría", y="Caudal (L/s)", color="Componente", color_discrete_map={"Consumo Neto": "#2980b9", "Pérdidas de Acueducto": "#e67e22", "Concesión Superficial": "#3498db", "Concesión Subterránea": "#2ecc71"}, title="Demanda Bruta vs Permisos Otorgados")
        fig_sub.add_hline(y=q_efectivo_dom, line_dash="dash", line_color="red", annotation_text="Límite Extracción Bruta")
        
        # Ajuste para evitar el warning de ancho de contenedor
        st.plotly_chart(fig_sub, use_container_width=True)
        
    st.divider()
    st.subheader("📋 Consolidado de Todos los Usos Registrados")
    if not df_usos_detalle.empty:
        c1, c2 = st.columns([2,1])
        with c1: 
            # ¡Solución Mágica de Arrow! Convertimos la tabla a texto puro para que Streamlit no colapse
            st.dataframe(df_usos_detalle.astype(str), use_container_width=True)
        with c2:
            csv = df_usos_detalle.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Desglose (CSV)", data=csv, file_name=f'Usos_SIRENA_{lugar_sel}.csv', mime='text/csv')
    else: 
        st.warning(f"⚠️ El territorio **{lugar_sel}** no registra datos formales en la corporación.")

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE CARGAS
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header(f"Inventario de Cargas Contaminantes ({anio_analisis})")
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
    # CORRECCIÓN: Ahora lee las nuevas variables q_necesario_dom y q_necesario_ind de la Pestaña 1
    q_efluente_lps = (q_necesario_dom * coef_retorno) + (q_necesario_ind * 0.8) + (vol_suero / 86400)
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
        st.subheader("Impacto Aguas Abajo")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = c_mix, title = {'text': "DBO5 Final (mg/L)", 'font': {'size': 20}},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {'axis': {'range': [None, max(20, c_mix + 5)]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 3], 'color': "#2ecc71"}, {'range': [3, 5], 'color': "#f1c40f"}, {'range': [5, 10], 'color': "#e67e22"}, {'range': [10, 100], 'color': "#e74c3c"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5.0}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN (NUEVO)
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Escenarios de Mitigación (CuencaVerde)")
    st.markdown("¿Qué impacto logramos si intervenimos el territorio? Ajusta las metas del proyecto y observa el resultado.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.subheader("Eficiencia de Redes")
        esc_perdidas = st.slider("Reducir pérdidas de acueducto a (%):", 0.0, 100.0, float(max(0, perd_dom - 10)), help="Evitar fugas reduce la extracción del río.")
    with col_e2:
        st.subheader("Saneamiento")
        esc_cobertura = st.slider("Aumentar Cobertura PTAR a (%):", 0.0, 100.0, float(min(100, cobertura_ptar + 30)))
    with col_e3:
        st.subheader("Tecnología PTAR")
        esc_eficiencia = st.slider("Mejorar Remoción DBO a (%):", 0.0, 100.0, float(min(100, eficiencia_ptar + 10)))
        
    st.divider()
    
    # Recálculos
    q_efectivo_esc = q_necesario_dom / (1 - (esc_perdidas/100)) if esc_perdidas < 100 else q_necesario_dom
    dbo_urbana_esc = pob_urbana * 0.050 * (1 - (esc_cobertura/100 * esc_eficiencia/100))
    carga_total_esc = dbo_urbana_esc + dbo_rural + dbo_suero + dbo_cerdos + dbo_agricola
    
    col_er1, col_er2 = st.columns([1, 1.5])
    with col_er1:
        st.metric("Extracción Bruta de Agua", f"{q_efectivo_esc:.1f} L/s", delta=f"{q_efectivo_esc - q_efectivo_dom:.1f} L/s (Agua salvada en la fuente)", delta_color="inverse")
        st.metric("Carga Contaminante DBO", f"{carga_total_esc:.1f} kg/día", delta=f"{carga_total_esc - carga_total_dbo:.1f} kg/día (Contaminación evitada)", delta_color="inverse")
    
    with col_er2:
        df_esc = pd.DataFrame({
            "Escenario": ["1. Situación Actual", "1. Situación Actual", "2. Con Proyecto CuencaVerde", "2. Con Proyecto CuencaVerde"],
            "Variable": ["Extracción de Agua (L/s)", "Carga DBO (kg/día)", "Extracción de Agua (L/s)", "Carga DBO (kg/día)"],
            "Valor": [q_efectivo_dom, carga_total_dbo, q_efectivo_esc, carga_total_esc]
        })
        fig_esc = px.bar(df_esc, x="Variable", y="Valor", color="Escenario", barmode="group", title="Impacto del Proyecto Ambiental", color_discrete_sequence=["#e74c3c", "#2ecc71"])
        st.plotly_chart(fig_esc, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: MAPA DE CALOR ESPACIAL Y TOPOLÓGICO
# ------------------------------------------------------------------------------
with tab_mapa:
    st.header("🗺️ Mapa de Calor y Análisis Espacial")
    st.markdown("Distribución territorial de cargas hídricas en el área seleccionada.")
    
    var_mapa = st.selectbox("Variable a cartografiar:", [
        "1. Cargas Contaminantes DBO Teóricas (Topología por Municipio)",
        "2. Caudal Requerido Teórico (Topología por Municipio)",
        "3. Densidad de Puntos de Concesión (Coordenadas SIRENA)", 
        "4. Densidad de Puntos de Vertimiento (Coordenadas CAR)"
    ])
    
    if "Teórica" in var_mapa:
        st.caption("Mapa de calor jerárquico (Treemap) basado en los cálculos matemáticos municipales.")
        df_agg = pd.DataFrame()
        
        # Filtra municipios según el contexto
        if nivel_sel_interno == "Nacional (Colombia)": df_m = df_mpios[df_mpios['año'] == anio_base].copy()
        elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)":
            car_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            mpios_car = set()
            if not df_concesiones.empty: mpios_car.update(df_concesiones[df_concesiones['car_norm'] == car_norm]['municipio_norm'].unique())
            df_m = df_mpios[(df_mpios['municipio_norm'].isin(mpios_car)) & (df_mpios['año'] == anio_base)].copy()
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
        st.caption("Mapa de densidad térmica 2D basado en coordenadas MAGNA-SIRGAS o WGS84.")
        df_map = df_concesiones.copy() if "Concesión" in var_mapa else df_vertimientos.copy()
        
        if not df_map.empty:
            lugar_norm = normalizar_texto(lugar_sel.replace("CAR: ", ""))
            if nivel_sel_interno == "Jurisdicción Ambiental (CAR)" and 'car_norm' in df_map.columns: df_map = df_map[df_map['car_norm'] == lugar_norm]
            elif nivel_sel_interno == "Municipal": df_map = df_map[df_map['municipio_norm'] == lugar_norm]
            
            col_z = 'caudal_lps' if "Concesión" in var_mapa else 'caudal_vert_lps'
            
            # BLINDAJE NUMÉRICO OBLIGATORIO PARA EVITAR TYPE-ERRORS
            df_map['coordenada_x'] = pd.to_numeric(df_map['coordenada_x'], errors='coerce')
            df_map['coordenada_y'] = pd.to_numeric(df_map['coordenada_y'], errors='coerce')
            df_map[col_z] = pd.to_numeric(df_map[col_z], errors='coerce')
            
            # Borrar filas que no tengan coordenadas válidas o caudal
            df_map = df_map.dropna(subset=['coordenada_x', 'coordenada_y', col_z])
            
            # Filtro para ignorar coordenadas atípicas (evita que el mapa se rompa)
            df_map = df_map[(df_map['coordenada_x'] > 1000) & (df_map['coordenada_y'] > 1000)]
            
            if not df_map.empty and df_map[col_z].sum() > 0:
                fig_dens = px.density_contour(df_map, x="coordenada_x", y="coordenada_y", z=col_z, histfunc="sum", fill=True, colorscale="Viridis", title=f"Densidad Espacial de Caudales (L/s)")
                st.plotly_chart(fig_dens, use_container_width=True)
            else:
                st.warning("Las coordenadas espaciales registradas en la base de datos presentan errores, están vacías o no son numéricas.")
        else:
            st.warning("No hay base de datos disponible para esta variable.")
            
# ------------------------------------------------------------------------------
# TAB 6: EXPLORADOR SIRENA (Data Mining Avanzado)
# ------------------------------------------------------------------------------
with tab_sirena:
    st.header("📊 Explorador Avanzado de Concesiones (SIRENA)")
    
    # AÑADIDO: Banner de Contexto para no confundir al usuario
    st.info(f"📍 **Contexto Global Activo:** Estás navegando la base de datos bajo la lupa de: **{nivel_sel} - {lugar_sel}**. (Usa los filtros de abajo para búsquedas específicas independientes).")
    
    st.markdown("Minería de datos sobre el universo total de resoluciones ambientales.")
    
    if not df_concesiones.empty:
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1: 
            edos = df_concesiones['estado'].dropna().unique() if 'estado' in df_concesiones.columns else []
            f_estado = st.multiselect("Estado del Trámite:", edos, default=["Activo"] if "Activo" in edos else None)
        with col_e2:
            f_tipo = st.multiselect("Fuente de Agua:", df_concesiones['tipo_agua'].unique())
        with col_e3:
            f_uso = st.multiselect("Uso Detallado:", sorted(df_concesiones['uso_detalle'].unique()))
        with col_e4:
            f_mpio = st.multiselect("Municipio(s):", sorted(df_concesiones['municipio'].unique()))

        df_exp = df_concesiones.copy()
        
        if 'cota_num' in df_exp.columns and df_exp['cota_num'].max() > 0:
            df_exp_valid_cota = df_exp[df_exp['cota_num'] >= 0]
            if not df_exp_valid_cota.empty:
                max_cota = float(df_exp_valid_cota['cota_num'].max())
                st.caption("Filtro de Elevación Topográfica:")
                rango_cota = st.slider("Rango de Cota (m.s.n.m):", 0.0, max_cota, (0.0, max_cota))
                df_exp = df_exp[((df_exp['cota_num'] >= rango_cota[0]) & (df_exp['cota_num'] <= rango_cota[1])) | (df_exp['cota_num'] == -1)]

        if f_estado: df_exp = df_exp[df_exp['estado'].isin(f_estado)]
        if f_tipo: df_exp = df_exp[df_exp['tipo_agua'].isin(f_tipo)]
        if f_uso: df_exp = df_exp[df_exp['uso_detalle'].isin(f_uso)]
        if f_mpio: df_exp = df_exp[df_exp['municipio'].isin(f_mpio)]
        
        st.divider()
        c_exp1, c_exp2 = st.columns([2, 1.5])
        with c_exp1:
            st.subheader(f"Registros Encontrados: {len(df_exp)}")
            st.dataframe(df_exp.astype(str), width="stretch")
            csv_exp = df_exp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exportar Resultados (CSV)", data=csv_exp, file_name="Reporte_SIRENA.csv", mime="text/csv")
            
        with c_exp2:
            st.subheader("Distribución de Caudales Netos")
            if not df_exp.empty and df_exp['caudal_lps'].sum() > 0:
                agrupador = st.selectbox("Agrupar gráfico por:", ["tipo_agua", "Sector_Sihcli", "uso_detalle", "municipio", "estado"], index=0)
                
                df_agg = df_exp.groupby(agrupador)['caudal_lps'].sum().reset_index()
                df_agg = df_agg[df_agg['caudal_lps'] > 0]
                
                fig_exp = px.pie(df_agg, values='caudal_lps', names=agrupador, hole=0.4, title=f"Caudal total filtrado: {df_agg['caudal_lps'].sum():,.1f} L/s")
                fig_exp.update_traces(textposition='inside', textinfo='value+label')
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.warning("No hay caudal numérico para graficar con los filtros seleccionados.")
    else:
        st.error("No se detectó la base de datos de Concesiones SIRENA.")
