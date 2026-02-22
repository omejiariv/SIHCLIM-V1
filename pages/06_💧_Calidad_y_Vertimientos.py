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
# 🔌 CONECTOR A BASES DE DATOS
# ==============================================================================
def leer_csv_robusto(ruta):
    try:
        # Forzamos lectura con punto y coma primero para el nuevo archivo
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
    ruta = "data/Concesiones_Corantioquia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty:
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
            
            # Autodetectar columnas de forma segura para el NUEVO archivo
            col_caudal = 'caudal_por_uso' if 'caudal_por_uso' in df.columns else ('caudal_usuario' if 'caudal_usuario' in df.columns else None)
            if not col_caudal: # Fallback al archivo viejo
                cands = [c for c in df.columns if 'caudal' in c and 'acumulado' not in c]
                col_caudal = cands[0] if cands else None
                
            col_uso = 'uso' if 'uso' in df.columns else None
            col_mpio = 'municipio' if 'municipio' in df.columns else None
            col_vereda = 'vereda' if 'vereda' in df.columns else None
            col_asunto = 'asunto' if 'asunto' in df.columns else None
            col_cota = 'cota' if 'cota' in df.columns else None
            col_estado = 'estado' if 'estado' in df.columns else None
            
            # Solo exigimos Municipio y Caudal para no perder datos
            if col_caudal and col_mpio:
                df = df.dropna(subset=[col_mpio]).copy() # No borramos por 'uso' vacío
                
                # Conversión de caudales
                if df[col_caudal].dtype == object:
                    df[col_caudal] = df[col_caudal].astype(str).str.replace(',', '.')
                df['caudal_lps'] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
                
                # Cota a numérico
                if col_cota:
                    df['cota_num'] = pd.to_numeric(df[col_cota], errors='coerce').fillna(-1)
                else:
                    df['cota_num'] = -1
                
                # Normalización
                df['municipio'] = df[col_mpio].astype(str).str.strip().str.title()
                df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
                
                if col_vereda: 
                    df['vereda'] = df[col_vereda].astype(str).str.strip().str.title()
                    df['vereda_norm'] = df['vereda'].apply(normalizar_texto)
                else:
                    df['vereda_norm'] = ""
                
                # FILTRO SUBTERRÁNEO VS SUPERFICIAL (Regex Inteligente)
                if col_asunto:
                    df['tipo_agua'] = np.where(df[col_asunto].str.lower().str.contains('subterran|subterrán|pozo|aljibe', regex=True, na=False), 'Subterránea',
                                      np.where(df[col_asunto].str.lower().str.contains('superficial|corriente', regex=True, na=False), 'Superficial', 'No Especificado'))
                else:
                    df['tipo_agua'] = 'No Especificado'

                # Rellenar usos nulos
                if col_uso:
                    df['uso_detalle'] = df[col_uso].fillna('Sin Información').astype(str).str.title().str.strip()
                else:
                    df['uso_detalle'] = 'Sin Información'

                def clasificar_uso_base(u):
                    u = normalizar_texto(u)
                    if any(x in u for x in ['domestico', 'consumo humano', 'abastecimiento', 'acueducto']): return 'Doméstico'
                    elif any(x in u for x in ['agricola', 'pecuario', 'acuicultura', 'agroindustrial', 'riego', 'piscicola', 'silvicultura']): return 'Agrícola/Pecuario'
                    elif any(x in u for x in ['industrial', 'mineria', 'minero', 'generacion de energia']): return 'Industrial'
                    else: return 'Otros'
                    
                df['Sector_Sihcli'] = df['uso_detalle'].apply(clasificar_uso_base)
                
                if col_estado:
                    df['estado'] = df[col_estado].fillna('Desconocido').astype(str).str.title().str.strip()
                else:
                    df['estado'] = 'Desconocido'
                    
                return df
    return pd.DataFrame()

df_mpios = cargar_municipios()
df_veredas = cargar_veredas()
df_concesiones = cargar_concesiones()

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

st.info(f"👥 Demografía proyectada para **{lugar_sel}** en el año **{anio_analisis}**:")
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
tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion, tab_sirena = st.tabs([
    "🚰 2. Demanda y Subregistro",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios",
    "📊 6. Explorador SIRENA"
])

anios_evo = np.arange(anio_analisis, anio_analisis + 31)
factor_evo = proyectar_curva(pob_t_base, anios_evo, anio_base, modelo_sel, tasa_r, k_man) / pob_t_base if pob_t_base > 0 else np.ones_like(anios_evo)
pob_evo = pob_total * (factor_evo / factor_proy)

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA (SUBREGISTRO Y DIAGNÓSTICO INTELIGENTE)
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header(f"🚰 Metabolismo Hídrico y Formalización")
    col_d1, col_d2 = st.columns([1, 1.5])
    
    with col_d1:
        st.subheader("1. Demanda Teórica")
        dotacion = st.number_input("Dotación Doméstica (L/hab/día):", value=120.0, step=5.0)
        q_teorico_dom = (pob_total * dotacion) / 86400
        st.metric(f"Caudal Doméstico Necesario", f"{q_teorico_dom:.2f} L/s")
        
        st.markdown("---")
        st.subheader("2. Demanda Legal (Autorizada)")
        
        q_sup, q_sub, q_legal_agr, q_legal_ind = 0.0, 0.0, 0.0, 0.0
        df_usos_detalle = pd.DataFrame()
        
        if not df_concesiones.empty and lugar_sel != "N/A":
            lugar_norm = normalizar_texto(lugar_sel)
            
            if nivel_sel == "Municipal": df_filtro_c = df_concesiones[df_concesiones['municipio_norm'] == lugar_norm]
            elif nivel_sel == "Veredal" and 'vereda_norm' in df_concesiones.columns: df_filtro_c = df_concesiones[df_concesiones['vereda_norm'] == lugar_norm]
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
        
        st.caption(f"Caudales formales extraídos de SIRENA para **{lugar_sel}**:")
        st.write(f"- **Superficial:** {q_sup:,.2f} L/s")
        st.write(f"- **Subterráneo:** {q_sub:,.2f} L/s")
        st.write(f"- **Total Legal Doméstico:** {q_concesionado_dom:,.2f} L/s")
        
    with col_d2:
        st.subheader("📊 Análisis de Ilegalidad o Subregistro")
        
        # CAJA DIAGNÓSTICA INTELIGENTE
        margen = 0.05 
        if q_concesionado_dom > q_teorico_dom * (1 + margen):
            st.error(f"🔴 **Sobreconcesión:** Se han otorgado {q_concesionado_dom - q_teorico_dom:,.1f} L/s por encima del requerimiento teórico poblacional.")
        elif q_concesionado_dom < q_teorico_dom * (1 - margen):
            st.warning(f"⚠️ **Riesgo de Ilegalidad / Subregistro:** La población necesita {q_teorico_dom - q_concesionado_dom:,.1f} L/s adicionales que no aparecen formalizados.")
        else:
            st.success(f"✅ **Equilibrio Hídrico:** El caudal otorgado ({q_concesionado_dom:,.1f} L/s) suple adecuadamente la demanda poblacional ({q_teorico_dom:,.1f} L/s).")

        df_chart = pd.DataFrame([
            {"Categoría": "Requerimiento Teórico", "Tipo": "Consumo Real Estimado", "Caudal (L/s)": q_teorico_dom},
            {"Categoría": "Registro SIRENA", "Tipo": "Concesión Superficial", "Caudal (L/s)": q_sup},
            {"Categoría": "Registro SIRENA", "Tipo": "Concesión Subterránea", "Caudal (L/s)": q_sub}
        ])
        
        fig_sub = px.bar(df_chart, x="Categoría", y="Caudal (L/s)", color="Tipo", 
                         color_discrete_map={"Consumo Real Estimado": "#e74c3c", "Concesión Superficial": "#3498db", "Concesión Subterránea": "#2ecc71"},
                         title="Comparativa: Realidad Demográfica vs Formalización")
        fig_sub.add_hline(y=q_teorico_dom, line_dash="dash", line_color="black", annotation_text="Brecha de Subregistro")
        st.plotly_chart(fig_sub, use_container_width=True)
        
    st.divider()
    st.subheader("📋 Consolidado de Todos los Usos Registrados (SIRENA)")
    if not df_usos_detalle.empty:
        col_t1, col_t2 = st.columns([2,1])
        with col_t1: st.dataframe(df_usos_detalle, use_container_width=True)
        with col_t2:
            csv = df_usos_detalle.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar Desglose (CSV)", data=csv, file_name=f'Usos_SIRENA_{lugar_sel}.csv', mime='text/csv')
    else:
        st.warning("No hay registros de concesiones para esta unidad territorial (o los nombres no coinciden).")

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
    q_efluente_lps = (q_teorico_dom * coef_retorno) + (q_legal_ind * 0.8) + (vol_suero / 86400)
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
        st.subheader("Impacto Aguas Abajo")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = c_mix, title = {'text': "DBO5 Final (mg/L)", 'font': {'size': 20}},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {'axis': {'range': [None, max(20, c_mix + 5)]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 3], 'color': "#2ecc71"}, {'range': [3, 5], 'color': "#f1c40f"}, {'range': [5, 10], 'color': "#e67e22"}, {'range': [10, 100], 'color': "#e74c3c"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5.0}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Intervenciones")
    st.info("Próxima fase de desarrollo.")

# ------------------------------------------------------------------------------
# TAB 5: EXPLORADOR SIRENA (Data Mining Avanzado)
# ------------------------------------------------------------------------------
with tab_sirena:
    st.header("📊 Explorador Avanzado de Concesiones (SIRENA)")
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
        
        # SLIDER DE COTA
        if 'cota_num' in df_exp.columns and df_exp['cota_num'].max() > 0:
            df_exp_valid_cota = df_exp[df_exp['cota_num'] >= 0]
            if not df_exp_valid_cota.empty:
                max_cota = float(df_exp_valid_cota['cota_num'].max())
                st.caption("Filtro de Elevación Topográfica:")
                rango_cota = st.slider("Rango de Cota (m.s.n.m):", 0.0, max_cota, (0.0, max_cota))
                df_exp = df_exp[((df_exp['cota_num'] >= rango_cota[0]) & (df_exp['cota_num'] <= rango_cota[1])) | (df_exp['cota_num'] == -1)]

        # Filtros
        if f_estado: df_exp = df_exp[df_exp['estado'].isin(f_estado)]
        if f_tipo: df_exp = df_exp[df_exp['tipo_agua'].isin(f_tipo)]
        if f_uso: df_exp = df_exp[df_exp['uso_detalle'].isin(f_uso)]
        if f_mpio: df_exp = df_exp[df_exp['municipio'].isin(f_mpio)]
        
        st.divider()
        c_exp1, c_exp2 = st.columns([2, 1.5])
        with c_exp1:
            st.subheader(f"Registros Encontrados: {len(df_exp)}")
            st.dataframe(df_exp, use_container_width=True)
            csv_exp = df_exp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exportar Resultados (CSV)", data=csv_exp, file_name="Reporte_SIRENA.csv", mime="text/csv")
            
        with c_exp2:
            st.subheader("Distribución de Caudales Netos")
            if not df_exp.empty and df_exp['caudal_lps'].sum() > 0:
                agrupador = st.selectbox("Agrupar gráfico por:", ["tipo_agua", "Sector_Sihcli", "uso_detalle", "municipio", "estado"], index=0)
                
                df_agg = df_exp.groupby(agrupador)['caudal_lps'].sum().reset_index()
                df_agg = df_agg[df_agg['caudal_lps'] > 0]
                
                fig_exp = px.pie(df_agg, values='caudal_lps', names=agrupador, hole=0.4, title=f"Caudal total filtrado: {df_agg['caudal_lps'].sum():,.1f} L/s")
                # Gráfico muestra Valores, NO porcentajes
                fig_exp.update_traces(textposition='inside', textinfo='value+label')
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.warning("No hay caudal numérico para graficar con los filtros seleccionados.")
    else:
        st.error("No se detectó la base de datos de Concesiones SIRENA.")
