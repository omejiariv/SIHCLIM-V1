# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
import time
import json
import unicodedata
import re
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Modelo Demográfico Integral", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico Integral (Proyección y Dasimetría)")
st.markdown("Ajuste matemático, simulación animada, mapas espaciales y proyección top-down de estructuras poblacionales (1952-2100).")
st.divider()

# --- FUNCION MÁGICA 1: EL ASPIRADOR DE TEXTOS (Match infalible) ---
def normalizar_texto(texto):
    if pd.isna(texto): return ""
    t = str(texto).upper()
    t = re.sub(r'\(.*?\)', '', t)
    
    # --- NUEVO: DESTRUCTOR DE PREFIJOS VEREDALES ---
    t = t.replace("VDA.", "").replace("VDA ", "").replace("VEREDA ", "").replace("SECTOR ", "").replace("CGE.", "")
    
    # 1. HOTFIX: Reparador de Caracteres Mutantes del DANE (UTF-8 a Latin-1)
    reparador = {
        "Ã\x81": "A", "Ã": "A", "Ã\x89": "E", "Ã‰": "E",
        "Ã\x8d": "I", "Ã": "I", "Ã\x93": "O", "Ã“": "O",
        "Ã\x9a": "U", "Ãš": "U", "Ã\x91": "N", "Ã‘": "N",
        "Ãœ": "U", "Ã\x9c": "U", "Ã¡": "A", "Ã©": "E", 
        "Ã­": "I", "Ã³": "O", "Ãº": "U", "Ã±": "N", "Ã¼": "U"
    }
    for malo, bueno in reparador.items():
        t = t.replace(malo, bueno)
        
    # 2. Quitar tildes y dejar solo letras/números
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = re.sub(r'[^A-Z0-9]', '', t)
    
    # 3. Traductor DANE-IGAC Actualizado (Con los 9 nombres pomposos)
    diccionario_rebeldes = {
        "BOGOTADC": "BOGOTA", "SANJOSEDECUCUTA": "CUCUTA", "LAGUAJIRA": "GUAJIRA", 
        "VALLEDELCAUCA": "VALLE", "VILLADESANDIEGODEUBATE": "UBATE", "SANTIAGODETOLU": "TOLU",
        "ELPENOL": "PENOL", "ELRETIRO": "RETIRO", "ELSANTUARIO": "SANTUARIO",
        "ELCARMENDEVIBORAL": "CARMENDEVIBORAL", "SANVICENTEFERRER": "SANVICENTE",
        "PUEBLORRICO": "PUEBLORICO", "SANANDRESDECUERQUIA": "SANANDRES",
        "SANPEDRODELOSMILAGROS": "SANPEDRO", "BRICENO": "BRICEN0",
        "PIZARRO": "BAJOBAUDO", "DOCORDO": "ELLITORALDELSANJUAN", "LITORALDELSANJUAN": "ELLITORALDELSANJUAN",
        "BAHIASOLANO": "BAHIASOLANOMUTIS", "TUMACO": "SANANDRESDETUMACO", "PATIA": "PATIAELBORDO",
        "LOPEZDEMICAY": "LOPEZ", "MAGUI": "MAGUIPAYAN", "ROBERTOPAYAN": "ROBERTOPAYANSANJOSE",
        "MALLAMA": "MALLAMAPIEDRAANCHA", "CUASPUD": "CUASPUDCARLOSAMA", "ALTOBAUDO": "ALTOBAUDOPIEDEPATO",
        "OLAYAHERRERA": "OLAYAHERRERABOCASDESATINGA", "SANTACRUZ": "SANTACRUZGUACHAVEZ",
        "LOSANDES": "LOSANDESSOTOMAYOR", "FRANCISCOPIZARRO": "FRANCISCOPIZARROSALAHONDA",
        "MEDIOSANJUAN": "ELLITORALDELSANJUANDOCORDO", "ELCANTONDELSANPABLO": "ELCANTONDESANPABLOMANAGRU",
        "ATRATO": "ATRATOYUTO", "LEGUIZAMO": "PUERTOLEGUIZAMO", "BARRANCOMINAS": "BARRANCOMINA",
        "MAPIRIPANA": "PANAPANA", "MORICHAL": "MORICHALNUEVO", "SANANDRESSOTAVENTO": "SANANDRESDESOTAVENTO",
        "SANLUISDESINCE": "SINCE", "SANVICENTEDECHUCURI": "SANVICENTEDELCHUCURI", "ELCARMENDECHUCURI": "ELCARMEN",
        "ARIGUANI": "ARIGUANIELDIFICIL", "SANMIGUEL": "SANMIGUELLADORADA", "VILLADELEYVA": "VILLADELEIVA",
        "PURACE": "PURACECOCONUCO", "ELTABLONDEGOMEZ": "ELTABLON", "ARMERO": "ARMEROGUAYABAL",
        "COLON": "COLONGENOVA", "SANPEDRODECARTAGO": "SANPEDRODECARTAGOCARTAGO", "CERROSANANTONIO": "CERRODESANANTONIO",
        "ARBOLEDA": "ARBOLEDABERRUECOS", "ENCINO": "ELENCINO", "MACARAVITA": "MARACAVITA",
        "TUNUNGUA": "TUNUNGA", "LAMONTANITA": "MONTANITA", "ELPAUJIL": "PAUJIL", "VILLARICA": "VILLARRICA",
        "GUADALAJARADEBUGA": "BUGA",
        # --- LOS 9 NUEVOS REBELDES (Nombres pomposos del nuevo mapa) ---
        "CARTAGENA": "CARTAGENADEINDIAS",
        "PIENDAMO": "PIENDAMOTUNIA",
        "MARIQUITA": "SANSEBASTIANDEMARIQUITA",
        "TOLUVIEJO": "SANJOSEDETOLUVIEJO",
        "SOTARA": "SOTARAPAISPAMBA",
        "PURISIMA": "PURISIMADELACONCEPCION",
        "GUICAN": "GUICANDELASIERRA",
        "PAPUNAUACD": "PAPUNAHUA",
        "PAPUNAUA": "PAPUNAHUA"
    }
    
    if t in diccionario_rebeldes:
        t = diccionario_rebeldes[t]
        
    return t
    
# --- 1. LECTURA DE DATOS LIMPIOS Y VEREDALES ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

REGIONES_COL = {
    'Caribe': ['Atlántico', 'Bolívar', 'Cesar', 'Córdoba', 'La Guajira', 'Magdalena', 'Sucre', 'Archipiélago De San Andrés'],
    'Pacífica': ['Cauca', 'Chocó', 'Nariño', 'Valle Del Cauca'],
    'Andina': ['Antioquia', 'Boyacá', 'Caldas', 'Cundinamarca', 'Huila', 'Norte De Santander', 'Quindio', 'Risaralda', 'Santander', 'Tolima', 'Bogotá, D.C.'],
    'Orinoquía': ['Arauca', 'Casanare', 'Meta', 'Vichada'],
    'Amazonía': ['Amazonas', 'Caquetá', 'Guainía', 'Guaviare', 'Putumayo', 'Vaupés']
}

@st.cache_data
def cargar_datos_limpios():
    try:
        ruta_nac = os.path.join(RUTA_RAIZ, "data", "PobCol1912_2100.csv")
        df_nac = pd.read_csv(ruta_nac, sep=',')
        if len(df_nac.columns) < 2: df_nac = pd.read_csv(ruta_nac, sep=';')
        df_nac.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().title() for c in df_nac.columns]
        df_nac = df_nac.rename(columns={'Male': 'Hombres', 'Female': 'Mujeres', 'Ano': 'Año'})
        
        ruta_mun_1 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_Colombia.csv")
        ruta_mun_2 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_colombia.csv")
        if os.path.exists(ruta_mun_1): df_mun = pd.read_csv(ruta_mun_1, sep=',')
        elif os.path.exists(ruta_mun_2): df_mun = pd.read_csv(ruta_mun_2, sep=',')
        else: raise FileNotFoundError("Archivo municipal no encontrado.")
        if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_1 if os.path.exists(ruta_mun_1) else ruta_mun_2, sep=';')
            
        df_mun.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().lower() for c in df_mun.columns]
        df_mun = df_mun.rename(columns={'poblacion': 'Total'})
        df_mun['depto_nom'] = df_mun['depto_nom'].str.title()
        df_mun['municipio'] = df_mun['municipio'].str.title()
        
        # MAGIA 2: Asignar Región con excepciones (Urabá)
        def asignar_region(fila):
            uraba_caribe = ["TURBO", "NECOCLI", "SAN JUAN DE URABA", "ARBOLETES", "SAN PEDRO DE URABA", "APARTADO", "CAREPA", "CHIGORODO", "MUTATA"]
            if normalizar_texto(fila['municipio']) in [normalizar_texto(m) for m in uraba_caribe]:
                return 'Caribe'
                
            depto = fila['depto_nom']
            for region, deptos in REGIONES_COL.items():
                if depto in deptos: return region
            return "Sin Región"
            
        df_mun['Macroregion'] = df_mun.apply(asignar_region, axis=1)
        
        df_ver = pd.DataFrame()
        ruta_ver_1 = os.path.join(RUTA_RAIZ, "data", "veredas_Antioquia.csv")
        ruta_ver_2 = os.path.join(RUTA_RAIZ, "data", "veredas_Antioquia.xlsx")
        
        if os.path.exists(ruta_ver_1): df_ver = pd.read_csv(ruta_ver_1, sep=';')
        elif os.path.exists(ruta_ver_2): df_ver = pd.read_excel(ruta_ver_2)
            
        if not df_ver.empty:
            df_ver.columns = [str(c).replace('\ufeff', '').strip() for c in df_ver.columns]
            if 'Municipio' in df_ver.columns: df_ver['Municipio'] = df_ver['Municipio'].str.title()
            if 'Vereda' in df_ver.columns: df_ver['Vereda'] = df_ver['Vereda'].str.title()
            if 'Poblacion_hab' in df_ver.columns: df_ver['Poblacion_hab'] = pd.to_numeric(df_ver['Poblacion_hab'], errors='coerce').fillna(0)

        return df_nac, df_mun, df_ver
    except Exception as e:
        st.error(f"🚨 Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_nac, df_mun, df_ver = cargar_datos_limpios()
if df_nac.empty or df_mun.empty: st.stop()

# --- 2. MODELOS MATEMÁTICOS ---
def modelo_lineal(x, m, b): return m * x + b
def modelo_exponencial(x, p0, r): return p0 * np.exp(r * (x - 2005))
def modelo_logistico(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))

# --- 3. CONFIGURACIÓN Y FILTROS LATERALES ---
st.sidebar.header("⚙️ 1. Selección Territorial")
escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Regional (Macroregiones)", "Departamental", "Municipal", "Veredal (Antioquia)"])

años_hist, pob_hist = [], []
df_mapa_base = pd.DataFrame()

if escala_sel == "Nacional":
    df_agrup_nac = df_nac.groupby('Año')[['Hombres', 'Mujeres']].sum().reset_index()
    años_hist = df_agrup_nac['Año'].values
    pob_hist = (df_agrup_nac['Hombres'] + df_agrup_nac['Mujeres']).values
    titulo_terr = "Colombia (Nacional)"
    df_mapa_base = df_mun.groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel == "Regional (Macroregiones)":
    regiones_list = sorted(df_mun['Macroregion'].unique())
    reg_sel = st.sidebar.selectbox("Seleccione la Macroregión", regiones_list)
    df_terr = df_mun[df_mun['Macroregion'] == reg_sel].groupby('año')['Total'].sum().reset_index()
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values
    titulo_terr = f"Región {reg_sel}"
    df_mapa_base = df_mun[df_mun['Macroregion'] == reg_sel].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel in ["Departamental", "Municipal"]:
    deptos = sorted(df_mun['depto_nom'].dropna().unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental":
        df_terr = df_mun[df_mun['depto_nom'] == depto_sel].groupby('año')['Total'].sum().reset_index()
        titulo_terr = depto_sel
        df_mapa_base = df_mun[df_mun['depto_nom'] == depto_sel].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
        df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].dropna().unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        df_terr = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby('año')['Total'].sum().reset_index()
        titulo_terr = f"{mpio_sel}, {depto_sel}"
        df_mapa_base = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
        df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})
        
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values

elif escala_sel == "Veredal (Antioquia)":
    # 1. Agregamos la opción "TODOS" al principio de la lista
    mpios_veredas = ["TODOS (Ver Mapa Completo)"] + sorted(df_ver['Municipio'].dropna().unique())
    mpio_sel = st.sidebar.selectbox("Municipio (Antioquia)", mpios_veredas)
    
    if mpio_sel == "TODOS (Ver Mapa Completo)":
        # Sumamos toda la población rural de Antioquia como base histórica general
        df_rural_ant = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['area_geografica'] == 'rural')]
        df_hist_rural = df_rural_ant.groupby('año')['Total'].sum().reset_index()
        
        años_hist = df_hist_rural['año'].values
        pob_hist = df_hist_rural['Total'].values
        titulo_terr = "Todas las Veredas (Región Disponible)"
        
        # Inyectamos la tabla completa de veredas al mapa
        df_mapa_base = df_ver.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'
        
    else:
        # El código original para un solo municipio
        veredas_lista = sorted(df_ver[df_ver['Municipio'] == mpio_sel]['Vereda'].dropna().unique())
        vereda_sel = st.sidebar.selectbox("Vereda", veredas_lista)
        
        df_rural_mpio = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['municipio'] == mpio_sel) & (df_mun['area_geografica'] == 'rural')]
        df_hist_rural = df_rural_mpio.groupby('año')['Total'].sum().reset_index()
        
        df_mpio_veredas = df_ver[df_ver['Municipio'] == mpio_sel]
        pob_total_veredas = df_mpio_veredas['Poblacion_hab'].sum()
        pob_ver_especifica = df_mpio_veredas[df_mpio_veredas['Vereda'] == vereda_sel]['Poblacion_hab'].sum()
        
        ratio_vereda = pob_ver_especifica / pob_total_veredas if pob_total_veredas > 0 else 0
        
        años_hist = df_hist_rural['año'].values
        pob_hist = df_hist_rural['Total'].values * ratio_vereda
        titulo_terr = f"Vereda {vereda_sel} ({mpio_sel})"
        
        df_mapa_base = df_mpio_veredas.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'
        
# --- 4. CÁLCULO DE PROYECCIONES ---
x_hist = np.array(años_hist, dtype=float)
y_hist = np.array(pob_hist, dtype=float)

año_maximo = int(max(df_nac['Año'].max(), 2100))
x_proj = np.arange(1950, año_maximo + 1, 1) 

proyecciones = {'Año': x_proj, 'Real': [np.nan]*len(x_proj)}
for i, año in enumerate(x_proj):
    if año in x_hist: proyecciones['Real'][i] = y_hist[np.where(x_hist == año)[0][0]]

try:
    popt_lin, _ = curve_fit(modelo_lineal, x_hist, y_hist)
    proyecciones['Lineal'] = np.maximum(0, modelo_lineal(x_proj, *popt_lin))
except: proyecciones['Lineal'] = [np.nan]*len(x_proj)

try:
    popt_exp, _ = curve_fit(modelo_exponencial, x_hist, y_hist, p0=[max(1, y_hist[0]), 0.01], maxfev=10000)
    proyecciones['Exponencial'] = modelo_exponencial(x_proj, *popt_exp)
except: proyecciones['Exponencial'] = proyecciones['Lineal']

try:
    K_guess = max(1, max(y_hist)) * 1.5
    popt_log, _ = curve_fit(modelo_logistico, x_hist, y_hist, p0=[K_guess, 0.05, 2020], maxfev=10000)
    proyecciones['Logístico'] = modelo_logistico(x_proj, *popt_log)
    param_K = popt_log[0]
except: 
    proyecciones['Logístico'] = proyecciones['Lineal']
    param_K = "N/A"

df_proj = pd.DataFrame(proyecciones)

# --- CONFIGURACIÓN DE PESTAÑAS (TABS) ---
tab_modelos, tab_mapas, tab_descargas = st.tabs(["📊 Pirámides y Modelos", "🌍 Geovisor Espacial", "💾 Descarga de Datos"])

# ==========================================
# PESTAÑA 1: MODELOS Y PIRÁMIDES ANIMADAS
# ==========================================
with tab_modelos:
    col_graf, col_param = st.columns([3, 1])
    with col_graf:
        st.subheader(f"📈 Curvas de Crecimiento Poblacional - {titulo_terr}")
        fig_curvas = go.Figure()
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Logístico'], mode='lines', name='Mod. Logístico', line=dict(color='#10b981', dash='dash')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Exponencial'], mode='lines', name='Mod. Exponencial', line=dict(color='#f59e0b', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Lineal'], mode='lines', name='Mod. Lineal', line=dict(color='#6366f1', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Reales (Censo)', marker=dict(color='#ef4444', size=8, symbol='diamond')))
        fig_curvas.update_layout(hovermode="x unified", xaxis_title="Año", yaxis_title="Habitantes", template="plotly_white")
        st.plotly_chart(fig_curvas, width='stretch')

    with col_param:
        st.subheader("🧮 Ecuaciones")
        st.latex(r"Log: P(t) = \frac{K}{1 + e^{-r(t - t_0)}}")
        if param_K != "N/A": st.success(f"**K:** {param_K:,.0f} hab.")
        st.latex(r"Exp: P(t) = P_0 \cdot e^{r(t - t_0)}")
        st.latex(r"Lin: P(t) = m \cdot t + b")

    st.divider()

    st.sidebar.header("🎯 2. Viaje en el Tiempo")
    modelo_sel = st.sidebar.radio("Base de cálculo para la pirámide:", ["Logístico", "Exponencial", "Lineal", "Dato Real (Si existe)"])
    columna_modelo = "Real" if modelo_sel == "Dato Real (Si existe)" else modelo_sel

    años_disp = sorted(df_nac['Año'].unique())
    año_sel = st.sidebar.select_slider("Selecciona un Año Estático:", options=años_disp, value=2024)

    st.sidebar.markdown("---")
    st.sidebar.subheader("▶️ Animación Temporal")
    velocidad_animacion = st.sidebar.slider("Velocidad (Segundos por año)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    iniciar_animacion = st.sidebar.button("▶️ Reproducir Evolución", type="primary", use_container_width=True)

    st.subheader(f"Estructura Poblacional Sintética - {titulo_terr}")
    ph_titulo_año = st.empty()

    col_pir1, col_pir2 = st.columns([2, 1])
    with col_pir1: ph_grafico_pir = st.empty()
    with col_pir2:
        st.markdown("### Resumen del Perfil")
        ph_metrica_pob = st.empty()
        ph_metrica_hom = st.empty()
        ph_metrica_muj = st.empty()
        ph_metrica_ind = st.empty()

    df_piramide_final = pd.DataFrame() 

    def renderizar_piramide(año_obj):
        global df_piramide_final
        try: pob_modelo = df_proj[df_proj['Año'] == año_obj][columna_modelo].values[0]
        except: pob_modelo = np.nan
            
        if pd.isna(pob_modelo): pob_modelo = df_proj[df_proj['Año'] == año_obj]['Logístico'].values[0]
        
        df_fnac = df_nac[df_nac['Año'] == año_obj].copy()
        pob_nacional_tot = df_fnac['Hombres'].sum() + df_fnac['Mujeres'].sum()
        df_fnac['Prop_H'] = df_fnac['Hombres'] / pob_nacional_tot
        df_fnac['Prop_M'] = df_fnac['Mujeres'] / pob_nacional_tot
        
        df_fnac['Hom_Terr'] = df_fnac['Prop_H'] * pob_modelo
        df_fnac['Muj_Terr'] = df_fnac['Prop_M'] * pob_modelo
        
        df_pir = pd.DataFrame({'Edad': df_fnac['Edad'], 'Hombres': df_fnac['Hom_Terr'] * -1, 'Mujeres': df_fnac['Muj_Terr']})
        cortes = list(range(0, 105, 5))
        etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)]
        df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
        df_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()
        
        df_melt = pd.melt(df_agrupado, id_vars=['Rango'], value_vars=['Hombres', 'Mujeres'], var_name='Sexo', value_name='Poblacion')
        df_piramide_final = df_melt.copy()
        
        fig_pir = px.bar(df_melt, y='Rango', x='Poblacion', color='Sexo', orientation='h', color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'})
        
        max_x = max(abs(df_melt['Poblacion'].min()), df_melt['Poblacion'].max()) * 1.1
        fig_pir.update_layout(barmode='relative', xaxis_title="Habitantes", yaxis_title="Edades", xaxis=dict(range=[-max_x, max_x]))
        fig_pir.update_traces(hovertemplate="%{y}: %{x:,.0f}")
        
        ph_titulo_año.markdown(f"### ⏳ **Año Proyectado: {año_obj}**")
        ph_grafico_pir.plotly_chart(fig_pir, width='stretch')
        
        tot_h, tot_m = df_fnac['Hom_Terr'].sum(), df_fnac['Muj_Terr'].sum()
        ind_m = (tot_h / tot_m) * 100 if tot_m > 0 else 0
        
        ph_metrica_pob.metric("Población Proyectada", f"{pob_modelo:,.0f}")
        ph_metrica_hom.metric("Total Hombres", f"{tot_h:,.0f}")
        ph_metrica_muj.metric("Total Mujeres", f"{tot_m:,.0f}")
        ph_metrica_ind.metric("Índice Masculinidad", f"{ind_m:.1f} H x 100 M")

    if iniciar_animacion:
        for a in años_disp:
            if a >= 1950:
                renderizar_piramide(a)
                time.sleep(velocidad_animacion)
    else:
        renderizar_piramide(año_sel)
        
# --- 7. MARCO METODOLÓGICO Y CONCEPTUAL ---
with st.expander("📚 Marco Conceptual, Metodológico y Matemático", expanded=False):
    st.markdown("""
    ### 1. Conceptos Teóricos
    La dinámica demográfica es el motor fundamental de la planificación hídrica y territorial. Conocer no solo *cuántos* somos, sino la *estructura por edades*, permite proyectar demandas futuras de acueductos, escenarios de presión sobre el recurso hídrico y necesidades de infraestructura.

    ### 2. Metodología de Mapeo Dasimétrico y Asignación Top-Down
    Ante la falta de censos poblacionales continuos en micro-territorios (como veredas), este modelo utiliza una **Estimación Sintética Anidada**:
    * **Paso 1 (Calibración):** Se utiliza la serie censal DANE a nivel municipal (Urbano/Rural) entre 2005 y 2020.
    * **Paso 2 (Dasimetría Veredal):** Se calcula el peso gravitacional de la población de cada vereda respecto a la población rural total de su municipio, asumiendo proporcionalidad espacial ($P_{vereda} = P_{rural\_mpio} \\times \\left( \\frac{P_{base\_vereda}}{\\sum P_{base\_veredas}} \\right)$).
    * **Paso 3 (Anidación Estructural):** Se aplica la "receta" porcentual de la pirámide de edades nacional del año correspondiente, a la masa poblacional calculada del micro-territorio.

    ### 3. Modelos Matemáticos de Ajuste Histórico
    Para viajar en el tiempo (1950-2100), la serie histórica se somete a regresiones no lineales (`scipy.optimize`):
    * **Logístico:** Modela ecosistemas limitados. La población crece hasta encontrar resistencia ambiental, estabilizándose en una *Capacidad de Carga* ($K$). Es el modelo más robusto para planeación a largo plazo.
    * **Exponencial:** Asume recursos infinitos. Útil para modelar cortos períodos de "explosión demográfica" en centros urbanos nuevos.
    * **Lineal:** Representa tendencias promedio sin aceleración.

    ### 4. Fuentes de Información
    * **Capa 1 (Estructura Nacional):** Proyecciones y retroproyecciones oficiales DANE (1950-2070).
    * **Capa 2 (Masa Municipal):** Series censales DANE conciliadas (2005-2020).
    * **Capa 3 (Filtro Veredal):** Base de datos espaciales y tabulares Gobernación de Antioquia / IGAC.
    """)

# ==========================================
# PESTAÑA 2: MAPA DEMOGRÁFICO (GEOVISOR)
# ==========================================
with tab_mapas:
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_terr} ({año_sel})")
    
    if escala_sel != "Veredal (Antioquia)":
        area_mapa = st.radio("Filtro de Zona Geográfica:", ["Total", "Urbano", "Rural"], horizontal=True)
    else:
        area_mapa = "Rural"
        st.info("ℹ️ A escala veredal, toda la población es considerada rural.")

    if 'año' in df_mapa_base.columns:
        df_mapa_año = df_mapa_base[df_mapa_base['año'] == min(max(df_mapa_base['año']), año_sel)].copy()
    else:
        df_mapa_año = df_mapa_base.copy()

    if area_mapa == "Total":
        df_mapa_plot = df_mapa_año.groupby(['Territorio', 'Padre'])['Total'].sum().reset_index()
    else:
        df_mapa_plot = df_mapa_año[df_mapa_año['area_geografica'] == area_mapa.lower()].copy()

    col_map1, col_map2 = st.columns([1, 3])
    
    with col_map1:
        st.markdown("**⚙️ Configuración del GeoJSON**")
        if escala_sel == "Veredal (Antioquia)": 
            sugerencia_geo = "Veredas_Antioquia_Oficial_DANE.geojson"
            sugerencia_prop = "properties.NOMBRE_VER"
            sugerencia_padre = "properties.MPIO_CNMBR"
        else: 
            sugerencia_geo = "mgn_municipios_optimizado.geojson" # Asegúrate de que este sea tu nombre de archivo
            sugerencia_prop = "properties.MPIO_CNMBR"
            sugerencia_padre = "properties.DPTO_CCDGO" # AHORA USAMOS EL CÓDIGO DANE
            
        archivo_geo_input = st.text_input("Archivo en GitHub:", value=sugerencia_geo)
        prop_geo_input = st.text_input("Llave Territorio:", value=sugerencia_prop)
        st.markdown("**🔗 Llave Doble (Anti-Homonimia)**")
        prop_padre_input = st.text_input("Llave Contexto:", value=sugerencia_padre)

    with col_map2:
        ruta_geo = os.path.join(RUTA_RAIZ, "data", archivo_geo_input)
        
        if os.path.exists(ruta_geo) and not df_mapa_plot.empty:
            try:
                # 1. Crear ADN Único en Pandas
                if prop_padre_input.strip() != "":
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot['Territorio'].apply(normalizar_texto) + "_" + df_mapa_plot['Padre'].apply(normalizar_texto)
                else:
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot['Territorio'].apply(normalizar_texto)

                with open(ruta_geo, encoding='utf-8') as f:
                    geo_data = json.load(f)
                
                # --- TRADUCTOR DANE DE CÓDIGOS A NOMBRES ---
                codigos_dane_deptos = {
                    "05": "ANTIOQUIA", "08": "ATLANTICO", "11": "BOGOTA", "13": "BOLIVAR", 
                    "15": "BOYACA", "17": "CALDAS", "18": "CAQUETA", "19": "CAUCA", 
                    "20": "CESAR", "23": "CORDOBA", "25": "CUNDINAMARCA", "27": "CHOCO", 
                    "41": "HUILA", "44": "GUAJIRA", "47": "MAGDALENA", "50": "META", 
                    "52": "NARINO", "54": "NORTEDESANTANDER", "63": "QUINDIO", "66": "RISARALDA", 
                    "68": "SANTANDER", "70": "SUCRE", "73": "TOLIMA", "76": "VALLE", 
                    "81": "ARAUCA", "85": "CASANARE", "86": "PUTUMAYO", "88": "ARCHIPIELAGODESANANDRES", 
                    "91": "AMAZONAS", "94": "GUAINIA", "95": "GUAVIARE", "97": "VAUPES", "99": "VICHADA"
                }
                
                # 2. Crear ADN Único en el GeoJSON
                prop_key = prop_geo_input.replace("properties.", "")
                padre_key = prop_padre_input.replace("properties.", "") if prop_padre_input.strip() else ""
                
                for feature in geo_data['features']:
                    val_terr = feature['properties'].get(prop_key, "")
                    val_padre = str(feature['properties'].get(padre_key, "")) if padre_key else ""
                    
                    # Fix caracteres mutantes desde la fuente
                    val_terr = normalizar_texto(val_terr)
                    
                    if val_padre.zfill(2) in codigos_dane_deptos:
                        val_padre = codigos_dane_deptos[val_padre.zfill(2)]
                    
                    # Fix espacial: Para distinguir los dos Manaures (El del Cesar es Balcón del Cesar)
                    if val_terr == "MANAUREBALCONDELCESAR": val_terr = "MANAURE"
                    
                    if val_padre:
                        feature['properties']['MATCH_ID'] = val_terr + "_" + normalizar_texto(val_padre)
                    else:
                        feature['properties']['MATCH_ID'] = val_terr
                        
                q_val = 0.85 if area_mapa == "Total" else 0.90
                max_color = df_mapa_plot['Total'].quantile(q_val) if len(df_mapa_plot) > 10 else df_mapa_plot['Total'].max()
                
                # 3. Renderizar Mapa
                fig_mapa = px.choropleth_mapbox(
                    df_mapa_plot,
                    geojson=geo_data,
                    locations='MATCH_ID',        
                    featureidkey='properties.MATCH_ID', 
                    color='Total',
                    color_continuous_scale="Viridis",
                    range_color=[0, max_color],  
                    mapbox_style="carto-positron",
                    zoom=5 if escala_sel != "Veredal (Antioquia)" else 9, 
                    center={"lat": 4.57, "lon": -74.29} if escala_sel != "Veredal (Antioquia)" else {"lat": 6.25, "lon": -75.56},
                    opacity=0.8,
                    labels={'Total': f'Población {area_mapa}'},
                    hover_data={'Total': ':,.0f', 'MATCH_ID': False, 'Territorio': True, 'Padre': True}
                )
                fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_mapa, width='stretch')
                
                geo_ids_disp = [f['properties'].get('MATCH_ID', '') for f in geo_data['features']]
                df_mapa_plot['En_Mapa'] = df_mapa_plot['MATCH_ID'].isin(geo_ids_disp)
                faltantes = df_mapa_plot[df_mapa_plot['En_Mapa'] == False]
                
                if not faltantes.empty:
                    st.warning(f"⚠️ {len(faltantes)} territorios de la tabla no cruzaron con el GeoJSON.")
                    
                    # --- NUEVO: REVELADOR DE VEREDAS (RAYOS X) ---
                    if escala_sel == "Veredal (Antioquia)":
                        with st.expander("🔍 Ver nombres exactos dentro del mapa GeoJSON (Para arreglar el Excel)"):
                            municipio_actual = normalizar_texto(df_mapa_plot['Padre'].iloc[0]) if not df_mapa_plot.empty else ""
                            
                            # Extraer las veredas que SÍ están dibujadas en el mapa para este municipio
                            veredas_en_mapa = []
                            for f in geo_data['features']:
                                m_padre = str(f['properties'].get(padre_key, ""))
                                if normalizar_texto(m_padre) == municipio_actual:
                                    veredas_en_mapa.append(f['properties'].get(prop_key, ""))
                                    
                            if veredas_en_mapa:
                                st.write(f"El mapa tiene **{len(veredas_en_mapa)}** polígonos para este municipio. Estos son sus nombres reales:")
                                st.dataframe(pd.DataFrame({"Nombres exactos en el GeoJSON": sorted(veredas_en_mapa)}), use_container_width=True)
                                st.info("💡 **Solución:** Busca tu vereda en esta lista. Si en tu Excel dice 'La Loma' pero aquí dice 'Vda. Loma', simplemente actualiza el nombre en tu Excel o añádelo al diccionario del código.")
                            else:
                                st.error("No se encontró ningún polígono para este municipio en el archivo GeoJSON. Verifica que la 'Llave Contexto' sea correcta.")
                
            except Exception as e:
                st.error(f"❌ Error dibujando mapa: {e}")
        else:
            st.warning(f"⚠️ No se encontró **{archivo_geo_input}** o no hay datos para esta vista.")
            
    st.dataframe(df_mapa_plot[['Territorio', 'Padre', 'Total', 'MATCH_ID', 'En_Mapa']].sort_values('Total', ascending=False), use_container_width=True)

# ==========================================
# PESTAÑA 3: DESCARGAS Y EXPORTACIÓN
# ==========================================
with tab_descargas:
    st.subheader("💾 Exportación de Resultados y Series de Tiempo")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("### 📈 Curvas de Proyección (1950-2100)")
        csv_proj = df_proj.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Descargar Proyecciones (CSV)", data=csv_proj, file_name=f"Proyecciones_{titulo_terr}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(df_proj.dropna(subset=['Real']).head(5), use_container_width=True)

    with col_d2:
        st.markdown(f"### 📊 Pirámide Sintética ({año_sel})")
        df_descarga_pir = df_piramide_final.copy()
        if not df_descarga_pir.empty:
            df_descarga_pir['Poblacion'] = df_descarga_pir['Poblacion'].abs()
            csv_pir = df_descarga_pir.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"⬇️ Descargar Pirámide {año_sel} (CSV)", data=csv_pir, file_name=f"Piramide_{año_sel}_{titulo_terr}.csv", mime="text/csv", use_container_width=True)
            st.dataframe(df_descarga_pir.head(5), use_container_width=True)
