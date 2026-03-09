# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Modelo Demográfico Integral", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico Integral (Proyección y Dasimetría)")
st.markdown("Ajuste de modelos matemáticos (Logístico, Exponencial, Lineal) y proyección top-down de estructuras poblacionales (1952-2070).")
st.divider()

# --- 1. LECTURA DE DATOS LIMPIOS ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
        
        if os.path.exists(ruta_mun_1):
            df_mun = pd.read_csv(ruta_mun_1, sep=',')
            if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_1, sep=';')
        elif os.path.exists(ruta_mun_2):
            df_mun = pd.read_csv(ruta_mun_2, sep=',')
            if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_2, sep=';')
        else:
            raise FileNotFoundError("Archivo municipal no encontrado.")
            
        df_mun.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().lower() for c in df_mun.columns]
        df_mun = df_mun.rename(columns={'poblacion': 'Total'})
        
        return df_nac, df_mun
    except Exception as e:
        st.error(f"🚨 Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_nac, df_mun = cargar_datos_limpios()
if df_nac.empty or df_mun.empty: st.stop()

# --- 2. MODELOS MATEMÁTICOS ---
def modelo_lineal(x, m, b): 
    return m * x + b

def modelo_exponencial(x, p0, r): 
    return p0 * np.exp(r * (x - 2005))

def modelo_logistico(x, K, r, x0): 
    return K / (1 + np.exp(-r * (x - x0)))

# --- 3. CONFIGURACIÓN Y FILTROS ---
st.sidebar.header("⚙️ 1. Selección Territorial")
escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Departamental", "Municipal"])

# Obtener la serie de tiempo real del territorio seleccionado
años_hist = []
pob_hist = []

if escala_sel == "Nacional":
    df_agrup_nac = df_nac.groupby('Año')[['Hombres', 'Mujeres']].sum().reset_index()
    df_agrup_nac['Total'] = df_agrup_nac['Hombres'] + df_agrup_nac['Mujeres']
    años_hist = df_agrup_nac['Año'].values
    pob_hist = df_agrup_nac['Total'].values
    titulo_terr = "Colombia (Nacional)"
else:
    deptos = sorted(df_mun['depto_nom'].unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental":
        df_terr = df_mun[df_mun['depto_nom'] == depto_sel].groupby('año')['Total'].sum().reset_index()
        titulo_terr = depto_sel
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        df_terr = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby('año')['Total'].sum().reset_index()
        titulo_terr = f"{mpio_sel}, {depto_sel}"
        
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values

# --- 4. AJUSTE DE CURVAS (REGRESIÓN) ---
x_hist = np.array(años_hist, dtype=float)
y_hist = np.array(pob_hist, dtype=float)
x_proj = np.arange(1950, 2071, 1)

# Diccionario para guardar proyecciones
proyecciones = {'Año': x_proj, 'Real': [np.nan]*len(x_proj)}
for i, año in enumerate(x_proj):
    if año in x_hist:
        proyecciones['Real'][i] = y_hist[np.where(x_hist == año)[0][0]]

# Ajuste Lineal
try:
    popt_lin, _ = curve_fit(modelo_lineal, x_hist, y_hist)
    proyecciones['Lineal'] = modelo_lineal(x_proj, *popt_lin)
except: proyecciones['Lineal'] = [np.nan]*len(x_proj)

# Ajuste Exponencial
try:
    p0_guess = y_hist[0]
    popt_exp, _ = curve_fit(modelo_exponencial, x_hist, y_hist, p0=[p0_guess, 0.01], maxfev=10000)
    proyecciones['Exponencial'] = modelo_exponencial(x_proj, *popt_exp)
except: proyecciones['Exponencial'] = [np.nan]*len(x_proj)

# Ajuste Logístico
try:
    K_guess = max(y_hist) * 1.5
    popt_log, _ = curve_fit(modelo_logistico, x_hist, y_hist, p0=[K_guess, 0.05, 2020], maxfev=10000)
    proyecciones['Logístico'] = modelo_logistico(x_proj, *popt_log)
    param_K = popt_log[0]
except: 
    proyecciones['Logístico'] = proyecciones['Lineal'] # Fallback
    param_K = "N/A"

df_proj = pd.DataFrame(proyecciones)

# --- 5. INTERFAZ GRÁFICA (CURVAS) ---
col_graf, col_param = st.columns([3, 1])

with col_graf:
    st.subheader(f"📈 Modelos de Crecimiento Poblacional - {titulo_terr}")
    fig_curvas = go.Figure()
    
    # Añadir líneas de modelos
    fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Logístico'], mode='lines', name='Mod. Logístico', line=dict(color='#10b981', dash='dash')))
    fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Exponencial'], mode='lines', name='Mod. Exponencial', line=dict(color='#f59e0b', dash='dot')))
    fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Lineal'], mode='lines', name='Mod. Lineal', line=dict(color='#6366f1', dash='dot')))
    
    # Añadir datos reales (Puntos)
    fig_curvas.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Reales (Censo)', marker=dict(color='#ef4444', size=8, symbol='diamond')))
    
    fig_curvas.update_layout(hovermode="x unified", xaxis_title="Año", yaxis_title="Población", template="plotly_white")
    st.plotly_chart(fig_curvas, width='stretch')

with col_param:
    st.subheader("🧮 Ecuaciones")
    st.info("**Logístico:** \nIdeal a largo plazo. Modela el agotamiento de recursos.")
    if param_K != "N/A":
        st.metric("Capacidad de Carga (K)", f"{param_K:,.0f} hab")
    
    st.warning("**Exponencial:** \nCrecimiento sin restricciones (Corto plazo).")
    st.success("**Lineal:** \nTendencia promedio constante.")

st.divider()

# --- 6. ESTIMACIÓN SINTÉTICA Y PIRÁMIDES ---
st.sidebar.header("🎯 2. Viaje en el Tiempo (Pirámide)")
modelo_sel = st.sidebar.radio("Elegir modelo para proyectar pirámide:", ["Logístico", "Exponencial", "Lineal", "Dato Real (Si existe)"])

años_disp = sorted(df_nac['Año'].unique())
año_sel = st.sidebar.select_slider("Selecciona el Año (1952-2070)", options=años_disp, value=2024)

# 1. Obtener la población total calculada por el modelo para el año elegido
pob_modelo_total = df_proj[df_proj['Año'] == año_sel][modelo_sel].values[0]

# Si eligió "Dato Real" pero no hay censo ese año, fallback al logístico
if pd.isna(pob_modelo_total):
    st.sidebar.warning("No hay dato real para ese año. Usando Logístico.")
    pob_modelo_total = df_proj[df_proj['Año'] == año_sel]['Logístico'].values[0]

# 2. Receta Nacional
df_filtrado_nac = df_nac[df_nac['Año'] == año_sel].copy()
pob_nacional_total = df_filtrado_nac['Hombres'].sum() + df_filtrado_nac['Mujeres'].sum()
df_filtrado_nac['Prop_Hombres'] = df_filtrado_nac['Hombres'] / pob_nacional_total
df_filtrado_nac['Prop_Mujeres'] = df_filtrado_nac['Mujeres'] / pob_nacional_total

# 3. Aplicar al territorio
df_filtrado_nac['Hombres_Terr'] = df_filtrado_nac['Prop_Hombres'] * pob_modelo_total
df_filtrado_nac['Mujeres_Terr'] = df_filtrado_nac['Prop_Mujeres'] * pob_modelo_total

col_pir1, col_pir2 = st.columns([2, 1])
with col_pir1:
    st.subheader(f"Pirámide Poblacional Sintética - {titulo_terr} ({año_sel})")
    df_pir = pd.DataFrame({
        'Edad': df_filtrado_nac['Edad'],
        'Hombres': df_filtrado_nac['Hombres_Terr'] * -1,
        'Mujeres': df_filtrado_nac['Mujeres_Terr']
    })
    cortes = list(range(0, 105, 5))
    etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)]
    df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
    df_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()
    
    df_melt = pd.melt(df_agrupado, id_vars=['Rango'], value_vars=['Hombres', 'Mujeres'], var_name='Sexo', value_name='Poblacion')
    fig_pir = px.bar(df_melt, y='Rango', x='Poblacion', color='Sexo', orientation='h', color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'})
    fig_pir.update_layout(barmode='relative', xaxis_title="Población Habitantes", yaxis_title="Grupos de Edad")
    fig_pir.update_traces(hovertemplate="%{y}: %{x:,.0f}")
    st.plotly_chart(fig_pir, width='stretch')

with col_pir2:
    st.subheader("Resumen del Modelo")
    pob_hombres = df_filtrado_nac['Hombres_Terr'].sum()
    pob_mujeres = df_filtrado_nac['Mujeres_Terr'].sum()
    ind_masc = (pob_hombres / pob_mujeres) * 100 if pob_mujeres > 0 else 0
    
    st.metric("Población Proyectada", f"{pob_modelo_total:,.0f}")
    st.metric("Total Hombres (Est.)", f"{pob_hombres:,.0f}")
    st.metric("Total Mujeres (Est.)", f"{pob_mujeres:,.0f}")
    st.metric("Índice Masculinidad", f"{ind_masc:.1f} H por cada 100 M")
