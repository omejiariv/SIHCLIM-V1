# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
import time
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Modelo Demográfico Integral", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico Integral (Proyección y Dasimetría)")
st.markdown("Ajuste matemático, simulación animada, mapas espaciales y proyección top-down de estructuras poblacionales (1952-2100).")
st.divider()

# --- 1. LECTURA DE DATOS LIMPIOS Y VEREDALES ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def cargar_datos_limpios():
    try:
        # Capa 1: Nacional
        ruta_nac = os.path.join(RUTA_RAIZ, "data", "PobCol1912_2100.csv")
        df_nac = pd.read_csv(ruta_nac, sep=',')
        if len(df_nac.columns) < 2: df_nac = pd.read_csv(ruta_nac, sep=';')
        df_nac.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().title() for c in df_nac.columns]
        df_nac = df_nac.rename(columns={'Male': 'Hombres', 'Female': 'Mujeres', 'Ano': 'Año'})
        
        # Capa 2: Municipal
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
        
        # Capa 3: Veredal (Antioquia)
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
escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Departamental", "Municipal", "Veredal (Antioquia)"])

años_hist, pob_hist = [], []
df_mapa_base = pd.DataFrame() # Base para el mapa

if escala_sel == "Nacional":
    df_agrup_nac = df_nac.groupby('Año')[['Hombres', 'Mujeres']].sum().reset_index()
    años_hist = df_agrup_nac['Año'].values
    pob_hist = (df_agrup_nac['Hombres'] + df_agrup_nac['Mujeres']).values
    titulo_terr = "Colombia (Nacional)"
    df_mapa_base = df_mun.groupby(['depto_nom', 'año'])['Total'].sum().reset_index() # Para mapa nacional, mostramos deptos

elif escala_sel in ["Departamental", "Municipal"]:
    deptos = sorted(df_mun['depto_nom'].dropna().unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental":
        df_terr = df_mun[df_mun['depto_nom'] == depto_sel].groupby('año')['Total'].sum().reset_index()
        titulo_terr = depto_sel
        df_mapa_base = df_mun[df_mun['depto_nom'] == depto_sel].groupby(['municipio', 'año'])['Total'].sum().reset_index()
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].dropna().unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        df_terr = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby('año')['Total'].sum().reset_index()
        titulo_terr = f"{mpio_sel}, {depto_sel}"
        df_mapa_base = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)] # Base urbana/rural
        
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values

elif escala_sel == "Veredal (Antioquia)":
    mpios_veredas = sorted(df_ver['Municipio'].dropna().unique())
    mpio_sel = st.sidebar.selectbox("Municipio (Antioquia)", mpios_veredas)
    
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

    # ANIMACIÓN DE PIRÁMIDES
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

    df_piramide_final = pd.DataFrame() # Para guardarla y descargarla luego

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
        df_piramide_final = df_melt.copy() # Guardar estado para descarga
        
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

st.divider()

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
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_terr}")
    
    st.info("💡 **Módulo Espacial Listo:** La lógica del mapa coroplético ya está programada. Para que los polígonos se dibujen en pantalla, por favor sube los archivos geométricos correspondientes (`.geojson`) a la carpeta `data/` del proyecto.")
    
    # Previsualización de la base de datos que alimentará el mapa
    st.markdown("### Tabla de Atributos Espaciales")
    st.markdown("Esta es la matriz de datos lista para inyectarse al mapa en tiempo real según los filtros seleccionados:")
    st.dataframe(df_mapa_base.head(15), use_container_width=True)
    
    st.markdown("""
    **¿Cómo activar el mapa real en el futuro?**
    1. Guarda el archivo geométrico en `data/` (ej. `municipios_colombia.geojson`).
    2. Usa la librería `plotly.express.choropleth_mapbox` enlazando esta tabla mostrada arriba (`df_mapa_base`) con el archivo geojson mediante la llave primaria (Nombre del municipio o Código DANE).
    """)

# ==========================================
# PESTAÑA 3: DESCARGAS Y EXPORTACIÓN
# ==========================================
with tab_descargas:
    st.subheader("💾 Exportación de Resultados y Series de Tiempo")
    st.markdown("Descarga los modelos matemáticos y las pirámides calculadas por la aplicación para usarlos en informes o software GIS.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("### 📈 Curvas de Proyección (1950-2100)")
        st.markdown("Incluye datos reales de censos y modelos Logístico, Lineal y Exponencial.")
        csv_proj = df_proj.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar Proyecciones (CSV)",
            data=csv_proj,
            file_name=f"Proyecciones_{titulo_terr}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_proj.dropna(subset=['Real']).head(5), use_container_width=True)

    with col_d2:
        st.markdown(f"### 📊 Pirámide Sintética ({año_sel})")
        st.markdown(f"Datos estructurados de la pirámide proyectada para el año seleccionado.")
        
        # Procesar valores absolutos para que el CSV no salga con hombres en negativo
        df_descarga_pir = df_piramide_final.copy()
        if not df_descarga_pir.empty:
            df_descarga_pir['Poblacion'] = df_descarga_pir['Poblacion'].abs()
            csv_pir = df_descarga_pir.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Descargar Pirámide {año_sel} (CSV)",
                data=csv_pir,
                file_name=f"Piramide_{año_sel}_{titulo_terr}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.dataframe(df_descarga_pir.head(5), use_container_width=True)
            
    st.warning("📸 **¿Quieres descargar las gráficas como imagen?** Pasa el mouse sobre cualquier gráfico (curvas o pirámide) y haz clic en el ícono de la cámara 📷 que aparece en la esquina superior derecha.")
