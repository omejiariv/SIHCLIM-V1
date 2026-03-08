# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Modelo Demográfico", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico y Calibración Top-Down")
st.markdown("Motor basado en datos depurados (1912-2100) con anidación a escala Municipal y Veredal.")
st.divider()

# --- 1. LECTURA DE DATOS LIMPIOS ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def cargar_datos_limpios():
    try:
        # --- Capa 1: Nacional Histórico ---
        ruta_nac = os.path.join(RUTA_RAIZ, "data", "PobCol1912_2100.csv")
        
        # Lectura "Todo Terreno": Detecta si viene separado por comas o punto y coma
        df_nac = pd.read_csv(ruta_nac, sep=',')
        if len(df_nac.columns) < 2:
            df_nac = pd.read_csv(ruta_nac, sep=';')
            
        # Limpiamos y forzamos a que la primera letra sea mayúscula (Ej: año -> Año)
        df_nac.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().title() for c in df_nac.columns]
        df_nac = df_nac.rename(columns={'Male': 'Hombres', 'Female': 'Mujeres', 'Ano': 'Año'})
        
        # --- Capa 2: Municipal ---
        ruta_mun_1 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_Colombia.csv")
        ruta_mun_2 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_colombia.csv")
        
        if os.path.exists(ruta_mun_1):
            df_mun = pd.read_csv(ruta_mun_1, sep=',')
            if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_1, sep=';')
        elif os.path.exists(ruta_mun_2):
            df_mun = pd.read_csv(ruta_mun_2, sep=',')
            if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_2, sep=';')
        else:
            raise FileNotFoundError("No encontré el archivo municipal.")
            
        # Limpiamos y forzamos a minúsculas para homologar con el motor
        df_mun.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().lower() for c in df_mun.columns]
        df_mun = df_mun.rename(columns={'poblacion': 'Total'})
        
        return df_nac, df_mun
        
    except Exception as e:
        st.error(f"🚨 Error interno de lectura: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_nac, df_mun = cargar_datos_limpios()

if df_nac.empty or df_mun.empty:
    st.stop()

# 🛑 ALARMA ANTI-ERRORES: Si la columna no existe, nos muestra qué está leyendo la máquina
if 'Año' not in df_nac.columns:
    st.error(f"❌ Archivo Nacional: No encontré la columna 'Año'. Esto es lo que Python está leyendo realmente: {list(df_nac.columns)}")
    st.stop()
    
if 'año' not in df_mun.columns:
    st.error(f"❌ Archivo Municipal: No encontré la columna 'año'. Esto es lo que Python está leyendo realmente: {list(df_mun.columns)}")
    st.stop()
    
# --- 2. MOTOR DE FILTROS EN CASCADA ---
st.sidebar.header("⚙️ Configuración")

escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional (1952-2100)", "Departamental (2005-2020)", "Municipal (2005-2020)"])

if escala_sel == "Nacional (1952-2100)":
    año_sel = st.sidebar.slider("Año", int(df_nac['Año'].min()), int(df_nac['Año'].max()), 2024)
    df_filtrado = df_nac[df_nac['Año'] == año_sel].copy()
    titulo_grafico = f"Pirámide Poblacional Colombia ({año_sel})"
    
else:
    año_sel = st.sidebar.slider("Año", int(df_mun['año'].min()), int(df_mun['año'].max()), 2020)
    deptos = sorted(df_mun['depto_nom'].unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental (2005-2020)":
        df_filtrado = df_mun[(df_mun['año'] == año_sel) & (df_mun['depto_nom'] == depto_sel)].copy()
        titulo_grafico = f"Distribución de Áreas en {depto_sel} ({año_sel})"
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        df_filtrado = df_mun[(df_mun['año'] == año_sel) & (df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].copy()
        titulo_grafico = f"Población en {mpio_sel}, {depto_sel} ({año_sel})"

# --- 3. RENDERIZADO VISUAL SEGÚN LA ESCALA ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(titulo_grafico)
    
    if escala_sel == "Nacional (1952-2100)":
        # Transformar para pirámide
        df_pir = pd.DataFrame({
            'Edad': df_filtrado['Edad'],
            'Hombres': df_filtrado['Hombres'] * -1,  # Negativo para dibujar a la izquierda
            'Mujeres': df_filtrado['Mujeres']
        })
        # Agrupamos por rangos quinquenales para que la gráfica se vea limpia
        cortes = list(range(0, 105, 5))
        etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)]
        df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
        # observed=True evita otra pequeña advertencia futura de Pandas
        df_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()
        
        df_melt = pd.melt(df_agrupado, id_vars=['Rango'], value_vars=['Hombres', 'Mujeres'], var_name='Sexo', value_name='Poblacion')
        
        fig = px.bar(df_melt, y='Rango', x='Poblacion', color='Sexo', orientation='h',
                     color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'})
        fig.update_layout(barmode='relative', xaxis_title="Población", yaxis_title="Edad")
        fig.update_traces(hovertemplate="%{y}: %{x}")
        # Corrección de la advertencia de Streamlit
        st.plotly_chart(fig, width='stretch')
        
    else:
        # Gráfica de Urbano vs Rural para Deptos y Municipios
        df_area = df_filtrado.groupby('area_geografica')['Total'].sum().reset_index()
        fig = px.pie(df_area, values='Total', names='area_geografica', hole=0.4, 
                     color='area_geografica', color_discrete_map={'urbano': '#3b82f6', 'rural': '#22c55e'})
        # Corrección de la advertencia de Streamlit
        st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Resumen Demográfico")
    
    if escala_sel == "Nacional (1952-2100)":
        pob_hombres = df_filtrado['Hombres'].sum()
        pob_mujeres = df_filtrado['Mujeres'].sum()
        pob_total = pob_hombres + pob_mujeres
        ind_masc = (pob_hombres / pob_mujeres) * 100 if pob_mujeres > 0 else 0
        
        st.metric("Población Total", f"{pob_total:,.0f}")
        st.metric("Total Hombres", f"{pob_hombres:,.0f}")
        st.metric("Total Mujeres", f"{pob_mujeres:,.0f}")
        st.metric("Índice Masculinidad", f"{ind_masc:.1f} H por cada 100 M")
    else:
        df_area = df_filtrado.groupby('area_geografica')['Total'].sum()
        pob_urbana = df_area.get('urbano', 0)
        pob_rural = df_area.get('rural', 0)
        pob_total = pob_urbana + pob_rural
        
        st.metric("Población Total", f"{pob_total:,.0f}")
        st.metric("Población Urbana", f"{pob_urbana:,.0f}")
        st.metric("Población Rural", f"{pob_rural:,.0f}")

st.success("✅ Base de datos 100% depurada conectada. Listos para inyectar Capa Veredal y Crecimiento Logístico.")
