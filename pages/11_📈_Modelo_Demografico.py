# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Modelo Demográfico", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico y Calibración Top-Down")
st.markdown("Motor basado en Estimación Sintética: Anidación proporcional desde la escala Nacional hasta la Municipal.")
st.divider()

# --- 1. LECTURA DE DATOS LIMPIOS ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def cargar_datos_limpios():
    try:
        # Capa 1: Nacional Histórico
        ruta_nac = os.path.join(RUTA_RAIZ, "data", "PobCol1912_2100.csv")
        df_nac = pd.read_csv(ruta_nac, sep=',')
        if len(df_nac.columns) < 2: df_nac = pd.read_csv(ruta_nac, sep=';')
            
        df_nac.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().title() for c in df_nac.columns]
        df_nac = df_nac.rename(columns={'Male': 'Hombres', 'Female': 'Mujeres', 'Ano': 'Año'})
        
        # Capa 2: Municipal
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
            
        df_mun.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().lower() for c in df_mun.columns]
        df_mun = df_mun.rename(columns={'poblacion': 'Total'})
        
        return df_nac, df_mun
        
    except Exception as e:
        st.error(f"🚨 Error interno de lectura: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_nac, df_mun = cargar_datos_limpios()

if df_nac.empty or df_mun.empty:
    st.stop()

if 'Año' not in df_nac.columns or 'año' not in df_mun.columns:
    st.error("Error en las columnas base. Revisa los archivos.")
    st.stop()

# --- 2. MOTOR DE FILTROS EN CASCADA ---
st.sidebar.header("⚙️ Configuración")

escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Departamental", "Municipal"])

# Control de Años Inteligente
if escala_sel == "Nacional":
    años_disponibles = sorted(df_nac['Año'].unique())
    año_defecto = 2024 if 2024 in años_disponibles else años_disponibles[-1]
    año_sel = st.sidebar.select_slider("Año", options=años_disponibles, value=año_defecto)
    
    df_filtrado_nac = df_nac[df_nac['Año'] == año_sel].copy()
    titulo_grafico = f"Pirámide Poblacional Colombia ({año_sel})"
    pob_total_territorio = df_filtrado_nac['Hombres'].sum() + df_filtrado_nac['Mujeres'].sum()
    
else:
    # Para Deptos y Municipios, los datos llegan hasta 2020 en esta base
    años_disponibles = sorted(df_mun['año'].unique())
    año_defecto = 2020 if 2020 in años_disponibles else años_disponibles[-1]
    año_sel = st.sidebar.select_slider("Año (2005-2020)", options=años_disponibles, value=año_defecto)
    
    deptos = sorted(df_mun['depto_nom'].unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental":
        # Sumamos la población de todo el departamento (urbano + rural)
        df_filtro_terr = df_mun[(df_mun['año'] == año_sel) & (df_mun['depto_nom'] == depto_sel)]
        pob_total_territorio = df_filtro_terr['Total'].sum()
        titulo_grafico = f"Pirámide Sintética - {depto_sel} ({año_sel})"
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        
        # Población exacta del municipio (urbano + rural)
        df_filtro_terr = df_mun[(df_mun['año'] == año_sel) & (df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)]
        pob_total_territorio = df_filtro_terr['Total'].sum()
        titulo_grafico = f"Pirámide Sintética - {mpio_sel}, {depto_sel} ({año_sel})"

    # 🧠 MOTOR DE ESTIMACIÓN SINTÉTICA (TOP-DOWN)
    # 1. Sacamos la "Receta" Nacional para ese mismo año
    df_filtrado_nac = df_nac[df_nac['Año'] == año_sel].copy()
    pob_nacional_total = df_filtrado_nac['Hombres'].sum() + df_filtrado_nac['Mujeres'].sum()
    
    # 2. Convertimos a proporciones (Ej: Hombres de 20 años son el 2% del país)
    df_filtrado_nac['Prop_Hombres'] = df_filtrado_nac['Hombres'] / pob_nacional_total
    df_filtrado_nac['Prop_Mujeres'] = df_filtrado_nac['Mujeres'] / pob_nacional_total
    
    # 3. Aplicamos la receta al territorio local
    df_filtrado_nac['Hombres'] = df_filtrado_nac['Prop_Hombres'] * pob_total_territorio
    df_filtrado_nac['Mujeres'] = df_filtrado_nac['Prop_Mujeres'] * pob_total_territorio

# --- 3. RENDERIZADO VISUAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(titulo_grafico)
    
    # Transformar para pirámide
    df_pir = pd.DataFrame({
        'Edad': df_filtrado_nac['Edad'],
        'Hombres': df_filtrado_nac['Hombres'] * -1,  # Negativo para dibujar a la izquierda
        'Mujeres': df_filtrado_nac['Mujeres']
    })
    
    # Agrupamos por rangos quinquenales
    cortes = list(range(0, 105, 5))
    etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)]
    df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
    df_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()
    
    df_melt = pd.melt(df_agrupado, id_vars=['Rango'], value_vars=['Hombres', 'Mujeres'], var_name='Sexo', value_name='Poblacion')
    
    fig = px.bar(df_melt, y='Rango', x='Poblacion', color='Sexo', orientation='h',
                 color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'})
    fig.update_layout(barmode='relative', xaxis_title="Población Habitantes", yaxis_title="Grupos de Edad")
    fig.update_traces(hovertemplate="%{y}: %{x:,.0f}")
    
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Resumen Demográfico")
    
    pob_hombres = df_filtrado_nac['Hombres'].sum()
    pob_mujeres = df_filtrado_nac['Mujeres'].sum()
    ind_masc = (pob_hombres / pob_mujeres) * 100 if pob_mujeres > 0 else 0
    
    st.metric("Población Total", f"{pob_total_territorio:,.0f}")
    st.metric("Total Hombres (Est.)", f"{pob_hombres:,.0f}")
    st.metric("Total Mujeres (Est.)", f"{pob_mujeres:,.0f}")
    st.metric("Índice Masculinidad", f"{ind_masc:.1f} H por cada 100 M")

st.success("✅ Motor de Estimación Sintética Activo. Proporciones nacionales anidadas a poblaciones reales locales.")
