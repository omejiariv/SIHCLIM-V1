# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Modelo Demográfico Avanzado", page_icon="📈", layout="wide")

st.title("📈 Modelo Demográfico y Dasimétrico")
st.markdown("Motor avanzado de simulación demográfica, calibración top-down y modelamiento de la estructura poblacional.")
st.divider()

# Creamos un "GPS" para encontrar la carpeta raíz del proyecto automáticamente
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def cargar_datos_piramides():
    ruta = os.path.join(RUTA_RAIZ, "data", "demografia_piramides_maestra.parquet")
    if os.path.exists(ruta):
        try:
            return pd.read_parquet(ruta)
        except Exception as e:
            st.error(f"🚨 El archivo existe, pero hubo un error al leerlo: {e}")
            st.info("💡 Asegúrate de tener 'pyarrow' o 'fastparquet' en tu archivo requirements.txt")
            return pd.DataFrame()
    else:
        # Modo depuración: Te muestra en pantalla dónde está buscando exactamente
        st.error(f"🔍 Busqué el archivo exactamente en: {ruta} y no está.")
        return pd.DataFrame()

@st.cache_data
def cargar_territorio():
    ruta = os.path.join(RUTA_RAIZ, "data", "territorio_maestro.xlsx")
    if os.path.exists(ruta):
        return pd.read_excel(ruta)
    return pd.DataFrame()

df_pir = cargar_datos_piramides()
df_terr = cargar_territorio()

if df_pir.empty:
    st.warning("⚠️ No se encontró la base de datos maestra de pirámides. Por favor ejecuta el script ETL local o verifica GitHub.")
    st.stop()

# --- 2. MOTOR DE FILTROS EN CASCADA ---
st.sidebar.header("⚙️ Configuración del Modelo")

# Filtro 1: Año
año_sel = st.sidebar.slider("Seleccione el Año", min_value=int(df_pir['año'].min()), max_value=int(df_pir['año'].max()), value=2024)

# Filtro 2: Área Geográfica
area_sel = st.sidebar.radio("Zona", ["Total", "Urbano", "Rural"], horizontal=True)
area_filtro = area_sel.lower() if area_sel != "Total" else ["urbano", "rural", "total"]

# Filtro 3: Escala
escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Departamental"])

departamento_sel = "Nacional"
if escala_sel == "Departamental":
    # Extraemos departamentos asegurando que sean textos válidos y no nulos
    deptos_disponibles = sorted([str(d).title() for d in df_pir['dpnom'].unique() if pd.notna(d) and str(d).upper() != "NACIONAL"])
    departamento_sel = st.sidebar.selectbox("Seleccione el Departamento", deptos_disponibles)
    
    # 🛡️ Salvavidas anti-NoneType (Si está vacío, elige el primero por defecto)
    if departamento_sel is None:
        departamento_sel = deptos_disponibles[0] if deptos_disponibles else "Nacional"

# --- 3. PROCESAMIENTO DINÁMICO (FILTRO BLINDADO) ---
# 1. Escala
if escala_sel == "Departamental":
    df_filtrado = df_pir[df_pir['dpnom'] == str(departamento_sel).upper()]
else:
    df_filtrado = df_pir[df_pir['dpnom'] == "NACIONAL"]

# 2. Año
df_filtrado = df_filtrado[df_filtrado['año'] == año_sel]

# 3. Limpieza Estricta de SEXO (Evitar dobles conteos)
# Atrapamos solo hombres y mujeres reales, ignorando filas que digan "Total" o "Ambos"
mask_hombres = df_filtrado['sexo'].str.contains('hombre', case=False, na=False)
mask_mujeres = df_filtrado['sexo'].str.contains('mujer', case=False, na=False)

df_filtrado.loc[mask_hombres, 'sexo_limpio'] = 'Hombres'
df_filtrado.loc[mask_mujeres, 'sexo_limpio'] = 'Mujeres'
df_filtrado = df_filtrado.dropna(subset=['sexo_limpio']) # Destruimos la basura

# 4. Homologación Estricta de ÁREA GEOGRÁFICA
# El DANE cambia los nombres con los años. Esto atrapa todas las variaciones.
mask_urbano = df_filtrado['area_geografica'].str.contains('urbano|cabecera', case=False, na=False)
mask_rural = df_filtrado['area_geografica'].str.contains('rural|resto|centro poblado', case=False, na=False)

if area_sel == "Urbano":
    df_filtrado = df_filtrado[mask_urbano]
elif area_sel == "Rural":
    df_filtrado = df_filtrado[mask_rural]
else: # Para "Total", sumamos Urbano + Rural, pero sin usar la fila "Total" del DANE
    df_filtrado = df_filtrado[mask_urbano | mask_rural]

# Agrupar datos ya purificados
df_agrupado = df_filtrado.groupby(['Rango_Edad', 'sexo_limpio'])['Poblacion'].sum().reset_index()

# Hombres en negativo para dibujar la pirámide hacia la izquierda
df_agrupado.loc[df_agrupado['sexo_limpio'] == 'Hombres', 'Poblacion'] *= -1

# --- 4. RENDERIZADO VISUAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Estructura Poblacional - {departamento_sel} ({año_sel})")
    
    fig = px.bar(
        df_agrupado, 
        y='Rango_Edad', 
        x='Poblacion', 
        color='sexo_limpio', 
        orientation='h',
        color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'},
        labels={'Poblacion': 'Habitantes', 'Rango_Edad': 'Rango de Edad', 'sexo_limpio': 'Sexo'},
        title=f"Pirámide Poblacional: {area_sel}"
    )
    
    fig.update_layout(
        barmode='relative',
        yaxis_title="Grupos de Edad",
        xaxis_title="Población (Hombres ← | → Mujeres)",
        xaxis=dict(tickformat=",.0f"),
        template="plotly_white",
        hovermode="y unified"
    )
    fig.update_traces(hovertemplate="%{y}: %{x}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Resumen Demográfico")
    
    df_absoluto = df_agrupado.copy()
    df_absoluto['Poblacion'] = df_absoluto['Poblacion'].abs()
    
    pob_total = df_absoluto['Poblacion'].sum()
    pob_hombres = df_absoluto[df_absoluto['sexo_limpio'] == 'Hombres']['Poblacion'].sum()
    pob_mujeres = df_absoluto[df_absoluto['sexo_limpio'] == 'Mujeres']['Poblacion'].sum()
    
    indice_masculinidad = (pob_hombres / pob_mujeres) * 100 if pob_mujeres > 0 else 0

    st.metric("Población Total", f"{pob_total:,.0f}")
    st.metric("Total Hombres", f"{pob_hombres:,.0f}")
    st.metric("Total Mujeres", f"{pob_mujeres:,.0f}")
    st.metric("Índice de Masculinidad", f"{indice_masculinidad:.1f} H por cada 100 M")

st.info("💡 Siguiente fase: Integración del modelo de crecimiento logístico ($K$, $r$) y filtros hasta escala veredal.")
