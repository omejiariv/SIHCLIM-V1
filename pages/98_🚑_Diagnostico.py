import streamlit as st
import pandas as pd
from modules.db_manager import get_engine

st.set_page_config(page_title="Buscando Coordenadas", layout="wide")
st.title("🕵️‍♂️ Búsqueda de Coordenadas Perdidas")

engine = get_engine()

st.subheader("1. Conteo de Salud")
try:
    # Contamos cuántas tienen coordenadas y cuántas no
    df_count = pd.read_sql("""
        SELECT 
            COUNT(*) as total,
            COUNT(latitud) as con_latitud,
            COUNT(longitud) as con_longitud
        FROM estaciones
    """, engine)
    
    total = df_count.iloc[0]['total']
    validas = df_count.iloc[0]['con_latitud']
    
    c1, c2 = st.columns(2)
    c1.metric("Total Estaciones", total)
    c2.metric("Con Coordenadas Válidas", validas)
    
    if validas == 0:
        st.error("🚨 ¡CERO! Ninguna estación tiene coordenadas en las columnas 'latitud'/'longitud'.")
    else:
        st.success(f"✅ Hay {validas} estaciones sanas.")

except Exception as e:
    st.error(str(e))

st.subheader("2. Inspección de Columnas (¿Dónde están?)")
try:
    # Traemos las primeras 5 filas COMPLETAS (todas las columnas)
    df_all = pd.read_sql("SELECT * FROM estaciones LIMIT 5", engine)
    st.write("Mira esta tabla. ¿Ves tus coordenadas en alguna columna extraña?")
    st.dataframe(df_all)
    
    # Análisis de columnas candidatas
    cols = df_all.columns.tolist()
    st.write(f"Columnas detectadas en BD: {cols}")

except Exception as e:
    st.error(str(e))