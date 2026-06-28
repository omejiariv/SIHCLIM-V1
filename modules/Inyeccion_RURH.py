# modules/Inyeccion_RURH.py

import os
import sys
import io
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import text

# Configuración inicial
st.set_page_config(page_title="Inyección RURH", page_icon="🏭", layout="wide")

try:
    from modules import selectors
    from modules.db_manager import get_engine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.db_manager import get_engine

selectors.renderizar_menu_navegacion("Inyección RURH")

st.title("🏭 Motor de Inyección RURH (Concesiones y Vertimientos)")
st.markdown("Este módulo descarga automáticamente los archivos Excel desde Supabase, consolida los caudales concesionados y vertimientos, y los inyecta en la base de datos para alimentar el simulador WEAP.")

def encontrar_columna(df, palabras_clave):
    """Busca una columna en el DataFrame basándose en coincidencias de palabras clave."""
    for col in df.columns:
        col_norm = str(col).upper().replace("Ó", "O").replace("Í", "I").replace(" ", "")
        if all(p in col_norm for p in palabras_clave):
            return col
    return None

def limpiar_y_sumar_caudales(df, tipo):
    """Limpia los datos, formatea el territorio y suma el caudal total."""
    # 1. Detectar columnas clave
    col_nombre = encontrar_columna(df, ["NOMBRE", "NSS3"])
    col_codigo = encontrar_columna(df, ["CODIGO", "NSS3"])
    col_caudal = encontrar_columna(df, ["CAUDAL"])

    if not col_nombre or not col_codigo or not col_caudal:
        return pd.DataFrame()

    # 2. Limpieza de datos
    df_clean = df.dropna(subset=[col_nombre, col_codigo, col_caudal]).copy()
    
    # Asegurar que el caudal sea numérico
    df_clean[col_caudal] = pd.to_numeric(df_clean[col_caudal], errors='coerce').fillna(0)

    # 3. Construir la Llave Universal (Ej: "Q. La Honda - (2308-01-04-24)")
    def formatear_territorio(row):
        nombre = str(row[col_nombre]).strip().title().replace("Q.", "Q. ")
        # Limpiar espacios dobles
        nombre = " ".join(nombre.split())
        codigo = str(row[col_codigo]).strip()
        return f"{nombre} - ({codigo})"

    df_clean['Territorio'] = df_clean.apply(formatear_territorio, axis=1)

    # 4. Agrupar y sumar (Convertir L/s a m3/s)
    df_agrupado = df_clean.groupby('Territorio')[col_caudal].sum().reset_index()
    df_agrupado.rename(columns={col_caudal: f'Caudal_{tipo}_Ls'}, inplace=True)
    df_agrupado[f'Caudal_{tipo}_m3s'] = df_agrupado[f'Caudal_{tipo}_Ls'] / 1000.0

    return df_agrupado

if st.button("🚀 Descargar, Procesar e Inyectar a Base de Datos", type="primary"):
    with st.spinner("Conectando con el Aleph de Supabase..."):
        try:
            url_concesiones = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/CONCESIONES_RURH_GUARNE_LAHONDA.xlsx"
            url_vertimientos = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/VERTIMIENTOS_LA_HONDA.xlsx"

            # Descargar archivos
            res_conc = requests.get(url_concesiones)
            res_vert = requests.get(url_vertimientos)

            # Leer todas las hojas y combinarlas
            dict_conc = pd.read_excel(io.BytesIO(res_conc.content), sheet_name=None)
            dict_vert = pd.read_excel(io.BytesIO(res_vert.content), sheet_name=None)

            df_conc_total = pd.concat(dict_conc.values(), ignore_index=True)
            df_vert_total = pd.concat(dict_vert.values(), ignore_index=True)

            st.info("📊 Archivos descargados. Procesando sumatorias espaciales...")

            # Procesar
            df_concesiones = limpiar_y_sumar_caudales(df_conc_total, "Concesionado")
            df_vertimientos = limpiar_y_sumar_caudales(df_vert_total, "Vertimiento")

            # Unir ambos DataFrames usando el Territorio
            if not df_concesiones.empty and not df_vertimientos.empty:
                df_final = pd.merge(df_concesiones, df_vertimientos, on="Territorio", how="outer").fillna(0.0)
            elif not df_concesiones.empty:
                df_final = df_concesiones
                df_final['Caudal_Vertimiento_Ls'] = 0.0
                df_final['Caudal_Vertimiento_m3s'] = 0.0
            else:
                st.error("No se encontraron columnas válidas en los archivos.")
                st.stop()

            # Calcular Presión Total RURH en m3/s
            df_final['Presion_Total_RURH_m3s'] = df_final['Caudal_Concesionado_m3s'] + df_final['Caudal_Vertimiento_m3s']

            st.dataframe(df_final.style.format({
                'Caudal_Concesionado_m3s': '{:.4f}',
                'Caudal_Vertimiento_m3s': '{:.4f}',
                'Presion_Total_RURH_m3s': '{:.4f}'
            }), use_container_width=True)

            # Inyectar a PostgreSQL
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS matriz_presiones_rurh"))
            df_final.to_sql('matriz_presiones_rurh', engine, if_exists='replace', index=False)

            st.success(f"✅ ¡Inyección Exitosa! {len(df_final)} cuencas actualizadas con presiones hídricas reales en PostgreSQL.")

        except Exception as e:
            st.error(f"🚨 Error durante la inyección: {e}")
