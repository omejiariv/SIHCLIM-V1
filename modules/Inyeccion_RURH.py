# modules/Inyeccion_RURH.py

import io
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import text
from modules.db_manager import get_engine

def procesar_e_inyectar_rurh():
    """Descarga el archivo RURH estandarizado de Supabase, lo procesa y lo inyecta en PostgreSQL."""
    with st.spinner("Conectando con el Aleph de Supabase (RURH Consolidado)..."):
        try:
            # 1. DESCARGA DEL ARCHIVO MAESTRO
            url_rurh = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/RURH_La_Honda.xlsx"
            res = requests.get(url_rurh)
            
            # 🔥 CORRECCIÓN DEL ERROR EXCEL: Forzamos el motor openpyxl
            df_rurh = pd.read_excel(io.BytesIO(res.content), engine='openpyxl')
            
            st.info("📊 Archivo RURH estandarizado descargado. Procesando sumatorias espaciales...")

            # 2. LIMPIEZA Y FILTRADO
            # Eliminamos filas que no tengan los datos críticos para el cruce
            df_clean = df_rurh.dropna(subset=['nombre_cuenca_nss3', 'código_cuenca_nss3', 'caudal_total_requerido_lps']).copy()
            
            # 3. FORJA DE LA LLAVE UNIVERSAL
            def formatear_territorio(row):
                nombre = str(row['nombre_cuenca_nss3']).strip().title().replace("Q.", "Q. ")
                # Colapsar espacios dobles
                nombre = " ".join(nombre.split())
                codigo = str(row['código_cuenca_nss3']).strip()
                return f"{nombre} - ({codigo})"

            df_clean['Territorio'] = df_clean.apply(formatear_territorio, axis=1)
            
            # Aseguramos formato numérico
            df_clean['caudal_total_requerido_lps'] = pd.to_numeric(df_clean['caudal_total_requerido_lps'], errors='coerce').fillna(0)

            # 4. SUMATORIA ESPACIAL (Agrupamos por cuenca)
            df_agrupado = df_clean.groupby('Territorio')['caudal_total_requerido_lps'].sum().reset_index()
            
            # Renombramos y convertimos a m3/s (1 m3/s = 1000 L/s)
            df_agrupado.rename(columns={'caudal_total_requerido_lps': 'Caudal_Concesionado_Ls'}, inplace=True)
            df_agrupado['Caudal_Concesionado_m3s'] = df_agrupado['Caudal_Concesionado_Ls'] / 1000.0
            
            # Espacio reservado para los vertimientos (cuando los estandarices de la misma forma)
            df_agrupado['Caudal_Vertimiento_m3s'] = 0.0
            df_agrupado['Presion_Total_RURH_m3s'] = df_agrupado['Caudal_Concesionado_m3s'] + df_agrupado['Caudal_Vertimiento_m3s']

            # Mostrar tabla de resultados en la interfaz
            st.dataframe(df_agrupado.style.format({
                'Caudal_Concesionado_m3s': '{:.4f}',
                'Caudal_Vertimiento_m3s': '{:.4f}',
                'Presion_Total_RURH_m3s': '{:.4f}'
            }), use_container_width=True)

            # 5. INYECCIÓN A POSTGRESQL
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS matriz_presiones_rurh"))
            df_agrupado.to_sql('matriz_presiones_rurh', engine, if_exists='replace', index=False)

            st.success(f"✅ ¡Inyección Exitosa! {len(df_agrupado)} cuencas actualizadas con presiones hídricas reales en PostgreSQL.")

        except Exception as e:
            st.error(f"🚨 Error durante la inyección: {e}")
