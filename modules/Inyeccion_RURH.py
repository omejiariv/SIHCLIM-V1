# modules/Inyeccion_RURH.py

import io
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import text
from modules.db_manager import get_engine

def procesar_e_inyectar_rurh():
    """Descarga el Metabolismo Maestro de Supabase, lo procesa y realiza una inyección dual en PostgreSQL."""
    with st.spinner("Conectando con el Aleph de Supabase (Metabolismo Maestro Departamental)..."):
        try:
            # 1. DESCARGA DEL ARCHIVO MAESTRO
            url_rurh = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Concesiones_SP_Metabolismo_Maestro.xls"
            res = requests.get(url_rurh)
            
            # 🔥 OJO: Como ahora es un archivo .xls, quitamos el engine='openpyxl' 
            # para que Pandas detecte automáticamente el motor correcto (xlrd o calamine).
            df_rurh = pd.read_excel(io.BytesIO(res.content))
            
            st.info("📊 Archivo Maestro descargado. Procesando el Metabolismo Hídrico de Antioquia...")

            # 2. LIMPIEZA Y FILTRADO BÁSICO
            df_clean = df_rurh.dropna(subset=['nombre_cuenca_nss3', 'código_cuenca_nss3']).copy()
            
            # 3. FORJA DE LA LLAVE UNIVERSAL (Tu lógica original intacta)
            def formatear_territorio(row):
                nombre = str(row['nombre_cuenca_nss3']).strip().title().replace("Q.", "Q. ")
                nombre = " ".join(nombre.split())
                codigo = str(row['código_cuenca_nss3']).strip()
                return f"{nombre} - ({codigo})"

            df_clean['Territorio'] = df_clean.apply(formatear_territorio, axis=1)
            
            # Convertimos a numérico no solo el total, sino también los sub-sectores
            cols_caudales = ['caudal_total_requerido_lps', 'caudal_domestico-lps', 'caudal_pecuario_lps', 'caudal_agricola_lps']
            for col in cols_caudales:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

            # 4. SUMATORIA ESPACIAL PARA EL WEAP (Matriz Agrupada)
            cols_existentes = [c for c in cols_caudales if c in df_clean.columns]
            df_agrupado = df_clean.groupby('Territorio')[cols_existentes].sum().reset_index()
            
            # Renombramos y convertimos a m3/s (Tu lógica matemática)
            if 'caudal_total_requerido_lps' in df_agrupado.columns:
                df_agrupado.rename(columns={'caudal_total_requerido_lps': 'Caudal_Concesionado_Ls'}, inplace=True)
                df_agrupado['Caudal_Concesionado_m3s'] = df_agrupado['Caudal_Concesionado_Ls'] / 1000.0
            
            # Agregamos los sub-caudales en m3/s
            if 'caudal_domestico-lps' in df_agrupado.columns:
                df_agrupado['Caudal_Domestico_m3s'] = df_agrupado['caudal_domestico-lps'] / 1000.0
            if 'caudal_pecuario_lps' in df_agrupado.columns:
                df_agrupado['Caudal_Pecuario_m3s'] = df_agrupado['caudal_pecuario_lps'] / 1000.0
            if 'caudal_agricola_lps' in df_agrupado.columns:
                df_agrupado['Caudal_Agricola_m3s'] = df_agrupado['caudal_agricola_lps'] / 1000.0
            
            # Espacio reservado para los vertimientos
            df_agrupado['Caudal_Vertimiento_m3s'] = 0.0
            df_agrupado['Presion_Total_RURH_m3s'] = df_agrupado.get('Caudal_Concesionado_m3s', 0.0) + df_agrupado['Caudal_Vertimiento_m3s']

            # Mostrar tabla de resultados en la interfaz (Mantenemos tu estilo de 4 decimales)
            st.markdown("### 🗄️ Muestra de la Matriz Agrupada (Para Motor WEAP)")
            # Formateamos solo las columnas que terminan en _m3s para no afectar los textos
            format_dict = {col: '{:.4f}' for col in df_agrupado.columns if '_m3s' in col}
            st.dataframe(df_agrupado.head(15).style.format(format_dict), use_container_width=True)

            # 5. INYECCIÓN DUAL A POSTGRESQL
            engine = get_engine()
            with engine.begin() as conn:
                # Inyección 1: La matriz sumarizada para el simulador WEAP
                conn.execute(text("DROP TABLE IF EXISTS matriz_presiones_rurh"))
                df_agrupado.to_sql('matriz_presiones_rurh', engine, if_exists='replace', index=False)
                
                # Inyección 2: La matriz cruda (con Coordenadas X/Y) para la página de Calidad
                conn.execute(text("DROP TABLE IF EXISTS concesiones_maestras_rurh_raw"))
                df_clean.to_sql('concesiones_maestras_rurh_raw', engine, if_exists='replace', index=False)

            st.success(f"✅ ¡Inyección Doble Exitosa! {len(df_agrupado)} cuencas consolidadas y {len(df_clean)} concesiones individuales guardadas en el Gemelo Digital.")

        except Exception as e:
            st.error(f"🚨 Error durante la inyección: {e}")
