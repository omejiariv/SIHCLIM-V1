# modules/Inyeccion_RURH.py

import io
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import text
from modules.db_manager import get_engine

def procesar_e_inyectar_rurh():
    """Descarga el Metabolismo Maestro, adapta dinámicamente sus columnas e inyecta en PostgreSQL."""
    with st.spinner("Conectando con el Aleph de Supabase (Metabolismo Maestro Departamental)..."):
        try:
            # 1. DESCARGA DEL ARCHIVO MAESTRO (Formato .xlsx)
            url_rurh = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Concesiones_SP_Metabolismo_Maestro.xlsx"
            res = requests.get(url_rurh)
            
            df_rurh = pd.read_excel(io.BytesIO(res.content), engine='openpyxl')
            cols = df_rurh.columns.tolist()
            
            st.info("📊 Archivo Maestro descargado. Activando Traductor Universal de Columnas...")

            # 2. ADAPTADOR DINÁMICO (La Piedra Rosetta)
            # Busca las columnas sin importar si es el archivo de La Honda o el Maestro
            col_cuenca = 'nombre_cuenca_nss3' if 'nombre_cuenca_nss3' in cols else 'Cuenca'
            col_codigo = 'código_cuenca_nss3' if 'código_cuenca_nss3' in cols else 'Codigo'
            col_caudal = 'caudal_total_requerido_lps' if 'caudal_total_requerido_lps' in cols else 'Caudal_Lps'
            col_uso = 'Uso_Agua' if 'Uso_Agua' in cols else None

            # Si ni siquiera encuentra la cuenca, detiene el proceso con un mensaje claro
            if col_cuenca not in cols:
                st.error(f"🚨 Falla estructural: No encuentro la columna de Cuenca. Columnas detectadas: {cols}")
                return

            # Limpieza básica
            df_clean = df_rurh.dropna(subset=[col_cuenca, col_caudal]).copy()
            df_clean[col_caudal] = pd.to_numeric(df_clean[col_caudal], errors='coerce').fillna(0)

            # 3. CLASIFICACIÓN SECTORIAL AUTOMÁTICA
            # Si el archivo tiene la columna Uso_Agua, calculamos los caudales específicos
            if col_uso:
                df_clean['Uso_Norm'] = df_clean[col_uso].astype(str).str.lower().str.strip()
                df_clean['caudal_domestico_lps'] = df_clean.apply(lambda x: x[col_caudal] if 'domest' in x['Uso_Norm'] or 'consumo' in x['Uso_Norm'] else 0, axis=1)
                df_clean['caudal_agricola_lps'] = df_clean.apply(lambda x: x[col_caudal] if 'agric' in x['Uso_Norm'] or 'riego' in x['Uso_Norm'] else 0, axis=1)
                df_clean['caudal_pecuario_lps'] = df_clean.apply(lambda x: x[col_caudal] if 'pecuar' in x['Uso_Norm'] else 0, axis=1)
            else:
                # Si las columnas ya venían separadas (como en La Honda), nos aseguramos de que existan
                for c in ['caudal_domestico_lps', 'caudal_agricola_lps', 'caudal_pecuario_lps']:
                    if c not in df_clean.columns: df_clean[c] = 0.0

            # 4. FORJA DE LA LLAVE UNIVERSAL
            def formatear_territorio(row):
                nombre = str(row[col_cuenca]).strip().title().replace("Q.", "Q. ")
                nombre = " ".join(nombre.split())
                codigo = str(row.get(col_codigo, "")).strip()
                
                # Si hay un código NSS3 válido, lo añade. Si no, usa solo el nombre (WEAP lo atrapará con su búsqueda flexible)
                if codigo and codigo.lower() != 'nan' and codigo != 'None':
                    return f"{nombre} - ({codigo})"
                return nombre

            df_clean['Territorio'] = df_clean.apply(formatear_territorio, axis=1)

            # 5. SUMATORIA ESPACIAL (Agrupación)
            cols_sumar = [col_caudal, 'caudal_domestico_lps', 'caudal_agricola_lps', 'caudal_pecuario_lps']
            df_agrupado = df_clean.groupby('Territorio')[cols_sumar].sum().reset_index()
            
            # Renombramos y convertimos a m3/s
            df_agrupado.rename(columns={col_caudal: 'Caudal_Concesionado_Ls'}, inplace=True)
            df_agrupado['Caudal_Concesionado_m3s'] = df_agrupado['Caudal_Concesionado_Ls'] / 1000.0
            df_agrupado['Caudal_Domestico_m3s'] = df_agrupado['caudal_domestico_lps'] / 1000.0
            df_agrupado['Caudal_Agricola_m3s'] = df_agrupado['caudal_agricola_lps'] / 1000.0
            df_agrupado['Caudal_Pecuario_m3s'] = df_agrupado['caudal_pecuario_lps'] / 1000.0
            
            # Espacio para vertimientos
            df_agrupado['Caudal_Vertimiento_m3s'] = 0.0
            df_agrupado['Presion_Total_RURH_m3s'] = df_agrupado['Caudal_Concesionado_m3s'] + df_agrupado['Caudal_Vertimiento_m3s']

            # Mostrar tabla de resultados en la interfaz
            st.markdown("### 🗄️ Matriz Agrupada Procesada (Para Motor WEAP)")
            format_dict = {col: '{:.4f}' for col in df_agrupado.columns if '_m3s' in col}
            st.dataframe(df_agrupado.head(15).style.format(format_dict), use_container_width=True)

            # 6. INYECCIÓN DUAL A POSTGRESQL
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS matriz_presiones_rurh"))
                df_agrupado.to_sql('matriz_presiones_rurh', engine, if_exists='replace', index=False)
                
                conn.execute(text("DROP TABLE IF EXISTS concesiones_maestras_rurh_raw"))
                df_clean.to_sql('concesiones_maestras_rurh_raw', engine, if_exists='replace', index=False)

            st.success(f"✅ ¡Inyección Doble Exitosa! {len(df_agrupado)} cuencas consolidadas y {len(df_clean)} concesiones guardadas en el Gemelo Digital.")

        except Exception as e:
            st.error(f"🚨 Error durante la inyección: {e}")
