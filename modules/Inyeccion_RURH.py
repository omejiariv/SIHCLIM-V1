import streamlit as st
import pandas as pd
import requests
import io
from sqlalchemy import text
from modules.db_manager import get_engine

def renderizar_inyeccion_rurh():
    st.markdown("### 🏭 Motor de Inyección RURH (Concesiones y Vertimientos)")
    st.info("Este módulo lee el archivo maestro alojado en Supabase, forja la Llave Maestra Espacial y consolida las presiones hídricas en PostgreSQL para el simulador WEAP.")

    url_excel = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Concesiones_SP_Metabolismo_Maestro.xls"
    
    if st.button("🚀 Iniciar Extracción, Transformación y Carga (ETL)", type="primary", use_container_width=True):
        try:
            with st.spinner("1. Descargando matriz maestra desde Supabase..."):
                response = requests.get(url_excel)
                response.raise_for_status()
                
                # Leemos el archivo Excel
                df = pd.read_excel(io.BytesIO(response.content))
                st.success(f"✅ Archivo descargado exitosamente. Filas detectadas: {len(df)}")

            with st.spinner("2. Forjando Llave Maestra Espacial y calculando presiones..."):
                # =====================================================================
                # 🗝️ FORJA DE LA LLAVE MAESTRA ESPACIAL
                # =====================================================================
                # Convertimos los nombres de las columnas a mayúsculas para una búsqueda segura
                cols_upper = [str(c).upper().strip() for c in df.columns]
                
                if 'NOM_NSS3' in cols_upper and 'NSS3' in cols_upper:
                    # Encontramos los nombres reales de las columnas en el dataframe
                    col_nom = df.columns[cols_upper.index('NOM_NSS3')]
                    col_cod = df.columns[cols_upper.index('NSS3')]
                    
                    # Construcción de la Llave: Ejemplo "Q. La Honda - (2308-01-04-24)"
                    df['Territorio'] = df[col_nom].astype(str).str.strip() + " - (" + df[col_cod].astype(str).str.strip() + ")"
                    st.write("🔑 **Llave Maestra forjada correctamente** a partir del Join Espacial.")
                else:
                    # Plan B en caso de que el Excel no traiga las columnas del Join
                    col_fallback = df.columns[cols_upper.index('SISTEMA_HIDRICO')] if 'SISTEMA_HIDRICO' in cols_upper else None
                    if col_fallback:
                        df['Territorio'] = df[col_fallback].fillna("Desconocido")
                        st.warning("⚠️ Columnas NSS3 no detectadas. Se usó el nombre básico del sistema hídrico.")
                    else:
                        df['Territorio'] = "Desconocido"
                        st.error("❌ No se encontraron columnas territoriales válidas en el archivo.")

                # =====================================================================
                # 🧮 CÁLCULO DE PRESIONES Y AGRUPACIÓN
                # =====================================================================
                col_caudal = next((c for c in df.columns if 'caudal' in str(c).lower() and 'concesionado' in str(c).lower()), None)
                
                if not col_caudal:
                    st.error("❌ No se encontró la columna de Caudal Concesionado en el Excel.")
                    return

                # Convertimos a numérico, forzando errores a NaN y luego a 0
                df[col_caudal] = pd.to_numeric(df[col_caudal], errors='coerce').fillna(0)
                
                # Convertimos L/s a m³/s si es necesario (ajustar si el Excel ya viene en m³/s)
                # Asumiendo que el Excel maestro suele venir en L/s según estándares corporativos
                # Si ya viene en m3/s, elimina la división por 1000
                df['Caudal_m3s'] = df[col_caudal] / 1000.0 

                # Agrupamos por la Llave Maestra
                df_consolidado = df.groupby('Territorio', as_index=False)['Caudal_m3s'].sum()
                df_consolidado.rename(columns={'Caudal_m3s': 'Presion_Total_RURH_m3s'}, inplace=True)
                
                st.success(f"✅ Presiones consolidadas en {len(df_consolidado)} unidades territoriales únicas.")
                
                with st.expander("👁️ Vista Previa del Consolidado RURH"):
                    st.dataframe(df_consolidado.head(10), use_container_width=True)

            with st.spinner("3. Inyectando resultados en PostgreSQL..."):
                engine = get_engine()
                with engine.begin() as conn:
                    # Opcional: Limpiar la tabla antes de inyectar los nuevos datos para evitar duplicados
                    # conn.execute(text("TRUNCATE TABLE matriz_presiones_rurh RESTART IDENTITY"))
                    
                    df_consolidado.to_sql(
                        'matriz_presiones_rurh',
                        con=conn,
                        if_exists='replace', # Reemplaza la tabla completa con la nueva verdad oficial
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                st.balloons()
                st.success("🎉 ¡Inyección exitosa! La base de datos RURH está actualizada y lista para el WEAP.")

        except Exception as e:
            st.error(f"🚨 Error crítico durante el proceso ETL: {str(e)}")
