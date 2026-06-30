import streamlit as st
import pandas as pd
import requests
import io
import geopandas as gpd
from sqlalchemy import text
from modules.db_manager import get_engine

def renderizar_inyeccion_rurh():
    st.markdown("### 🏭 Motor de Inyección RURH (ETL Geoespacial)")
    st.info("Este módulo lee las concesiones con coordenadas (Supabase), realiza un **Join Espacial en vivo** contra la capa de cuencas (PostgreSQL) para forjar la Llave Maestra, y consolida las presiones.")

    url_excel = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Concesiones_SP_Metabolismo_Maestro.xls"
    
    if st.button("🚀 Ejecutar Join Espacial y Consolidar (ETL)", type="primary", use_container_width=True):
        try:
            with st.spinner("1. Descargando matriz de puntos (RURH) desde Supabase..."):
                response = requests.get(url_excel)
                response.raise_for_status()
                df = pd.read_excel(io.BytesIO(response.content))
                st.success(f"✅ Archivo crudo descargado. Filas detectadas: {len(df)}")

            with st.spinner("2. Detectando Coordenadas y Caudal..."):
                # Búsqueda dinámica de las columnas (insensible a mayúsculas y espacios)
                cols_upper = {str(c).upper().strip(): c for c in df.columns}
                
                # 🔥 FIX: Búsqueda exacta de las columnas de coordenadas y caudal
                col_caudal = cols_upper.get('CAUDAL_LPS')
                col_norte = cols_upper.get('LATITUD_N')
                col_oeste = cols_upper.get('LONGITUD_W')

                if not col_caudal:
                    st.error("❌ No se encontró la columna exacta 'Caudal_Lps' en el archivo.")
                    return
                if not col_norte or not col_oeste:
                    st.error("❌ No se encontraron las columnas de coordenadas ('Latitud_N' y 'Longitud_W').")
                    return

                # Limpieza de coordenadas (Aseguramos que sean números)
                df['y_coord'] = pd.to_numeric(df[col_norte], errors='coerce')
                df['x_coord'] = pd.to_numeric(df[col_oeste], errors='coerce')
                
                # Ajuste técnico para Colombia: Si la longitud "Longitud_W" viene positiva, la volvemos negativa para WGS84
                df['x_coord'] = df['x_coord'].apply(lambda x: -x if pd.notnull(x) and x > 0 and x < 100 else x)
                df_puntos = df.dropna(subset=['x_coord', 'y_coord']).copy()
                
                # Convertimos el DataFrame a un GeoDataFrame (Asumimos EPSG:4326 - WGS84 como base)
                gdf_puntos = gpd.GeoDataFrame(
                    df_puntos, 
                    geometry=gpd.points_from_xy(df_puntos['x_coord'], df_puntos['y_coord']),
                    crs="EPSG:4326"
                )
                st.success(f"✅ Geometrías creadas: {len(gdf_puntos)} puntos de captación/vertimiento listos.")

            with st.spinner("3. Ejecutando Join Espacial contra polígonos de Cuencas en PostgreSQL..."):
                engine = get_engine()
                # Traemos la capa oficial desde la base de datos (NSS3, NOM_NSS3 y su geometría)
                query_poligonos = text('SELECT nss3, nom_nss3, geometry FROM cuencas')
                gdf_cuencas = gpd.read_postgis(query_poligonos, engine, geom_col='geometry')
                
                if gdf_cuencas.crs is None:
                    gdf_cuencas.set_crs("EPSG:4326", inplace=True)
                
                # Proyección estandarizada: forzamos que ambos hablen el mismo sistema de coordenadas
                if gdf_puntos.crs != gdf_cuencas.crs:
                    gdf_puntos = gdf_puntos.to_crs(gdf_cuencas.crs)

                # 🔥 LA MAGIA: Join Espacial (Point in Polygon)
                gdf_cruce = gpd.sjoin(gdf_puntos, gdf_cuencas, how="inner", predicate="intersects")
                
                if gdf_cruce.empty:
                    st.error("❌ El Join Espacial falló. Ningún punto cayó dentro de los polígonos de cuencas (Verificar sistema de coordenadas o si los puntos caen en el mar).")
                    return

                st.success(f"✅ Join Espacial exitoso: {len(gdf_cruce)} concesiones asociadas a una cuenca IDEAM.")

            with st.spinner("4. Forjando Llave Maestra y Consolidando Metabolismo..."):
                # 🗝️ FORJA DE LA LLAVE MAESTRA
                gdf_cruce['Territorio'] = gdf_cruce['nom_nss3'].astype(str).str.strip() + " - (" + gdf_cruce['nss3'].astype(str).str.strip() + ")"
                
                # 🧮 CÁLCULO DE CAUDAL
                gdf_cruce['Caudal_m3s'] = pd.to_numeric(gdf_cruce[col_caudal], errors='coerce').fillna(0) / 1000.0
                
                # 📦 MATRIZ 1: PREPARAR DATOS CRUDOS (RAW) PARA EL MAPA Y CÁLCULOS DETALLADOS
                # Eliminamos la geometría para guardarlo en formato tabular estándar en SQL
                df_raw_to_sql = pd.DataFrame(gdf_cruce).drop(columns=['geometry'], errors='ignore')
                
                # 📊 MATRIZ 2: AGRUPACIÓN FINAL (RESUMEN)
                df_consolidado = gdf_cruce.groupby('Territorio', as_index=False)['Caudal_m3s'].sum()
                df_consolidado.rename(columns={'Caudal_m3s': 'Presion_Total_RURH_m3s'}, inplace=True)

                with st.expander("👁️ Vista Previa del Consolidado Final"):
                    st.dataframe(df_consolidado.sort_values(by='Presion_Total_RURH_m3s', ascending=False).head(15), use_container_width=True)

            with st.spinner("5. Inyectando bases de datos a PostgreSQL..."):
                with engine.begin() as conn:
                    # 1. Guardamos la tabla RAW (70,000 registros listos para la Pág 07)
                    df_raw_to_sql.to_sql('concesiones_maestras_rurh_raw', con=conn, if_exists='replace', index=False, method='multi')
                    
                    # 2. Guardamos la tabla consolidada (Para simuladores rápidos como WEAP)
                    df_consolidado.to_sql('matriz_presiones_rurh', con=conn, if_exists='replace', index=False, method='multi')
                
                st.balloons()
                st.success("🎉 ¡Inyección maestra completada! Se han guardado los registros crudos y el consolidado oficial en PostgreSQL.")
