# modules/Inyeccion_RURH.py

import io
import requests
import pandas as pd
import geopandas as gpd
import streamlit as st
from sqlalchemy import text
from modules.db_manager import get_engine

def procesar_e_inyectar_rurh():
    """Descarga el Metabolismo Maestro, realiza Cruce Espacial con la tabla 'cuencas' de PostGIS e inyecta."""
    with st.spinner("Conectando con el Aleph (Descargando Concesiones y leyendo Mapa PostGIS)..."):
        try:
            engine = get_engine()
            
            # 1. DESCARGA DEL ARCHIVO MAESTRO DE CONCESIONES (.xls)
            url_concesiones = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Concesiones_SP_Metabolismo_Maestro.xls"
            res_c = requests.get(url_concesiones)
            df_concesiones = pd.read_excel(io.BytesIO(res_c.content))
            
            # 2. LECTURA DE LA CAPA OFICIAL DE CUENCAS (Directo desde PostgreSQL)
            try:
                # Leemos la tabla geométrica usando el motor PostGIS
                q_cuencas = text("SELECT * FROM cuencas")
                gdf_cuencas = gpd.read_postgis(q_cuencas, engine, geom_col='geometry')
            except Exception as e:
                st.error(f"🚨 No se pudo leer la tabla 'cuencas' de la base de datos: {e}")
                return
            
            st.info("📊 Datos listos. Ejecutando Fusión Espacial (Point-in-Polygon)...")

            # 3. DETECCIÓN DINÁMICA DE COORDENADAS (Concesiones)
            cols = df_concesiones.columns.tolist()
            col_x = next((c for c in cols if str(c).lower() in ['coordenada_x', 'longitud_w', 'lon', 'x']), None)
            col_y = next((c for c in cols if str(c).lower() in ['coordenada_y', 'latitud_n', 'lat', 'y']), None)
            
            if not col_x or not col_y:
                st.error(f"🚨 No encuentro columnas de coordenadas en el Excel. Columnas: {cols}")
                return
            
            # Limpiar nulos y asegurar formato numérico
            df_clean = df_concesiones.dropna(subset=[col_x, col_y]).copy()
            df_clean[col_x] = pd.to_numeric(df_clean[col_x], errors='coerce')
            df_clean[col_y] = pd.to_numeric(df_clean[col_y], errors='coerce')
            df_clean = df_clean.dropna(subset=[col_x, col_y])

            # 4. CREACIÓN DEL GEODATAFRAME DE PUNTOS
            gdf_puntos = gpd.GeoDataFrame(
                df_clean,
                geometry=gpd.points_from_xy(df_clean[col_x], df_clean[col_y]),
                crs="EPSG:4326" 
            )

            # Alineación de sistemas de coordenadas
            if gdf_cuencas.crs is None:
                gdf_cuencas.set_crs(epsg=4326, inplace=True)
            gdf_puntos = gdf_puntos.to_crs(gdf_cuencas.crs)

            # 5. 💥 CRUCE ESPACIAL (La magia de heredar atributos)
            gdf_cruzado = gpd.sjoin(gdf_puntos, gdf_cuencas, how="inner", predicate="intersects")
            
            if gdf_cruzado.empty:
                st.error("🚨 El cruce espacial resultó vacío. Revisa que las coordenadas de las concesiones caigan dentro del mapa de Antioquia.")
                return

            # 6. ADAPTADOR DE CAUDALES Y SECTORES
            col_caudal = next((c for c in gdf_cruzado.columns if str(c).lower() in ['caudal_total_requerido_lps', 'caudal_lps', 'caudal']), None)
            col_uso = next((c for c in gdf_cruzado.columns if 'uso' in str(c).lower()), None)
            
            if not col_caudal:
                st.error("🚨 No se encontró columna de caudal en las concesiones.")
                return
                
            gdf_cruzado[col_caudal] = pd.to_numeric(gdf_cruzado[col_caudal], errors='coerce').fillna(0)

            if col_uso:
                gdf_cruzado['Uso_Norm'] = gdf_cruzado[col_uso].astype(str).str.lower().str.strip()
                gdf_cruzado['caudal_domestico_lps'] = gdf_cruzado.apply(lambda x: x[col_caudal] if 'domest' in x['Uso_Norm'] or 'consumo' in x['Uso_Norm'] else 0, axis=1)
                gdf_cruzado['caudal_agricola_lps'] = gdf_cruzado.apply(lambda x: x[col_caudal] if 'agric' in x['Uso_Norm'] or 'riego' in x['Uso_Norm'] else 0, axis=1)
                gdf_cruzado['caudal_pecuario_lps'] = gdf_cruzado.apply(lambda x: x[col_caudal] if 'pecuar' in x['Uso_Norm'] else 0, axis=1)
            else:
                for c in ['caudal_domestico_lps', 'caudal_agricola_lps', 'caudal_pecuario_lps']:
                    if c not in gdf_cruzado.columns: gdf_cruzado[c] = 0.0

            # 7. FORJA DE LA LLAVE UNIVERSAL (Usando las columnas que vi en tu Panel de Control)
            def formatear_territorio(row):
                # Usamos nombre_cuenca o n_nss3 que tu panel de control asegura que existan
                nombre_base = str(row.get('nombre_cuenca', row.get('n_nss3', ''))).strip()
                nombre = nombre_base.title().replace("Q.", "Q. ")
                nombre = " ".join(nombre.split())
                
                # Intentamos usar n_nss3 como código si id_cuenca no está claro, o viceversa
                codigo = str(row.get('n_nss3', row.get('id_cuenca', ''))).strip()
                
                if codigo and codigo.lower() != 'nan' and codigo != 'none':
                    return f"{nombre} - ({codigo})"
                return nombre

            gdf_cruzado['Territorio'] = gdf_cruzado.apply(formatear_territorio, axis=1)

            # 8. SUMATORIA ESPACIAL
            cols_sumar = [col_caudal, 'caudal_domestico_lps', 'caudal_agricola_lps', 'caudal_pecuario_lps']
            df_agrupado = gdf_cruzado.groupby('Territorio')[cols_sumar].sum().reset_index()
            
            # Conversión a m3/s
            df_agrupado.rename(columns={col_caudal: 'Caudal_Concesionado_Ls'}, inplace=True)
            df_agrupado['Caudal_Concesionado_m3s'] = df_agrupado['Caudal_Concesionado_Ls'] / 1000.0
            df_agrupado['Caudal_Domestico_m3s'] = df_agrupado['caudal_domestico_lps'] / 1000.0
            df_agrupado['Caudal_Agricola_m3s'] = df_agrupado['caudal_agricola_lps'] / 1000.0
            df_agrupado['Caudal_Pecuario_m3s'] = df_agrupado['caudal_pecuario_lps'] / 1000.0
            df_agrupado['Caudal_Vertimiento_m3s'] = 0.0
            df_agrupado['Presion_Total_RURH_m3s'] = df_agrupado['Caudal_Concesionado_m3s'] + df_agrupado['Caudal_Vertimiento_m3s']

            # Mostrar resultados
            st.markdown("### 🗄️ Matriz Agrupada post-Cruce Espacial (SQL Puro)")
            format_dict = {c: '{:.4f}' for c in df_agrupado.columns if '_m3s' in c}
            st.dataframe(df_agrupado.head(15).style.format(format_dict), use_container_width=True)

            # 9. INYECCIÓN DUAL A POSTGRESQL
            # Extirpamos la columna 'geometry' porque PostgreSQL necesita el texto plano
            df_raw_sql = pd.DataFrame(gdf_cruzado.drop(columns=['geometry'])) 

            # 🔥 FIX ANTI-BLOQUEOS: 
            # Eliminamos el 'with engine.begin()' y el 'DROP TABLE' explícito.
            # Delegamos el control total de la transacción al motor de Pandas.
            df_agrupado.to_sql('matriz_presiones_rurh', engine, if_exists='replace', index=False)
            df_raw_sql.to_sql('concesiones_maestras_rurh_raw', engine, if_exists='replace', index=False)

            st.success(f"✅ ¡Fusión y Doble Inyección Exitosa! {len(df_agrupado)} cuencas consolidadas y {len(df_raw_sql)} concesiones ancladas usando el mapa oficial de PostGIS.")

        except Exception as e:
            st.error(f"🚨 Error durante la inyección o cruce espacial: {e}")
