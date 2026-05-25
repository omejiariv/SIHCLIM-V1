# pages/11_⚙️_Generador.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import tempfile
import zipfile
import io
import os
import warnings
import re
import unicodedata
import requests
import time


from modules import climate_api

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Generador Espacial", page_icon="⚙️", layout="wide")

st.title("⚙️ Centro de Geoprocesamiento y Transformación")
st.info("Herramientas de administrador para cruces espaciales, compresión y estandarización de cartografía web.")

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "AdminPoter"
                if clave_beta == st.secrets.get("CLAVE_BETA", "AdminPoter"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo
muro_de_acceso_beta()
# ==============================================================================

# --- ACTUALIZACIÓN: AGREGAMOS LA CUARTA PESTAÑA ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧩 1. Intersecciones Espaciales (Cuencas)", 
    "🔄 2. Convertidor GeoJSON (Soporta ZIP y Simplificación)", 
    "🗜️ 3. Compresor/Extractor ZIP",
    "🔁 4. Convertidor GeoJSON a SHP + Atributos",
    "🔗 5. Homologador de Veredas",
    "🌧️ 6. Fusión Hidrometeorológica"
])

# =====================================================================
# PESTAÑA 1: INTERSECCIÓN ESPACIAL (MUNICIPIOS O VEREDAS)
# =====================================================================
with tab1:
    st.subheader("Generador de Intersecciones Espaciales")
    st.write("Cruce de mapas para calcular la proporción de cada territorio dentro de las cuencas.")
    
    nivel_cruce = st.radio("Selecciona el Nivel Territorial a cruzar con las Cuencas:", ["Municipios", "Veredas"], horizontal=True)
    
    if st.button(f"🚀 Iniciar Cruce Espacial ({nivel_cruce})", type="primary"):
        with st.spinner("1. Cargando y proyectando mapas (EPSG:9377)..."):
            try:
                # 1. Cargar Cuencas (Siempre es el mismo)
                cuencas = gpd.read_file('data/SubcuencasAinfluencia.geojson')
                cuencas = cuencas.to_crs(epsg=9377)
                
                # 2. Cargar Capa Territorial según la selección
                if nivel_cruce == "Municipios":
                    territorio = gpd.read_file('data/mgn_municipios_optimizado.geojson')
                    col_nombre = 'MPIO_CNMBR'
                    col_padre = 'DPTO_CNMBR' # Opcional
                    nombre_salida = 'cuencas_mpios_proporcion.csv'
                else:
                    territorio = gpd.read_file('data/Veredas_Antioquia_TOTAL_UrbanoyRural.geojson')
                    col_nombre = 'NOMBRE_VER'
                    col_padre = 'NOMB_MPIO' # Necesario en veredas por homonimia
                    nombre_salida = 'cuencas_veredas_proporcion.csv'
                
                territorio = territorio.to_crs(epsg=9377)
                st.success(f"Mapas cargados. Cuencas: {len(cuencas)} | {nivel_cruce}: {len(territorio)}")

                # 3. Calcular área original del territorio
                territorio['Area_Original_Ha'] = territorio.geometry.area / 10000

                # 4. Intersección
                st.info("2. Ejecutando geoprocesamiento de intersección (Esto puede tardar unos minutos)...")
                interseccion = gpd.overlay(territorio, cuencas, how='intersection')

                # 5. Calcular nueva área y proporción
                interseccion['Area_Interseccion_Ha'] = interseccion.geometry.area / 10000
                interseccion['Porcentaje'] = (interseccion['Area_Interseccion_Ha'] / interseccion['Area_Original_Ha']) * 100
                
                # Limpiamos porcentajes mayores a 100 (errores de topología menores)
                interseccion['Porcentaje'] = interseccion['Porcentaje'].apply(lambda x: min(x, 100.0))

                # 6. Preparar tabla final
                if nivel_cruce == "Municipios":
                    df_final = interseccion[['MPIO_CNMBR', 'ZH', 'SUBC_LBL', 'N_NSS1', 'Area_Interseccion_Ha', 'Porcentaje']].copy()
                    df_final.columns = ['Municipio', 'Zona_Hidrografica', 'Subcuenca', 'Sistema', 'Area_Ha', 'Porcentaje']
                else:
                    df_final = interseccion[['NOMBRE_VER', 'NOMB_MPIO', 'ZH', 'SUBC_LBL', 'N_NSS1', 'Area_Interseccion_Ha', 'Porcentaje']].copy()
                    df_final.columns = ['Vereda', 'Municipio', 'Zona_Hidrografica', 'Subcuenca', 'Sistema', 'Area_Ha', 'Porcentaje']

                st.success("✅ Cruce completado con éxito.")
                st.dataframe(df_final.head())

                # 7. Botón de descarga
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Descargar {nombre_salida}",
                    data=csv,
                    file_name=nombre_salida,
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"🚨 Ocurrió un error en el cruce: {e}")

# =====================================================================
# PESTAÑA 2: EL TRANSFORMADOR DE SHAPEFILES (SOPORTA ZIP Y SIMPLIFICACIÓN)
# =====================================================================
with tab2:
    st.subheader("Transformador y Optimizador Web de Cartografía")
    st.markdown("Sube un archivo **.zip** (ej. de 65 MB) que contenga tu Shapefile. El sistema lo descomprimirá, lo simplificará para la web y te entregará un GeoJSON ligero.")
    
    modo_carga = st.radio("Método de Carga:", ["Subir archivo .zip (Recomendado)", "Subir archivos sueltos"], horizontal=True)
    
    # --- CONTROL DE SIMPLIFICACIÓN ---
    st.markdown("---")
    st.markdown("#### 📉 Optimización Topológica (Adelgazar Mapa)")
    simplificar = st.checkbox("Activar simplificación de fronteras (Crucial para mapas Nacionales grandes)", value=True)
    factor_simp = st.slider(
        "Tolerancia (Grados). Más alto = Más liviano pero bordes más rectos. 0.005 es ideal para Municipios.", 
        min_value=0.001, max_value=0.050, value=0.005, step=0.001, format="%.3f"
    )
    st.markdown("---")
    
    if modo_carga == "Subir archivo .zip (Recomendado)":
        archivo_zip = st.file_uploader("Sube tu archivo ZIP (debe contener un .shp adentro)", type=['zip'])
        
        if archivo_zip and st.button("⚙️ Descomprimir, Simplificar y Transformar"):
            with st.spinner("Procesando ZIP en la nube..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # 1. Extraer ZIP
                        with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                            
                        # 2. Buscar archivo .shp
                        ruta_shp = None
                        for root, dirs, files in os.walk(tmpdir):
                            for file in files:
                                if file.lower().endswith('.shp'):
                                    ruta_shp = os.path.join(root, file)
                                    break
                            if ruta_shp: break
                            
                        if not ruta_shp:
                            st.error("❌ No se encontró ningún archivo .shp dentro del ZIP.")
                        else:
                            # 3. Leer con GeoPandas
                            st.toast("Leyendo polígonos...")
                            gdf = gpd.read_file(ruta_shp)
                            
                            # 4. Proyección a WGS84 (Web)
                            if gdf.crs is None: gdf.set_crs(epsg=3116, inplace=True)
                            if gdf.crs.to_string() != "EPSG:4326": 
                                st.toast("Reproyectando coordenadas a WGS84...")
                                gdf = gdf.to_crs(epsg=4326)
                            
                            # 5. SIMPLIFICACIÓN TOPOLÓGICA MÁGICA
                            if simplificar:
                                st.toast(f"Simplificando geometrías (Tolerancia: {factor_simp})...")
                                gdf['geometry'] = gdf['geometry'].simplify(tolerance=factor_simp, preserve_topology=True)
                            
                            st.toast("Convirtiendo a GeoJSON...")
                            geojson_data = gdf.to_json()
                            nombre_base = os.path.basename(ruta_shp).replace('.shp', '').replace('.SHP', '') + "_optimizado"
                            
                            st.success(f"✅ ¡Capa '{nombre_base}' lista! ({len(gdf)} polígonos procesados).")
                            st.download_button(
                                label=f"📥 Descargar {nombre_base}.geojson",
                                data=geojson_data,
                                file_name=f"{nombre_base}.geojson",
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"❌ Error procesando el ZIP: {str(e)}")
                    
    else: # Modo archivos sueltos
        archivos_subidos = st.file_uploader("Selecciona archivos (Mínimo .shp, .shx y .dbf)", accept_multiple_files=True)
        if archivos_subidos:
            archivo_shp = next((f for f in archivos_subidos if f.name.lower().endswith('.shp')), None)
            
            if archivo_shp and st.button("⚙️ Transformar sueltos a GeoJSON"):
                with st.spinner("Procesando archivos sueltos..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            for f in archivos_subidos:
                                with open(os.path.join(tmpdir, f.name), "wb") as f_out:
                                    f_out.write(f.getvalue())
                                    
                            ruta_shp_temporal = os.path.join(tmpdir, archivo_shp.name)
                            gdf = gpd.read_file(ruta_shp_temporal)
                            
                            if gdf.crs is None: gdf.set_crs(epsg=3116, inplace=True)
                            if gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
                            
                            if simplificar:
                                gdf['geometry'] = gdf['geometry'].simplify(tolerance=factor_simp, preserve_topology=True)
                            
                            geojson_data = gdf.to_json()
                            nombre_base = archivo_shp.name.replace('.shp', '').replace('.SHP', '') + "_optimizado"
                            
                            st.success(f"✅ ¡Capa estandarizada! ({len(gdf)} registros).")
                            st.download_button(
                                label=f"📥 Descargar {nombre_base}.geojson", data=geojson_data,
                                file_name=f"{nombre_base}.geojson", mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            elif not archivo_shp:
                st.warning("⚠️ Falta el archivo principal .shp en tu selección.")

# =====================================================================
# PESTAÑA 3: UTILIDAD DE COMPRESIÓN / DESCOMPRESIÓN ZIP
# =====================================================================
with tab3:
    st.subheader("🗜️ Gestor de Archivos ZIP")
    
    col_zip1, col_zip2 = st.columns(2)
    
    with col_zip1:
        st.markdown("### 📦 Comprimir Archivos")
        st.markdown("Sube varios archivos sueltos para empaquetarlos en un solo `.zip`.")
        archivos_a_comprimir = st.file_uploader("Selecciona archivos a comprimir", accept_multiple_files=True, key="uploader_comp")
        
        if archivos_a_comprimir:
            nombre_zip = st.text_input("Nombre del archivo final:", value="mapas_comprimidos")
            if st.button("📦 Crear Archivo ZIP"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for archivo in archivos_a_comprimir:
                        zip_file.writestr(archivo.name, archivo.getvalue())
                
                st.success("✅ ZIP creado exitosamente.")
                st.download_button(
                    label=f"📥 Descargar {nombre_zip}.zip",
                    data=zip_buffer.getvalue(),
                    file_name=f"{nombre_zip}.zip",
                    mime="application/zip"
                )

    with col_zip2:
        st.markdown("### 📂 Descomprimir ZIP")
        st.markdown("Sube un archivo `.zip` para extraer su contenido.")
        archivo_a_descomprimir = st.file_uploader("Sube un archivo ZIP", type=['zip'], key="uploader_desc")
        
        if archivo_a_descomprimir:
            with zipfile.ZipFile(archivo_a_descomprimir, 'r') as zip_ref:
                lista_archivos = zip_ref.namelist()
                st.write(f"**Contiene {len(lista_archivos)} archivos:**")
                for nombre_archivo in lista_archivos:
                    if not nombre_archivo.endswith('/'): 
                        datos_archivo = zip_ref.read(nombre_archivo)
                        st.download_button(
                            label=f"⬇️ {os.path.basename(nombre_archivo)}",
                            data=datos_archivo,
                            file_name=os.path.basename(nombre_archivo),
                            key=f"desc_{nombre_archivo}"
                        )

# =====================================================================
# PESTAÑA 4: CONVERTIDOR GEOJSON A SHAPEFILE + VISOR DE ATRIBUTOS
# =====================================================================
with tab4:
    st.subheader("🔁 Convertidor GeoJSON a Shapefile (SHP) y Visor de Atributos")
    st.markdown("Sube un archivo **.geojson** o **.json**. Podrás previsualizar su tabla de atributos y exportarlo como un Shapefile listo para software GIS (QGIS/ArcGIS).")
    
    archivo_geojson_in = st.file_uploader("Sube tu archivo GeoJSON", type=['geojson', 'json'], key="up_geo2shp")
    
    if archivo_geojson_in:
        with st.spinner("Leyendo estructura del archivo espacial..."):
            try:
                # 1. Leer el GeoJSON
                gdf_in = gpd.read_file(archivo_geojson_in)
                
                # 2. Mostrar Información de la Tabla
                st.success(f"✅ Archivo cargado correctamente. Contiene **{len(gdf_in)}** geometrías/polígonos.")
                
                st.markdown("#### 📋 Estructura de Campos (Atributos)")
                
                # Extraer nombres de campos y tipos de datos (ignorando la geometría)
                df_info = pd.DataFrame({
                    "Campo": gdf_in.columns,
                    "Tipo de Dato": gdf_in.dtypes.astype(str)
                })
                df_info = df_info[df_info["Campo"] != "geometry"].reset_index(drop=True)
                
                col_info1, col_info2 = st.columns([1, 2.5])
                with col_info1:
                    st.dataframe(df_info, use_container_width=True)
                
                with col_info2:
                    st.markdown("**Vista Previa de los Datos (Primeras 5 filas):**")
                    # Mostrar el dataframe sin la columna geometry para que sea ligero visualmente
                    df_vista = pd.DataFrame(gdf_in.drop(columns='geometry'))
                    st.dataframe(df_vista.head(), use_container_width=True)
                
                # 3. Conversión a Shapefile empaquetado en ZIP
                st.markdown("---")
                st.markdown("#### 📦 Exportar a Shapefile (.shp)")
                st.info("💡 **Nota:** El formato Shapefile exige que esté compuesto por varios archivos complementarios (.shp, .shx, .dbf, .prj, etc.). Te entregaremos un archivo **.zip** con todos los componentes debidamente empaquetados.")
                
                nombre_export = st.text_input("Nombre base para el archivo final:", value="Capa_Exportada")
                
                if st.button("🚀 Convertir y Empaquetar a SHP", type="primary"):
                    with st.spinner("Generando y empaquetando archivos Shapefile..."):
                        
                        # Usamos un directorio temporal para guardar los múltiples archivos del SHP
                        with tempfile.TemporaryDirectory() as tmpdir:
                            shp_path = os.path.join(tmpdir, f"{nombre_export}.shp")
                            
                            # GeoPandas exporta el SHP. Advierte si los nombres superan los 10 caracteres (limitación del formato SHP)
                            gdf_in.to_file(shp_path, driver='ESRI Shapefile')
                            
                            # Empaquetamos todo el contenido del directorio temporal en un ZIP en memoria
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                                for root, dirs, files in os.walk(tmpdir):
                                    for file in files:
                                        filepath = os.path.join(root, file)
                                        zip_file.write(filepath, arcname=file)
                            
                            st.success("✅ ¡Geometría y tabla de atributos empaquetados exitosamente!")
                            st.download_button(
                                label=f"📥 Descargar {nombre_export}.zip",
                                data=zip_buffer.getvalue(),
                                file_name=f"{nombre_export}.zip",
                                mime="application/zip"
                            )
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo GeoJSON: {str(e)}")

# =====================================================================
# PESTAÑA 5: Generador de Diccionario Maestro (Veredas)
# =====================================================================

    with tab5:
        st.markdown("### 🔗 Generador de Diccionario Maestro (Veredas)")
        st.info("Cruza las bases tabulares (DANE) con las espaciales (GeoJSON) usando el Código DANE para generar el diccionario `homologacion_veredas.csv`.")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            file_dane = st.file_uploader("1. Archivo Matriz (ej. veredas_dane.csv)", type=['csv'], key="dane_csv")
            st.caption("Debe tener columnas: `codigo_vereda`, `NOMB_MPIO`, `NOMBRE_VER`")
        with col_h2:
            file_gis = st.file_uploader("2. Archivo Espacial (ej. veredas_gisant.csv)", type=['csv'], key="gis_csv")
            st.caption("Debe tener columnas: `codigo_vereda`, `NOMB_MPIO`, `NOMBRE_VER`")

        if file_dane and file_gis:
            if st.button("🚀 Generar Diccionario de Homologación", type="primary", use_container_width=True):
                with st.spinner("Realizando exorcismo de datos y cruzando bases..."):
                    try:
                        # 🛡️ LECTOR MEGA-BLINDADO (Anti-BOM y Anti-Punto y Coma)
                        def leer_csv_robusto(file_obj):
                            file_obj.seek(0)
                            # Detectar separador manualmente leyendo una muestra
                            sample = file_obj.read(1000).decode('utf-8', errors='ignore')
                            sep_char = ';' if ';' in sample else ','
                            file_obj.seek(0)
                            
                            try:
                                # utf-8-sig DESTRUYE el carácter fantasma \ufeff de Excel
                                return pd.read_csv(file_obj, sep=sep_char, encoding='utf-8-sig')
                            except UnicodeDecodeError:
                                file_obj.seek(0)
                                return pd.read_csv(file_obj, sep=sep_char, encoding='latin1')

                        # 1. Leer archivos
                        df_dane = leer_csv_robusto(file_dane)
                        df_gis = leer_csv_robusto(file_gis)

                        # 2. Exorcismo de columnas (Eliminar TODO rastro de caracteres invisibles)
                        df_dane.columns = [str(c).replace('\ufeff', '').replace('\xef\xbb\xbf', '').strip() for c in df_dane.columns]
                        df_gis.columns = [str(c).replace('\ufeff', '').replace('\xef\xbb\xbf', '').strip() for c in df_gis.columns]

                        # 3. Forzar el nombre de la llave (Si dice "codigo" en algún lado, lo volvemos codigo_vereda)
                        df_dane.columns = ['codigo_vereda' if 'codigo' in c.lower() else c for c in df_dane.columns]
                        df_gis.columns = ['codigo_vereda' if 'codigo' in c.lower() else c for c in df_gis.columns]

                        # 4. Asegurar que sean strings limpios de 8 dígitos
                        df_dane['codigo_vereda'] = df_dane['codigo_vereda'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
                        df_gis['codigo_vereda'] = df_gis['codigo_vereda'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)

                        # 5. Cruzar bases
                        df_cruce = pd.merge(df_dane, df_gis, on='codigo_vereda', suffixes=('_DANE', '_GIS'))

                        # Verificar si cruzaron
                        if df_cruce.empty:
                            st.error("❌ Los archivos se leyeron bien, pero no hay códigos DANE que coincidan. Revisa los datos de las columnas de código.")
                            st.stop()

                        # Función limpiadora interna
                        def limpiar_texto(t, municipio=""):
                            if not isinstance(t, str): return ""
                            t = str(t).upper().strip()
                            t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
                            for word in [r'\bVEREDA\b', r'\bVDA\.?\b', r'\bSECTOR\b', r'\bCASERIO\b']:
                                t = re.sub(word, '', t)
                            if municipio:
                                mun = ''.join(c for c in unicodedata.normalize('NFD', str(municipio).upper()) if unicodedata.category(c) != 'Mn')
                                id_final = t + "_" + mun
                            else:
                                id_final = t
                            return re.sub(r'[^A-Z0-9_]', '', id_final)

                        # Construir los ADN
                        df_cruce['ID_TABLA'] = df_cruce.apply(lambda row: limpiar_texto(row.get('NOMBRE_VER_DANE', ''), row.get('NOMB_MPIO_DANE', '')), axis=1)
                        df_cruce['ID_MAPA'] = df_cruce.apply(lambda row: limpiar_texto(row.get('NOMBRE_VER_GIS', ''), row.get('NOMB_MPIO_GIS', '')), axis=1)

                        # Extraer rebeldes
                        df_rebeldes = df_cruce[df_cruce['ID_TABLA'] != df_cruce['ID_MAPA']]
                        df_final = df_rebeldes[['ID_TABLA', 'ID_MAPA']].drop_duplicates()

                        st.success(f"✅ ¡Éxito Total! Se cruzaron los datos y se detectaron {len(df_final)} discrepancias territoriales.")
                        st.dataframe(df_final, use_container_width=True)

                        # Botón de descarga
                        csv_dict = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar homologacion_veredas.csv",
                            data=csv_dict,
                            file_name="homologacion_veredas.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        st.info("💡 Instrucción: Descarga este archivo y guárdalo en la carpeta `data/` de tu proyecto. El `Modelo Demográfico` lo leerá automáticamente para arreglar los mapas.")

                    except Exception as e:
                        st.error(f"❌ Error interno durante el procesamiento: {e}")

# ------------------------------------------------------------------------------
# 🚀 PESTAÑA 6: GEMELO DIGITAL (CONSOLIDADOR MAESTRO DE REDES)
# ------------------------------------------------------------------------------
with tab6:
    st.header("🌧️ Gemelo Digital: Súper-Consolidador de Redes Hidrometeorológicas")
    st.markdown("""
    Este motor unifica todo tu arsenal de datos en una única base de datos coherente:
    1. Combina secuencialmente el **Histórico ENSO**, la **Transición 2010-2025** y las **Automáticas API en Tiempo Real**.
    2. Aplica el **Bisturí Temporal 2010** para homogeneizar sensores modernos con líneas base antiguas.
    3. Levanta el **Gemelo Digital** para imputar vacíos mediante correlación espacial de vecindad (>0.65) y fases ENSO.
    """)
    
    if 'csv_fusionado_data' not in st.session_state:
        st.session_state['csv_fusionado_data'] = None
        st.session_state['fusion_resumen'] = ""
        st.session_state['fusion_preview'] = None

    # Interfaz organizada para los 5 flujos de datos posibles
    col_1, col_2 = st.columns(2)
    with col_1:
        file_maestro = st.file_uploader("🥇 1. Histórico IDEAM Base (1970-2020)", type=['csv'])
        file_auto_ideam = st.file_uploader("🥉 3. IDEAM Automáticas API (Bot Matriz)", type=['csv'])
    with col_2:
        file_parche_1 = st.file_uploader("🥈 2. IDEAM Transición Reciente (2010-2025)", type=['csv'])
        file_satelital = st.file_uploader("🚀 4. Satelital Copernicus (Opcional)", type=['csv'])
        file_calibracion = st.file_uploader("📥 Matriz Calibración Copernicus (Opcional)", type=['csv'])
        
    usar_imputacion = st.checkbox("Activar Imputación Avanzada del Gemelo Digital (Espacial + ENSO)", value=True)
        
    # Exigimos los 3 archivos terrestres como mínimo para encender el motor
    if file_maestro and file_parche_1 and file_auto_ideam:
        if st.button("🔄 Ejecutar Fusión e Imputación Maestra", type="primary", use_container_width=True):
            with st.spinner("1. Tejiendo la red del IDEAM (Histórica + Transición + Automática)..."):
                try:
                    def leer_csv_seguro(file_obj):
                        if file_obj is None: return None
                        try:
                            return pd.read_csv(file_obj, sep=';', encoding='utf-8')
                        except UnicodeDecodeError:
                            file_obj.seek(0)
                            return pd.read_csv(file_obj, sep=';', encoding='latin1')
                            
                    df_m = leer_csv_seguro(file_maestro)
                    df_p1 = leer_csv_seguro(file_parche_1)
                    df_auto = leer_csv_seguro(file_auto_ideam)
                    df_sat = leer_csv_seguro(file_satelital) 

                    # 🧹 ADUANA DE DATOS Y ESTANDARIZACIÓN
                    import numpy as np
                    codigos_falsos = [999.9, 999.0, 999, 9999.9, 9999.0, 9999, -99.9, -99.0, -999.0, -9999.0]
                    dfs_procesados = []
                    
                    archivos_activos = [df for df in [df_m, df_p1, df_auto, df_sat] if df is not None]
                    
                    for df_temp in archivos_activos:
                        # 🚨 CINTURÓN DE SEGURIDAD: Obligar a que todos los códigos sean texto limpio
                        df_temp.columns = [str(c).strip() for c in df_temp.columns]
                        
                        if 'date' in df_temp.columns: df_temp.rename(columns={'date': 'fecha'}, inplace=True)
                        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'], errors='coerce')
                        df_temp.dropna(subset=['fecha'], inplace=True)
                        df_temp['fecha'] = df_temp['fecha'].dt.to_period('M').dt.to_timestamp()
                        df_temp = df_temp.groupby('fecha').first().reset_index()
                        df_temp.set_index('fecha', inplace=True)
                        
                        # Ahora sí podemos buscar números de forma segura
                        cols_est = [c for c in df_temp.columns if c.isnumeric()]
                        for col in cols_est:
                            if df_temp[col].dtype == object:
                                df_temp[col] = df_temp[col].astype(str).str.replace(',', '.', regex=False)
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                            df_temp[col] = df_temp[col].replace(codigos_falsos, np.nan)
                            df_temp.loc[(df_temp[col] < 0) | (df_temp[col] > 1500.0), col] = np.nan
                            
                        dfs_procesados.append(df_temp)
                        
                    # Despaquetar fuentes terrestres limpias
                    df_m_clean = dfs_procesados[0]
                    df_p1_clean = dfs_procesados[1]
                    df_auto_clean = dfs_procesados[2]

                    # ==========================================================
                    # 🌍 PASO 1: CONSOLIDACIÓN SECUENCIAL DE LA RED TERRESTE
                    # Combina Histórico -> parche 2010-2025 -> Automáticas nuevas
                    # ==========================================================
                    df_terrestre = df_m_clean.combine_first(df_p1_clean).combine_first(df_auto_clean)
                    
                    # Capturar límites de nacimiento
                    cols_totales = [c for c in df_terrestre.columns if str(c).isnumeric()]
                    limites_nacimiento = {}
                    for col in cols_totales:
                        datos_validos = df_terrestre[col].dropna()
                        limites_nacimiento[col] = datos_validos.index.min() if not datos_validos.empty else None

                    # ==========================================================
                    # ⚖️ PASO 2: BISTURÍ TEMPORAL (HOMOGENEIZACIÓN DE ERAS)
                    # ==========================================================
                    AÑO_RUPTURA = 2010
                    log_homogeneidad = 0
                    
                    for col in cols_totales:
                        serie_hist = df_terrestre.loc[df_terrestre.index.year < AÑO_RUPTURA, col].dropna()
                        serie_reciente = df_terrestre.loc[df_terrestre.index.year >= AÑO_RUPTURA, col].dropna()
                        
                        if len(serie_hist) > 60 and len(serie_reciente) > 24:
                            media_hist = serie_hist[serie_hist > 0].mean()
                            media_reciente = serie_reciente[serie_reciente > 0].mean()
                            
                            if pd.notna(media_hist) and pd.notna(media_reciente) and media_reciente > 0:
                                factor = media_hist / media_reciente
                                if abs(1 - factor) > 0.12:
                                    factor = max(0.35, min(factor, 2.5))
                                    df_terrestre.loc[df_terrestre.index.year >= AÑO_RUPTURA, col] = df_terrestre.loc[df_terrestre.index.year >= AÑO_RUPTURA, col] * factor
                                    log_homogeneidad += 1
                                    
                    st.success(f"⚖️ Homogeneidad Terrestre: Se calibró el bloque moderno de {log_homogeneidad} estaciones.")
                    
                    df_final = df_terrestre.copy()
                    mensaje_estatus = "Ecosistema unificado basado en 3 redes del IDEAM."

                    # ==========================================================
                    # 🛰️ PASO 3: INGESTIÓN SATELITAL (OPCIONAL)
                    # ==========================================================
                    if df_sat is not None:
                        df_satelite = dfs_procesados[3]
                        st.info("🛰️ Datos satelitales detectados. Integrando capa adicional...")
                        
                        df_cal = None
                        if file_calibracion is not None:
                            try:
                                df_cal = pd.read_csv(file_calibracion, sep=';', decimal=',')
                                df_cal['Estacion'] = df_cal['Estacion'].astype(str).str.strip()
                                df_cal.set_index('Estacion', inplace=True)
                            except Exception: pass
                            
                        cols_comunes_sat = [c for c in df_satelite.columns if c in df_terrestre.columns and str(c).isnumeric()]
                        
                        for col in cols_comunes_sat:
                            aplicado_eq = False
                            if df_cal is not None and col in df_cal.index:
                                m, b, r2 = df_cal.loc[col, 'Pendiente_m'], df_cal.loc[col, 'Intercepto_b'], df_cal.loc[col, 'R2']
                                if pd.notna(m) and pd.notna(b) and pd.notna(r2) and r2 >= 0.2: 
                                    df_satelite[col] = (df_satelite[col] * m) + b
                                    df_satelite[col] = df_satelite[col].clip(lower=0) 
                                    aplicado_eq = True
                            
                            if not aplicado_eq:
                                hist_terr = df_terrestre[df_terrestre[col] > 0][col]
                                hist_s = df_satelite[df_satelite[col] > 0][col]
                                if not hist_terr.empty and not hist_s.empty:
                                    df_satelite[col] = df_satelite[col] * max(0.3, min(hist_terr.mean() / hist_s.mean(), 3.0))
                                    
                        df_final = df_terrestre.combine_first(df_satelite)
                        mensaje_estatus = "Ecosistema unificado Híbrido (3 Redes IDEAM + Copernicus)."

                    df_final = df_final[df_final.index <= pd.Timestamp.today()]

                    # ==========================================================
                    # 🧩 PASO 4: GEMELO DIGITAL (IMPUTACIÓN ESPACIAL)
                    # ==========================================================
                    cols_estaciones = [c for c in df_final.columns if str(c).isnumeric()]
                    
                    if usar_imputacion:
                        with st.spinner("2. Levantando Gemelo Digital: Interconectando estaciones..."):
                            from modules import climate_api  
                            
                            Q1_mes = df_final.groupby(df_final.index.month)[cols_estaciones].transform(lambda x: x.quantile(0.25))
                            Q3_mes = df_final.groupby(df_final.index.month)[cols_estaciones].transform(lambda x: x.quantile(0.75))
                            techo_mensual = Q3_mes + (3 * (Q3_mes - Q1_mes))

                            for col in cols_estaciones:
                                df_final.loc[df_final[col] > np.maximum(250.0, techo_mensual[col]), col] = np.nan

                            MIN_DATOS_REALES = 24
                            conteo_reales = df_final[cols_estaciones].notna().sum()
                            estaciones_robustas = conteo_reales[conteo_reales >= MIN_DATOS_REALES].index.tolist()

                            if estaciones_robustas:
                                matriz_corr = df_final[estaciones_robustas].corr(method='pearson')
                                climatologia_mensual = df_final.groupby(df_final.index.month)[estaciones_robustas].mean()
                                UMBRAL_CORRELACION = 0.65
                                
                                df_imputado_espacial = df_final[cols_estaciones].copy()
                                
                                for estacion in estaciones_robustas:
                                    registros_vacios = df_final[df_final[estacion].isna()].index
                                    if len(registros_vacios) > 0:
                                        vecinas_ordenadas = matriz_corr[estacion].drop(estacion).sort_values(ascending=False)
                                        vecinas_validas = vecinas_ordenadas[vecinas_ordenadas >= UMBRAL_CORRELACION].index.tolist()
                                        
                                        for fecha_idx in registros_vacios:
                                            mes_actual = fecha_idx.month
                                            for vecina in vecinas_validas:
                                                valor_vecina = df_final.loc[fecha_idx, vecina]
                                                if not np.isnan(valor_vecina):
                                                    media_target = climatologia_mensual.loc[mes_actual, estacion]
                                                    media_vecina = climatologia_mensual.loc[mes_actual, vecina]
                                                    if media_vecina > 0:
                                                        df_imputado_espacial.loc[fecha_idx, estacion] = valor_vecina * (media_target / media_vecina)
                                                        break 
                                
                                df_final[cols_estaciones] = df_imputado_espacial

                            if estaciones_robustas:
                                df_clima_vivo = climate_api.get_live_oni_data()
                                if df_clima_vivo is not None and not df_clima_vivo.empty:
                                    df_clima_vivo.set_index('fecha', inplace=True)
                                    df_final = df_final.join(df_clima_vivo[['fase_enso']], how='left')
                                    df_promedios_enso = df_final.groupby([df_final.index.month, 'fase_enso'])[estaciones_robustas].transform('mean')
                                    df_final[estaciones_robustas] = df_final[estaciones_robustas].fillna(df_promedios_enso)
                                    df_final.drop(columns=['fase_enso'], inplace=True)
                                
                                df_promedios_mensuales = df_final.groupby(df_final.index.month)[estaciones_robustas].transform('mean')
                                df_final[estaciones_robustas] = df_final[estaciones_robustas].fillna(df_promedios_mensuales)

                    # 🛡️ CORTAFUEGOS DE NACIMIENTO
                    if cols_estaciones:
                        for col in cols_estaciones:
                            inicio = limites_nacimiento.get(col)
                            if inicio:
                                df_final.loc[df_final.index < inicio, col] = np.nan
                        df_final.dropna(subset=cols_estaciones, how='all', inplace=True)
                    
                    df_final = df_final.sort_index().reset_index()
                    df_final['fecha'] = df_final['fecha'].dt.strftime('%Y-%m-%d')
                    
                    st.session_state['csv_fusionado_data'] = df_final.to_csv(sep=';', decimal=',', index=False, encoding='utf-8-sig').encode('utf-8')
                    st.session_state['fusion_resumen'] = f"✅ Operación Exitosa. {mensaje_estatus}"
                    st.session_state['fusion_preview'] = df_final.tail(10)
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error procesando los archivos: {e}")

    # Renderizado
    if st.session_state['csv_fusionado_data'] is not None:
        st.markdown("---")
        st.success("✅ ¡Súper-Matriz Unificada Generada con Éxito!")
        st.info(st.session_state['fusion_resumen'])
        st.dataframe(st.session_state['fusion_preview'], use_container_width=True)
        
        st.download_button(
            label="📥 Descargar Matriz Definitiva Total (Apta para Supabase)",
            data=st.session_state['csv_fusionado_data'],
            file_name="DatosPptnmes_Maestro_Integral.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
# Extraer datos del IDEAM

def extraer_ideam_datos_abiertos(lista_estaciones, limite_registros=50000):
    """
    Se conecta al API de Datos Abiertos de Colombia (Socrata) para extraer 
    registros de precipitación del IDEAM.
    """
    print("📡 Iniciando conexión con el portal de Datos Abiertos del Gobierno...")
    
    # El ID del dataset oficial del IDEAM en datos.gov.co (Catálogo de precipitación)
    # Nota: Este ID corresponde a Catálogo de Datos Climáticos, puede requerir ajuste 
    # si el IDEAM migra el servidor, pero es el punto de entrada estándar.
    DATASET_ID = "s54a-sgyg"
    URL_BASE = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"
    
    resultados = []
    
    for estacion in lista_estaciones:
        print(f"Buscando estación: {estacion}...")
        
        # Construimos la consulta SoQL (Socrata Query Language)
        parametros = {
            "codigoestacion": str(estacion),
            "$limit": limite_registros,
            "$order": "fechaobservacion ASC"
        }
        
        try:
            respuesta = requests.get(URL_BASE, params=parametros)
            
            if respuesta.status_code == 200:
                datos = respuesta.json()
                if datos:
                    df_temp = pd.DataFrame(datos)
                    resultados.append(df_temp)
                    print(f"  ✅ {len(datos)} registros encontrados.")
                else:
                    print("  ⚠️ Sin datos recientes en la API.")
            else:
                print(f"  ❌ Error de conexión: {respuesta.status_code}")
                
        except Exception as e:
            print(f"  ❌ Fallo crítico en estación {estacion}: {e}")
            
        # Pausa táctica para no saturar el servidor del gobierno
        time.sleep(1)
        
    if resultados:
        df_final = pd.concat(resultados, ignore_index=True)
        print("\n✅ Extracción completada.")
        return df_final
    else:
        return pd.DataFrame()

# ==========================================
# 🚀 ZONA DE PRUEBAS
# ==========================================
if __name__ == "__main__":
    # Pon aquí 3 o 4 códigos de estaciones que sepas que tienen vacíos post-2010
    mis_estaciones = ["23060110", "23080950", "27010810"] 
    
    df_descarga = extraer_ideam_datos_abiertos(mis_estaciones)
    
    if not df_descarga.empty:
        # Guardamos el botín para inspeccionarlo
        df_descarga.to_csv("Botin_IDEAM_Directo.csv", index=False, sep=';', encoding='utf-8')
        print(df_descarga.head())
