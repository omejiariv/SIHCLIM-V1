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
# 🚀 PESTAÑA 6: FUSIÓN HIDROMETEOROLÓGICA EN CASCADA (AUTÓNOMA)
# ------------------------------------------------------------------------------
with tab6:
    st.header("🌧️ Fusión Inteligente en Cascada (5 Capas)")
    st.markdown("""
    Esta herramienta rellena vacíos hidrometeorológicos usando un arsenal jerárquico. 
    **El nivel superior jamás se sobrescribe**, solo se rellenan sus `NaN` (huecos) con el nivel inferior.
    """)
    
    if 'csv_fusionado_data' not in st.session_state:
        st.session_state['csv_fusionado_data'] = None
        st.session_state['fusion_resumen'] = ""
        st.session_state['fusion_preview'] = None

    # Reorganizamos en 2 columnas para que quepan los 4 archivos
    col_1, col_2 = st.columns(2)
    with col_1:
        file_maestro = st.file_uploader("🥇 1. Histórico In Situ", type=['csv'], key='up_maestro')
        file_parche_2 = st.file_uploader("🥉 3. Satelital (Copérnicus)", type=['csv'], key='up_parche2')
    with col_2:
        file_parche_1 = st.file_uploader("🥈 2. Institucional (IDEAM)", type=['csv'], key='up_parche1')
        file_indices = st.file_uploader("🌍 4. Índices (NOAA/ONI) *Para Imputación Inteligente*", type=['csv'], key='up_indices')
        
    st.markdown("---")
    st.markdown("#### 🧩 Arsenal Matemático (Capas de Imputación)")
    usar_imputacion = st.checkbox(
        "Activar Imputación Climatológica (Capa 4 & 5): Rellena baches calculando promedios por fase ENSO.", 
        value=True
    )
        
    if file_maestro and file_parche_1 and file_parche_2:
        if st.button("🔄 Ejecutar Fusión Integral", type="primary", use_container_width=True):
            with st.spinner("1. Limpiando anomalías, alineando fechas y cruzando estaciones..."):
                try:
                    def leer_csv_seguro(file_obj):
                        try:
                            return pd.read_csv(file_obj, sep=';', encoding='utf-8')
                        except UnicodeDecodeError:
                            file_obj.seek(0)
                            return pd.read_csv(file_obj, sep=';', encoding='latin1')
                            
                    df_m = leer_csv_seguro(file_maestro)
                    df_p1 = leer_csv_seguro(file_parche_1)
                    df_p2 = leer_csv_seguro(file_parche_2)

                    for df_temp in [df_m, df_p1, df_p2]:
                        if 'date' in df_temp.columns:
                            df_temp.rename(columns={'date': 'fecha'}, inplace=True)
                            
                    if 'fecha' not in df_m.columns or 'fecha' not in df_p1.columns or 'fecha' not in df_p2.columns:
                        st.error("❌ Los tres archivos deben contener una columna llamada 'fecha' o 'date'.")
                        st.stop()

                    # ==========================================================
                    # 🧹 ADUANA DE DATOS ENTRADA
                    # ==========================================================
                    import numpy as np
                    codigos_falsos = [999.9, 999.0, 999, 9999.9, 9999.0, 9999, -99.9, -99.0, -999.0, -9999.0]
                    UMBRAL_MAX_PPT = 1200.0  
                    
                    dfs_procesados = []
                    for df_temp in [df_m, df_p1, df_p2]:
                        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'], errors='coerce')
                        df_temp.dropna(subset=['fecha'], inplace=True)
                        df_temp['fecha'] = df_temp['fecha'].dt.to_period('M').dt.to_timestamp()
                        df_temp = df_temp.groupby('fecha').first().reset_index()
                        df_temp.set_index('fecha', inplace=True)
                        
                        cols_est = [c for c in df_temp.columns if str(c).isnumeric()]
                        for col in cols_est:
                            if df_temp[col].dtype == object:
                                df_temp[col] = df_temp[col].astype(str).str.replace(',', '.', regex=False)
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                            df_temp[col] = df_temp[col].replace(codigos_falsos, np.nan)
                            df_temp.loc[(df_temp[col] < 0) | (df_temp[col] > UMBRAL_MAX_PPT), col] = np.nan
                            
                        dfs_procesados.append(df_temp)
                        
                    df_m, df_p1, df_p2 = dfs_procesados
                    df_final = df_m.combine_first(df_p1).combine_first(df_p2)
                    df_final = df_final[df_final.index <= pd.Timestamp.today()]
                    
                    # ==========================================================
                    # 🧩 IMPUTACIÓN HÍBRIDA (CORRELACIÓN ESPACIAL + CLIMÁTICA ENSO)
                    # ==========================================================
                    cols_estaciones = [c for c in df_final.columns if str(c).isnumeric()]
                    
                    # 🛡️ NUEVO: CAPTURAR EL "CICLO DE VIDA" DE CADA ESTACIÓN
                    limites_vida = {}
                    for col in cols_estaciones:
                        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                        datos_validos = df_final[col].dropna()
                        if not datos_validos.empty:
                            limites_vida[col] = (datos_validos.index.min(), datos_validos.index.max())
                        else:
                            limites_vida[col] = (None, None)
                    
                    if usar_imputacion:
                        with st.spinner("2. Forjando Matriz de Correlación e Imputando Espacialmente..."):
                            from modules import climate_api  
                            import numpy as np
                            
                            # Forzar conversión a numérico puro
                            for col in cols_estaciones:
                                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

                            # --------------------------------------------------
                            # CAPA A: FILTRO DE TUKEY BI-DIMENSIONAL (Limpieza previa)
                            # --------------------------------------------------
                            Q1_global = df_final[cols_estaciones].quantile(0.25)
                            Q3_global = df_final[cols_estaciones].quantile(0.75)
                            IQR_global = Q3_global - Q1_global
                            techo_global = Q3_global + (4 * IQR_global) 

                            Q1_mes = df_final.groupby(df_final.index.month)[cols_estaciones].transform(lambda x: x.quantile(0.25))
                            Q3_mes = df_final.groupby(df_final.index.month)[cols_estaciones].transform(lambda x: x.quantile(0.75))
                            IQR_mes = Q3_mes - Q1_mes
                            techo_mensual = Q3_mes + (3 * IQR_mes)

                            for col in cols_estaciones:
                                limite_final = np.maximum(250.0, np.minimum(techo_global[col], techo_mensual[col]))
                                df_final.loc[df_final[col] > limite_final, col] = np.nan

                            # --------------------------------------------------
                            # 🛡️ DEFINICIÓN DE ESTACIONES ROBUSTAS (El eslabón perdido)
                            # --------------------------------------------------
                            MIN_DATOS_REALES = 24
                            conteo_reales = df_final[cols_estaciones].notna().sum()
                            estaciones_robustas = conteo_reales[conteo_reales >= MIN_DATOS_REALES].index.tolist()

                            # --------------------------------------------------
                            # 🔥 CAPA B: MOTOR DE CORRELACIÓN ESPACIAL PROPORCIONAL
                            # --------------------------------------------------
                            st.text("📡 Calculando coeficientes de Pearson de la red real...")
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
                                                        valor_imputado = valor_vecina * (media_target / media_vecina)
                                                        df_imputado_espacial.loc[fecha_idx, estacion] = valor_imputado
                                                        break 
                                
                                df_final[cols_estaciones] = df_imputado_espacial

                            # --------------------------------------------------
                            # CAPA C: RED DE SEGURIDAD CLIMÁTICA (ENSO / Mensual)
                            # --------------------------------------------------
                            if estaciones_robustas:
                                df_clima_vivo = climate_api.get_live_oni_data()
                                col_enso_name = None
                                
                                if df_clima_vivo is not None and not df_clima_vivo.empty:
                                    df_clima_vivo.set_index('fecha', inplace=True)
                                    df_final = df_final.join(df_clima_vivo[['fase_enso']], how='left')
                                    col_enso_name = 'fase_enso'
                                
                                if col_enso_name:
                                    df_promedios_enso = df_final.groupby([df_final.index.month, col_enso_name])[estaciones_robustas].transform('mean')
                                    df_final[estaciones_robustas] = df_final[estaciones_robustas].fillna(df_promedios_enso)
                                    df_final.drop(columns=[col_enso_name], inplace=True)
                                
                                df_promedios_mensuales = df_final.groupby(df_final.index.month)[estaciones_robustas].transform('mean')
                                df_final[estaciones_robustas] = df_final[estaciones_robustas].fillna(df_promedios_mensuales)
                                df_final[estaciones_robustas] = df_final[estaciones_robustas].fillna(df_final[estaciones_robustas].mean())

                    # ==========================================================
                    # 🛡️ EXPORTACIÓN SEGURA (APLICAR CORTAFUEGOS DE VIDA ÚTIL)
                    # ==========================================================
                    if cols_estaciones:
                        # 🚫 Borramos los datos imputados fuera de la vida real de la estación
                        for col in cols_estaciones:
                            inicio, fin = limites_vida.get(col, (None, None))
                            if inicio and fin:
                                df_final.loc[df_final.index < inicio, col] = np.nan
                                df_final.loc[df_final.index > fin, col] = np.nan
                                
                        df_final.dropna(subset=cols_estaciones, how='all', inplace=True)
                    
                    df_final = df_final.sort_index().reset_index()
                    df_final['fecha'] = df_final['fecha'].dt.strftime('%Y-%m-%d')
                    
                    st.session_state['csv_fusionado_data'] = df_final.to_csv(sep=';', decimal=',', index=False, encoding='utf-8-sig').encode('utf-8')
                    st.session_state['fusion_resumen'] = f"Operación exitosa. Matriz blindada con Híbrido Espacial-Climático y límites de vida útiles."
                    st.session_state['fusion_preview'] = df_final.tail(10)
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error procesando los archivos: {e}")

    # Renderizado
    if st.session_state['csv_fusionado_data'] is not None:
        st.markdown("---")
        st.success("✅ ¡Matriz Única de Trabajo Generada con Éxito!")
        st.info(st.session_state['fusion_resumen'])
        st.dataframe(st.session_state['fusion_preview'], use_container_width=True)
        
        st.download_button(
            label="📥 Descargar Matriz Definitiva (Apta para Excel y Supabase)",
            data=st.session_state['csv_fusionado_data'],
            file_name="DatosPptnmes_Maestro_Integral.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
