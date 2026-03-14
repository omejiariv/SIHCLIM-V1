# pages/11_⚙️_Generador.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import tempfile
import zipfile
import io
import os
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Generador Espacial", page_icon="⚙️", layout="wide")

st.title("⚙️ Centro de Geoprocesamiento y Transformación")
st.info("Herramientas de administrador para cruces espaciales, compresión y estandarización de cartografía web.")

tab1, tab2, tab3 = st.tabs([
    "🧩 1. Intersecciones Espaciales (Cuencas)", 
    "🔄 2. Convertidor GeoJSON (Soporta ZIP y Simplificación)", 
    "🗜️ 3. Compresor/Extractor ZIP"
])

# =====================================================================
# PESTAÑA 1: INTERSECCIÓN ESPACIAL (MUNICIPIOS O VEREDAS)
# =====================================================================
with tab1:
    st.subheader("Generador de Intersecciones Espaciales")
    st.write("Cruce de mapas para calcular la proporción de cada territorio dentro de las cuencas.")
    
    # NUEVO: Selector de nivel territorial
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
