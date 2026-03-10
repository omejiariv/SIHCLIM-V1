# pages/96_⚙️_Generador.py

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
st.info("Herramientas de administrador para cruces espaciales, compresión de archivos y estandarización de cartografía web.")

# Nueva estructura de pestañas
tab1, tab2, tab3 = st.tabs([
    "🧩 1. Intersección Cuencas-Municipios", 
    "🔄 2. Convertidor a GeoJSON (Soporta ZIP)", 
    "🗜️ 3. Compresor/Extractor ZIP"
])

# =====================================================================
# PESTAÑA 1: TU CÓDIGO ORIGINAL INTACTO
# =====================================================================
with tab1:
    st.subheader("Generador de Intersecciones Espaciales")
    st.write("Cruce de mapas para calcular la proporción de cada municipio dentro de las cuencas.")
    
    if st.button("🚀 Iniciar Cruce Espacial (Tarda ~1 minuto)"):
        with st.spinner("1. Cargando y proyectando mapas (EPSG:9377)..."):
            try:
                mpios = gpd.read_file('data/MunicipiosAntioquia.geojson')
                cuencas = gpd.read_file('data/SubcuencasAinfluencia.geojson')
                
                mpios = mpios.to_crs(epsg=9377)
                cuencas = cuencas.to_crs(epsg=9377)
                
                mpios['area_mpio_ha'] = mpios.geometry.area / 10000
            except Exception as e:
                st.error(f"Error cargando mapas. Verifica que existan en la carpeta data/. Detalles: {e}")
                st.stop()
            
        with st.spinner("2. Geometría en proceso: Cruzando polígonos..."):
            interseccion = gpd.overlay(mpios, cuencas, how='intersection')
            interseccion['area_fragmento_ha'] = interseccion.geometry.area / 10000
            interseccion['porcentaje_en_cuenca'] = (interseccion['area_fragmento_ha'] / interseccion['area_mpio_ha']) * 100
            
            df_final = interseccion[interseccion['porcentaje_en_cuenca'] > 0.1].copy()
            
            df_export = df_final[['MPIO_CNMBR', 'Zona', 'SUBC_LBL', 'N_NSS1', 'area_fragmento_ha', 'porcentaje_en_cuenca']]
            df_export.columns = ['Municipio', 'Zona_Hidrografica', 'Subcuenca', 'Sistema', 'Area_Ha', 'Porcentaje']
            
            csv = df_export.to_csv(index=False).encode('utf-8')
            
        st.success("✅ ¡Matemática espacial completada con éxito!")
        st.dataframe(df_export.head()) 
        
        st.download_button(
            label="📥 DESCARGAR ARCHIVO MAESTRO CSV",
            data=csv,
            file_name='cuencas_mpios_proporcion.csv',
            mime='text/csv',
        )

# =====================================================================
# PESTAÑA 2: TRANSFORMADOR DE SHAPEFILES (AHORA SOPORTA ZIP DIRECTO)
# =====================================================================
with tab2:
    st.subheader("Transformador Web de Cartografía a GeoJSON")
    st.markdown("Puedes subir un archivo **.zip** que contenga tu Shapefile, o seleccionar los archivos sueltos (`.shp`, `.dbf`, `.shx`, etc.) al mismo tiempo.")
    
    modo_carga = st.radio("Método de Carga:", ["Subir archivo .zip (Recomendado)", "Subir archivos sueltos"], horizontal=True)
    
    if modo_carga == "Subir archivo .zip (Recomendado)":
        archivo_zip = st.file_uploader("Sube tu archivo ZIP (debe contener un .shp adentro)", type=['zip'])
        
        if archivo_zip and st.button("⚙️ Descomprimir y Transformar a GeoJSON"):
            with st.spinner("Procesando ZIP y reproyectando a WGS84..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Extraer todo el ZIP en la carpeta temporal
                        with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                            
                        # Buscar recursivamente el archivo .shp
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
                            # Leer con GeoPandas
                            gdf = gpd.read_file(ruta_shp)
                            
                            # Estandarización de Coordenadas
                            if gdf.crs is None: gdf.set_crs(epsg=3116, inplace=True)
                            if gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
                            
                            geojson_data = gdf.to_json()
                            nombre_base = os.path.basename(ruta_shp).replace('.shp', '').replace('.SHP', '')
                            
                            st.success(f"✅ ¡Capa '{nombre_base}' estandarizada! ({len(gdf)} polígonos procesados).")
                            st.download_button(
                                label=f"📥 Descargar {nombre_base}.geojson",
                                data=geojson_data,
                                file_name=f"{nombre_base}.geojson",
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"❌ Error procesando el ZIP: {str(e)}")
                    
    else: # Modo archivos sueltos (tu código original mejorado)
        archivos_subidos = st.file_uploader("Selecciona archivos (Mínimo .shp, .shx y .dbf)", accept_multiple_files=True)
        if archivos_subidos:
            archivo_shp = next((f for f in archivos_subidos if f.name.lower().endswith('.shp')), None)
            
            if archivo_shp and st.button("⚙️ Transformar sueltos a GeoJSON"):
                with st.spinner("Reconstruyendo Shapefile y reproyectando..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            for f in archivos_subidos:
                                with open(os.path.join(tmpdir, f.name), "wb") as f_out:
                                    f_out.write(f.getvalue())
                                    
                            ruta_shp_temporal = os.path.join(tmpdir, archivo_shp.name)
                            gdf = gpd.read_file(ruta_shp_temporal)
                            
                            if gdf.crs is None: gdf.set_crs(epsg=3116, inplace=True)
                            if gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
                            
                            geojson_data = gdf.to_json()
                            nombre_base = archivo_shp.name.replace('.shp', '').replace('.SHP', '')
                            
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
                # Crear ZIP en memoria
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
        st.markdown("Sube un archivo `.zip` para ver y descargar su contenido por separado.")
        archivo_a_descomprimir = st.file_uploader("Sube un archivo ZIP", type=['zip'], key="uploader_desc")
        
        if archivo_a_descomprimir:
            with zipfile.ZipFile(archivo_a_descomprimir, 'r') as zip_ref:
                lista_archivos = zip_ref.namelist()
                st.write(f"**Contiene {len(lista_archivos)} archivos:**")
                
                # Mostrar botones de descarga individuales
                for nombre_archivo in lista_archivos:
                    # Evitar carpetas vacías
                    if not nombre_archivo.endswith('/'): 
                        datos_archivo = zip_ref.read(nombre_archivo)
                        st.download_button(
                            label=f"⬇️ {os.path.basename(nombre_archivo)}",
                            data=datos_archivo,
                            file_name=os.path.basename(nombre_archivo),
                            key=f"desc_{nombre_archivo}"
                        )
