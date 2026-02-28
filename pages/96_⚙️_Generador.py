# pages/96_⚙️_Generador.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Generador Espacial", page_icon="⚙️", layout="wide")

st.title("⚙️ Centro de Geoprocesamiento y Transformación")
st.info("Herramientas de administrador para cruces espaciales y estandarización de cartografía hacia la web.")

tab1, tab2 = st.tabs(["🧩 1. Intersección Cuencas-Municipios", "🔄 2. Convertidor Shapefile a GeoJSON (Nube)"])

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
# PESTAÑA 2: EL TRANSFORMADOR DE SHAPEFILES MULTI-ARCHIVO
# =====================================================================
with tab2:
    st.subheader("Transformador Web de Cartografía")
    st.markdown("Sube **todos** los archivos que componen tu capa `.shp` al mismo tiempo (el `.shp`, `.dbf`, `.shx`, `.prj`, etc.). No necesitas comprimirlos en ZIP.")
    
    # El truco: accept_multiple_files=True permite seleccionar varios archivos sueltos
    archivos_subidos = st.file_uploader("Selecciona los archivos de la capa (Mínimo .shp, .shx y .dbf)", accept_multiple_files=True)

    if archivos_subidos:
        # Buscamos cuál de todos los archivos es el .shp principal para nombrarlo
        archivo_shp = next((f for f in archivos_subidos if f.name.endswith('.shp')), None)
        
        if archivo_shp:
            if st.button("⚙️ Transformar a GeoJSON"):
                with st.spinner("Reconstruyendo el Shapefile en la nube y reproyectando a WGS84..."):
                    try:
                        # Crear un directorio temporal seguro en la nube
                        with tempfile.TemporaryDirectory() as tmpdir:
                            # Escribir todos los archivos subidos en ese directorio temporal
                            for f in archivos_subidos:
                                filepath = os.path.join(tmpdir, f.name)
                                with open(filepath, "wb") as f_out:
                                    f_out.write(f.getvalue())
                            
                            # Ahora Geopandas puede leer el .shp y encontrará a sus "hermanos" (.dbf, .shx) ahí mismo
                            ruta_shp_temporal = os.path.join(tmpdir, archivo_shp.name)
                            gdf = gpd.read_file(ruta_shp_temporal)
                            
                            # Estandarización del Sistema de Coordenadas
                            if gdf.crs is None:
                                # Asumimos Magna Sirgas origen Nacional o Central si no hay .prj
                                gdf.set_crs(epsg=3116, inplace=True)
                                
                            if gdf.crs.to_string() != "EPSG:4326":
                                gdf = gdf.to_crs(epsg=4326)
                                
                            # Conversión final a GeoJSON
                            geojson_data = gdf.to_json()
                            nombre_base = archivo_shp.name.replace('.shp', '')
                            
                            st.success(f"✅ ¡Capa '{nombre_base}' estandarizada! ({len(gdf)} registros procesados).")
                            
                            st.download_button(
                                label=f"📥 Descargar {nombre_base}.geojson",
                                data=geojson_data,
                                file_name=f"{nombre_base}.geojson",
                                mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"❌ Error procesando los archivos: {str(e)}")
        else:
            st.warning("⚠️ Asegúrate de incluir el archivo que termina en '.shp' en tu selección.")
