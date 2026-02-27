import streamlit as st
import geopandas as gpd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.title("⚙️ Generador de Intersecciones Espaciales")
st.info("Esta es una herramienta temporal de administrador para cruzar los mapas de municipios y cuencas.")

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
        
        # Ajustamos a las columnas de tus GeoJSON
        df_export = df_final[['MPIO_CNMBR', 'Zona', 'SUBC_LBL', 'N_NSS1', 'area_fragmento_ha', 'porcentaje_en_cuenca']]
        df_export.columns = ['Municipio', 'Zona_Hidrografica', 'Subcuenca', 'Sistema', 'Area_Ha', 'Porcentaje']
        
        # Convertimos a formato CSV en memoria
        csv = df_export.to_csv(index=False).encode('utf-8')
        
    st.success("✅ ¡Matemática espacial completada con éxito!")
    st.dataframe(df_export.head()) # Te mostramos una vista previa
    
    # EL BOTÓN MÁGICO PARA DESCARGAR A TU COMPUTADORA
    st.download_button(
        label="📥 DESCARGAR ARCHIVO MAESTRO CSV",
        data=csv,
        file_name='cuencas_mpios_proporcion.csv',
        mime='text/csv',
    )
