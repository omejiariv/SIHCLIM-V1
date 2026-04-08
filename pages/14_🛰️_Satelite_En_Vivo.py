import streamlit as st
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="Radar Satelital Vivo", page_icon="🛰️", layout="wide")
st.title("🛰️ Radar Dynamic World (En Vivo)")

# ==========================================
# 1. MOTOR DE AUTENTICACIÓN
# ==========================================
@st.cache_resource
def iniciar_conexion_gee():
    try:
        # Extraemos las credenciales completas de los secrets
        credenciales_dict = dict(st.secrets["gcp_service_account"])
        
        # Earth Engine necesita un objeto de credenciales específico
        credentials = ee.ServiceAccountCredentials(
            email=credenciales_dict["client_email"],
            key_data=credenciales_dict["private_key"]
        )
        # Inicializamos el superordenador
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Fallo en la comunicación con el satélite: {e}")
        return False

# ==========================================
# 2. RENDERIZADO DEL MAPA
# ==========================================
if iniciar_conexion_gee():
    st.success("✅ Enlace satelital establecido con Google Earth Engine. Leyendo datos en vivo...")
    
    with st.spinner("Procesando datos en los servidores de Google..."):
        # Punto focal: La Fe / Medellín
        punto_focal = ee.Geometry.Point([-75.5, 6.1]) 
        roi_ee = punto_focal.buffer(10000) # Buffer de 10km
        
        # Colección Dynamic World (Mosaico más reciente)
        dw_coleccion = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                       .filterBounds(roi_ee) \
                       .filterDate('2023-01-01', '2024-01-01')
        
        dw_imagen = dw_coleccion.select('label').mode().clip(roi_ee)
        
        # Paleta de colores oficial de Dynamic World
        dw_vis = {
            'min': 0, 'max': 8,
            'palette': [
                '#419BDF', # Agua
                '#397D49', # Bosque
                '#88B053', # Pastos
                '#7A87C6', # Cultivos
                '#E49635', # Matorrales
                '#DFC35A', # Suelo Desnudo
                '#C4281B', # Urbano (Rojo)
                '#A59B8F', # Nieve/Nubes
                '#B39FE1'  # Otros
            ]
        }
        
        # Pintar el mapa con Geemap
        m = geemap.Map(location=[6.1, -75.5], zoom_start=12)
        m.add_basemap('SATELLITE') # Fondo satelital base
        m.addLayer(dw_imagen, dw_vis, 'Cobertura Dynamic World (10m)')
        
        # Renderizar en Streamlit
        st_folium(m, width=1000, height=600)
