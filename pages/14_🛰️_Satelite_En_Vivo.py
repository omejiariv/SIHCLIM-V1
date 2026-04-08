import streamlit as st
import ee
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Radar Satelital Vivo", page_icon="🛰️", layout="wide")
st.title("🛰️ Radar Dynamic World (En Vivo)")

# ==========================================
# 1. MOTOR DE AUTENTICACIÓN
# ==========================================
@st.cache_resource
def iniciar_conexion_gee():
    try:
        credenciales_dict = dict(st.secrets["gcp_service_account"])
        credentials = ee.ServiceAccountCredentials(
            email=credenciales_dict["client_email"],
            key_data=credenciales_dict["private_key"]
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Fallo en la comunicación con el satélite: {e}")
        return False

# ==========================================
# 2. PUENTE NATIVO FOLIUM - EARTH ENGINE
# ==========================================
# 🧠 Esta es la ingeniería pura: Le enseñamos a Folium a leer la API de Google
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Inyectamos nuestro método dentro de la clase original de Folium
folium.Map.add_ee_layer = add_ee_layer

# ==========================================
# 3. RENDERIZADO DEL MAPA
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
        
        # Pintar el mapa base
        m = folium.Map(location=[6.1, -75.5], zoom_start=12)
        
        # Añadir capa satelital base real
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Inyectar la capa procesada por IA de Earth Engine
        m.add_ee_layer(dw_imagen, dw_vis, 'Cobertura Dynamic World (10m)')
        
        # Añadir panel de control de capas
        folium.LayerControl().add_to(m)
        
        # Renderizar
        st_folium(m, width=1000, height=600)
