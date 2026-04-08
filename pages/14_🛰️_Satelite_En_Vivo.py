import os
import sys
import json
import streamlit as st
import ee
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE PÁGINA (Debe ir primero) ---
st.set_page_config(page_title="Radar Satelital Vivo", page_icon="🛰️", layout="wide")

# --- IMPORTACIÓN DE MÓDULOS (SIDEBAR Y SELECTORES) ---
try:
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo

# ==========================================
# MENÚ Y SELECTORES TERRITORIALES
# ==========================================
# 1. Recuperamos el menú lateral izquierdo
selectors.renderizar_menu_navegacion("Satélite en Vivo")

try:
    inicializar_torrente_sanguineo()
except Exception:
    pass

st.title("🛰️ Radar Dynamic World (En Vivo)")
st.markdown("Clasificación de coberturas de la tierra en tiempo real mediante Inteligencia Artificial (Sentinel-2 / Google Earth Engine).")

# 2. Recuperamos el selector espacial superior (Municipios/Subcuencas)
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

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
def add_ee_layer(self, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(self)
    except Exception as e:
        print(f"Error añadiendo capa EE: {e}")

folium.Map.add_ee_layer = add_ee_layer

# ==========================================
# 3. RENDERIZADO DEL MAPA DINÁMICO (Conexión SIG)
# ==========================================
if iniciar_conexion_gee():
    # Validamos que el usuario haya seleccionado una zona válida en el selector
    if gdf_zona is not None and not gdf_zona.empty:
        st.success(f"✅ Satélite enlazado. Escaneando la zona: **{nombre_zona}**")
        
        with st.spinner("Calculando órbitas y aplicando IA de clasificación... esto puede tomar unos segundos."):
            # 1. Convertir el polígono Geopandas a formato Google Earth Engine
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            
            # Extraemos la geometría exacta (con todas sus curvas) para recortar el mapa
            geom_unificada = gdf_zona.geometry.unary_union
            # Convertimos esa figura maestra al lenguaje que entiende el satélite
            roi_ee = ee.Geometry(geom_unificada.__geo_interface__)
            
            # 2. Consultar Dynamic World para el último año
            dw_coleccion = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                           .filterBounds(roi_ee) \
                           .filterDate('2023-01-01', '2024-01-01')
            
            # Magia: Tomamos el promedio (mode) y lo RECORTAMOS con la forma exacta de la cuenca
            dw_imagen = dw_coleccion.select('label').mode().clip(roi_ee)
            
            # ==========================================================
            # 🧠 INVENTARIO TOTAL DE COBERTURAS EN TIEMPO REAL
            # ==========================================================
            try:
                # 1. Agrupar píxeles por clase y sumar su área (m2)
                pixel_area = ee.Image.pixelArea()
                area_por_clase = pixel_area.addBands(dw_imagen).reduceRegion(
                    reducer=ee.Reducer.sum().group(groupField=1, groupName='clase'),
                    geometry=roi_ee,
                    scale=10, # Resolución de 10x10 metros
                    maxPixels=1e9
                ).get('groups')
                
                # 2. Traer los datos desde los servidores de Google a Python
                estadisticas = area_por_clase.getInfo()
                
                # 3. Diccionario de Clases Oficial de Dynamic World
                nombres_clases = {
                    0: "Agua", 1: "Bosques", 2: "Pastos", 3: "Cultivos", 
                    4: "Matorrales", 5: "Suelo Desnudo", 6: "Urbano / Infraestructura", 
                    7: "Nieve", 8: "Nubes"
                }
                
                # 4. Procesar resultados
                resultados = []
                ha_bosque_satelite = 0.0
                
                for item in estadisticas:
                    clase_id = int(item['clase'])
                    area_ha = item['sum'] / 10000.0 # Convertir m2 a Hectáreas
                    nombre = nombres_clases.get(clase_id, "Desconocido")
                    resultados.append({"Cobertura": nombre, "Área (ha)": area_ha})
                    
                    if clase_id == 1: # Guardamos el Bosque para el Simulador WRI
                        ha_bosque_satelite = area_ha
                
                st.session_state['satelite_ha_bosque'] = ha_bosque_satelite
                
                # 5. Renderizar Gráfico y Tabla en Streamlit
                if resultados:
                    import pandas as pd
                    import plotly.express as px
                    
                    df_coberturas = pd.DataFrame(resultados).sort_values(by="Área (ha)", ascending=False)
                    
                    st.markdown("### 📊 Inventario Satelital de Uso de Suelo")
                    c_graf, c_tabla = st.columns([2, 1])
                    
                    with c_tabla:
                        st.dataframe(df_coberturas.style.format({"Área (ha)": "{:,.1f}"}), use_container_width=True)
                        
                    with c_graf:
                        fig_pie = px.pie(
                            df_coberturas, values='Área (ha)', names='Cobertura', hole=0.4, 
                            color='Cobertura', color_discrete_map={
                                "Agua": "#419BDF", "Bosques": "#397D49", "Pastos": "#88B053", 
                                "Cultivos": "#7A87C6", "Matorrales": "#E49635", "Suelo Desnudo": "#DFC35A", 
                                "Urbano / Infraestructura": "#C4281B", "Nieve": "#A59B8F", "Nubes": "#B39FE1"
                            }
                        )
                        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
            except Exception as e:
                st.warning(f"Error procesando estadísticas satelitales: {e}")
            # ==========================================================
            # ==========================================================
            
            # 3. Paleta Oficial de Dynamic World
            dw_vis = {
                'min': 0, 'max': 8,
                'palette': [
                    '#419BDF', # 0: Agua
                    '#397D49', # 1: Árboles / Bosques
                    '#88B053', # 2: Pastos
                    '#7A87C6', # 3: Cultivos
                    '#E49635', # 4: Matorrales
                    '#DFC35A', # 5: Suelo Desnudo
                    '#C4281B', # 6: Construido / Urbano
                    '#A59B8F', # 7: Nieve / Hielo
                    '#B39FE1'  # 8: Nubes / Sin datos
                ]
            }
            
            # 4. Crear Mapa Base Centrado en la Zona Seleccionada
            centro_y = (miny + maxy) / 2
            centro_x = (minx + maxx) / 2
            m = folium.Map(location=[centro_y, centro_x], zoom_start=11)
            
            # Fondo satelital normal de Google
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google',
                name='Satélite HD Base',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Inyectar la capa de IA
            m.add_ee_layer(dw_imagen, dw_vis, f'Cobertura IA ({nombre_zona})')
            
            # Dibujar un borde blanco brillante alrededor del territorio
            folium.GeoJson(
                gdf_zona,
                name="Límite Territorial",
                style_function=lambda x: {'color': 'white', 'weight': 3, 'fillOpacity': 0}
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            # Renderizar en la pantalla de Streamlit
            st_folium(m, width="100%", height=600, returned_objects=[])
            
            # Leyenda visual
            st.markdown("""
            **Leyenda de Coberturas (Dynamic World a 10m de resolución):**
            🟦 Agua | 🟩 Bosques | 🟨 Pastos | 🟪 Cultivos | 🟧 Matorrales | 🟫 Suelo Desnudo | 🟥 Infraestructura / Urbano
            """)
    else:
        st.info("👈 Por favor, selecciona un territorio en el panel lateral para iniciar el escaneo satelital.")
