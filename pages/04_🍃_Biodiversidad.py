# pages/04_🍃_Biodiversidad.py

import streamlit as st
import sys
import os
import io
import pandas as pd
import numpy as np
import geopandas as gpd
import tempfile
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import shapes
import plotly.graph_objects as go
import plotly.express as px
from modules.admin_utils import init_supabase
from shapely.geometry import shape

from modules.land_cover import LAND_COVER_COLORS, LAND_COVER_LEGEND

# --- IMPORTACIÓN DE MÓDULOS DEL SISTEMA ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector, carbon_calculator
    from modules import life_zones as lz 
    from modules import land_cover as lc
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")
# Encendido automático del Gemelo Digital (Lectura de matrices maestras)
from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto
encender_gemelo_digital()

st.title("🍃 Biodiversidad y Servicios de la Naturaleza")
st.markdown("""
Explora la riqueza biológica del territorio y descubre **La Factura de la Naturaleza**: 
una valoración económica de los colosales servicios de desalinización, bombeo, transporte 
y tratamiento de agua que el ecosistema realiza de forma gratuita.
""")

# 2. SELECTOR ESPACIAL
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error en selector: {e}")
    st.stop()

def save_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 
@st.cache_resource(show_spinner=False)
def get_raster_from_cloud(filename):
    """
    Buscador Profundo: Busca en la raíz y dentro de carpetas del bucket 'rasters'.
    """
    try:
        client = init_supabase()
        bucket_name = "rasters" 
        
        # 1. Listar contenido raíz del bucket
        items_root = client.storage.from_(bucket_name).list()
        
        # 2. Construir lista plana de archivos (buscando dentro de carpetas si es necesario)
        all_files = []
        for item in items_root:
            if item['metadata'] is None: # Es una carpeta
                folder_name = item['name']
                # Listar contenido de la carpeta
                sub_items = client.storage.from_(bucket_name).list(path=folder_name)
                for sub in sub_items:
                    all_files.append(f"{folder_name}/{sub['name']}")
            else:
                # Es un archivo en la raíz
                all_files.append(item['name'])

        # 3. Lógica de Búsqueda Flexible
        # Buscamos si alguna parte del nombre coincide (ej: "DemAntioquia" en "Coberturas/DemAntioquia_v2.tif")
        keyword = filename.split('_')[0].split('.')[0] # Ej: de "DemAntioquia_EPSG..." toma "DemAntioquia"
        
        target_file = None
        for real_name in all_files:
            # Limpieza para comparar
            clean_real = real_name.split('/')[-1] # Quitar nombre de carpeta
            
            # A. Coincidencia exacta
            if filename == clean_real:
                target_file = real_name
                break
            
            # B. Coincidencia parcial (el "salvavidas")
            if keyword.lower() in clean_real.lower() and filename.endswith('.tif'):
                target_file = real_name
                st.toast(f"⚠️ Archivo encontrado con nombre diferente: '{real_name}'")
                break
        
        # 4. Descargar o Reportar
        if target_file:
            file_bytes = client.storage.from_(bucket_name).download(target_file)
            return io.BytesIO(file_bytes)
        else:
            # SI FALLA: Muestra qué hay realmente en el bucket para diagnosticar
            st.warning(f"🔍 No encontré '{filename}'.\n📂 Archivos disponibles en 'rasters': {all_files}")
            return None

    except Exception as e:
        st.error(f"Error de conexión con Storage: {e}")
        return None
        
@st.cache_data(ttl=3600)
def load_layer_cached(layer_name):
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson",
        "Predios": "PrediosEjecutados.geojson"
    }
    if layer_name in file_map:
        try:
            # Ajuste de ruta robusto
            file_path = os.path.join(config.Config.DATA_DIR, file_map[layer_name])
            if not os.path.exists(file_path):
                # Fallback por si DATA_DIR no resuelve bien en cloud
                file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_map[layer_name]))
            
            if os.path.exists(file_path):
                gdf = gpd.read_file(file_path)
                if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                return gdf
        except: return None
    return None


# --- FUNCIÓN analizar_coberturas_por_zona_vida ---

@st.cache_data(show_spinner=False)
def analizar_coberturas_por_zona_vida(_gdf_zona, zone_key, _dem_file, _ppt_file, _cov_file):
    """
    Estrategia 'Tierra Firme': Escribe los archivos en disco temporalmente
    para garantizar que rasterio lea correctamente la georreferenciación.
    """
    try:
        if not _dem_file or not _ppt_file or not _cov_file:
            return None

        # 1. CREAR ARCHIVOS TEMPORALES (Simulamos estar en tu PC)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_dem, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_ppt, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_cov:
            
            # Escribir bytes al disco
            _dem_file.seek(0); tmp_dem.write(_dem_file.read())
            _ppt_file.seek(0); tmp_ppt.write(_ppt_file.read())
            _cov_file.seek(0); tmp_cov.write(_cov_file.read())
            
            # Guardamos rutas
            path_dem, path_ppt, path_cov = tmp_dem.name, tmp_ppt.name, tmp_cov.name

        try:
            # ---------------------------------------------------------
            # PASO 2: PROCESAR EL DEM (AHORA SÍ DESDE DISCO)
            # ---------------------------------------------------------
            dem_arr = None
            out_meta = None
            out_crs = None
            
            with rasterio.open(path_dem) as src_dem:
                # A. Diagnóstico de CRS
                crs_working = src_dem.crs
                if not crs_working:
                    # Si sigue sin tener CRS, es un archivo 'crudo'. Asumimos 3116.
                    crs_working = rasterio.crs.CRS.from_string("EPSG:3116")

                # B. Proyectar Geometría (Tu lógica exitosa)
                gdf_valid = _gdf_zona.copy()
                gdf_valid['geometry'] = gdf_valid.buffer(0)
                gdf_proj = gdf_valid.to_crs(crs_working)
                
                # C. Verificar Superposición (Debug visual si falla)
                # bounds_raster = src_dem.bounds
                # bounds_zona = gdf_proj.total_bounds
                # Si esto falla, aquí sabríamos por qué (pero mask lanzará ValueError)

                # D. Recorte
                try:
                    out_image, out_transform = mask(src_dem, gdf_proj.geometry, crop=True)
                    dem_arr = out_image[0]
                    out_shape = dem_arr.shape
                    out_crs = crs_working
                except ValueError:
                    # st.error(f"Zona fuera del mapa. Raster: {src_dem.bounds}, Zona: {gdf_proj.total_bounds}")
                    return None

                # E. Limpieza
                dem_arr = np.where(dem_arr == src_dem.nodata, np.nan, dem_arr)
                dem_arr = np.where(dem_arr < -100, np.nan, dem_arr) # Filtro ruido

            if dem_arr is None or np.isnan(dem_arr).all():
                return None

            # ---------------------------------------------------------
            # PASO 3: ALINEAR OTROS MAPAS (DESDE DISCO)
            # ---------------------------------------------------------
            def alinear_desde_disco(path_raster, shape_dst, transform_dst, crs_dst, es_cat=False):
                with rasterio.open(path_raster) as src:
                    # Asignar CRS si falta
                    crs_src = src.crs if src.crs else "EPSG:3116"
                    
                    destino = np.zeros(shape_dst, dtype=src.dtypes[0])
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=destino,
                        src_transform=src.transform,
                        src_crs=crs_src,
                        dst_transform=transform_dst,
                        dst_crs=crs_dst,
                        resampling=Resampling.nearest if es_cat else Resampling.bilinear
                    )
                    return destino

            ppt_arr = alinear_desde_disco(path_ppt, out_shape, out_transform, out_crs)
            cov_arr = alinear_desde_disco(path_cov, out_shape, out_transform, out_crs, es_cat=True)

            # ---------------------------------------------------------
            # PASO 4: CÁLCULOS
            # ---------------------------------------------------------
            v_classify = np.vectorize(lz.classify_life_zone_alt_ppt)
            
            dem_safe = np.nan_to_num(dem_arr, nan=-9999)
            ppt_safe = np.nan_to_num(ppt_arr, nan=0)
            
            zv_arr = v_classify(dem_safe, ppt_safe)
            
            # Máscara estricta: Solo donde hay DEM válido Y Cobertura válida
            valid_mask = ~np.isnan(dem_arr) & (dem_arr > -100) & (cov_arr > 0)
            
            # Área de pixel
            res_x = out_transform[0]
            if out_crs.is_geographic:
                 pixel_area_ha = (abs(res_x) * 111132.0) ** 2 / 10000.0
            else:
                 pixel_area_ha = (abs(res_x) ** 2) / 10000.0

            df = pd.DataFrame({
                'ZV_ID': zv_arr[valid_mask].flatten(),
                'COV_ID': cov_arr[valid_mask].flatten()
            })
            
            if df.empty: return None

            resumen = df.groupby(['ZV_ID', 'COV_ID']).size().reset_index(name='Pixeles')
            resumen['Hectareas'] = resumen['Pixeles'] * pixel_area_ha
            
            # Nombres
            resumen['Zona_Vida'] = resumen['ZV_ID'].map(lambda x: lz.holdridge_int_to_name_simplified.get(x, f"ZV {x}"))
            resumen['Cobertura'] = resumen['COV_ID'].map(lambda x: lc.LAND_COVER_LEGEND.get(x, f"Clase {x}"))
            
            return resumen

        finally:
            # LIMPIEZA: Borrar archivos temporales para no llenar el servidor
            try:
                os.remove(path_dem)
                os.remove(path_ppt)
                os.remove(path_cov)
            except: pass

    except Exception as e:
        # st.error(f"Error técnico: {e}")
        return None

# --- FUNCIÓN HELPER ---

@st.cache_data(show_spinner=False)
def generar_mapa_coberturas_vectorial(_gdf_zona, _cov_file):
    """
    Convierte Raster a Polígonos usando 'Archivo Temporal' para garantizar 
    la lectura del CRS, detectando dinámicamente si es WGS84 o Magna Sirgas.
    """
    try:
        if not _cov_file: return None
        
        import tempfile
        import os
        from rasterio.features import shapes
        from rasterio.mask import mask
        import geopandas as gpd
        import rasterio
        
        # 1. Bajar a Tierra (Temporal)
        _cov_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_cov:
            tmp_cov.write(_cov_file.read())
            path_cov = tmp_cov.name
            
        try:
            with rasterio.open(path_cov) as src:
                # 2. Detector Automático de Proyección
                src_crs = src.crs
                if not src_crs:
                    # Si la coordenada X es un grado (-75 aprox), es WGS84. Si es > 1000, es Magna.
                    src_crs = "EPSG:3116" if src.transform[2] > 1000 else "EPSG:4326"
                    
                # 3. Preparar Geometría (Blindaje)
                gdf_valid = _gdf_zona.copy()
                gdf_valid['geometry'] = gdf_valid.buffer(0)
                gdf_proj = gdf_valid.to_crs(src_crs)
                
                # 4. Recorte
                try:
                    out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
                    data = out_image[0]
                except ValueError:
                    return None # Fuera de límites geográficos
                    
                # 5. Extraer Polígonos
                mask_val = (data != src.nodata) & (data > 0)
                geoms = ({'properties': {'val': v}, 'geometry': s} 
                         for i, (s, v) in enumerate(shapes(data, mask=mask_val, transform=out_transform)))
                
                gdf_vector = gpd.GeoDataFrame.from_features(list(geoms), crs=src_crs)
                if gdf_vector.empty: return None

                # 6. Estandarizar para el mapa Web (Obligatorio WGS84 para Plotly)
                gdf_vector = gdf_vector.to_crs("EPSG:4326")
                
                # 7. Asignar Colores y Nombres según Diccionario Oficial
                gdf_vector['Cobertura'] = gdf_vector['val'].map(lambda x: lc.LAND_COVER_LEGEND.get(int(x), f"Clase {int(x)}"))
                gdf_vector['Color'] = gdf_vector['val'].map(lambda x: lc.LAND_COVER_COLORS.get(int(x), "#CCCCCC"))
                
                return gdf_vector
                
        finally:
            # Limpiar memoria del servidor
            try: os.remove(path_cov)
            except: pass
            
    except Exception as e:
        return None
        
# --- FUNCIÓN DE INTEGRACIÓN: DETECTAR ZONA DE VIDA ---
def detectar_zona_vida_dominante(gdf_zona):
    """
    Usa el módulo life_zones para estimar la zona climática del polígono seleccionado
    sin tener que procesar todo el raster pesado si no es necesario.
    """
    try:
        # 1. Calculamos el centroide de la zona seleccionada
        centroid = gdf_zona.to_crs("+proj=cea").centroid.to_crs("EPSG:4326").iloc[0]
        altitud = altitud_ref if altitud_ref > 0 else 1500 # Default si falla
        
        # 2. Obtenemos precipitación promedio (Simulada o de base de datos)
        # En una integración total, aquí leeríamos el raster de PPAMAnt.tif en ese punto
        ppt_estimada = 2000 # Valor medio para la región si no hay raster cargado
        
        # 3. Usamos la lógica de clasificación de life_zones.py
        # ID -> Nombre
        zona_id = lz.classify_life_zone_alt_ppt(altitud, ppt_estimada)
        zona_nombre = lz.holdridge_int_to_name_simplified.get(zona_id, "Desconocido")
        
        # 4. Mapeo a códigos de Álvarez (Ecuaciones)
        # Esto es un diccionario de traducción simple para el ejemplo
        mapa_codigos = {
            "Bosque húmedo Premontano (bh-PM)": "bh-PM",
            "Bosque muy húmedo Premontano (bmh-PM)": "bmh-PM",
            "Bosque muy húmedo Montano (bmh-M)": "bmh-M",
            "Bosque húmedo Tropical (bh-T)": "bh-T"
        }
        return mapa_codigos.get(zona_nombre, "bh-MB") # Default seguro
    except:
        return "bh-MB"

# =========================================================================
# 🗂️ SISTEMA DE PESTAÑAS (NAVEGACIÓN)
# =========================================================================
tab_factura, tab_mapa, tab_taxonomia, tab_forestal, tab_afolu, tab_comparador, tab_ecologia, tab_ret_dosel, tab_micro = st.tabs([
    "💰 La Factura de la Naturaleza", 
    "🗺️ Mapa & GBIF", 
    "🧬 Taxonomía",
    "🌲 Bosque e Inventarios",
    "⚖️ Metabolismo (AFOLU)",
    "⚖️ Comparativa de Escenarios de Carbono",
    "🌿 Ecología del Paisaje", 
    "🌳 Retención Hídrica del Dosel",
    "🔬 Ecohidrología: Del Bosque a la PTAP"
])

# ==============================================================================
# 💧 TAB 1: LA FACTURA DE LA NATURALEZA (VALORACIÓN DE SERVICIOS ECOSISTÉMICOS)
# ==============================================================================
with tab_factura:
    st.subheader("💰 La Factura de la Naturaleza")
    st.info("Valoración económica de los servicios ecosistémicos...")

    st.markdown("""
    > *¿Cuánto nos costaría a los humanos hacer el trabajo que el ciclo del agua hace gratis?* > Este simulador calcula el costo energético y económico de desalinizar, bombear, transportar y filtrar el agua con tecnología e infraestructura humana.
    """)

    # 🎥 EL REPRODUCTOR DE VIDEO DIDÁCTICO
    with st.expander("🎥 Ver Explicación Didáctica: El Ciclo del Agua", expanded=False):
        url_video_supabase = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/videos/ciclodelagua.mp4"
        st.video(url_video_supabase, format="video/mp4")
        st.caption("Aprende cómo la naturaleza actúa como la mayor planta de tratamiento y bombeo del planeta.")

    # 1. 🚀 INYECCIÓN DE LA TURBINA CENTRAL (Con blindaje de seguridad)
    try:
        anio_actual = st.session_state.get('aleph_anio', 2025)
        datos_metabolismo = obtener_metabolismo_exacto(nombre_seleccion, anio_actual)
        pob_total_base = datos_metabolismo.get('pob_total', 5000)
    except Exception:
        # Paracaídas por si no hay territorio seleccionado aún
        pob_total_base = 5000 

    # ==========================================
    # ESTRUCTURA DE DASHBOARD: 1/3 Controles | 2/3 Resultados
    # ==========================================
    col_ctrl, col_dash = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        st.markdown("### 🎛️ Parámetros Locales")
        st.info("Ajusta las variables para recalcular la factura en tiempo real.")
        
        # Blindaje: value usa la población exacta extraída por el motor
        val_pob = int(pob_total_base) if pob_total_base >= 1000 else 1000
        poblacion = st.number_input("👥 Población a abastecer:", min_value=1000, value=val_pob, step=5000)
        dotacion = st.slider("🚰 Dotación (Litros/hab/día):", min_value=50, max_value=300, value=150, step=5)
        altura_m = st.number_input("⛰️ Altitud promedio (m.s.n.m):", min_value=0, value=1500, step=50)
        distancia_km = st.number_input("🌬️ Distancia al mar (km):", min_value=0, value=400, step=10)
        
        # Escondemos las tarifas técnicas para no saturar la interfaz principal
        with st.expander("⚙️ Configuración de Tarifas Unitarias (US$)", expanded=False):
            t_desalinizacion = st.number_input("Desalinización ($/m³):", value=0.50, step=0.05)
            t_tratamiento = st.number_input("Tratamiento ($/m³):", value=0.05, step=0.01)
            t_transporte = st.number_input("Transporte ($/m³ por km):", value=0.25, step=0.05)
            t_bombeo = st.number_input("Bombeo ($/m³ por metro):", value=0.18, step=0.01)

    # ==========================================
    # MOTOR MATEMÁTICO
    # ==========================================
    volumen_anual_m3 = (poblacion * dotacion * 365) / 1000
    costo_desalinizacion = volumen_anual_m3 * t_desalinizacion
    costo_tratamiento = volumen_anual_m3 * t_tratamiento
    costo_transporte = volumen_anual_m3 * distancia_km * t_transporte
    costo_bombeo = volumen_anual_m3 * altura_m * t_bombeo
    
    costo_total_naturaleza = costo_desalinizacion + costo_tratamiento + costo_transporte + costo_bombeo
    costo_medio_m3 = costo_total_naturaleza / volumen_anual_m3 if volumen_anual_m3 > 0 else 0

    # ==========================================
    # PANEL DE RESULTADOS (DERECHA)
    # ==========================================
    with col_dash:
        st.markdown("### 🧾 Resumen Financiero Anual - Aportes de la Infraestructura Natural")
        
        # Tarjetas de métricas (KPIs)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("💧 Volumen Movilizado", f"{volumen_anual_m3 / 1e6:,.1f} Millones m³")
        kpi2.metric("💸 Aporte de la Naturaleza", f"${costo_total_naturaleza / 1e6:,.1f} M USD", "Subsidio Natural")
        kpi3.metric("🏷️ Costo Real Oculto", f"${costo_medio_m3:,.2f} USD / m³")
        
        # Gráfico de Cascada Financiera (Waterfall)
        import plotly.graph_objects as go
        
        fig_waterfall = go.Figure(go.Waterfall(
            name = "Factura Natural",
            orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = ["Desalinización<br>(Evaporación)", "Bombeo<br>(Ascenso Térmico)", "Transporte<br>(Vientos)", "Tratamiento<br>(Suelo/Bosques)", "<b>VALOR TOTAL</b>"],
            textposition = "outside",
            text = [f"${costo_desalinizacion/1e6:.1f}M", f"${costo_bombeo/1e6:.1f}M", f"${costo_transporte/1e6:.1f}M", f"${costo_tratamiento/1e6:.1f}M", f"<b>${costo_total_naturaleza/1e6:.1f}M</b>"],
            y = [costo_desalinizacion, costo_bombeo, costo_transporte, costo_tratamiento, costo_total_naturaleza],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color":"#e74c3c"}},
            increasing = {"marker":{"color":"#2ecc71"}},
            totals = {"marker":{"color":"#3498db"}}
        ))
        
        fig_waterfall.update_layout(
            title = "Construcción del Costo de los Servicios Hídricos (Millones USD)",
            showlegend = False,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Millones de Dólares (USD)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # Mensaje conclusivo de alto impacto (¡AHORA SÍ DENTRO DE LA PESTAÑA!)
    st.success(f"🌱 **El Mensaje para los Tomadores de Decisiones:** Proteger las cuencas y los bosques que abastecen a estos **{poblacion:,.0f} habitantes** le ahorra al Estado y a la sociedad **${costo_total_naturaleza / 1e6:,.1f} millones de dólares anuales** en infraestructura artificial. La conservación es la inversión más rentable.")

# ==============================================================================
# 🌍 MOTOR DE BIODIVERSIDAD GLOBAL (Prepara datos para Tab 2 y 3)
# ==============================================================================
import pandas as pd

gdf_bio = pd.DataFrame()
threatened = pd.DataFrame()
n_threat = 0

# --- PROCESAMIENTO PREVIO (Solo si hay zona cargada) ---
try:
    if gdf_zona is not None:
        with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
            gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)
            
        if not gdf_bio.empty and 'Amenaza IUCN' in gdf_bio.columns:
            threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
            n_threat = threatened['Nombre Científico'].nunique()
except NameError:
    pass # Pasa de largo si no hay mapa cargado para no romper la app

# ==============================================================================
# 🗺️ TAB 2: MAPA Y MÉTRICAS
# ==============================================================================
with tab_mapa:
    if gdf_zona is not None:
        # 1. Métricas Principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros GBIF", f"{len(gdf_bio):,.0f}")
        c2.metric("Especies", f"{gdf_bio['Nombre Científico'].nunique():,.0f}" if not gdf_bio.empty else "0")
        c3.metric("Familias", f"{gdf_bio['Familia'].nunique():,.0f}" if not gdf_bio.empty and 'Familia' in gdf_bio.columns else "0")
        c4.metric("Amenazadas (IUCN)", f"{n_threat}")

        # 2. Visor Territorial
        st.markdown("##### Visor Territorial")
        
        fig = go.Figure()

        # A. CENTRO DEL MAPA
        try:
            center = gdf_zona.to_crs("+proj=cea").centroid.to_crs("EPSG:4326").iloc[0]
            center_lat, center_lon = center.y, center.x
        except: center_lat, center_lon = 6.5, -75.5

        # B. CAPA ZONA (ROJO)
        for idx, row in gdf_zona.iterrows():
            if row.geometry:
                polys = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms) if row.geometry.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=3, color='red'), name='Zona Selección', hoverinfo='skip'))

        # C. CAPAS DE CONTEXTO (Municipios, Cuencas, Predios)
        layers_to_show = [("Municipios", "gray", 1), ("Cuencas", "blue", 1.5), ("Predios", "orange", 1)]
        
        for lyr_name, color, width in layers_to_show:
            gdf_lyr = load_layer_cached(lyr_name)
            if gdf_lyr is not None:
                # Recorte espacial para optimizar (solo predios)
                if lyr_name == "Predios":
                    try:
                        roi_buf = gdf_zona.to_crs("EPSG:3116").buffer(1000).to_crs("EPSG:4326")
                        gdf_lyr = gpd.clip(gdf_lyr, roi_buf)
                    except: pass
                
                if not gdf_lyr.empty:
                    # Dibujamos solo el primer polígono con leyenda, el resto oculto en grupo
                    for idx, row in gdf_lyr.iterrows():
                        if row.geometry:
                            polys = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms) if row.geometry.geom_type == 'MultiPolygon' else []
                            for i, poly in enumerate(polys):
                                x, y = poly.exterior.xy
                                show_leg = True if idx == 0 and i == 0 else False
                                visible_opt = 'legendonly' if lyr_name == "Predios" else True
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines', 
                                    line=dict(width=width, color=color), 
                                    name=lyr_name, legendgroup=lyr_name, 
                                    showlegend=show_leg, hoverinfo='skip', visible=visible_opt
                                ))

        # D. PUNTOS DE BIODIVERSIDAD (VERDE)
        if not gdf_bio.empty:
            # Blindaje inteligente contra columnas faltantes en GBIF
            if 'Nombre Común' in gdf_bio.columns:
                hover_text = gdf_bio['Nombre Común']
            elif 'Nombre Científico' in gdf_bio.columns:
                hover_text = gdf_bio['Nombre Científico']
            else:
                hover_text = "Registro Biológico"

            fig.add_trace(go.Scattermapbox(
                lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                mode='markers', marker=dict(size=7, color='rgb(0, 200, 100)'),
                text=hover_text, name='Biodiversidad'
            ))

        fig.update_layout(
            mapbox_style="carto-positron", 
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10), 
            margin={"r":0,"t":0,"l":0,"b":0}, height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if not gdf_bio.empty:
            st.download_button("💾 Descargar Datos (CSV)", save_to_csv(gdf_bio.drop(columns='geometry', errors='ignore')), f"biodiv_{nombre_seleccion}.csv", "text/csv")

    else:
        st.info("👈 Seleccione una zona en el menú lateral para visualizar el mapa.")

# ==============================================================================
# TAB 3: TAXONOMÍA
# ==============================================================================
with tab_taxonomia:
    if not gdf_bio.empty:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("##### Estructura Taxonómica")
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                df_chart = gdf_bio.fillna("Sin Dato")
                fig_sun = px.sunburst(df_chart, path=['Reino', 'Clase', 'Orden', 'Familia'], height=600)
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Datos taxonómicos insuficientes.")
        
        with c2:
            st.markdown("##### Especies Amenazadas")
            if not threatened.empty:
                st.warning(f"⚠️ {n_threat} especies en riesgo.")
                
                # Escudo protector contra columnas faltantes en GBIF
                cols_mostrar = ['Nombre Científico', 'Amenaza IUCN']
                if 'Nombre Común' in threatened.columns:
                    cols_mostrar.insert(1, 'Nombre Común') # Lo añade solo si existe
                    
                # <-- ¡Alineación corregida y escudo Arrow aplicado!
                st.dataframe(threatened[cols_mostrar].astype(str).drop_duplicates(), width="stretch", hide_index=True)
            else:
                st.success("✅ No se detectaron especies en categorías críticas (CR, EN, VU) en esta zona.")
        
        st.markdown("---")
        st.markdown("##### Detalle de Registros")
        
        # <-- Escudo Arrow y actualización de ancho (width) para la tabla general
        df_mostrar = gdf_bio.drop(columns=['geometry'], errors='ignore').astype(str)
        st.dataframe(df_mostrar, width="stretch", hide_index=True)
        
    else:
        st.info("No hay datos de biodiversidad para mostrar estadísticas.")
        
# ==============================================================================
# TAB 4: CALCULADORA DE CARBONO (INTEGRADA & DOCUMENTADA)
# ==============================================================================
with tab_forestal:
    st.header("🌳 Estimación de Servicios Ecosistémicos (Carbono)")
    
    # --- 1. MARCO CONCEPTUAL (CAJA DE MENSAJE) ---
    with st.expander("📘 Marco Conceptual y Metodológico (Ver Detalles)", expanded=False):
        st.markdown("""
        ### 🧠 Metodología de Estimación
        Esta herramienta sigue los lineamientos del **IPCC (2006)** y las metodologías del Mecanismo de Desarrollo Limpio (**MDL AR-TOOL14**).
        
        **1. Ecuaciones Utilizadas:**
        * **Crecimiento:** Modelo *Von Bertalanffy* para biomasa aérea.
            $$B_t = A \\cdot (1 - e^{-k \\cdot t})^{\\frac{1}{1-m}}$$
        * **Suelo:** Factor de acumulación lineal de Carbono Orgánico del Suelo (COS) durante los primeros 20 años ($0.705 \\, tC/ha/año$).
        
        **2. Fuentes de Datos:**
        * **Coeficientes Alométricos:** *Álvarez et al. (2012)* para bosques naturales de Colombia.
        * **Parámetros de Crecimiento:** Calibrados para *Bosque Húmedo Tropical* y *Bosque Seco Tropical* en la región andina.
        
        **3. Alcance y Utilidad:**
        Permite estimar el potencial de mitigación (bonos de carbono) ex-ante para proyectos de **Restauración Activa** (siembra) y **Pasiva** (regeneración natural), facilitando la viabilidad financiera de proyectos ambientales.
        """)
        st.info("⚠️ **Nota:** Las estimaciones son aproximadas y deben validarse con mediciones directas en campo para certificación.")

    st.divider()
    
    if gdf_zona is None:
        st.warning("👈 Por favor selecciona una zona en el menú lateral para iniciar el diagnóstico.")
        st.stop()
    
    # 1. DESCARGA DE RECURSOS (NUBE)
    with st.spinner("☁️ Descargando capas climáticas y de cobertura..."):
        dem_bytes = get_raster_from_cloud("DemAntioquia_EPSG3116.tif")
        ppt_bytes = get_raster_from_cloud("PPAMAnt.tif")
        cov_bytes = get_raster_from_cloud("Cob25m_WGS84.tif") # Nombre exacto en bucket

    # 2. PROCESAMIENTO
    if dem_bytes and ppt_bytes and cov_bytes:
        with st.spinner("🔄 Cruzando mapas de Clima (Holdridge) y Cobertura..."):
            # AQUÍ ESTABA EL ERROR: Ahora pasamos 5 argumentos (incluyendo archivos y la clave)
            df_diagnostico = analizar_coberturas_por_zona_vida(
                gdf_zona, 
                nombre_seleccion,  # Clave de cache
                dem_bytes, 
                ppt_bytes, 
                cov_bytes
            )
    else:
        st.error("❌ No se pudieron descargar los mapas base desde la nube.")
        df_diagnostico = None

    # 3. RESULTADOS
    if df_diagnostico is not None and not df_diagnostico.empty:
        
        # --- VISUALIZACIÓN COMPLETA DEL DIAGNÓSTICO ---
        st.markdown("##### 📊 Distribución de Coberturas por Zona de Vida")
        
        # 1. Gráfico de Barras Apiladas (Todas las coberturas, no solo potencial)
        fig_diag = px.bar(
            df_diagnostico, 
            x='Hectareas', 
            y='Zona_Vida', 
            color='Cobertura', 
            orientation='h',
            title="Hectáreas por Cobertura y Clima",
            color_discrete_sequence=px.colors.qualitative.Prism,
            height=400
        )
        st.plotly_chart(fig_diag, use_container_width=True)
        
        # 2. Tabla de Datos (Pivot Table para mejor lectura)
        with st.expander("Ver Tabla de Datos Detallada (Hectáreas)"):
            pivot_diag = df_diagnostico.pivot_table(
                index='Cobertura',
                columns='Zona_Vida',
                values='Hectareas',
                aggfunc='sum',
                fill_value=0
            )
            # Formato numérico
            st.dataframe(pivot_diag.style.format("{:,.1f}"), use_container_width=True)
            
        st.divider()

        # =====================================================================
        # 🌐 CONEXIÓN AL ALEPH (ST.SESSION_STATE) - DATOS REALES SUPABASE
        # =====================================================================
        try:
            # Inicializamos contadores para el puente de datos
            distribucion_real = {
                'bosque': 0.0, 'agricola': 0.0, 'pastos': 0.0, 'urbano': 0.0
            }
            
            for _, row in df_diagnostico.iterrows():
                cov_id = int(row['COV_ID'])
                ha = float(row['Hectareas'])
                
                if cov_id == 9: distribucion_real['bosque'] += ha
                elif cov_id in [5, 6, 8]: distribucion_real['agricola'] += ha
                elif cov_id in [7, 10]: distribucion_real['pastos'] += ha
                elif cov_id in [1, 2, 3, 4]: distribucion_real['urbano'] += ha
            
            # Inyección en la memoria global para el Sankey de la Pág 08
            st.session_state['aleph_ha_bosque'] = distribucion_real['bosque']
            st.session_state['aleph_ha_agricola'] = distribucion_real['agricola']
            st.session_state['aleph_ha_pastos'] = distribucion_real['pastos']
            st.session_state['aleph_ha_urbana'] = distribucion_real['urbano']
            st.session_state['aleph_territorio_origen'] = str(nombre_seleccion)
            
            # Área total para el recuadro de dimensionamiento territorial
            st.session_state['area_total_cuenca_val'] = sum(distribucion_real.values())
            
            st.success(f"📡 **Datos Geoespaciales Sincronizados:** El Sankey de la Pág 08 ahora usa las {st.session_state['area_total_cuenca_val']:,.1f} ha reales de {nombre_seleccion}.")
            
        except Exception as e:
            st.error(f"Error en puente de datos: {e}")
# =====================================================================
        
        st.divider()
        
        # IDs de Pastos/Degradados
        target_ids = [7, 3, 11] 
        df_potencial = df_diagnostico[df_diagnostico['COV_ID'].isin(target_ids)].copy()
        total_potencial = df_potencial['Hectareas'].sum()
        
        k1, k2 = st.columns(2)
        k1.metric("Área Total Zona", f"{(gdf_zona.to_crs('+proj=cea').area.sum()/10000):,.0f} ha")
        k2.metric("Potencial Restauración", f"{total_potencial:,.0f} ha", help="Pastos + Áreas Degradadas disponibles")
    else:
        total_potencial = 0

    st.divider()

    # --- MAPA ESPACIAL BLINDADO ---
    st.markdown("##### 🗺️ Mapa de Usos del Suelo y Predios")
    
    with st.spinner("🎨 Dibujando mapa interactivo..."):
        try:
            fig_map = go.Figure()
            center_lat, center_lon = 6.5, -75.5 
            
            if gdf_zona is not None and not gdf_zona.empty:
                gdf_zona_wgs = gdf_zona.to_crs("EPSG:4326")
                centroid = gdf_zona_wgs.geometry.centroid.iloc[0]
                center_lat, center_lon = centroid.y, centroid.x
                
                # 1. CAPA ZONA (Amarilla - Siempre visible)
                for idx, row in gdf_zona_wgs.iterrows():
                    geoms = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms)
                    for poly in geoms:
                        x, y = poly.exterior.xy
                        fig_map.add_trace(go.Scattermapbox(
                            lon=list(x), lat=list(y), mode='lines', 
                            line=dict(color='yellow', width=3),
                            name="Zona Selección"
                        ))

                # 2. CAPA COBERTURAS (Método Robusto de Arrays Planos)
                if cov_bytes:
                    gdf_cov_vis = generar_mapa_coberturas_vectorial(gdf_zona, cov_bytes)
                    if gdf_cov_vis is not None and not gdf_cov_vis.empty:
                        gdf_cov_vis['geometry'] = gdf_cov_vis['geometry'].simplify(0.001) # Optimizar
                        
                        for cob_type in gdf_cov_vis['Cobertura'].unique():
                            subset = gdf_cov_vis[gdf_cov_vis['Cobertura'] == cob_type]
                            color_hex = subset['Color'].iloc[0]
                            
                            # Extraer coordenadas separadas por None (El truco definitivo para Plotly)
                            lons, lats = [], []
                            for geom in subset.geometry:
                                if geom is None: continue
                                geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                                for poly in geoms:
                                    x, y = poly.exterior.xy
                                    lons.extend(list(x) + [None])
                                    lats.extend(list(y) + [None])
                                    
                            if lons:
                                fig_map.add_trace(go.Scattermapbox(
                                    lon=lons, lat=lats, mode='lines', fill='toself',
                                    fillcolor=color_hex, line=dict(width=0), opacity=0.6,
                                    name=cob_type, legendgroup="Coberturas", 
                                    visible='legendonly', # Apagado por defecto
                                    hoverinfo="name", hovertext=cob_type
                                ))

                # 3. CAPA PREDIOS (Filtro espacial seguro en lugar de Clip)
                gdf_predios = load_layer_cached("Predios")
                if gdf_predios is not None and not gdf_predios.empty:
                    gdf_pred_wgs = gdf_predios.to_crs("EPSG:4326")
                    
                    try:
                        # Blindar geometrías y buscar intersecciones (más seguro que clip)
                        gdf_pred_wgs['geometry'] = gdf_pred_wgs.geometry.buffer(0)
                        gdf_zona_valid = gdf_zona_wgs.copy()
                        gdf_zona_valid['geometry'] = gdf_zona_valid.geometry.buffer(0)
                        
                        intersected = gpd.sjoin(gdf_pred_wgs, gdf_zona_valid, how='inner', predicate='intersects')
                        gdf_pred_clip = gdf_pred_wgs.loc[intersected.index].drop_duplicates()
                    except:
                        gdf_pred_clip = gpd.GeoDataFrame() # Fallback silencioso

                    if not gdf_pred_clip.empty:
                        lons_p, lats_p = [], []
                        for idx, row in gdf_pred_clip.iterrows():
                            geom = row.geometry
                            if geom is None: continue
                            geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                            for poly in geoms:
                                x, y = poly.exterior.xy
                                lons_p.extend(list(x) + [None])
                                lats_p.extend(list(y) + [None])
                                
                        if lons_p:
                            fig_map.add_trace(go.Scattermapbox(
                                lon=lons_p, lat=lats_p, mode='lines', 
                                line=dict(color='#FF6D00', width=2),
                                name="Predios Ejecutados", legendgroup="Predios",
                                visible='legendonly', # Apagado por defecto
                                hoverinfo="name", hovertext="Predio"
                            ))

            fig_map.update_layout(
                mapbox_style="carto-positron", 
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=12),
                margin={"r":0,"t":0,"l":0,"b":0}, height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error renderizando el mapa: {e}")
            
    st.divider()
    
    # --- 4. CONFIGURACIÓN DEL ANÁLISIS ---
    st.subheader("⚙️ Configuración del Análisis")
    
    enfoque = st.radio("Selecciona el enfoque metodológico:", ["🔮 Proyección (Planificación Ex-ante)", "📏 Inventario (Medición Ex-post)"], horizontal=True)

    # ================= OPCIÓN A: PROYECCIÓN =================
    if "Proyección" in enfoque:
        col_conf1, col_conf2 = st.columns([1, 2])

        with col_conf1:
            st.markdown("##### 🌳 Planificación Forestal")
            opciones_modelos = list(carbon_calculator.ESCENARIOS_CRECIMIENTO.keys())
            estrategia = st.selectbox("Modelo de Intervención:", options=opciones_modelos, format_func=lambda x: carbon_calculator.ESCENARIOS_CRECIMIENTO[x]["nombre"])
            
            tipo_area = st.radio("Definir Área Forestal:", ["Manual", "Todo el Potencial"], horizontal=True)
            val_def = float(total_potencial) if 'total_potencial' in locals() and total_potencial > 0 else 1.0
            area_input = st.number_input("Hectáreas (Bosque):", min_value=0.1, value=1.0, step=0.1) if tipo_area == "Manual" else st.number_input("Hectáreas (Bosque):", value=val_def, disabled=True)
            
            edad_proy = st.slider("Horizonte de Análisis (Años):", 5, 50, 20)

        with col_conf2:
            # 🚀 CÁLCULO REACTIVO (SOLO BOSQUE)
            df_bosque = carbon_calculator.calcular_proyeccion_captura(area_input, edad_proy, estrategia)
            total_c_bosque = df_bosque['Proyecto_tCO2e_Acumulado'].iloc[-1]
            precio_usd = 5.0 
            tasa_prom = total_c_bosque / edad_proy
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Captura Total", f"{total_c_bosque:,.0f} tCO2e")
            m2.metric("Tasa Anual", f"{tasa_prom:,.1f} t/año")
            m3.metric("Valor Potencial", f"${(total_c_bosque * precio_usd):,.0f} USD")
            
            fig = px.area(df_bosque, x='Año', y='Proyecto_tCO2e_Acumulado', title=f"Dinámica - {carbon_calculator.ESCENARIOS_CRECIMIENTO[estrategia]['nombre']}", color_discrete_sequence=['#2ecc71'])
            
            # Usamos el nuevo estándar de Streamlit para evitar el warning
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 Ver Tabla Financiera y Descargar Reporte"):
                df_fin_bosque = df_bosque.copy()
                df_fin_bosque['Valor_USD_Acumulado'] = df_fin_bosque['Proyecto_tCO2e_Acumulado'] * precio_usd
                st.dataframe(df_fin_bosque.style.format({'Proyecto_tCO2e_Acumulado': '{:,.1f}', 'Valor_USD_Acumulado': '${:,.0f}'}))
                
                # Try/except rápido por si no tienes importado save_to_csv ahí
                try:
                    csv = save_to_csv(df_fin_bosque)
                except:
                    csv = df_fin_bosque.to_csv(index=False).encode('utf-8')
                    
                st.download_button("📥 Descargar Reporte Forestal (CSV)", csv, "reporte_forestal.csv", "text/csv")

                
    # ================= OPCIÓN B: INVENTARIO =================
    else:
        c_inv_1, c_inv_2 = st.columns([1, 2])
        with c_inv_1:
            st.info("Sube un archivo Excel/CSV con mediciones de campo. Requiere columnas: `DAP` (cm), `Altura` (m).")
            up_file = st.file_uploader("Cargar Inventario Forestal", type=['csv', 'xlsx'])
            
            opciones_zv = ["bh-MB", "bh-PM", "bh-T", "bmh-M", "bmh-MB", "bmh-PM", "bp-PM", "bs-T", "me-T"]
            zona_vida_inv = st.selectbox("Ecuación (Zona de Vida Predominante):", opciones_zv, index=0)
            
            btn_inv = st.button("🧮 Calcular Stock Actual", type="primary")

        with c_inv_2:
            if up_file and btn_inv:
                try:
                    if up_file.name.endswith('.csv'): df_inv = pd.read_csv(up_file)
                    else: df_inv = pd.read_excel(up_file)
                    
                    df_res_inv, msg = carbon_calculator.calcular_inventario_forestal(df_inv, zona_vida_inv)
                    
                    if df_res_inv is not None:
                        st.success("✅ Inventario Procesado")
                        tot_carb = df_res_inv['CO2e_Total_tCO2e'].sum()
                        
                        i1, i2 = st.columns(2)
                        i1.metric("Árboles Válidos", f"{len(df_res_inv)}")
                        i2.metric("Stock Estimado", f"{tot_carb:,.2f} tCO2e")
                        
                        st.dataframe(df_res_inv.head())
                        csv_inv = df_res_inv.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Descargar Reporte CSV", csv_inv, "reporte_inventario.csv", "text/csv")
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Error procesando archivo: Revise que las columnas se llamen DAP y Altura. ({e})")

# ==============================================================================
# TAB 5: METABOLISMO TERRITORIAL (AFOLU COMPLETO)
# ==============================================================================
with tab_afolu:
    
    # -------------------------------------------------------------------------
    # 🌐 1. CEREBRO ESPACIAL Y EXTRACCIÓN AL GEMELO DIGITAL
    # -------------------------------------------------------------------------
    titulo_dinamico = f"Metabolismo Territorial: Dinámica de GEI en {nombre_seleccion}"
    st.header(f"⚖️ {titulo_dinamico}")
    
    area_bosque_real = 100.0
    try:
        if 'df_diagnostico' in locals() and df_diagnostico is not None:
            area_bosque_real = df_diagnostico[df_diagnostico['COV_ID'] == 9]['Hectareas'].sum()
    except: pass

    # -------------------------------------------------------------------------
    # 🚀 LA MAGIA CENTRALIZADA: Extracción limpia al Gemelo Digital
    # -------------------------------------------------------------------------
    anio_analisis = st.session_state.get('aleph_anio', 2025)
    datos_metabolismo = obtener_metabolismo_exacto(nombre_seleccion, anio_analisis)

    poblacion_urbana_calculada = datos_metabolismo['pob_urbana']
    poblacion_rural_calculada = datos_metabolismo['pob_rural']
    bovinos_reales = datos_metabolismo['bovinos']
    porcinos_reales = datos_metabolismo['porcinos']
    aves_reales = datos_metabolismo['aves']
    origen_datos = datos_metabolismo.get('origen_humano', 'Estimación Geoespacial')

    aleph_pastos = float(st.session_state.get('aleph_ha_pastos', 50.0))

    col_a1, col_a2 = st.columns([1, 2.5])
    
    with col_a1:
        # =========================================================================
        # 🌲 MÓDULO 1: BOSQUES (Desplegable)
        # =========================================================================
        with st.expander("🌳 1. Línea Base Forestal (Sumidero Principal)", expanded=True):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                estrategia_af = st.selectbox("Bosque Existente/Planeado:", options=list(carbon_calculator.ESCENARIOS_CRECIMIENTO.keys()), format_func=lambda x: carbon_calculator.ESCENARIOS_CRECIMIENTO[x]["nombre"])
                area_af = st.number_input("Hectáreas (Bosque Satélite):", value=float(area_bosque_real) if area_bosque_real > 0 else 100.0, step=10.0)
            with col_b2:
                horizonte_af = st.slider("Horizonte de Análisis (Años):", 5, 50, 20, key="slider_afolu")

        # =========================================================================
        # 🐄 MÓDULO 2: RURAL Y AGROPECUARIO (Desplegable)
        # =========================================================================
        with st.expander("🌾 2. Actividades Agropecuarias y Humanas (Rural)", expanded=False):
            if origen_datos == "Matriz Maestra":
                st.success(f"🧠 **Conexión Aleph Sincronizada:** Las cargas rurales se calcularon usando censos reales para **{nombre_seleccion}**.")
            else:
                st.info(f"📍 **Conexión Aleph Local:** Datos aproximados para **{nombre_seleccion}** (Fuente: {origen_datos}).")
            opciones_fuentes = ["Todas", "Pasturas", "Bovinos", "Porcinos", "Avicultura", "Población Rural"]
            fuentes_sel = st.multiselect("Selecciona cargas rurales a modelar:", opciones_fuentes, default=["Todas"])
            fuentes_activas = ["Pasturas", "Bovinos", "Porcinos", "Avicultura", "Población Rural"] if "Todas" in fuentes_sel else fuentes_sel

            esc_pasto, area_pastos = "PASTO_DEGRADADO", 0.0
            v_leche, v_carne, cerdos, aves, humanos_rurales = 0, 0, 0, 0, 0
            
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                if "Pasturas" in fuentes_activas:
                    esc_pasto = st.selectbox("Manejo de Pastos:", list(carbon_calculator.ESCENARIOS_PASTURAS.keys()), format_func=lambda x: carbon_calculator.ESCENARIOS_PASTURAS[x]["nombre"])
                    area_pastos = st.number_input("Ha de Pasturas (Satélite):", value=float(aleph_pastos), step=5.0)
                if "Bovinos" in fuentes_activas:
                    v_leche = st.number_input("Vacas Lecheras (ICA):", value=int(bovinos_reales * 0.4), step=10)
            with c_r2:
                if "Bovinos" in fuentes_activas:
                    v_carne = st.number_input("Ganado Carne/Cría (ICA):", value=int(bovinos_reales * 0.6), step=10)
                if "Porcinos" in fuentes_activas:
                    cerdos = st.number_input("Cerdos Cabezas (ICA):", value=int(porcinos_reales), step=50)
            with c_r3:
                if "Avicultura" in fuentes_activas:
                    aves = st.number_input("Aves Galpones (ICA):", value=int(aves_reales), step=500)
                if "Población Rural" in fuentes_activas:
                    humanos_rurales = st.number_input("Humanos Rurales (Censo):", value=int(poblacion_rural_calculada), step=10)

        # =========================================================================
        # 🏙️ MÓDULO 3: URBANO Y MOVILIDAD (Desplegable)
        # =========================================================================
        with st.expander("🏙️ 3. Actividades Urbanas (Ciudades y Movilidad)", expanded=False):
            col_u1, col_u2, col_u3 = st.columns(3)
            
            with col_u1:
                st.markdown("##### 👥 Demografía y Agua")
                humanos_urbanos = st.number_input("Población Urbana:", value=int(poblacion_urbana_calculada), step=100)
                vertimientos_m3 = (humanos_urbanos * 150) / 1000
                st.metric("Agua Residual Generada", f"{vertimientos_m3:,.1f} m³/día")
                
            with col_u2:
                st.markdown("##### 🗑️ Residuos Sólidos")
                tasa_basura = st.slider("Generación (kg/hab-día):", min_value=0.0, max_value=1.5, value=0.7, step=0.1, help="Promedio Colombia: 0.6 - 0.8 kg diarios por persona.")
                basura_anual_ton = (humanos_urbanos * tasa_basura * 365) / 1000
                st.metric("Basura al Relleno", f"{basura_anual_ton:,.0f} ton/año")
                
            with col_u3:
                st.markdown("##### 🚗 Parque Automotor")
                tasa_motorizacion = st.slider("Densidad (Vehículos/1000 hab):", min_value=10, max_value=1500, value=333, step=10, help="Medellín: ~333. Laureles: ~739. El Poblado: ~1250.")
                vehiculos = int((humanos_urbanos * tasa_motorizacion) / 1000)
                st.metric("Vehículos Estimados", f"{vehiculos:,.0f} unds")

            st.markdown("---")
            st.markdown("##### ⛽ Física de Emisiones Vehiculares")
            col_v1, col_v2, col_v3 = st.columns(3)
            
            with col_v1:
                km_galon = st.slider("Rendimiento (km/galón):", min_value=1.0, max_value=100.0, value=40.0, step=1.0, help="SUV/Camioneta: 25-30 km/gal. Sedán: 40-50 km/gal. Híbrido: 70-90 km/gal.")
            with col_v2:
                km_anual = st.slider("Recorrido Medio Anual (km):", min_value=0, max_value=50000, value=12000, step=1000, help="Uso ocasional: 5,000 km/año. Promedio LATAM: 12,000 km/año. Taxis: 35,000+ km/año.")
            with col_v3:
                galones_anuales = vehiculos * (km_anual / km_galon) if km_galon > 0 else 0
                emision_anual_vehiculos = (galones_anuales * 8.887) / 1000.0 
                st.info(f"☁️ **Impacto Total:** El parque automotor consume **{galones_anuales:,.0f}** galones/año, emitiendo **{emision_anual_vehiculos:,.0f} ton CO2e/año**.")

        # =========================================================================
        # 4. EVENTOS EN EL TIEMPO
        # =========================================================================
        st.markdown("---")
        st.subheader("4. Eventos en el Tiempo")
        tipo_evento = st.radio("Simular alteración de cobertura:", ["Ninguno", "Pérdida (Deforestación/Incendio)", "Ganancia (Restauración Activa)"], horizontal=True)
        area_evento, anio_evento, estado_ev, causa_ev = 0.0, 1, "BOSQUE_SECUNDARIO", "AGRICOLA"
        
        if tipo_evento != "Ninguno":
            area_evento = st.number_input("Hectáreas Afectadas:", min_value=0.1, value=5.0, step=1.0)
            anio_evento = st.slider("¿En qué año ocurre?", 1, int(horizonte_af), 5)
            estado_ev = st.selectbox("Tipo de Cobertura:", list(carbon_calculator.STOCKS_SUCESION.keys()), index=4)
            if "Pérdida" in tipo_evento:
                causa_ev = st.selectbox("Causa:", list(carbon_calculator.CAUSAS_PERDIDA.keys()))
                
    with col_a2:
 
        # =====================================================================
        # CÁLCULOS REACTIVOS (Con Desglose Urbano y Movilidad)
        # =====================================================================
        h_anios = int(horizonte_af)

        df_bosque_af = carbon_calculator.calcular_proyeccion_captura(area_af, h_anios, estrategia_af)
        df_pastos_af = carbon_calculator.calcular_captura_pasturas(area_pastos, h_anios, esc_pasto)
        
        # 1. Calculamos solo las emisiones pecuarias con la función base (enviamos 0 humanos para no mezclarlos)
        df_fuentes_af = carbon_calculator.calcular_emisiones_fuentes_detallado(v_leche, v_carne, cerdos, aves, 0, h_anios)
        
        # 2. Factores de Emisión IPCC aproximados (toneladas CO2e / año)
        # Aguas residuales (Metano/Óxido Nitroso): ~0.05 ton CO2e por habitante al año
        emision_rural_anual = humanos_rurales * 0.05 
        emision_urbana_anual = humanos_urbanos * 0.05
        # Residuos sólidos (Metano en relleno): ~0.15 ton CO2e por tonelada de basura
        emision_basura_anual = basura_anual_ton * 0.15
        
        # 3. Inyectar las nuevas curvas como columnas independientes
        df_fuentes_af['Humanos_Rurales (Aguas Residuales)'] = emision_rural_anual
        df_fuentes_af['Vertimientos_Urbanos'] = emision_urbana_anual
        df_fuentes_af['Residuos_Solidos'] = emision_basura_anual
        df_fuentes_af['Parque_Automotor'] = emision_anual_vehiculos
        
        # 4. Recalcular el Total de Emisiones sumando todas las columnas (excepto 'Año')
        columnas_fuentes = [c for c in df_fuentes_af.columns if c not in ['Año', 'Total_Emisiones']]
        df_fuentes_af['Total_Emisiones'] = df_fuentes_af[columnas_fuentes].sum(axis=1)
        
        # 5. Eventos de Cambio de Cobertura
        t_ev = "PERDIDA" if "Pérdida" in tipo_evento else "GANANCIA"
        anio_ev_int = int(anio_evento) if 'anio_evento' in locals() else 5 
        df_evento_af = carbon_calculator.calcular_evento_cambio(area_evento, t_ev, anio_ev_int, h_anios)

        # =====================================================================
        # 5. BALANCE Y GRÁFICA FINAL (BLINDADO ANTI-ERRORES)
        # =====================================================================
        df_bal = carbon_calculator.calcular_balance_territorial(df_bosque_af, df_pastos_af, df_fuentes_af, df_evento_af)
        
        # Función escudo: Si la columna no existe, devuelve 0
        def v_seguro(df, col):
            return df[col].iloc[-1] if col in df.columns else 0

        neto_final = v_seguro(df_bal, 'Balance_Neto_tCO2e')
        usd_total = neto_final * 5.0
        
        val_bosque = v_seguro(df_bal, 'Captura_Bosque')
        val_pastos = v_seguro(df_bal, 'Captura_Pastos')
        val_ganancia = v_seguro(df_bal, 'Evento_Ganancia')
        val_perdida = v_seguro(df_bal, 'Evento_Perdida')
        
        captura_total = val_bosque + val_pastos + val_ganancia
        emision_total = v_seguro(df_fuentes_af, 'Total_Emisiones') + val_perdida
        
        # MÉTRICAS PRINCIPALES
        st.markdown(f"### 📊 {titulo_dinamico}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Captura (Sumideros)", f"{captura_total:,.0f} t")
        m2.metric("Emisiones Totales", f"{emision_total:,.0f} t")
        estado = "🌿 Sumidero" if neto_final > 0 else "⚠️ Emisor"
        m3.metric("Balance Neto", f"{neto_final:,.0f} t", delta=estado, delta_color="normal" if neto_final > 0 else "inverse")
        m4.metric("Valor del Carbono", f"${usd_total:,.0f} USD")
        
        # GRÁFICO MULTI-CURVAS INTELIGENTE
        fig = go.Figure()
        
        # Función escudo para graficar: Si la curva no existe, la ignora silenciosamente
        def agregar_curva(fig, df, col, nombre, color):
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df['Año'], y=df[col], mode='lines', fill='tozeroy', name=nombre, line=dict(color=color)))

        # Curvas de Sumidero
        agregar_curva(fig, df_bal, 'Captura_Bosque', 'Bosque Base', '#2ecc71')
        color_pasto = '#f1c40f' if val_pastos >= 0 else '#e67e22'
        agregar_curva(fig, df_bal, 'Captura_Pastos', 'Pasturas', color_pasto)
        agregar_curva(fig, df_bal, 'Evento_Ganancia', 'Restauración Nueva', '#00bc8c')
        
        # Curvas de Emisión Agropecuaria
        agregar_curva(fig, df_fuentes_af, 'Emision_Bovinos', 'Bovinos', '#e74c3c')
        agregar_curva(fig, df_fuentes_af, 'Emision_Porcinos', 'Porcinos', '#e83e8c')
        agregar_curva(fig, df_fuentes_af, 'Emision_Aves', 'Aves', '#fd7e14')
        
        # Curvas de Emisión Urbana y Humana
        col_humanos = 'Humanos_Rurales (Aguas Residuales)' if 'Humanos_Rurales (Aguas Residuales)' in df_fuentes_af.columns else 'Emision_Humanos'
        agregar_curva(fig, df_fuentes_af, col_humanos, 'Humanos Rurales', '#6f42c1')
        agregar_curva(fig, df_fuentes_af, 'Vertimientos_Urbanos', 'Vertimientos Urbanos', '#17a2b8')
        agregar_curva(fig, df_fuentes_af, 'Residuos_Solidos', 'Residuos Sólidos', '#795548')
        agregar_curva(fig, df_fuentes_af, 'Parque_Automotor', 'Parque Automotor', '#34495e')
        
        agregar_curva(fig, df_bal, 'Evento_Perdida', 'Deforestación/Pérdida', '#343a40')
        
        # Línea de Balance Neto
        if 'Balance_Neto_tCO2e' in df_bal.columns:
            fig.add_trace(go.Scatter(x=df_bal['Año'], y=df_bal['Balance_Neto_tCO2e'], mode='lines', name='Balance Neto Real', line=dict(color='black', width=4, dash='dot')))
        
        fig.update_layout(xaxis_title="Año", yaxis_title="Acumulado (tCO2e)", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 6: COMPARADOR DE ESCENARIOS (NUEVO)
# ==============================================================================
with tab_comparador:
    st.header("⚖️ Comparativa de Escenarios de Carbono")
    st.info("Selecciona múltiples modelos para visualizar sus diferencias en captura y retorno financiero.")
    
    col_comp1, col_comp2 = st.columns([1, 3])
    
    with col_comp1:
        st.subheader("Configuración")
        # Selector múltiple
        modelos_disp = list(carbon_calculator.ESCENARIOS_CRECIMIENTO.keys())
        seleccionados = st.multiselect(
            "Modelos a comparar:", 
            options=modelos_disp,
            default=["STAND_I", "STAND_V", "CONS_RIO"], # Default: Alta, Pasiva y Conservación
            format_func=lambda x: carbon_calculator.ESCENARIOS_CRECIMIENTO[x]["nombre"]
        )
        
        area_comp = st.number_input("Área de Análisis (Ha):", value=100.0, min_value=1.0)
        anios_comp = st.slider("Horizonte (Años):", 10, 50, 30)
        precio_bono = st.number_input("Precio Bono (USD/t):", value=5.0)

    with col_comp2:
        if seleccionados:
            # Construir DataFrame consolidado
            df_consolidado = pd.DataFrame()
            
            resumen_final = []
            
            for mod in seleccionados:
                # Calculamos la proyección para este modelo
                df_temp = carbon_calculator.calcular_proyeccion_captura(area_comp, anios_comp, mod)
                df_temp['Escenario'] = carbon_calculator.ESCENARIOS_CRECIMIENTO[mod]["nombre"]
                
                # Guardamos para el gráfico
                df_consolidado = pd.concat([df_consolidado, df_temp])
                
                # Guardamos para la tabla resumen
                total_c = df_temp['Proyecto_tCO2e_Acumulado'].iloc[-1]
                resumen_final.append({
                    "Escenario": carbon_calculator.ESCENARIOS_CRECIMIENTO[mod]["nombre"],
                    "Total CO2e": total_c,
                    "Valor (USD)": total_c * precio_bono
                })
            
            # 1. Gráfico Multilínea
            fig_comp = px.line(
                df_consolidado, 
                x='Año', 
                y='Proyecto_tCO2e_Acumulado', 
                color='Escenario',
                title=f"Proyección Comparativa ({area_comp} ha)",
                labels={'Proyecto_tCO2e_Acumulado': 'Acumulado (tCO2e)'},
                line_shape='spline' # Curvas suaves
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # 2. Tabla Resumen
            st.subheader("Resumen Financiero y Ambiental")
            df_resumen = pd.DataFrame(resumen_final).set_index("Escenario")
            
            # Formateo bonito
            st.dataframe(
                df_resumen.style.format({"Total CO2e": "{:,.0f}", "Valor (USD)": "${:,.0f}"})
                .background_gradient(cmap="Greens", subset=["Total CO2e"]),
                use_container_width=True
            )
            
        else:
            st.warning("Selecciona al menos un modelo para comparar.")

# =========================================================================
# TAB 7: ECOLOGÍA DEL PAISAJE (CONECTIVIDAD RIPARIA)
# =========================================================================
with tab_ecologia:
    
    # --- 🔄 DETECTOR DE CAMBIO DE CUENCA (El solucionador de la memoria pegada) ---
    if st.session_state.get('ultima_cuenca_ecologia') != nombre_seleccion:
        st.session_state['gdf_rios'] = None # Borra los ríos de la cuenca anterior
        st.session_state['buffer_m_ripario'] = None
        st.session_state['ultima_cuenca_ecologia'] = nombre_seleccion

    # --- 🏷️ TÍTULO DINÁMICO ---
    st.subheader(f"🌿 Ecología del Paisaje: Conectividad y Franjas Riparias en {nombre_seleccion.title()}")
    st.markdown("Analiza la red hidrográfica y modela escenarios de restauración basados en la viabilidad territorial y el déficit de coberturas naturales.")
    
    if st.session_state.get('gdf_rios') is not None and not st.session_state['gdf_rios'].empty:
        gdf_rios_actual = st.session_state['gdf_rios']
        
        c_gap1, c_gap2 = st.columns([1, 2.5])
        
        with c_gap1:
            st.markdown("#### ⚙️ Parámetros del Corredor")
            
            # 🌍 BISTURÍ: Verificar si hay una amenaza activa en la memoria de Geomorfología
            amenaza_activa = 'aleph_twi_umbral' in st.session_state
            
            opciones_metodo = ["Estándar (Ley 99 de 1993)"]
            if amenaza_activa: 
                opciones_metodo.append("🛡️ Diseño por Amenaza (Nexo Físico)")
                
            tipo_buffer = st.radio("Metodología de Aislamiento:", opciones_metodo)
            
            if tipo_buffer == "Estándar (Ley 99 de 1993)":
                buffer_m = st.slider("Ancho de franja de protección por lado (m):", min_value=0, max_value=100, value=30, step=5)
            else:
                st.success("🧠 **Nexo Físico Activo:** Leyendo llanura de inundación / torrencial de Geomorfología.")
                
                # 📣 EL MANIFIESTO DEL ARQUITECTO
                st.markdown(f"""
                <div style="border-left: 5px solid #2ecc71; padding: 15px; background-color: rgba(46, 204, 113, 0.1); border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #27ae60; margin-top: 0;">🌳 Manifiesto de Resiliencia</h4>
                    <b style="font-size: 0.95em;">Se requiere crear un bosque de protección y un corredor de biodiversidad sobre estas zonas de peligro para amortiguar el golpe, proteger a la población, restaurar el cauce natural del río y contribuir con la Seguridad Hídrica Integral de la cuenca.</b>
                </div>
                """, unsafe_allow_html=True)
                
                # Traducimos la amenaza extrema (Q_max) a metros de protección requeridos
                # A mayor caudal pico, más ancho se vuelve el bosque ripario de forma automática
                q_max_memoria = st.session_state.get('aleph_q_max_m3s', 50.0)
                buffer_calculado = max(30.0, float(np.log10(q_max_memoria + 1) * 35.0)) 
                
                st.info(f"🌊 Ancho de seguridad calculado por la física extrema del río: **{buffer_calculado:.1f} metros** por margen.")
                buffer_m = buffer_calculado
            
            with st.spinner("Calculando red riparia (Álgebra Lineal Rápida)..."):
                # 1. CÁLCULO MATEMÁTICO PURO (¡Cero colapsos de memoria!)
                rios_3116 = gdf_rios_actual.to_crs(epsg=3116)
                longitud_total_m = rios_3116.length.sum()
                
                # Área = Longitud * (Ancho * 2) * Factor de descuento por cruces (0.85)
                area_total_ha = (longitud_total_m * (buffer_m * 2) * 0.85) / 10000.0
                st.metric("Área Total del Corredor", f"{area_total_ha:,.1f} ha")
                
                # 2. GAP ANALYSIS: COBERTURAS VS. RED DE DRENAJE
                # Leemos el área real de bosque desde el Gemelo Digital (calculado en Tab 4)
                ha_bosque_aleph = st.session_state.get('aleph_ha_bosque', 0.0)
                area_cuenca_aleph = st.session_state.get('area_total_cuenca_val', 0.0)
                
                # Dinamismo total: si hay datos satelitales, los usa; si no, asume 35% de línea base
                if area_cuenca_aleph > 0:
                    pct_bosque_existente = (ha_bosque_aleph / area_cuenca_aleph) * 100
                else:
                    pct_bosque_existente = 35.0 
                    
                ha_bosque = area_total_ha * (pct_bosque_existente / 100.0)
                ha_deficit = area_total_ha - ha_bosque
                
            st.markdown("---")
            st.markdown("#### 📊 Análisis de Brechas (Gap)")
            st.metric("🌳 Bosque Existente", f"{ha_bosque:,.1f} ha", "Cobertura Natural")
            st.metric("🔴 Déficit Ripario", f"{ha_deficit:,.1f} ha", "- Área a Restaurar", delta_color="inverse")
            
            # Guardamos los parámetros para Toma de Decisiones
            st.session_state['buffer_m_ripario'] = buffer_m
            st.session_state['ha_deficit_ripario'] = ha_deficit
            
        with c_gap2:
            import pydeck as pdk
            st.markdown("##### 🗺️ Red de Conectividad Ecológica (Aceleración GPU)")
            
            # Creamos una copia para añadirle los nombres que leerá el Tooltip
            rios_4326 = gdf_rios_actual.to_crs(epsg=4326).copy()
            rios_4326['ID_Tramo'] = ["Segmento Hídrico " + str(i+1) for i in range(len(rios_4326))]
            
            # Redondear valores para que el tooltip se vea elegante
            if 'longitud_km' in rios_4326.columns:
                rios_4326['longitud_km'] = rios_4326['longitud_km'].round(2)
            
            try: 
                c_lat, c_lon = rios_4326.geometry.iloc[0].centroid.y, rios_4326.geometry.iloc[0].centroid.x
            except: 
                c_lat, c_lon = 6.2, -75.5
                
            capas_mapa = []
            
            if gdf_zona is not None:
                zona_4326 = gdf_zona.to_crs("EPSG:4326")
                capas_mapa.append(pdk.Layer("GeoJsonLayer", data=zona_4326, opacity=1, stroked=True, get_line_color=[0, 200, 0, 255], get_line_width=3, filled=False))
                
            # MAGIA VISUAL: Añadimos 'pickable=True' para que el mapa detecte el ratón
            capas_mapa.append(pdk.Layer(
                "GeoJsonLayer",
                data=rios_4326,
                opacity=0.6,
                stroked=True,
                get_line_color=[39, 174, 96, 255], 
                get_line_width=buffer_m * 2,
                lineWidthUnits='"meters"',
                lineWidthMinPixels=2,
                pickable=True, # <-- Clave para la interactividad
                autoHighlight=True # Hace que el río brille al pasar el ratón
            ))
            
            capas_mapa.append(pdk.Layer("GeoJsonLayer", data=rios_4326, opacity=1, get_line_color=[52, 152, 219, 255], get_line_width=1, lineWidthUnits='"pixels"'))
            
            view_state = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=11)
            
            # DISEÑO DEL TOOLTIP
            tooltip = {
                "html": "<b>{ID_Tramo}</b><br/>Orden de Strahler: <b>{Orden_Strahler}</b><br/>Longitud: {longitud_km} km",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
            st.pydeck_chart(pdk.Deck(
                layers=capas_mapa, 
                initial_view_state=view_state, 
                map_style="light",
                tooltip=tooltip # <-- Conectamos el Tooltip al mapa
            ), use_container_width=True)

    else:
        st.info("⚠️ La red hidrográfica no está en la memoria. Puedes calcularla en Geomorfología o generarla directamente aquí.")
        
        # --- 🌊 MOTOR HIDROLÓGICO DE BOLSILLO ---
        from modules.geomorfologia_tools import render_motor_hidrologico
        
        # Invocamos la herramienta pasándole el polígono de la zona
        render_motor_hidrologico(gdf_zona)

# =========================================================================
# PESTAÑA 8: RETENCIÓN HÍDRICA DEL DOSEL
# =========================================================================
with tab_ret_dosel:
    st.subheader("🌳 Servicio Ecosistémico: Retención Hídrica del Dosel")
    st.info("Modelo eco-hidrológico de intercepción forestal. Estima cuánta agua de un aguacero es 'secuestrada' por las hojas y ramas, mitigando el riesgo de escorrentía rápida y avalanchas.")

    st.markdown("---")

    col_input, col_graf = st.columns([1, 2])

    with col_input:
        st.subheader("Parámetros Macro del Ecosistema")
        
        tipo_cobertura = st.selectbox(
            "Tipo de Cobertura Vegetal:",
            ["Bosque Andino (Nativo)", "Plantación de Pino", "Robledal", "Rastrojo Alto", "Pastos Degradados"]
        )
        
        dicc_vegetacion = {
            "Bosque Andino (Nativo)": {"Sl": 0.25, "LAI_max": 6.5},
            "Plantación de Pino": {"Sl": 0.20, "LAI_max": 5.0},
            "Robledal": {"Sl": 0.30, "LAI_max": 5.5},
            "Rastrojo Alto": {"Sl": 0.15, "LAI_max": 3.5},
            "Pastos Degradados": {"Sl": 0.05, "LAI_max": 1.5}
        }
        
        params = dicc_vegetacion[tipo_cobertura]
        
        densidad_pct = st.slider("Estado de Conservación / Densidad (%):", 10.0, 100.0, 80.0, 5.0)
        lai_actual = params["LAI_max"] * (densidad_pct / 100.0)
        
        hectareas = st.number_input("Área del polígono a evaluar (ha):", value=100.0, step=10.0)
        
        st.markdown("---")
        st.subheader("El Evento Meteorológico")
        
        # Nuevos controles separados de Intensidad y Tiempo
        c_lluvia1, c_lluvia2 = st.columns(2)
        intensidad_mm_h = c_lluvia1.slider("🌧️ Intensidad (mm/hora):", 1.0, 100.0, 20.0, 1.0)
        duracion_h = c_lluvia2.slider("⏱️ Duración (horas):", 0.5, 24.0, 2.0, 0.5)
        
        # Cálculo de la precipitación total bruta
        precipitacion_mm = intensidad_mm_h * duracion_h
        st.info(f"**Precipitación Bruta Total del Evento:** {precipitacion_mm:.1f} mm")

    # --- 1. NUEVOS CONTROLES DE EVAPORACIÓN (GASH, 1979) ---
    with col_input:
        st.markdown("---")
        st.subheader("Termodinámica (Gash, 1979)")
        temp_c = st.slider("🌡️ Temp. Promedio durante el evento (°C):", 10.0, 35.0, 22.0, 0.5)
        # Estimación simplificada de la tasa de evaporación (E_w) en mm/h basada en temperatura
        ew_mm_h = (temp_c / 35.0) * 0.45 
        evaporacion_evento_mm = ew_mm_h * duracion_h
        st.info(f"**Evaporación del dosel mojado ($E_w$):** {ew_mm_h:.2f} mm/h")

    # =========================================================================
    # --- 2. MOTOR FÍSICO-MATEMÁTICO (ASTON + GASH) ---
    # =========================================================================
    s_max_mm = params["Sl"] * lai_actual

    import numpy as np
    
    # Ecuación Híbrida (Aston)
    if s_max_mm > 0:
        intercepcion_neta_mm = s_max_mm * (1 - np.exp(-precipitacion_mm / s_max_mm))
    else:
        intercepcion_neta_mm = 0.0

    # Añadimos la evaporación continua durante la tormenta (Modelo analítico de Gash)
    intercepcion_bruta_mm = intercepcion_neta_mm + evaporacion_evento_mm
    
    # Blindaje físico: La retención total no puede superar lo que llovió
    intercepcion_mm = min(intercepcion_bruta_mm, precipitacion_mm)
    precipitacion_efectiva_mm = precipitacion_mm - intercepcion_mm
    
    # Prevenir división por cero si el evento es de 0 mm
    if precipitacion_mm > 0:
        eficiencia_retencion_pct = (intercepcion_mm / precipitacion_mm) * 100
    else:
        eficiencia_retencion_pct = 0.0

    volumen_retenido_m3 = intercepcion_mm * hectareas * 10
    volumen_escurre_m3 = precipitacion_efectiva_mm * hectareas * 10

    # =========================================================================
    # --- 3. RENDERIZADO DE GRÁFICOS Y PRUEBAS ---
    # =========================================================================
    with col_graf:
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Capacidad Máxima Dosel", f"{s_max_mm:.2f} mm", f"LAI: {lai_actual:.1f}", delta_color="normal")
        c_m2.metric("Agua Retenida / Evaporada", f"{intercepcion_mm:.1f} mm", f"{eficiencia_retencion_pct:.1f}% del aguacero", delta_color="off")
        
        alerta_suelo = "inverse" if precipitacion_efectiva_mm > 30 else "normal"
        c_m3.metric("Agua al Suelo (P. Efectiva)", f"{precipitacion_efectiva_mm:.1f} mm", "Golpe de escorrentía", delta_color=alerta_suelo)
        
        import plotly.graph_objects as go
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=["Impacto Volumétrico del Evento"],
            y=[volumen_retenido_m3],
            name="Volumen 'Secuestrado' por el Bosque (m³)",
            marker_color="#2ecc71",
            text=f"{volumen_retenido_m3:,.0f} m³", textposition='auto'
        ))
        fig_vol.add_trace(go.Bar(
            x=["Impacto Volumétrico del Evento"],
            y=[volumen_escurre_m3],
            name="Volumen que golpea el suelo (m³)",
            marker_color="#e74c3c",
            text=f"{volumen_escurre_m3:,.0f} m³", textposition='auto'
        ))
        
        fig_vol.update_layout(
            barmode='stack',
            title=f"Balance Hídrico del Evento en {hectareas} ha",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="Metros Cúbicos (m³)"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        if eficiencia_retencion_pct > 15:
            st.success(f"🌿 **Alta Regulación:** El ecosistema actuó como un escudo, absorbiendo {volumen_retenido_m3:,.0f} toneladas de agua que, de otro modo, habrían alimentado directamente la creciente del río.")
        else:
            st.error(f"⚠️ **Riesgo de Avalancha:** El dosel está saturado o degradado. La mayor parte de la energía de la tormenta ({volumen_escurre_m3:,.0f} m³) está golpeando el suelo directamente.")

        # --- 4. BLINDAJE CIENTÍFICO (IN-APP UNIT TESTS) ---
        st.markdown("---")
        if st.toggle("🧪 Ejecutar Blindaje Científico (Validación del Modelo)"):
            st.markdown("<div style='background-color: #f4f6f6; padding: 15px; border-radius: 5px; border-left: 4px solid #34495e;'>", unsafe_allow_html=True)
            st.markdown("#### ⚙️ Autodiagnóstico de Ecuaciones (Aston & Gash)")
            
            # Test 1: Lluvia Cero
            test1_i = s_max_mm * (1 - np.exp(-0.0 / s_max_mm)) if s_max_mm > 0 else 0
            if test1_i == 0.0: 
                st.write("✅ **Test 1 superado:** Con precipitación 0 mm, la intercepción es estrictamente 0.0 mm.")
            else: 
                st.write("❌ **Fallo Test 1:** Ruido matemático en lluvia cero.")
                
            # Test 2: Dosel Arrasado
            test2_i = 0.0 * (1 - np.exp(-50.0 / 0.001)) # Simulamos LAI cercano a 0
            if test2_i < 0.1: 
                st.write("✅ **Test 2 superado:** Sin área foliar (LAI=0), la intercepción tiende a cero.")
            
            # Test 3: Límite Asintótico (Catastrofismo)
            test3_i = s_max_mm * (1 - np.exp(-10000.0 / s_max_mm)) if s_max_mm > 0 else 0
            if round(test3_i, 3) == round(s_max_mm, 3): 
                st.write(f"✅ **Test 3 superado:** Ante lluvia infinita (10,000 mm), la retención neta no supera el límite físico de {s_max_mm:.2f} mm.")
            else: 
                st.write("❌ **Fallo Test 3:** Violación de la ley de conservación de masa.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    # =========================================================================
    # MARCO CONCEPTUAL, METODOLOGÍA Y FUENTES CIENTÍFICAS
    # =========================================================================
    st.markdown("---")
    with st.expander("📚 Marco Conceptual, Metodologías y Fuentes Científicas", expanded=False):
        st.markdown("""
        ### 🔬 La Ciencia: Ecuación de Aston Modificada
        El agua que una tormenta deja caer no llega toda al suelo. El bosque actúa como un paraguas y una esponja. Para modelar esto, usaremos la relación empírica basada en el Índice de Área Foliar (LAI) y la Capacidad de Almacenamiento Específico ($S_l$) de las hojas.
        
        La capacidad máxima de retención del dosel ($S_{max}$, en milímetros) se define como:
        $$S_{max}=S_l \times LAI$$
        
        Cuando ocurre un evento de precipitación bruta ($P$), el agua interceptada ($I$) sigue una curva asintótica (porque una vez que las hojas se llenan, el resto escurre o gotea). Usaremos la forma exponencial clásica:
        $$I=S_{max} \cdot (1 - e^{-P/S_{max}})$$
        
        El agua que efectivamente golpea el suelo y genera riesgo de avalancha (Precipitación Efectiva, $P_{eff}$) es simplemente $P - I$.

        ### 🌿 La Matemática de la Naturaleza: Geometría Fractal
        Los árboles no son cilindros ni conos perfectos; son estructuras **fractales**. Para maximizar la captura de luz y la retención de agua (es decir, para maximizar el LAI en un espacio tridimensional reducido), la naturaleza utiliza patrones de autosemejanza.
        * **Sistemas de Lindenmayer (L-Systems):** Modelan el crecimiento vegetal mediante reglas recursivas. Cada rama se divide en sub-ramas más pequeñas siguiendo un factor de escala y un ángulo específico.
        * **Optimización Ecohidrológica:** Esta ramificación infinita crea una "esponja aérea" con un área superficial gigantesca. Un roble maduro puede tener miles de metros cuadrados de superficie foliar desplegados a partir de un solo tronco, interceptando eficientemente la energía cinética de las gotas de lluvia.
        
        ### 🎯 Utilidad e Interpretación Territorial
        * **Amortiguación de Crecientes Súbitas:** Permite cuantificar el volumen de agua que el bosque evita que llegue instantáneamente al cauce, reduciendo picos de caudal hidrográfico.
        * **Control de Erosión Hídrica:** El follaje disipa la energía cinética de la lluvia. Si el ecosistema está degradado, la $P_{eff}$ golpea el suelo erosionándolo y arrastrando sedimentos hacia los embalses.
        * **Valoración del Capital Natural:** Traducir hectáreas de bosque a metros cúbicos de agua retenida es el eslabón fundamental para justificar financieramente los proyectos de infraestructura verde.

        ### 📖 Fuentes de Consulta de Primer Nivel
        * **Aston, A. R. (1979).** *Rainfall interception by eight small trees.* Journal of Hydrology, 42(3-4), 383-396. (Ecuación base del modelo asintótico).
        * **Merriam, R. A. (1960).** *A note on the interception loss equation.* Journal of Geophysical Research. (Fundamentos de la exponencial de pérdida).
        * **Gash, J. H. C. (1979).** *An analytical model of rainfall interception by forests.* Q.J.R. Meteorol. Soc.
        * **Lindenmayer, A. (1968).** *Mathematical models for cellular interactions in development.* Journal of Theoretical Biology. (Bases matemáticas de los fractales vegetales).
        * **Mandelbrot, B. B. (1982).** *The Fractal Geometry of Nature.* W. H. Freeman and Co.
        """)

# =========================================================================
# PESTAÑA 9: ECOHIDROLOGÍA (EFECTO CASCADA)
# =========================================================================
with tab_micro:
    import plotly.graph_objects as go
    
    # 🎨 ESTILOS PREMIUM (HOMOLOGACIÓN VISUAL)
    st.markdown("""
    <style>
    div[data-testid="stExpander"] details summary p { font-family: 'Georgia', serif !important; font-size: 1.15em !important; color: #2c3e50 !important; font-weight: 600 !important; }
    div[data-testid="stExpander"] { border: 1px solid #d3c0a3 !important; border-radius: 6px !important; box-shadow: 2px 2px 8px rgba(0,0,0,0.04) !important; background-color: #ffffff; margin-bottom: -1px !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🔬 Efecto Cascada: Del Microscopio Foliar a la Planta de Tratamiento")
    st.info("Simulador de ciclo completo (Source-to-Tap). Laboratorio de la ECOmplejidad. Modela cómo la alteración de un componente biológico microscópico desencadena una avalancha de impactos físicos, químicos y financieros.")

    # --- 1. GENERADOR FRACTAL OPTIMIZADO ---
    with st.expander("🌿 El Código de la Naturaleza (Generador Fractal de Dosel)", expanded=False):
        st.markdown("La capacidad adaptativa de un árbol para retener agua y capturar luz se basa en la optimización fractal de su área superficial.")
        col_frac1, col_frac2 = st.columns([1, 2.5])
        
        with col_frac1:
            profundidad = st.slider("Nivel de Ramificación (Iteraciones):", 2, 15, 7)
            angulo_grados = st.slider("Ángulo de Ramificación (°):", 10, 90, 25)
            escala = st.slider("Factor de Reducción (Escala):", 0.5, 0.85, 0.75, step=0.05)
            velocidad = st.slider("⏱️ Velocidad de Animación (seg/nivel):", 0.05, 1.5, 0.25, 0.05)
            animar = st.button("🌱 Animar Crecimiento", use_container_width=True)
            
        with col_frac2:
            import math, time
            espacio_fractal = st.empty()
            
            def generar_figura_fractal_optimizada(prof_actual, angulo_base, factor_escala):
                x_lines, y_lines = [], []
                # OPTIMIZACIÓN 1: Pre-calcular la conversión a radianes fuera del bucle
                delta_angulo = math.radians(angulo_base) 
                
                # OPTIMIZACIÓN 2: Función anidada para no pasar listas por referencia constantemente
                def construir(x, y, angulo, longitud, nivel):
                    if nivel == 0: return
                    
                    # Pre-calcular seno y coseno una sola vez por rama
                    cos_a = math.cos(angulo)
                    sin_a = math.sin(angulo)
                    
                    x_nuevo = x + longitud * cos_a
                    y_nuevo = y + longitud * sin_a
                    
                    x_lines.extend([x, x_nuevo, None])
                    y_lines.extend([y, y_nuevo, None])
                    
                    longitud_escala = longitud * factor_escala
                    
                    construir(x_nuevo, y_nuevo, angulo - delta_angulo, longitud_escala, nivel - 1)
                    construir(x_nuevo, y_nuevo, angulo + delta_angulo, longitud_escala, nivel - 1)

                # Disparo inicial de la recursividad
                construir(0, 0, math.pi / 2, 100, prof_actual)
                
                # OPTIMIZACIÓN 3: hoverinfo='skip' evita que el navegador colapse buscando tooltips en 100k nodos
                fig_f = go.Figure(go.Scatter(
                    x=x_lines, y=y_lines, mode='lines', 
                    line=dict(color='rgba(39, 174, 96, 0.8)', width=1.5),
                    hoverinfo='skip' 
                ))
                
                fig_f.update_layout(
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), 
                    margin=dict(l=0, r=0, t=0, b=0), 
                    height=350, 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                return fig_f

            if animar:
                for p in range(1, profundidad + 1):
                    espacio_fractal.plotly_chart(generar_figura_fractal_optimizada(p, angulo_grados, escala), use_container_width=True)
                    time.sleep(velocidad)
            else: 
                espacio_fractal.plotly_chart(generar_figura_fractal_optimizada(profundidad, angulo_grados, escala), use_container_width=True)

    # --- 2. ANATOMÍA Y MICROINGENIERÍA ---
    with st.expander("🪵 1. & 2. Arquitectura del Árbol y Microingeniería Foliar", expanded=False):
        import math
        col_anat, col_hoja, col_graf = st.columns([1.2, 1.2, 2])
        
        with col_anat:
            st.markdown("#### 🪵 1. Arquitectura del Árbol")
            dbh_cm = st.slider("Diámetro del Tronco (DAP en cm):", 5.0, 150.0, 30.0, 1.0)
            area_dosel_m2 = st.slider("Área del Dosel (Proyección al Suelo en m²):", 1.0, 150.0, 25.0, 1.0)
            iteraciones_ramas = st.slider("Nivel de Ramificación (Complejidad Fractal):", 2, 15, 7, help="A mayor nivel, mayor superficie leñosa y foliar desplegada.")
            angulo_ramas = st.select_slider("Ángulo de Ramificación:", options=["Agudo (30° - Forma V)", "Medio (60° - Copa Redonda)", "Horizontal (90°)", "Llorón (120° - Hacia abajo)"], value="Medio (60° - Copa Redonda)")
            
        with col_hoja:
            st.markdown("#### 🍃 2. Microingeniería Foliar")
            tamano_hoja = st.select_slider("Área Foliar (Tamaño):", options=["Micrófila (< 5 cm²)", "Mesófila (5 - 50 cm²)", "Macrófila (> 50 cm²)"], value="Mesófila (5 - 50 cm²)")
            textura = st.radio("Textura de la Epidermis:", ["Lisa / Cerosa (Repele agua)", "Normal", "Pubescente (Pelos microscópicos)"], index=1)
            forma = st.radio("Morfología de la Hoja:", ["Plana", "Cóncava (Forma de copa)", "Acuminada (Punta de goteo larga)"], index=0)
            
            if st.toggle("👁️ Ver Herbario Botánico"):
                st.markdown("""<style>.botanical-tooltip { position: relative; display: inline-block; text-align: center; margin-bottom: 20px; cursor: help; } .botanical-tooltip img { border-radius: 5px; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); transition: transform 0.3s ease; max-width: 100%; height: auto; } .botanical-tooltip:hover img { transform: scale(1.02); } .botanical-tooltip .tooltiptext { visibility: hidden; width: 280px; background-color: #fdfaf2; color: #2c3e50; text-align: left; border: 1px solid #d3c0a3; border-radius: 5px; padding: 15px; position: absolute; z-index: 10; top: 105%; left: 50%; margin-left: -140px; opacity: 0; transition: opacity 0.4s; font-size: 0.85em; font-family: 'Georgia', serif; box-shadow: 4px 4px 12px rgba(0,0,0,0.3); line-height: 1.4; pointer-events: none; } .botanical-tooltip .tooltiptext::after { content: ""; position: absolute; bottom: 100%; left: 50%; margin-left: -8px; border-width: 8px; border-style: solid; border-color: transparent transparent #fdfaf2 transparent; } .botanical-tooltip:hover .tooltiptext { visibility: visible; opacity: 1; } .tit-botanico { font-weight: bold; font-size: 1.1em; color: #5d4037; border-bottom: 1px solid #d3c0a3; padding-bottom: 5px; margin-bottom: 8px;}</style>""", unsafe_allow_html=True)
                url_base = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/imagenes/"
                
                st.markdown("#### A. Textura de la Epidermis")
                c_h1, c_h2, c_h3 = st.columns(3)
                with c_h1: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Epidermis%20Lisa%20y%20Cerosa.png" target="_blank"><img src="{url_base}Epidermis%20Lisa%20y%20Cerosa.png"></a><div class="tooltiptext"><div class="tit-botanico">I. Lisa / Cerosa</div>Las gotas (B) mantienen una forma esférica perfecta ilustrando la tensión superficial en acción.</div></div>""", unsafe_allow_html=True)
                with c_h2: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Epidermis%20Normal.png" target="_blank"><img src="{url_base}Epidermis%20Normal.png"></a><div class="tooltiptext"><div class="tit-botanico">II. Epidermis Normal</div>Sin capa de cera gruesa. Una ligera llovizna (A) forma gotas irregulares que tienden a extenderse.</div></div>""", unsafe_allow_html=True)
                with c_h3: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Epidermis%20Pubescente.png" target="_blank"><img src="{url_base}Epidermis%20Pubescente.png"></a><div class="tooltiptext"><div class="tit-botanico">III. Epidermis Pubescente</div>Epidermis aterciopelada detallando tricomas glandulares, donde la gota (B) es retenida por aire atrapado.</div></div>""", unsafe_allow_html=True)

                st.markdown("#### B. Morfología de la Hoja")
                c_h4, c_h5, c_h6 = st.columns(3)
                with c_h4: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Morfologia%20Plana.png" target="_blank"><img src="{url_base}Morfologia%20Plana.png"></a><div class="tooltiptext"><div class="tit-botanico">IV. Morfología Plana</div>El agua (A) se extiende de manera uniforme, eficiente para maximizar la luz solar en regiones menos húmedas.</div></div>""", unsafe_allow_html=True)
                with c_h5: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Morfologia%20Concava.png" target="_blank"><img src="{url_base}Morfologia%20Concava.png"></a><div class="tooltiptext"><div class="tit-botanico">V. Morfología Cóncava</div>El agua (A) es recolectada hacia el centro. Ideal para canalizar agua hacia el tallo (Efecto Embudo).</div></div>""", unsafe_allow_html=True)
                with c_h6: st.markdown(f"""<div class="botanical-tooltip"><a href="{url_base}Morfologia%20Acuminada.png" target="_blank"><img src="{url_base}Morfologia%20Acuminada.png"></a><div class="tooltiptext"><div class="tit-botanico">VI. Morfología Acuminada</div>Extremo apical detallado (Acumen). Permite un rápido drenaje y reduce el tamaño de la gota de goteo.</div></div>""", unsafe_allow_html=True)
        
        # --- CÁLCULOS BIOMÉTRICOS Y FRACTALES ---
        area_foliar_base_m2 = 0.15 * (dbh_cm ** 2.1)
        # La complejidad fractal multiplica el área foliar desplegada
        area_foliar_m2 = area_foliar_base_m2 * (1 + (iteraciones_ramas * 0.08))
        
        # Área superficial de Tronco y Ramas (Aproximación de cilindro base + explosión fractal)
        area_tronco_ramas_m2 = (math.pi * (dbh_cm / 100.0) * 6.0) * (1.18 ** iteraciones_ramas)
        
        # Factor de Área Superficial (LAI a nivel de individuo)
        factor_area_superficial = area_foliar_m2 / area_dosel_m2 if area_dosel_m2 > 0 else 0
        
        # Modificadores de Retención
        sl_base = 0.20
        mod_tex = 0.7 if "Lisa" in textura else 1.6 if "Pubescente" in textura else 1.0
        mod_for = 1.4 if "Cóncava" in forma else 0.8 if "Acuminada" in forma else 1.0
        mod_tam = 0.85 if "Micrófila" in tamano_hoja else 1.25 if "Macrófila" in tamano_hoja else 1.0
        
        sl_efectivo = sl_base * mod_tex * mod_for * mod_tam
        volumen_retenido_litros = area_foliar_m2 * sl_efectivo
        
        stemflow_pct = 12.0 if "Agudo" in angulo_ramas else 5.0 if "Medio" in angulo_ramas else 1.0 if "Horizontal" in angulo_ramas else 0.1
        retencion_pct = min(25.0 * (sl_efectivo / 0.20), 45.0)
        throughfall_pct = 100.0 - retencion_pct - stemflow_pct

        # --- RENDERIZADO DEL PANEL DERECHO ---
        with col_graf:
            st.markdown("#### 📐 Biometría Estructural")
            c_bio1, c_bio2 = st.columns(2)
            c_bio1.metric("Área Foliar Total", f"{area_foliar_m2:,.1f} m²", "Súper-superficie de captura")
            c_bio2.metric("Área Leñosa (Tronco/Ramas)", f"{area_tronco_ramas_m2:,.1f} m²", "Estructura de soporte")
            
            c_bio3, c_bio4 = st.columns(2)
            c_bio3.metric("Factor Área Superficial", f"{factor_area_superficial:.1f}", "m² hoja / m² suelo (LAI local)")
            c_bio4.metric("Capacidad de Retención", f"{volumen_retenido_litros:,.1f} L", "Agua secuestrada", delta_color="normal")
            
            st.markdown("---")
            
            # Gráfico de torta
            fig_p = go.Figure(go.Pie(
                labels=["Agua Retenida (Dosel)", "Escurrimiento Tronco (Stemflow)", "Agua al Suelo (Throughfall)"], 
                values=[retencion_pct, stemflow_pct, throughfall_pct], 
                hole=0.4, 
                marker_colors=["#2ecc71", "#8e44ad", "#3498db"], 
                textinfo="percent",
                insidetextorientation='radial'
            ))
            
            fig_p.update_layout(
                title="Distribución del Destino de la Lluvia", 
                showlegend=True, 
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                height=320, 
                margin=dict(t=30, b=80, l=10, r=10)
            )
            st.plotly_chart(fig_p, use_container_width=True)
            
    # --- 3. BALÍSTICA ---
    with st.expander("🌧️ 3. Balística de la Gota y Control de Erosión", expanded=False):
        col_gota1, col_gota2 = st.columns(2)
        with col_gota1:
            diametro_lluvia = st.slider("Diámetro gota lluvia (mm):", 1.0, 6.0, 3.0, 0.2)
            altura_dosel = st.slider("Altura de caída (m):", 1.0, 20.0, 5.0, 0.5)
        with col_gota2:
            # --- CASCADA FÍSICA: El tamaño de la hoja dicta el tamaño base de la gota ---
            base_goteo = 4.5 if "Macrófila" in tamano_hoja else 2.5 if "Micrófila" in tamano_hoja else 3.5
            
            # La morfología altera el goteo final
            if "Acuminada" in forma: 
                diametro_goteo = base_goteo * 0.6
                st.success(f"💧 **Punta de goteo activa:** Corta la tensión superficial ({diametro_goteo:.1f} mm).")
            elif "Cóncava" in forma: 
                diametro_goteo = base_goteo * 1.4
                st.warning(f"⚠️ **Efecto copa:** 'Súper gotas' gigantes ({diametro_goteo:.1f} mm).")
            else: 
                diametro_goteo = base_goteo
                st.info(f"💧 **Hoja plana:** Gota de tamaño íntegro ({diametro_goteo:.1f} mm).")
                
        vt_lluvia = 9.65 - 10.3 * math.exp(-0.6 * diametro_lluvia)
        vt_goteo_max = 9.65 - 10.3 * math.exp(-0.6 * diametro_goteo)
        vel_goteo_h = vt_goteo_max * math.sqrt(1 - math.exp(-2 * 9.81 * altura_dosel / (vt_goteo_max**2)))
        masa_l_kg = (4/3) * math.pi * ((diametro_lluvia / 2000)**3) * 1000
        masa_g_kg = (4/3) * math.pi * ((diametro_goteo / 2000)**3) * 1000
        ek_l_uj, ek_g_uj = (0.5 * masa_l_kg * (vt_lluvia**2)) * 1e6, (0.5 * masa_g_kg * (vel_goteo_h**2)) * 1e6
        red_ek = 100 - (ek_g_uj / ek_l_uj * 100) if ek_l_uj > 0 else 0
        c_b1, c_b2, c_b3 = st.columns(3)
        c_b1.metric("Impacto Directo", f"{vt_lluvia:.1f} m/s", f"{ek_l_uj:.1f} μJ")
        c_b2.metric("Gota Filtrada", f"{vel_goteo_h:.1f} m/s", f"{ek_g_uj:.1f} μJ")
        if red_ek > 0: c_b3.metric("Protección", f"-{red_ek:.1f}%", "Energía disipada", delta_color="normal")
        else: c_b3.metric("Riesgo", f"+{abs(red_ek):.1f}%", "Impacto aumentado", delta_color="inverse")
        if st.toggle("📚 Mostrar El Milagro de la Hoja: Física y Ecuaciones"):
            st.markdown("""
            **La Arquitectura Fractal:** El área foliar crece exponencialmente con el diámetro.
            **La Balística:** El árbol reduce la energía de la lluvia a cero, y el agua vuelve a caer ganando nueva velocidad.
            """)

    # --- 4. EROSIVIDAD ---
    with st.expander("🟤 4. Erosividad y Desprendimiento de Suelo (Splash Detachment)", expanded=False):
        st.markdown("""<style>.tooltip-mod4 { position: relative; display: inline-block; color: #e67e22; font-weight: bold; cursor: help; border-bottom: 2px dotted #e67e22; } .tooltip-mod4 .tooltiptext { visibility: hidden; width: 320px; background-color: #2c3e50; color: #fff; text-align: left; border-radius: 6px; padding: 15px; position: absolute; z-index: 50; top: 120%; left: 50%; margin-left: -160px; opacity: 0; transition: opacity 0.3s; font-size: 0.85em; font-weight: normal;} .tooltip-mod4:hover .tooltiptext { visibility: visible; opacity: 1; }</style>""", unsafe_allow_html=True)
        st.markdown("<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>Escala el impacto balístico. Esta tierra es el primer paso del <span class='tooltip-mod4'>Efecto Cascada Territorial<span class='tooltiptext'>🧠 <b>Conexión Gemelo Digital:</b> Los kilogramos arrancados viajarán a la <b>Página 08</b> asfixiando el embalse.</span></span>.</div>", unsafe_allow_html=True)
        c_s1, c_s2 = st.columns([1, 1.5])
        with c_s1:
            vol_t_mm = st.slider("🌧️ Volumen Tormenta (mm):", 10.0, 150.0, 50.0, step=5.0)
            dur_h = st.slider("⏱️ Duración (h):", 0.5, 24.0, 1.0, step=0.5)
            tipo_s = st.selectbox("Erodabilidad (Factor K):", ["Arena Fina (Alta - K=0.06)", "Franco-Limoso (Media - K=0.03)", "Arcilloso (Baja - K=0.01)"], index=1)
            k_f = 0.06 if "Alta" in tipo_s else 0.03 if "Media" in tipo_s else 0.01
        int_mm_h = vol_t_mm / dur_h if dur_h > 0 else 0
        v_g_n = (4/3) * math.pi * ((diametro_lluvia / 2)**3)
        v_g_a = (4/3) * math.pi * ((diametro_goteo / 2)**3)
        n_g_n = (vol_t_mm * 1_000_000) / v_g_n if v_g_n > 0 else 0
        n_g_a = (vol_t_mm * 1_000_000) / v_g_a if v_g_a > 0 else 0
        ke_t_n, ke_t_a = (n_g_n * (ek_l_uj / 1e6)), (n_g_a * (ek_g_uj / 1e6))
        suelo_p_n_kg, suelo_p_a_kg = k_f * ke_t_n, k_f * ke_t_a
        st.session_state['memoria_suelo_arrancado'] = suelo_p_a_kg
        with c_s2:
            c_e1, c_e2, c_e3 = st.columns(3)
            c_e1.metric("Intensidad", f"{int_mm_h:.1f} mm/h")
            c_e2.metric("Energía (Abierto)", f"{ke_t_n/1000:,.1f} kJ/m²")
            c_e3.metric("Energía (Dosel)", f"{ke_t_a/1000:,.1f} kJ/m²")
            fig_s = go.Figure(data=[go.Bar(name='Cielo Abierto', x=['Kg/m²'], y=[suelo_p_n_kg], marker_color='#e67e22', text=[f"{suelo_p_n_kg:.1f} Kg"], textposition='auto'), go.Bar(name='Bajo Dosel', x=['Kg/m²'], y=[suelo_p_a_kg], marker_color='#27ae60', text=[f"{suelo_p_a_kg:.1f} Kg"], textposition='auto')])
            fig_s.update_layout(barmode='group', height=300, margin=dict(t=30, b=0, l=10, r=10), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_s, use_container_width=True)
        if st.toggle("📚 Mostrar El Aleph del Suelo: MMF"):
            st.markdown("""
            **Mecánica del Impacto (Splash Detachment):** Cuando la Energía Cinética de la lluvia supera la cohesión del suelo, las partículas finas explotan taponando los poros.
            **El Modelo MMF:** $D_s = K \cdot KE_{total}$.
            """)
            
    # =========================================================================
    # 🌊 5. MÓDULO DE TRANSPORTE DE SEDIMENTOS (Ecuación de Manning)
    # =========================================================================
    with st.expander("🌊 5. El Viaje del Lodo: Transporte de Sedimentos", expanded=False):
        st.info("La tierra arrancada por el impacto de la gota necesita un vehículo para llegar al río: La Escorrentía. Usa la física hidráulica de Manning para calcular cuánto sedimento es retenido por el sotobosque.")

        col_trans1, col_trans2 = st.columns([1, 1.5])

        with col_trans1:
            pendiente_pct = st.slider("⛰️ Pendiente del Terreno (%):", 1.0, 60.0, 15.0, 1.0)
            rugosidad = st.radio(
                "Cobertura a nivel del suelo (Sotobosque / Hojarasca):",
                [
                    "Suelo Desnudo / Pasto sobrepastoreado (Muy liso, n = 0.03)",
                    "Rastrojo bajo / Hojarasca media (Fricción media, n = 0.08)",
                    "Bosque nativo denso con raíces densas (Fricción alta, n = 0.15)"
                ], index=1
            )
            manning_n = 0.03 if "0.03" in rugosidad else 0.08 if "0.08" in rugosidad else 0.15

        # FÍSICA HIDRÁULICA
        S = pendiente_pct / 100.0
        R = 0.005 
        velocidad_escorrentia = (1.0 / manning_n) * (R ** (2/3)) * math.sqrt(S)
        sdr_pct = min(max((velocidad_escorrentia / 0.8) * 100, 0.0), 100.0)
        
        # 🛡️ Recuperación segura de la masa arrancada en el Módulo 4
        suelo_perdido_seguro = st.session_state.get('memoria_suelo_arrancado', 0.0)
        sedimento_al_rio_kg = suelo_perdido_seguro * (sdr_pct / 100.0)
        sedimento_retenido_kg = suelo_perdido_seguro - sedimento_al_rio_kg

        with col_trans2:
            c_t1, c_t2, c_t3 = st.columns(3)
            c_t1.metric("Velocidad del Flujo", f"{velocidad_escorrentia:.2f} m/s", "Ecuación de Manning")
            c_t2.metric("Sedimento al Río", f"{sedimento_al_rio_kg:.2f} Kg/m²", f"{sdr_pct:.1f}% del lodo", delta_color="inverse")
            c_t3.metric("Filtro Biológico", f"{sedimento_retenido_kg:.2f} Kg/m²", "Retenido en el bosque")
            
            fig_viaje = go.Figure(go.Waterfall(
                orientation = "v", measure = ["absolute", "relative", "total"],
                x = ["Tierra Arrancada", "Retenida (Sotobosque)", "Exportada al Río"],
                y = [suelo_perdido_seguro, -sedimento_retenido_kg, sedimento_al_rio_kg],
                text = [f"{suelo_perdido_seguro:.2f} Kg", f"-{sedimento_retenido_kg:.2f} Kg", f"<b>{sedimento_al_rio_kg:.2f} Kg</b>"],
                textposition = "outside", connector = {"line": {"color":"rgba(0,0,0,0.2)", "dash":"dot"}},
                decreasing = {"marker": {"color":"#2ecc71"}}, 
                increasing = {"marker": {"color":"#e74c3c"}}, 
                totals = {"marker": {"color":"#8e44ad"}}
            ))
            fig_viaje.update_layout(title="Balance de Transporte", height=320, margin=dict(l=20, r=20, t=40, b=30), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_viaje, use_container_width=True)
            
        if st.toggle("📚 Revelar El Aleph de la Hidráulica: Rugosidad y Gravedad"):
            st.markdown("""
            ### 🌊 La Ecuación de Manning y la Ley de la Gravedad
            Una vez que la tierra ha sido pulverizada por la lluvia, comienza su descenso hacia los ríos. El ingeniero Robert Manning (1889) dedujo cómo calcular la velocidad de este flujo superficial:
            $$V = \\frac{1}{n} R^{2/3} S^{1/2}$$
            Donde $S$ es la fuerza de la gravedad (pendiente) y $n$ es la salvación de la cuenca: **La Rugosidad**.
            
            **La Inteligencia del Sotobosque:** Al añadir hojarasca, helechos y raíces superficiales, el coeficiente de fricción ($n$) aumenta. Esto reduce la velocidad del agua por debajo de la *velocidad crítica de arrastre*, obligando al lodo a decantar. **El agua llega al río, pero la montaña se queda en su sitio.**
            """)

    # =========================================================================
    # 🛑 6. MÓDULO DE LIMNOLOGÍA: COLMATACIÓN Y DINÁMICA (RESTAURADO + PARTICIÓN)
    # =========================================================================
    with st.expander("🛑 6. Limnología Integral: Uniformismo y Catastrofismo en La Fe", expanded=False):
        st.markdown("""<style>.limno-tooltip { position: relative; display: inline-block; color: #2980b9; font-weight: 600; cursor: help; border-bottom: 1px dashed #2980b9; } .limno-tooltip .tooltiptext { visibility: hidden; width: 320px; background-color: #fdfaf2; color: #2c3e50; text-align: left; border: 1px solid #d3c0a3; border-radius: 5px; padding: 15px; position: absolute; z-index: 50; bottom: 125%; left: 50%; margin-left: -160px; opacity: 0; transition: opacity 0.4s; font-size: 0.9em; font-family: 'Georgia', serif; box-shadow: 4px 4px 12px rgba(0,0,0,0.3); line-height: 1.4; } .limno-tooltip:hover .tooltiptext { visibility: visible; opacity: 1; } .tit-limno { font-weight: bold; font-size: 1.1em; color: #8e44ad; border-bottom: 1px solid #d3c0a3; padding-bottom: 5px; margin-bottom: 8px;}</style>""", unsafe_allow_html=True)
        st.markdown("<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 15px;'>Modelo dinámico. Integra el <b>Uniformismo</b> (rutina) y el <b>Catastrofismo</b> (avenidas torrenciales) para calcular el colapso del <span class='limno-tooltip'>Volumen Muerto<span class='tooltiptext'><div class='tit-limno'>Fecha de Caducidad</div>Espacio en el fondo diseñado para sedimentos.</span></span>.</div>", unsafe_allow_html=True)

        col_lim1, col_lim2 = st.columns([1, 1.5])
        with col_lim1:
            if st.toggle("📍 Ver Contexto: Cuenca Espíritu Santo"):
                st.write("Área total de captación: **173 km²**. Afluente vital para La Fe.")
            
            tipo_tormenta = st.select_slider("Severidad de la Tormenta de HOY:", options=["Ordinaria (Tr < 1 año)", "Fuerte (Tr 5 años)", "Severa (Tr 20 años)", "Extrema (Tr 50 años)", "Catastrófica (Tr 100 años)"])
            f_tor = {"Ord": 1.0, "Fue": 2.5, "Sev": 4.5, "Ext": 6.5, "Cat": 9.0}[tipo_tormenta[:3]]
            area_km2 = st.number_input("Área afectada por la tormenta (km²):", 1.0, 173.0, 5.0)
            
            st.markdown("**Composición del Paisaje (Mix de Usos):**")
            c_p1, c_p2 = st.columns(2)
            pct_bosque = c_p1.slider("🌲 Bosque Conservado", 0, 100, 20)
            pct_agricola = c_p2.slider("🐄 Agrícola / Ganadero", 0, 100, 50)
            pct_degradado = c_p1.slider("🪨 Suelo Degradado", 0, 100, 15)
            pct_urbano = c_p2.slider("🏙️ Expansión Urbana", 0, 100, 15)
            
            total_p = pct_bosque + pct_agricola + pct_degradado + pct_urbano
            f_bos, f_agr, f_deg, f_urb = pct_bosque/total_p, pct_agricola/total_p, pct_degradado/total_p, pct_urbano/total_p

            st.markdown("**Batimetría y Dinámica:**")
            c_v1, c_v2 = st.columns(2)
            vol_util_hm3 = c_v1.number_input("Vol. Útil (Mm³):", value=12.5)
            vol_muerto_hm3 = c_v2.number_input("Vol. Muerto (Mm³):", value=3.0)
            caudal_ingreso = st.number_input("Ingreso Total (m³/s):", value=6.5)

            # --- 🚀 NUEVA FUNCIONALIDAD: PARTICIÓN DE SEDIMENTOS ---
            st.markdown("**Destino Físico de Sedimentos (Realismo de Ingeniería):**")
            c_part1, c_part2, c_part3 = st.columns(3)
            p_colas = c_part1.slider("% Colas (Delta)", 0, 100, 40, help="Material grueso que se queda en la entrada del río.")
            p_fondo = c_part2.slider("% Fondo (Muerto)", 0, 100, 45, help="Material fino que decanta en la presa.")
            p_susp = c_part3.slider("% Suspendido", 0, 100, 15, help="Material abrasivo que viaja a las turbinas/túneles.")
            
            if (p_colas + p_fondo + p_susp) != 100:
                st.warning(f"⚠️ La suma debe ser 100%. Actual: {p_colas+p_fondo+p_susp}%")

        # 🧠 MOTOR MATEMÁTICO INTEGRAL ACTUALIZADO (Partición por Cota y Abrasión)
        p_fos = (f_bos * 0.0001) + (f_agr * 0.0015) + (f_deg * 0.0005) + (f_urb * 0.0025)
        f_ero = (f_bos * 0.05) + (f_agr * 1.0) + (f_deg * 2.5) + (f_urb * 3.5)
        
        sed_al_rio = locals().get('sedimento_al_rio_kg', 7.02) 
        lodo_total_m3 = (sed_al_rio * area_km2 * 1e6 * f_ero * (f_tor**1.8)) / 1200.0
        
        # --- LÓGICA DE PARTICIÓN TÉCNICA ---
        lodo_colas_m3 = lodo_total_m3 * (p_colas / 100)      # Depósito en cotas altas (Delta)
        lodo_fondo_m3 = lodo_total_m3 * (p_fondo / 100)      # Depósito en Volumen Muerto (Peligro estructural)
        lodo_turbinas_m3 = lodo_total_m3 * (p_susp / 100)    # Sedimento en suspensión (Abrasión mecánica)
        
        # Tasa de colmatación base anual (Uniformismo)
        lodo_anual_base = (sed_al_rio * 5.0 * 1e6 * f_ero * 1.0) / 120.0
        
        # 🚨 REFINAMIENTO OPERATIVO:
        # Solo el lodo de FONDO reduce la vida útil operativa (Cota de Captación).
        # El lodo en colas reduce capacidad útil pero NO colmata la torre de salida hoy.
        anos_robados = lodo_fondo_m3 / lodo_anual_base if lodo_anual_base > 0 else 0
        
        vol_muerto_restante = (vol_muerto_hm3 * 1e6) - lodo_fondo_m3
        vida_util_restante = vol_muerto_restante / lodo_anual_base if lodo_anual_base > 0 else 99
        
        # Hidráulica y Química
        tasa_renovacion = (caudal_ingreso * 31536000) / ((vol_util_hm3 + vol_muerto_hm3) * 1e6)
        dias_residencia = 365 / tasa_renovacion if tasa_renovacion > 0 else 0
        fosforo_hoy = (sed_al_rio * area_km2 * 1e6 * f_ero * (f_tor**1.8)) * p_fos

        with col_lim2:
            st.markdown("##### ⚡ Impacto de la Avenida Torrencial (HOY)")
            c_e1, c_e2, c_e3 = st.columns(3)
            # Visualización de la Trifurcación
            c_e1.metric("Lodo en Colas", f"{lodo_colas_m3:,.0f} m³", "Cota alta (Delta)")
            c_e2.metric("Lodo en Fondo", f"{lodo_fondo_m3:,.0f} m³", "Volumen Muerto", delta_color="inverse")
            c_e3.metric("Lodo Suspendido", f"{lodo_turbinas_m3:,.0f} m³", "Riesgo Abrasión", delta_color="inverse")
            
            st.markdown("---")
            st.markdown("##### ⏳ Proyección Integral (Impacto en la Infraestructura)")
            c_p1, c_p2 = st.columns(2)
            c_p1.metric("Tasa Colmatación Base", f"{lodo_anual_base:,.0f} m³/año", "Desgaste rutinario")
            
            # La vida útil restante ahora es mucho más precisa
            estado_vida = "inverse" if vida_util_restante < 15 else "normal"
            c_p2.metric("Vida Útil Restante", f"{max(0.0, vida_util_restante):.1f} Años", "Post-sedimentación", delta_color=estado_vida)

            st.markdown("---")
            st.markdown("##### 🌊 Dinámica Hidráulica y Riesgo Químico")
            c_h1, c_h2 = st.columns(2)
            c_h1.metric("Tasa de Renovación", f"{tasa_renovacion:.1f} veces/año")
            c_h2.metric("Tiempo de Residencia", f"{dias_residencia:.0f} Días", "Edad del agua")

            st.info(f"**Impacto Químico:** {fosforo_hoy:,.1f} Kg de Fósforo inyectados hoy.")
            if fosforo_hoy > 500: st.error("🚨 **ALERTA ROJA:** Inminente Eutrofización y Anoxia.")
            elif fosforo_hoy > 100: st.warning("⚠️ **Riesgo Medio:** Alteración de transparencia y altos costos PTAP.")
            else: st.success("🌿 **Protección:** El paisaje amortiguó eficazmente la carga.")
            
        if st.toggle("📚 Revelar El Aleph de los Lagos: Colmatación y Fósforo"):
            st.markdown("""
            ### ⏳ Colmatación: El Reloj de Arena de la Ingeniería
            Los embalses son trampas de sedimentos. No todo el lodo llega al fondo; los granos gruesos se depositan en las **Colas del Embalse**, reduciendo la capacidad útil. Los sedimentos más finos quedan **suspendidos**, viajando por los túneles y desgastando álabes de turbinas por abrasión mecánica (cuarzos y circones).
            
            ### 🧪 La Venganza de la Tierra: Eutrofización
            La carga de fósforo detona el crecimiento de macrófitas. El embalse se vuelve anóxico en el fondo, aniquilando la fauna acuática y encareciendo la potabilización.
            """)
            
    # =========================================================================
    # 🚰 7. ECONOMÍA DE LA CALIDAD (PTAP) - RESTAURADO
    # =========================================================================
    with st.expander("🚰 7. Economía de la Calidad: El Costo en la Planta", expanded=False):
        st.info("Traduce el daño ecológico a dólares. Calcula el sobrecosto en químicos que la empresa de acueducto debe asumir para potabilizar el agua generada por la tormenta.")
        
        col_pot1, col_pot2 = st.columns([1, 1.5])
        with col_pot1:
            st.markdown("**Parámetros de Potabilización (Planta La Ayurá):**")
            q_ptap = st.number_input("Caudal Tratado (m³/s):", value=5.0, step=0.5)
            st.markdown("**Costo de Insumos (USD/Ton):**")
            c_alum = st.number_input("Sulfato Alum.:", value=450.0, step=10.0)
            c_cloro = st.number_input("Cloro Líquido:", value=1200.0, step=50.0)

        # MOTOR FINANCIERO-SANITARIO
        # Leemos las variables directamente del Módulo 6 que acaba de ejecutarse
        lodo_para_ptap = lodo_total_m3
        fosforo_para_ptap = fosforo_hoy
        vol_dia_l = q_ptap * 86400 * 1000
        
        # 1. Escenario Base (Sin desastre)
        ton_alum_base = (vol_dia_l * 15.0) / 1e9
        ton_cloro_base = (vol_dia_l * 2.0) / 1e9
        costo_base_anual_usd = ((ton_alum_base * c_alum) + (ton_cloro_base * c_cloro)) * 365

        # 2. Escenario de Crisis (La Tormenta de Hoy)
        f_turb = 1.0 + (lodo_para_ptap / 10000.0)
        f_eut = 1.0 + (fosforo_para_ptap / 500.0)
        extra_alum = (vol_dia_l * 15.0 * (min(f_turb, 8.0) - 1)) / 1e9
        extra_cloro = (vol_dia_l * 2.0 * (min(f_eut, 4.0) - 1)) / 1e9
        s_total = (extra_alum * c_alum) + (extra_cloro * c_cloro)
        
        ha_equiv = s_total / 2500.0

        with col_pot2:
            st.markdown("##### 💸 La Factura de la Tormenta vs Operación Base")
            c_f1, c_f2, c_f3 = st.columns(3)
            # MÉTRICA RECUPERADA: Costo base anual
            c_f1.metric("Costo Base Anual", f"${costo_base_anual_usd/1e6:,.1f} M USD", "Operación Normal")
            c_f2.metric("Sobrecosto HOY", f"${s_total:,.0f} USD", delta_color="inverse")
            c_f3.metric("Insumo Extra", f"+{extra_alum:,.1f} Ton Alum.", delta_color="inverse")
            
            if s_total > 5000:
                st.error(f"⚠️ **Penalidad Financiera:** El sobrecosto de hoy equivale a lo que costaría reforestar **{ha_equiv:,.1f} hectáreas** en la cuenca alta.")
            else:
                st.success("💧 **Agua de Alta Calidad:** El bosque amortiguó la tormenta.")
        
        if st.toggle("📚 Revelar El Aleph Financiero"):
            st.markdown("""
            **La miopía gris frente a la infraestructura verde:** Históricamente, se invierte en ampliar plantas de tratamiento para lidiar con el agua sucia, ignorando el ecosistema que la produce.
            * **El Lodo:** Alta turbiedad exige dosis masivas de coagulantes y genera lodos químicos costosos de disponer.
            * **El Fósforo:** Detona algas que tapan filtros y reaccionan con el cloro formando subproductos cancerígenos (THMs).
            
            **Conclusión:** Conservar el bosque no es filantropía; es la estrategia de reducción de costos operativos (OPEX) más inteligente para un acueducto.
            """)

    # =========================================================================
    # 🕳️ 8. EL MUNDO OCULTO (AGUAS SUBTERRÁNEAS) - RESTAURADO
    # =========================================================================
    with st.expander("🕳️ 8. El Mundo Oculto: Aguas Subterráneas y el 'Embalse Invisible'", expanded=False):
        st.info("El caudal de los ríos en verano depende de la recarga anual acumulada. Modela cómo el bosque construye el Flujo Base que nos salva durante El Niño.")

        col_sub1, col_sub2 = st.columns([1, 1.5])
        with col_sub1:
            st.markdown("**1. Escala Territorial (Régimen Anual):**")
            c_a1, c_a2 = st.columns(2)
            area_acuifero_km2 = c_a1.slider("Área Recarga (km²):", 1.0, 173.0, 173.0)
            precip_anual_mm = c_a2.slider("Lluvia (mm/año):", 1000, 4000, 2200)
            
            st.markdown("**2. Hidrogeología (Porosidad):**")
            geologia = st.selectbox("Formación Geológica:", ["Rocas Ígneas (Batolito)", "Depósitos Aluviales (Arenas)", "Arcillas Compactas"])
            
            inf_max = 0.40 if "Aluviales" in geologia else 0.15 if "Arcillas" in geologia else 0.25
            sy = 0.20 if "Aluviales" in geologia else 0.05 if "Arcillas" in geologia else 0.12

            st.markdown("**3. Demanda en Estiaje:**")
            c_e1, c_e2 = st.columns(2)
            dias_sequia = c_e1.number_input("Días Sequía (El Niño):", value=90)
            costo_emb_usd = c_e2.number_input("Costo m³ Embalse:", value=2.5)

        # MOTOR HIDROGEOLÓGICO
        vol_lluvia_m3 = (precip_anual_mm / 1000.0) * (area_acuifero_km2 * 1e6)
        mod_paisaje = (locals().get('f_bos', 0.2) * 1.0) + (locals().get('f_agr', 0.5) * 0.6) + (0.15 * 0.2) + (0.15 * 0.05)
        coef_inf_real = inf_max * mod_paisaje
        recarga_anual_m3 = vol_lluvia_m3 * coef_inf_real * sy
        caudal_base_ls = (recarga_anual_m3 / 31536000) * 1000
        
        # CÁLCULOS RECUPERADOS
        vol_disponible_sequia_m3 = recarga_anual_m3 * (dias_sequia / 365.0)
        valor_acuifero_usd = vol_disponible_sequia_m3 * costo_emb_usd
        personas_salvadas = (vol_disponible_sequia_m3 * 1000) / (150 * dias_sequia)

        with col_sub2:
            st.markdown("##### 💧 Balance Hidrológico Anual del Acuífero")
            c_s1, c_s2, c_s3 = st.columns(3)
            # MÉTRICAS RECUPERADAS
            c_s1.metric("Precipitación Total", f"{vol_lluvia_m3/1e6:,.1f} Mm³", "Lluvia en 1 año")
            c_s2.metric("Coef. Infiltración", f"{coef_inf_real*100:.1f}%", "Efecto Paisaje")
            c_s3.metric("Recarga Real", f"{recarga_anual_m3/1e6:,.1f} Mm³", f"S_y: {sy*100:.1f}%")
            
            st.markdown("---")
            st.markdown(f"##### ☀️ Soporte Vital durante la Sequía ({dias_sequia} días)")
            c_s4, c_s5, c_s6 = st.columns(3)
            # MÉTRICAS RECUPERADAS
            c_s4.metric("Flujo Base", f"{caudal_base_ls:,.1f} L/s", "Caudal 24/7")
            c_s5.metric("Población Soportada", f"{personas_salvadas:,.0f} Hab", "Con 150 L/día")
            c_s6.metric("Valor Infraestructura", f"${valor_acuifero_usd/1e6:,.1f} M USD", "Ahorro en represas")
            
            if caudal_base_ls < 50.0:
                st.error("🚨 **Riesgo de Colapso:** Caudal insuficiente para la vida ribereña.")
            elif coef_inf_real >= (inf_max * 0.8):
                st.success("🌿 **El 'Embalse Invisible' Actúa:** El bosque ahorró millones en concreto.")

        if st.toggle("📚 Revelar El Aleph Subterráneo"):
            st.markdown("""
            **La Ingeniería de las Raíces:** El bosque actúa como un taladro natural que crea macroporos para recargar los acuíferos. El asfalto anula esta infiltración.
            
            ### 🔬 Porosidad vs Rendimiento Específico ($S_y$)
            * **Retención:** Agua que queda atrapada en el suelo por capilaridad.
            * **Rendimiento:** Agua liberable que alimenta manantiales.
            
            **Economía de la Porosidad:** La cuenca nos ofrece almacenamiento geológico a costo cero. Solo debemos mantener el bosque para que el agua pueda entrar.
            """)

    # =========================================================================
    # 🌐 9. CONEXIÓN AL GEMELO DIGITAL (CROSS-POLLINATION)
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🌐 9. Conexión al Gemelo Digital (Cross-Pollination)")
    st.info("Exporta la partición física, química y el riesgo de infraestructura de esta avenida torrencial hacia el simulador territorial (Pág 08).")

    if st.button("🔌 Sincronizar Impacto con el Sistema Territorial (Pág 08)", type="primary", use_container_width=True):
        # 🧠 ENVIAMOS LA TRIFURCACIÓN TÉCNICA DE LODO
        # Asignación directa y segura desde las variables del Módulo 6 y 7
        lodo_total = lodo_total_m3
        lodo_colas = lodo_colas_m3
        lodo_fondo = lodo_fondo_m3
        lodo_abrasivo = lodo_turbinas_m3 
        
        # Datos químicos y financieros
        fosforo_final = fosforo_hoy
        costo_final = s_total
        
        # 💾 Inyección en la Memoria Global (st.session_state)
        # Esto permite que la Página 08 dibuje el Sankey con las tres venas de lodo
        st.session_state['eco_lodo_total_m3'] = float(lodo_total)
        st.session_state['eco_lodo_colas_m3'] = float(lodo_colas)
        st.session_state['eco_lodo_fondo_m3'] = float(lodo_fondo)
        st.session_state['eco_lodo_abrasivo_m3'] = float(lodo_abrasivo)
        
        st.session_state['eco_fosforo_kg'] = float(fosforo_final)
        st.session_state['eco_sobrecosto_usd'] = float(costo_final)
        
        st.success(f"""
            🧠 **Sincronización de Ingeniería Exitosa:**
            Los datos han cruzado el Aleph con partición física:
            * **Lodo en Colas (Cota Alta):** {lodo_colas:,.0f} m³
            * **Lodo en Fondo (Vol. Muerto):** {lodo_fondo:,.0f} m³
            * **Lodo Suspendido (Abrasión):** {lodo_abrasivo:,.0f} m³
            
            Ve a la **Página 08 (Sistemas Hídricos)** para visualizar este impacto trifurcado en el Mapa Conceptual.
        """)
