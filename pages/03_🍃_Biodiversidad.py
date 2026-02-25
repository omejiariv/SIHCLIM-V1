# pages/03_🍃_Biodiversidad.py

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
st.title("🍃 Biodiversidad y Servicios Ecosistémicos Integrados")

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

# --- DEFINICIÓN DE TABS PRINCIPALES ---
tab_mapa, tab_taxonomia, tab_forestal, tab_afolu, tab_comparador = st.tabs([
    "🗺️ Mapa & GBIF", "🧬 Taxonomía", "🌲 Bosque e Inventarios", "⚖️ Metabolismo Territorial", "📈 Comparador"
])

# Variable global para datos de biodiversidad
gdf_bio = pd.DataFrame()
threatened = pd.DataFrame()
n_threat = 0

# --- PROCESAMIENTO PREVIO (Solo si hay zona) ---
if gdf_zona is not None:
    with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)
        
    if not gdf_bio.empty and 'Amenaza IUCN' in gdf_bio.columns:
        threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
        n_threat = threatened['Nombre Científico'].nunique()

# ==============================================================================
# TAB 1: MAPA Y MÉTRICAS
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
# TAB 2: TAXONOMÍA
# ==============================================================================
with tab_taxonomia:
    if not gdf_bio.empty:
        c1, c2 = st.columns([2,1])
        
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
                    
                st.dataframe(threatened[cols_mostrar].drop_duplicates(), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No se detectaron especies en categorías críticas (CR, EN, VU) en esta zona.")
        
        st.markdown("---")
        st.markdown("##### Detalle de Registros")
        st.dataframe(gdf_bio.drop(columns='geometry', errors='ignore'), use_container_width=True)
        
    else:
        st.info("No hay datos de biodiversidad para mostrar estadísticas.")

# ==============================================================================
# TAB 3: CALCULADORA DE CARBONO (INTEGRADA & DOCUMENTADA)
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
        # 🌐 CONEXIÓN AL ALEPH (ST.SESSION_STATE) - AHORA VISIBLE Y BLINDADO
        # =====================================================================
        try:
            area_agricola_total = 0.0
            area_urbana_total = 0.0
            area_pastos_total = 0.0
            
            for index, row in df_diagnostico.iterrows():
                cov_id = int(row['COV_ID'])
                hectareas = float(row['Hectareas'])
                
                if cov_id in [5, 6, 8]: area_agricola_total += hectareas
                elif cov_id in [7, 10]: area_pastos_total += hectareas
                elif cov_id in [1, 2, 3, 4]: area_urbana_total += hectareas
            
            st.session_state['aleph_ha_agricola'] = area_agricola_total
            st.session_state['aleph_ha_pastos'] = area_pastos_total
            st.session_state['aleph_area_urbana'] = area_urbana_total
            st.session_state['aleph_territorio_origen'] = str(nombre_seleccion)
            
            st.success(f"🌐 **El Aleph activo:** Las áreas de {nombre_seleccion} ({area_agricola_total:,.1f} Ha Agrícolas y {area_pastos_total:,.1f} Ha en Pastos) han sido enviadas al simulador de Calidad de Agua.")
            
        except Exception as e:
            st.error(f"Error de conexión en Aleph: {e}")
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
# TAB 4: METABOLISMO TERRITORIAL (AFOLU COMPLETO)
# ==============================================================================
with tab_afolu:
    
    # -------------------------------------------------------------------------
    # 🌐 1. CAPTURA ABSOLUTA DE VARIABLES Y CEREBRO ESPACIAL
    # -------------------------------------------------------------------------
    from modules.data_processor import cargar_censo_ica, normalizar_texto, cargar_territorio_maestro
    
    # 1.1 El "Cazador" de variables locales (Evita el 'Desconocido')
    try: 
        nivel_sel = nivel_seleccion # Intenta leer tu variable local del sidebar
    except NameError: 
        nivel_sel = st.session_state.get('nivel_seleccion', st.session_state.get('nivel_agregacion', 'Por Municipio'))
        
    try: 
        nombre_sel = nombre_seleccion # Intenta leer tu variable local del sidebar
    except NameError: 
        nombre_sel = st.session_state.get('nombre_seleccion', st.session_state.get('territorio_seleccionado', 'Desconocido'))

    nombre_sel_norm = normalizar_texto(nombre_sel)
    termino_busqueda = nombre_sel_norm.replace("region", "").replace("provincia", "").replace("car", "").strip()
    
    # 1.2 Conexión a la nueva base maestra
    df_territorio = cargar_territorio_maestro()
    
    # Blindaje: Si el procesador no creó las columnas norm, las creamos aquí en vivo
    if not df_territorio.empty:
        if 'region_norm' not in df_territorio.columns and 'region' in df_territorio.columns:
            df_territorio['region_norm'] = df_territorio['region'].astype(str).apply(normalizar_texto)
        if 'municipio_norm' not in df_territorio.columns and 'municipio' in df_territorio.columns:
            df_territorio['municipio_norm'] = df_territorio['municipio'].astype(str).apply(normalizar_texto)
            
    mpios_activos = []
    es_departamento = False
    
    # 1.3 Motor de Agregación Flexible
    if "departamento" in nivel_sel.lower() or "antioquia" in termino_busqueda:
        es_departamento = True
    elif not df_territorio.empty and "regi" in nivel_sel.lower():
        # Busca el término (ej. 'oriente') dentro de la columna region_norm de tu nuevo excel
        mask = df_territorio['region_norm'].astype(str).str.contains(termino_busqueda, na=False)
        mpios_activos = df_territorio[mask]['municipio_norm'].tolist()
    elif not df_territorio.empty and ("car" in nivel_sel.lower() or "jurisdicci" in nivel_sel.lower()):
        if 'car' in df_territorio.columns:
            mask = df_territorio['car'].astype(str).apply(normalizar_texto).str.contains(termino_busqueda, na=False)
            mpios_activos = df_territorio[mask]['municipio_norm'].tolist()
        
    # Fallback si no encontró nada o la base falló
    if not mpios_activos and not es_departamento:
        if "aburra" in termino_busqueda:
            mpios_activos = ["medellin", "bello", "itagui", "envigado", "sabaneta", "copacabana", "la estrella", "girardota", "caldas", "barbosa"]
        else:
            mpios_activos = [nombre_sel_norm]

    # --- TÍTULO DINÁMICO ---
    if "municipio" in nivel_sel.lower(): prefijo = "el Municipio de"
    elif "regi" in nivel_sel.lower(): prefijo = "la Región"
    elif "departamento" in nivel_sel.lower(): prefijo = "el Departamento de"
    elif "cuenca" in nivel_sel.lower(): prefijo = "la Cuenca"
    elif "car" in nivel_sel.lower() or "jurisdicci" in nivel_sel.lower(): prefijo = "la Jurisdicción de"
    else: prefijo = "el territorio de"
    
    titulo_dinamico = f"Metabolismo Territorial: Dinámica de GEI en {nombre_sel}"

    st.header(f"⚖️ {titulo_dinamico}")
    
    # Un pequeño indicador visual para que tú sepas si está leyendo la base bien
    if es_departamento: st.success(f"📍 Mapeando Departamento Completo")
    else: st.info(f"📍 Agrupando {len(mpios_activos)} municipio(s): {', '.join(mpios_activos[:5])}{'...' if len(mpios_activos)>5 else ''}")

    # -------------------------------------------------------------------------
    # 🌐 2. EXTRACCIÓN Y SUMA DE DATOS (BOSQUE, ICA, DANE)
    # -------------------------------------------------------------------------
    area_bosque_real = 100.0
    try:
        if 'df_diagnostico' in locals() and df_diagnostico is not None:
            area_bosque_real = df_diagnostico[df_diagnostico['COV_ID'] == 9]['Hectareas'].sum()
    except: pass

    df_bov = cargar_censo_ica('bovino')
    df_porc = cargar_censo_ica('porcino')
    df_aves = cargar_censo_ica('aviar')
    
    # Motor de sumatoria masiva
    def filtrar_y_sumar(df, columna_mpio, columna_valor):
        if df.empty or columna_valor not in df.columns: return 0
        if es_departamento:
            if 'DEPARTAMENTO' in df.columns:
                return df[df['DEPARTAMENTO'].astype(str).str.lower().str.contains('antioquia')][columna_valor].sum()
            return df[columna_valor].sum()
        return df[df[columna_mpio].isin(mpios_activos)][columna_valor].sum()

    bovinos_reales = int(filtrar_y_sumar(df_bov, 'MUNICIPIO_NORM', 'TOTALBOVINOS'))
    porcinos_reales = int(filtrar_y_sumar(df_porc, 'MUNICIPIO_NORM', 'TOTAL_CERDOS'))
    col_aves = 'TOTAL_AVES_CAPACIDAD_OCUPADA_MAS_AVES_TRASPATIO' if not df_aves.empty and 'TOTAL_AVES_CAPACIDAD_OCUPADA_MAS_AVES_TRASPATIO' in df_aves.columns else 'TOTAL_AVES_CAPACIDAD_OCUPADA'
    aves_reales = int(filtrar_y_sumar(df_aves, 'MUNICIPIO_NORM', col_aves))

    poblacion_rural_calculada, poblacion_urbana_calculada = 0, 0
    try:
        df_mpios = pd.read_csv("data/Pob_mpios_colombia.csv", sep=';', low_memory=False)
        if es_departamento: df_m_filt = df_mpios[df_mpios['depto_nom'].astype(str).str.lower().str.contains('antioquia')]
        else: df_m_filt = df_mpios[df_mpios['municipio'].astype(str).apply(normalizar_texto).isin(mpios_activos)]
            
        if not df_m_filt.empty:
            anio_max = df_m_filt['año'].max()
            df_m_filt = df_m_filt[df_m_filt['año'] == anio_max]
            areas_str = df_m_filt['area_geografica'].astype(str).str.lower()
            poblacion_rural_calculada = int(df_m_filt[areas_str.str.contains('rural|resto|centro', na=False)]['Poblacion'].sum())
            poblacion_urbana_calculada = int(df_m_filt[areas_str.str.contains('urbano|cabecera', na=False)]['Poblacion'].sum())
    except: pass
    
    if poblacion_rural_calculada == 0: poblacion_rural_calculada = 50 
    if poblacion_urbana_calculada == 0: poblacion_urbana_calculada = 1000 
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
            st.info(f"📍 **Conexión Aleph:** Datos extraídos para **{nombre_seleccion}**.")
            opciones_fuentes = ["Todas", "Pasturas", "Bovinos", "Porcinos", "Avicultura", "Población Rural"]
            fuentes_sel = st.multiselect("Selecciona cargas rurales a modelar:", opciones_fuentes, default=["Todas"])
            fuentes_activas = ["Pasturas", "Bovinos", "Porcinos", "Avicultura", "Población Rural"] if "Todas" in fuentes_sel else fuentes_sel

            esc_pasto, area_pastos = "PASTO_DEGRADADO", 0.0
            v_leche, v_carne, cerdos, aves, humanos_rurales = 0, 0, 0, 0, 0
            
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                if "Pasturas" in fuentes_activas:
                    esc_pasto = st.selectbox("Manejo de Pastos:", list(carbon_calculator.ESCENARIOS_PASTURAS.keys()), format_func=lambda x: carbon_calculator.ESCENARIOS_PASTURAS[x]["nombre"])
                    area_pastos = st.number_input("Ha de Pasturas (Satélite):", value=aleph_pastos, step=5.0)
                if "Bovinos" in fuentes_activas:
                    v_leche = st.number_input("Vacas Lecheras (ICA):", value=int(bovinos_reales * 0.4), step=10)
            with c_r2:
                if "Bovinos" in fuentes_activas:
                    v_carne = st.number_input("Ganado Carne/Cría (ICA):", value=int(bovinos_reales * 0.6), step=10)
                if "Porcinos" in fuentes_activas:
                    cerdos = st.number_input("Cerdos Cabezas (ICA):", value=porcinos_reales, step=50)
            with c_r3:
                if "Avicultura" in fuentes_activas:
                    aves = st.number_input("Aves Galpones (ICA):", value=aves_reales, step=500)
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
# TAB 4: COMPARADOR DE ESCENARIOS (NUEVO)
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



























