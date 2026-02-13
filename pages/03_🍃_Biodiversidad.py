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
import plotly.graph_objects as go
import plotly.express as px
from modules.admin_utils import init_supabase

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

# --- FUNCIÓN DETECTIVE (VERSIÓN DIAGNÓSTICO) ---

# --- REEMPLAZAR LA FUNCIÓN get_raster_from_cloud POR ESTA VERSIÓN ---

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
tab_mapa, tab_tax, tab_carbon = st.tabs(["🗺️ Mapa & GBIF", "📊 Taxonomía", "🌳 Calculadora Carbono"])

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
            fig.add_trace(go.Scattermapbox(
                lon=gdf_bio['lon'], lat=gdf_bio['lat'], 
                mode='markers', marker=dict(size=7, color='rgb(0, 200, 100)'), 
                text=gdf_bio['Nombre Común'], name='Biodiversidad'
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
with tab_tax:
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
                st.dataframe(threatened[['Nombre Científico', 'Nombre Común', 'Amenaza IUCN']].drop_duplicates(), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No se detectaron especies en categorías críticas (CR, EN, VU) en esta zona.")
        
        st.markdown("##### Detalle de Registros")
        st.dataframe(gdf_bio.drop(columns='geometry', errors='ignore'), use_container_width=True)
    else:
        st.info("No hay datos de biodiversidad para mostrar estadísticas.")

# ==============================================================================
# TAB 3: CALCULADORA DE CARBONO (INTEGRADA & DOCUMENTADA)
# ==============================================================================
with tab_carbon:
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
        # IDs de Pastos/Degradados
        target_ids = [7, 3, 11] 
        df_potencial = df_diagnostico[df_diagnostico['COV_ID'].isin(target_ids)].copy()
        total_potencial = df_potencial['Hectareas'].sum()
        
        k1, k2 = st.columns(2)
        k1.metric("Área Total Zona", f"{(gdf_zona.to_crs('+proj=cea').area.sum()/10000):,.1f} ha")
        k2.metric("Potencial Restauración", f"{total_potencial:,.1f} ha", help="Pastos + Áreas Degradadas disponibles")
    else:
        total_potencial = 0

    st.divider()

    # --- 3. CONFIGURACIÓN DEL PROYECTO ---
    st.subheader("⚙️ Configuración del Proyecto")
    
    col_conf1, col_conf2 = st.columns([1, 2])
    
    with col_conf1:
        # A. ESTRATEGIA (Aquí integramos Densidad/Sucesión)
        estrategia = st.selectbox(
            "Estrategia de Intervención:",
            options=["ACTIVA_ALTA", "ACTIVA_MEDIA", "PASIVA"],
            format_func=lambda x: carbon_calculator.ESCENARIOS_CRECIMIENTO[x]["nombre"],
            help="Define la densidad de siembra y velocidad de crecimiento."
        )
        
        # Mostrar descripción dinámica
        desc = carbon_calculator.ESCENARIOS_CRECIMIENTO[estrategia]["desc"]
        st.caption(f"ℹ️ *{desc}*")
        
        # B. ÁREA
        tipo_area = st.radio("Definir Área:", ["Manual", "Todo el Potencial"], horizontal=True)
        if tipo_area == "Manual":
            area_input = st.number_input("Hectáreas:", min_value=0.1, value=1.0)
        else:
            val_def = float(total_potencial) if total_potencial > 0 else 1.0
            area_input = st.number_input("Hectáreas:", value=val_def, disabled=True)
            
        # C. EDAD (Horizonte)
        edad_proy = st.slider("Edad / Horizonte (Años):", 5, 50, 20, help="Tiempo de proyección del análisis.")
        
        calc_btn = st.button("🚀 Calcular Carbono", type="primary")

    with col_conf2:
        # Lógica de Persistencia: Si se pulsa el botón, calculamos y guardamos.
        # Si no se pulsa, pero ya existe un cálculo previo, lo mostramos.
        if calc_btn:
            # 1. Calcular
            df_res = carbon_calculator.calcular_proyeccion_captura(
                hectareas=area_input, 
                anios=edad_proy, 
                escenario_key=estrategia
            )
            # 2. Guardar en Memoria de Sesión (ESTO FALTABA)
            st.session_state['carbon_results'] = {
                'df': df_res,
                'total': df_res['Proyecto_tCO2e_Acumulado'].iloc[-1],
                'estrategia': estrategia,
                'area': area_input
            }

        # Revisamos si hay resultados en memoria para mostrar
        if 'carbon_results' in st.session_state:
            res = st.session_state['carbon_results']
            df_res = res['df']
            total_c = res['total']
            
            # (Opcional) Advertencia si cambiaste parámetros pero no recalculaste
            # if res['area'] != area_input or res['estrategia'] != estrategia:
            #     st.caption("⚠️ *Los resultados mostrados corresponden al cálculo anterior. Pulsa 'Calcular' para actualizar.*")

            # Resultados Clave
            tasa_prom = total_c / edad_proy
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Captura Total", f"{total_c:,.0f} tCO2e")
            m2.metric("Tasa Anual", f"{tasa_prom:,.1f} t/año")
            m3.metric("Valor (5 USD/t)", f"${(total_c*5):,.0f} USD")
            
            # Gráfico de Curva S (Sigmoide)
            fig = px.area(df_res, x='Año', y='Proyecto_tCO2e_Acumulado',
                          title=f"Dinámica de Carbono - {carbon_calculator.ESCENARIOS_CRECIMIENTO[res['estrategia']]['nombre']}",
                          labels={'Proyecto_tCO2e_Acumulado': 'Acumulado (tCO2e)'},
                          color_discrete_sequence=['#2ecc71'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver Tabla Detallada"):
                st.dataframe(df_res)
        
        elif not calc_btn:
            st.info("👈 Configura los parámetros y pulsa 'Calcular Carbono' para ver la proyección.")
