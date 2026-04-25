# pages/01_🌦️_Clima_e_Hidrologia.py

import io
import os
import sys

# --- PARCHE UNIVERSAL PARA WINDOWS/PROJ (GDAL HELL) ---
# Esto arregla Rasterio tanto en el script principal como en los módulos importados
if os.name == 'nt': # Solo en Windows
    try:
        import pyproj
        # Forzamos a que todo el sistema use el diccionario de coordenadas de Python
        os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    except: pass
# ------------------------------------------------------

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from rasterio.io import MemoryFile

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS (SOLUCIÓN AL NAMEERROR) ---
try:
    from modules import selectors
    from modules.admin_utils import init_supabase
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.admin_utils import init_supabase
    
# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌦️", layout="wide")
warnings.filterwarnings("ignore")

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Clima e Hidrología")

# ==============================================================================
# 📡 1. CONEXIÓN SATELITAL (AUTO-FETCH ENSO DESDE COLUMBIA UNIVERSITY)
# ==============================================================================
if 'enso_fase' not in st.session_state:
    try:
        from modules.forecasting import get_iri_enso_forecast
        df_enso, _ = get_iri_enso_forecast()
        
        # Tomamos el pronóstico del trimestre actual (primera fila)
        if df_enso is not None and not df_enso.empty:
            prob_nina = df_enso.iloc[0]['Niña']
            prob_nino = df_enso.iloc[0]['Niño']
            
            if prob_nina > 50: estado_actual = "Niña 🌧️"
            elif prob_nino > 50: estado_actual = "Niño ☀️"
            else: estado_actual = "Neutro ⚖️"
            
            st.session_state['enso_fase'] = estado_actual
            st.toast(f"📡 Clima Global Sincronizado: Condición actual {estado_actual}", icon="✅")
    except Exception as e:
        st.session_state['enso_fase'] = "Neutro ⚖️"
        st.toast("⚠️ No se pudo conectar con el servidor IRI. Usando clima Neutro.", icon="🔌")

# --- 2. IMPORTACIONES ROBUSTAS ---
try:
    # Módulos Base (Tus archivos subidos)
    from modules.config import Config
    from modules.db_manager import get_engine
    from modules.data_processor import complete_series, load_and_process_all_data
    from modules.reporter import generate_pdf_report
    
    # Módulos Críticos de Visualización y Selección
    from modules import selectors
    from modules import visualizer as viz
    
    # Módulos de Física y Utilidades (Manejo de errores si faltan dependencias)
    try:
        from modules import hydro_physics as physics
        from modules.admin_utils import download_raster_to_temp
        PHYSICS_AVAILABLE = True
    except ImportError as e:
        PHYSICS_AVAILABLE = False
        st.toast(f"⚠️ Módulos físicos limitados: {e}", icon="⚠️")

    # Análisis
    try:
        from modules.analysis import calculate_trends_mann_kendall
    except ImportError:
        calculate_trends_mann_kendall = None

except Exception as e:
    st.error(f"❌ Error Crítico de Importación: {e}")
    st.stop()

# --- FUNCIÓN MAESTRA DE CARGA (VERSIÓN STORAGE / BUCKET) ---
# Esta versión conecta con la pestaña "Coberturas" del Panel de Admin
@st.cache_resource(show_spinner=False)
def cargar_raster_db(filename):
    """
    Descarga un archivo raster (TIF) desde el Storage (Bucket 'rasters') de Supabase
    y lo devuelve como un objeto en memoria (BytesIO).
    """
    try:
        # 1. Iniciamos cliente usando tus credenciales existentes
        client = init_supabase()
        
        # 2. Descargamos los bytes del bucket 'rasters'
        # (El Panel de Admin sube los archivos a este bucket por defecto)
        file_bytes = client.storage.from_("rasters").download(filename)
        
        # 3. Devolvemos el objeto en memoria
        return io.BytesIO(file_bytes)
        
    except Exception as e:
        # Si falla (ej. archivo no existe), retornamos None
        # st.error(f"Error descargando '{filename}' del Storage: {e}") 
        return None
# -----------------------------------------------------------

# --- 3. CARGA DE DATOS UNIFICADA (Con Caché) ---
@st.cache_resource(show_spinner="📡 Consultando Sistema de Información...", ttl=3600)
def load_all_data_cached():
    """Wrapper con caché para la carga pesada de data_processor"""
    return load_and_process_all_data()

# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================
def main():
    
    # --- A. SELECTOR ESPACIAL (Módulo selectors.py) ---
    try:
        # Llama a tu selector espacial existente
        ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
    except Exception as e:
        st.sidebar.error(f"Error en Selector: {e}")
        st.stop()

    # Validación de Selección
    if not ids_estaciones:
        st.info("👈 Seleccione una Cuenca o Municipio en el menú lateral para comenzar.")
        # Opcional: Mostrar mapa general o mensaje de bienvenida aquí
        st.stop()

    # --- B. CARGA DE DATOS ---
    try:
        (gdf_stations, gdf_municipios, df_all_rain, df_enso, gdf_subcuencas, gdf_predios) = load_all_data_cached()
    except Exception as e:
        st.error(f"Error cargando datos base: {e}")
        st.stop()

    # Filtro de datos según selección del usuario
    if df_all_rain is not None and not df_all_rain.empty and ids_estaciones:
        # Asegurar tipos string para cruce exacto
        df_all_rain['id_estacion'] = df_all_rain['id_estacion'].astype(str).str.strip()
        ids_estaciones = [str(x).strip() for x in ids_estaciones]
        
        # Filtro Principal
        df_long = df_all_rain[df_all_rain['id_estacion'].isin(ids_estaciones)].copy()
        
        # Filtrar GeoDataFrame de estaciones
        if gdf_stations is not None:
            gdf_stations['id_estacion'] = gdf_stations['id_estacion'].astype(str).str.strip()
            gdf_filtered = gdf_stations[gdf_stations['id_estacion'].isin(ids_estaciones)]
        else:
            gdf_filtered = gpd.GeoDataFrame()
    else:
        st.warning("No hay datos de lluvia disponibles en la base de datos para esta selección.")
        st.stop()

    if df_long.empty:
        st.warning(f"La zona '{nombre_zona}' no tiene registros históricos de precipitación.")
        st.stop()

    stations_for_analysis = df_long[Config.STATION_NAME_COL].unique().tolist()

    # =========================================================================
    # --- C. JERARQUÍA DE NAVEGACIÓN (Centro de Comando Principal) ---
    # =========================================================================
    
    # 🛡️ PARCHE CSS: Evita que el módulo 'Inicio' elimine el margen superior
    st.markdown("""
    <style>
    .block-container { padding-top: 3rem !important; }
    /* Estilo premium para botones */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #2c3e50, #3498db); border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1a252f, #2980b9); transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title(f"🌦️ Análisis Hidroclimático: {nombre_zona}")
    st.markdown("### 🎛️ Centro de Comando y Módulos de Análisis")
    
    # 1. Agrupamos los módulos en Categorías para no abrumar al usuario
    col_nav1, col_nav2, col_nav3 = st.columns([1.5, 1.5, 1])
    
    with col_nav1:
        categoria_nav = st.selectbox(
            "📂 Categoría de Análisis:",
            ["📊 Analítica y Monitoreo Base", "🔬 Ciencia y Pronóstico Climático", "🌍 Gemelo Digital (Modelación Avanzada)"]
        )
        
    with col_nav2:
        if "Analítica" in categoria_nav:
            # 🚀 AQUÍ ESTÁ EL CAMBIO: Movimos "🗺️ Distribución" a la posición 0
            opciones = ["🗺️ Distribución", "🚨 Monitoreo", "📈 Gráficos", "📊 Estadísticas", "📄 Reporte", "🏠 Inicio"]
        elif "Ciencia" in categoria_nav:
            opciones = ["🔮 Pronóstico Climático", "📉 Tendencias", "⚠️ Anomalías", "🔗 Correlación", "🌊 Extremos", "🧪 Sesgo"]
        else:
            opciones = ["🌍 Mapas Avanzados (Motor Aleph)", "✨ Mapas Isoyetas HD", "🌿 Cobertura", "🌱 Zonas Vida", "🌡️ Clima Futuro"]
            
        selected_module_raw = st.selectbox("🎯 Módulo Específico:", opciones)

    with col_nav3:
        st.markdown("<br>", unsafe_allow_html=True)
        # El botón salvavidas, ahora visible y accesible en la barra superior
        if st.button("🔄 Refrescar Memoria", help="Limpia el caché y recarga los datos desde cero", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            keys_to_delete = ['df_long', 'gdf_stations', 'gdf_subcuencas', 'uploaded_file_hash']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.markdown("---")

    # --- D. FILTROS EN BARRA LATERAL (TIEMPO Y LIMPIEZA) ---
    with st.sidebar:
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            min_y = int(df_long[Config.YEAR_COL].min())
            max_y = int(df_long[Config.YEAR_COL].max())
            year_range = st.slider("📅 Años:", min_y, max_y, (min_y, max_y))
            
            c1, c2 = st.columns(2)
            ignore_zeros = c1.checkbox("🚫 Sin Ceros", value=False)
            ignore_nulls = c2.checkbox("🚫 Sin Nulos", value=False)
            apply_interp = st.checkbox("🔄 Interpolación", value=False)

    # --- E. PROCESAMIENTO ---
    mask_time = (df_long[Config.YEAR_COL] >= year_range[0]) & (df_long[Config.YEAR_COL] <= year_range[1])
    df_monthly_filtered = df_long.loc[mask_time].copy()
    
    if ignore_zeros: df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]
    if ignore_nulls: df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

    if apply_interp:
        with st.spinner("Interpolando series..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
    
    df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()

    # Argumentos Globales para Visualizer
    display_args = {
        "df_long": df_monthly_filtered, "df_complete": df_monthly_filtered,
        "gdf_stations": gdf_stations, "gdf_filtered": gdf_filtered,
        "gdf_municipios": gdf_municipios, "gdf_subcuencas": gdf_subcuencas,
        "gdf_predios": gdf_predios, "df_enso": df_enso,
        "stations_for_analysis": stations_for_analysis, "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered, "analysis_mode": "Anual",
        "selected_regions": [], "selected_municipios": [],
        "selected_months": list(range(1, 13)), "year_range": year_range,
        "start_date": pd.to_datetime(f"{year_range[0]}-01-01"), 
        "end_date": pd.to_datetime(f"{year_range[1]}-12-31"),
        "gdf_coberturas": gdf_predios, "interpolacion": "Si" if apply_interp else "No",
        "user_loc": None, "gdf_zona": gdf_zona 
    }

    # Limpiamos el nombre para que los 'elif' de tu código sigan funcionando intactos
    selected_module = "🌍 Mapas Avanzados" if "Mapas Avanzados" in selected_module_raw else selected_module_raw

    # =========================================================================
    # --- F. ENRUTADOR DE MÓDULOS ---
    # =========================================================================

    # 👑 TRATAMIENTO VIP PARA EL MOTOR ALEPH (Si fue seleccionado)
    if selected_module == "🌍 Mapas Avanzados":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px; box-shadow: 0 10px 20px rgba(0,0,0,0.15);">
            <h2 style="color: white; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.2em;">🌍</span> Motor Aleph: Modelación Hidrológica Distribuida
            </h2>
            <p style="font-size: 1.05em; opacity: 0.95; line-height: 1.5; margin-bottom: 0;">
                El nodo central del Gemelo Digital. Aquí la meteorología se fusiona con la topografía (Modelos de Elevación Digital) y las coberturas vegetales satelitales para calcular recarga, infiltración y caudales en cada pixel del territorio.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Módulos Estándar (Usando visualizer.py)
    if selected_module == "🏠 Inicio": viz.display_welcome_tab()
    elif selected_module == "🚨 Monitoreo": viz.display_realtime_dashboard(df_monthly_filtered, gdf_stations, gdf_filtered)
    elif selected_module == "🗺️ Distribución": viz.display_spatial_distribution_tab(**display_args)
    elif selected_module == "📈 Gráficos": viz.display_graphs_tab(**display_args)
    elif selected_module == "📊 Estadísticas": 
        viz.display_stats_tab(**display_args)
        st.markdown("---")
        viz.display_station_table_tab(**display_args)
   
    elif selected_module == "🔮 Pronóstico Climático": 
        # 1. Llamada original al visualizador (Mantiene intacto lo que ya tenías)
        viz.display_climate_forecast_tab(**display_args)
        
        # =========================================================================
        # 2. NUEVA SECCIÓN: EMISOR CLIMÁTICO (ENSO) HACIA EL METABOLISMO
        # =========================================================================
        st.markdown("---")
        st.subheader("📡 Emisor Climático: Conexión con el Metabolismo Territorial")
        st.info("Integra el pronóstico atmosférico con la realidad terrestre. Exporta un escenario de anomalía climática a la Memoria Global para evaluar su impacto en la seguridad hídrica (WRI) y los embalses.")
        
        col_enso1, col_enso2 = st.columns([1.5, 1])
        with col_enso1:
            escenario_enso = st.selectbox("🌊 Seleccione el Escenario Climático a Simular:", 
                                         ["Condiciones Neutras (Histórico Promedio)", 
                                          "🟡 El Niño Moderado (-15% Lluvia / Oferta)", 
                                          "🔴 El Niño Severo (-35% Lluvia / Oferta)", 
                                          "🟢 La Niña Moderada (+15% Lluvia / Oferta)", 
                                          "🔵 La Niña Fuerte (+35% Lluvia / Oferta)"])
            
            # Asignación del multiplicador matemático (factor_clima_enso)
            factor_clima = 1.0
            if "Niño Moderado" in escenario_enso: factor_clima = 0.85
            elif "Niño Severo" in escenario_enso: factor_clima = 0.65
            elif "Niña Moderada" in escenario_enso: factor_clima = 1.15
            elif "Niña Fuerte" in escenario_enso: factor_clima = 1.35
            
        with col_enso2:
            st.metric("Factor de Anomalía de Oferta Natural", f"{factor_clima}x", 
                      f"{(factor_clima-1)*100:+.0f}% respecto a la base", 
                      delta_color="normal" if factor_clima >= 1 else "inverse")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Enviar Escenario al Modelo WRI", use_container_width=True):
                # Guardamos las variables en el cerebro del sistema (st.session_state)
                st.session_state['factor_clima_enso'] = factor_clima
                st.session_state['nombre_escenario_enso'] = escenario_enso
                st.success(f"✅ ¡Escenario guardado! Ve a la página de 'Sistemas Hídricos Territoriales' para ver cómo este clima afecta el metabolismo.")
                
    elif selected_module == "📉 Tendencias": viz.display_trends_and_forecast_tab(**display_args)
    elif selected_module == "⚠️ Anomalías": viz.display_anomalies_tab(**display_args)
    elif selected_module == "🔗 Correlación": viz.display_correlation_tab(**display_args)
    elif selected_module == "🌊 Extremos": viz.display_drought_analysis_tab(**display_args)
    
# --- MÓDULO: MAPAS AVANZADOS (VERSIÓN DEFINITIVA CORREGIDA) ---
    elif selected_module == "🌍 Mapas Avanzados":
        
        # --- A. FUNCIONES AUXILIARES LOCALES ---
        def local_warper_force_4326(tif_path, bounds_wgs84, shape_out):
            """Reproyección forzada a WGS84 (Compatible con Nube y Local)."""
            import rasterio
            from rasterio.warp import reproject, Resampling, calculate_default_transform
            from rasterio.io import MemoryFile
            import os
            import io
            import numpy as np
            
            # Parche para Windows/PROJ
            if os.name == 'nt':
                try:
                    import pyproj
                    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
                except: pass

            try:
                # --- LÓGICA CENTRAL (Para evitar errores de indentación al repetir) ---
                def _core_process(src):
                    transform, width, height = calculate_default_transform(
                        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
                    )
                    minx, miny, maxx, maxy = bounds_wgs84
                    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, shape_out[0], shape_out[1])
                    destination = np.zeros(shape_out, dtype=np.float32)
                    
                    reproject(
                        source=rasterio.band(src, 1), 
                        destination=destination,
                        src_transform=src.transform, 
                        src_crs=src.crs,
                        dst_transform=dst_transform, 
                        dst_crs='EPSG:4326',
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan
                    )
                    return destination

                # --- APERTURA SEGÚN TIPO DE ARCHIVO ---
                
                # CASO 1: Archivo en Memoria (BytesIO desde Supabase)
                if isinstance(tif_path, (io.BytesIO, bytes)) or hasattr(tif_path, 'read'):
                    
                    # 🛡️ BLINDAJE BytesIO: Rebobinar el puntero al inicio (EOF Fix)
                    if hasattr(tif_path, 'seek'):
                        tif_path.seek(0)
                        
                    with MemoryFile(tif_path) as memfile:
                        with memfile.open() as src:
                            return _core_process(src)
                # CASO 2: Ruta de archivo normal (String)
                else:
                    with rasterio.open(tif_path) as src:
                        return _core_process(src)

            except Exception:
                return None

        def generar_popup_html_avanzado(row):
            """Popup HTML enriquecido."""
            nombre = str(row.get(Config.STATION_NAME_COL, 'Estación'))
            muni = str(row.get('municipio', 'N/A'))
            alt = float(row.get('altitud', 0))
            ppt = float(row.get('ppt_media', 0))
            std = float(row.get('ppt_std', 0))
            
            return f"""
            <div style='font-family:sans-serif; font-size:12px; min-width:160px; line-height:1.4;'>
                <b style='color:#1f77b4; font-size:14px'>{nombre}</b><hr style='margin:4px 0; border-top:1px solid #ddd'>
                📍 <b>Mpio:</b> {muni}<br>
                ⛰️ <b>Altitud:</b> {alt:.0f} msnm<br>
                💧 <b>P. Anual:</b> {ppt:.0f} mm<br>
                📊 <b>Desv. Std:</b> ±{std:.0f} mm
            </div>
            """

        if not PHYSICS_AVAILABLE:
            st.error("❌ Módulos 'hydro_physics' o 'admin_utils' no disponibles.")
        else:
            st.header("🌍 Modelación Hidrológica Distribuida (Aleph)")
            
            # --- 0. CARGA DE RECURSOS (NUBE SUPABASE) ---
            # Inicializamos variables
            dem_path = None # En este caso, será un objeto en memoria (BytesIO), no un "path" de texto
            cov_path = None
            gdf_bocatomas = None 

            # A. Carga de Rasters desde BD (Sin archivos locales)
            try:
                # Usamos la función global cargar_raster_db definida al inicio del script
                # (Si no está definida arriba, avísame para dártela)
                dem_bytes = cargar_raster_db("DemAntioquia_EPSG3116.tif")
                cov_bytes = cargar_raster_db("Cob25m_WGS84.tif") # Asegúrate de subir este también si lo usas
                
                if dem_bytes:
                    dem_path = dem_bytes # Asignamos el objeto bytes a la variable
                else:
                    st.error("⛔ Falta el DEM en la Base de Datos. Súbelo en el Panel Admin.")
                    st.stop()
                    
                if cov_bytes:
                    cov_path = cov_bytes
            except NameError:
                st.error("⚠️ La función 'cargar_raster_db' no está definida. Revisa el Paso 2 de las instrucciones.")
                st.stop()

            # B. Bocatomas (Carga Segura)
            try:
                # Intentamos importar get_engine solo si es necesario
                from modules.db_manager import get_engine
                engine = get_engine()
                if engine:
                    gdf_bocatomas = gpd.read_postgis("SELECT * FROM bocatomas", engine, geom_col="geometry")
                    # Estandarización de nombre para visualizer
                    if 'nombre_bocatoma' in gdf_bocatomas.columns: 
                        gdf_bocatomas['nombre_predio'] = gdf_bocatomas['nombre_bocatoma'] 
                    elif 'nombre' in gdf_bocatomas.columns: 
                        gdf_bocatomas['nombre_predio'] = gdf_bocatomas['nombre']
                    else: 
                        gdf_bocatomas['nombre_predio'] = "Bocatoma"
            except Exception as e:
                # Si falla la BD o no existe la tabla, seguimos sin bocatomas (no es crítico)
                gdf_bocatomas = None

            # 1. CONFIGURACIÓN GRID Y SINCRONIZACIÓN ESPACIAL
            # 🤝 LECTURA MAESTRA: El lienzo toma el tamaño exacto que el Sidebar le dicte
            buffer_km = float(st.session_state.get("buffer_global_km", 15.0))
            
            st.info(f"📏 Modelando con un área de influencia de **{buffer_km} km** (Configurado en el panel lateral).")
            
            # Dejamos solo el slider de resolución para no saturar la interfaz
            grid_res = st.slider("Resolución del Grid (Celdas)", min_value=50, max_value=400, value=150, help="Mayor resolución = más detalle pero más tiempo de cálculo.")
            
            # 2. GEOMETRÍA
            if gdf_zona is None: gdf_zona = gdf_filtered
            
            # Conversión aproximada de km a grados (1 grado ~ 111 km en el ecuador)
            buffer_deg = buffer_km / 111.0
            
            # Aplicamos el buffer a la zona para definir la "caja" (Bounding Box) del mapa
            gdf_buffer = gdf_zona.buffer(buffer_deg) if buffer_km > 0 else gdf_zona
            minx, miny, maxx, maxy = gdf_buffer.total_bounds
            
            # Creamos la malla matemática (Meshgrid)
            xi = np.linspace(minx, maxx, grid_res)
            yi = np.linspace(miny, maxy, grid_res)
            grid_x, grid_y = np.meshgrid(xi, yi)
            bounds_calc = (minx, miny, maxx, maxy)
            
            # 3. DATOS ESTACIONES
            df_annual_sums = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
            df_stats = df_annual_sums.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'std']).reset_index()
            df_stats.columns = [Config.STATION_NAME_COL, 'ppt_media', 'ppt_std']
            
            gdf_calc = gdf_filtered.merge(df_stats, on=Config.STATION_NAME_COL)
            gdf_calc = gdf_calc.dropna(subset=['ppt_media', 'geometry'])
            gdf_calc['ppt_media'] = pd.to_numeric(gdf_calc['ppt_media'], errors='coerce').fillna(0)
            gdf_calc['ppt_std'] = pd.to_numeric(gdf_calc['ppt_std'], errors='coerce').fillna(0)
            gdf_calc['popup_html'] = gdf_calc.apply(generar_popup_html_avanzado, axis=1)

            # 4. PROCESAMIENTO DEM Y COBERTURAS
            # Usamos las rutas directas de Config (que ahora apuntan a la nube)
            dem_path = Config.DEM_FILE_PATH
            cov_path = Config.LAND_COVER_RASTER_PATH

            dem_array = None
            with st.spinner("🏔️ Conectando con Supabase y procesando topografía..."):
                try: 
                    # Usa el mismo warper que construimos para hydro_physics
                    dem_array = physics.warper_raster_to_grid(dem_path, bounds_calc, grid_x.shape)
                except Exception as e: 
                    st.warning(f"Aviso Topografía: {e}")
            
            # 5. EJECUCIÓN
            metodo = st.selectbox("Método Interpolación", ['kriging', 'idw', 'spline', 'ked'])
            
            if st.button("🚀 Ejecutar Modelo"):
                st.session_state['ejecutar_aleph'] = True
            
            if st.session_state.get('ejecutar_aleph', False):
                col_close = st.columns([6, 1])[1]
                if col_close.button("❌ Cerrar"):
                    st.session_state['ejecutar_aleph'] = False; st.rerun()

                with st.spinner("Calculando balance hídrico distribuido (esto puede tardar unos segundos)..."):
                    try:
                        # 0. CARGAR MUNICIPIOS (Bypass v2)
                        gdf_municipios = None
                        try:
                            from modules.db_manager import get_engine
                            eng = get_engine()
                            # Leemos la tabla v2 que acabas de subir
                            gdf_municipios = gpd.read_postgis("SELECT * FROM municipios_v2", eng, geom_col="geometry")
                            
                            # Forzar sistema de coordenadas para Folium
                            if gdf_municipios.crs is None:
                                gdf_municipios.set_crs("EPSG:4326", inplace=True)
                            elif gdf_municipios.crs != "EPSG:4326":
                                gdf_municipios = gdf_municipios.to_crs("EPSG:4326")
                            
                            # Limpieza de nombres para el popup
                            gdf_municipios = gdf_municipios.rename(columns={'MPIO_CNMBR': 'Municipio'})
                            
                        except Exception as e:
                            st.warning(f"Capa de municipios v2 no disponible aún: {e}")
                            
                        # 1. DATOS ESTACIONES
                        df_anios_count = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.YEAR_COL].nunique().reset_index(name='n_anios')
                        gdf_calc = gdf_calc.merge(df_anios_count, on=Config.STATION_NAME_COL, how='left').fillna(0)

                        # A. INTERPOLACIÓN HISTÓRICA BASE
                        dem_safe = np.nan_to_num(dem_array, nan=0) if dem_array is not None else None
                        
                        # 💉 BISTURÍ: Enrutamos el Motor Aleph hacia nuestro nuevo interpolador blindado
                        try:
                            from modules.interpolation import interpolador_maestro
                            Z_P_base, Z_Err = interpolador_maestro(
                                df_puntos=gdf_calc, 
                                col_val='ppt_media', 
                                grid_x=grid_x, 
                                grid_y=grid_y, 
                                metodo=str(metodo).lower(), 
                                dem_grid=dem_safe
                            )
                        except Exception as e:
                            st.warning(f"Error en interpolador maestro: {e}. Usando fallback...")
                            Z_P_base, Z_Err = physics.interpolar_variable(
                                gdf_calc, 'ppt_media', grid_x, grid_y, method=metodo, dem_array=dem_safe
                            )
                            
                        if metodo == 'ked': Z_P_base = np.maximum(Z_P_base, 0)

                        # B. BISTURÍ: INYECCIÓN DE ESCENARIOS CMIP6 (Cambio Climático)
                        delta_ppt_sim = st.session_state.get('sim_delta_ppt', 0.0) 
                        delta_temp_sim = st.session_state.get('sim_delta_temp', 0.0)

                        Z_P = Z_P_base * (1 + (delta_ppt_sim / 100.0))
                        
                        # C. MOTOR FÍSICO DISTRIBUIDO (Conectado a la Nube)
                        # AQUÍ ASEGURAMOS QUE cov_path y dem_path SE ENVÍAN AL MOTOR
                        matrices_finales = physics.run_distributed_model(
                            Z_P, grid_x, grid_y, {'dem': dem_path, 'cobertura': cov_path}, bounds_calc
                        )
                        
                        # C. INCERTIDUMBRE
                        if Z_Err is not None:
                            matrices_finales['12. Incertidumbre Interpolación (Std)'] = Z_Err

                        # D. VECTORES (Predios / Bocatomas)
                        gdf_predios_viz = None
                        try:
                            # Intentamos recargar DIRECTAMENTE de la BD para tener todas las columnas (pk_predios, nombre_pre, etc.)
                            from modules.db_manager import get_engine
                            gdf_predios_viz = gpd.read_postgis("SELECT * FROM predios", get_engine(), geom_col="geometry")
                            
                            # Aseguramos proyección WGS84 para el mapa
                            if gdf_predios_viz.crs and gdf_predios_viz.crs.to_string() != "EPSG:4326":
                                gdf_predios_viz = gdf_predios_viz.to_crs("EPSG:4326")
                        except Exception as e:
                            # Si falla la recarga (o no hay BD conectada), usamos la variable global como respaldo
                            if gdf_predios is not None and not gdf_predios.empty:
                                gdf_predios_viz = gdf_predios.copy()

                        # Bocatomas (Misma lógica de seguridad)
                        gdf_bocatomas_viz = None
                        if gdf_bocatomas is not None and not gdf_bocatomas.empty:
                             gdf_bocatomas_viz = gdf_bocatomas.copy()

                        # E. VISUALIZACIÓN
                        viz.display_advanced_maps_tab(
                            df_long=df_monthly_filtered,
                            gdf_stations=gdf_calc,
                            matrices=matrices_finales, 
                            grid=(grid_x, grid_y),
                            mask=None, 
                            gdf_zona=gdf_zona, 
                            gdf_buffer=gdf_buffer, 
                            gdf_predios=gdf_predios_viz, # <--- Aquí va la versión recargada y completa
                            gdf_bocatomas=gdf_bocatomas_viz,
                            gdf_municipios=gdf_municipios
                        )
                        
                        # F. DASHBOARD DE ESTADÍSTICAS (5 COLUMNAS - SUPER COMPLETO)
                        st.markdown("---")
                        st.markdown(f"### 📊 Diagnóstico Hidrológico Integral: {nombre_zona}")
                        
                        try:
                            # 🔥 FIX ANTI-COLAPSO (TOPOLOGY EXCEPTION)
                            # Reparamos cualquier "nudo" o cruce inválido en el polígono antes de medirlo
                            gdf_zona_valida = gdf_zona.copy()
                            gdf_zona_valida['geometry'] = gdf_zona_valida.geometry.make_valid()

                            # --- 1. CÁLCULOS GEOMÉTRICOS ---
                            gdf_zona_proj = gdf_zona_valida.to_crs(epsg=3116) # Proyectar a metros (Magna Sirgas)
                            area_km2 = gdf_zona_proj.area.sum() / 1e6
                            perim_km = gdf_zona_proj.length.sum() / 1e3
                            
                            # Índice de Forma (Compacidad de Gravelius): Kc = 0.28 * P / sqrt(A)
                            ind_gravelius = (0.282 * perim_km) / (np.sqrt(area_km2)) if area_km2 > 0 else 0

                            # --- 2. EXTRACCIÓN DE VARIABLES DEL MODELO ---
                            from shapely.vectorized import contains
                            
                            # Usamos la geometría ya reparada y simplificada levemente si es muy compleja
                            geom_unificada = gdf_zona_valida.geometry.simplify(0.001).unary_union
                            mask_exact = contains(geom_unificada, grid_x, grid_y)
                            
                            def get_avg(keyword): 
                                """Busca la capa en matrices_finales y calcula el promedio zonal."""
                                for k, v in matrices_finales.items():
                                    if keyword in k and v is not None: 
                                        return np.nanmean(v[mask_exact])
                                return 0
                            
                            v_ppt = get_avg("Precipitación")
                            v_temp = get_avg("Temperatura")
                            v_etr = get_avg("Evapotranspiración")
                            v_esc = get_avg("Escorrentía")
                            v_inf = get_avg("Infiltración")
                            v_rec_pot = get_avg("Recarga Potencial")
                            v_rec_real = get_avg("Recarga Real")
                            
                            # --- 3. HIDROLOGÍA Y CAUDALES ---
                            # Factor Q (m3/s) = (mm/año * km2 * 1000) / (31536000 s/año)
                            factor_q = (area_km2 * 1000) / 31536000
                            
                            # Caudales Estimados
                            Q_medio = (v_esc + v_rec_real) * factor_q # Oferta Hídrica Total
                            Q_base = v_rec_real * factor_q # Flujo base (aprox Caudal Mínimo sostenido)
                            Q_maximo = Q_medio * 2.5 # Estimación empírica pico anual (sin datos diarios)
                            Q_ecologico = Q_medio * 0.25 # 25% del Medio (Criterio MADS usual)
                            
                            # Rendimiento Hídrico (m3/ha-año)
                            # 1 mm = 10 m3/ha
                            Rendimiento_m3ha = (v_esc + v_rec_real) * 10 

                            # --- 4. ÍNDICES CLIMÁTICOS Y DE EROSIÓN ---
                            # Índice de Aridez (Martonne): I = P / (T + 10)
                            # 0-10 (Árido), 20-30 (Húmedo), >55 (Perhúmedo)
                            ind_martonne_aridez = v_ppt / (v_temp + 10) if v_temp else 0
                            
                            # Índice Climático (Lang): I = P / T
                            ind_lang = v_ppt / v_temp if v_temp > 0 else 0
                            
                            # Erosividad (Aprox Fournier Modificado o R-USLE simplificado)
                            # R = 0.0739 * P^1.8 (Aprox tropical)
                            ind_erosividad = 0.07 * (v_ppt ** 1.5)

                            # --- 5. RENDERIZADO (5 COLUMNAS) ---
                            # 🎨 INYECCIÓN CSS: Convertimos st.metric en Tarjetas Premium
                            st.markdown("""
                            <style>
                            div[data-testid="metric-container"] {
                                background-color: #fdfdfd;
                                border: 1px solid #e0e0e0;
                                padding: 15px;
                                border-radius: 8px;
                                box-shadow: 2px 2px 8px rgba(0,0,0,0.04);
                                transition: transform 0.2s ease;
                            }
                            div[data-testid="metric-container"]:hover {
                                transform: translateY(-2px);
                                box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
                                border-color: #3498db;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                            k1, k2, k3, k4, k5 = st.columns(5)
                            
                            # COL 1: MORFOMETRÍA
                            k1.markdown("#### 📏 Morfometría")
                            k1.metric("Área", f"{area_km2:.2f} km²")
                            k1.metric("Perímetro", f"{perim_km:.1f} km")
                            k1.metric("Índice Gravelius", f"{ind_gravelius:.2f}", "Forma (Kc)")
                            k1.metric("Estaciones", f"{len(gdf_calc)}")
                            
                            # COL 2: BALANCE HÍDRICO
                            k2.markdown("#### 💧 Balance (mm)")
                            k2.metric("Precipitación", f"{v_ppt:.0f} mm")
                            k2.metric("ETR", f"{v_etr:.0f} mm", "Pérdida")
                            k2.metric("Escorrentía", f"{v_esc:.0f} mm", "Superficial")
                            k2.metric("Infiltración", f"{v_inf:.0f} mm", "Suelo")
                            
                            # COL 3: CAUDALES
                            k3.markdown("#### 🌊 Caudales (m³/s)")
                            k3.metric("Caudal Medio", f"{Q_medio:.2f}")
                            k3.metric("Caudal Mínimo", f"{Q_base:.2f}", "Base Est.")
                            k3.metric("Caudal Máximo", f"{Q_maximo:.2f}", "Pico Est.")
                            k3.metric("Caudal Ecológico", f"{Q_ecologico:.2f}", "25% Qm")
                            
                            # COL 4: ÍNDICES
                            k4.markdown("#### 📉 Índices")
                            k4.metric("Rendimiento", f"{Rendimiento_m3ha:.0f}", "m³/ha-año")
                            k4.metric("Aridez (Martonne)", f"{ind_martonne_aridez:.1f}")
                            k4.metric("Factor Lang", f"{ind_lang:.1f}", "Clima")
                            k4.metric("Erosividad", f"{ind_erosividad:.0f}", "Potencial")

                            # COL 5: AGUAS SUBTERRÁNEAS (NUEVA)
                            k5.markdown("#### ⏬ Aguas Subt.")
                            k5.metric("Recarga Potencial", f"{v_rec_pot:.0f} mm", "Infiltración Total")
                            k5.metric("Recarga Real", f"{v_rec_real:.0f} mm", "Acuífero")
                            k5.metric("Volumen Recarga", f"{(v_rec_real * area_km2 * 1000):.2e} m³", "Anual")
                            
                            # 🌐 INYECCIÓN AL ALEPH (Memoria Global para todo el Gemelo Digital)
                            st.session_state['aleph_recarga_mm'] = float(v_rec_real)
                            st.session_state['aleph_area_km2'] = float(area_km2)
                            
                            # 🚀 LAS NUEVAS CONEXIONES HIDROLÓGICAS
                            st.session_state['aleph_q_rio_m3s'] = float(Q_medio)  # Caudal Medio Oferta
                            st.session_state['aleph_q_min_m3s'] = float(Q_base)   # Caudal Mínimo (Estiaje / Q95)
                            st.session_state['aleph_q_max_m3s'] = float(Q_maximo) # Caudal Máximo Pico
                            
                            st.success("🧠 **¡Conexión Exitosa!** Caudales y recarga inyectados en vivo al Gemelo Digital. Las páginas de 'Toma de Decisiones' y 'Sistemas Hídricos' ahora usarán la física de esta cuenca.")
                            
                        except Exception as e:
                            st.warning(f"Cálculos parciales: {e}")

                    except Exception as e:
                        st.error(f"Error crítico: {e}")

    # --- OTROS MÓDULOS ---
    elif selected_module == "🧪 Sesgo": viz.display_bias_correction_tab(**display_args)
    elif selected_module == "🌿 Cobertura": viz.display_land_cover_analysis_tab(**display_args)

    # --- ZONAS DE VIDA (NATIVO EN LA NUBE) ---
    elif selected_module == "🌱 Zonas Vida":
        # 1. Le pasamos las URLs directas a los argumentos del visualizador
        display_args['dem_file'] = Config.DEM_FILE_PATH
        display_args['ppt_file'] = Config.PRECIP_RASTER_PATH
        
        # 2. Llamamos al visualizador
        try:
            viz.display_life_zones_tab(**display_args)
        except Exception as e:
            st.error(f"⚠️ Error renderizando Zonas de Vida: {e}. Asegúrate de que los Rasters estén en Supabase.")
            
    elif selected_module == "🌡️ Clima Futuro": viz.display_climate_scenarios_tab(**display_args)
    
    # --- ISOYETAS HD (CORREGIDO) ---
    elif selected_module == "✨ Mapas Isoyetas HD":
        st.header("🗺️ Isoyetas Alta Definición (RBF)")
        col1, col2 = st.columns([1,3])
        year_iso = col1.selectbox("Año:", range(int(year_range[1]), int(year_range[0])-1, -1))
        suavidad = col1.slider("Suavizado:", 0.0, 2.0, 0.5)
        
        ids_validos = tuple(gdf_filtered['id_estacion'].unique())
        
        if len(ids_validos) > 2:
            try:
                # Importaciones necesarias
                from modules.db_manager import get_engine
                from sqlalchemy import text
                
                engine = get_engine()
                ids_sql = str(ids_validos) if len(ids_validos) > 1 else f"('{ids_validos[0]}')"
                
                q = text(f"""
                    SELECT e.nombre, e.latitud as lat, e.longitud as lon, SUM(p.valor) as valor
                    FROM precipitacion p JOIN estaciones e ON p.id_estacion = e.id_estacion
                    WHERE extract(year from p.fecha) = :y AND e.id_estacion IN {ids_sql}
                    GROUP BY e.id_estacion, e.nombre, e.latitud, e.longitud
                """)
                
                df_iso = pd.read_sql(q, engine, params={"y": year_iso})
                
                if not df_iso.empty:
                    from scipy.interpolate import Rbf
                    import plotly.graph_objects as go
                    
                    # --- DEFINICIÓN DE LÍMITES (CRUCIAL) ---
                    minx, miny, maxx, maxy = gdf_filtered.total_bounds
                    # ---------------------------------------

                    # Crear grid
                    gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                    
                    rbf = Rbf(df_iso['lon'], df_iso['lat'], df_iso['valor'], function='thin_plate', smooth=suavidad)
                    z = rbf(gx, gy)
                    
                    # Visualización
                    fig = go.Figure(go.Contour(z=z.T, x=np.linspace(minx,maxx,200), y=np.linspace(miny,maxy,200), colorscale="Viridis"))
                    
                    if hasattr(viz, 'add_context_layers_ghost'):
                        viz.add_context_layers_ghost(fig, gdf_filtered)
                        
                    fig.add_trace(go.Scatter(x=df_iso['lon'], y=df_iso['lat'], mode='markers', text=df_iso['nombre']))
                    st.plotly_chart(fig, use_container_width=True)
                
                else: 
                    st.warning("Datos insuficientes para interpolar (Mínimo 3 estaciones con datos en este año).")
            
            except Exception as e: 
                st.error(f"Error en Isoyetas: {e}")
        else: 
            st.warning("Se requieren mín. 3 estaciones para calcular isoyetas.")

    # --- REPORTE (SOLUCIONADO EL DUPLICADO) ---
    elif selected_module == "📄 Reporte":
        st.header("Generación de Informe")
        if st.button("📄 Crear PDF"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf: 
                st.download_button("Descargar PDF", pdf, "reporte_hidro.pdf", "application/pdf")

    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)
    
if __name__ == "__main__":

    main()

    # ==============================================================================
    # ⚙️ CUARTO DE MÁQUINAS: FORJA DE LA MATRIZ MAESTRA
    # ==============================================================================
    st.markdown("---")
    st.subheader("⚙️ Motor Administrativo: Matriz Hidrológica Maestra")
    
    # 📣 EL MANIFIESTO DEL ESTADO DINÁMICO
    st.markdown("""
    <div style="border-left: 5px solid #f39c12; padding: 15px; background-color: rgba(243, 156, 18, 0.1); border-radius: 5px; margin-bottom: 15px;">
        <h4 style="color: #d35400; margin-top: 0;">⚠️ Advertencia de Sincronización Territorial</h4>
        <b style="font-size: 0.95em;">Esta Matriz Maestra es una fotografía del estado dinámico de la realidad. Es el motor de alta velocidad que alimenta la Toma de Decisiones. Si hay cambios sustanciales en el sistema (inclusión de nuevas áreas de análisis, nuevas cuencas, nueva información hidrometeorológica, actualizaciones de coberturas o correcciones topográficas), se DEBE generar una nueva matriz maestra para asegurar que la planificación se base en la verdad territorial más reciente.</b>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🚀 Ejecutar Generación Masiva Multiescala (Recálculo Global)", expanded=False):
        st.info("Esta herramienta recorrerá espacialmente el territorio y calculará la hidrología para TODOS los niveles jerárquicos (NSS3, NSS2, SZH, etc.) en una sola ejecución. La matriz resultante será la fuente de verdad universal.")
        
        # 🔌 INYECCIÓN DE LA LLAVE DE LA BASE DE DATOS (ENGINE)
        from modules.db_manager import get_engine
        engine = get_engine()

        if st.button("⚡ Forjar Matriz Multiescala Definitiva", key="btn_gen_rep_multi", type="primary"):
            try:
                import numpy as np
                import geopandas as gpd
                from sqlalchemy import text
                import pandas as pd
                
                prog_nivel = st.progress(0, text="Iniciando motores espaciales...")

                # 1. Cargamos el clima global (Estaciones y Lluvia)
                with st.spinner("📡 Cargando red de Estaciones Climáticas..."):
                    q_est = text("SELECT id_estacion, altitud, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
                    gdf_est = gpd.read_postgis(q_est, engine, geom_col="geometry").to_crs("EPSG:3116")
                    gdf_est['id_estacion'] = gdf_est['id_estacion'].astype(str)
                    
                    df_rain = pd.read_sql("SELECT id_estacion, AVG(valor)*12 as ppt FROM precipitacion GROUP BY id_estacion", engine)
                    df_rain['id_estacion'] = df_rain['id_estacion'].astype(str)

                res_multiescala = []

                # 🧠 EL NÚCLEO FÍSICO: Función maestra que procesa CUALQUIER mapa que le pasemos
                def procesar_capa_espacial(gdf_base, dicc_niveles, offset_progreso, total_pasos):
                    gdf_base['geometry'] = gdf_base.geometry.make_valid()
                    gdf_base = gdf_base[gdf_base.geometry.notnull()]
                    
                    paso_actual = offset_progreso
                    for nombre_nivel, col_bd in dicc_niveles.items():
                        if col_bd not in gdf_base.columns: continue
                        
                        prog_nivel.progress(paso_actual / total_pasos, text=f"Calculando Hidrología para: {nombre_nivel}")
                        paso_actual += 1
                        
                        # Magia Espacial: Agrupamos y fusionamos los polígonos
                        gdf_nivel = gdf_base.dissolve(by=col_bd).reset_index()
                        
                        for i, row in gdf_nivel.iterrows():
                            nom_territorio = str(row.get(col_bd, "Desconocido")).strip()
                            if nom_territorio in ["", "None", "nan", "Cuenca Sin Nombre"]: continue
                            
                            area_km2 = row.geometry.area / 1e6
                            if area_km2 <= 0: area_km2 = 1.0
                            
                            # Cruce Climático (Qué estaciones caen cerca de este territorio)
                            buf = row.geometry.buffer(20000) # 20km de búsqueda
                            est_in = gdf_est[gdf_est.geometry.within(buf)]
                            
                            ppt_media, altitud_media = 2500.0, 1500.0
                            if not est_in.empty:
                                ids = est_in['id_estacion'].tolist()
                                vals_ppt = df_rain[df_rain['id_estacion'].isin(ids)]['ppt']
                                if not vals_ppt.empty: ppt_media = vals_ppt.mean()
                                vals_alt = est_in['altitud'].dropna()
                                if not vals_alt.empty: altitud_media = vals_alt.mean()
                                
                            # 🔬 MODELACIÓN FÍSICA Y TERMODINÁMICA
                            temp_calc = max(5.0, 28.0 - (0.006 * altitud_media))
                            L = 300 + 25*temp_calc + 0.05*(temp_calc**3)
                            etr = ppt_media / np.sqrt(0.9 + (ppt_media/L)**2) if L > 0 else 0
                            escorrentia_total = ppt_media - etr
                            
                            # Partición del Flujo
                            recarga_mm = escorrentia_total * 0.15
                            escorrentia_superficial = escorrentia_total * 0.85
                            q_medio = (escorrentia_superficial * area_km2 * 1000) / 31536000
                            q_base = (recarga_mm * area_km2 * 1000) / 31536000
                            
                            res_multiescala.append({
                                "Jerarquia": nombre_nivel,
                                "Territorio": nom_territorio,
                                "Area_km2": round(area_km2, 2), 
                                "Altitud_m": round(altitud_media, 0),
                                "Temp_C": round(temp_calc, 1),
                                "Lluvia_mm": round(ppt_media, 0), 
                                "ETR_mm": round(etr, 0),
                                "Recarga_mm": round(recarga_mm, 0),
                                "Escorrentia_mm": round(escorrentia_superficial, 0),
                                "Caudal_Medio_m3s": round(q_medio + q_base, 3)
                            })
                    return paso_actual

                # --- 🗺️ FASE 1: PROCESAR EL MAPA DE CUENCAS ---
                with st.spinner("💧 Calculando red hidrográfica completa..."):
                    gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry").to_crs("EPSG:3116")
                    niveles_cuencas = {'NSS3': 'nom_nss3', 'NSS2': 'nom_nss2', 'NSS1': 'nom_nss1', 'SZH': 'nom_szh', 'ZH': 'nomzh', 'AH': 'nomah'}
                    pasos_totales = len(niveles_cuencas) + 4 # Sumamos los 4 niveles administrativos que vienen
                    paso_actual = procesar_capa_espacial(gdf_cuencas, niveles_cuencas, 0, pasos_totales)

                # --- 🏛️ FASE 2: PROCESAR EL MAPA ADMINISTRATIVO (VERSIÓN CLIENTE SUPABASE) ---
                with st.spinner("🏛️ Sincronizando con Territorio Maestro y calculando escalas superiores..."):
                    try:
                        import io
                        from modules.utils import normalizar_texto
                        from modules.admin_utils import init_supabase
                        
                        # 1. CARGA SEGURA DEL DICCIONARIO MAESTRO (Usando el cliente oficial)
                        supabase_client = init_supabase()
                        if not supabase_client:
                            raise Exception("No se pudo iniciar el cliente de Supabase para descargar el maestro.")
                            
                        # Descargamos el archivo directamente del bucket
                        bucket_name = "sihcli_maestros"
                        file_path = "territorio_maestro.csv"
                        
                        response = supabase_client.storage.from_(bucket_name).download(file_path)
                        
                        if not response:
                            raise Exception("El archivo CSV vino vacío desde Supabase.")
                            
                        df_maestro = pd.read_csv(io.BytesIO(response))
                        df_maestro['match_id'] = df_maestro['municipio'].apply(normalizar_texto)

                        # 2. CARGA Y CRUCE CON EL MAPA
                        gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry").to_crs("EPSG:3116")
                        col_mun_mapa = 'mpio_cnmbr' if 'mpio_cnmbr' in gdf_mun.columns else ('MPIO_CNMBR' if 'MPIO_CNMBR' in gdf_mun.columns else 'municipio')
                        gdf_mun['match_id'] = gdf_mun[col_mun_mapa].apply(normalizar_texto)
                        
                        # Unimos el mapa con el Excel para heredar las jerarquías oficiales
                        gdf_mun = gdf_mun.merge(df_maestro[['match_id', 'municipio', 'subregion', 'region', 'depto_nom']], on='match_id', how='left')
                        
                        niveles_admin = {
                            'MUNICIPAL': 'municipio',
                            'REGION': 'subregion',
                            'MACROREGION': 'region',
                            'DEPARTAMENTO': 'depto_nom'
                        }
                        
                        # Pasamos el mapa enriquecido al motor físico
                        procesar_capa_espacial(gdf_mun, niveles_admin, paso_actual, pasos_totales)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error en Sincronización Maestra Administrativa: {e}")

                prog_nivel.progress(1.0, text="¡Física territorial procesada al 100%!")
                
                # --- 💾 GUARDADO MAESTRO (VERSIÓN BLINDADA CON LLAVE UNIVERSAL) ---
                with st.spinner("Forjando Llaves Universales y Guardando Cerebro Hidrológico en Supabase..."):
                    from modules.utils import normalizar_texto
                    from sqlalchemy import text
                    
                    df_matriz = pd.DataFrame(res_multiescala)
                    
                    # 1. Función Forjadora de Llave
                    def forjar_llave_hidro(row):
                        jerarquia = str(row.get('Jerarquia', '')).upper()
                        if jerarquia == "MUNICIPAL": jerarquia = "MUNICIPAL"
                        elif jerarquia == "REGIONAL": jerarquia = "REGION"
                        elif jerarquia == "DEPARTAMENTAL": jerarquia = "DEPARTAMENTO"
                        
                        terr_limpio = normalizar_texto(row.get('Territorio', ''))
                        terr = str(terr_limpio).replace(" ", "_").upper()
                        return f"{jerarquia}_{terr}_TOTAL"

                    # 2. PRIMERO aplicamos la llave a todos los datos para igualarlos
                    df_matriz['LLAVE_UNIVERSAL'] = df_matriz.apply(forjar_llave_hidro, axis=1)
                    
                    # 3. LUEGO aplicamos el Filtro Anti-Duplicados usando la LLAVE
                    # Esto colapsa las variaciones de mapa (ej. "CHIMA" y "CHIMÃ") en una sola fila.
                    df_matriz = df_matriz.drop_duplicates(subset=['LLAVE_UNIVERSAL'], keep='first')
                    
                    # 4. Inyección Segura a PostgreSQL
                    try:
                        with engine.begin() as conn:
                            conn.execute(text('ALTER TABLE matriz_hidrologica_maestra ADD COLUMN IF NOT EXISTS "LLAVE_UNIVERSAL" TEXT;'))
                            conn.execute(text("DELETE FROM matriz_hidrologica_maestra;"))
                            
                        df_matriz.to_sql("matriz_hidrologica_maestra", engine, if_exists='append', index=False)
                        st.cache_data.clear()
                        
                        st.success(f"✅ EL ALEPH ESTÁ COMPLETO. Matriz Hidrológica forjada con {len(df_matriz)} territorios únicos.")
                        
                        resumen = df_matriz['Jerarquia'].value_counts().reset_index()
                        resumen.columns = ['Nivel', 'Cantidad de Territorios']
                        st.dataframe(resumen, hide_index=True)

                        csv_matriz = df_matriz.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Descargar Gran Matriz (CSV)", csv_matriz, "Matriz_Hidro_Multiescala.csv", "text/csv")            
                    except Exception as e:
                        st.error(f"🚨 Error inyectando a SQL: {e}")

            except Exception as e:
                st.error(f"❌ Error crítico procesando la información espacial: {e}")
