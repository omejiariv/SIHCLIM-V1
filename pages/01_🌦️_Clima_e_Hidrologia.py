# pages/Clima_e_Hidrologia.py

import os
import sys
import warnings
import io
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
from rasterio.io import MemoryFile

# 1. CONFIGURACIÓN DE PÁGINA (PRIMERA LÍNEA SIEMPRE)
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌦️", layout="wide")
warnings.filterwarnings("ignore")

# 2. PARCHE UNIVERSAL (GDAL/PROJ)
if os.name == 'nt':
    try:
        import pyproj
        os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    except: pass

# 3. IMPORTACIÓN ROBUSTA DE MÓDULOS (PATH FIX)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import selectors, admin_utils, reporter, visualizer as viz
from modules.config import Config
from modules.db_manager import get_engine
from modules.data_processor import complete_series, load_and_process_all_data
from modules.dem_extractor import completar_altitudes_con_dem

# 4. IMPORTACIÓN Y BLINDAJE DE MÓDULOS AVANZADOS
try:
    from modules import hydro_physics as physics
    from modules.admin_utils import download_raster_to_temp
    from modules.utils import cargar_capa_espacial_cache
    
    # Intentamos importar la función de análisis
    try:
        from modules.analysis import calculate_trends_mann_kendall
    except ImportError:
        # SI FALLA: Definimos un "wrapper" de emergencia usando la librería mk
        import pymannkendall as mk
        def calculate_trends_mann_kendall(serie):
            return mk.original_test(serie)
            
    PHYSICS_AVAILABLE = True
except ImportError as e:
    PHYSICS_AVAILABLE = False
    st.toast(f"⚠️ Módulos avanzados limitados: {e}", icon="⚠️")

# 5. MENÚ DE NAVEGACIÓN
selectors.renderizar_menu_navegacion("Clima e Hidrología")

# 6. SINCRONIZACIÓN CLIMÁTICA (NOAA/IRI)
if 'enso_fase' not in st.session_state:
    try:
        from modules.climate_api import get_iri_enso_forecast
        df_enso, _ = get_iri_enso_forecast()
        if df_enso is not None and not df_enso.empty:
            f = df_enso.iloc[0]
            # Usamos update para mantener el código limpio
            st.session_state.update({
                'enso_fase': "Niña 🌧️" if f.get('La Niña', 0) > 50 else ("Niño ☀️" if f.get('El Niño', 0) > 50 else "Neutro ⚖️"),
                'aleph_iri_nino': int(f.get('El Niño', 0)),
                'aleph_iri_neutro': int(f.get('Neutral', 0)),
                'aleph_iri_nina': int(f.get('La Niña', 0)),
                'aleph_iri_trimestre': str(f.get('Trimestre', 'N/A'))
            })
            st.toast(f"📡 Clima Global Sincronizado: {st.session_state['enso_fase']}", icon="✅")
    except Exception:
        st.session_state.update({'enso_fase': "Neutro ⚖️", 'aleph_iri_nino': 0, 'aleph_iri_neutro': 100, 'aleph_iri_nina': 0})
        st.toast("⚠️ Usando clima Neutro por fallo de conexión.", icon="🔌")
        
# 6. CACHÉ DE RASTERS (OPTIMIZADO)
@st.cache_resource(show_spinner=False)
def cargar_raster_db(filename):
    try:
        client = admin_utils.init_supabase()
        return io.BytesIO(client.storage.from_("rasters").download(filename))
    except: return None

@st.cache_resource(show_spinner="📡 Consultando Sistema...")
def load_all_data_cached():
    return load_and_process_all_data()

# ==========================================
# APLICACIÓN PRINCIPAL
# ==========================================
def main():
    # 🛡️ IMPORTACIÓN DE SEGURIDAD INTERNA
    import pandas as pd
    import geopandas as gpd
    
    # --- A. SELECTOR ESPACIAL ---
    try:
        ids_estaciones_dummy, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
    except Exception as e:
        st.sidebar.error(f"Error en Selector: {e}")
        st.stop()

    # 🔥 FIX 1: Validamos contra el nombre del territorio
    if not nombre_zona or nombre_zona in ["-- Seleccione --", "Sin Selección", "NINGUNO", "Antioquia"]:
        st.info("👈 Seleccione una Cuenca o Municipio en el menú lateral para comenzar.")
        st.stop()

    # 🚀 FIX 2: RESTAURACIÓN DEL BUFFER ESPACIAL EN EL SIDEBAR
    st.sidebar.markdown("---")
    buffer_km = st.sidebar.slider(
        "🎯 Radio de Búsqueda (Buffer en km):", 
        min_value=0.0, max_value=50.0, value=15.0, step=1.0, 
        help="Expande la zona de búsqueda para capturar estaciones vecinas. Es vital para cuencas pequeñas (como NSS3) que no tienen estaciones dentro de su límite exacto."
    )
    # Guardamos en sesión para que el Motor Aleph (Mapas) respete este mismo buffer
    st.session_state['buffer_global_km'] = buffer_km

    # --- B. CARGA DE DATOS ---
    try:
        (gdf_stations, gdf_municipios, df_all_rain, df_enso, gdf_subcuencas, gdf_predios) = load_all_data_cached()
    except Exception as e:
        st.error(f"Error cargando datos base: {e}")
        st.stop()

    # 🌍 FIX 3: INTERSECCIÓN ESPACIAL GEODÉSICA (CON BUFFER)
    ids_estaciones = []
    if gdf_zona is not None and not gdf_zona.empty and gdf_stations is not None and not gdf_stations.empty:
        # Proyectamos a Magna-Sirgas (EPSG:3116) para poder medir en metros exactos
        gdf_zona_proj = gdf_zona.to_crs(epsg=3116)
        gdf_stations_proj = gdf_stations.to_crs(epsg=3116)
        
        # Aplicamos el radio de expansión (convertimos km a metros)
        if buffer_km > 0:
            gdf_zona_proj['geometry'] = gdf_zona_proj.geometry.buffer(buffer_km * 1000)
        
        # Cruzamos las geometrías: ¿Qué estaciones caen en el área expandida?
        estaciones_dentro = gpd.sjoin(gdf_stations_proj, gdf_zona_proj, predicate='intersects')
        ids_estaciones = estaciones_dentro['id_estacion'].tolist()

    if not ids_estaciones:
        st.warning(f"⚠️ No se encontraron estaciones meteorológicas a menos de {buffer_km} km de la zona: {nombre_zona}.")
        st.info("💡 Sugerencia: Aumenta el 'Radio de Búsqueda' en el panel lateral.")
        st.stop()

    # =========================================================================
    # --- C. DESCARGA DINÁMICA DE LLUVIAS (EL FIX ARQUITECTÓNICO) ---
    # =========================================================================
    # En lugar de descargar millones de datos, le pedimos a Supabase SOLO 
    # los datos de las estaciones que caen en nuestra cuenca.
    with st.spinner(f"☁️ Descargando historial de precipitaciones para {len(ids_estaciones)} estaciones..."):
        try:
            # 🛡️ FIX: Importaciones locales blindadas para evitar el error de "local variable 'text'"
            from sqlalchemy import text
            from modules.db_manager import get_engine

            ids_fmt = ",".join([f"'{str(x).strip()}'" for x in ids_estaciones])
            q_lluvia = text(f"SELECT id_estacion, fecha, valor FROM precipitacion WHERE id_estacion IN ({ids_fmt})")
            
            engine_lluvia = get_engine()
            with engine_lluvia.connect() as conn:
                df_all_rain = pd.read_sql(q_lluvia, conn)
            
            if not df_all_rain.empty:
                # Damos formato a las columnas
                df_all_rain['id_estacion'] = df_all_rain['id_estacion'].astype(str).str.strip()
                df_all_rain = df_all_rain.rename(columns={'fecha': Config.DATE_COL, 'valor': Config.PRECIPITATION_COL})
                df_all_rain[Config.DATE_COL] = pd.to_datetime(df_all_rain[Config.DATE_COL])
                df_all_rain[Config.PRECIPITATION_COL] = pd.to_numeric(df_all_rain[Config.PRECIPITATION_COL], errors='coerce')
                df_all_rain = df_all_rain.dropna(subset=[Config.DATE_COL])
                df_all_rain[Config.YEAR_COL] = df_all_rain[Config.DATE_COL].dt.year
                df_all_rain[Config.MONTH_COL] = df_all_rain[Config.DATE_COL].dt.month
                
                # Unimos con los nombres de las estaciones
                df_long = pd.merge(
                    df_all_rain,
                    gdf_stations[["id_estacion", Config.STATION_NAME_COL]],
                    on="id_estacion",
                    how="inner"
                )
            else:
                df_long = pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error descargando historial de lluvias: {e}")
            df_long = pd.DataFrame()

    if df_long.empty:
        st.warning(f"La zona '{nombre_zona}' no tiene registros históricos de precipitación válidos.")
        st.stop()

    # 🛡️ 2. FILTRO DE INTEGRIDAD FÍSICA (PURGA DE ANOMALÍAS EXTREMAS)
    if Config.PRECIPITATION_COL in df_long.columns:
        umbral_maximo = 8000  # Límite físico en mm (Ej. diluvios irreales > 30,000mm)
        picos_mask = df_long[Config.PRECIPITATION_COL] > umbral_maximo
        
        if picos_mask.any():
            num_picos = picos_mask.sum()
            st.warning(f"⚠️ Alerta de Calidad: Se neutralizaron {num_picos} registros físicamente imposibles (> {umbral_maximo}mm) en esta zona. Han sido pasados a nulos para no sesgar la interpolación.")
            df_long.loc[picos_mask, Config.PRECIPITATION_COL] = np.nan
    
    # 3. 📍 Filtro de geometrías espaciales
    if gdf_stations is not None:
        gdf_stations['id_estacion'] = gdf_stations['id_estacion'].astype(str).str.strip()
        gdf_filtered = gdf_stations[gdf_stations['id_estacion'].isin(ids_estaciones)].copy()
    else:
        gdf_filtered = gpd.GeoDataFrame()

    # =========================================================================
    # --- JERARQUÍA DE NAVEGACIÓN ---
    # =========================================================================
    st.markdown("""
    <style>
    .block-container { padding-top: 3rem !important; }
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
    
    col_nav1, col_nav2, col_nav3 = st.columns([1.5, 1.5, 1])
    
    with col_nav1:
        categoria_nav = st.selectbox(
            "📂 Categoría de Análisis:",
            ["📊 Analítica y Monitoreo Base", "🔬 Ciencia y Pronóstico Climático", "🌍 Gemelo Digital (Modelación Avanzada)"]
        )
        
    with col_nav2:
        if "Analítica" in categoria_nav:
            # 🧠 Inyectamos la nueva opción en la lista base
            opciones = ["🗺️ Distribución", "🚨 Monitoreo", "📈 Gráficos", "📊 Estadísticas", "🧠 Peritaje y Consistencia", "📄 Reporte", "🏠 Inicio"]
        elif "Ciencia" in categoria_nav:
            opciones = ["🔮 Pronóstico Climático", "📉 Tendencias", "⚠️ Anomalías", "🔗 Correlación", "🌊 Extremos", "🧪 Sesgo"]
        else:
            opciones = ["🌍 Mapas Avanzados (Motor Aleph)", "✨ Mapas Isoyetas HD", "🌿 Cobertura", "🌱 Zonas Vida", "🌡️ Clima Futuro"]
            
        selected_module_raw = st.selectbox("🎯 Módulo Específico:", opciones)

    with col_nav3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refrescar Memoria", help="Limpia el caché y recarga los datos desde cero", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            keys_to_delete = ['df_long', 'gdf_stations', 'gdf_subcuencas', 'uploaded_file_hash']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # --- D. FILTROS EN BARRA LATERAL (TIEMPO Y LIMPIEZA) ---
    with st.sidebar:
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            
            # 🛡️ FIX: Blindaje del Slider de Tiempo
            if df_long is not None and not df_long.empty:
                min_y = int(df_long[Config.YEAR_COL].min())
                max_y = int(df_long[Config.YEAR_COL].max())
            else:
                min_y, max_y = 1980, 2026 # Valores de seguridad
                
            year_range = st.slider("📅 Años:", min_y, max_y, (min_y, max_y))
            
            c1, c2 = st.columns(2)
            ignore_zeros = c1.checkbox("🚫 Sin Ceros", value=False)
            ignore_nulls = c2.checkbox("🚫 Sin Nulos", value=False)
            apply_interp = st.checkbox("🔄 Interpolación", value=False)

        # 🛡️ INYECCIÓN QUIRÚRGICA: Filtro de Homogenización
        st.markdown("---")
        st.markdown("### ⏳ Filtro de Homogenización")
        min_years = st.slider(
            "Años mínimos de registro continuos", 
            min_value=0, max_value=50, value=0, step=1, 
            help="Filtra estaciones que no cumplan con este mínimo de años históricos. Útil para eliminar el 'escalón' de estaciones post-2010."
        )

    # --- E. PROCESAMIENTO ESTRATÉGICO ---
    # 1. Aplicar filtro temporal primero
    mask_time = (df_long[Config.YEAR_COL] >= year_range[0]) & (df_long[Config.YEAR_COL] <= year_range[1])
    df_monthly_filtered = df_long.loc[mask_time].copy()
    
    # 2. LIMPIEZA DE CALIDAD (Fundamental hacerlo antes de contar años)
    if ignore_zeros: 
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]
    if ignore_nulls: 
        df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

    # 3. ⏳ FILTRO DE HOMOGENIZACIÓN (Ahora sí, contando solo la verdad)
    if min_years > 0 and not df_monthly_filtered.empty:
        # Contamos años únicos que sobrevivieron a la purga de ceros y nulos
        conteo_agnos = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.YEAR_COL].nunique()
        estaciones_validas = conteo_agnos[conteo_agnos >= min_years].index.tolist()
        
        # Recortamos la matriz
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].isin(estaciones_validas)]
        st.sidebar.success(f"✅ {len(estaciones_validas)} estaciones cumplen con >{min_years} años reales.")

    # 4. Extraemos la lista final y purificada para los módulos de análisis
    if not df_monthly_filtered.empty:
        stations_for_analysis = df_monthly_filtered[Config.STATION_NAME_COL].unique().tolist()
    else:
        stations_for_analysis = []
        st.warning(f"Ninguna estación cumple con el requisito de tener {min_years} años de datos continuos.")
        st.stop()

    # 5. Interpolación (Solo a las estaciones sobrevivientes históricas)
    if apply_interp:
        with st.spinner("Interpolando series..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
    
    df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum(min_count=10).reset_index()

    # 4. Ajuste de seguridad para year_range
    if 'year_range' not in locals() or year_range is None:
        year_range = [2020, 2026]

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
    elif selected_module == "🗺️ Distribución": 
        viz.display_spatial_distribution_tab(**display_args)
        
        # =====================================================================
        # --- CAJA DESPLEGABLE: GUÍA DE CODIFICACIÓN HIDROLÓGICA (IDEAM) ---
        # =====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Guía de Navegación: ¿Cómo leer los códigos hidrológicos (IDEAM)?", expanded=False):
            st.markdown("""
            ### El Sistema de las "Muñecas Rusas" (Metodología Pfafstetter)
            Para organizar el agua en Colombia, el IDEAM utiliza un sistema jerárquico. Imagina que el agua de nuestro territorio se organiza como un árbol, desde el tronco más grueso hasta la hoja más pequeña. Cada nivel añade más dígitos al código:
            
            * **🌊 AH (Área Hidrográfica - 1 dígito):** *El Gran Tronco.*
            * **Ríos Principales:**
                * **💧 ZH (Zona Hidrográfica - 2 dígitos):** *Las Ramas Principales.*
                * **🌿 SZH (Subzona Hidrográfica - 4 dígitos):** *Las Ramas Secundarias.*
            * **Tributarios y Microcuencas (Niveles Subsiguientes - NSS):**
                * **🍃 NSS1 (Nivel 4):** Afluentes directos de la SZH.
                * **🌱 NSS2 (Nivel 5):** Microcuencas locales.
                * **💧 NSS3 (Nivel 6):** Pequeños drenajes y nacimientos.
            
            ---
            ### 🗺️ Diccionario Rápido: Códigos Principales en Antioquia
            Para facilitar tu búsqueda en la plataforma, aquí tienes la relación exacta de los códigos mayores (Áreas y Zonas Hidrográficas) que rigen nuestro departamento:
            
            | 🌊 Área Hidrográfica (AH) | Código AH | 💧 Zona Hidrográfica (ZH) | Código ZH |
            | :--- | :---: | :--- | :---: |
            | **Magdalena Cauca** | `2` | Medio Magdalena | `23` |
            | **Magdalena Cauca** | `2` | Cauca | `26` |
            | **Magdalena Cauca** | `2` | Nechí | `27` |
            | **Caribe** | `1` | Atrato - Darién | `11` |
            | **Caribe** | `1` | Caribe - Litoral | `12` |
            | **Caribe** | `1` | Sinú | `13` |
            
            > **💡 Tip de Navegación:** Si buscas el Río Porce (Código `2601`), fíjate que empieza por `26`. Esto te indica inmediatamente que pertenece a la Zona Hidrográfica del **Cauca** y, por consiguiente, al Área Hidrográfica **Magdalena Cauca** (`2`).
            """)
            
            # --- SECCIÓN DE DESCARGA DEL DOCUMENTO OFICIAL ---
            st.markdown("---")
            st.markdown("#### 📚 Documento Oficial de Referencia")
            st.info("Consulta el manual completo de codificación y zonificación hidrológica para comprender los delineamientos técnicos a fondo.")
            
            # Botón de descarga en Streamlit
            try:
                # Asegúrate de que la ruta coincida con donde guardes el PDF en tu servidor/proyecto
                ruta_pdf = "Codificacion Hidrologica Antioquia IDEAM.pdf" 
                with open(ruta_pdf, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                
                st.download_button(
                    label="📥 Descargar Manual de Codificación IDEAM (PDF)",
                    data=PDFbyte,
                    file_name="Codificacion_Hidrologica_Antioquia_IDEAM.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            except FileNotFoundError:
                st.error("⚠️ El archivo PDF de referencia no se encontró en el servidor. (Asegúrate de subir 'Codificacion Hidrologica Antioquia IDEAM.pdf' a la carpeta principal de tu aplicación).")
            
    elif selected_module == "📈 Gráficos": viz.display_graphs_tab(**display_args)
    elif selected_module == "📊 Estadísticas": 
        viz.display_stats_tab(**display_args)
        st.markdown("---")
        viz.display_station_table_tab(**display_args)

    # ==============================================================================
    # 🧠 NUEVO MÓDULO: PERITAJE Y CONSISTENCIA HIDROMETEOROLÓGICA
    # ==============================================================================
    elif selected_module == "🧠 Peritaje y Consistencia":
        st.subheader("🧠 Analítica Avanzada: Peritaje de Consistencia Hidrometeorológica")
        st.markdown("Evaluación forense automática de la calidad de las series de tiempo y su respuesta ante macro-eventos globales (ENOS).")
        
        # Ruta al Modelo de Elevación Digital ráster para Antioquia
        RUTA_DEM = os.path.join("data", "dem_antioquia.tif")
        
        # ----------------------------------------------------------------------
        # 🏢 1. CAJA MENSAJE PERMANENTE: BALANCE MACROREGIONAL (AHORA EXPANDIBLE)
        # ----------------------------------------------------------------------
        if df_monthly_filtered is not None and not df_monthly_filtered.empty and stations_for_analysis:
            with st.expander("🌍 Diagnóstico Macroregional de la Selección", expanded=True):
                df_macro = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)].copy()
                
                if not df_macro.empty:
                    df_macro_anual = df_macro.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].agg(['sum', 'count']).reset_index()
                    df_macro_anual.columns = ['Estación', 'Año', 'Precipitación', 'Meses_Validos']
                    
                    # Filtro de integridad regional robusto (>700 mm)
                    df_macro_integro = df_macro_anual[(df_macro_anual['Meses_Validos'] >= 10) & (df_macro_anual['Precipitación'] >= 700.0)]
                    
                    if not df_macro_integro.empty:
                        idx_macro_max = df_macro_integro['Precipitación'].idxmax()
                        idx_macro_min = df_macro_integro['Precipitación'].idxmin()
                        
                        est_max_reg = df_macro_integro.loc[idx_macro_max, 'Estación']
                        ano_max_reg = int(df_macro_integro.loc[idx_macro_max, 'Año'])
                        val_max_reg = df_macro_integro.loc[idx_macro_max, 'Precipitación']
                        
                        est_min_reg = df_macro_integro.loc[idx_macro_min, 'Estación']
                        ano_min_reg = int(df_macro_integro.loc[idx_macro_min, 'Año'])
                        val_min_reg = df_macro_integro.loc[idx_macro_min, 'Precipitación']
                        
                        df_medias = df_macro_integro.groupby('Estación')['Precipitación'].mean().reset_index()
                        est_mas_humeda = df_medias.loc[df_medias['Precipitación'].idxmax(), 'Estación']
                        media_mas_humeda = df_medias['Precipitación'].max()
                        
                        est_mas_seca = df_medias.loc[df_medias['Precipitación'].idxmin(), 'Estación']
                        media_mas_seca = df_medias['Precipitación'].min()
                        
                        c_macro1, c_macro2 = st.columns(2)
                        with c_macro1:
                            st.markdown(f"**🌧️ Núcleo de Alta Precipitación (Más Húmedo):**")
                            st.markdown(f"* **Estación Predominante:** `{est_mas_humeda}`")
                            st.markdown(f"* **Rendimiento Medio Anual:** `{media_mas_humeda:,.1f} mm/año`")
                            st.markdown(f"* **Récord Histórico Absoluto:** `{val_max_reg:,.1f} mm` en el año **{ano_max_reg}** (`{est_max_reg}`)")
                            
                        with c_macro2:
                            st.markdown(f"**🌵 Núcleo de Sombra de Lluvia (Más Seco):**")
                            st.markdown(f"* **Estación Predominante:** `{est_mas_seca}`")
                            st.markdown(f"* **Rendimiento Medio Anual:** `{media_mas_seca:,.1f} mm/año`")
                            st.markdown(f"* **Mínimo Histórico Absoluto:** `{val_min_reg:,.1f} mm` en el año **{ano_min_reg}** (`{est_min_reg}`)")
                    else:
                        st.warning("⚠️ No hay suficientes años con registros que superen el umbral de integridad de 700 mm en esta selección geográfica.")
                else:
                    st.info("No se encontraron registros de series temporales para las estaciones de este sector.")

        # ----------------------------------------------------------------------
        # 🔍 SELECTOR LOCAL Y PROCESAMIENTO DE LA ESTACIÓN
        # ----------------------------------------------------------------------
        if stations_for_analysis:
            est_sel = st.selectbox("🔍 Seleccione la Estación para Peritaje Forense:", stations_for_analysis)
            
            if est_sel and df_monthly_filtered is not None and not df_monthly_filtered.empty:
                
                nombre_estacion_puro = est_sel.split('[')[0].strip() if '[' in est_sel else est_sel
                codigo_puro = est_sel.split('[')[1].replace(']', '').strip() if '[' in est_sel else ""
                
                df_est = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == est_sel].copy()
                if df_est.empty:
                    df_est = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == nombre_estacion_puro].copy()
                if df_est.empty and codigo_puro:
                    df_est = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].astype(str).str.contains(codigo_puro)].copy()
                
                df_est = df_est.dropna(subset=[Config.PRECIPITATION_COL])
                
                df_anual = df_est.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].agg(['sum', 'count']).reset_index()
                df_anual.columns = ['Año', 'Precipitación', 'Meses_Validos']
                df_integro = df_anual[(df_anual['Meses_Validos'] >= 10) & (df_anual['Precipitación'] >= 700.0)]
                
                # --- DISEÑO BI-PANEL: FICHA GEOGRÁFICA Y VALIDACIÓN MACROCLIMÁTICA ---
                col_panel1, col_panel2 = st.columns([1.5, 2])
                
                # 🏢 2. FICHA TÉCNICA GEOGRÁFICA (AHORA EXPANDIBLE)
                with col_panel1:
                    with st.expander("🛡️ Ficha Técnica Geográfica", expanded=True):
                        municipio_est = "No especificado"
                        altitud_est = "N/D"
                        
                        if gdf_stations is not None and not gdf_stations.empty:
                            meta_est = gdf_stations[gdf_stations[Config.STATION_NAME_COL].astype(str).str.contains(codigo_puro)] if codigo_puro else gdf_stations[gdf_stations[Config.STATION_NAME_COL] == est_sel]
                            if not meta_est.empty:
                                col_muni = [c for c in meta_est.columns if 'muni' in c.lower() or 'id_muni' in c.lower()]
                                col_alt = [c for c in meta_est.columns if 'alt' in c.lower() or 'ele' in c.lower() or 'msnm' in c.lower()]
                                
                                if col_muni: municipio_est = str(meta_est.iloc[0][col_muni[0]]).upper()
                                
                                if col_alt and pd.notna(meta_est.iloc[0][col_alt[0]]) and str(meta_est.iloc[0][col_alt[0]]).strip() != "":
                                    altitud_est = f"{int(meta_est.iloc[0][col_alt[0]]):,} msnm"
                                elif os.path.exists(RUTA_DEM):
                                    try:
                                        import rasterio
                                        punto_geom = meta_est.geometry.iloc[0]
                                        with rasterio.open(RUTA_DEM) as src:
                                            coord_par = [(punto_geom.x, punto_geom.y)]
                                            for val in src.sample(coord_par):
                                                alt_dem = val[0]
                                                if alt_dem > -9999:
                                                    altitud_est = f"{int(alt_dem):,} msnm (Vía DEM)"
                                    except:
                                        altitud_est = "Error muestreo DEM"
                        
                        st.markdown(f"**📍 Estación:** `{est_sel}`")
                        st.markdown(f"**🏢 Municipio:** `{municipio_est}`")
                        st.markdown(f"**⛰️ Altitud:** `{altitud_est}`")
                        st.divider()
                        
                        años_totales = df_anual.shape[0]
                        años_completos = df_integro.shape[0]
                        años_reconstruidos = df_anual[(df_anual['Precipitación'].notna()) & (df_anual['Precipitación'] % 1 != 0)].shape[0]
                        
                        st.metric(label="Años con Datos Activos", value=f"{años_totales} años")
                        st.metric(label="Años con Integridad (>10 meses)", value=f"{años_completos} años")
                        st.metric(label="Años Reparados por Gemelo Digital", value=f"{años_reconstruidos} años")
                
                # 🌍 3. VALIDACIÓN MACROCLIMÁTICA BASE CON EXTREMOS MENSUALES (AHORA EXPANDIBLE)
                with col_panel2:
                    with st.expander("🌍 Validación Macroclimática Base", expanded=True):
                        
                        if not df_integro.empty:
                            # Cálculos anuales
                            idx_max = df_integro['Precipitación'].idxmax()
                            idx_min = df_integro['Precipitación'].idxmin()
                            
                            max_año = int(df_integro.loc[idx_max, 'Año'])
                            max_val = df_integro.loc[idx_max, 'Precipitación']
                            min_año = int(df_integro.loc[idx_min, 'Año'])
                            min_val = df_integro.loc[idx_min, 'Precipitación']
                            
                            anios_niña_hitos = [1975, 1988, 1999, 2010, 2011, 2022]
                            anios_niño_hitos = [1983, 1992, 1997, 2015, 2023]
                            
                            diag_max = "✅ **Consistente:** Coincide con un evento histórico de excesos por La Niña." if max_año in anios_niña_hitos else "ℹ️ Máximo local regular."
                            diag_min = "✅ **Consistente:** Coincide con una sequía crítica por El Niño fuerte." if min_año in anios_niño_hitos else "ℹ️ Mínimo local regular."
                            
                            # Fila 1: Máximo Anual
                            st.markdown(f"**Año Más Lluvioso Registrado:**")
                            st.markdown(f"🥇 {max_año} con **{max_val:,.1f} mm/año**")
                            st.caption(diag_max)
                            st.divider()
                            
                            # Fila 2: Mínimo Anual
                            st.markdown(f"**Año Menos Lluvioso Registrado:**")
                            st.markdown(f"🌵 {min_año} con **{min_val:,.1f} mm/año**")
                            st.caption(diag_min)
                            st.divider()
                            
                            # Fila 3: Hitos Mensuales Absolutos de la Serie
                            st.markdown(f"**📊 Hitos Mensuales Absolutos de la Serie:**")
                            if Config.MONTH_COL in df_est.columns and not df_est.empty:
                                meses_dict = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                                              7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
                                
                                idx_mes_max = df_est[Config.PRECIPITATION_COL].idxmax()
                                idx_mes_min = df_est[Config.PRECIPITATION_COL].idxmin()
                                
                                val_mes_max = df_est.loc[idx_mes_max, Config.PRECIPITATION_COL]
                                num_mes_max = int(df_est.loc[idx_mes_max, Config.MONTH_COL])
                                ano_mes_max = int(df_est.loc[idx_mes_max, Config.YEAR_COL])
                                
                                val_mes_min = df_est.loc[idx_mes_min, Config.PRECIPITATION_COL]
                                num_mes_min = int(df_est.loc[idx_mes_min, Config.MONTH_COL])
                                ano_mes_min = int(df_est.loc[idx_mes_min, Config.YEAR_COL])
                                
                                st.markdown(f"🌊 **Mes Más Lluvioso:** {meses_dict.get(num_mes_max, str(num_mes_max))} de {ano_mes_max} con **{val_mes_max:,.1f} mm**")
                                st.markdown(f"🍂 **Mes Menos Lluvioso:** {meses_dict.get(num_mes_min, str(num_mes_min))} de {ano_mes_min} con **{val_mes_min:,.1f} mm**")
                        else:
                            st.warning("⚠️ Sin suficientes años íntegros para emitir un dictamen macroclimático robusto.")
                
                # 📈 4. EVOLUCIÓN MULTIANUAL DE SOPORTE (AHORA EXPANDIBLE)
                st.markdown("---")
                with st.expander("📈 Evolución Multianual de Soporte", expanded=True):
                    df_grafico = df_anual.set_index('Año')[['Precipitación']]
                    st.bar_chart(df_grafico, color="#3498db")
                
                # ----------------------------------------------------------------------
                # 📊 REPORTES CONSOLIDADOS DE LA SELECCIÓN (TABLA MAESTRA)
                # ----------------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 📊 Reporte Consolidado Forense de la Selección")
                st.markdown("Matriz unificada de estadísticas críticas para todas las estaciones activas en la delimitación actual.")
                
                registros_tabla = []
                for est_item in stations_for_analysis:
                    code_item = est_item.split('[')[1].replace(']', '').strip() if '[' in est_item else ""
                    name_item = est_item.split('[')[0].strip() if '[' in est_item else est_item
                    
                    df_sub = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].astype(str).str.contains(code_item if code_item else name_item)].copy()
                    df_sub = df_sub.dropna(subset=[Config.PRECIPITATION_COL])
                    
                    if not df_sub.empty:
                        df_sub_anual = df_sub.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].agg(['sum', 'count']).reset_index()
                        df_sub_anual.columns = ['Año', 'Precipitación', 'Meses_Validos']
                        df_sub_integro = df_sub_anual[(df_sub_anual['Meses_Validos'] >= 10) & (df_sub_anual['Precipitación'] >= 700.0)]
                        
                        anios_activos = df_sub_anual.shape[0]
                        anios_integros = df_sub_integro.shape[0]
                        media_anual = df_sub_integro['Precipitación'].mean() if not df_sub_integro.empty else float('nan')
                        
                        if not df_sub_integro.empty:
                            idx_sub_max = df_sub_integro['Precipitación'].idxmax()
                            idx_sub_min = df_sub_integro['Precipitación'].idxmin()
                            max_val_anual = f"{df_sub_integro.loc[idx_sub_max, 'Precipitación']:,.1f} ({int(df_sub_integro.loc[idx_sub_max, 'Año'])})"
                            min_val_anual = f"{df_sub_integro.loc[idx_sub_min, 'Precipitación']:,.1f} ({int(df_sub_integro.loc[idx_sub_min, 'Año'])})"
                        else:
                            max_val_anual, min_val_anual = "N/D", "N/D"
                            
                        registros_tabla.append({
                            "Código/Estación": est_item,
                            "Años Activos": anios_activos,
                            "Años Íntegros": anios_integros,
                            "Media Anual (mm)": round(media_anual, 1) if pd.notna(media_anual) else "N/D",
                            "Máximo Anual Histór. (Año)": max_val_anual,
                            "Mínimo Anual Histór. (Año)": min_val_anual
                        })
                
                if registros_tabla:
                    df_reporte_final = pd.DataFrame(registros_tabla)
                    st.dataframe(df_reporte_final, use_container_width=True, hide_index=True)
                    
                    csv_reporte = df_reporte_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar Reporte de Estadísticas Básicas (CSV)",
                        data=csv_reporte,
                        file_name="Reporte_Consolidado_Forense_Hidro.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error("❌ No se pudieron procesar las series para la estación seleccionada.")
        else:
            st.info("👈 Seleccione una Cuenca o Municipio con estaciones activas en el menú lateral para iniciar el peritaje.")
            
    elif selected_module == "🔮 Pronóstico Climático": 
        # Llama al visualizador reconstruido (que ahora tiene historia, NOAA y Prophet)
        viz.display_climate_forecast_tab(**display_args)
        
        # =========================================================================
        # 2. NUEVA SECCIÓN: EMISOR CLIMÁTICO (MULTIVERSO ENSO) HACIA EL METABOLISMO
        # =========================================================================

        st.markdown("---")
        st.subheader("📡 Emisor Climático: Multiverso Probabilístico (IRI)")
        st.info("Integra el pronóstico atmosférico en vivo con la realidad terrestre. Exporta el paquete de probabilidades completo a la Memoria Global para evaluar el Riesgo Esperado en el metabolismo territorial y la toma de decisiones.")
        
        # Leemos las probabilidades vivas del Aleph (o asignamos 100% neutro si apenas está cargando)
        p_nino = st.session_state.get('aleph_iri_nino', 0)
        p_neutro = st.session_state.get('aleph_iri_neutro', 100) 
        p_nina = st.session_state.get('aleph_iri_nina', 0)
        
        # Definimos los impactos físicos (anomalías) para cada universo (Ajustables según tu criterio)
        impacto_nino = 0.65   # -35% de Oferta Hídrica (Sequía)
        impacto_neutro = 1.00 # 100% de Oferta Hídrica (Normal)
        impacto_nina = 1.35   # +35% de Oferta Hídrica (Exceso/Inundación)
        
        # Cálculo de la Esperanza Matemática: Factor de Anomalía Esperado Ponderado
        factor_esperado = ((p_nino * impacto_nino) + (p_neutro * impacto_neutro) + (p_nina * impacto_nina)) / 100.0
        
        col_enso1, col_enso2 = st.columns([1.5, 1])
        with col_enso1:
            st.markdown("##### 🌌 Universos Paralelos Integrados:")
            st.markdown(f"- ☀️ **Universo Niño ({p_nino}% prob):** Oferta al 65% (Déficit)")
            st.markdown(f"- ⚖️ **Universo Neutro ({p_neutro}% prob):** Oferta al 100% (Histórico)")
            st.markdown(f"- 🌧️ **Universo Niña ({p_nina}% prob):** Oferta al 135% (Exceso)")
            
        with col_enso2:
            st.metric("Factor de Oferta Esperado (Riesgo Ponderado)", f"{factor_esperado:.2f}x", 
                      f"{(factor_esperado-1)*100:+.1f}% impacto real proyectado", 
                      delta_color="normal" if factor_esperado >= 1 else "inverse")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Enviar Multiverso al Aleph", use_container_width=True):
                # 1. Enviamos el paquete completo para el Optimizador de la Pág 09
                st.session_state['aleph_clima_multiverso'] = {
                    "Niño": impacto_nino,
                    "Neutro": impacto_neutro,
                    "Niña": impacto_nina
                }
                
                # 2. Compatibilidad hacia atrás: enviamos el factor esperado para la Pág 08 (Sistemas Hídricos)
                st.session_state['factor_clima_enso'] = factor_esperado
                st.session_state['nombre_escenario_enso'] = f"Multiverso IRI ({factor_esperado:.2f}x)"
                
                st.success("✅ ¡Multiverso inyectado! Ve a Toma de Decisiones o Sistemas Hídricos para ver el impacto probabilístico.")
                
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
                    gdf_bocatomas = cargar_capa_espacial_cache("SELECT * FROM bocatomas", engine, geom_col="geometry")
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
            
            # Buscamos el valor en las posibles variables del sidebar (por si cambió de nombre)
            buffer_km = float(st.session_state.get("buffer_km", st.session_state.get("buffer_global_km", 25.0)))
            
            # Forzamos la actualización visual para que refleje el número exacto
            st.info(f"📏 Modelando con un área de influencia de **{buffer_km} km** (Sincronizado con el panel lateral).")
            
            # Dejamos solo el slider de resolución para no saturar la interfaz
            grid_res = st.slider("Resolución del Grid (Celdas)", min_value=50, max_value=500, value=100, help="Mayor resolución = más detalle pero más tiempo de cálculo.")
            
            # 2. GEOMETRÍA
            if gdf_zona is None: gdf_zona = gdf_filtered
            
            # Conversión aproximada de km a grados (1 grado ~ 111 km en el ecuador)
            buffer_deg = buffer_km / 111.0
            
            # Aplicamos el buffer a la zona para definir la "caja" (Bounding Box) del mapa
            gdf_buffer = gdf_zona.buffer(buffer_deg) if buffer_km > 0 else gdf_zona
            minx, miny, maxx, maxy = gdf_buffer.total_bounds
            
            # Creamos la malla matemática (Meshgrid)
            import numpy as np  # <--- INYECTA ESTA LÍNEA AQUÍ
            xi = np.linspace(minx, maxx, grid_res)
            yi = np.linspace(miny, maxy, grid_res)
            grid_x, grid_y = np.meshgrid(xi, yi)
            bounds_calc = (minx, miny, maxx, maxy)
            
            # 3. DATOS ESTACIONES
            df_annual_sums = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum(min_count=10).reset_index()
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
                            import geopandas as gpd
                            from modules.db_manager import get_engine
                            eng = get_engine()
                            # Leemos la tabla v2 que acabas de subir
                            gdf_municipios = cargar_capa_espacial_cache("SELECT * FROM municipios_v2", eng, geom_col="geometry")
                            
                            # Forzar sistema de coordenadas para Folium
                            if gdf_municipios.crs is None:
                                gdf_municipios.set_crs("EPSG:4326", inplace=True)
                            elif gdf_municipios.crs != "EPSG:4326":
                                gdf_municipios = gdf_municipios.to_crs("EPSG:4326")
                            
                            # Limpieza de nombres para el popup
                            gdf_municipios = gdf_municipios.rename(columns={'MPIO_CNMBR': 'Municipio'})

                        except ImportError:
                            st.error("❌ Error Crítico: La librería 'geopandas' no está instalada en el servidor.")
                            
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
                            gdf_predios_viz = cargar_capa_espacial_cache("SELECT * FROM predios", get_engine(), geom_col="geometry")
                            
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

    # --- REPORTE INTEGRADO (ACTUALIZADO) ---
    elif selected_module == "📄 Reporte":
        st.header("Generación de Informe")
        
        # 1. IMPORTAR EL NUEVO MOTOR
        from modules import reporter
        
        # 2. CAPTURAR EL CAPÍTULO ACTUAL
        fig_actual = st.session_state.get('fig_clima_principal') 
        capitulo = {
            'title': 'Análisis Hidroclimático',
            'text': f'Análisis realizado con {len(stations_for_analysis)} estaciones. Periodo: {year_range}.',
            'fig': fig_actual
        }
        
        # 3. GUARDAR EN EL ESTADO GLOBAL
        if 'pdf_chapters' not in st.session_state:
            st.session_state.pdf_chapters = []
        if capitulo not in st.session_state.pdf_chapters:
            st.session_state.pdf_chapters.append(capitulo)
            st.success("✅ Capítulo de Clima capturado para el reporte final.")
        
        # 4. GENERAR PDF CONSOLIDADO
        st.info("Este reporte consolidará todo lo que hayas visitado en otras pestañas.")
        if st.button("📥 Generar PDF Consolidado de SIHCLI"):
            # AQUÍ ESTÁ LA CLAVE: Llamamos al nombre correcto de la función
            pdf_data = reporter.generate_consolidated_pdf(st.session_state.pdf_chapters)
            st.download_button(
                "Descargar PDF Final", 
                pdf_data, 
                "Reporte_Sihcli_Final.pdf", 
                "application/pdf"
            )
    
    # ==============================================================================
    # ⚙️ BÓVEDA DE ADMINISTRADOR: FORJA DE LA MATRIZ MAESTRA
    # ==============================================================================
    st.markdown("---")
    with st.expander("⚙️ Área Restringida: Motor de Generación de Matriz Hidrológica", expanded=False):
        st.subheader("Matriz Hidrológica Maestra (Recálculo Global)")
        
        # 🔐 SISTEMA DE SEGURIDAD
        admin_pwd = st.text_input("🔑 Ingrese clave de administrador para desbloquear el motor:", type="password", key="admin_pwd_forja")
        
        if admin_pwd == "AdminPoter":
            st.success("🔓 Acceso Concedido. Protocolos de forja desbloqueados.")
            
            st.markdown("""
            <div style="border-left: 5px solid #f39c12; padding: 15px; background-color: rgba(243, 156, 18, 0.1); border-radius: 5px; margin-bottom: 15px;">
                <h4 style="color: #d35400; margin-top: 0;">⚠️ Advertencia de Sincronización Territorial</h4>
                <b style="font-size: 0.95em;">Esta Matriz Maestra es el motor de alta velocidad que alimenta la Toma de Decisiones. Solo debe ejecutarse si se han modificado las bases de datos espaciales (nuevos municipios, cuencas o capas topográficas).</b>
            </div>
            """, unsafe_allow_html=True)

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
                        gdf_est = cargar_capa_espacial_cache(q_est, geom_col="geometry").to_crs("EPSG:3116")
                        gdf_est['id_estacion'] = gdf_est['id_estacion'].astype(str)
                        
                        df_rain = pd.read_sql("SELECT id_estacion, AVG(valor)*12 as ppt FROM precipitacion GROUP BY id_estacion", engine)
                        # 🚀 FIX QUIRÚRGICO: Verificar que la columna exista antes de modificarla
                        if not df_rain.empty and 'id_estacion' in df_rain.columns:
                            df_rain['id_estacion'] = df_rain['id_estacion'].astype(str)
                        else:
                            st.warning("⚠️ No se encontraron datos históricos de lluvia. Los cálculos asumirán 2500mm por defecto.")
                            df_rain = pd.DataFrame(columns=['id_estacion', 'ppt'])

                    res_multiescala = []

                    def procesar_capa_espacial(gdf_base, dicc_niveles, offset_progreso, total_pasos):
                        gdf_base['geometry'] = gdf_base.geometry.make_valid()
                        gdf_base = gdf_base[gdf_base.geometry.notnull()]
                        
                        paso_actual = offset_progreso
                        for nombre_nivel, col_bd in dicc_niveles.items():
                            if col_bd not in gdf_base.columns: continue
                            
                            prog_nivel.progress(paso_actual / total_pasos, text=f"Calculando Hidrología para: {nombre_nivel}")
                            paso_actual += 1
                            
                            gdf_nivel = gdf_base.dissolve(by=col_bd).reset_index()
                            
                            for i, row in gdf_nivel.iterrows():
                                nom_territorio = str(row.get(col_bd, "Desconocido")).strip()
                                if nom_territorio in ["", "None", "nan", "Cuenca Sin Nombre"]: continue
                                
                                area_km2 = row.geometry.area / 1e6
                                if area_km2 <= 0: area_km2 = 1.0
                                
                                buffer_seguro = float(st.session_state.get("buffer_global_km", 15.0))
                                buf = row.geometry.buffer(buffer_seguro * 1000)
                                est_in = gdf_est[gdf_est.geometry.within(buf)]
                                
                                ppt_media, altitud_media = 2500.0, 1500.0
                                if not est_in.empty:
                                    ids = est_in['id_estacion'].tolist()
                                    vals_ppt = df_rain[df_rain['id_estacion'].isin(ids)]['ppt']
                                    if not vals_ppt.empty: ppt_media = vals_ppt.mean()
                                    vals_alt = est_in['altitud'].dropna()
                                    if not vals_alt.empty: altitud_media = vals_alt.mean()
                                    
                                temp_calc = max(5.0, 28.0 - (0.006 * altitud_media))
                                L = 300 + 25*temp_calc + 0.05*(temp_calc**3)
                                etr = ppt_media / np.sqrt(0.9 + (ppt_media/L)**2) if L > 0 else 0
                                escorrentia_total = ppt_media - etr
                                
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

                    # --- 🗺️ FASE 1: PROCESAR CUENCAS ---
                    with st.spinner("Calculando red hidrográfica..."):
                        gdf_cuencas = cargar_capa_espacial_cache("SELECT * FROM cuencas", geom_col="geometry").to_crs("EPSG:3116")

                        # 🧬 INYECCIÓN DE ADN (LLAVE ÚNICA)
                        # Fusionamos Nombre + Código para evitar que cuencas homónimas se derritan en una sola
                        mapa_columnas = {'nom_nss3': 'nss3', 'nom_nss2': 'nss2', 'nom_nss1': 'nss1', 'nom_szh': 'szh', 'nomzh': 'zh', 'nomah': 'ah'}
                        for col_nom, col_cod in mapa_columnas.items():
                            if col_nom in gdf_cuencas.columns and col_cod in gdf_cuencas.columns:
                                gdf_cuencas[col_nom] = gdf_cuencas.apply(
                                    lambda r: f"{str(r[col_nom]).strip()} - ({str(r[col_cod]).strip()})"
                                    if pd.notnull(r[col_nom]) and str(r[col_nom]).strip() != "" and pd.notnull(r[col_cod])
                                    else r[col_nom], axis=1
                                )
                        
                        if 'CorpoAmb' in gdf_cuencas.columns and 'depto_regi' in gdf_cuencas.columns:
                            gdf_cuencas['CorpoAmb'] = gdf_cuencas['CorpoAmb'].str.strip()
                            gdf_cuencas['depto_regi'] = gdf_cuencas['depto_regi'].str.strip()
                            mask_aburra = gdf_cuencas['depto_regi'].str.contains('Aburr', case=False, na=False)
                            gdf_cuencas.loc[mask_aburra, 'CorpoAmb'] = 'AMVA'

                        niveles_cuencas = {'NSS3': 'nom_nss3', 'NSS2': 'nom_nss2', 'NSS1': 'nom_nss1', 'SZH': 'nom_szh', 'ZH': 'nomzh', 'AH': 'nomah', 'CAR': 'CorpoAmb', 'REGION_CUENCA': 'depto_regi'}
                        pasos_totales = len(niveles_cuencas) + 5
                        paso_actual = procesar_capa_espacial(gdf_cuencas, niveles_cuencas, 0, pasos_totales)

                    # --- 🏛️ FASE 2: PROCESAR MUNICIPIOS ---
                    with st.spinner("🏛️ Sincronizando Niveles Administrativos..."):
                        try:
                            import io, requests
                            url_maestro = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
                            res_m = requests.get(url_maestro, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=15)
                            df_maestro = pd.read_excel(io.BytesIO(res_m.content))
                            
                            # 🚀 FIX QUIRÚRGICO 1: Forzar 5 dígitos exactos en el maestro (05079)
                            df_maestro['dp_mp'] = df_maestro['dp_mp'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(5)
                            
                            gdf_mun = cargar_capa_espacial_cache("SELECT * FROM municipios", geom_col="geometry").to_crs("EPSG:3116")
                            posibles_cols = ['mpio_cdpmp', 'MPIO_CDPMP', 'dp_mp', 'mpio_cdp', 'MPIO_CDP']
                            col_id_mapa = next((c for c in posibles_cols if c in gdf_mun.columns), None)
                            
                            # 🚀 FIX QUIRÚRGICO 2: Forzar 5 dígitos exactos en el mapa PostGIS
                            gdf_mun['dp_mp'] = gdf_mun[col_id_mapa].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(5)
                            
                            # Cruce perfecto por Código DANE Dpto-Municipio
                            gdf_mun = gdf_mun.merge(df_maestro[['dp_mp', 'municipio', 'subregion', 'region', 'depto_nom', 'car']], on='dp_mp', how='left')
                            
                            # Homologar Valle de Aburrá
                            mask_aburra = gdf_mun['subregion'].str.contains('Aburr', case=False, na=False)
                            gdf_mun.loc[mask_aburra, 'car'] = 'AMVA'
                            
                            col_nom_mapa = next((c for c in ['mpio_cnmbr', 'MPIO_CNMBR', 'municipio'] if c in gdf_mun.columns), 'municipio')
                            gdf_mun['municipio_clean'] = gdf_mun['municipio'].fillna(gdf_mun[col_nom_mapa])
                            gdf_mun['subregion'] = gdf_mun['subregion'].fillna('Sin Region')
                            gdf_mun['region'] = gdf_mun['region'].fillna('Sin Macroregion')
                            gdf_mun['depto_nom'] = gdf_mun['depto_nom'].fillna('Antioquia')
                            
                            # Filtrar solo Antioquia (código 05)
                            gdf_mun = gdf_mun[gdf_mun['dp_mp'].str.startswith('05')].copy()
                            
                            niveles_admin = {'MUNICIPAL': 'municipio_clean', 'REGION': 'subregion', 'MACROREGION': 'region', 'DEPARTAMENTO': 'depto_nom', 'CAR': 'car'}
                            paso_actual = procesar_capa_espacial(gdf_mun, niveles_admin, paso_actual, pasos_totales)
                        except Exception as e:
                            st.error(f"Error en Fase 2: {e}")
                            st.stop()
                            
                    prog_nivel.progress(1.0, text="¡Física territorial procesada al 100%!")
                    
                    # --- 💾 GUARDADO MAESTRO CON LLAVES UNIVERSALES PERFECTAS ---
                    with st.spinner("Forjando Llaves Universales e Inyectando a PostgreSQL..."):
                        from modules.utils import normalizar_texto
                        df_matriz = pd.DataFrame(res_multiescala)
                        
                        def forjar_llave_hidro(row):
                            jerarquia = str(row.get('Jerarquia', '')).upper()
                            # 🚀 FIX QUIRÚRGICO 3: Homologar nombres de jerarquías con selectors.py
                            if jerarquia in ["DEPARTAMENTAL", "DEPARTAMENTO"]: 
                                jerarquia = "DEPARTAMENTO"
                            elif jerarquia in ["REGIONAL", "REGION", "SUBREGION"]: 
                                jerarquia = "REGION"
                            elif jerarquia in ["MUNICIPAL", "MUNICIPIO"]: 
                                jerarquia = "MUNICIPIO"
                                
                            if row.get('Territorio') is None or str(row.get('Territorio')) == 'nan': 
                                return None
                                
                            terr = str(normalizar_texto(row.get('Territorio', ''))).replace(" ", "_").upper()
                            return f"{jerarquia}_{terr}_TOTAL"

                        df_matriz['LLAVE_UNIVERSAL'] = df_matriz.apply(forjar_llave_hidro, axis=1)
                        df_matriz = df_matriz.drop_duplicates(subset=['LLAVE_UNIVERSAL'], keep='first')
                        
                        with engine.begin() as conn:
                            conn.execute(text('ALTER TABLE matriz_hidrologica_maestra ADD COLUMN IF NOT EXISTS "LLAVE_UNIVERSAL" TEXT;'))
                            conn.execute(text("DELETE FROM matriz_hidrologica_maestra;"))
                        df_matriz.to_sql("matriz_hidrologica_maestra", engine, if_exists='append', index=False)
                        st.cache_data.clear()
                        st.success(f"✅ EL ALEPH ESTÁ COMPLETO. {len(df_matriz)} territorios sincronizados.")
                        
                        csv_data = df_matriz.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Descargar Matriz Recién Forjada (CSV)", csv_data, "Matriz_Hidro_Multiescala.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error crítico: {e}")

            st.markdown("---")
            if st.button("🔍 Preparar Descarga de Base Guardada"):
                try:
                    df_export = pd.read_sql("SELECT * FROM matriz_hidrologica_maestra", engine)
                    if not df_export.empty:
                        st.write(f"✅ Se encontraron {len(df_export)} territorios listos para exportar.")
                        csv_export = df_export.to_csv(index=False).encode('utf-8')
                        st.download_button(label="💾 Descargar Matriz Maestra (CSV)", data=csv_export, file_name="Matriz_Hidro_Maestra_SQL.csv", mime="text/csv")
                    else:
                        st.warning("La base de datos está vacía. Ejecuta la forja primero.")
                except Exception as e:
                    st.error(f"Error al conectar con la base: {e}")
                    
        elif admin_pwd != "":
            st.error("❌ Clave incorrecta. Acceso denegado.")

if __name__ == "__main__":
    main()

