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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌦️", layout="wide")
warnings.filterwarnings("ignore")

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

# --- FUNCIÓN MAESTRA DE CARGA (SOLUCIÓN A ERRORES 1 Y 2) ---
@st.cache_resource(show_spinner=False)
def cargar_raster_db(filename):
    """
    Descarga un archivo raster (TIF) desde la tabla 'system_assets' de Supabase
    y lo devuelve como un objeto en memoria (BytesIO), listo para Rasterio.
    """
    try:
        engine = get_engine()
        if engine is None:
            return None
            
        with engine.connect() as conn:
            # Buscamos el binario por nombre
            result = conn.execute(
                text("SELECT file_data FROM system_assets WHERE filename = :f"),
                {"f": filename}
            ).fetchone()
            
            if result:
                return io.BytesIO(result[0]) # Éxito: Devolvemos el archivo en RAM
            else:
                # Si no existe, retornamos None para manejar el error después
                return None
    except Exception as e:
        st.error(f"Error descargando mapa '{filename}': {e}")
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

    # --- C. BARRA LATERAL (NAVEGACIÓN) ---
    with st.sidebar:
        st.divider()
        st.markdown("### 🚀 Navegación")
        selected_module = st.radio(
            "Ir a:",
            [
                "🏠 Inicio", "🚨 Monitoreo", "🗺️ Distribución", "📈 Gráficos", 
                "📊 Estadísticas", "🔮 Pronóstico Climático", "📉 Tendencias", 
                "⚠️ Anomalías", "🔗 Correlación", "🌊 Extremos", 
                "🌍 Mapas Avanzados", "🧪 Sesgo", "🌿 Cobertura", 
                "🌱 Zonas Vida", "🌡️ Clima Futuro", "📄 Reporte", "✨ Mapas Isoyetas HD"
            ]
        )
        st.markdown("---")

        # Filtro de Tiempo
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            min_y = int(df_long[Config.YEAR_COL].min())
            max_y = int(df_long[Config.YEAR_COL].max())
            year_range = st.slider("📅 Años:", min_y, max_y, (min_y, max_y))
            
            c1, c2 = st.columns(2)
            ignore_zeros = c1.checkbox("🚫 Sin Ceros", value=False)
            ignore_nulls = c2.checkbox("🚫 Sin Nulos", value=False)
            apply_interp = st.checkbox("🔄 Interpolación", value=False)

        # --- AQUÍ VA EL BOTÓN SALVAVIDAS ---
        if st.button("🔄 Refrescar Datos", help="Borra memoria y recarga"):
            # 1. Borrar caché de funciones (consultas SQL)
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # 2. 🔥 CLAVE: Borrar las variables guardadas en memoria 🔥
            keys_to_delete = ['df_long', 'gdf_stations', 'gdf_subcuencas', 'uploaded_file_hash']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            
            # 3. Recargar
            st.rerun()

    # --- D. PROCESAMIENTO ---
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

    # --- E. ENRUTADOR DE MÓDULOS ---
    st.title(f"🌦️ Análisis: {nombre_zona}")

    # Módulos Estándar (Usando visualizer.py)
    if selected_module == "🏠 Inicio": viz.display_welcome_tab()
    elif selected_module == "🚨 Monitoreo": viz.display_realtime_dashboard(df_monthly_filtered, gdf_stations, gdf_filtered)
    elif selected_module == "🗺️ Distribución": viz.display_spatial_distribution_tab(**display_args)
    elif selected_module == "📈 Gráficos": viz.display_graphs_tab(**display_args)
    elif selected_module == "📊 Estadísticas": 
        viz.display_stats_tab(**display_args)
        st.markdown("---")
        viz.display_station_table_tab(**display_args)
    elif selected_module == "🔮 Pronóstico Climático": viz.display_climate_forecast_tab(**display_args)
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
            
            # --- 0. CARGA DE RECURSOS ---
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

            # 1. CONFIGURACIÓN GRID
            c1, c2 = st.columns(2)
            buffer_km = c1.slider("Buffer Análisis (km)", 0.0, 50.0, 20.0)
            grid_res = c2.slider("Resolución Grid (Celdas)", 50, 400, 150)
            
            # 2. GEOMETRÍA
            if gdf_zona is None: gdf_zona = gdf_filtered
            buffer_deg = buffer_km / 111.0
            gdf_buffer = gdf_zona.buffer(buffer_deg) if buffer_km > 0 else gdf_zona
            minx, miny, maxx, maxy = gdf_buffer.total_bounds
            
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

            # 4. PROCESAMIENTO DEM (Visualización Previa)
            dem_array = None
            with st.spinner("🏔️ Procesando topografía..."):
                try: dem_array = local_warper_force_4326(dem_path, bounds_calc, grid_x.shape)
                except: pass
            
            # 5. EJECUCIÓN
            metodo = st.selectbox("Método Interpolación", ['kriging', 'idw', 'spline', 'ked'])
            
            if st.button("🚀 Ejecutar Modelo"):
                st.session_state['ejecutar_aleph'] = True
            
            # ... (código previo dentro del if session_state) ...
            if st.session_state.get('ejecutar_aleph', False):
                col_close = st.columns([6, 1])[1]
                if col_close.button("❌ Cerrar"):
                    st.session_state['ejecutar_aleph'] = False; st.rerun()

                with st.spinner("Calculando balance hídrico distribuido..."):
                    try:
                        # 0. CARGAR MUNICIPIOS (Capa Contexto)
                        gdf_municipios = None
                        try:
                            from modules.db_manager import get_engine
                            eng = get_engine()
                            # Intentamos cargar municipios si existen en BD
                            gdf_municipios = gpd.read_postgis("SELECT * FROM municipios", eng, geom_col="geometry")
                            # Aseguramos proyección
                            if gdf_municipios.crs != "EPSG:4326":
                                gdf_municipios = gdf_municipios.to_crs("EPSG:4326")
                        except: pass

                        # 1. DATOS ESTACIONES
                        df_anios_count = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.YEAR_COL].nunique().reset_index(name='n_anios')
                        gdf_calc = gdf_calc.merge(df_anios_count, on=Config.STATION_NAME_COL, how='left').fillna(0)

                        # A. INTERPOLACIÓN
                        dem_safe = np.nan_to_num(dem_array, nan=0)
                        Z_P, Z_Err = physics.interpolar_variable(
                            gdf_calc, 'ppt_media', grid_x, grid_y, method=metodo, dem_array=dem_safe
                        )
                        if metodo == 'ked': Z_P = np.maximum(Z_P, 0)

                        # B. MOTOR FÍSICO
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
                        st.subheader("📊 Diagnóstico Hidrológico Integral")
                        
                        try:
                            # --- 1. CÁLCULOS GEOMÉTRICOS ---
                            gdf_zona_proj = gdf_zona.to_crs(epsg=3116) # Proyectar a metros (Magna Sirgas)
                            area_km2 = gdf_zona_proj.area.sum() / 1e6
                            perim_km = gdf_zona_proj.length.sum() / 1e3
                            
                            # Índice de Forma (Compacidad de Gravelius): Kc = 0.28 * P / sqrt(A)
                            # Kc = 1.0 (Círculo perfecto), > 1.0 (Alargada)
                            ind_gravelius = (0.282 * perim_km) / (np.sqrt(area_km2)) if area_km2 > 0 else 0

                            # --- 2. EXTRACCIÓN DE VARIABLES DEL MODELO ---
                            from shapely.vectorized import contains
                            mask_exact = contains(gdf_zona.unary_union, grid_x, grid_y)
                            
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
                            
                        except Exception as e:
                            st.warning(f"Cálculos parciales: {e}")

                    except Exception as e:
                        st.error(f"Error crítico: {e}")


    # --- OTROS MÓDULOS ---
    elif selected_module == "🧪 Sesgo": viz.display_bias_correction_tab(**display_args)
    elif selected_module == "🌿 Cobertura": viz.display_land_cover_analysis_tab(**display_args)

    # --- ZONAS DE VIDA (CONECTADO A SUPABASE) ---
    elif selected_module == "🌱 Zonas Vida":
        # 1. Aseguramos tener los mapas en memoria
        try:
            # Usamos la función de carga que definimos antes (cargar_raster_db)
            dem_bytes = cargar_raster_db("DemAntioquia_EPSG3116.tif")
            pp_bytes = cargar_raster_db("PPAMAnt.tif")
            
            if dem_bytes and pp_bytes:
                # 2. Inyectamos los archivos en memoria a los argumentos
                # Esto sobreescribe cualquier ruta de texto vieja que tuviera display_args
                display_args['dem_file'] = dem_bytes
                display_args['ppt_file'] = pp_bytes
                
                # 3. Llamamos al visualizador
                viz.display_life_zones_tab(**display_args)
            else:
                st.error("⚠️ No se encontraron los mapas 'DemAntioquia_EPSG3116.tif' o 'PPAMAnt.tif' en Supabase.")
                st.info("Ve al Panel de Administración -> Pestaña Mapas y súbelos.")
        except NameError:
             st.error("Falta definir la función 'cargar_raster_db' al inicio del archivo.")
    
    elif selected_module == "🌡️ Clima Futuro": viz.display_climate_scenarios_tab(**display_args)
    
    # --- ISOYETAS HD (Tu código original preservado) ---
    elif selected_module == "✨ Mapas Isoyetas HD":
        st.header("🗺️ Isoyetas Alta Definición (RBF)")
        col1, col2 = st.columns([1,3])
        year_iso = col1.selectbox("Año:", range(int(year_range[1]), int(year_range[0])-1, -1))
        suavidad = col1.slider("Suavizado:", 0.0, 2.0, 0.5)
        
        ids_validos = tuple(gdf_filtered['id_estacion'].unique())
        
        if len(ids_validos) > 2:
            try:
                # --- CORRECCIÓN AQUÍ: Importamos explícitamente ---
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
                    
                    minx, miny, maxx, maxy = gdf_filtered.total_bounds
                    # ------------------------------------------

                    # Ahora sí podemos crear el grid
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
            
    elif selected_module == "📄 Reporte":
        st.header("Generación de Informe")
        if st.button("📄 Crear PDF"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf: st.download_button("Descargar PDF", pdf, "reporte_hidro.pdf", "application/pdf")

    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":

    main()




















