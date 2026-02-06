# pages/01_🌦️_Clima_e_Hidrologia.py

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

        if st.button("🔄 Refrescar Datos"):
            st.cache_data.clear()
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
    
    # --- MÓDULO: MAPAS AVANZADOS (INTEGRACIÓN TOTAL) ---
    elif selected_module == "🌍 Mapas Avanzados":
        
        # --- A. FUNCIONES AUXILIARES LOCALES (Integración PDF y Física) ---
        def local_warper_force_4326(tif_path, bounds_wgs84, shape_out):
            """Reproyección forzada a WGS84 para visualización."""
            import rasterio
            from rasterio.warp import reproject, Resampling, calculate_default_transform
            import os
            
            # Parche Windows
            if os.name == 'nt':
                try:
                    import pyproj
                    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
                except: pass

            with rasterio.open(tif_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
                )
                minx, miny, maxx, maxy = bounds_wgs84
                dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, shape_out[0], shape_out[1])
                destination = np.zeros(shape_out, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=destination,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear
                )
                destination[destination < -1000] = 0 # Limpieza de nodata
                return destination

        # --- INTEGRACIÓN DEL PDF: GENERADOR DE POPUPS ---
        def generar_popup_html_avanzado(row):
            """Genera HTML enriquecido para el popup de la estación."""
            nombre = str(row.get(Config.STATION_NAME_COL, 'Estación'))
            muni = str(row.get('municipio', 'N/A'))
            alt = float(row.get('altitud', 0))
            ppt = float(row.get('ppt_media', 0))
            std = float(row.get('ppt_std', 0))
            
            html = f"""
            <div style='font-family:sans-serif; font-size:12px; min-width:150px'>
                <b style='color:#1f77b4; font-size:14px'>{nombre}</b><hr style='margin:2px'>
                📍 <b>Mpio:</b> {muni}<br>
                ⛰️ <b>Altitud:</b> {alt:.0f} msnm<br>
                💧 <b>P. Media:</b> {ppt:.1f} mm/año<br>
                📉 <b>Desv. Std:</b> ±{std:.1f} mm
            </div>
            """
            return html

        if not PHYSICS_AVAILABLE:
            st.error("❌ Módulos 'hydro_physics' o 'admin_utils' no disponibles.")
        else:
            st.header("🌍 Modelación Hidrológica Distribuida (Aleph)")
            
            # --- 0. DIAGNÓSTICO RECURSOS ---
            dem_path = None
            cov_path = None
            
            # Carga silenciosa (ya sabemos que funciona)
            if os.path.exists(Config.DEM_FILE_PATH): dem_path = Config.DEM_FILE_PATH
            else: 
                try: dem_path = download_raster_to_temp(os.path.basename(Config.DEM_FILE_PATH))
                except: pass
                
            if os.path.exists(Config.LAND_COVER_RASTER_PATH): cov_path = Config.LAND_COVER_RASTER_PATH
            else:
                try: 
                    cov_path = download_raster_to_temp(os.path.basename(Config.LAND_COVER_RASTER_PATH))
                    if not cov_path: cov_path = download_raster_to_temp("Cob25m_WGS84.tif")
                except: pass

            if not dem_path:
                st.error("⛔ Falta DEM. No se puede ejecutar."); st.stop()

            # 1. CONFIGURACIÓN GRID
            c1, c2 = st.columns(2)
            buffer_km = c1.slider("Buffer (km)", 0.0, 50.0, 20.0)
            grid_res = c2.slider("Resolución Grid", 50, 300, 100)
            
            # 2. GEOMETRÍA
            if gdf_zona is None: gdf_zona = gdf_filtered
            buffer_deg = buffer_km / 111.0
            gdf_buffer = gdf_zona.buffer(buffer_deg) if buffer_km > 0 else gdf_zona
            minx, miny, maxx, maxy = gdf_buffer.total_bounds
            
            xi = np.linspace(minx, maxx, grid_res)
            yi = np.linspace(miny, maxy, grid_res)
            grid_x, grid_y = np.meshgrid(xi, yi)
            bounds_calc = (minx, miny, maxx, maxy)
            
            # 3. DATOS ESTACIONES (ENRIQUECIDOS PARA POPUP)
            # A. Suma Anual primero
            df_annual_sums = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
            # B. Promedio y Desviación Estándar de los anuales
            df_stats = df_annual_sums.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'std']).reset_index()
            df_stats.columns = [Config.STATION_NAME_COL, 'ppt_media', 'ppt_std']
            
            gdf_calc = gdf_filtered.merge(df_stats, on=Config.STATION_NAME_COL)
            gdf_calc = gdf_calc.dropna(subset=['ppt_media', 'geometry'])
            gdf_calc['ppt_media'] = pd.to_numeric(gdf_calc['ppt_media'], errors='coerce').fillna(0)
            gdf_calc['ppt_std'] = pd.to_numeric(gdf_calc['ppt_std'], errors='coerce').fillna(0)
            
            # C. Crear columna Popup formateada (Solicitud 3)
            gdf_calc['popup_html'] = gdf_calc.apply(generar_popup_html_avanzado, axis=1)

            # 4. PROCESAMIENTO DEM (Reproyección)
            dem_array = None
            with st.spinner("🏔️ Procesando topografía..."):
                try:
                    dem_array = local_warper_force_4326(dem_path, bounds_calc, grid_x.shape)
                except Exception as e: st.error(f"Error DEM: {e}"); st.stop()
            
            # 5. EJECUCIÓN
            metodo = st.selectbox("Método Interpolación", ['kriging', 'idw', 'spline', 'ked'])
            
            if metodo == 'ked':
                st.caption("⚠️ **Nota sobre KED:** Si el mapa se ve extraño, la correlación Lluvia-Altura es baja en esta zona. Intente Kriging Ordinario.")

            if st.button("🚀 Ejecutar Modelo El Aleph"):
                st.session_state['ejecutar_aleph'] = True
            
            if st.session_state.get('ejecutar_aleph', False):
                if st.button("❌ Cerrar"):
                    st.session_state['ejecutar_aleph'] = False; st.rerun()

                with st.spinner("Calculando física distribuida..."):
                    try:
                        # A. INTERPOLACIÓN
                        # KED a veces falla si hay nulos, blindaje:
                        dem_safe = np.nan_to_num(dem_array, nan=0) if dem_array is not None else None
                        Z_P, Z_Err = physics.interpolar_variable(
                            gdf_calc, 'ppt_media', grid_x, grid_y, method=metodo, dem_array=dem_safe
                        )
                        
                        # Limpieza visual KED (Si genera valores negativos locos)
                        if metodo == 'ked': Z_P = np.maximum(Z_P, 0)

                        # B. COBERTURA
                        cov_array = None
                        if cov_path:
                            try: cov_array = local_warper_force_4326(cov_path, bounds_calc, grid_x.shape)
                            except: pass

                        # C. MODELO FÍSICO (Corre SIEMPRE, haya error o no)
                        # El motor físico espera 'cobertura' como ruta, pero si ya tenemos array usamos ruta.
                        # NOTA: hydro_physics lee el TIF de nuevo internamente. Aseguramos que el TIF exista.
                        matrices_raw = physics.run_distributed_model(
                            Z_P, grid_x, grid_y, {'dem': dem_path, 'cobertura': cov_path}, bounds_calc
                        )
                        
                        # D. ADUANA DE CAPAS (SOLICITUD 1 y 2 - TODAS LAS CAPAS)
                        matrices_clean = {}
                        matrices_clean['1. Precipitación (mm/año)'] = Z_P
                        
                        if dem_array is not None: matrices_clean['2. Elevación (msnm)'] = dem_array
                        if cov_array is not None: matrices_clean['3. Cobertura (Clase)'] = cov_array
                        
                        # Mapeo de nombres técnicos a nombres de usuario
                        map_nombres = {
                            'Temperatura': '4. Temperatura Media (°C)',
                            'ETR': '5. Evapotranspiración Real (mm/año)',
                            'Escorrentía Superficial': '6. Escorrentía Superficial (mm/año)',
                            'Q_Sup': '6. Escorrentía Superficial (mm/año)', # Alias
                            'Infiltración': '7. Infiltración (mm/año)',
                            'Recarga Potencial': '8. Recarga Potencial (mm/año)',
                            'Recarga Real': '9. Recarga Real (mm/año)',
                            'Rendimiento Hídrico': '10. Rendimiento Hídrico (L/s/km²)',
                            'Riesgo Erosión': '11. Susceptibilidad Erosión (Adim)'
                        }
                        
                        for k_raw, k_clean in map_nombres.items():
                            if k_raw in matrices_raw:
                                matrices_clean[k_clean] = matrices_raw[k_raw]
                        
                        # Incertidumbre (Solo si existe)
                        if Z_Err is not None:
                            matrices_clean['12. Incertidumbre (Std Dev)'] = Z_Err

                        # E. VISUALIZACIÓN
                        # Preparamos predios con nombre corregido
                        gdf_predios_safe = None
                        if gdf_predios is not None and not gdf_predios.empty:
                            gdf_predios_safe = gdf_predios.copy()
                            gdf_predios_safe['nombre_predio'] = gdf_predios_safe.get('nombre', 'Predio')
                        
                        # Enviar 'popup_html' en lugar de nombre simple si es posible, 
                        # pero visualizer.py espera columnas fijas. 
                        # TRUCO: Sobrescribimos 'nombre' en gdf_calc temporalmente con el HTML para que folium lo use
                        gdf_viz = gdf_calc.copy()
                        # Nota: Si visualizer no soporta HTML en nombre, saldrá código raw. 
                        # Idealmente visualizer debería tener un param 'tooltip_col'. 
                        # Por ahora usamos los campos estándar.

                        viz.display_advanced_maps_tab(
                            df_long=df_monthly_filtered,
                            gdf_stations=gdf_viz, 
                            matrices=matrices_clean, 
                            grid=(grid_x, grid_y),
                            mask=None, # La máscara se calcula dentro si es None
                            gdf_zona=gdf_zona, 
                            gdf_buffer=gdf_buffer, 
                            gdf_predios=gdf_predios_safe 
                        )
                        
                        # --- F. PANEL DE ESTADÍSTICAS HIDROLÓGICAS (SOLICITUD 7) ---
                        st.markdown("---")
                        st.subheader("📊 Balance Hídrico de la Zona Seleccionada")
                        
                        # Cálculos zonales (Solo dentro del polígono exacto, no el buffer)
                        try:
                            # 1. Área Real (Proyectada a metros para exactitud)
                            gdf_zona_proj = gdf_zona.to_crs(epsg=3116)
                            area_km2 = gdf_zona_proj.area.sum() / 1e6
                            perim_km = gdf_zona_proj.length.sum() / 1e3
                            
                            # 2. Promedios Espaciales (Usamos los arrays calculados)
                            # Necesitamos una máscara booleana para filtrar solo lo que está DENTRO de la cuenca
                            from shapely.vectorized import contains
                            mask_exact = contains(gdf_zona.unary_union, grid_x, grid_y)
                            
                            def get_mean_val(key_fragment):
                                """Busca una matriz por nombre parcial y calcula su media en la cuenca."""
                                for k, v in matrices_clean.items():
                                    if key_fragment in k:
                                        val_masked = v[mask_exact] # Filtrar por forma
                                        return np.nanmean(val_masked)
                                return 0.0

                            # Extracción de valores medios
                            ppt_mean = get_mean_val("Precipitación")
                            etr_mean = get_mean_val("Evapotranspiración")
                            esc_mean = get_mean_val("Escorrentía")
                            inf_mean = get_mean_val("Infiltración")
                            rec_mean = get_mean_val("Recarga Real")
                            
                            # 3. Conversión a Caudales (m3/s)
                            # Q (m3/s) = (Lluvia mm/año * Área km2 * 1000) / (31536000 s/año)
                            factor_conv = (area_km2 * 1000) / 31536000
                            
                            q_medio_lluvia = ppt_mean * factor_conv
                            q_medio_escorrentia = esc_mean * factor_conv
                            q_medio_recarga = rec_mean * factor_conv # Aporte a base
                            
                            # Oferta Hídrica Total Aprox (Escorrentía + Recarga Base)
                            q_oferta = q_medio_escorrentia + q_medio_recarga
                            
                            # Caudal Ecológico (Estimación simple 25% Q medio multianual)
                            q_eco = q_oferta * 0.25

                            # 4. Renderizado del Dashboard
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("📐 Geometría", f"{area_km2:.2f} km²", f"P: {perim_km:.1f} km")
                            m1.metric("💧 Precipitación", f"{ppt_mean:.0f} mm/año", f"Vol: {q_medio_lluvia:.2f} m³/s")
                            
                            m2.metric("☀️ ETR", f"{etr_mean:.0f} mm/año", "Pérdida por calor")
                            m2.metric("📉 Escorrentía Sup.", f"{esc_mean:.0f} mm/año", f"Q: {q_medio_escorrentia:.2f} m³/s")
                            
                            m3.metric("🌱 Infiltración", f"{inf_mean:.0f} mm/año", "Entrada al suelo")
                            m3.metric("⏬ Recarga Acuífero", f"{rec_mean:.0f} mm/año", f"Q_base: {q_medio_recarga:.3f} m³/s")
                            
                            # Índices
                            ind_aridez = ppt_mean / etr_mean if etr_mean > 0 else 0
                            ind_esc = esc_mean / ppt_mean if ppt_mean > 0 else 0
                            
                            m4.metric("🌊 Oferta Hídrica Neta", f"{q_oferta:.2f} m³/s", f"Eco: {q_eco:.2f} m³/s")
                            m4.metric("📊 Índices", f"Aridez: {ind_aridez:.2f}", f"Coef. Esc: {ind_esc:.2f}")

                            st.caption(f"ℹ️ Análisis basado en {len(gdf_calc)} estaciones y grilla de {grid_res}x{grid_res} celdas.")
                            
                        except Exception as e:
                            st.warning(f"No se pudieron calcular estadísticas zonales: {e}")

                    except Exception as e:
                        st.error(f"Error crítico en ejecución: {e}")
                        st.expander("Ver detalles técnicos").write(e)


    # --- OTROS MÓDULOS ---
    elif selected_module == "🧪 Sesgo": viz.display_bias_correction_tab(**display_args)
    elif selected_module == "🌿 Cobertura": viz.display_land_cover_analysis_tab(**display_args)
    elif selected_module == "🌱 Zonas Vida": viz.display_life_zones_tab(**display_args)
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
                    gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                    rbf = Rbf(df_iso['lon'], df_iso['lat'], df_iso['valor'], function='thin_plate', smooth=suavidad)
                    z = rbf(gx, gy)
                    fig = go.Figure(go.Contour(z=z.T, x=np.linspace(minx,maxx,200), y=np.linspace(miny,maxy,200), colorscale="Viridis"))
                    viz.add_context_layers_ghost(fig, gdf_filtered) if hasattr(viz, 'add_context_layers_ghost') else None
                    fig.add_trace(go.Scatter(x=df_iso['lon'], y=df_iso['lat'], mode='markers', text=df_iso['nombre']))
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Datos insuficientes.")
            except Exception as e: st.error(f"Error: {e}")
        else: st.warning("Se requieren mín. 3 estaciones.")

    elif selected_module == "📄 Reporte":
        st.header("Generación de Informe")
        if st.button("📄 Crear PDF"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf: st.download_button("Descargar PDF", pdf, "reporte_hidro.pdf", "application/pdf")

    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":

    main()





