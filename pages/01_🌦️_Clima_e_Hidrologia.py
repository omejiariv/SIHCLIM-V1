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
    
    # --- MÓDULO: MAPAS AVANZADOS (CON REPROYECCIÓN FORZADA) ---
    elif selected_module == "🌍 Mapas Avanzados":
        
        # --- FUNCIÓN DE AUXILIO LOCAL: REPROYECCIÓN SEGURA Y BLINDADA ---
        def local_warper_force_4326(tif_path, bounds_wgs84, shape_out):
            """
            Fuerza la lectura del Raster transformándolo a WGS84 (EPSG:4326).
            Versión Universal (Limpia variables de entorno conflictivas).
            """
            import os
            import rasterio
            from rasterio.warp import reproject, Resampling, calculate_default_transform
            
            # --- LIMPIEZA DE CONFLICTOS PROJ ---
            # En la nube, a veces es mejor NO setear nada y dejar que rasterio se autoconfigure.
            # Solo si estamos en Windows forzamos pyproj.
            if os.name == 'nt': # Si es Windows
                try:
                    import pyproj
                    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
                except: pass
            # -----------------------------------

            with rasterio.open(tif_path) as src:
                # ... (El resto del código sigue IDÉNTICO a como lo tenías) ...
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
                )
                
                minx, miny, maxx, maxy = bounds_wgs84
                
                dst_transform = rasterio.transform.from_bounds(
                    minx, miny, maxx, maxy, shape_out[0], shape_out[1]
                )
                
                destination = np.zeros(shape_out, dtype=np.float32)
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear
                )
                
                destination[destination < -1000] = 0
                return destination

        if not PHYSICS_AVAILABLE:
            st.error("❌ Módulos 'hydro_physics' o 'admin_utils' no disponibles.")
        else:
            st.header("🌍 Modelación Hidrológica Distribuida (Aleph)")
            
            # --- 0. DIAGNÓSTICO RÁPIDO ---
            dem_path = None
            cov_path = None
            
            if os.path.exists(Config.DEM_FILE_PATH):
                dem_path = Config.DEM_FILE_PATH
            else:
                try: dem_path = download_raster_to_temp(os.path.basename(Config.DEM_FILE_PATH))
                except: pass
                
            if os.path.exists(Config.LAND_COVER_RASTER_PATH):
                cov_path = Config.LAND_COVER_RASTER_PATH
            else:
                try: cov_path = download_raster_to_temp("Cob25m_WGS84.tif")
                except: pass

            if not dem_path:
                st.error("⛔ Falta el archivo DEM. Verifique la carpeta 'data/'.")
                st.stop()

            # 1. Configuración del Grid
            c1, c2 = st.columns(2)
            buffer_km = c1.slider("Buffer (km)", 0.0, 50.0, 20.0)
            grid_res = c2.slider("Resolución Grid", 50, 300, 100)
            
            # 2. Geometría y Grid
            if gdf_zona is None: gdf_zona = gdf_filtered
            
            buffer_deg = buffer_km / 111.0
            gdf_buffer = gdf_zona.buffer(buffer_deg) if buffer_km > 0 else gdf_zona
            minx, miny, maxx, maxy = gdf_buffer.total_bounds
            
            xi = np.linspace(minx, maxx, grid_res)
            yi = np.linspace(miny, maxy, grid_res)
            grid_x, grid_y = np.meshgrid(xi, yi)
            bounds_calc = (minx, miny, maxx, maxy)
            
            # 3. Datos Estaciones
            df_mean = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index(name='ppt_media')
            gdf_calc = gdf_filtered.merge(df_mean, on=Config.STATION_NAME_COL)
            gdf_calc = gdf_calc.dropna(subset=['ppt_media', 'geometry'])
            gdf_calc['ppt_media'] = pd.to_numeric(gdf_calc['ppt_media'], errors='coerce').fillna(0)

            # 4. Procesamiento DEM (Usando el warper local seguro)
            dem_array = None
            with st.spinner("🏔️ Procesando topografía (Reproyección)..."):
                try:
                    # Usamos la función LOCAL que acabamos de definir arriba
                    dem_array = local_warper_force_4326(dem_path, bounds_calc, grid_x.shape)
                except Exception as e:
                    st.error(f"Error procesando DEM: {e}")
                    st.stop()
            
            # 5. Ejecución
            metodo = st.selectbox("Método Interpolación", ['kriging', 'idw', 'spline', 'ked'] if dem_array is not None else ['kriging', 'idw', 'spline'])
            
            if st.button("🚀 Ejecutar Modelo"):
                st.session_state['ejecutar_aleph'] = True
            
            # B. Bloque Persistente
            if st.session_state.get('ejecutar_aleph', False):
                
                # Botón de cierre
                col_close = st.columns([6, 1])[1]
                if col_close.button("❌ Cerrar Mapa"):
                    st.session_state['ejecutar_aleph'] = False
                    st.rerun()

                with st.spinner("Calculando física distribuida..."):
                    try:
                        # 1. INTERPOLACIÓN LLUVIA (La joya de la corona)
                        Z_P, Z_Err = physics.interpolar_variable(
                            gdf_calc, 'ppt_media', grid_x, grid_y, method=metodo, dem_array=dem_array
                        )
                        
                        # 2. PROCESAMIENTO DE COBERTURA (Local y Seguro)
                        cov_array = None
                        if cov_path and os.path.exists(cov_path):
                            try:
                                cov_array = local_warper_force_4326(cov_path, bounds_calc, grid_x.shape)
                            except: pass

                        # 3. EJECUCIÓN DEL MODELO FÍSICO
                        # (Calcula escorrentía, infiltración, etc. internamente)
                        matrices_raw = physics.run_distributed_model(
                            Z_P, grid_x, grid_y, {'dem': dem_path, 'cobertura': cov_path}, bounds_calc
                        )
                        
                        # --- 4. LA ADUANA (Limpieza de Duplicados y Nombres) ---
                        matrices_clean = {}

                        # A. Mapas Base (Priorizamos los nuestros que sabemos que se ven bien)
                        matrices_clean['1. Precipitación (mm)'] = Z_P
                        
                        if dem_array is not None:
                            matrices_clean['2. Elevación (msnm)'] = dem_array
                        
                        if cov_array is not None:
                            matrices_clean['3. Cobertura de Suelo (Clase)'] = cov_array

                        # B. Resultados del Modelo (Los rescatamos del output raw)
                        # Buscamos nombres probables que devuelve hydro_physics y los estandarizamos
                        keys_raw = matrices_raw.keys()
                        
                        # Escorrentía (Runoff)
                        if 'Escorrentía Superficial' in keys_raw:
                            matrices_clean['4. Escorrentía Superficial (mm)'] = matrices_raw['Escorrentía Superficial']
                        elif 'Q_Sup' in keys_raw:
                            matrices_clean['4. Escorrentía Superficial (mm)'] = matrices_raw['Q_Sup']

                        # Infiltración / Recarga
                        if 'Recarga Potencial' in keys_raw:
                            matrices_clean['5. Recarga Potencial (mm)'] = matrices_raw['Recarga Potencial']
                        
                        # Rendimiento Hídrico
                        if 'Rendimiento Hídrico' in keys_raw:
                            matrices_clean['6. Rendimiento Hídrico (L/s/km2)'] = matrices_raw['Rendimiento Hídrico']

                        # Inyectar Error si existe
                        if Z_Err is not None:
                            matrices_clean['7. Incertidumbre (Kriging Std)'] = Z_Err

                        # --- 5. VISUALIZACIÓN ---
                        # Pasamos 'matrices_clean' en lugar de 'matrices'
                        
                        # Preparamos Predios
                        gdf_predios_safe = None
                        if gdf_predios is not None and not gdf_predios.empty:
                            gdf_predios_safe = gdf_predios[gdf_predios.geometry.notnull()].copy()
                            if 'nombre' in gdf_predios_safe.columns:
                                gdf_predios_safe['nombre_predio'] = gdf_predios_safe['nombre']
                            if gdf_predios_safe.empty: gdf_predios_safe = None
    
                        viz.display_advanced_maps_tab(
                            df_long=df_monthly_filtered,
                            gdf_stations=gdf_calc, 
                            matrices=matrices_clean,  # <--- AQUÍ ESTÁ LA CLAVE
                            grid=(grid_x, grid_y),
                            mask=None, 
                            gdf_zona=gdf_zona, 
                            gdf_buffer=gdf_buffer, 
                            gdf_predios=gdf_predios_safe 
                        )
                        
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


