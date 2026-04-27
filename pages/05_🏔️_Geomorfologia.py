# modules/05_🏔️_Geomorfologia.py

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import features
from rasterio.io import MemoryFile
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from shapely.geometry import shape, LineString, MultiLineString, Polygon

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA ---
# Nota: set_page_config SIEMPRE debe ser el primer comando de Streamlit
st.set_page_config(page_title="Geomorfología Pro", page_icon="🏔️", layout="wide")

# Aseguramos que python encuentre los módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 🚀 IMPORTACIONES CLOUD NATIVE Y MÓDULOS ---
from modules import selectors
from modules.config import Config
from modules.hydro_physics import download_raster_secure

try:
    from modules.db_manager import get_engine
except ImportError:
    from db_manager import get_engine

# Intentamos importar pysheds
try:
    from pysheds.grid import Grid
    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False

try:
    from modules import land_cover
except ImportError:
    land_cover = None

# Inicializar conexión a Base de Datos
engine = get_engine()

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Geomorfología")

# --- INICIALIZACIÓN DE VARIABLES DE ESTADO ---
if 'gdf_contours' not in st.session_state: st.session_state['gdf_contours'] = None
if 'catchment_raster' not in st.session_state: st.session_state['catchment_raster'] = None
if 'gdf_rios' not in st.session_state: st.session_state['gdf_rios'] = None     
if 'df_indices' not in st.session_state: st.session_state['df_indices'] = None 
    
st.title("🏔️ Análisis Geomorfológico y Terreno 3D")
st.markdown("""
Esta herramienta utiliza el **Modelo Digital de Elevación (DEM)** para modelar el terreno, 
calcular pendientes, extraer vectores de drenaje y realizar diagnósticos hidrológicos automáticos.
""")

# --- 1. BARRA LATERAL (SELECTOR) ---
ids, nombre_zona, alt_ref, gdf_zona_seleccionada = selectors.render_selector_espacial()

# Corrección Geometría (Puntos -> Caja)
if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
    if gdf_zona_seleccionada.geom_type.isin(['Point', 'MultiPoint']).any():
        if len(gdf_zona_seleccionada) == 1:
            gdf_zona_seleccionada['geometry'] = gdf_zona_seleccionada.buffer(0.045) 
        else:
            bbox = gdf_zona_seleccionada.unary_union.envelope
            gdf_zona_seleccionada = gpd.GeoDataFrame({'geometry': [bbox]}, crs=gdf_zona_seleccionada.crs)

# --- 2. CARGA DEL DEM (CONECTADO A LA NUBE) ---
DEM_PATH = Config.DEM_FILE_PATH

@st.cache_data(show_spinner="Descargando y procesando terreno desde la Nube...")
def cargar_y_cortar_dem(ruta_dem, _gdf_corte, zona_id):
    if _gdf_corte is None or _gdf_corte.empty: 
        return None, None, None
    
    # Descarga Segura desde Supabase
    safe_path = download_raster_secure(ruta_dem)
    if not safe_path: 
        return None, None, None
            
    try:
        # 1. Blindaje de Geometría (Repara polígonos inválidos)
        geometria_valida = _gdf_corte.copy()
        geometria_valida['geometry'] = geometria_valida.buffer(0) 

        with rasterio.open(safe_path) as src:
            crs_dem = src.crs
            gdf_proyectado = geometria_valida.to_crs(crs_dem)
            geoms = gdf_proyectado.geometry.values
            
            try:
                out_image, out_transform = mask(src, geoms, crop=True)
            except ValueError:
                return None, "OUT_OF_BOUNDS", None
                
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": 1
            })
            
            dem_array = out_image[0]
            # Limpieza de datos NoData y valores erróneos profundos
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            dem_array = np.where(dem_array < -100, np.nan, dem_array)
            
            if np.isnan(dem_array).all(): 
                return None, "EMPTY_DATA", None
                
            return dem_array, out_meta, out_transform
            
    except Exception as e:
        st.error(f"Error procesando DEM: {e}")
        return None, None, None

# --- CEREBRO DEL ANALISTA 🧠 ---
def analista_hidrologico(pendiente_media, hi_value):
    if pendiente_media > 25:
        txt_pend = "un relieve fuertemente escarpado"
        riesgo = "alto potencial de flujos torrenciales y respuesta rápida"
    elif pendiente_media > 12:
        txt_pend = "un relieve moderadamente ondulado"
        riesgo = "velocidades de flujo moderadas"
    else:
        txt_pend = "un relieve predominantemente plano"
        riesgo = "propensión al encharcamiento y flujos lentos"

    if hi_value > 0.50:
        tipo = "Cuenca Joven (En Desequilibrio)"
        txt_hi = "fase activa de erosión (Juventud)"
    elif hi_value < 0.35:
        tipo = "Cuenca Vieja (Senil)"
        txt_hi = "fase avanzada de sedimentación (Senectud)"
    else:
        tipo = "Cuenca Madura"
        txt_hi = "equilibrio dinámico"

    return f"""
    **Diagnóstico del Analista:**
    La zona presenta **{txt_pend}** (Pendiente media: {pendiente_media:.1f}°), sugiriendo {riesgo}.
    
    Evolutivamente, es una **{tipo}** (HI: {hi_value:.3f}), indicando una {txt_hi}.
    """

# --- FUNCIÓN DE VECTORIZACIÓN DE RÍOS 🌊 ---
@st.cache_data(show_spinner="Trazando red de drenaje...")
def extraer_vectores_rios(_grid, _fdir, _acc, umbral, _crs_in, cache_id):
    try:
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        branches = _grid.extract_river_network(_fdir, _acc > umbral, dirmap=dirmap)
        if not branches or not branches['features']: return None
        gdf = gpd.GeoDataFrame.from_features(branches['features'])
        if gdf.crs is None: gdf.set_crs(_crs_in, inplace=True)
        return gdf
    except Exception:
        return None
        
# --- FUNCIONES DE DESCARGA ---
def to_tif(arr, meta):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(arr.astype(rasterio.float32), 1)
        return memfile.read()

# --- LÓGICA PRINCIPAL ---
if gdf_zona_seleccionada is not None:
    if not DEM_PATH:
        st.error("⚠️ La ruta del DEM no está configurada en config.py")
    else:
        arr_elevacion, meta, transform = cargar_y_cortar_dem(DEM_PATH, gdf_zona_seleccionada, nombre_zona)
        
        if meta == "OUT_OF_BOUNDS":
            st.warning(f"⚠️ Zona fuera de cobertura del DEM actual.")
        elif meta == "EMPTY_DATA":
            st.warning(f"⚠️ Datos vacíos en el recorte.")
        elif arr_elevacion is not None and not np.isnan(arr_elevacion).all():
            
            # --- CÁLCULOS GLOBALES ---
            elevs_valid = arr_elevacion[~np.isnan(arr_elevacion)].flatten()
            min_el, max_el = np.min(elevs_valid), np.max(elevs_valid)
            mean_el = np.mean(elevs_valid)
            hi_global = (mean_el - min_el) / (max_el - min_el) if (max_el - min_el) > 0 else 0.5
            
            # Pendientes
            dy, dx = np.gradient(arr_elevacion, 30.0)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_mean = np.nanmean(slope_deg)
            
            texto_analisis = analista_hidrologico(slope_mean, hi_global)

            # =========================================================================
            # 🧠 MOTOR HIDROLÓGICO EN BACKGROUND
            # =========================================================================
            import tempfile
            
            grid = None; acc = None; fdir = None
            crs_actual = meta.get('crs', 'EPSG:3116')
            
            with st.spinner("Despertando motor hidrológico (PySheds)..."):
                if st.session_state.get('ultima_zona_procesada') != nombre_zona or st.session_state.get('gdf_rios') is None:
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                        try:
                            meta_t = meta.copy()
                            meta_t.update(driver='GTiff', dtype='float64', nodata=-9999.0)
                            dem_clean = np.where(np.isnan(arr_elevacion), -9999.0, arr_elevacion)
                            
                            with rasterio.open(tmp.name, 'w', **meta_t) as dst: 
                                dst.write(dem_clean.astype('float64'), 1)
                            
                            if PYSHEDS_AVAILABLE:
                                grid = Grid.from_raster(tmp.name)
                                dem_grid = grid.read_raster(tmp.name, nodata=-9999.0)
                                
                                flooded = grid.fill_depressions(dem_grid)
                                resolved = grid.resolve_flats(flooded)
                                
                                dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                                fdir = grid.flowdir(resolved, dirmap=dirmap)
                                acc = grid.accumulation(fdir, dirmap=dirmap)
                                
                                branches = grid.extract_river_network(fdir, acc > 100, dirmap=dirmap)
                                
                                if branches and len(branches["features"]) > 0:
                                    gdf_streams_raw = gpd.GeoDataFrame.from_features(branches["features"], crs=crs_actual)
                                    
                                    if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
                                        zona_crs = gdf_zona_seleccionada.to_crs(crs_actual)
                                        gdf_streams = gpd.clip(gdf_streams_raw, zona_crs)
                                    else:
                                        gdf_streams = gdf_streams_raw
                                        
                                    if not gdf_streams.empty:
                                        gdf_streams_m = gdf_streams.to_crs(epsg=3116)
                                        gdf_streams['longitud_km'] = gdf_streams_m.length / 1000.0
                                        st.session_state['gdf_rios'] = gdf_streams
                                    else:
                                        st.session_state['gdf_rios'] = None
                                else:
                                    st.session_state['gdf_rios'] = None
                                    
                            st.session_state['grid_obj'] = grid
                            st.session_state['acc_obj'] = acc
                            st.session_state['fdir_obj'] = fdir
                            st.session_state['ultima_zona_procesada'] = nombre_zona
                            
                        except Exception:
                            pass
                        finally:
                            try: os.remove(tmp.name)
                            except: pass

            grid = st.session_state.get('grid_obj')
            acc = st.session_state.get('acc_obj')
            fdir = st.session_state.get('fdir_obj')

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mínima", f"{min_el:.0f} m")
            c2.metric("Máxima", f"{max_el:.0f} m")
            c3.metric("Media", f"{mean_el:.0f} m")
            c4.metric("Rango", f"{max_el - min_el:.0f} m")

            tab1, tab2, tab3, tab4, tab6, tab7, tab5 = st.tabs([
                "🗺️ 3D", "📐 Pendientes", "📈 Hipsometría", 
                "🌊 Hidrología", "📊 Índices Morfo-métricos", "🚨 Amenazas", "📥 Descargas"
            ])
            
            h, w = arr_elevacion.shape
            factor = max(1, int(max(h, w) / 200)) 

            # --- TAB 1: 3D Y CURVAS ---
            with tab1:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown("#### Visualización")
                    exag = st.slider("Exageración Vertical:", 0.5, 5.0, 1.5, 0.1, key="z_exag")
                    st.markdown("---")
                    ver_curvas = st.toggle("Ver Curvas de Nivel", value=True)
                    intervalo_curvas = st.select_slider("Intervalo (m):", options=[10, 25, 50, 100], value=50)
                    
                with c2:
                    arr_3d = arr_elevacion[::factor, ::factor]
                    contours_conf = dict(z=dict(show=ver_curvas, start=np.nanmin(arr_elevacion), end=np.nanmax(arr_elevacion), size=intervalo_curvas, color="white", usecolormap=False, project_z=False))
                    fig = go.Figure(data=[go.Surface(z=arr_3d, colorscale='Earth', contours=contours_conf, name="Terreno")])

                    if ver_curvas:
                        try:
                            min_z, max_z = np.nanmin(arr_elevacion), np.nanmax(arr_elevacion)
                            levels = np.arange(np.floor(min_z), np.ceil(max_z), intervalo_curvas)
                            contours_obj = plt.contour(arr_elevacion, levels=levels)
                            geoms_2d = [] 
                            for level, collection in zip(levels, contours_obj.collections):
                                for path in collection.get_paths():
                                    v = path.vertices
                                    if len(v) < 2: continue 
                                    xs_geo, ys_geo = rasterio.transform.xy(transform, v[:, 0], v[:, 1])
                                    geoms_2d.append({'geometry': LineString(zip(xs_geo, ys_geo)), 'elevation': level})
                            plt.close() 
                            if geoms_2d: st.session_state['gdf_contours'] = gpd.GeoDataFrame(geoms_2d, crs=meta['crs'])
                        except: pass

                    fig.update_layout(title="Terreno 3D (Curvas Nativas)", autosize=True, height=900, scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2*exag), camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), margin=dict(l=0, r=0, b=0, t=40))
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # --- TAB 2: PENDIENTES ---
            with tab2:
                st.subheader("Mapa de Pendientes")
                st.info(texto_analisis, icon="🤖") 
                fig_slope = px.imshow(slope_deg[::factor, ::factor], color_continuous_scale='Turbo', labels={'color': 'Grados'}, title=f"Pendientes: {nombre_zona}")
                fig_slope.update_xaxes(fixedrange=False) 
                fig_slope.update_yaxes(fixedrange=False) 
                fig_slope.update_layout(height=700, dragmode='pan', hovermode='closest')
                st.plotly_chart(fig_slope, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

                slope_flat = slope_deg[~np.isnan(slope_deg)].flatten()
                bins = [0, 3, 7, 12, 25, 50, 90]
                labels = ['Plano', 'Suave', 'Inclinado', 'Ondulado', 'Escarpado', 'Muy Escarpado']
                cats = pd.cut(slope_flat, bins=bins, labels=labels)
                counts = cats.value_counts().sort_index()
                df_slopes = pd.DataFrame({'Categoría': counts.index, '%': (counts.values/len(slope_flat)*100).round(1)})
                st.dataframe(df_slopes.T, use_container_width=True)

            # --- TAB 3: HIPSOMETRÍA ---
            with tab3:
                elevs_sorted = np.sort(elevs_valid)[::-1]
                total_pixels = len(elevs_sorted)
                x_pct = np.linspace(0, 100, total_pixels)
                idx = np.linspace(0, total_pixels-1, 500, dtype=int)
                
                c_hip1, c_hip2 = st.columns(2)
                with c_hip1:
                    fig_hyp = go.Figure()
                    fig_hyp.add_trace(go.Scatter(x=x_pct[idx], y=elevs_sorted[idx], fill='tozeroy', name='Real', line=dict(color='#2E86C1')))
                    fig_hyp.update_layout(title="Curva Hipsométrica (Absoluta)", xaxis_title="% Área Acumulada", yaxis_title="Altitud (m.s.n.m)", height=450, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_hyp, use_container_width=True, config={'scrollZoom': True})

                with c_hip2:
                    h_min, h_max = np.min(elevs_sorted), np.max(elevs_sorted)
                    h_rel = (elevs_sorted[idx] - h_min) / (h_max - h_min)
                    a_rel = x_pct[idx] / 100.0 
                    
                    fig_adim = go.Figure()
                    fig_adim.add_trace(go.Scatter(x=a_rel, y=h_rel, name='Cuenca Actual', line=dict(color='#E74C3C', width=3)))
                    fig_adim.add_trace(go.Scatter(x=[0, 1], y=[1, 0], name='Equilibrio (Ref)', line=dict(color='gray', dash='dot')))
                    fig_adim.update_layout(title="Curva Adimensional (Ciclo de Erosión)", xaxis_title="Área Relativa (a/A)", yaxis_title="Altura Relativa (h/H)", height=450, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_adim, use_container_width=True, config={'scrollZoom': True})
                    
                st.info("**Interpretación Adimensional:**\n* **Curva Convexa:** Cuenca joven.\n* **Curva Concava:** Cuenca vieja.\n* **Forma de 'S':** Cuenca madura.")
                
                st.markdown("---")
                st.markdown("#### 🧮 Ecuación Hipsométrica $A(h)$")
                st.caption("Modelo matemático que relaciona la altitud ($h$) con el porcentaje de área acumulada ($A$).")
                
                z_poly = np.polyfit(elevs_sorted[idx], x_pct[idx], 3)
                col_eq1, col_eq2 = st.columns([2, 1])
                with col_eq1: st.latex(f"A(h) = {z_poly[0]:.3e} \\cdot h^3 {z_poly[1]:+.3e} \\cdot h^2 {z_poly[2]:+.3e} \\cdot h {z_poly[3]:+.3e}")
                
                with col_eq2:
                    p_func = np.poly1d(z_poly)
                    y_pred = p_func(elevs_sorted[idx])
                    y_real = x_pct[idx]
                    r2 = 1 - (np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2))
                    st.metric("Precisión del Ajuste ($R^2$)", f"{r2:.4f}")

                # =================================================================
                # 🚀 INYECCIÓN A LA MATRIZ MAESTRA HIPSOMÉTRICA (GEMELO DIGITAL)
                # =================================================================
                st.markdown("---")
                st.markdown("#### 💾 Forja de Ecuaciones (Matriz Maestra)")
                st.info("Guarda esta ecuación polinómica en SQL. El módulo de **Calidad y Vertimientos** usará estos coeficientes para calcular el caudal exacto del río en cualquier altitud.")
                
                if st.button("🚀 Inyectar Ecuación a Base de Datos", type="primary"):
                    try:
                        from sqlalchemy import text
                        from modules.utils import normalizar_texto
                        
                        # Crear llave universal (Mismo formato del Módulo 09)
                        nivel_llave = "MUNICIPAL" if "Municipio" in nombre_zona else "CUENCA" # Ajustar según tu selector
                        t_norm = normalizar_texto(nombre_zona).upper().replace(" ", "_")
                        llave_u = f"{nivel_llave}_{t_norm}_TOTAL"
                        
                        # Estructurar los datos
                        df_hipso_db = pd.DataFrame([{
                            "LLAVE_UNIVERSAL": llave_u,
                            "Territorio": nombre_zona,
                            "Nivel": nivel_llave,
                            "Coef_C3": float(z_poly[0]),
                            "Coef_C2": float(z_poly[1]),
                            "Coef_C1": float(z_poly[2]),
                            "Coef_C0": float(z_poly[3]),
                            "R2_Ajuste": float(r2),
                            "H_Minima": float(h_min),
                            "H_Maxima": float(h_max)
                        }])
                        
                        # Asegurar que la tabla existe y guardar
                        engine_sql = get_engine()
                        with engine_sql.connect() as conn:
                            conn.execute(text('''
                                CREATE TABLE IF NOT EXISTS matriz_maestra_hipsometrica (
                                    "LLAVE_UNIVERSAL" TEXT PRIMARY KEY,
                                    "Territorio" TEXT, "Nivel" TEXT,
                                    "Coef_C3" FLOAT, "Coef_C2" FLOAT, "Coef_C1" FLOAT, "Coef_C0" FLOAT,
                                    "R2_Ajuste" FLOAT, "H_Minima" FLOAT, "H_Maxima" FLOAT
                                );
                            '''))
                            # Upsert: Borrar si existe y volver a insertar
                            conn.execute(text(f"DELETE FROM matriz_maestra_hipsometrica WHERE \"LLAVE_UNIVERSAL\" = '{llave_u}'"))
                            
                        df_hipso_db.to_sql('matriz_maestra_hipsometrica', engine_sql, if_exists='append', index=False)
                        st.success(f"✅ ¡Ecuación A(h) para **{nombre_zona}** guardada exitosamente en el Gemelo Digital!")
                        
                    except Exception as e:
                        st.error(f"🚨 Error inyectando a SQL: {e}")
                    
            # --- TAB 4: HIDROLOGÍA ---
            with tab4:
                import math
                st.subheader(f"🌊 Hidrología: Red de Drenaje y Cuencas - {nombre_zona}")
                
                c_conf, c_map = st.columns([1, 3])
                with c_conf:
                    st.markdown("#### ⚙️ Configuración")
                    opciones = ["Vectores (Líneas)", "Raster (Acumulación)", "Catchment (Mascara)", "Divisoria (Línea)"]
                    modo_viz = st.radio("Visualización:", opciones)
                    umbral = st.slider("Umbral Acumulación", 5, 5000, 100, 5)

                with c_map:
                    if grid is not None and acc is not None and fdir is not None:
                        with st.spinner("Aplicando Leyes de Horton y extrayendo red..."):
                            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                            branches = grid.extract_river_network(fdir, acc > umbral, dirmap=dirmap)
                            
                            if branches and len(branches["features"]) > 0:
                                gdf_streams_raw = gpd.GeoDataFrame.from_features(branches["features"], crs=crs_actual)
                                
                                if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
                                    gdf_streams = gpd.clip(gdf_streams_raw, gdf_zona_seleccionada.to_crs(crs_actual))
                                else:
                                    gdf_streams = gdf_streams_raw
                                
                                if not gdf_streams.empty:
                                    gdf_streams_m = gdf_streams.to_crs(epsg=3116)
                                    gdf_streams['longitud_km'] = gdf_streams_m.length / 1000.0
                                    
                                    try:
                                        inv_affine = ~grid.affine
                                        orden_list = []
                                        Ra = 4.5 
                                        
                                        for geom in gdf_streams.geometry:
                                            acc_vals = []
                                            lineas = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
                                            
                                            for linea in lineas:
                                                if linea.is_empty: continue
                                                for p in list(linea.coords):
                                                    try:
                                                        c_f, r_f = inv_affine * p
                                                        c, r = int(round(c_f)), int(round(r_f))
                                                        rmin, rmax = max(0, r-1), min(acc.shape[0], r+2)
                                                        cmin, cmax = max(0, c-1), min(acc.shape[1], c+2)
                                                        window = acc[rmin:rmax, cmin:cmax]
                                                        if window.size > 0: acc_vals.append(np.max(window))
                                                    except: pass
                                            
                                            if acc_vals:
                                                acc_max = max(acc_vals)
                                                orden = int(math.floor(math.log(acc_max / umbral, Ra))) + 1 if acc_max >= umbral else 1
                                            else: orden = 1
                                            orden_list.append(max(1, orden))
                                        
                                        gdf_streams['Orden_Strahler'] = orden_list
                                    except Exception:
                                        gdf_streams['Orden_Strahler'] = 1 
                                    
                                    st.session_state['gdf_rios'] = gdf_streams
                                    st.session_state['geomorfo_strahler_df'] = gdf_streams.groupby('Orden_Strahler').agg(Num_Segmentos=('geometry', 'count'), Longitud_Km=('longitud_km', 'sum')).reset_index()
                                    st.success(f"✅ Red generada: {len(gdf_streams)} segmentos.")
                                else:
                                    st.session_state['gdf_rios'] = None
                                    st.session_state['geomorfo_strahler_df'] = None
                            else:
                                st.session_state['gdf_rios'] = None
                                st.session_state['geomorfo_strahler_df'] = None
                                
                    if grid is not None and acc is not None:
                        lat_c, lon_c = 0, 0
                        r_smart, c_smart = 0, 0
                        
                        if gdf_zona_seleccionada is not None:
                            cent = gdf_zona_seleccionada.to_crs("EPSG:4326").geometry.centroid.iloc[0]
                            lat_c, lon_c = cent.y, cent.x
                        
                        idx_max = np.nanargmax(acc)
                        r_smart, c_smart = np.unravel_index(idx_max, acc.shape)
                        try:
                            if gdf_zona_seleccionada is not None:
                                mask_poly = features.rasterize(((g, 1) for g in gdf_zona_seleccionada.to_crs(meta['crs']).geometry), out_shape=acc.shape, transform=transform, fill=0, dtype='uint8')
                                acc_masked = np.where(mask_poly==1, acc, -1)
                                r_smart, c_smart = np.unravel_index(np.argmax(acc_masked), acc_masked.shape)
                        except: pass

                        with st.expander(f"📍 Puntos Clave: {nombre_zona}", expanded=True):
                            k1, k2, k3 = st.columns(3)
                            with k1: st.markdown("**Centro**"); st.caption(f"{lat_c:.4f}, {lon_c:.4f}")
                            with k2: st.markdown("**Salida Detectada**"); st.code(f"X:{c_smart} Y:{r_smart}")
                            with k3:
                                if st.button("🎯 Usar Salida", type="primary"):
                                    st.session_state['x_pour_calib'] = int(c_smart)
                                    st.session_state['y_pour_calib'] = int(r_smart)
                                    st.rerun()

                        if modo_viz == "Raster (Acumulación)":
                            acc_viz = acc.copy()
                            try:
                                if gdf_zona_seleccionada is not None:
                                    mask_poly = features.rasterize(((g, 1) for g in gdf_zona_seleccionada.to_crs(meta['crs']).geometry), out_shape=acc.shape, transform=transform, fill=0, dtype='uint8')
                                    acc_viz = np.where(mask_poly == 1, acc_viz, np.nan)
                            except: pass

                            h, w = acc_viz.shape
                            factor = int(max(h, w) / 800) if h > 1000 or w > 1000 else 1
                            fig = px.imshow(np.log1p(acc_viz[::factor, ::factor]), color_continuous_scale='Blues', title=f"Acumulación de Flujo: {nombre_zona}")
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)

                        else:
                            fig = go.Figure()
                            if gdf_zona_seleccionada is not None:
                                poly = gdf_zona_seleccionada.to_crs("EPSG:4326").geometry.iloc[0]
                                xx, yy = poly.exterior.coords.xy if poly.geom_type=='Polygon' else max(poly.geoms, key=lambda a:a.area).exterior.coords.xy
                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xx), lat=list(yy), line={'width':2, 'color':'#00FF00'}, name="Oficial"))

                            if modo_viz == "Vectores (Líneas)":
                                r_viz = st.session_state.get('gdf_rios')
                                if r_viz is not None and not r_viz.empty:
                                    r_viz_4326 = r_viz.to_crs("EPSG:4326")
                                    colores_orden = {1: '#85C1E9', 2: '#3498DB', 3: '#2874A6', 4: '#1A5276', 5: '#0E2F44'}
                                    for orden in sorted(r_viz_4326['Orden_Strahler'].unique()):
                                        l, lt, tx = [], [], []
                                        for _, row in r_viz_4326[r_viz_4326['Orden_Strahler'] == orden].iterrows():
                                            g = row.geometry
                                            if g.geom_type == 'LineString': p = [g]
                                            elif g.geom_type == 'MultiLineString': p = g.geoms
                                            else: continue
                                            for s in p: 
                                                x, y = s.xy
                                                l.extend(list(x) + [None]); lt.extend(list(y) + [None])
                                                tx.extend([f"Río Orden {orden}"] * (len(x) + 1))
                                        fig.add_trace(go.Scattermapbox(mode="lines", lon=l, lat=lt, text=tx, hoverinfo='text', line={'width': 1 + (orden * 0.8), 'color': colores_orden.get(orden, '#0E2F44')}, name=f"Orden {orden}"))
                                    st.success(f"✅ Red generada exitosamente (Ley de Horton). Se detectaron hasta ríos de Orden {int(r_viz_4326['Orden_Strahler'].max())}.")
                                else: st.warning("No hay ríos visibles. Disminuya el umbral de acumulación.")
                                    
                            elif modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                                if 'x_pour_calib' not in st.session_state: 
                                    st.session_state['x_pour_calib']=int(c_smart); st.session_state['y_pour_calib']=int(r_smart)
                                st.markdown("##### 🔧 Ajuste Manual")
                                c1, c2 = st.columns(2)
                                with c1: xp=st.number_input("X:", value=st.session_state['x_pour_calib'])
                                with c2: yp=st.number_input("Y:", value=st.session_state['y_pour_calib'])
                                st.session_state['x_pour_calib']=xp; st.session_state['y_pour_calib']=yp
                                
                                try:
                                    catch = grid.catchment(x=xp, y=yp, fdir=fdir, dirmap=dirmap, xytype='index')
                                    geoms = [shape(g) for g, v in features.shapes(np.ascontiguousarray(catch, dtype=np.uint8), transform=transform) if v > 0]
                                    if geoms:
                                        gdf_c = gpd.GeoDataFrame({'geometry': geoms}, crs=crs_actual).dissolve().to_crs("EPSG:4326")
                                        if modo_viz=="Catchment (Mascara)": fig.add_trace(go.Choroplethmapbox(geojson=gdf_c.geometry.__geo_interface__, locations=gdf_c.index, z=[1]*len(gdf_c), colorscale=[[0,'#3366CC'],[1,'#3366CC']], showscale=False, marker_opacity=0.5, name="Calculada"))
                                        else: 
                                            xc, yc = gdf_c.geometry.iloc[0].exterior.coords.xy
                                            fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xc), lat=list(yc), line={'width':3, 'color':'red'}, name="Divisoria"))
                                        st.success(f"Área: {gdf_c.to_crs('EPSG:3116').area.sum()/1e6:.2f} km²")
                                except: pass

                            fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center={"lat": lat_c, "lon": lon_c}, height=600, margin=dict(l=0,r=0,t=0,b=0))
                            st.session_state['fig_mapa_hidro'] = fig 
                            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                    else: st.warning("Procesando...")
                        
            # --- TAB 6: ÍNDICES Y MODELACIÓN ---
            with tab6:
                st.subheader(f"📊 Panel Hidrológico: {nombre_zona}")
                try:
                    gdf_metric = gdf_zona_seleccionada.to_crs("EPSG:3116")
                    geom = gdf_metric.geometry.iloc[0]
                    area_km2, perimetro_km = geom.area / 1e6, geom.length / 1000
                    kc = 0.282 * perimetro_km / np.sqrt(area_km2) 
                    bounds = geom.bounds
                    longitud_axial_km = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) / 1000
                    kf = area_km2 / (longitud_axial_km ** 2) 
                    
                    dd_str = "N/A (Calcule ríos primero)"
                    longitud_rios_km = 0
                    if st.session_state.get('gdf_rios') is not None:
                        longitud_rios_km = st.session_state['gdf_rios']['longitud_km'].sum()
                        dd_str = f"{longitud_rios_km / area_km2:.2f} km/km²"

                    desnivel_m = max_el - min_el
                    pendiente_cauce_m_m = desnivel_m / (longitud_axial_km * 1000)
                    
                    st.markdown("##### 📐 Índices Morfométricos")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Área (A)", f"{area_km2:.2f} km²")
                    c2.metric("Perímetro (P)", f"{perimetro_km:.2f} km")
                    c3.metric("Gravelius (Kc)", f"{kc:.3f}", help=">1: Alargada, ~1: Redonda")
                    c4.metric("Densidad Drenaje", dd_str)

                    with st.expander("Ver Tabla Detallada de Parámetros"):
                        df_morfo = pd.DataFrame({
                            "Parámetro": ["Área", "Perímetro", "Longitud Axial", "Longitud Total Ríos", "Desnivel (H)", "Pendiente Media Cuenca", "Pendiente Aprox. Cauce"],
                            "Valor": [area_km2, perimetro_km, longitud_axial_km, longitud_rios_km, desnivel_m, slope_mean, pendiente_cauce_m_m * 100],
                            "Unidad": ["km²", "km", "km", "km", "m", "Grados", "%"]
                        })
                        st.dataframe(df_morfo.style.format({"Valor": "{:.2f}"}), use_container_width=True)
                        st.session_state['df_indices'] = df_morfo 
                    
                    st.markdown("---")
                    
                    if st.session_state.get('geomorfo_strahler_df') is not None:
                        st.markdown("##### 🌊 Red de Drenaje y Potencial Ripario (Strahler)")
                        df_str = st.session_state['geomorfo_strahler_df']
                        
                        c_str1, c_str2 = st.columns([1, 1.5])
                        with c_str1:
                            st.dataframe(df_str.style.format({'Longitud_Km': '{:.2f}'}), use_container_width=True, hide_index=True)
                            rb_list = [df_str['Num_Segmentos'].iloc[i] / df_str['Num_Segmentos'].iloc[i+1] for i in range(len(df_str)-1) if df_str['Num_Segmentos'].iloc[i+1] > 0]
                            rb_mean = sum(rb_list)/len(rb_list) if rb_list else 0
                            st.metric("Relación de Bifurcación ($R_b$)", f"{rb_mean:.2f}", help="Si Rb está entre 3 y 5, la cuenca es geológicamente estable. Valores altos indican riesgo de crecientes súbitas.")
                            
                        with c_str2:
                            fig_str = px.bar(df_str, x='Orden_Strahler', y='Longitud_Km', title="Longitud de Ríos por Orden de Strahler", labels={'Orden_Strahler': 'Orden', 'Longitud_Km': 'Longitud (Km)'}, color='Orden_Strahler', color_continuous_scale='Blues')
                            fig_str.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), xaxis=dict(tickmode='linear', dtick=1))
                            st.plotly_chart(fig_str, use_container_width=True)
                    
                    st.markdown("##### ⏱️ Tiempo de Concentración (Tc) y Caudales")
                    col_tc, col_q = st.columns(2)
                    
                    with col_tc:
                        st.markdown("**1. Tiempo de Concentración (Tc)**")
                        L_m = longitud_axial_km * 1000
                        S_mm = pendiente_cauce_m_m
                        
                        if S_mm > 0:
                            tc_kirpich_min = 0.01947 * (L_m**0.77) * (S_mm**-0.385)
                            tc_calif_hr = 0.87 * ((longitud_axial_km**3) / desnivel_m)**0.385
                            st.info(f"⏱️ **Kirpich:** {tc_kirpich_min:.1f} min ({tc_kirpich_min/60:.2f} h)")
                            st.write(f"⏱️ **California:** {tc_calif_hr*60:.1f} min ({tc_calif_hr:.2f} h)")
                        else:
                            st.warning("Pendiente nula, no se puede calcular Tc.")
                            tc_kirpich_min = 0

                    with col_q:
                        st.markdown("**2. Caudal Pico Extremo (Q) - Método Racional Integrado**")
                        c_sugerido = 0.50 
                        detalle_cob = "No hay datos de cobertura."
                        
                        # 🚀 NUBE: CÁLCULO AUTOMÁTICO DE C (COBERTURAS EN LA NUBE)
                        PATH_COB = Config.LAND_COVER_RASTER_PATH
                        if land_cover and PATH_COB:
                            try:
                                stats_cob = land_cover.calcular_estadisticas_zona(gdf_zona_seleccionada, PATH_COB)
                                if stats_cob:
                                    c_pond = sum((0.85 if "Urbano" in cob or "Industrial" in cob else 0.60 if "Cultivo" in cob else 0.45 if "Pasto" in cob or "Herbácea" in cob else 0.30 if "Bosque" in cob else 1.0 if "Agua" in cob else 0.5) * (pct/100.0) for cob, pct in stats_cob.items())
                                    c_sugerido = c_pond
                                    detalle_cob = ", ".join([f"{k} ({v:.0f}%)" for k,v in sorted(stats_cob.items(), key=lambda x: x[1], reverse=True)[:3]])
                            except: pass

                        c_runoff = st.slider("Coeficiente de Escorrentía (C):", 0.1, 1.0, float(c_sugerido), 0.05, help=f"Valor sugerido basado en coberturas satelitales: {c_sugerido:.2f}\nPredomina: {detalle_cob}")
                        if c_sugerido != 0.5: st.caption(f"🛰️ **C Calculado:** {c_sugerido:.2f} ({detalle_cob})")

                        # 🌍 BISTURÍ: INYECCIÓN ESTADÍSTICA (GUMBEL Y DESAGREGACIÓN)
                        st.markdown("---")
                        st.markdown("##### ⛈️ Tormenta de Diseño (Nexo Estadístico)")
                        
                        ppt_100a_memoria = float(st.session_state.get('aleph_ppt_100a', 120.0))
                        
                        if 'aleph_ppt_100a' in st.session_state:
                            st.success(f"🧠 **Gumbel Sincronizado:** Extremo Histórico Tr=100 años es de **{ppt_100a_memoria:.1f} mm**.")
                            st.caption("💡 *Nota: Si este valor proviene del historial mensual, representa el mes más lluvioso, no un evento de 24 horas.*")
                        
                        # Corrector de Escala (Mensual a 24h)
                        es_mensual = st.checkbox("🔄 El valor sincronizado es Mensual (Desagregar a 24h)", value=ppt_100a_memoria > 300)
                        
                        if es_mensual:
                            factor_24h = st.slider("Factor de escala (Mes a 24h):", 0.1, 0.5, 0.30, 0.05, help="En zonas tropicales andinas, el día más lluvioso suele aportar entre el 25% y 35% de la lluvia del mes más extremo.")
                            p_24h_sugerido = ppt_100a_memoria * factor_24h
                        else:
                            p_24h_sugerido = ppt_100a_memoria
                            
                        p_diseno = st.number_input("Precipitación Extrema 24h ($P_{24}$) [mm]:", min_value=10.0, value=float(p_24h_sugerido), step=5.0)
                        
                        # --- MAGIA HIDROLÓGICA: CURVA IDF SINTÉTICA ---
                        # Para no asumir que todo P24 cae en el Tc, usamos una desagregación típica de tormenta
                        # I = (P24 / 24) * (24 / Tc)^0.65  <-- Muy usada en Colombia cuando no hay IDF local
                        tc_horas = tc_kirpich_min / 60 if tc_kirpich_min > 0 else 1.0
                        
                        i_rain_calc = (p_diseno / 24.0) * ((24.0 / tc_horas) ** 0.65)
                        
                        st.info(f"⚡ **Intensidad Desagregada (I):** {i_rain_calc:.1f} mm/h *(Ajustada con curva IDF sintética para un Tc de {tc_horas:.2f}h)*.")
                        
                        # Cálculo Final Racional
                        q_peak = 0.278 * c_runoff * i_rain_calc * area_km2
                        st.session_state['geomorfo_q_pico_racional'] = float(q_peak)
                        
                        origen_q = "Modelo Racional (Geomorfología)"
                        if 'aleph_q_max_m3s' not in st.session_state or st.session_state['aleph_q_max_m3s'] == 0.0: 
                            st.session_state['aleph_q_max_m3s'] = float(q_peak)
                        else: 
                            origen_q = "Aleph Distribuido (Pág 01 - Preservado)"
                        
                        st.metric("Caudal Pico (Q)", f"{q_peak:,.2f} m³/s", f"Método: {origen_q}")
                        st.caption("Fórmula Racional: $Q = 0.278 \cdot C \cdot I \cdot A$")
                
                except Exception as e:
                    st.error(f"Error en cálculos hidrológicos: {e}")

            # --- TAB 7: AMENAZAS ---
            with tab7:
                st.subheader("🚨 Zonificación de Amenazas Hidrológicas")
                def mapa_amenaza_contexto(mask_binaria, color_amenaza, titulo, gdf_oficial):
                    from rasterio import features
                    fig = go.Figure()

                    if gdf_oficial is not None:
                        gdf_4326 = gdf_oficial.to_crs("EPSG:4326")
                        poly = gdf_4326.geometry.iloc[0]
                        xo, yo = poly.exterior.coords.xy if poly.geom_type == 'Polygon' else max(poly.geoms, key=lambda a: a.area).exterior.coords.xy
                        fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xo), lat=list(yo), line={'width': 3, 'color': '#00FF00'}, name="Cuenca Oficial"))
                        c_lat, c_lon = gdf_4326.centroid.y.mean(), gdf_4326.centroid.x.mean()
                    else: c_lat, c_lon = 6.2, -75.5

                    try:
                        try: r = gpd.read_postgis("SELECT * FROM red_drenaje", engine, geom_col='geometry')
                        except: r = gpd.read_postgis("SELECT * FROM red_drenaje", engine, geom_col='geom')
                        
                        if gdf_oficial is not None:
                            r_clip = gpd.clip(r.to_crs("EPSG:3116"), gdf_oficial.to_crs("EPSG:3116"))
                            if not r_clip.empty:
                                r_viz = r_clip.to_crs("EPSG:4326")
                                l, lt = [], []
                                for g in r_viz.geometry:
                                    if g.geom_type=='LineString': x,y=g.xy; l.extend(list(x)+[None]); lt.extend(list(y)+[None])
                                    elif g.geom_type=='MultiLineString': 
                                        for s in g.geoms: x,y=s.xy; l.extend(list(x)+[None]); lt.extend(list(y)+[None])
                                fig.add_trace(go.Scattermapbox(mode="lines", lon=l, lat=lt, line={'width':1.5, 'color':'#0044FF'}, name="Red Drenaje"))
                    except: pass

                    geoms = [shape(g) for g, v in features.shapes(np.ascontiguousarray(mask_binaria, dtype=np.uint8), transform=transform) if v == 1]
                    if geoms:
                        gdf_threat = gpd.GeoDataFrame({'geometry': geoms}, crs=meta['crs']).to_crs("EPSG:3116")
                        gdf_threat = gdf_threat[gdf_threat.geometry.area > 900]
                        if not gdf_threat.empty:
                            gdf_threat['geometry'] = gdf_threat.simplify(20).to_crs("EPSG:4326")
                            fig.add_trace(go.Choroplethmapbox(geojson=gdf_threat.geometry.__geo_interface__, locations=gdf_threat.index, z=[1]*len(gdf_threat), colorscale=[[0, color_amenaza], [1, color_amenaza]], showscale=False, marker_opacity=0.6, name="Zona Amenaza"))
                        else: st.info("✅ Zona segura (Amenazas pequeñas filtradas).")
                    else: st.info("✅ Sin amenazas detectadas.")

                    fig.update_layout(title=titulo, mapbox_style="carto-positron", mapbox_zoom=12, mapbox_center={"lat": c_lat, "lon": c_lon}, height=600, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

                def caja_analisis_ai(mask_riesgo, tipo):
                    pct = (np.sum(mask_riesgo) / mask_riesgo.size) * 100
                    color = "red" if "Torrencial" in tipo else "#0099FF"
                    nivel = "CRÍTICO" if pct > 5 else ("Medio" if pct > 1 else "Bajo")
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding: 15px; background-color: rgba(240,242,246,0.5); border-radius: 5px; margin-bottom: 20px;">
                        <strong style="color: {color}; font-size: 1.1em;">📊 Diagnóstico: {tipo}</strong>
                        <ul style="margin-bottom: 0;"><li><b>Área Afectada:</b> {pct:.2f}% de la zona visible.</li><li><b>Nivel de Alerta:</b> {nivel}</li></ul>
                    </div>""", unsafe_allow_html=True)

                if 'acc' in locals() and acc is not None and 'slope_deg' in locals():
                    min_h, min_w = min(slope_deg.shape[0], acc.shape[0]), min(slope_deg.shape[1], acc.shape[1])
                    s_core, acc_raw = slope_deg[:min_h, :min_w], acc[:min_h, :min_w]
                    a_core_log = np.log1p(acc_raw)
                    
                    t1, t2 = st.tabs(["🔴 Avenida Torrencial", "🔵 Inundación (TWI)"])
                    
                    with t1:
                        c_t1, c_t2 = st.columns([1, 3])
                        with c_t1:
                            st.markdown("#### Parámetros Físicos y Energía")
                            q_max_aleph = st.session_state.get('aleph_q_max_m3s', 0.0)
                            s_range = st.slider("Rango de Pendiente Crítica (°)", 0.0, 60.0, (15.0, 45.0), key="s_torrencial")
                            if q_max_aleph > 0:
                                st.success(f"🧠 **Caudal Pico Sincronizado:** {q_max_aleph:,.1f} m³/s")
                                a_sugerido = max(4.0, 8.5 - np.log10(q_max_aleph + 1))
                            else:
                                st.warning("⚠️ Usando energía teórica. Ejecuta el Aleph en la Pág 01.")
                                a_sugerido = 6.0
                                
                            a_umb = st.slider("Umbral de Energía (Acumulación Log):", min_value=4.0, max_value=9.0, value=float(a_sugerido), step=0.1, key="a_torrencial")
                            
                            if st.button("🚀 Guardar Zona Torrencial (Para Biodiversidad)"):
                                st.session_state['aleph_twi_umbral'] = a_umb
                                st.session_state['aleph_pendiente_max'] = s_range[1]
                                st.session_state['aleph_pendiente_min'] = s_range[0]
                                st.toast("✅ Zona torrencial guardada para diseño de bosques de protección.", icon="🪨")

                        with c_t2:
                            mask_t = (s_core >= s_range[0]) & (s_core <= s_range[1]) & (a_core_log >= a_umb)
                            caja_analisis_ai(mask_t, "Avenida Torrencial (Flujo de Detritos)")
                            mapa_amenaza_contexto(mask_t, "#e74c3c", f"Zonas de Alta Energía / Susceptibilidad Torrencial", gdf_zona_seleccionada)

                    with t2:
                        c_in1, c_in2 = st.columns([1, 3])
                        with c_in1:
                            st.markdown("#### Motor Hidráulico (Manning-TWI)")
                            q_max_aleph = st.session_state.get('aleph_q_max_m3s', 0.0)
                            if q_max_aleph > 0:
                                st.success(f"🧠 **Caudal Pico Sincronizado:** {q_max_aleph:,.1f} m³/s")
                                q_diseno = q_max_aleph
                            else:
                                st.warning("⚠️ Usando caudal teórico. Ejecuta el Aleph en la Pág 01 para usar hidrología real.")
                                q_diseno = 50.0 
                                
                            twi_sugerido = max(5.0, 16.0 - np.log1p(q_diseno))
                            twi_val = st.slider("Umbral de Desbordamiento (TWI):", min_value=5.0, max_value=25.0, value=float(twi_sugerido), step=0.5, key="twi_slider")
                            
                            with st.spinner("Inundando llanura aluvial..."):
                                s_rad = np.deg2rad(s_core)
                                tan_s = np.where(np.tan(s_rad) < 0.001, 0.001, np.tan(s_rad))
                                twi = np.log((acc_raw * 30.0 + 1) / tan_s)
                                
                            strict = st.checkbox("Restringir a valles (< 5°)", value=True)
                            
                            if st.button("🚀 Guardar Zona de Riesgo (Para Corredores Riparios)"):
                                st.session_state['aleph_twi_umbral'] = twi_val
                                st.session_state['aleph_pendiente_max'] = 5 if strict else 90
                                st.toast("✅ Zona de inundación guardada en Memoria Global.", icon="🌊")

                        with c_in2:
                            mask_i = (twi >= twi_val) & (s_core <= 5) if strict else (twi >= twi_val)
                            caja_analisis_ai(mask_i, "Inundación por Desbordamiento")
                            mapa_amenaza_contexto(mask_i, "#0099FF", f"Llanura de Inundación (Q = {q_diseno:,.1f} m³/s)", gdf_zona_seleccionada)
                else:
                    st.warning("⚠️ Calcula la Hidrología en el Tab 4 primero.")
                    
            # --- TAB 5: DESCARGAS ---
            with tab5:
                st.subheader("Centro de Descargas")
                st.caption("Descarga los productos generados en las pestañas anteriores.")
                
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                
                with c1:
                    st.write("🏔️ **DEM**")
                    c1.download_button("💾 .TIF", to_tif(arr_elevacion, meta), f"DEM_{nombre_zona}.tif")
                
                with c2:
                    st.write("〰️ **Curvas**")
                    if st.session_state['gdf_contours'] is not None:
                        c2.download_button("💾 .JSON", st.session_state['gdf_contours'].to_json(), f"Curvas_{nombre_zona}.geojson", "application/json")
                    else: st.warning("⚠️ Ver Tab 3D")

                with c3:
                    st.write("📐 **Pendiente**")
                    slope_meta = meta.copy(); slope_meta.update(dtype=rasterio.float32)
                    c3.download_button("💾 .TIF", to_tif(slope_deg, slope_meta), f"Slope_{nombre_zona}.tif")

                with c4:
                    st.write("📈 **Hipso**")
                    try:
                        elevs_sort = np.sort(elevs_valid)[::-1]
                        df_hyp = pd.DataFrame({"Porcentaje_Area": np.linspace(0, 100, len(elevs_sort)), "Altitud": elevs_sort})
                        c4.download_button("💾 .CSV", df_hyp.to_csv(index=False).encode('utf-8'), f"Hipsometria_{nombre_zona}.csv", "text/csv")
                    except: st.error("Error calc.")

                with c5:
                    st.write("🌊 **Ríos**")
                    if st.session_state['gdf_rios'] is not None:
                        c5.download_button("💾 .JSON", st.session_state['gdf_rios'].to_json(), f"Rios_{nombre_zona}.geojson", "application/json")
                    else: st.warning("⚠️ Ver Tab Hidro")

                with c6:
                    st.write("📊 **Índices**")
                    if st.session_state['df_indices'] is not None:
                        c6.download_button("💾 .CSV", st.session_state['df_indices'].to_csv(index=False).encode('utf-8'), f"Indices_{nombre_zona}.csv", "text/csv")
                    else: st.warning("⚠️ Ver Tab Índices")

                with c7:
                    st.write("🟦 **Cuenca**")
                    if st.session_state['catchment_raster'] is not None:
                        catch_meta = meta.copy(); catch_meta.update(dtype=rasterio.uint8, nodata=0)
                        c7.download_button("💾 .TIF", to_tif(st.session_state['catchment_raster'].astype(np.uint8), catch_meta), f"Catchment_{nombre_zona}.tif")
                    else: st.warning("⚠️ Calc. Tab Hidro")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🌐 Mapas Interactivos y Vectores SIG")
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    if 'fig_mapa_hidro' in st.session_state and st.session_state['fig_mapa_hidro'] is not None:
                        st.download_button("📥 Mapa Interactivo (HTML)", st.session_state['fig_mapa_hidro'].to_html(include_plotlyjs='cdn'), f"Mapa_Hidrologia_{nombre_zona}.html", "text/html", use_container_width=True)
                    else: st.info("⚠️ Genera el mapa en Tab Hidrología primero.")
                
                with col_d2:
                    if st.session_state.get('gdf_rios') is not None:
                        st.download_button("📥 Red de Drenaje (GeoJSON)", st.session_state['gdf_rios'].to_crs("EPSG:4326").to_json(), f"Red_Drenaje_Strahler_{nombre_zona}.geojson", "application/json", use_container_width=True)
                    else: st.info("⚠️ Calcula los ríos en Tab Hidrología primero.")
                        
                with col_d3:
                    if st.session_state.get('gdf_rios') is not None:
                        import tempfile, zipfile
                        from io import BytesIO
                        
                        def create_shp_zip():
                            temp_dir = tempfile.mkdtemp()
                            shp_path = os.path.join(temp_dir, f"Drenaje_{nombre_zona}".replace(" ", "_") + ".shp")
                            gdf_export = st.session_state['gdf_rios'].to_crs("EPSG:4326").copy()
                            if 'Orden_Strahler' in gdf_export.columns: gdf_export = gdf_export.rename(columns={'Orden_Strahler': 'Orden_Stra'})
                            if 'longitud_km' in gdf_export.columns: gdf_export = gdf_export.rename(columns={'longitud_km': 'Long_km'})
                            gdf_export.to_file(shp_path, driver='ESRI Shapefile')
                            
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files: zip_file.write(os.path.join(root, file), arcname=file)
                            return zip_buffer.getvalue()
                            
                        st.download_button("📥 Red de Drenaje (Shapefile .zip)", create_shp_zip(), f"Shapefile_Drenaje_{nombre_zona}.zip", "application/zip", use_container_width=True)
                    else: st.info("⚠️ Calcula los ríos en Tab Hidrología primero.")

else:
    st.info("👈 Selecciona una zona.")

# ==============================================================================
# 🚀 ⚙️ FORJA MASIVA: MATRIZ HIDRO-GEOMORFOLÓGICA MAESTRA
# ==============================================================================
st.markdown("---")
with st.expander("⚙️ PANEL DE ADMINISTRADOR: Forja Masiva de Matriz Hidro-Geomorfológica", expanded=False):
    st.warning("⚠️ **Atención:** Este proceso iterará sobre TODAS las cuencas de la base de datos, recortará el Modelo Digital de Elevación (DEM), calculará las curvas hipsométricas y forjará las ecuaciones de caudal por altitud. Puede tardar varios minutos.")
    
    if st.button("⚡ Iniciar Forja Masiva Global", type="primary", use_container_width=True):
        import time
        from sqlalchemy import text
        from modules.utils import normalizar_texto
        from modules.db_manager import get_engine
        
        engine_sql = get_engine()
        
        # 1. Extraer todas las cuencas desde la base de datos oficial
        try:
            gdf_todas_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine_sql, geom_col="geometry")
        except Exception as e:
            st.error(f"Error cargando mapa de cuencas: {e}")
            st.stop()
            
        if gdf_todas_cuencas.empty:
            st.error("No se encontraron cuencas en la base de datos.")
            st.stop()
            
        # Determinar columnas de nombres (Ajusta si tus columnas se llaman distinto en la tabla 'cuencas')
        columnas_niveles = {
            "AH": "nomah",
            "ZH": "nomzh",
            "SZH": "nom_szh",
            "NSS1": "nom_nss1",
            "NSS2": "nom_nss2",
            "NSS3": "nom_nss3"
        }
        
        barra_progreso = st.progress(0)
        texto_progreso = st.empty()
        
        resultados_forja = []
        total_entidades = 0
        
        # Pre-contar entidades válidas para la barra de progreso
        entidades_a_procesar = []
        for nivel, col in columnas_niveles.items():
            if col in gdf_todas_cuencas.columns:
                nombres_unicos = gdf_todas_cuencas[col].dropna().unique()
                for nombre in nombres_unicos:
                    entidades_a_procesar.append((nivel, col, nombre))
                    
        total_entidades = len(entidades_a_procesar)
        st.info(f"🔍 Detectadas {total_entidades} entidades hidrográficas para modelar.")

        # 2. Bucle Maestro de Forja
        for i, (nivel, col, nombre) in enumerate(entidades_a_procesar):
            try:
                texto_progreso.text(f"Forjando [{i+1}/{total_entidades}]: {nivel} - {nombre}...")
                
                # Extraer polígono
                gdf_poligono = gdf_todas_cuencas[gdf_todas_cuencas[col] == nombre].copy()
                if gdf_poligono.empty: continue
                
                # Unir si hay múltiples fragmentos
                if len(gdf_poligono) > 1:
                    gdf_poligono = gpd.GeoDataFrame({'geometry': [gdf_poligono.unary_union]}, crs=gdf_todas_cuencas.crs)
                    
                # Recortar DEM
                arr_dem, meta_dem, _ = cargar_y_cortar_dem(DEM_PATH, gdf_poligono, nombre)
                
                if arr_dem is not None and not np.isnan(arr_dem).all():
                    elevs_valid = arr_dem[~np.isnan(arr_dem)].flatten()
                    if len(elevs_valid) < 10: continue # Ignorar si es muy pequeño
                    
                    elevs_sorted = np.sort(elevs_valid)[::-1]
                    total_pixels = len(elevs_sorted)
                    x_pct = np.linspace(0, 100, total_pixels)
                    
                    # Reducir muestra para el polinomio y mejorar velocidad
                    idx = np.linspace(0, total_pixels-1, min(500, total_pixels), dtype=int)
                    
                    # Forjar Polinomio de Grado 3 (A(h))
                    z_poly = np.polyfit(elevs_sorted[idx], x_pct[idx], 3)
                    p_func = np.poly1d(z_poly)
                    
                    # Calcular R2
                    y_pred = p_func(elevs_sorted[idx])
                    y_real = x_pct[idx]
                    r2 = 1 - (np.sum((y_real - y_pred) ** 2) / (np.sum((y_real - np.mean(y_real)) ** 2) + 1e-9))
                    
                    # Generar Llave Universal
                    t_norm = normalizar_texto(nombre).upper().replace(" ", "_")
                    llave_u = f"{nivel.upper()}_{t_norm}_TOTAL"
                    
                    resultados_forja.append({
                        "LLAVE_UNIVERSAL": llave_u,
                        "Territorio": nombre,
                        "Nivel": nivel,
                        "Coef_C3": float(z_poly[0]),
                        "Coef_C2": float(z_poly[1]),
                        "Coef_C1": float(z_poly[2]),
                        "Coef_C0": float(z_poly[3]),
                        "R2_Ajuste": float(r2),
                        "H_Minima": float(np.min(elevs_valid)),
                        "H_Maxima": float(np.max(elevs_valid)),
                        "H_Media": float(np.mean(elevs_valid))
                    })
                    
            except Exception as e:
                # Si una cuenca falla (fuera del DEM, error topológico), la saltamos silenciosamente
                pass
                
            barra_progreso.progress((i + 1) / total_entidades)

        texto_progreso.text("🚀 Empaquetando Matriz Maestra y enviando a SQL...")
        
        # 3. Inyección a SQL
        if resultados_forja:
            df_matriz_geomorfo = pd.DataFrame(resultados_forja)
            
            try:
                with engine_sql.connect() as conn:
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS matriz_hidrogeomorfologica_maestra (
                            "LLAVE_UNIVERSAL" TEXT PRIMARY KEY,
                            "Territorio" TEXT, "Nivel" TEXT,
                            "Coef_C3" FLOAT, "Coef_C2" FLOAT, "Coef_C1" FLOAT, "Coef_C0" FLOAT,
                            "R2_Ajuste" FLOAT, "H_Minima" FLOAT, "H_Maxima" FLOAT, "H_Media" FLOAT
                        );
                    '''))
                    # Limpiamos tabla vieja para un reemplazo limpio
                    conn.execute(text("DELETE FROM matriz_hidrogeomorfologica_maestra;"))
                    
                df_matriz_geomorfo.to_sql('matriz_hidrogeomorfologica_maestra', engine_sql, if_exists='append', index=False)
                
                barra_progreso.progress(1.0)
                st.success(f"✅ **¡FORJA MASIVA COMPLETADA!** Se han inyectado {len(df_matriz_geomorfo)} ecuaciones hipsométricas a la Matriz Hidro-Geomorfológica Maestra.")
                
                with st.expander("Ver Muestra de la Matriz Generada"):
                    st.dataframe(df_matriz_geomorfo.head(20))
                    
            except Exception as e:
                st.error(f"🚨 Error inyectando la matriz maestra a SQL: {e}")
        else:
            st.warning("No se generó ningún resultado válido durante la forja.")
