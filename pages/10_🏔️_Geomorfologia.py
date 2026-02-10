import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import features
from rasterio.io import MemoryFile
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from shapely.geometry import shape, LineString, MultiLineString, Polygon
from modules import selectors

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

# Configuración de Página
st.set_page_config(page_title="Geomorfología Pro", page_icon="🏔️", layout="wide")

# --- INICIALIZACIÓN DE VARIABLES DE ESTADO ---
# (Asegúrate de que este bloque tenga todas estas líneas)
if 'gdf_contours' not in st.session_state: st.session_state['gdf_contours'] = None
if 'catchment_raster' not in st.session_state: st.session_state['catchment_raster'] = None
if 'gdf_rios' not in st.session_state: st.session_state['gdf_rios'] = None     # <--- NUEVO
if 'df_indices' not in st.session_state: st.session_state['df_indices'] = None # <--- NUEVO
    
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

# --- 2. CARGA DEL DEM ---
DEM_PATH = os.path.join("data", "DemAntioquia_EPSG3116.tif")

@st.cache_data(show_spinner="Procesando terreno...")
def cargar_y_cortar_dem(ruta_dem, _gdf_corte, zona_id):
    if _gdf_corte is None or _gdf_corte.empty: return None, None, None
    try:
        if not os.path.exists(ruta_dem): return None, None, None
        with rasterio.open(ruta_dem) as src:
            crs_dem = src.crs
            gdf_proyectado = _gdf_corte.to_crs(crs_dem)
            geoms = gdf_proyectado.geometry.values
            try:
                out_image, out_transform = mask(src, geoms, crop=True)
            except ValueError:
                return None, "OUT_OF_BOUNDS", None
            
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform, "count": 1})
            dem_array = out_image[0]
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            dem_array = np.where(dem_array < -100, np.nan, dem_array)
            if np.isnan(dem_array).all(): return None, "EMPTY_DATA", None
            return dem_array, out_meta, out_transform
    except Exception as e:
        st.error(f"Error DEM: {e}")
        return None, None, None

# --- CEREBRO DEL ANALISTA (RECUPERADO) 🧠 ---
def analista_hidrologico(pendiente_media, hi_value):
    # Pendiente
    if pendiente_media > 25:
        txt_pend = "un relieve fuertemente escarpado"
        riesgo = "alto potencial de flujos torrenciales y respuesta rápida"
    elif pendiente_media > 12:
        txt_pend = "un relieve moderadamente ondulado"
        riesgo = "velocidades de flujo moderadas"
    else:
        txt_pend = "un relieve predominantemente plano"
        riesgo = "propensión al encharcamiento y flujos lentos"

    # Hipsometría
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

# --- FUNCIÓN DE VECTORIZACIÓN DE RÍOS (MEJORADA: LÍNEAS DE FLUJO) 🌊 ---
@st.cache_data(show_spinner="Trazando red de drenaje...")
def extraer_vectores_rios(_grid, _fdir, _acc, umbral, _crs_in, cache_id):
    """
    Usa PySheds para extraer líneas de flujo reales (Centerlines).
    """
    try:
        # PySheds native extraction
        # dirmap estándar: (N, NE, E, SE, S, SW, W, NW)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        branches = _grid.extract_river_network(_fdir, _acc > umbral, dirmap=dirmap)
        
        if not branches or not branches['features']:
            return None

        # Convertir GeoJSON a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(branches['features'])
        
        # Asignar CRS manualmente si viene vacío
        if gdf.crs is None:
            gdf.set_crs(_crs_in, inplace=True)
            
        return gdf

    except Exception as e:
        return None
        
# --- FUNCIONES DE DESCARGA ---
def to_tif(arr, meta):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(arr.astype(rasterio.float32), 1)
        return memfile.read()

# --- LÓGICA PRINCIPAL ---
if gdf_zona_seleccionada is not None:
    if not os.path.exists(DEM_PATH):
        st.error(f"⚠️ Archivo no encontrado: {DEM_PATH}")
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
            
            # Pendientes (NumPy)
            dy, dx = np.gradient(arr_elevacion, 30.0)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_mean = np.nanmean(slope_deg)
            
            # Texto Analista
            texto_analisis = analista_hidrologico(slope_mean, hi_global)

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mínima", f"{min_el:.0f} m")
            c2.metric("Máxima", f"{max_el:.0f} m")
            c3.metric("Media", f"{mean_el:.0f} m")
            c4.metric("Rango", f"{max_el - min_el:.0f} m")

            tab1, tab2, tab3, tab4, tab6, tab7, tab5 = st.tabs([
                "🗺️ 3D", "📐 Pendientes", "📈 Hipsometría", 
                "🌊 Hidrología", "📊 Índices (Nuevo)", "🚨 Amenazas", "📥 Descargas"
            ])
            
            # Factor de reducción visual
            h, w = arr_elevacion.shape
            factor = max(1, int(max(h, w) / 200)) # Mejor resolución para el 3D

            # --- TAB 1: 3D Y CURVAS DE NIVEL (NATIVAS + DESCARGA) ---
            with tab1:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown("#### Visualización")
                    exag = st.slider("Exageración Vertical:", 0.5, 5.0, 1.5, 0.1, key="z_exag")
                    st.markdown("---")
                    ver_curvas = st.toggle("Ver Curvas de Nivel", value=True)
                    intervalo_curvas = st.select_slider("Intervalo (m):", options=[10, 25, 50, 100], value=50)
                    
                with c2:
                    # Preparar Terreno 3D
                    arr_3d = arr_elevacion[::factor, ::factor]
                    
                    # --- CONFIGURACIÓN DE CURVAS NATIVAS (VISUAL) ---
                    contours_conf = dict(
                        z=dict(
                            show=ver_curvas,
                            start=np.nanmin(arr_elevacion),
                            end=np.nanmax(arr_elevacion),
                            size=intervalo_curvas,
                            color="white", # Color de la línea en el mapa
                            usecolormap=False,
                            project_z=False # False = Pegadas al terreno (lo que te gusta)
                        )
                    )

                    fig = go.Figure(data=[go.Surface(
                        z=arr_3d, 
                        colorscale='Earth', 
                        contours=contours_conf, # ¡Aquí está la magia visual!
                        name="Terreno"
                    )])

                    # --- CÁLCULO SILENCIOSO PARA DESCARGA (VECTOR) ---
                    # Esto ocurre en background solo para generar el archivo JSON
                    if ver_curvas:
                        try:
                            # Usamos matplotlib solo para matemáticas, no para pintar
                            min_z, max_z = np.nanmin(arr_elevacion), np.nanmax(arr_elevacion)
                            levels = np.arange(np.floor(min_z), np.ceil(max_z), intervalo_curvas)
                            contours_obj = plt.contour(arr_elevacion, levels=levels)
                            
                            geoms_2d = [] 
                            for level, collection in zip(levels, contours_obj.collections):
                                for path in collection.get_paths():
                                    v = path.vertices
                                    if len(v) < 2: continue # Ignorar puntos
                                    # Transformar Pixel -> Lat/Lon
                                    xs_geo, ys_geo = rasterio.transform.xy(transform, v[:, 0], v[:, 1])
                                    geoms_2d.append({'geometry': LineString(zip(xs_geo, ys_geo)), 'elevation': level})
                            
                            plt.close() # Limpiar memoria
                            
                            if geoms_2d:
                                st.session_state['gdf_contours'] = gpd.GeoDataFrame(geoms_2d, crs=meta['crs'])
                        except: pass

                    fig.update_layout(
                        title="Terreno 3D (Curvas Nativas)", autosize=True, height=600, 
                        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2*exag)),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: PENDIENTES (ZOOM FULL) ---
            with tab2:
                st.subheader("Mapa de Pendientes")
                st.info(texto_analisis, icon="🤖") 
                
                fig_slope = px.imshow(
                    slope_deg[::factor, ::factor], 
                    color_continuous_scale='Turbo',
                    labels={'color': 'Grados'},
                    title=f"Pendientes: {nombre_zona}"
                )
                
                # Configuración explícita de ejes para permitir zoom libre
                fig_slope.update_xaxes(fixedrange=False) # Permitir zoom X
                fig_slope.update_yaxes(fixedrange=False) # Permitir zoom Y
                
                fig_slope.update_layout(
                    height=700, # Más alto
                    dragmode='pan', 
                    hovermode='closest'
                )
                st.plotly_chart(fig_slope, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

                # Tabla
                slope_flat = slope_deg[~np.isnan(slope_deg)].flatten()
                bins = [0, 3, 7, 12, 25, 50, 90]
                labels = ['Plano', 'Suave', 'Inclinado', 'Ondulado', 'Escarpado', 'Muy Escarpado']
                cats = pd.cut(slope_flat, bins=bins, labels=labels)
                counts = cats.value_counts().sort_index()
                df_slopes = pd.DataFrame({'Categoría': counts.index, '%': (counts.values/len(slope_flat)*100).round(1)})
                st.dataframe(df_slopes.T, use_container_width=True)

            # --- TAB 3: HIPSOMETRÍA (ESTÁNDAR Y ADIMENSIONAL) ---
            with tab3:
                # Datos base
                elevs_sorted = np.sort(elevs_valid)[::-1]
                total_pixels = len(elevs_sorted)
                x_pct = np.linspace(0, 100, total_pixels)
                
                # Reducción para graficar rápido
                idx = np.linspace(0, total_pixels-1, 500, dtype=int)
                
                c_hip1, c_hip2 = st.columns(2)
                
                with c_hip1:
                    # GRÁFICO 1: Curva Hipsométrica Clásica (Absoluta)
                    fig_hyp = go.Figure()
                    fig_hyp.add_trace(go.Scatter(x=x_pct[idx], y=elevs_sorted[idx], fill='tozeroy', name='Real', line=dict(color='#2E86C1')))
                    fig_hyp.update_layout(
                        title="Curva Hipsométrica (Absoluta)",
                        xaxis_title="% Área Acumulada", yaxis_title="Altitud (m.s.n.m)",
                        height=450, margin=dict(l=0,r=0,t=40,b=0)
                    )
                    st.plotly_chart(fig_hyp, use_container_width=True)

                with c_hip2:
                    # GRÁFICO 2: Curva Adimensional (Relativa)
                    # Eje Y: (h - h_min) / (h_max - h_min)
                    # Eje X: a / A
                    h_min, h_max = np.min(elevs_sorted), np.max(elevs_sorted)
                    h_rel = (elevs_sorted[idx] - h_min) / (h_max - h_min)
                    a_rel = x_pct[idx] / 100.0 # De 0 a 1
                    
                    fig_adim = go.Figure()
                    fig_adim.add_trace(go.Scatter(x=a_rel, y=h_rel, name='Cuenca Actual', line=dict(color='#E74C3C', width=3)))
                    # Referencia de Equilibrio (Recta)
                    fig_adim.add_trace(go.Scatter(x=[0, 1], y=[1, 0], name='Equilibrio (Ref)', line=dict(color='gray', dash='dot')))
                    
                    fig_adim.update_layout(
                        title="Curva Adimensional (Ciclo de Erosión)",
                        xaxis_title="Área Relativa (a/A)", yaxis_title="Altura Relativa (h/H)",
                        height=450, margin=dict(l=0,r=0,t=40,b=0)
                    )
                    st.plotly_chart(fig_adim, use_container_width=True)
                    
                st.info("""
                **Interpretación Adimensional:**
                * **Curva Convexa (Arriba de la recta):** Cuenca joven, en fase activa de erosión.
                * **Curva Concava (Debajo de la recta):** Cuenca vieja, sedimentada y estabilizada.
                * **Forma de 'S':** Cuenca madura en transición.
                """)

            # --- TAB 4: HIDROLOGÍA (ACTUALIZADO: COORDENADAS EXTREMAS) ---
            gdf_rios_export = None
            _raster_export = None

            with tab4:
                st.subheader("🌊 Hidrología: Red de Drenaje y Cuencas")
                
                if not PYSHEDS_AVAILABLE:
                    st.error("⚠️ Instala `pysheds` para usar este módulo.")
                else:
                    c_conf, c_map = st.columns([1, 3])
                    
                    with c_conf:
                        st.markdown("#### ⚙️ Configuración")
                        opciones_viz = [
                            "Vectores (Líneas)", 
                            " (Mascara)",
                            "Divisoria (Línea)",
                            "Raster (Acumulación)" 
                        ]
                        modo_viz = st.radio("Visualización:", opciones_viz)
                        
                        umbral = 0
                        if modo_viz == "Vectores (Líneas)":
                            umbral = st.slider("Umbral Acumulación", 2, 2000, 50, 5, key=f"umb_{nombre_zona}")
                            st.info("Baja el valor (<50) para ver detalles finos.")

                    with c_map:
                        import tempfile
                        from shapely.geometry import shape
                        
                        # 1. PREPARACIÓN HIDROLÓGICA (Corregido y Blindado)
                        grid = None; acc = None; fdir = None
                        
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            meta_temp = meta.copy(); meta_temp.update(driver='GTiff', dtype='float64') 
                            with rasterio.open(tmp.name, 'w', **meta_temp) as dst:
                                dst.write(arr_elevacion.astype('float64'), 1)
                            try:
                                grid = Grid.from_raster(tmp.name)
                                dem_grid = grid.read_raster(tmp.name)
                                
                                # Procesos Hidrológicos
                                pit_filled = grid.fill_pits(dem_grid)
                                resolved = grid.resolve_flats(pit_filled)
                                dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                                fdir = grid.flowdir(resolved, dirmap=dirmap)
                                acc = grid.accumulation(fdir, dirmap=dirmap)
                                
                            except Exception as e: st.error(f"Error: {e}")
                            finally: 
                                try: os.remove(tmp.name)
                                except: pass

                        if grid is not None and acc is not None:
                            crs_actual = meta.get('crs', 'EPSG:3116')

                            # --- CÁLCULO DE REFERENCIAS (FIX NaN) ---
                            # Convertimos a numpy array puro y enmascaramos los NoData
                            dem_arr = dem_grid.view(np.ndarray)
                            # Asumimos que valores muy bajos o nodata son inválidos para la búsqueda
                            dem_safe = np.where(dem_arr < -100, np.nan, dem_arr)
                            
                            # Búsqueda de índices ignorando NaNs
                            try:
                                # Mínimo (Salida teórica)
                                idx_min_flat = np.nanargmin(dem_safe)
                                y_min, x_min = np.unravel_index(idx_min_flat, dem_safe.shape)
                                h_min = dem_safe[y_min, x_min]
                                
                                # Máximo (Cabecera)
                                idx_max_flat = np.nanargmax(dem_safe)
                                y_max, x_max = np.unravel_index(idx_max_flat, dem_safe.shape)
                                h_max = dem_safe[y_max, x_max]
                            except:
                                y_min, x_min, h_min = 0, 0, 0
                                y_max, x_max, h_max = 0, 0, 0

                            with st.expander("📍 Coordenadas de Referencia (Matriz)", expanded=True):
                                c_ref1, c_ref2 = st.columns(2)
                                c_ref1.info(f"**Punto Más Bajo (Posible Salida):**\n\nFila (Y): {y_min} | Col (X): {x_min}\nAltitud: {h_min:.1f} m")
                                c_ref2.success(f"**Punto Más Alto:**\n\nFila (Y): {y_max} | Col (X): {x_max}\nAltitud: {h_max:.1f} m")
                                
                            # --- MODO 1: RASTER ---
                            if modo_viz == "Raster (Acumulación)":
                                log_acc = np.log1p(acc)
                                fig = px.imshow(log_acc, color_continuous_scale='Blues', title="Acumulación de Flujo (Log)")
                                fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
                                st.plotly_chart(fig, use_container_width=True)

                            # --- MODO 2: CATCHMENT / DIVISORIA (CÓDIGO ORIGINAL ESTABLE) ---
                            elif modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                                # 1. Punto Inicial (Automático Global)
                                if 'x_pour_calib' not in st.session_state:
                                    try:
                                        idx_max_acc = np.nanargmax(acc)
                                        y_auto, x_auto = np.unravel_index(idx_max_acc, acc.shape)
                                    except: y_auto, x_auto = 0, 0
                                    st.session_state['x_pour_calib'] = int(x_auto)
                                    st.session_state['y_pour_calib'] = int(y_auto)

                                # 2. Controles de Calibración
                                with st.expander("🔧 Calibración de Punto de Desfogue", expanded=True):
                                    c_coord, c_snap = st.columns([3, 1])
                                    with c_coord:
                                        c_x, c_y = st.columns(2)
                                        # Usamos session_state para permitir actualizaciones
                                        x_pour = c_x.number_input("Columna (X):", value=st.session_state['x_pour_calib'], min_value=0, max_value=acc.shape[1]-1, step=1, key="num_x")
                                        y_pour = c_y.number_input("Fila (Y):", value=st.session_state['y_pour_calib'], min_value=0, max_value=acc.shape[0]-1, step=1, key="num_y")
                                    
                                    with c_snap:
                                        st.write("") 
                                        st.write("") 
                                        if st.button("🧲 Atraer", help="Busca el pixel con mayor flujo en un radio de 5 celdas."):
                                            r = 5
                                            y_curr, x_curr = y_pour, x_pour
                                            y_s, y_e = max(0, y_curr-r), min(acc.shape[0], y_curr+r+1)
                                            x_s, x_e = max(0, x_curr-r), min(acc.shape[1], x_curr+r+1)
                                            window = acc[y_s:y_e, x_s:x_e]
                                            if window.size > 0:
                                                loc_max = np.unravel_index(np.nanargmax(window), window.shape)
                                                st.session_state['x_pour_calib'] = int(x_s + loc_max[1])
                                                st.session_state['y_pour_calib'] = int(y_s + loc_max[0])
                                                st.toast(f"✅ Ajustado a: ({st.session_state['x_pour_calib']}, {st.session_state['y_pour_calib']})", icon="🎯")
                                                st.rerun()

                                # 3. Calcular Catchment
                                catch = None
                                try:
                                    catch = grid.catchment(x=x_pour, y=y_pour, fdir=fdir, dirmap=dirmap, xytype='index')
                                    st.session_state['catchment_raster'] = catch
                                except Exception as e: st.error(f"Error: {e}")

                                # 4. Visualización
                                if catch is not None:
                                    # Truco de memoria original
                                    catch_int = np.ascontiguousarray(catch, dtype=np.uint8)
                                    shapes_gen = features.shapes(catch_int, transform=transform)
                                    geoms = [shape(geom) for geom, val in shapes_gen if val > 0]
                                    
                                    if geoms:
                                        gdf_c = gpd.GeoDataFrame({'geometry': geoms}, crs=crs_actual).dissolve()
                                        gdf_calc_4326 = gdf_c.to_crs("EPSG:4326")
                                        gdf_off_4326 = gdf_zona_seleccionada.to_crs("EPSG:4326")
                                        
                                        if modo_viz == "Catchment (Mascara)":
                                            fig = px.choropleth_mapbox(
                                                geojson=gdf_calc_4326.geometry.__geo_interface__,
                                                locations=gdf_calc_4326.index, 
                                                mapbox_style="carto-positron",
                                                center={"lat": gdf_calc_4326.centroid.y.mean(), "lon": gdf_calc_4326.centroid.x.mean()},
                                                zoom=10, 
                                                opacity=0.5, 
                                                color_discrete_sequence=["#0099FF"]
                                            )
                                            # Validación
                                            if not gdf_off_4326.empty:
                                                poly = gdf_off_4326.geometry.iloc[0]
                                                if poly.geom_type == 'Polygon': x, y = poly.exterior.coords.xy
                                                else: x, y = max(poly.geoms, key=lambda a: a.area).exterior.coords.xy
                                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(x), lat=list(y), line={'width':2, 'color':'#00FF00'}, name="Oficial"))
                                            
                                            fig.update_layout(title="Catchment (Área Drenante)", height=600, margin=dict(l=0,r=0,t=30,b=0))
                                            st.plotly_chart(fig, use_container_width=True)

                                        elif modo_viz == "Divisoria (Línea)":
                                            fig = go.Figure()
                                            # Roja (Calculada)
                                            p_c = gdf_calc_4326.geometry.iloc[0]
                                            if p_c.geom_type == 'Polygon': xc, yc = p_c.exterior.coords.xy
                                            else: xc, yc = max(p_c.geoms, key=lambda a: a.area).exterior.coords.xy
                                            fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xc), lat=list(yc), line={'width':3, 'color':'red'}, name="Calculada (DEM)"))
                                            
                                            # Verde (Oficial)
                                            if not gdf_off_4326.empty:
                                                p_o = gdf_off_4326.geometry.iloc[0]
                                                if p_o.geom_type == 'Polygon': xo, yo = p_o.exterior.coords.xy
                                                else: xo, yo = max(p_o.geoms, key=lambda a: a.area).exterior.coords.xy
                                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xo), lat=list(yo), line={'width':2, 'color':'#00FF00'}, name="Oficial (IGAC)"))
                                            
                                            clat = gdf_calc_4326.centroid.y.mean()
                                            clon = gdf_calc_4326.centroid.x.mean()
                                            fig.update_layout(title="Comparativa de Divisorias", mapbox=dict(style="carto-positron", zoom=10, center={"lat": clat, "lon": clon}), height=600, margin=dict(l=0,r=0,t=30,b=0))
                                            st.plotly_chart(fig, use_container_width=True)
                                            
            # --- TAB 6: ÍNDICES Y MODELACIÓN (FASE A + B) ---
            with tab6:
                st.subheader(f"📊 Panel Hidrológico: {nombre_zona}")
                
                # Usamos la geometría oficial para cálculos base
                try:
                    gdf_metric = gdf_zona_seleccionada.to_crs("EPSG:3116")
                    geom = gdf_metric.geometry.iloc[0]
                    
                    # --- FASE A: MORFOMETRÍA ---
                    area_km2 = geom.area / 1e6
                    perimetro_km = geom.length / 1000
                    
                    # Índices de Forma
                    kc = 0.282 * perimetro_km / np.sqrt(area_km2) # Gravelius
                    # Longitud Axial (Aprox. lado mayor del bounding box)
                    bounds = geom.bounds
                    longitud_axial_km = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) / 1000
                    kf = area_km2 / (longitud_axial_km ** 2) # Factor de Forma
                    
                    # Densidad de Drenaje (Requiere ríos calculados)
                    dd_str = "N/A (Calcule ríos primero)"
                    longitud_rios_km = 0
                    if 'gdf_rios' in locals() and gdf_rios is not None:
                        gdf_rios_metric = gdf_rios.to_crs("EPSG:3116")
                        longitud_rios_km = gdf_rios_metric.length.sum() / 1000
                        dd = longitud_rios_km / area_km2
                        dd_str = f"{dd:.2f} km/km²"

                    # Pendientes
                    # Pendiente Media Cuenca (Sm) ya calculada globalmente como slope_mean
                    # Pendiente Cauce Principal (Aproximación: Desnivel / Longitud Axial)
                    desnivel_m = max_el - min_el
                    pendiente_cauce_m_m = desnivel_m / (longitud_axial_km * 1000)
                    
                    # Visualización Fase A
                    st.markdown("##### 📐 Índices Morfométricos")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Área (A)", f"{area_km2:.2f} km²")
                    c2.metric("Perímetro (P)", f"{perimetro_km:.2f} km")
                    c3.metric("Gravelius (Kc)", f"{kc:.3f}", help=">1: Alargada, ~1: Redonda")
                    c4.metric("Densidad Drenaje", dd_str)

                    with st.expander("Ver Tabla Detallada de Parámetros"):
                        # Creamos el DataFrame local 'df_morfo'
                        df_morfo = pd.DataFrame({
                            "Parámetro": ["Área", "Perímetro", "Longitud Axial", "Longitud Total Ríos", "Desnivel (H)", "Pendiente Media Cuenca", "Pendiente Aprox. Cauce"],
                            "Valor": [area_km2, perimetro_km, longitud_axial_km, longitud_rios_km, desnivel_m, slope_mean, pendiente_cauce_m_m * 100],
                            "Unidad": ["km²", "km", "km", "km", "m", "Grados", "%"]
                        })
                        
                        st.dataframe(df_morfo.style.format({"Valor": "{:.3f}"}), use_container_width=True)
                        
                        # CORRECCIÓN: Guardamos 'df_morfo' en la llave 'df_indices' del session_state
                        # Así la Tab 5 podrá encontrarlo para descargarlo
                        st.session_state['df_indices'] = df_morfo 
                    
                    st.markdown("---")
                    
                    # --- FASE B: HIDROLOGÍA SINTÉTICA ---
                    st.markdown("##### ⏱️ Tiempo de Concentración (Tc) y Caudales")
                    st.caption("Estimaciones basadas en fórmulas empíricas (Método Racional).")
                    
                    col_tc, col_q = st.columns(2)
                    
                    with col_tc:
                        st.markdown("**1. Tiempo de Concentración (Tc)**")
                        # Kirpich: Tc (min) = 0.01947 * L^0.77 * S^-0.385 (L en metros, S en m/m)
                        # Usamos longitud axial como proxy de longitud de cauce principal si no hay red detallada
                        L_m = longitud_axial_km * 1000
                        S_mm = pendiente_cauce_m_m
                        
                        if S_mm > 0:
                            tc_kirpich_min = 0.01947 * (L_m**0.77) * (S_mm**-0.385)
                            # California (aprox): Tc = 0.87 * (L^3 / H)^0.385 (L en km, H en m) -> resultado en horas
                            tc_calif_hr = 0.87 * ((longitud_axial_km**3) / desnivel_m)**0.385
                            
                            st.info(f"⏱️ **Kirpich:** {tc_kirpich_min:.1f} min ({tc_kirpich_min/60:.2f} h)")
                            st.write(f"⏱️ **California:** {tc_calif_hr*60:.1f} min ({tc_calif_hr:.2f} h)")
                        else:
                            st.warning("Pendiente nula, no se puede calcular Tc.")
                            tc_kirpich_min = 0

                    with col_q:
                        st.markdown("**2. Caudal Pico (Q) - Método Racional**")
                        
                        # --- CÁLCULO AUTOMÁTICO DE C (COBERTURAS) ---
                        c_sugerido = 0.50 # Default
                        detalle_cob = "No hay datos de cobertura."
                        
                        PATH_COB = "data/Cob25m_WGS84.tif"
                        if land_cover and os.path.exists(PATH_COB):
                            try:
                                # Calcular estadísticas de cobertura para la zona
                                stats_cob = land_cover.calcular_estadisticas_zona(gdf_zona_seleccionada, PATH_COB)
                                if stats_cob:
                                    # Ponderación simple según valores típicos del método racional
                                    # Urbano: 0.85, Cultivo: 0.6, Pasto: 0.5, Bosque: 0.3, Agua: 1.0
                                    c_pond = 0
                                    for cob, pct in stats_cob.items():
                                        peso = pct / 100.0
                                        val_c = 0.5
                                        if "Urbano" in cob or "Industrial" in cob: val_c = 0.85
                                        elif "Cultivo" in cob: val_c = 0.60
                                        elif "Pasto" in cob or "Herbácea" in cob: val_c = 0.45
                                        elif "Bosque" in cob: val_c = 0.30
                                        elif "Agua" in cob: val_c = 1.0
                                        c_pond += val_c * peso
                                    
                                    c_sugerido = c_pond
                                    # Texto resumen
                                    top_3 = sorted(stats_cob.items(), key=lambda x: x[1], reverse=True)[:3]
                                    detalle_cob = ", ".join([f"{k} ({v:.0f}%)" for k,v in top_3])
                            except: pass

                        i_rain = st.slider("Intensidad de Lluvia (I) [mm/h]:", 10, 200, 50, 10)
                        
                        c_runoff = st.slider(
                            "Coeficiente de Escorrentía (C):", 
                            0.1, 1.0, float(c_sugerido), 0.05,
                            help=f"Valor sugerido basado en coberturas satelitales: {c_sugerido:.2f}\nPredomina: {detalle_cob}"
                        )
                        
                        if c_sugerido != 0.5:
                            st.caption(f"🛰️ **C Calculado:** {c_sugerido:.2f} ({detalle_cob})")
                        
                        q_peak = 0.278 * c_runoff * i_rain * area_km2
                        
                        st.metric("Caudal Pico Estimado (Q)", f"{q_peak:.2f} m³/s")
                        st.caption("Fórmula: $Q = 0.278 \cdot C \cdot I \cdot A$")
    
                except Exception as e:
                    st.error(f"Error en cálculos: {e}")

            # --- TAB 7: AMENAZAS (ESPEJOS LÓGICOS) ---
            with tab7:
                st.subheader("🚨 Zonificación de Amenazas Hidrológicas")
                
                if 'acc' in locals() and acc is not None:
                    # Preparar datos comunes
                    min_h = min(slope_deg.shape[0], acc.shape[0])
                    min_w = min(slope_deg.shape[1], acc.shape[1])
                    s_core = slope_deg[:min_h, :min_w]
                    a_core = np.log1p(acc[:min_h, :min_w])
                    
                    # Pestañas para separar los mapas
                    t1, t2 = st.tabs(["🔴 Avenida Torrencial", "🔵 Inundación Plana"])
                    
                    # --- ESPEJO 1: TU CÓDIGO ORIGINAL (Torrencial) ---
                    with t1:
                        st.markdown("**Identificación de zonas críticas donde convergen alta pendiente y alto flujo.**")
                        c_par, c_vis = st.columns([1, 3])
                        with c_par:
                            st.markdown("#### Criterios")
                            s_umb = st.slider("Pendiente Crítica (> Grados)", 15, 45, 30)
                            a_umb = st.slider("Acumulación Log (> Umbral)", 1.0, 10.0, 5.5)
                            
                            st.info("""
                            **Semáforo:**
                            * 🔴 **Muy Alta:** Pendiente Alta + Flujo Alto.
                            * 🟠 **Alta:** Pendiente Alta.
                            * 🟡 **Media:** Flujo Alto (Plano).
                            """)
                        with c_vis:
                            risk = np.zeros_like(s_core, dtype=np.uint8)
                            mask_steep = s_core >= s_umb
                            mask_flow = a_core >= a_umb
                            
                            risk[mask_flow] = 1          # Amarillo
                            risk[mask_steep] = 2         # Naranja
                            risk[mask_steep & mask_flow] = 3 # Rojo
                            
                            colors = [[0.0, "rgba(0,0,0,0)"], [0.33, "#FFD700"], [0.66, "#FF8C00"], [1.0, "#FF0000"]]
                            
                            fig_risk = px.imshow(risk, color_continuous_scale=colors)
                            fig_risk.update_layout(coloraxis_showscale=False, height=550, margin=dict(l=0,r=0,t=0,b=0))
                            fig_risk.update_xaxes(visible=False); fig_risk.update_yaxes(visible=False)
                            st.plotly_chart(fig_risk, use_container_width=True)

                    # --- ESPEJO 2: NUEVO CÓDIGO (Inundación) ---
                    with t2:
                        st.markdown("**Identificación de zonas planas propensas a empozamiento.**")
                        c_par, c_vis = st.columns([1, 3])
                        with c_par:
                            st.markdown("#### Criterios")
                            # Aquí la lógica es inversa: Buscamos pendiente BAJA
                            s_flat = st.slider("Pendiente Plana (< Grados)", 0.5, 10.0, 3.0)
                            a_umb_i = st.slider("Acumulación Río (> Log)", 1.0, 10.0, 5.5, key="a_flood")
                            
                            st.info("""
                            **Semáforo:**
                            * 🔵 **Inundación:** Pendiente Plana + Flujo Alto.
                            * 🟡 **Río Normal:** Flujo Alto (Con pendiente).
                            """)
                        with c_vis:
                            risk_i = np.zeros_like(s_core, dtype=np.uint8)
                            mask_flat = s_core <= s_flat # Condición inversa
                            mask_flow = a_core >= a_umb_i
                            
                            risk_i[mask_flow] = 1          # Amarillo (Río normal)
                            risk_i[mask_flat & mask_flow] = 2 # AZUL (Inundación)
                            
                            # Escala de azules
                            colors_i = [[0.0, "rgba(0,0,0,0)"], [0.5, "#FFD700"], [1.0, "#0099FF"]]
                            
                            fig_i = px.imshow(risk_i, color_continuous_scale=colors_i)
                            fig_i.update_layout(coloraxis_showscale=False, height=550, margin=dict(l=0,r=0,t=0,b=0))
                            fig_i.update_xaxes(visible=False); fig_i.update_yaxes(visible=False)
                            st.plotly_chart(fig_i, use_container_width=True)

                else:
                    st.warning("⚠️ Calcula primero la hidrología.")
                                  
            # --- TAB 5: DESCARGAS (7 COLUMNAS COMPLETA) ---
            with tab5:
                st.subheader("Centro de Descargas")
                st.caption("Descarga los productos generados en las pestañas anteriores.")
                
                # Definimos 7 columnas
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                
                # 1. DEM (TIF)
                with c1:
                    st.write("🏔️ **DEM**")
                    c1.download_button("💾 .TIF", to_tif(arr_elevacion, meta), f"DEM_{nombre_zona}.tif")
                
                # 2. Curvas de Nivel (Vector)
                with c2:
                    st.write("〰️ **Curvas**")
                    if st.session_state['gdf_contours'] is not None:
                        geojson = st.session_state['gdf_contours'].to_json()
                        c2.download_button("💾 .JSON", geojson, f"Curvas_{nombre_zona}.geojson", "application/json")
                    else:
                        st.warning("⚠️ Ver Tab 3D")

                # 3. Pendientes (TIF)
                with c3:
                    st.write("📐 **Pendiente**")
                    slope_meta = meta.copy(); slope_meta.update(dtype=rasterio.float32)
                    c3.download_button("💾 .TIF", to_tif(slope_deg, slope_meta), f"Slope_{nombre_zona}.tif")

                # 4. Datos Hipsométricos (CSV)
                with c4:
                    st.write("📈 **Hipso**")
                    try:
                        # Recálculo rápido para descarga
                        elevs_sort = np.sort(elevs_valid)[::-1]
                        pcts = np.linspace(0, 100, len(elevs_sort))
                        df_hyp = pd.DataFrame({"Porcentaje_Area": pcts, "Altitud": elevs_sort})
                        csv_hyp = df_hyp.to_csv(index=False).encode('utf-8')
                        c4.download_button("💾 .CSV", csv_hyp, f"Hipsometria_{nombre_zona}.csv", "text/csv")
                    except:
                        st.error("Error calc.")

                # 5. Ríos (GEOJSON)
                with c5:
                    st.write("🌊 **Ríos**")
                    if st.session_state['gdf_rios'] is not None:
                        rios_json = st.session_state['gdf_rios'].to_json()
                        c5.download_button("💾 .JSON", rios_json, f"Rios_{nombre_zona}.geojson", "application/json")
                    else:
                        st.warning("⚠️ Ver Tab Hidro")

                # 6. CSV Índices
                with c6:
                    st.write("📊 **Índices**")
                    if st.session_state['df_indices'] is not None:
                        csv_ind = st.session_state['df_indices'].to_csv(index=False).encode('utf-8')
                        c6.download_button("💾 .CSV", csv_ind, f"Indices_{nombre_zona}.csv", "text/csv")
                    else:
                        st.warning("⚠️ Ver Tab Índices")

                # 7. Catchment Raster
                with c7:
                    st.write("🟦 **Cuenca**")
                    if st.session_state['catchment_raster'] is not None:
                        catch_meta = meta.copy(); catch_meta.update(dtype=rasterio.uint8, nodata=0)
                        c7.download_button(
                            "💾 .TIF", 
                            to_tif(st.session_state['catchment_raster'].astype(np.uint8), catch_meta), 
                            f"Catchment_{nombre_zona}.tif"
                        )
                    else:
                        st.warning("⚠️ Calc. Tab Hidro")

else:
    st.info("👈 Selecciona una zona.")


