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
    if _gdf_corte is None or _gdf_corte.empty: 
        return None, None, None
    
    try:
        if not os.path.exists(ruta_dem): 
            return None, None, None
            
        # 1. Blindaje de Geometría (Repara polígonos inválidos)
        # Esto evita errores silenciosos al recortar
        geometria_valida = _gdf_corte.copy()
        geometria_valida['geometry'] = geometria_valida.buffer(0) 

        with rasterio.open(ruta_dem) as src:
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

            # --- TAB 4: HIDROLOGÍA (OPTIMIZADO Y BLINDADO) ---
            with tab4:
                st.subheader("🌊 Hidrología: Red de Drenaje y Cuencas")
                
                # Configuración en columnas
                c_conf, c_map = st.columns([1, 3])
                
                with c_conf:
                    st.markdown("#### ⚙️ Configuración")
                    opciones_viz = [
                        "Vectores (Líneas)", 
                        "Catchment (Mascara)",
                        "Divisoria (Línea)",
                        "Raster (Acumulación)" 
                    ]
                    modo_viz = st.radio("Visualización:", opciones_viz)
                    
                    # Slider de umbral (Fundamental para definir qué es río y qué no)
                    umbral = st.slider("Umbral Acumulación", 
                                     min_value=10, max_value=5000, value=100, step=10, 
                                     help="Celdas mínimas drenando a un punto para considerarlo río.")

                with c_map:
                    # 1. PREPARACIÓN DE DATOS (PYSHEDS)
                    import tempfile
                    from shapely.geometry import shape, LineString, MultiLineString, Point
                    
                    grid = None
                    acc = None
                    fdir = None
                    crs_actual = meta.get('crs', 'EPSG:3116')

                    # Procesamiento
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                        try:
                            # Escribir raster temporal para pysheds
                            meta_temp = meta.copy()
                            meta_temp.update(driver='GTiff', dtype='float64') 
                            with rasterio.open(tmp.name, 'w', **meta_temp) as dst:
                                dst.write(arr_elevacion.astype('float64'), 1)
                            
                            # Inicializar Grid
                            grid = Grid.from_raster(tmp.name)
                            dem_grid = grid.read_raster(tmp.name)
                            
                            # Operaciones hidrológicas (Fill Pits -> Flow Dir -> Accumulation)
                            pit_filled = grid.fill_pits(dem_grid)
                            resolved = grid.resolve_flats(pit_filled)
                            
                            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                            fdir = grid.flowdir(resolved, dirmap=dirmap)
                            acc = grid.accumulation(fdir, dirmap=dirmap)

                        except Exception as e:
                            st.error(f"Error procesando hidrología: {e}")
                        finally: 
                            try: os.remove(tmp.name)
                            except: pass

                    # 2. LÓGICA DE INTERACCIÓN Y ANÁLISIS
                    if grid is not None and acc is not None:
                        
                        # --- SECCIÓN NUEVA: LOCALIZADOR DE PUNTOS CLAVE ---
                        with st.expander("📍 Localizador de Puntos Clave (Calibración)", expanded=False):
                            c_info1, c_info2, c_info3 = st.columns(3)
                            
                            # A. Punto de Máxima Acumulación (Salida natural del río principal)
                            idx_max_acc = np.nanargmax(acc)
                            y_acc, x_acc = np.unravel_index(idx_max_acc, acc.shape)
                            
                            # B. Punto Más Alto (Elevación Máxima - Cabecera potencial)
                            idx_max_elev = np.nanargmax(arr_elevacion)
                            y_high, x_high = np.unravel_index(idx_max_elev, arr_elevacion.shape)
                            
                            # C. Punto Más Bajo (Elevación Mínima - Salida natural topográfica)
                            idx_min_elev = np.nanargmin(arr_elevacion)
                            y_low, x_low = np.unravel_index(idx_min_elev, arr_elevacion.shape)

                            # Botones para setear
                            with c_info1:
                                st.metric("Máx. Acumulación", f"x:{x_acc}, y:{y_acc}")
                                if st.button("Usar como Desfogue", key="btn_use_acc"):
                                    st.session_state['x_pour_calib'] = int(x_acc)
                                    st.session_state['y_pour_calib'] = int(y_acc)
                                    st.rerun()
                            
                            with c_info2:
                                st.metric("Punto Más Bajo", f"x:{x_low}, y:{y_low}")
                                if st.button("Usar como Desfogue", key="btn_use_low"):
                                    st.session_state['x_pour_calib'] = int(x_low)
                                    st.session_state['y_pour_calib'] = int(y_low)
                                    st.rerun()

                            with c_info3:
                                st.metric("Punto Más Alto", f"x:{x_high}, y:{y_high}")
                                st.caption("Cabecera (No usar como desfogue)")

                        # --- CALIBRACIÓN MANUAL DEL DESFOGUE ---
                        # Inicializar si no existe
                        if 'x_pour_calib' not in st.session_state:
                            st.session_state['x_pour_calib'] = int(x_acc)
                            st.session_state['y_pour_calib'] = int(y_acc)

                        if modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                            st.markdown("##### 🔧 Ajuste Fino del Punto de Cierre (Outlet)")
                            c_adj, c_tools = st.columns([2, 2])
                            
                            with c_adj:
                                x_p = st.number_input("Coord X (Columna):", value=st.session_state['x_pour_calib'], step=1)
                                y_p = st.number_input("Coord Y (Fila):", value=st.session_state['y_pour_calib'], step=1)
                                # Actualizar estado
                                st.session_state['x_pour_calib'] = x_p
                                st.session_state['y_pour_calib'] = y_p

                            with c_tools:
                                st.info("💡 Si la cuenca es muy pequeña, usa el imán.")
                                if st.button("🧲 IMÁN (Atraer al río más cercano)"):
                                    # Buscar en un radio de 20 pixeles el punto de mayor acumulación
                                    r = 20
                                    y0, y1 = max(0, y_p-r), min(acc.shape[0], y_p+r+1)
                                    x0, x1 = max(0, x_p-r), min(acc.shape[1], x_p+r+1)
                                    win = acc[y0:y1, x0:x1]
                                    
                                    if win.size > 0:
                                        # Encontrar max en la ventana local
                                        max_local_idx = np.nanargmax(win)
                                        loc_y, loc_x = np.unravel_index(max_local_idx, win.shape)
                                        
                                        # Convertir a coordenadas globales
                                        new_x = x0 + loc_x
                                        new_y = y0 + loc_y
                                        
                                        st.session_state['x_pour_calib'] = int(new_x)
                                        st.session_state['y_pour_calib'] = int(new_y)
                                        st.success(f"Ajustado a río principal en: {new_x}, {new_y}")
                                        st.rerun()

                        # 3. VISUALIZACIÓN
                        # A. RASTER
                        if modo_viz == "Raster (Acumulación)":
                            log_acc = np.log1p(acc)
                            fig = px.imshow(log_acc, color_continuous_scale='Blues', title="Acumulación de Flujo (Log)")
                            fig.update_layout(height=600)
                            fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True)

                        # B. VECTORES (RÍOS)
                        elif modo_viz == "Vectores (Líneas)":
                            gdf_rios = extraer_vectores_rios(grid, fdir, acc, umbral, crs_actual)
                            if gdf_rios is not None:
                                gdf_4326 = gdf_rios.to_crs("EPSG:4326")
                                # ... (Código de ploteo de líneas igual al anterior) ...
                                # (He simplificado esta parte en el texto para no hacerlo eterno, 
                                # pero usa tu lógica de 'lons, lats' o geopandas plot)
                                
                                # Visualización rápida con Plotly Express para Geodataframes (más moderno)
                                lat_c = gdf_4326.geometry.centroid.y.mean()
                                lon_c = gdf_4326.geometry.centroid.x.mean()
                                
                                fig = px.choropleth_mapbox(
                                    geojson=gdf_4326.geometry.__geo_interface__,
                                    locations=gdf_4326.index,
                                    color_discrete_sequence=["#0077BE"],
                                    mapbox_style="carto-positron",
                                    zoom=11, center={"lat": lat_c, "lon": lon_c},
                                    opacity=0.8
                                )
                                fig.update_layout(title=f"Red de Drenaje (Umbral: {umbral})", height=600)
                                st.plotly_chart(fig, use_container_width=True)

                        # C. CATCHMENT (CUENCA)
                        elif modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                            try:
                                # Calcular Catchment
                                catch = grid.catchment(x=x_p, y=y_p, fdir=fdir, dirmap=dirmap, xytype='index')
                                
                                # Vectorizar resultado
                                catch_int = np.ascontiguousarray(catch, dtype=np.uint8)
                                shapes_gen = features.shapes(catch_int, transform=transform)
                                geoms = [shape(g) for g, v in shapes_gen if v > 0]
                                
                                if geoms:
                                    gdf_c = gpd.GeoDataFrame({'geometry': geoms}, crs=crs_actual).dissolve()
                                    gdf_4326 = gdf_c.to_crs("EPSG:4326")
                                    
                                    # Crear punto de desfogue para visualizarlo
                                    # Necesitamos las coordenadas Lat/Lon del punto x_p, y_p para pintarlo
                                    # Usamos affine transform para pasar de pixel a coordenadas proy, luego a 4326
                                    affine = meta['transform']
                                    coords_proy = affine * (x_p + 0.5, y_p + 0.5) # +0.5 para centro de pixel
                                    pt_gdf = gpd.GeoDataFrame(
                                        {'geometry': [Point(coords_proy[0], coords_proy[1])]}, 
                                        crs=crs_actual
                                    ).to_crs("EPSG:4326")
                                    pt_lon = pt_gdf.geometry.iloc[0].x
                                    pt_lat = pt_gdf.geometry.iloc[0].y

                                    lat_c = gdf_4326.centroid.y.mean()
                                    lon_c = gdf_4326.centroid.x.mean()

                                    fig = go.Figure()

                                    # 1. Pintar Cuenca
                                    if modo_viz == "Catchment (Mascara)":
                                        fig = px.choropleth_mapbox(
                                            geojson=gdf_4326.geometry.__geo_interface__,
                                            locations=gdf_4326.index,
                                            mapbox_style="carto-positron",
                                            center={"lat": lat_c, "lon": lon_c}, zoom=11,
                                            opacity=0.5, color_discrete_sequence=["#0099FF"]
                                        )
                                    else: # Solo Línea
                                        poly = gdf_4326.geometry.iloc[0]
                                        if poly.geom_type == 'Polygon': xx, yy = poly.exterior.coords.xy
                                        else: xx, yy = max(poly.geoms, key=lambda a: a.area).exterior.coords.xy
                                        fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xx), lat=list(yy), line={'width':3, 'color':'red'}, name="Divisoria"))
                                        fig.update_layout(mapbox_style="carto-positron", mapbox_center={"lat": lat_c, "lon": lon_c}, mapbox_zoom=11)

                                    # 2. 🔥 PINTAR EL PUNTO DE DESFOGUE (IMPORTANTE PARA DEBUG)
                                    fig.add_trace(go.Scattermapbox(
                                        mode="markers",
                                        lon=[pt_lon], lat=[pt_lat],
                                        marker={'size': 12, 'color': 'red', 'symbol': 'circle'},
                                        name="Punto de Desfogue"
                                    ))

                                    fig.update_layout(height=600, margin=dict(l=0,r=0,t=30,b=0), title="Análisis de Cuenca")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Info de área
                                    area_km2 = gdf_c.area.sum() / 1e6
                                    st.success(f"✅ Cuenca delimitada con éxito. Área: {area_km2:.2f} km²")

                                else:
                                    st.warning("⚠️ La cuenca generada es nula. El punto de desfogue probablemente está fuera del flujo acumulado.")
                                    st.info("Prueba usando el botón '🧲 IMÁN' o selecciona 'Usar como Desfogue' en el Localizador de Puntos.")

                            except Exception as e:
                                st.error(f"Error en trazado: {e}")

                    else:
                        st.warning("Esperando cálculo hidrológico...")
                                            
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
                        
                        st.dataframe(df_morfo.style.format({"Valor": "{:.2f}"}), use_container_width=True)
                        
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

            # --- TAB 7: AMENAZAS (MODELACIÓN FÍSICA AVANZADA) ---
            with tab7:
                st.subheader("🚨 Zonificación de Amenazas Hidrológicas")

                # --- 1. FUNCIÓN DE MAPA OPTIMIZADA (Anti-Pantalla Blanca) ---
                def mapa_con_fondo(mask_binaria, color_hex, titulo):
                    from rasterio import features
                    from shapely.geometry import shape
                    
                    # A. Convertir a formato ligero
                    mask_safe = np.ascontiguousarray(mask_binaria, dtype=np.uint8)
                    
                    # B. Vectorizar solo zonas activas
                    shapes_gen = features.shapes(mask_safe, transform=transform)
                    geoms = []
                    for g, v in shapes_gen:
                        if v == 1:
                            geoms.append(shape(g))
                    
                    if not geoms:
                        st.info("✅ No se detectan zonas de riesgo con estos parámetros.")
                        return

                    # C. Crear DataFrame y Simplificar
                    gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=meta['crs'])
                    
                    try:
                        # Proyectar a metros para medir área
                        if gdf.crs.to_string() != "EPSG:3116":
                            gdf = gdf.to_crs("EPSG:3116")
                        
                        # FILTRO 1: Eliminar ruido (polígonos < 900m2)
                        gdf = gdf[gdf.geometry.area > 900]
                        
                        # FILTRO 2: Simplificar vértices (Tolerancia 20m)
                        gdf['geometry'] = gdf.simplify(tolerance=20)
                        
                        # Volver a Lat/Lon para el mapa
                        gdf = gdf.to_crs("EPSG:4326")
                    except Exception as e:
                        st.warning(f"Aviso de proyección: {e}")
                        if gdf.crs.to_string() != "EPSG:4326":
                            gdf = gdf.to_crs("EPSG:4326")

                    if gdf.empty:
                        st.success("✅ Zona segura (Ruido filtrado).")
                        return

                    # D. Visualizar
                    c_lat = gdf.geometry.centroid.y.mean()
                    c_lon = gdf.geometry.centroid.x.mean()
                    
                    fig = px.choropleth_mapbox(
                        geojson=gdf.geometry.__geo_interface__,
                        locations=gdf.index,
                        mapbox_style="carto-positron",
                        center={"lat": c_lat, "lon": c_lon}, 
                        zoom=12,
                        opacity=0.5, 
                        color_discrete_sequence=[color_hex]
                    )
                    fig.update_layout(title=titulo, margin=dict(l=0,r=0,t=30,b=0), height=550)
                    st.plotly_chart(fig, use_container_width=True)

                # --- 2. FUNCIÓN DE DIAGNÓSTICO AI ---
                def caja_analisis_ai(mask_riesgo, tipo):
                    total = mask_riesgo.size
                    afectado = np.sum(mask_riesgo)
                    pct = (afectado / total) * 100
                    
                    color = "red" if "Torrencial" in tipo else "#0099FF"
                    nivel = "Bajo" if pct < 1 else ("Medio" if pct < 5 else "CRÍTICO")
                    criterio = "Energía cinética en canales de transporte." if "Torrencial" in tipo else "Saturación topográfica (TWI) alta."
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding: 15px; background-color: rgba(240,242,246,0.5); border-radius: 5px; margin-bottom: 20px;">
                        <strong style="color: {color}; font-size: 1.1em;">🤖 Diagnóstico Inteligente ({tipo})</strong>
                        <ul style="margin-bottom: 0;">
                            <li><b>Cobertura:</b> {pct:.2f}% del área.</li>
                            <li><b>Nivel:</b> {nivel}</li>
                            <li><b>Física:</b> {criterio}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # --- LÓGICA PRINCIPAL (Ahora dentro de Tab 7 con 16 espacios) ---
                if 'acc' in locals() and acc is not None and 'slope_deg' in locals():
                    # Recortes de seguridad
                    min_h = min(slope_deg.shape[0], acc.shape[0])
                    min_w = min(slope_deg.shape[1], acc.shape[1])
                    s_core = slope_deg[:min_h, :min_w]
                    acc_raw = acc[:min_h, :min_w]
                    
                    # Log para visualización, pero RAW para cálculos físicos
                    a_core_log = np.log1p(acc_raw) 
                    
                    t1, t2 = st.tabs(["🔴 Avenida Torrencial", "🔵 Inundación (TWI)"])
                    
                    # 1. AVENIDA TORRENCIAL
                    with t1:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.markdown("#### Energía de Flujo")
                            # Rangos ajustados a la realidad andina
                            s_range = st.slider("Pendiente Crítica (°)", 0.0, 60.0, (5.0, 40.0), 
                                              help="Rango donde el agua gana velocidad y transporta material.")
                            a_umb = st.slider("Caudal (Log)", 4.0, 9.0, 6.0)
                            st.error("Modelando Flujos Rápidos")
                            
                        with c2:
                            # El agua debe existir (a_umb) Y la pendiente debe permitir transporte (s_range)
                            mask_t = (s_core >= s_range[0]) & (s_core <= s_range[1]) & (a_core_log >= a_umb)
                            caja_analisis_ai(mask_t, "Avenida Torrencial")
                            mapa_con_fondo(mask_t, "red", "Amenaza: Flujo de Escombros / Av. Torrencial")

                    # 2. INUNDACIÓN (TWI CIENTÍFICO)
                    with t2:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.markdown("#### Índice Topográfico (TWI)")
                            with st.spinner("Calculando saturación de suelos..."):
                                slope_rad = np.deg2rad(s_core)
                                tan_slope = np.tan(slope_rad)
                                tan_slope = np.where(tan_slope < 0.001, 0.001, tan_slope)
                                
                                resolucion_pixel = 30.0  
                                specific_catchment_area = acc_raw * resolucion_pixel
                                
                                twi = np.log((specific_catchment_area + 1) / tan_slope)
                            
                            twi_val = st.slider("Sensibilidad de Humedad", 5.0, 25.0, 10.0, 
                                              help="Valores > 10 suelen indicar saturación/río.")
                            
                            strict_flat = st.checkbox("Restringir a zonas planas (< 5°)", value=True)
                            st.info("Modelando Empozamiento")
                            
                        with c2:
                            if strict_flat:
                                mask_i = (twi >= twi_val) & (s_core <= 5)
                            else:
                                mask_i = (twi >= twi_val)
                            
                            caja_analisis_ai(mask_i, "Inundación Plana")
                            mapa_con_fondo(mask_i, "#0099FF", f"Amenaza: Inundación (TWI > {twi_val})")
                else:
                    st.warning("⚠️ Ve a la pestaña 'Hidrología' primero para calcular el flujo de agua.")

            # --- TAB 5: DESCARGAS (ALINEADO A 12 ESPACIOS) ---
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
