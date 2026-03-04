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

import sys
# Aseguramos que python encuentre los módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from modules.db_manager import get_engine
except ImportError:
    # Fallback por si la estructura de carpetas varía
    from db_manager import get_engine

# Inicializar conexión
engine = get_engine()

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
                "🌊 Hidrología", "📊 Índices Morfo-métricos", "🚨 Amenazas", "📥 Descargas"
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
                        title="Terreno 3D (Curvas Nativas)", 
                        autosize=True, 
                        height=900, # <--- AUMENTADO DE 600 A 900
                        scene=dict(
                            aspectmode='manual', 
                            aspectratio=dict(x=1, y=1, z=0.2*exag),
                            # Cámara inicial más alejada para ver todo
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

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
                    st.plotly_chart(fig_hyp, use_container_width=True, config={'scrollZoom': True})

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
                    st.plotly_chart(fig_adim, use_container_width=True, config={'scrollZoom': True})
                    
                st.info("""
                **Interpretación Adimensional:**
                * **Curva Convexa (Arriba de la recta):** Cuenca joven, en fase activa de erosión.
                * **Curva Concava (Debajo de la recta):** Cuenca vieja, sedimentada y estabilizada.
                * **Forma de 'S':** Cuenca madura en transición.
                """)

            # --- TAB 4: HIDROLOGÍA (VERSIÓN FINAL INTEGRADA Y CORREGIDA) ---
            with tab4:
                import sys
                sys.setrecursionlimit(20000) # Estabilidad
                
                st.subheader("🌊 Hidrología: Red de Drenaje y Cuencas")
                
                c_conf, c_map = st.columns([1, 3])
                with c_conf:
                    st.markdown("#### ⚙️ Configuración")
                    # Raster de primero en la lista para probar fácil
                    opciones = ["Vectores (Líneas)", "Raster (Acumulación)", "Catchment (Mascara)", "Divisoria (Línea)"]
                    modo_viz = st.radio("Visualización:", opciones)
                    umbral = st.slider("Umbral Acumulación", 10, 5000, 100, 10)

                with c_map:
                    # 1. PROCESAMIENTO (CÁLCULO PURO)
                    import tempfile
                    from shapely.geometry import shape, Point
                    from rasterio import features
                    from rasterio.transform import rowcol
                    
                    grid = None; acc = None; fdir = None
                    crs_actual = meta.get('crs', 'EPSG:3116')

                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                        try:
                            meta_t = meta.copy(); meta_t.update(driver='GTiff', dtype='float64') 
                            with rasterio.open(tmp.name, 'w', **meta_t) as dst: dst.write(arr_elevacion.astype('float64'), 1)
                            grid = Grid.from_raster(tmp.name)
                            dem_grid = grid.read_raster(tmp.name)
                            pit_filled = grid.fill_pits(dem_grid)
                            resolved = grid.resolve_flats(pit_filled)
                            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                            fdir = grid.flowdir(resolved, dirmap=dirmap)
                            acc = grid.accumulation(fdir, dirmap=dirmap)
                            
                            # --- 🌟 MAGIA ESPACIAL: VECTORIZACIÓN Y STRAHLER ---
                            branches = grid.extract_river_network(fdir, acc > umbral, dirmap=dirmap)
                            if branches and len(branches["features"]) > 0:
                                gdf_streams = gpd.GeoDataFrame.from_features(branches["features"], crs=crs_actual)
                                
                                # Longitud real en kilómetros (Proyectado a Magna Sirgas EPSG:3116)
                                gdf_streams_m = gdf_streams.to_crs(epsg=3116)
                                gdf_streams['longitud_km'] = gdf_streams_m.length / 1000.0
                                
                                # Cálculo del Orden de Strahler mediante lectura de píxeles
                                try:
                                    strahler_raster = grid.stream_order(fdir, dirmap=dirmap)
                                    inv_affine = ~grid.affine
                                    orden_list = []
                                    import statistics
                                    for geom in gdf_streams.geometry:
                                        orders = []
                                        for p in list(geom.coords):
                                            try:
                                                c, r = inv_affine * p
                                                val = strahler_raster[int(r), int(c)]
                                                if val > 0: orders.append(val)
                                            except: pass
                                        orden_list.append(int(statistics.mode(orders)) if orders else 1)
                                    gdf_streams['Orden_Strahler'] = orden_list
                                except Exception as e:
                                    gdf_streams['Orden_Strahler'] = 1 # Fallback de seguridad
                                    
                                # Guardamos los ríos en el cerebro global
                                st.session_state['gdf_rios'] = gdf_streams
                                
                                # Resumen estadístico agrupado por Orden
                                df_strahler = gdf_streams.groupby('Orden_Strahler').agg(
                                    Num_Segmentos=('geometry', 'count'),
                                    Longitud_Km=('longitud_km', 'sum')
                                ).reset_index()
                                st.session_state['geomorfo_strahler_df'] = df_strahler
                            else:
                                st.session_state['gdf_rios'] = None
                                st.session_state['geomorfo_strahler_df'] = None
                            # -----------------------------------------------------
                        except Exception as e: 
                            st.warning(f"Error procesando hidrología: {e}")
                        finally: 
                            try: os.remove(tmp.name)
                            except: pass

                    # 2. LÓGICA DE VISUALIZACIÓN
                    if grid is not None and acc is not None:
                        
                        # --- PUNTOS CLAVE ---
                        lat_c, lon_c = 0,0
                        r_smart, c_smart = 0,0
                        
                        if gdf_zona_seleccionada is not None:
                            cent = gdf_zona_seleccionada.to_crs("EPSG:4326").geometry.centroid.iloc[0]
                            lat_c, lon_c = cent.y, cent.x
                        
                        # Salida Smart
                        idx_max = np.nanargmax(acc)
                        r_smart, c_smart = np.unravel_index(idx_max, acc.shape)
                        try:
                            if gdf_zona_seleccionada is not None:
                                gdf_p = gdf_zona_seleccionada.to_crs(meta['crs'])
                                shapes = ((g, 1) for g in gdf_p.geometry)
                                mask_poly = features.rasterize(shapes, out_shape=acc.shape, transform=transform, fill=0, dtype='uint8')
                                acc_masked = np.where(mask_poly==1, acc, -1)
                                idx_s = np.argmax(acc_masked)
                                r_smart, c_smart = np.unravel_index(idx_s, acc_masked.shape)
                        except: pass

                        # --- CAJA DE INFORMACIÓN ---
                        with st.expander(f"📍 Puntos Clave: {nombre_zona}", expanded=True):
                            k1, k2, k3 = st.columns(3)
                            with k1:
                                st.markdown("**Centro**")
                                st.caption(f"{lat_c:.4f}, {lon_c:.4f}")
                            with k2:
                                st.markdown("**Salida Detectada**")
                                st.code(f"X:{c_smart} Y:{r_smart}")
                            with k3:
                                if st.button("🎯 Usar Salida", type="primary"):
                                    st.session_state['x_pour_calib'] = int(c_smart)
                                    st.session_state['y_pour_calib'] = int(r_smart)
                                    st.rerun()

                        # --- MAPAS ---
                        
                        # CASO A: RASTER (USAMOS IF AQUÍ PARA QUE SEA LA PRIMERA OPCIÓN LÓGICA)
                        if modo_viz == "Raster (Acumulación)":
                            # Cálculo dinámico: Si el mapa es gigante (>1000px), reducimos resolución
                            h, w = acc.shape
                            factor = 1
                            if h > 1000 or w > 1000: factor = int(max(h, w) / 800)
                            
                            fig = px.imshow(
                                np.log1p(acc[::factor, ::factor]), 
                                color_continuous_scale='Blues', 
                                title=f"Acumulación de Flujo (Escala 1:{factor})",
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)

                        # CASO B: GEOGRÁFICO (Vectores, Catchment, etc.)
                        else:
                            fig = go.Figure()
                            
                            # 1. Oficial (Verde)
                            if gdf_zona_seleccionada is not None:
                                poly = gdf_zona_seleccionada.to_crs("EPSG:4326").geometry.iloc[0]
                                if poly.geom_type=='Polygon': xx,yy=poly.exterior.coords.xy
                                else: xx,yy=max(poly.geoms, key=lambda a:a.area).exterior.coords.xy
                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xx), lat=list(yy), line={'width':2, 'color':'#00FF00'}, name="Oficial"))

                            # 2. VECTORES (LÍNEAS CALCULADAS EN VIVO)
                            if modo_viz == "Vectores (Líneas)":
                                r_viz = st.session_state.get('gdf_rios')
                                if r_viz is not None and not r_viz.empty:
                                    r_viz_4326 = r_viz.to_crs("EPSG:4326")
                                    l, lt, tx = [], [], []
                                    
                                    for _, row in r_viz_4326.iterrows():
                                        g = row.geometry
                                        orden = row.get('Orden_Strahler', 1)
                                        n = f"Río Orden {orden}"
                                        
                                        if g.geom_type == 'LineString': p = [g]
                                        elif g.geom_type == 'MultiLineString': p = g.geoms
                                        else: continue
                                        
                                        for s in p: 
                                            x, y = s.xy
                                            l.extend(list(x) + [None])
                                            lt.extend(list(y) + [None])
                                            tx.extend([n] * (len(x) + 1))
                                    
                                    # El color variará según el orden gracias al hover
                                    fig.add_trace(go.Scattermapbox(mode="lines", lon=l, lat=lt, text=tx, hoverinfo='text', line={'width': 2, 'color': '#0044FF'}, name="Red Drenaje (Modelo)"))
                                    st.success(f"✅ {len(r_viz)} tramos de río calculados. Deslice el ratón sobre ellos para ver su Orden de Strahler.")
                                else:
                                    st.warning(f"No se detectaron ríos con un umbral de {umbral}. Disminuya el valor del control deslizante para forzar la aparición de arroyos menores.")

                            # 3. CATCHMENT / DIVISORIA
                            elif modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                                if 'x_pour_calib' not in st.session_state: st.session_state['x_pour_calib']=int(c_smart); st.session_state['y_pour_calib']=int(r_smart)
                                st.markdown("##### 🔧 Ajuste Manual")
                                c1,c2 = st.columns(2)
                                with c1: xp=st.number_input("X:", value=st.session_state['x_pour_calib'])
                                with c2: yp=st.number_input("Y:", value=st.session_state['y_pour_calib'])
                                st.session_state['x_pour_calib']=xp; st.session_state['y_pour_calib']=yp
                                
                                try:
                                    catch = grid.catchment(x=xp, y=yp, fdir=fdir, dirmap=dirmap, xytype='index')
                                    catch_int = np.ascontiguousarray(catch, dtype=np.uint8)
                                    shapes_gen = features.shapes(catch_int, transform=transform)
                                    geoms = [shape(g) for g, v in shapes_gen if v > 0]
                                    if geoms:
                                        gdf_c = gpd.GeoDataFrame({'geometry': geoms}, crs=crs_actual).dissolve().to_crs("EPSG:4326")
                                        if modo_viz=="Catchment (Mascara)":
                                            fig.add_trace(go.Choroplethmapbox(geojson=gdf_c.geometry.__geo_interface__, locations=gdf_c.index, z=[1]*len(gdf_c), colorscale=[[0,'#3366CC'],[1,'#3366CC']], showscale=False, marker_opacity=0.5, name="Calculada"))
                                        else:
                                            xc, yc = gdf_c.geometry.iloc[0].exterior.coords.xy
                                            fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xc), lat=list(yc), line={'width':3, 'color':'red'}, name="Divisoria"))
                                        st.success(f"Área: {gdf_c.to_crs('EPSG:3116').area.sum()/1e6:.2f} km²")
                                except: pass

                            fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center={"lat": lat_c, "lon": lon_c}, height=600, margin=dict(l=0,r=0,t=0,b=0))
                            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

                    else: st.warning("Procesando...")
                        
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
                    if st.session_state.get('gdf_rios') is not None:
                        gdf_rios_st = st.session_state['gdf_rios']
                        longitud_rios_km = gdf_rios_st['longitud_km'].sum()
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

            # --- TAB 7: AMENAZAS (CONTEXTO GEOESPACIAL COMPLETO) ---
            with tab7:
                st.subheader("🚨 Zonificación de Amenazas Hidrológicas")

                # --- FUNCIÓN DE MAPA SUPERPUESTO (CON RÍOS OFICIALES) ---
                def mapa_amenaza_contexto(mask_binaria, color_amenaza, titulo, gdf_oficial):
                    from rasterio import features
                    from shapely.geometry import shape
                    
                    fig = go.Figure()

                    # CAPA 1: Polígono Oficial (Verde)
                    if gdf_oficial is not None:
                        gdf_4326 = gdf_oficial.to_crs("EPSG:4326")
                        poly = gdf_4326.geometry.iloc[0]
                        if poly.geom_type == 'Polygon': xo, yo = poly.exterior.coords.xy
                        else: xo, yo = max(poly.geoms, key=lambda a: a.area).exterior.coords.xy
                        fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xo), lat=list(yo), line={'width': 3, 'color': '#00FF00'}, name="Cuenca Oficial"))
                        c_lat, c_lon = gdf_4326.centroid.y.mean(), gdf_4326.centroid.x.mean()
                    else: c_lat, c_lon = 6.2, -75.5

                    # CAPA 2: RED DE DRENAJE (RECORTE EXACTO PARA NO BLOQUEAR)
                    try:
                        # 1. Cargar BD
                        try: r = gpd.read_postgis("SELECT * FROM red_drenaje", engine, geom_col='geometry')
                        except: r = gpd.read_postgis("SELECT * FROM red_drenaje", engine, geom_col='geom')
                        
                        # 2. Recorte Estricto en Metros (Igual que en Hidrología)
                        if gdf_oficial is not None:
                            poly_m = gdf_oficial.to_crs("EPSG:3116")
                            r_m = r.to_crs("EPSG:3116")
                            
                            # Clip exacto (elimina lo que sobre)
                            r_clip = gpd.clip(r_m, poly_m)
                            
                            if not r_clip.empty:
                                # Solo convertimos a WGS84 lo que quedó dentro
                                r_viz = r_clip.to_crs("EPSG:4326")
                                l, lt = [], []
                                for g in r_viz.geometry:
                                    if g.geom_type=='LineString': x,y=g.xy
                                    elif g.geom_type=='MultiLineString': 
                                        for s in g.geoms: x,y=s.xy; l.extend(list(x)+[None]); lt.extend(list(y)+[None]); continue
                                    else: continue
                                    l.extend(list(x)+[None]); lt.extend(list(y)+[None])
                                
                                fig.add_trace(go.Scattermapbox(mode="lines", lon=l, lat=lt, line={'width':1.5, 'color':'#0044FF'}, name="Red Drenaje"))
                    except: pass

                    # CAPA 3: Amenaza Vectorizada
                    mask_safe = np.ascontiguousarray(mask_binaria, dtype=np.uint8)
                    shapes_gen = features.shapes(mask_safe, transform=transform)
                    geoms = [shape(g) for g, v in shapes_gen if v == 1]
                    
                    if geoms:
                        gdf_threat = gpd.GeoDataFrame({'geometry': geoms}, crs=meta['crs'])
                        gdf_threat = gdf_threat.to_crs("EPSG:3116")
                        # Filtro de ruido: eliminamos polígonos menores a 900m2
                        gdf_threat = gdf_threat[gdf_threat.geometry.area > 900]
                        
                        if not gdf_threat.empty:
                            gdf_threat['geometry'] = gdf_threat.simplify(20).to_crs("EPSG:4326")
                            fig.add_trace(go.Choroplethmapbox(geojson=gdf_threat.geometry.__geo_interface__, locations=gdf_threat.index, z=[1]*len(gdf_threat), colorscale=[[0, color_amenaza], [1, color_amenaza]], showscale=False, marker_opacity=0.6, name="Zona Amenaza"))
                        else: st.info("✅ Zona segura (Amenazas pequeñas filtradas).")
                    else: st.info("✅ Sin amenazas detectadas.")

                    fig.update_layout(title=titulo, mapbox_style="carto-positron", mapbox_zoom=12, mapbox_center={"lat": c_lat, "lon": c_lon}, height=600, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

                # --- ANÁLISIS AI ---
                def caja_analisis_ai(mask_riesgo, tipo):
                    total = mask_riesgo.size
                    afectado = np.sum(mask_riesgo)
                    pct = (afectado / total) * 100
                    color = "red" if "Torrencial" in tipo else "#0099FF"
                    nivel = "CRÍTICO" if pct > 5 else ("Medio" if pct > 1 else "Bajo")
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding: 15px; background-color: rgba(240,242,246,0.5); border-radius: 5px; margin-bottom: 20px;">
                        <strong style="color: {color}; font-size: 1.1em;">📊 Diagnóstico: {tipo}</strong>
                        <ul style="margin-bottom: 0;">
                            <li><b>Área Afectada:</b> {pct:.2f}% de la zona visible.</li>
                            <li><b>Nivel de Alerta:</b> {nivel}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # --- LÓGICA PRINCIPAL ---
                if 'acc' in locals() and acc is not None and 'slope_deg' in locals():
                    # Recortes seguros
                    min_h = min(slope_deg.shape[0], acc.shape[0])
                    min_w = min(slope_deg.shape[1], acc.shape[1])
                    s_core = slope_deg[:min_h, :min_w]
                    acc_raw = acc[:min_h, :min_w]
                    a_core_log = np.log1p(acc_raw)
                    
                    t1, t2 = st.tabs(["🔴 Avenida Torrencial", "🔵 Inundación (TWI)"])
                    
                    # 1. TORRENCIAL
                    with t1:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.markdown("#### Parámetros Físicos")
                            s_range = st.slider("Pendiente Crítica (°)", 0.0, 60.0, (5.0, 40.0), key="s_torrencial")
                            a_umb = st.slider("Acumulación (Log)", 4.0, 9.0, 6.0, key="a_torrencial")
                            st.warning("Detectando flujos de alta energía.")
                        with c2:
                            mask_t = (s_core >= s_range[0]) & (s_core <= s_range[1]) & (a_core_log >= a_umb)
                            caja_analisis_ai(mask_t, "Avenida Torrencial")
                            # Pasamos gdf_zona_seleccionada para dibujar el borde verde
                            mapa_amenaza_contexto(mask_t, "red", "Amenaza: Avenida Torrencial", gdf_zona_seleccionada)

                    # 2. INUNDACIÓN
                    with t2:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.markdown("#### Índice TWI")
                            # Cálculo TWI robusto
                            with st.spinner("Procesando terreno..."):
                                s_rad = np.deg2rad(s_core)
                                tan_s = np.tan(s_rad)
                                tan_s = np.where(tan_s < 0.001, 0.001, tan_s)
                                resolucion = 30.0 # Aprox Sentinel/SRTM
                                area_esp = acc_raw * resolucion
                                twi = np.log((area_esp + 1) / tan_s)
                            
                            twi_val = st.slider("Umbral Humedad", 5.0, 25.0, 10.0, key="twi_slider")
                            strict = st.checkbox("Solo planos (< 5°)", value=True)
                            st.info("Detectando zonas de empozamiento.")
                        with c2:
                            if strict: mask_i = (twi >= twi_val) & (s_core <= 5)
                            else: mask_i = (twi >= twi_val)
                            
                            caja_analisis_ai(mask_i, "Inundación")
                            mapa_amenaza_contexto(mask_i, "#0099FF", f"Amenaza: Inundación (TWI > {twi_val})", gdf_zona_seleccionada)
                else:
                    st.warning("⚠️ Calcula la Hidrología en el Tab 4 primero.")
                    
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
