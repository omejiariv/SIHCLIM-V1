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
                
                # --- ECUACIÓN HIPSOMÉTRICA MATEMÁTICA ---
                st.markdown("---")
                st.markdown("#### 🧮 Ecuación Hipsométrica $A(h)$")
                st.caption("Modelo matemático que relaciona la altitud ($h$) con el porcentaje de área acumulada ($A$). Permite calcular qué porcentaje de la cuenca se inundaría o afectaría a una cota específica.")
                
                # Ajuste polinómico de 3er grado: Area(%) = f(Elevación)
                z_poly = np.polyfit(elevs_sorted[idx], x_pct[idx], 3)
                
                col_eq1, col_eq2 = st.columns([2, 1])
                with col_eq1:
                    st.latex(f"A(h) = {z_poly[0]:.3e} \\cdot h^3 {z_poly[1]:+.3e} \\cdot h^2 {z_poly[2]:+.3e} \\cdot h {z_poly[3]:+.3e}")
                
                with col_eq2:
                    p_func = np.poly1d(z_poly)
                    y_pred = p_func(elevs_sorted[idx])
                    y_real = x_pct[idx]
                    ss_res = np.sum((y_real - y_pred) ** 2)
                    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    st.metric("Precisión del Ajuste ($R^2$)", f"{r2:.4f}")
                    
            # --- TAB 4: HIDROLOGÍA (VERSIÓN FINAL INTEGRADA Y CORREGIDA) ---
            with tab4:
                import sys
                sys.setrecursionlimit(20000) # Estabilidad
                
                st.subheader(f"🌊 Hidrología: Red de Drenaje y Cuencas - {nombre_zona}")
                
                c_conf, c_map = st.columns([1, 3])
                with c_conf:
                    st.markdown("#### ⚙️ Configuración")
                    # Raster de primero en la lista para probar fácil
                    opciones = ["Vectores (Líneas)", "Raster (Acumulación)", "Catchment (Mascara)", "Divisoria (Línea)"]
                    modo_viz = st.radio("Visualización:", opciones)
                    umbral = st.slider("Umbral Acumulación", 5, 5000, 100, 5)

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
                            # 1. Blindaje de bordes: Convertimos NaN a -9999 para que PySheds reconozca las paredes
                            meta_t = meta.copy()
                            meta_t.update(driver='GTiff', dtype='float64', nodata=-9999.0)
                            dem_clean = np.where(np.isnan(arr_elevacion), -9999.0, arr_elevacion)
                            
                            with rasterio.open(tmp.name, 'w', **meta_t) as dst: 
                                dst.write(dem_clean.astype('float64'), 1)
                            
                            grid = Grid.from_raster(tmp.name)
                            dem_grid = grid.read_raster(tmp.name, nodata=-9999.0)
                            
                            # --- 🌟 LA CURA PARA LOS RÍOS ROTOS ---
                            # Usamos fill_depressions (más poderoso) en lugar de fill_pits
                            flooded = grid.fill_depressions(dem_grid)
                            resolved = grid.resolve_flats(flooded)
                            
                            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                            fdir = grid.flowdir(resolved, dirmap=dirmap)
                            acc = grid.accumulation(fdir, dirmap=dirmap)
                            
                            # Vectorización y Strahler
                            branches = grid.extract_river_network(fdir, acc > umbral, dirmap=dirmap)
                            if branches and len(branches["features"]) > 0:
                                gdf_streams_raw = gpd.GeoDataFrame.from_features(branches["features"], crs=crs_actual)
                                
                                # --- ✂️ EL BISTURÍ ESPACIAL: RECORTE ESTRICTO A LA CUENCA ---
                                if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
                                    zona_crs = gdf_zona_seleccionada.to_crs(crs_actual)
                                    gdf_streams = gpd.clip(gdf_streams_raw, zona_crs)
                                else:
                                    gdf_streams = gdf_streams_raw
                                
                                if not gdf_streams.empty:
                                    gdf_streams_m = gdf_streams.to_crs(epsg=3116)
                                    gdf_streams['longitud_km'] = gdf_streams_m.length / 1000.0
                                    
                                    # --- 🌟 MAGIA MATEMÁTICA: LEY DE ÁREAS DE HORTON ---
                                    # En lugar de topología frágil, usamos física de fluidos logarítmica.
                                    try:
                                        inv_affine = ~grid.affine
                                        orden_list = []
                                        import math
                                        
                                        Ra = 4.5 # Relación de Área de Horton estándar
                                        
                                        for geom in gdf_streams.geometry:
                                            acc_vals = []
                                            lineas = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
                                            
                                            for linea in lineas:
                                                if linea.is_empty: continue
                                                for p in list(linea.coords):
                                                    try:
                                                        c_f, r_f = inv_affine * p
                                                        c, r = int(round(c_f)), int(round(r_f))
                                                        # Leemos la acumulación en el río
                                                        rmin, rmax = max(0, r-1), min(acc.shape[0], r+2)
                                                        cmin, cmax = max(0, c-1), min(acc.shape[1], c+2)
                                                        window = acc[rmin:rmax, cmin:cmax]
                                                        if window.size > 0:
                                                            acc_vals.append(np.max(window))
                                                    except: pass
                                                    
                                            if acc_vals:
                                                acc_max = max(acc_vals)
                                                if acc_max >= umbral:
                                                    # Fórmula de Horton
                                                    orden = int(math.floor(math.log(acc_max / umbral, Ra))) + 1
                                                else:
                                                    orden = 1
                                            else:
                                                orden = 1
                                                
                                            orden_list.append(max(1, orden))
                                            
                                        gdf_streams['Orden_Strahler'] = orden_list
                                        
                                    except Exception as e:
                                        gdf_streams['Orden_Strahler'] = 1 
                                        
                                    st.session_state['gdf_rios'] = gdf_streams
                                    df_strahler = gdf_streams.groupby('Orden_Strahler').agg(
                                        Num_Segmentos=('geometry', 'count'),
                                        Longitud_Km=('longitud_km', 'sum')
                                    ).reset_index()
                                    st.session_state['geomorfo_strahler_df'] = df_strahler
                                else:
                                    st.session_state['gdf_rios'] = None
                                    st.session_state['geomorfo_strahler_df'] = None
                            else:
                                st.session_state['gdf_rios'] = None
                                st.session_state['geomorfo_strahler_df'] = None
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
                        
                        # CASO A: RASTER (RECORTE EXACTO AL POLÍGONO)
                        if modo_viz == "Raster (Acumulación)":
                            # Máscara para ocultar el agua fuera de la cuenca
                            acc_viz = acc.copy()
                            try:
                                if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
                                    gdf_p = gdf_zona_seleccionada.to_crs(meta['crs'])
                                    shapes = ((g, 1) for g in gdf_p.geometry)
                                    mask_poly = features.rasterize(shapes, out_shape=acc.shape, transform=transform, fill=0, dtype='uint8')
                                    # Ponemos np.nan (nulo) a lo que está fuera para que sea transparente
                                    acc_viz = np.where(mask_poly == 1, acc_viz, np.nan)
                            except Exception as e: pass

                            # Cálculo dinámico de resolución
                            h, w = acc_viz.shape
                            factor = 1
                            if h > 1000 or w > 1000: factor = int(max(h, w) / 800)
                            
                            fig = px.imshow(
                                np.log1p(acc_viz[::factor, ::factor]), 
                                color_continuous_scale='Blues', 
                                title=f"Acumulación de Flujo: {nombre_zona}",
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
                                    
                                    # Paleta de colores hidro: de claro a oscuro según jerarquía
                                    colores_orden = {1: '#85C1E9', 2: '#3498DB', 3: '#2874A6', 4: '#1A5276', 5: '#0E2F44'}
                                    
                                    # Dibujamos iterando orden por orden
                                    for orden in sorted(r_viz_4326['Orden_Strahler'].unique()):
                                        l, lt, tx = [], [], []
                                        tramos = r_viz_4326[r_viz_4326['Orden_Strahler'] == orden]
                                        color = colores_orden.get(orden, '#0E2F44')
                                        grosor = 1 + (orden * 0.8) # Más grueso a medida que sube el orden
                                        
                                        for _, row in tramos.iterrows():
                                            g = row.geometry
                                            n = f"Río Orden {orden}"
                                            
                                            if g.geom_type == 'LineString': p = [g]
                                            elif g.geom_type == 'MultiLineString': p = g.geoms
                                            else: continue
                                            
                                            for s in p: 
                                                x, y = s.xy
                                                l.extend(list(x) + [None])
                                                lt.extend(list(y) + [None])
                                                tx.extend([n] * (len(x) + 1))
                                        
                                        fig.add_trace(go.Scattermapbox(mode="lines", lon=l, lat=lt, text=tx, hoverinfo='text', line={'width': grosor, 'color': color}, name=f"Orden {orden}"))
                                    
                                    max_o = int(r_viz_4326['Orden_Strahler'].max())
                                    st.success(f"✅ Red generada exitosamente (Ley de Horton). Se detectaron hasta ríos de Orden {max_o}.")
                                else:
                                    st.warning("No hay ríos visibles. Disminuya el umbral de acumulación.")
                                    
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
                            
                            # 🌟 Guardamos el mapa en la memoria para poder descargarlo luego
                            st.session_state['fig_mapa_hidro'] = fig 
                            
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
                    
                    # --- PANEL DE STRAHLER Y RIPARIOS (INFRAESTRUCTURA VERDE) ---
                    if st.session_state.get('geomorfo_strahler_df') is not None:
                        st.markdown("##### 🌊 Red de Drenaje y Potencial Ripario (Strahler)")
                        df_str = st.session_state['geomorfo_strahler_df']
                        
                        c_str1, c_str2 = st.columns([1, 1.5])
                        with c_str1:
                            st.dataframe(df_str.style.format({'Longitud_Km': '{:.2f}'}), use_container_width=True, hide_index=True)
                            
                            # Cálculo de la Relación de Bifurcación (Rb) - Indicador clave de estabilidad
                            rb_list = []
                            for i in range(len(df_str)-1):
                                n_u = df_str['Num_Segmentos'].iloc[i]
                                n_u_next = df_str['Num_Segmentos'].iloc[i+1]
                                if n_u_next > 0: rb_list.append(n_u / n_u_next)
                            rb_mean = sum(rb_list)/len(rb_list) if rb_list else 0
                            
                            st.metric("Relación de Bifurcación ($R_b$)", f"{rb_mean:.2f}", 
                                      help="Si Rb está entre 3 y 5, la cuenca es geológicamente estable. Valores altos indican riesgo de crecientes súbitas.")
                            
                        with c_str2:
                            import plotly.express as px
                            fig_str = px.bar(df_str, x='Orden_Strahler', y='Longitud_Km', 
                                             title="Longitud de Ríos por Orden de Strahler",
                                             labels={'Orden_Strahler': 'Orden', 'Longitud_Km': 'Longitud (Km)'},
                                             color='Orden_Strahler', color_continuous_scale='Blues')
                            fig_str.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), 
                                                  xaxis=dict(tickmode='linear', dtick=1))
                            st.plotly_chart(fig_str, use_container_width=True)
                            
                        # --- GLOSARIO Y METODOLOGÍA (NUEVO) ---
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("📚 Fundamentos Teóricos: Geometría Fractal y Leyes de Horton-Strahler", expanded=False):
                            st.markdown("""
                            ### 🌊 Jerarquización de la Red de Drenaje
                            La cuantificación de la red hídrica transforma la geografía cualitativa en modelos matemáticos predictivos. Nuestro motor utiliza la física de fluidos y la topología para clasificar el territorio:
                            
                            * **Orden de Strahler (1957):** Sistema de clasificación topológica donde los nacimientos (sin afluentes) son de Orden 1. Cuando dos cauces del mismo orden se unen, forman uno de orden superior ($n+1$). Es el estándar mundial para dimensionar infraestructura verde y franjas riparias.
                            * **Leyes de Horton (1945):** Descubiertas por Robert E. Horton, demuestran que las cuencas crecen con patrones fractales (crecimiento geométrico). En nuestro modelo, para evitar la "fragmentación" de líneas vectoriales al recortar los polígonos, usamos la **Ley de las Áreas**, deduciendo el orden jerárquico directamente del volumen de agua acumulada, garantizando precisión matemática continua.
                            
                            ### 📐 Relación de Bifurcación ($R_b$)
                            Es el cociente entre el número de cauces de un orden dado y el número de cauces del orden inmediatamente superior ($R_b = N_u / N_{u+1}$). 
                            * **Interpretación (< 3.0):** Cuencas muy planas o altamente alteradas.
                            * **Interpretación (3.0 - 5.0):** Rango natural de estabilidad geológica. Cuencas sobre sustratos homogéneos donde la red de drenaje se desarrolló sin un control tectónico fuerte.
                            * **Interpretación (> 5.0):** Cuencas escarpadas, alargadas o con fuerte control estructural (fallas). Indican un **alto riesgo torrencial**, ya que el agua viaja rápidamente por cauces estrechos generando picos de crecida severos.
                            
                            ### 🌿 Utilidad en Soluciones Basadas en la Naturaleza (SbN)
                            Tener los kilómetros exactos por orden permite costear presupuestos de conservación. Un cauce de Orden 1 (nacimiento) puede requerir 30 metros de aislamiento ripario, mientras que un Orden 4 requiere más de 50 metros. La suma de estas áreas define el presupuesto exacto de restauración territorial.
                            
                            **Referencias Clave:**
                            1. Horton, R. E. (1945). *Erosional development of streams and their drainage basins; hydrophysical approach to quantitative morphology*. Geological society of America bulletin, 56(3), 275-370.
                            2. Strahler, A. N. (1957). *Quantitative analysis of watershed geomorphology*. Eos, Transactions American Geophysical Union, 38(6), 913-920.
                            3. Schumm, S. A. (1956). *Evolution of drainage systems and slopes in badlands at Perth Amboy, New Jersey*. Geological society of America bulletin, 67(5), 597-646.
                            """)
                    
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
                        st.markdown("**2. Caudal Pico Extremo (Q) - Método Racional Integrado**")
                        
                        # --- CÁLCULO AUTOMÁTICO DE C (COBERTURAS) ---
                        c_sugerido = 0.50 # Default
                        detalle_cob = "No hay datos de cobertura."
                        
                        PATH_COB = "data/Cob25m_WGS84.tif"
                        if land_cover and os.path.exists(PATH_COB):
                            try:
                                stats_cob = land_cover.calcular_estadisticas_zona(gdf_zona_seleccionada, PATH_COB)
                                if stats_cob:
                                    c_pond = 0
                                    for cob, pct in stats_cob.items():
                                        peso = pct / 100.0
                                        val_c = 0.5
                                        if "Urbano" in cob or "Industrial" in cob: val_c = 0.85
                                        elif "Cultivo" in cob: val_c = 0.60
                                        elif "Pasto" in  cob or "Herbácea" in cob: val_c = 0.45
                                        elif "Bosque" in cob: val_c = 0.30
                                        elif "Agua" in cob: val_c = 1.0
                                        c_pond += val_c * peso
                                    c_sugerido = c_pond
                                    top_3 = sorted(stats_cob.items(), key=lambda x: x[1], reverse=True)[:3]
                                    detalle_cob = ", ".join([f"{k} ({v:.0f}%)" for k,v in top_3])
                            except: pass

                        c_runoff = st.slider(
                            "Coeficiente de Escorrentía (C):", 
                            0.1, 1.0, float(c_sugerido), 0.05,
                            help=f"Valor sugerido basado en coberturas satelitales: {c_sugerido:.2f}\nPredomina: {detalle_cob}"
                        )
                        if c_sugerido != 0.5:
                            st.caption(f"🛰️ **C Calculado:** {c_sugerido:.2f} ({detalle_cob})")

                        # 🌍 BISTURÍ: INYECCIÓN ESTADÍSTICA (GUMBEL)
                        st.markdown("---")
                        st.markdown("##### ⛈️ Tormenta de Diseño (Nexo Estadístico)")
                        
                        ppt_100a_memoria = st.session_state.get('aleph_ppt_100a', 120.0) # Fallback educado
                        
                        if 'aleph_ppt_100a' in st.session_state:
                            st.success(f"🧠 **Gumbel Sincronizado:** Lluvia Tr=100 años es de **{ppt_100a_memoria:.1f} mm**.")
                        
                        p_diseno = st.number_input("Precipitación Extrema 24h [mm]:", min_value=10.0, value=float(ppt_100a_memoria), step=5.0, help="Representa la lluvia de un evento extremo (ej. Periodo de Retorno de 100 años).")
                        
                        # Fusión Física: Intensidad = Precipitación / Tiempo de Concentración
                        tc_horas = tc_kirpich_min / 60 if tc_kirpich_min > 0 else 1.0
                        i_rain_calc = p_diseno / tc_horas
                        
                        st.info(f"⚡ **Intensidad Desagregada (I):** {i_rain_calc:.1f} mm/h *(Asumiendo el peor escenario: toda la lluvia cae concentrada en el Tc de {tc_horas:.2f}h)*.")
                        
                        # Cálculo Final Racional
                        q_peak = 0.278 * c_runoff * i_rain_calc * area_km2
                        
                        # 🌐 INYECCIÓN AL ALEPH: Reemplazamos el caudal máximo global con este, que es hiper-preciso
                        st.session_state['aleph_q_max_m3s'] = float(q_peak)
                        
                        st.metric("Caudal Pico Definitivo (Q)", f"{q_peak:,.2f} m³/s", "Inyectado a la Turbina Central (Amenazas)")
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
                    
                    # 1. TORRENCIAL (NEXO FÍSICO CON ALEPH)
                    with t1:
                        c_t1, c_t2 = st.columns([1, 3])
                        with c_t1:
                            st.markdown("#### Parámetros Físicos y Energía")
                            
                            # 🌍 BISTURÍ: Inyección de Oferta Extrema (Desde Pág 01)
                            q_max_aleph = st.session_state.get('aleph_q_max_m3s', 0.0)
                            
                            s_range = st.slider("Rango de Pendiente Crítica (°)", 0.0, 60.0, (15.0, 45.0), key="s_torrencial", help="Las avenidas torrenciales (flujos de detritos) ocurren típicamente en laderas con más de 15° de inclinación.")
                            
                            if q_max_aleph > 0:
                                st.success(f"🧠 **Caudal Pico Sincronizado:** {q_max_aleph:,.1f} m³/s")
                                # Ecuación empírica: A mayor caudal, menor acumulación necesaria para detonar un flujo
                                a_sugerido = max(4.0, 8.5 - np.log10(q_max_aleph + 1))
                            else:
                                st.warning("⚠️ Usando energía teórica. Ejecuta el Aleph en la Pág 01.")
                                a_sugerido = 6.0
                                
                            a_umb = st.slider(
                                "Umbral de Energía (Acumulación Log):", 
                                min_value=4.0, max_value=9.0, 
                                value=float(a_sugerido), step=0.1, key="a_torrencial",
                                help="El modelo ajusta esto según el caudal pico. Valores más bajos indican que ríos más pequeños tendrán suficiente energía para arrastrar rocas."
                            )
                            st.warning("Detectando zonas de alta energía cinética y arrastre.")
                            
                            if st.button("🚀 Guardar Zona Torrencial (Para Biodiversidad)"):
                                st.session_state['aleph_twi_umbral'] = a_umb
                                st.session_state['aleph_pendiente_max'] = s_range[1]
                                st.session_state['aleph_pendiente_min'] = s_range[0]
                                st.toast("✅ Zona torrencial guardada para diseño de bosques de protección.", icon="🪨")

                        with c_t2:
                            mask_t = (s_core >= s_range[0]) & (s_core <= s_range[1]) & (a_core_log >= a_umb)
                            caja_analisis_ai(mask_t, "Avenida Torrencial (Flujo de Detritos)")
                            mapa_amenaza_contexto(mask_t, "#e74c3c", f"Zonas de Alta Energía / Susceptibilidad Torrencial", gdf_zona_seleccionada)

                    # 2. INUNDACIÓN (NEXO FÍSICO CON ALEPH)
                    with t2:
                        c_in1, c_in2 = st.columns([1, 3])
                        with c_in1:
                            st.markdown("#### Motor Hidráulico (Manning-TWI)")
                            
                            # 🌍 BISTURÍ: Inyección de Oferta Extrema (Desde Pág 01)
                            q_max_aleph = st.session_state.get('aleph_q_max_m3s', 0.0)
                            
                            if q_max_aleph > 0:
                                st.success(f"🧠 **Caudal Pico Sincronizado:** {q_max_aleph:,.1f} m³/s")
                                q_diseno = q_max_aleph
                            else:
                                st.warning("⚠️ Usando caudal teórico. Ejecuta el Aleph en la Pág 01 para usar hidrología real.")
                                q_diseno = 50.0 # Fallback
                                
                            # Ecuación empírica para derivar el umbral TWI desde el caudal
                            # A mayor caudal, el umbral de TWI baja (más celdas se inundan)
                            twi_sugerido = max(5.0, 16.0 - np.log1p(q_diseno))
                            
                            twi_val = st.slider(
                                "Umbral de Desbordamiento (TWI):", 
                                min_value=5.0, max_value=25.0, 
                                value=float(twi_sugerido), step=0.5, key="twi_slider",
                                help="El modelo calculó este umbral basado en el Caudal Máximo. Si lo bajas, simulas un nivel de agua más alto."
                            )
                            
                            # Cálculo TWI robusto
                            with st.spinner("Inundando llanura aluvial..."):
                                s_rad = np.deg2rad(s_core)
                                tan_s = np.tan(s_rad)
                                tan_s = np.where(tan_s < 0.001, 0.001, tan_s)
                                resolucion = 30.0 # Aprox Sentinel/SRTM
                                area_esp = acc_raw * resolucion
                                twi = np.log((area_esp + 1) / tan_s)
                                
                            strict = st.checkbox("Restringir a valles (< 5°)", value=True)
                            
                            # Inyectar el buffer de inundación resultante al Aleph (Para Biodiversidad)
                            if st.button("🚀 Guardar Zona de Riesgo (Para Corredores Riparios)"):
                                st.session_state['aleph_twi_umbral'] = twi_val
                                st.session_state['aleph_pendiente_max'] = 5 if strict else 90
                                st.toast("✅ Zona de inundación guardada en Memoria Global.", icon="🌊")

                        with c_in2:
                            if strict: mask_i = (twi >= twi_val) & (s_core <= 5)
                            else: mask_i = (twi >= twi_val)
                            
                            caja_analisis_ai(mask_i, "Inundación por Desbordamiento")
                            mapa_amenaza_contexto(mask_i, "#0099FF", f"Llanura de Inundación (Q = {q_diseno:,.1f} m³/s)", gdf_zona_seleccionada)
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

                # --- DESCARGA AVANZADA DE MAPAS Y VECTORES SIG ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🌐 Mapas Interactivos y Vectores SIG")
                st.caption("Exporta la red de drenaje calculada con sus atributos (Orden de Strahler, Longitud) para software SIG (QGIS/ArcGIS) o como un mapa web interactivo.")
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                # 1. HTML Interactivo
                with col_d1:
                    if 'fig_mapa_hidro' in st.session_state and st.session_state['fig_mapa_hidro'] is not None:
                        html_mapa = st.session_state['fig_mapa_hidro'].to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="📥 Mapa Interactivo (HTML)",
                            data=html_mapa,
                            file_name=f"Mapa_Hidrologia_{nombre_zona}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    else:
                        st.info("⚠️ Genera el mapa en Tab Hidrología primero.")
                
                # 2. GeoJSON
                with col_d2:
                    if st.session_state.get('gdf_rios') is not None:
                        rios_json = st.session_state['gdf_rios'].to_crs("EPSG:4326").to_json()
                        st.download_button(
                            label="📥 Red de Drenaje (GeoJSON)",
                            data=rios_json,
                            file_name=f"Red_Drenaje_Strahler_{nombre_zona}.geojson",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.info("⚠️ Calcula los ríos en Tab Hidrología primero.")
                        
                # 3. Shapefile (ZIP)
                with col_d3:
                    if st.session_state.get('gdf_rios') is not None:
                        import tempfile
                        import zipfile
                        from io import BytesIO
                        
                        # Función para empaquetar el Shapefile al vuelo
                        def create_shp_zip():
                            temp_dir = tempfile.mkdtemp()
                            base_name = f"Drenaje_{nombre_zona}".replace(" ", "_")
                            shp_path = os.path.join(temp_dir, f"{base_name}.shp")
                            
                            # Preparar GDF: Los Shapefiles solo soportan nombres de columnas de hasta 10 caracteres
                            gdf_export = st.session_state['gdf_rios'].to_crs("EPSG:4326").copy()
                            if 'Orden_Strahler' in gdf_export.columns:
                                gdf_export = gdf_export.rename(columns={'Orden_Strahler': 'Orden_Stra'})
                            if 'longitud_km' in gdf_export.columns:
                                gdf_export = gdf_export.rename(columns={'longitud_km': 'Long_km'})
                                
                            gdf_export.to_file(shp_path, driver='ESRI Shapefile')
                            
                            # Comprimir la carpeta en un ZIP
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        zip_file.write(os.path.join(root, file), arcname=file)
                            return zip_buffer.getvalue()
                            
                        st.download_button(
                            label="📥 Red de Drenaje (Shapefile .zip)",
                            data=create_shp_zip(),
                            file_name=f"Shapefile_Drenaje_{nombre_zona}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.info("⚠️ Calcula los ríos en Tab Hidrología primero.")

else:
    st.info("👈 Selecciona una zona.")
