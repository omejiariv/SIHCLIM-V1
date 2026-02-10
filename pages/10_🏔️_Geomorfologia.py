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
import os
from shapely.geometry import shape, LineString, MultiLineString, Polygon
from modules import selectors

# Intentamos importar pysheds
try:
    from pysheds.grid import Grid
    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False

# Configuración de Página
st.set_page_config(page_title="Geomorfología Pro", page_icon="🏔️", layout="wide")

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

            tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs([
                "🗺️ 3D", "📐 Pendientes", "📈 Hipsometría", 
                "🌊 Hidrología", "📊 Índices (Nuevo)", "📥 Descargas"
            ])
            
            # Factor de reducción visual
            h, w = arr_elevacion.shape
            factor = max(1, int(max(h, w) / 200)) # Mejor resolución para el 3D

            # --- TAB 1: 3D (EXAGERACIÓN VISUAL, NO DE DATOS - SELECTOR DE COLOR) ---
            with tab1:
                col_ctrl, col_map = st.columns([1, 4])
                with col_ctrl:
                    st.markdown("##### 🎛️ Control 3D")
                    exageracion = st.slider("Exageración Vertical:", 0.5, 5.0, 1.5, 0.1, key=f"slider_3d_{nombre_zona}")
                    
                    # 🎨 Nuevo Selector de Color
                    paletas = {
                        "Tierra (Natural)": "Earth",
                        "Terreno (Verde/Café)": "bluyl", # Aprox a terreno
                        "Viridis (Científico)": "Viridis",
                        "Turbo (Alto Contraste)": "Turbo",
                        "Agua (Azul/Verde)": "YlGnBu",
                        "Magma (Oscuro)": "Magma"
                    }
                    paleta_nombre = st.selectbox("Paleta de Color:", list(paletas.keys()), index=0)
                    paleta_codigo = paletas[paleta_nombre]
                
                with col_map:
                    arr_3d = arr_elevacion[::factor, ::factor]
                    
                    # Usamos la paleta seleccionada
                    fig_surf = go.Figure(data=[go.Surface(z=arr_3d, colorscale=paleta_codigo)])
                    
                    z_aspect = 0.2 * exageracion
                    fig_surf.update_layout(
                        title=f"Topografía 3D ({paleta_nombre})",
                        autosize=True, height=650,
                        scene=dict(
                            xaxis_title='X', yaxis_title='Y', zaxis_title='Altitud (m)',
                            aspectmode='manual', aspectratio=dict(x=1, y=1, z=z_aspect)
                        ),
                        margin=dict(l=10, r=10, b=10, t=40)
                    )
                    st.plotly_chart(fig_surf, use_container_width=True)

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

            # --- TAB 3: HIPSOMETRÍA (RECUPERADA) ---
            with tab3:
                elevs_sorted = np.sort(elevs_valid)[::-1]
                n_px = len(elevs_sorted)
                area_pct = np.arange(1, n_px + 1) / n_px * 100
                
                # Optimización Gráfica
                idx = np.linspace(0, n_px - 1, 300, dtype=int) if n_px > 300 else np.arange(n_px)
                
                # Cálculo de Ecuación
                eq_str = "N/A"
                try:
                    coeffs = np.polyfit(area_pct[idx], elevs_sorted[idx], 3)
                    eq_str = f"H = {coeffs[0]:.2e}A³ + {coeffs[1]:.2e}A² + {coeffs[2]:.2e}A + {coeffs[3]:.0f}"
                except: pass

                st.markdown(f"**📐 Ecuación:** `$ {eq_str} $`")
                st.success(f"**Diagnóstico:** (HI: {hi_global:.3f}) - {('Joven/Erosiva' if hi_global>0.5 else 'Vieja/Sedimentaria')}")

                fig_hypso = go.Figure()
                fig_hypso.add_trace(go.Scatter(x=area_pct[idx], y=elevs_sorted[idx], fill='tozeroy'))
                fig_hypso.update_layout(height=500, title="Curva Hipsométrica", xaxis_title="% Área", yaxis_title="Altitud")
                st.plotly_chart(fig_hypso, use_container_width=True)

            # --- TAB 4: HIDROLOGÍA (ACTUALIZADO: COORDENADAS EXTREMAS) ---
            gdf_rios_export = None
            catchment_raster_export = None

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
                            "Catchment (Mascara)",
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
                        
                        # 1. PREPARACIÓN HIDROLÓGICA
                        grid = None; acc = None; fdir = None
                        
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            meta_temp = meta.copy(); meta_temp.update(driver='GTiff', dtype='float64') 
                            with rasterio.open(tmp.name, 'w', **meta_temp) as dst:
                                dst.write(arr_elevacion.astype('float64'), 1)
                            try:
                                grid = Grid.from_raster(tmp.name)
                                dem_grid = grid.read_raster(tmp.name) # Leer normal para conservar metadatos
                                pit_filled = grid.fill_pits(dem_grid)
                                resolved = grid.resolve_flats(pit_filled)
                                dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                                fdir = grid.flowdir(resolved, dirmap=dirmap)
                                acc = grid.accumulation(fdir, dirmap=dirmap)
                            except Exception as e: st.error(f"Error: {e}")
                            finally: try: os.remove(tmp.name); except: pass

                        if grid is not None and acc is not None:
                            crs_actual = meta.get('crs', 'EPSG:3116')

                            # --- INFO DE REFERENCIA (NUEVO) ---
                            # Encontrar coordenadas de Altura Mínima y Máxima
                            idx_min_h = np.argmin(dem_grid); y_min, x_min = np.unravel_index(idx_min_h, dem_grid.shape)
                            idx_max_h = np.argmax(dem_grid); y_max, x_max = np.unravel_index(idx_max_h, dem_grid.shape)
                            
                            with st.expander("📍 Coordenadas de Referencia (Matriz)", expanded=False):
                                c_ref1, c_ref2 = st.columns(2)
                                c_ref1.info(f"**Punto Más Bajo (Posible Salida):**\n\nFila (Y): {y_min} | Col (X): {x_min}\nAltitud: {np.min(dem_grid):.1f} m")
                                c_ref2.success(f"**Punto Más Alto:**\n\nFila (Y): {y_max} | Col (X): {x_max}\nAltitud: {np.max(dem_grid):.1f} m")

                            # --- MODO 1: RASTER ---
                            if modo_viz == "Raster (Acumulación)":
                                log_acc = np.log1p(acc)
                                fig = px.imshow(log_acc, color_continuous_scale='Blues', title="Acumulación de Flujo (Log)")
                                fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
                                st.plotly_chart(fig, use_container_width=True)

                            # --- MODO 2: CATCHMENT / DIVISORIA ---
                            elif modo_viz in ["Catchment (Mascara)", "Divisoria (Línea)"]:
                                idx_max_acc = np.argmax(acc)
                                y_auto, x_auto = np.unravel_index(idx_max_acc, acc.shape)
                                
                                # Calibración Manual
                                with st.expander("🔧 Calibrar Punto de Desfogue", expanded=True):
                                    st.caption("Ajusta las coordenadas para coincidir con la salida real del río.")
                                    c_x, c_y = st.columns(2)
                                    x_pour = c_x.number_input("Columna (X):", value=int(x_auto), min_value=0, max_value=acc.shape[1]-1, step=1)
                                    y_pour = c_y.number_input("Fila (Y):", value=int(y_auto), min_value=0, max_value=acc.shape[0]-1, step=1)

                                # Calcular Catchment
                                catch = None
                                try:
                                    catch = grid.catchment(x=x_pour, y=y_pour, fdir=fdir, dirmap=dirmap, xytype='index')
                                    catchment_raster_export = catch
                                except: pass

                                # Visualización
                                if catch is not None:
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
                                                locations=gdf_calc_4326.index, mapbox_style="carto-positron",
                                                center={"lat": gdf_calc_4326.centroid.y.mean(), "lon": gdf_calc_4326.centroid.x.mean()},
                                                zoom=10, opacity=0.5, color_discrete_sequence=["#0099FF"]
                                            )
                                            # Validación (Línea Verde)
                                            if not gdf_off_4326.empty:
                                                poly = gdf_off_4326.geometry.iloc[0]
                                                if poly.geom_type == 'Polygon': x, y = poly.exterior.coords.xy
                                                else: x, y = max(poly.geoms, key=lambda a: a.area).exterior.coords.xy
                                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(x), lat=list(y), line={'width':2, 'color':'#00FF00'}, name="Oficial"))
                                            
                                            fig.update_layout(title="Catchment", height=600, margin=dict(l=0,r=0,t=30,b=0))
                                            st.plotly_chart(fig, use_container_width=True)

                                        elif modo_viz == "Divisoria (Línea)":
                                            fig = go.Figure()
                                            # Roja (Calculada)
                                            p_c = gdf_calc_4326.geometry.iloc[0]
                                            if p_c.geom_type == 'Polygon': xc, yc = p_c.exterior.coords.xy
                                            else: xc, yc = max(p_c.geoms, key=lambda a: a.area).exterior.coords.xy
                                            fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xc), lat=list(yc), line={'width':3, 'color':'red'}, name="Calculada"))
                                            
                                            # Verde (Oficial)
                                            if not gdf_off_4326.empty:
                                                p_o = gdf_off_4326.geometry.iloc[0]
                                                if p_o.geom_type == 'Polygon': xo, yo = p_o.exterior.coords.xy
                                                else: xo, yo = max(p_o.geoms, key=lambda a: a.area).exterior.coords.xy
                                                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(xo), lat=list(yo), line={'width':2, 'color':'#00FF00'}, name="Oficial"))
                                            
                                            clat = gdf_calc_4326.centroid.y.mean()
                                            clon = gdf_calc_4326.centroid.x.mean()
                                            fig.update_layout(title="Comparativa", mapbox=dict(style="carto-positron", zoom=10, center={"lat": clat, "lon": clon}), height=600, margin=dict(l=0,r=0,t=30,b=0))
                                            st.plotly_chart(fig, use_container_width=True)

                            # --- MODO 3: VECTORES ---
                            elif modo_viz == "Vectores (Líneas)":
                                gdf_rios = extraer_vectores_rios(grid, fdir, acc, umbral, meta['crs'], nombre_zona)
                                if gdf_rios is not None:
                                    gdf_4326 = gdf_rios.to_crs("EPSG:4326")
                                    lons, lats = [], []
                                    for geom in gdf_4326.geometry:
                                        if geom.geom_type == 'LineString': x, y = geom.xy; lons.extend(list(x)+[None]); lats.extend(list(y)+[None])
                                        elif geom.geom_type == 'MultiLineString': 
                                            for g in geom.geoms: x, y = g.xy; lons.extend(list(x)+[None]); lats.extend(list(y)+[None])
                                    
                                    fig = go.Figure(go.Scattermapbox(mode="lines", lon=lons, lat=lats, line={'width': 2, 'color': '#0077BE'}))
                                    center = gdf_4326.geometry.centroid.iloc[0]
                                    fig.update_layout(mapbox=dict(style="carto-positron", zoom=10, center={"lat": center.y, "lon": center.x}), height=600, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                else: st.warning("Baja el umbral.")

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
                        df_morfo = pd.DataFrame({
                            "Parámetro": ["Área", "Perímetro", "Longitud Axial", "Longitud Total Ríos", "Desnivel (H)", "Pendiente Media Cuenca", "Pendiente Aprox. Cauce"],
                            "Valor": [area_km2, perimetro_km, longitud_axial_km, longitud_rios_km, desnivel_m, slope_mean, pendiente_cauce_m_m * 100],
                            "Unidad": ["km²", "km", "km", "km", "m", "Grados", "%"]
                        })
                        st.dataframe(df_morfo.style.format({"Valor": "{:.3f}"}), use_container_width=True)

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
                        # Q = 0.278 * C * I * A  (Q m3/s, I mm/h, A km2)
                        
                        i_rain = st.slider("Intensidad de Lluvia (I) [mm/h]:", 10, 200, 50, 10)
                        c_runoff = st.select_slider("Coeficiente de Escorrentía (C):", 
                                                    options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                                                    value=0.5,
                                                    help="0.2: Bosque denso, 0.9: Urbano pavimentado")
                        
                        q_peak = 0.278 * c_runoff * i_rain * area_km2
                        
                        st.metric("Caudal Pico Estimado (Q)", f"{q_peak:.2f} m³/s")
                        st.caption("Fórmula: $Q = 0.278 \cdot C \cdot I \cdot A$")

                except Exception as e:
                    st.error(f"Error en cálculos: {e}")
                                
            # --- TAB 5: DESCARGAS ---
            with tab5:
                st.subheader("Centro de Descargas")
                c1, c2, c3, c4 = st.columns(4)
                
                # 1. DEM (TIF)
                c1.download_button("📥 DEM (.tif)", to_tif(arr_elevacion, meta), f"DEM_{nombre_zona}.tif")
                
                # 2. Pendientes (TIF)
                slope_meta = meta.copy(); slope_meta.update(dtype=rasterio.float32)
                c2.download_button("📥 Pendientes (.tif)", to_tif(slope_deg, slope_meta), f"Slope_{nombre_zona}.tif")
                
                # 3. Datos Hipsométricos (CSV) - Recalculamos rápido para descargar
                try:
                    elevs_sorted = np.sort(elevs_valid)[::-1]
                    n_px = len(elevs_sorted)
                    area_pct = np.arange(1, n_px + 1) / n_px * 100
                    # Submuestreo para CSV manejable (max 5000 filas)
                    step = max(1, n_px // 5000)
                    df_hypso_export = pd.DataFrame({
                        "Porcentaje_Area": area_pct[::step],
                        "Altitud_m": elevs_sorted[::step]
                    })
                    csv_hypso = df_hypso_export.to_csv(index=False).encode('utf-8')
                    c3.download_button("📊 Curva Hipsométrica (.csv)", csv_hypso, f"Hipsometria_{nombre_zona}.csv", "text/csv")
                except:
                    c3.warning("Error generando CSV")

                # 4. RÍOS (GEOJSON)
                if gdf_rios_export is not None:
                    try:
                        # 🔥 CORRECCIÓN: Usamos meta['crs'] aquí también
                        crs_actual = meta.get('crs', 'EPSG:3116')
                        gdf_export_4326 = gdf_rios_export.set_crs(crs_actual, allow_override=True).to_crs("EPSG:4326")
                        json_str = gdf_export_4326.to_json()
                        c4.download_button("🌊 Red Drenaje (.geojson)", json_str, f"Rios_{nombre_zona}.geojson", "application/json")
                    except Exception as e:
                        c4.error(f"Error proyec: {e}")

                # CSV Índices
                try:
                    csv_ind = df_indices.to_csv(index=False).encode('utf-8')
                    c3.download_button("📊 Índices (.csv)", csv_ind, f"Indices_{nombre_zona}.csv", "text/csv")
                except: pass

                # Catchment Raster (NUEVO)
                if catchment_raster_export is not None:
                    # Convertimos el array de 0/1 a TIF
                    catch_meta = meta.copy(); catch_meta.update(dtype=rasterio.uint8, nodata=0)
                    c4.download_button(
                        "🟦 Catchment (.tif)", 
                        to_tif(catchment_raster_export.astype(np.uint8), catch_meta), 
                        f"Catchment_{nombre_zona}.tif"
                    )
                else:
                    c4.info("Calcula la cuenca en la pestaña 'Hidrología' para descargar.")

else:
    st.info("👈 Selecciona una zona.")
