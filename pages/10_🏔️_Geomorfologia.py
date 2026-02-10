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
from shapely.geometry import shape, LineString, MultiLineString
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
def extraer_vectores_rios(_grid, _fdir, _acc, umbral, crs_in="EPSG:3116"):
    """
    Usa PySheds para extraer líneas de flujo reales (Centerlines).
    """
    try:
        # PySheds extrae las ramas del río como líneas vectoriales
        # dirmap=(64, 128, 1, 2, 4, 8, 16, 32) es el estándar
        branches = _grid.extract_river_network(_fdir, _acc > umbral)
        
        # Validar si encontró algo
        if not branches or not branches['features']:
            return None

        # Convertir GeoJSON a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(branches['features'])
        
        # Asignar CRS (Sistema de Coordenadas)
        if gdf.crs is None:
            gdf.set_crs(crs_in, inplace=True)
            
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
            
            # Pendientes
            dy, dx = np.gradient(arr_elevacion, 30.0)
            slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            slope_mean = np.nanmean(slope_deg)
            
            # Texto Analista
            texto_analisis = analista_hidrologico(slope_mean, hi_global)

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mínima", f"{min_el:.0f} m")
            c2.metric("Máxima", f"{max_el:.0f} m")
            c3.metric("Media", f"{mean_el:.0f} m")
            c4.metric("Rango", f"{max_el - min_el:.0f} m")

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 3D", "📐 Pendientes", "📈 Hipsometría", "🌊 Drenaje (Vector)", "📥 Descargas"])
            
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

            # --- TAB 4: RED DE DRENAJE (VECTORES REALES) ---
            gdf_rios_export = None 
            with tab4:
                st.subheader("Red de Drenaje (Vectores)")
                
                if not PYSHEDS_AVAILABLE:
                    st.error("Instala `pysheds` para ver esto.")
                else:
                    c_param, c_viz = st.columns([1, 4])
                    with c_param:
                        st.info("Configuración Hidrológica")
                        # Slider ajustado a la realidad de micro-cuencas
                        # Inicia en 50 para no saturar, pero permite bajar a 2
                        umbral = st.slider("Umbral Acumulación", 2, 2000, 50, 5, key=f"umb_rio_{nombre_zona}")
                        st.caption(f"Se dibujarán líneas donde el flujo supere {umbral} celdas.")
                        st.markdown("""
                        * **Valor Bajo:** Muestra cabeceras y arroyos.
                        * **Valor Alto:** Muestra solo cauces principales.
                        """)

                    with c_viz:
                        import tempfile
                        # 1. Procesamiento PySheds
                        # Necesitamos guardar el TIF temporalmente porque PySheds lee rutas de archivo
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            meta_temp = meta.copy(); meta_temp.update(driver='GTiff')
                            with rasterio.open(tmp.name, 'w', **meta_temp) as dst:
                                dst.write(arr_elevacion.astype(rasterio.float32), 1)
                            
                            try:
                                # Instanciar Grid
                                grid = Grid.from_raster(tmp.name)
                                dem_grid = grid.read_raster(tmp.name)
                                
                                # A. Rellenar depresiones (Crucial para continuidad)
                                pit_filled = grid.fill_pits(dem_grid)
                                # B. Resolver zonas planas
                                resolved = grid.resolve_flats(pit_filled)
                                # C. Dirección de flujo
                                fdir = grid.flowdir(resolved)
                                # D. Acumulación
                                acc = grid.accumulation(fdir)
                                
                            except Exception as e:
                                st.error(f"Error cálculo flujo: {e}")
                                acc = None
                            finally:
                                try: os.remove(tmp.name)
                                except: pass

                        # 2. Vectorización (Ahora pasamos el grid y fdir)
                        if acc is not None:
                            # Obtenemos CRS correcto
                            crs_actual = meta.get('crs', 'EPSG:3116')
                            
                            # Llamada a la nueva función
                            gdf_rios = extraer_vectores_rios(grid, fdir, acc, umbral, crs_in=crs_actual)
                            
                            if gdf_rios is not None and not gdf_rios.empty:
                                gdf_rios_export = gdf_rios.copy()
                                
                                try:
                                    # Reproyección a Lat/Lon para mapa web
                                    gdf_rios_4326 = gdf_rios.to_crs("EPSG:4326")
                                    
                                    # Extraer coordenadas para Plotly (LineString)
                                    lons, lats = [], []
                                    for geom in gdf_rios_4326.geometry:
                                        if geom.geom_type == 'LineString':
                                            xs, ys = geom.xy
                                            lons.extend(list(xs) + [None]) # None corta la línea
                                            lats.extend(list(ys) + [None])
                                        elif geom.geom_type == 'MultiLineString':
                                            for g in geom.geoms:
                                                xs, ys = g.xy
                                                lons.extend(list(xs) + [None])
                                                lats.extend(list(ys) + [None])

                                    # Crear Mapa
                                    fig_map = go.Figure()
                                    
                                    # Líneas de Drenaje
                                    fig_map.add_trace(go.Scattermapbox(
                                        mode = "lines", 
                                        lon = lons, lat = lats,
                                        line = {'width': 2.5, 'color': '#0099FF'}, # Azul sólido
                                        name = "Red Hídrica",
                                        hoverinfo='skip'
                                    ))
                                    
                                    # Ajustar vista
                                    center_lat = gdf_rios_4326.geometry.centroid.y.mean()
                                    center_lon = gdf_rios_4326.geometry.centroid.x.mean()

                                    fig_map.update_layout(
                                        mapbox_style="carto-positron",
                                        mapbox_center={"lat": center_lat, "lon": center_lon},
                                        mapbox_zoom=11, 
                                        margin={"r":0,"t":0,"l":0,"b":0}, 
                                        height=650,
                                        showlegend=False
                                    )
                                    
                                    st.success(f"✅ Red trazada: {len(gdf_rios)} segmentos.")
                                    st.plotly_chart(fig_map, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"Error pintando mapa: {e}")
                            else:
                                st.warning(f"No se detectaron ríos con umbral {umbral}. Baja el valor.")
                                
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

else:
    st.info("👈 Selecciona una zona.")
