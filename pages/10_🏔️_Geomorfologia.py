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

# --- FUNCIÓN DE VECTORIZACIÓN DE RÍOS (NUEVA) 🌊 ---
@st.cache_data(show_spinner="Vectorizando red de drenaje...")
def extraer_vectores_rios(acc_array, transform, umbral):
    """Convierte la matriz de acumulación en líneas GeoJSON."""
    # Crear máscara booleana
    mask_rios = (acc_array > umbral).astype(np.int16)
    
    if np.sum(mask_rios) == 0:
        return None

    # Extraer formas del raster (Vectorización)
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            features.shapes(mask_rios, transform=transform))
        if v == 1 # Solo nos interesa donde hay río (valor 1)
    )
    
    # Convertir a GeoDataFrame
    geoms = list(results)
    if not geoms:
        return None
        
    gdf = gpd.GeoDataFrame.from_features(geoms)
    # Asignar CRS (asumimos el del DEM)
    # Nota: para descargar GeoJSON necesitamos lat/lon (EPSG:4326), 
    # pero para plotear en coords locales usamos coordenadas crudas.
    # Aquí asumiremos coordenadas planas locales del DEM para visualización.
    return gdf

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

            # --- TAB 1: 3D ---
            with tab1:
                col_ctrl, col_map = st.columns([1, 4])
                with col_ctrl:
                    st.markdown("##### 🎛️ Control 3D")
                    # Usamos una key única para forzar redraw si cambia
                    exageracion = st.slider("Exageración Vertical:", 1.0, 10.0, 1.0, 0.5, key=f"slider_3d_{nombre_zona}")
                
                with col_map:
                    arr_3d = arr_elevacion[::factor, ::factor]
                    # Aplicar la exageración directamente a los datos Z
                    z_data = arr_3d * exageracion
                    
                    fig_surf = go.Figure(data=[go.Surface(z=z_data, colorscale='Earth')])
                    fig_surf.update_layout(
                        title=f"Topografía 3D (x{exageracion})",
                        autosize=True, height=650,
                        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)), # Z controlado
                        margin=dict(l=10, r=10, b=10, t=40)
                    )
                    st.plotly_chart(fig_surf, use_container_width=True)

            # --- TAB 2: PENDIENTES ---
            with tab2:
                st.subheader("Mapa de Pendientes")
                st.info(texto_analisis, icon="🤖") # El Analista ha vuelto
                
                fig_slope = px.imshow(
                    slope_deg[::factor, ::factor], 
                    color_continuous_scale='Turbo',
                    labels={'color': 'Grados'}
                )
                fig_slope.update_layout(height=600, dragmode='zoom') # Habilitar zoom explícito
                st.plotly_chart(fig_slope, use_container_width=True)

                # Tabla Resumen
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

            # --- TAB 4: RED DE DRENAJE VECTORIAL 🌊 ---
            gdf_rios_export = None # Para el botón de descarga
            with tab4:
                st.subheader("Red de Drenaje (Vectores)")
                
                if not PYSHEDS_AVAILABLE:
                    st.error("Instala `pysheds` para ver esto.")
                else:
                    c_param, c_viz = st.columns([1, 4])
                    with c_param:
                        st.info("Configura la sensibilidad:")
                        # Slider con key única para que se actualice
                        umbral = st.slider("Umbral Acumulación", 500, 10000, 2000, 500, key=f"umb_rio_{nombre_zona}")
                        st.caption("Menor valor = Más detalles (ríos pequeños).")

                    with c_viz:
                        # 1. Procesamiento PySheds
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            with rasterio.open(tmp.name, 'w', **meta) as dst:
                                dst.write(arr_elevacion.astype(rasterio.float32), 1)
                            
                            grid = Grid.from_raster(tmp.name)
                            dem_grid = grid.read_raster(tmp.name)
                            # Rellenar y calcular flujo
                            pit_filled = grid.fill_pits(dem_grid)
                            fdir = grid.flowdir(pit_filled)
                            acc = grid.accumulation(fdir)
                            acc_arr = acc.view(np.ndarray)
                            
                            os.remove(tmp.name)

                        # 2. Vectorización (Pixels -> Líneas)
                        # Usamos 'transform' para que las líneas queden en coords reales
                        gdf_rios = extraer_vectores_rios(acc_arr, transform, umbral)
                        
                        if gdf_rios is not None:
                            # Simplificar geometría para que sea rápido
                            gdf_rios['geometry'] = gdf_rios.simplify(10) 
                            gdf_rios_export = gdf_rios.copy() # Guardar copia para descarga
                            
                            # Plotly no soporta GeoDataFrame local plano fácilmente sin Mapbox.
                            # Truco: Ploteamos las líneas como Scatter (XY)
                            fig_net = go.Figure()
                            
                            # Mapa base (DEM en gris de fondo)
                            fig_net.add_trace(go.Heatmap(
                                z=arr_elevacion[::factor, ::factor],
                                colorscale='Greys', opacity=0.4, showscale=False
                            ))

                            # Convertir líneas a coordenadas para Plotly
                            lats = []
                            lons = []
                            for line in gdf_rios.geometry:
                                if line.geom_type == 'LineString':
                                    x, y = line.xy
                                    # Convertir coordenadas planas a índices de pixel aproximados para coincidir con el heatmap
                                    # O mejor: Plotear todo en coordenadas planas (quitar heatmap de pixel)
                                    pass

                            # MEJOR OPCIÓN VISUAL: Plotly Express con GeoJSON
                            # Pero necesitamos convertir a Lat/Lon para que GeoJSON funcione bien en mapas web.
                            # Si gdf_rios está en EPSG:3116, convertir a 4326.
                            try:
                                gdf_rios_4326 = gdf_rios.set_crs("EPSG:3116", allow_override=True).to_crs("EPSG:4326")
                                
                                fig_map = px.choropleth_mapbox(
                                    geojson=gdf_rios_4326.geometry.__geo_interface__,
                                    locations=gdf_rios_4326.index,
                                    mapbox_style="carto-positron",
                                    zoom=8,
                                    center={"lat": gdf_rios_4326.geometry.centroid.y.mean(), "lon": gdf_rios_4326.geometry.centroid.x.mean()},
                                    opacity=0.5
                                )
                                # ScatterMapbox para las líneas de ríos (Más bonito)
                                lat_lines = []
                                lon_lines = []
                                for geom in gdf_rios_4326.geometry:
                                    if geom.geom_type == 'LineString':
                                        xs, ys = geom.xy
                                        lon_lines.extend(list(xs) + [None])
                                        lat_lines.extend(list(ys) + [None])
                                
                                fig_map = go.Figure(go.Scattermapbox(
                                    mode = "lines",
                                    lon = lon_lines,
                                    lat = lat_lines,
                                    marker = {'size': 10},
                                    line = {'width': 2, 'color': 'blue'}
                                ))
                                fig_map.update_layout(
                                    mapbox_style="open-street-map",
                                    mapbox_center={"lat": gdf_rios_4326.geometry.centroid.y.mean(), "lon": gdf_rios_4326.geometry.centroid.x.mean()},
                                    mapbox_zoom=10,
                                    margin={"r":0,"t":0,"l":0,"b":0},
                                    height=600
                                )
                                st.plotly_chart(fig_map, use_container_width=True)
                                
                            except Exception as e:
                                st.warning("No se pudo reproyectar para mapa base. Mostrando esquema local.")
                                # Fallback: Esquema local (Coordenadas de la imagen)
                                fig_local = px.imshow(np.log1p(acc_arr), color_continuous_scale='Blues', title="Red de Flujo (Local)")
                                st.plotly_chart(fig_local, use_container_width=True)

                        else:
                            st.warning("No se detectaron ríos con ese umbral. Intenta bajar el valor.")

            # --- TAB 5: DESCARGAS ---
            with tab5:
                st.subheader("Centro de Descargas")
                c1, c2, c3 = st.columns(3)
                
                # DEM
                c1.download_button("📥 Descargar DEM (.tif)", to_tif(arr_elevacion, meta), f"DEM_{nombre_zona}.tif")
                
                # Pendientes
                slope_meta = meta.copy(); slope_meta.update(dtype=rasterio.float32)
                c2.download_button("📥 Descargar Pendientes (.tif)", to_tif(slope_deg, slope_meta), f"Slope_{nombre_zona}.tif")
                
                # RÍOS (GEOJSON) - NUEVO 🌟
                if gdf_rios_export is not None:
                    # Intentamos convertir a LatLon para que sea útil en Google Earth/QGIS web
                    try:
                        gdf_export_4326 = gdf_rios_export.set_crs("EPSG:3116", allow_override=True).to_crs("EPSG:4326")
                        json_str = gdf_export_4326.to_json()
                        c3.download_button("📥 Descargar Ríos (.geojson)", json_str, f"Rios_{nombre_zona}.geojson", "application/json")
                    except:
                        st.warning("No se pudo reproyectar ríos para descarga.")

else:
    st.info("👈 Selecciona una zona.")
