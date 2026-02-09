import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO
from modules import selectors

# Intentamos importar pysheds, si falla, manejamos el error elegantemente
try:
    from pysheds.grid import Grid
    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False

# Configuración de Página
st.set_page_config(page_title="Geomorfología Avanzada", page_icon="🏔️", layout="wide")

st.title("🏔️ Análisis Geomorfológico y Terreno 3D")
st.markdown("""
Esta herramienta utiliza el **Modelo Digital de Elevación (DEM)** para modelar el terreno, 
calcular pendientes, definir redes de drenaje y permitir la descarga de productos cartográficos.
""")

# --- 1. BARRA LATERAL (SELECTOR) ---
ids, nombre_zona, alt_ref, gdf_zona_seleccionada = selectors.render_selector_espacial()

# 🛠️ CORRECCIÓN CLAVE: Convertir Puntos (Regiones) en Polígono (Caja)
if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
    if gdf_zona_seleccionada.geom_type.isin(['Point', 'MultiPoint']).any():
        if len(gdf_zona_seleccionada) == 1:
            gdf_zona_seleccionada['geometry'] = gdf_zona_seleccionada.buffer(0.045) 
        else:
            bbox = gdf_zona_seleccionada.unary_union.envelope
            gdf_zona_seleccionada = gpd.GeoDataFrame({'geometry': [bbox]}, crs=gdf_zona_seleccionada.crs)

# --- 2. CARGA DEL DEM (RASTER) ---
DEM_PATH = os.path.join("data", "DemAntioquia_EPSG3116.tif")

@st.cache_data(show_spinner="Procesando terreno...")
def cargar_y_cortar_dem(ruta_dem, _gdf_corte, zona_id):
    """Corta el DEM grande usando la geometría seleccionada."""
    if _gdf_corte is None or _gdf_corte.empty:
        return None, None, None

    try:
        if not os.path.exists(ruta_dem):
            return None, None, None

        with rasterio.open(ruta_dem) as src:
            crs_dem = src.crs
            gdf_proyectado = _gdf_corte.to_crs(crs_dem)
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
                "count": 1 # Asegurar que sea 1 banda
            })
            
            dem_array = out_image[0]
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            dem_array = np.where(dem_array < -100, np.nan, dem_array)
            
            if np.isnan(dem_array).all():
                 return None, "EMPTY_DATA", None

            return dem_array, out_meta, out_transform

    except Exception as e:
        st.error(f"Error técnico procesando el DEM: {e}")
        return None, None, None

# --- FUNCIONES AUXILIARES DE DESCARGA ---
def convert_array_to_tif_bytes(arr, meta):
    """Convierte un array numpy a bytes de un archivo TIF para descarga."""
    try:
        with MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(arr.astype(rasterio.float32), 1)
            return memfile.read()
    except Exception as e:
        return None

# --- LÓGICA PRINCIPAL ---

if gdf_zona_seleccionada is not None:
    if not os.path.exists(DEM_PATH):
        st.error(f"⚠️ No encuentro el archivo DEM en: {DEM_PATH}")
    else:
        # Procesar DEM
        arr_elevacion, meta, transform = cargar_y_cortar_dem(DEM_PATH, gdf_zona_seleccionada, nombre_zona)
        
        # MANEJO DE CASOS DE ERROR
        if meta == "OUT_OF_BOUNDS":
            st.warning(f"⚠️ **Fuera de Cobertura:** La zona '{nombre_zona}' está fuera de los límites.")
        elif meta == "EMPTY_DATA":
             st.warning(f"⚠️ **Datos Vacíos:** El recorte no contiene datos válidos.")
             
        elif arr_elevacion is not None and not np.isnan(arr_elevacion).all():
            
            # --- CÁLCULOS GLOBALES ---
            elevs_valid = arr_elevacion[~np.isnan(arr_elevacion)].flatten()
            min_el, max_el = np.min(elevs_valid), np.max(elevs_valid)
            mean_el = np.mean(elevs_valid)
            
            # Pendientes
            pixel_size = 30.0 
            dy, dx = np.gradient(arr_elevacion, pixel_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_mean_global = np.nanmean(slope_deg)

            # --- VISUALIZACIÓN DE MÉTRICAS ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Elevación Mínima", f"{min_el:.0f} m")
            c2.metric("Elevación Máxima", f"{max_el:.0f} m")
            c3.metric("Elevación Media", f"{mean_el:.0f} m")
            c4.metric("Rango Altitudinal", f"{max_el - min_el:.0f} m")

            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🗺️ Elevación 3D", 
                "📐 Pendientes", 
                "📈 Hipsometría", 
                "🌊 Red de Drenaje",
                "📥 Descargas"
            ])
            
            h, w = arr_elevacion.shape
            factor = max(1, int(max(h, w) / 150))

            # --- TAB 1: 3D CON EXAGERACIÓN VERTICAL ---
            with tab1:
                col_ctrl, col_map = st.columns([1, 4])
                with col_ctrl:
                    st.markdown("##### 🎛️ Controles")
                    exageracion = st.slider("Exageración Vertical:", 1.0, 5.0, 1.0, 0.1, help="Multiplica la altura para resaltar el relieve.")
                
                with col_map:
                    st.subheader(f"Modelo 3D: {nombre_zona}")
                    arr_3d = arr_elevacion[::factor, ::factor]
                    
                    # Aplicar exageración solo para visualización
                    z_data = arr_3d * exageracion
                    
                    fig_surf = go.Figure(data=[go.Surface(z=z_data, colorscale='Earth')])
                    fig_surf.update_layout(
                        title=f"Topografía 3D (x{exageracion})",
                        autosize=True,
                        height=600,
                        scene=dict(
                            xaxis_title='X', yaxis_title='Y', zaxis_title='Altitud (m)',
                            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5) # Z más bajo por defecto
                        ),
                        margin=dict(l=10, r=10, b=10, t=40)
                    )
                    st.plotly_chart(fig_surf, use_container_width=True)

            # --- TAB 2: PENDIENTES CON TABLA CLASIFICADA ---
            with tab2:
                st.subheader(f"📐 Mapa de Pendientes y Clasificación")
                
                # 1. Mapa
                fig_slope = px.imshow(
                    slope_deg[::factor, ::factor], 
                    color_continuous_scale='Turbo',
                    labels={'color': 'Pendiente (°)'}
                )
                fig_slope.update_layout(height=500, title="Mapa de Pendientes")
                fig_slope.update_xaxes(showticklabels=False); fig_slope.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_slope, use_container_width=True)

                # 2. Tabla de Clasificación (FAO/IGAC simplificado)
                st.markdown("##### 📊 Distribución de Pendientes")
                slope_flat = slope_deg[~np.isnan(slope_deg)].flatten()
                total_area_px = len(slope_flat)
                
                # Rangos
                bins = [0, 3, 7, 12, 25, 50, 90]
                labels = ['Plano (0-3°)', 'Suave (3-7°)', 'Inclinado (7-12°)', 'Ondulado (12-25°)', 'Escarpado (25-50°)', 'Muy Escarpado (>50°)']
                
                # Clasificación usando Pandas Cut
                cats = pd.cut(slope_flat, bins=bins, labels=labels, include_lowest=True)
                counts = cats.value_counts().sort_index()
                
                df_slopes = pd.DataFrame({
                    'Categoría': counts.index,
                    'Píxeles': counts.values,
                    'Porcentaje (%)': (counts.values / total_area_px * 100).round(1),
                    'Área Estimada (ha)': (counts.values * (pixel_size**2) / 10000).round(1) # 30m x 30m = 900m2
                })
                
                st.dataframe(df_slopes, use_container_width=True)

            # --- TAB 3: HIPSOMETRÍA (Simplificada para no repetir código largo) ---
            with tab3:
                st.subheader(f"📈 Curva Hipsométrica")
                elevs_sorted = np.sort(elevs_valid)[::-1]
                n_pixels = len(elevs_sorted)
                area_percent = np.arange(1, n_pixels + 1) / n_pixels * 100
                
                # Reducción para graficar
                idx = np.linspace(0, n_pixels - 1, 200, dtype=int) if n_pixels > 200 else np.arange(n_pixels)
                
                fig_hypso = go.Figure()
                fig_hypso.add_trace(go.Scatter(
                    x=area_percent[idx], y=elevs_sorted[idx], 
                    mode='lines', fill='tozeroy', line=dict(color='#2E86C1')
                ))
                fig_hypso.update_layout(title="Curva Hipsométrica", xaxis_title="% Área", yaxis_title="Altitud (m)", height=500)
                st.plotly_chart(fig_hypso, use_container_width=True)

            # --- TAB 4: RED DE DRENAJE (CON PYSHEDS) ---
            with tab4:
                st.subheader("🌊 Red de Drenaje (Generada Automáticamente)")
                
                if not PYSHEDS_AVAILABLE:
                    st.error("⚠️ La librería 'pysheds' no está instalada. Por favor agrega `pysheds` a tu requirements.txt.")
                else:
                    try:
                        col_params, col_net = st.columns([1, 4])
                        with col_params:
                            st.info("Ajusta el umbral para definir qué es un río.")
                            umbral_rio = st.slider("Umbral de Acumulación", 100, 5000, 1000, 100, 
                                                 help="Número de celdas que drenan a un punto para considerarlo río. Menor valor = más ríos.")

                        with col_net:
                            # 1. Crear Grid de PySheds desde el Array
                            # PySheds prefiere leer de archivo, pero podemos simularlo o guardar temp
                            # Para eficiencia en nube, usamos un truco con MemoryFile si es posible, 
                            # o instanciamos Grid directamente si tenemos coordenadas.
                            
                            # Opción Robusta: Guardar temporalmente el recorte para PySheds
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                tmp_name = tmp.name
                                with rasterio.open(
                                    tmp_name, 'w', 
                                    **meta
                                ) as dst:
                                    dst.write(arr_elevacion.astype(rasterio.float32), 1)
                            
                            # Procesamiento PySheds
                            grid = Grid.from_raster(tmp_name)
                            dem_grid = grid.read_raster(tmp_name)
                            
                            # A. Rellenar Sumideros (Fill Pits)
                            pit_filled = grid.fill_pits(dem_grid)
                            
                            # B. Dirección de Flujo (Flow Direction)
                            fdir = grid.flowdir(pit_filled)
                            
                            # C. Acumulación (Flow Accumulation)
                            acc = grid.accumulation(fdir)
                            
                            # D. Definir Red (Stream Network)
                            # Convertimos a array numpy para visualizar
                            acc_arr = acc.view(np.ndarray)
                            stream_mask = acc_arr > umbral_rio
                            
                            # Limpieza archivo temp
                            os.remove(tmp_name)

                            # Visualización: Superponer Ríos sobre DEM
                            # Creamos una imagen RGBA donde los ríos son azules y el resto transparente
                            rios_img = np.zeros((acc_arr.shape[0], acc_arr.shape[1], 4)) # RGBA
                            rios_img[stream_mask] = [0, 0, 1, 1] # Azul opaco
                            
                            # Mapa Base (DEM)
                            fig_drain = px.imshow(
                                arr_elevacion[::factor, ::factor], 
                                color_continuous_scale='gray', 
                                title=f"Red de Drenaje (Umbral: {umbral_rio} celdas)"
                            )
                            # Nota: Plotly no soporta capas raster superpuestas fácilmente en imshow.
                            # Usaremos Scattergl para los puntos de río si no son demasiados, o solo mostramos la red.
                            
                            # Opción visual rápida: Mostrar la acumulación Logarítmica
                            log_acc = np.log1p(acc_arr[::factor, ::factor])
                            fig_flow = px.imshow(
                                log_acc,
                                color_continuous_scale='Blues',
                                title="Acumulación de Flujo (Simulación de Ríos)"
                            )
                            fig_flow.update_layout(height=600)
                            fig_flow.update_xaxes(showticklabels=False); fig_flow.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig_flow, use_container_width=True)
                            
                            st.caption("Nota: Las zonas más oscuras representan los cauces principales.")

                    except Exception as e:
                        st.error(f"Error generando red de drenaje: {e}")

            # --- TAB 5: DESCARGAS ---
            with tab5:
                st.subheader("📥 Centro de Descargas")
                st.markdown("Descarga los datos procesados para usarlos en QGIS, ArcGIS o Excel.")
                
                c1, c2, c3 = st.columns(3)
                
                # 1. Descargar DEM (TIF)
                tif_bytes = convert_array_to_tif_bytes(arr_elevacion, meta)
                if tif_bytes:
                    c1.download_button(
                        label="🗺️ Descargar DEM (.tif)",
                        data=tif_bytes,
                        file_name=f"DEM_{nombre_zona}.tif",
                        mime="image/tiff"
                    )
                
                # 2. Descargar Pendientes (TIF)
                slope_meta = meta.copy()
                slope_meta.update(dtype=rasterio.float32)
                slope_bytes = convert_array_to_tif_bytes(slope_deg, slope_meta)
                if slope_bytes:
                    c2.download_button(
                        label="📐 Descargar Pendientes (.tif)",
                        data=slope_bytes,
                        file_name=f"Slope_{nombre_zona}.tif",
                        mime="image/tiff"
                    )

                # 3. Descargar Estadísticas Pendientes (CSV)
                csv = df_slopes.to_csv(index=False).encode('utf-8')
                c3.download_button(
                    label="📊 Tabla Pendientes (.csv)",
                    data=csv,
                    file_name=f"Estadisticas_Pendientes_{nombre_zona}.csv",
                    mime="text/csv"
                )
                
                st.info("ℹ️ Los archivos TIF incluyen georreferenciación (EPSG:3116) y pueden abrirse directamente en software SIG.")

        else:
            st.warning("El recorte del DEM resultó en datos vacíos.")
else:
    st.info("👈 Selecciona una zona en la barra lateral.")
