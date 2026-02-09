import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import plotly.express as px
import os
from modules import selectors
import plotly.graph_objects as go

# Configuración de Página
st.set_page_config(page_title="Geomorfología Avanzada", page_icon="🏔️", layout="wide")

st.title("🏔️ Análisis Geomorfológico y Terreno 3D")
st.markdown("""
Esta herramienta utiliza el **Modelo Digital de Elevación (DEM)** para modelar el terreno, 
calcular pendientes y realizar diagnósticos hidrológicos automáticos.
""")

# --- 1. BARRA LATERAL (SELECTOR) ---
ids, nombre_zona, alt_ref, gdf_zona_seleccionada = selectors.render_selector_espacial()

# 🛠️ CORRECCIÓN CLAVE: Convertir Puntos (Regiones) en Polígono (Caja)
# Esto permite que zonas como "Oriente" o "Bajo Cauca" funcionen.
if gdf_zona_seleccionada is not None and not gdf_zona_seleccionada.empty:
    if gdf_zona_seleccionada.geom_type.isin(['Point', 'MultiPoint']).any():
        # Buffer pequeño si es un solo punto, o caja envolvente si son varios (Región)
        if len(gdf_zona_seleccionada) == 1:
            gdf_zona_seleccionada['geometry'] = gdf_zona_seleccionada.buffer(0.045) # aprox 5km
        else:
            bbox = gdf_zona_seleccionada.unary_union.envelope
            gdf_zona_seleccionada = gpd.GeoDataFrame({'geometry': [bbox]}, crs=gdf_zona_seleccionada.crs)

# --- 2. CARGA DEL DEM (RASTER) MEJORADA ---
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
            
            # Intentamos cortar. Si la zona está fuera del mapa, rasterio lanza ValueError
            try:
                out_image, out_transform = mask(src, geoms, crop=True)
            except ValueError:
                # Esto ocurre cuando el polígono no toca el mapa (Urabá, Bajo Cauca, etc.)
                return None, "OUT_OF_BOUNDS", None
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            dem_array = out_image[0]
            
            # Filtros de limpieza
            # Reemplazamos el valor NoData del archivo por NaN
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            # Filtro adicional para errores negativos o vacíos absolutos
            dem_array = np.where(dem_array < -100, np.nan, dem_array)
            
            # Verificación final: Si todo es NaN, es un recorte vacío
            if np.isnan(dem_array).all():
                 return None, "EMPTY_DATA", None

            return dem_array, out_meta, out_transform

    except Exception as e:
        # Error técnico real
        st.error(f"Error técnico procesando el DEM: {e}")
        return None, None, None

# --- 3. CEREBRO DEL ANALISTA INTELIGENTE 🧠 ---
def analista_hidrologico(pendiente_media, hi_value):
    diagnostico = ""
    tipo_cuenca = ""
    
    # Pendiente
    if pendiente_media > 25:
        txt_pendiente = "un relieve fuertemente escarpado"
        riesgo_pendiente = "alto potencial de flujos torrenciales y tiempos de concentración muy cortos"
    elif pendiente_media > 12:
        txt_pendiente = "un relieve moderadamente ondulado"
        riesgo_pendiente = "velocidades de flujo moderadas"
    else:
        txt_pendiente = "un relieve predominantemente plano"
        riesgo_pendiente = "baja velocidad de flujo, propensión al encharcamiento"

    # Hipsometría
    if hi_value > 0.50:
        tipo_cuenca = "Cuenca Joven (En Desequilibrio)"
        txt_hi = "indica una fase activa de erosión (Juventud)"
    elif hi_value < 0.35:
        tipo_cuenca = "Cuenca Vieja (Senil)"
        txt_hi = "indica una fase avanzada de sedimentación (Senectud)"
    else:
        tipo_cuenca = "Cuenca Madura"
        txt_hi = "indica un estado de equilibrio dinámico"

    # Diagnóstico Final
    diagnostico = f"""
    **Diagnóstico del Analista:**
    La zona analizada presenta **{txt_pendiente}** (Pendiente media: {pendiente_media:.1f}°), lo que sugiere {riesgo_pendiente}.
    
    Desde el punto de vista evolutivo, se clasifica como una **{tipo_cuenca}** (HI: {hi_value:.3f}). Esto {txt_hi}.
    
    **Implicación Hidrológica:** {'⚠️ Se recomienda monitoreo de avenidas torrenciales y erosión de laderas.' if pendiente_media > 20 else 'ℹ️ La gestión debe enfocarse en el control de inundaciones lentas y drenaje.'}
    """
    return diagnostico

# --- 4. LÓGICA PRINCIPAL ---

if gdf_zona_seleccionada is not None:
    if not os.path.exists(DEM_PATH):
        st.error(f"⚠️ No encuentro el archivo DEM en: {DEM_PATH}")
    else:
        # Procesar DEM
        arr_elevacion, meta, transform = cargar_y_cortar_dem(DEM_PATH, gdf_zona_seleccionada, nombre_zona)
        
        # MANEJO DE CASOS DE ERROR CONTROLADO
        if meta == "OUT_OF_BOUNDS":
            st.warning(f"⚠️ **Fuera de Cobertura:** La zona '{nombre_zona}' está fuera de los límites del mapa de elevación actual (DEM).")
            st.info("💡 Por favor selecciona una zona en la región andina/central de Antioquia o carga un DEM de mayor cobertura.")
        
        elif meta == "EMPTY_DATA":
             st.warning(f"⚠️ **Datos Vacíos:** El recorte se realizó, pero no contiene datos de elevación válidos.")
             
        elif arr_elevacion is not None and not np.isnan(arr_elevacion).all():
            
            # --- CÁLCULOS GLOBALES ---
            elevs_valid = arr_elevacion[~np.isnan(arr_elevacion)].flatten()
            min_el, max_el = np.min(elevs_valid), np.max(elevs_valid)
            mean_el = np.mean(elevs_valid)
            
            # Protección contra división por cero en HI
            rango = max_el - min_el
            hi_global = (mean_el - min_el) / rango if rango > 0 else 0.5
            
            # Pendientes
            pixel_size = 30.0 
            dy, dx = np.gradient(arr_elevacion, pixel_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_mean_global = np.nanmean(slope_deg)
            max_slope = np.nanmax(slope_deg)
            
            # Texto del Analista
            texto_analisis = analista_hidrologico(slope_mean_global, hi_global)

            # --- VISUALIZACIÓN DE MÉTRICAS ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Elevación Mínima", f"{min_el:.0f} m")
            c2.metric("Elevación Máxima", f"{max_el:.0f} m")
            c3.metric("Elevación Media", f"{mean_el:.0f} m")
            c4.metric("Rango Altitudinal", f"{max_el - min_el:.0f} m")

            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "🗺️ Elevación 3D", 
                "📐 Pendientes", 
                "📈 Hipsometría", 
                "🌊 Red de Drenaje (Beta)"
            ])
            
            # Factor de reducción para gráficos
            h, w = arr_elevacion.shape
            factor = max(1, int(max(h, w) / 150))

            # --- TAB 1: 3D ---
            with tab1:
                st.subheader(f"Modelo Digital de Elevación 3D: {nombre_zona}")
                arr_3d = arr_elevacion[::factor, ::factor]
                
                fig_surf = go.Figure(data=[go.Surface(z=arr_3d, colorscale='Earth')])
                fig_surf.update_layout(
                    title=f"Topografía 3D - {nombre_zona}",
                    autosize=True,
                    height=700,
                    scene=dict(
                        xaxis_title='Oeste - Este',
                        yaxis_title='Sur - Norte',
                        zaxis_title='Altitud (m)',
                        aspectmode='auto' 
                    ),
                    margin=dict(l=10, r=10, b=10, t=40)
                )
                st.plotly_chart(fig_surf, use_container_width=True)
                st.caption("Usa el mouse para rotar el modelo.")

            # --- TAB 2: PENDIENTES (CORREGIDO Y BLINDADO) ---
            with tab2:
                st.subheader(f"📐 Mapa de Pendientes y Riesgo")
                
                # 1. CÁLCULO SEGURO DE ESTADÍSTICAS (Anti-ZeroDivision)
                total_pixeles_validos = np.count_nonzero(~np.isnan(slope_deg))
                
                if total_pixeles_validos > 0:
                    mean_slope = np.nanmean(slope_deg)
                    max_slope = np.nanmax(slope_deg)
                    count_escarpado = np.count_nonzero((slope_deg > 30) & (~np.isnan(slope_deg)))
                    pct_escarpado = (count_escarpado / total_pixeles_validos) * 100
                else:
                    mean_slope = 0.0
                    max_slope = 0.0
                    pct_escarpado = 0.0

                # 2. MOSTRAR MÉTRICAS
                col_met1, col_met2, col_met3 = st.columns(3)
                col_met1.metric("Pendiente Media", f"{mean_slope:.1f}°")
                col_met2.metric("Pendiente Máxima", f"{max_slope:.1f}°")
                col_met3.metric("% Área Escarpada (>30°)", f"{pct_escarpado:.1f}%")
                
                # 3. VISUALIZACIÓN DEL MAPA
                if total_pixeles_validos > 0:
                    fig_slope = px.imshow(
                        slope_deg[::factor, ::factor], 
                        color_continuous_scale='Turbo',
                        title=f"Mapa de Pendientes - {nombre_zona}",
                        labels={'color': 'Pendiente (°)'}
                    )
                    fig_slope.update_xaxes(showticklabels=False) 
                    fig_slope.update_yaxes(showticklabels=False)
                    fig_slope.update_layout(height=600)
                    st.plotly_chart(fig_slope, use_container_width=True)
                else:
                    st.warning("⚠️ No hay datos de terreno suficientes para calcular pendientes.")

                # 4. DIAGNÓSTICO DEL ANALISTA (Solo si hay datos)
                if total_pixeles_validos > 0:
                    st.info(texto_analisis, icon="🤖")

            # --- TAB 3: HIPSOMETRÍA ---
            with tab3:
                st.subheader(f"📈 Curva Hipsométrica")
                
                elevs_sorted = np.sort(elevs_valid)[::-1]
                n_pixels = len(elevs_sorted)
                area_percent = np.arange(1, n_pixels + 1) / n_pixels * 100
                
                if n_pixels > 200:
                    indices = np.linspace(0, n_pixels - 1, 200, dtype=int)
                    elevations_plot = elevs_sorted[indices]
                    area_plot = area_percent[indices]
                else:
                    elevations_plot = elevs_sorted
                    area_plot = area_percent

                # Ecuación
                eq_str = "N/A"
                try:
                    coeffs = np.polyfit(area_plot, elevations_plot, 3)
                    eq_str = (
                        f"H = {coeffs[0]:.2e}A³ "
                        f"{'+' if coeffs[1]>=0 else '-'} {abs(coeffs[1]):.2e}A² "
                        f"{'+' if coeffs[2]>=0 else '-'} {abs(coeffs[2]):.2e}A "
                        f"{'+' if coeffs[3]>=0 else '-'} {abs(coeffs[3]):.2f}"
                    )
                except: pass

                st.markdown(f"**📐 Ecuación del Relieve:** `$ {eq_str} $`")

                fig_hypso = go.Figure()
                fig_hypso.add_trace(go.Scatter(
                    x=area_plot, y=elevations_plot, mode='lines', name='Curva Real',
                    line=dict(color='#2E86C1', width=3), fill='tozeroy'
                ))
                fig_hypso.update_layout(
                    title="Distribución de Altitudes",
                    xaxis_title="% Área Acumulada",
                    yaxis_title="Altitud (m)",
                    height=500,
                    template="plotly_white"
                )
                st.plotly_chart(fig_hypso, use_container_width=True)
                st.success(f"**Diagnóstico Hipsométrico:** Se clasifica como una **{('Cuenca Joven' if hi_global > 0.5 else 'Cuenca Vieja')}** (HI: {hi_global:.3f}).")

            # --- TAB 4: RED DE DRENAJE ---
            with tab4:
                st.subheader("🌊 Red de Drenaje Teórica (Beta)")
                st.warning("🚧 Módulo en construcción.")

        else:
            st.warning("El recorte del DEM resultó en datos vacíos.")
else:
    st.info("👈 Selecciona una zona en la barra lateral.")
