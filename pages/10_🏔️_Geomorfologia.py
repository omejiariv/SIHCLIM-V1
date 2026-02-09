import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import plotly.express as px
import os
import pydeck as pdk
from modules import selectors

# Configuración de Página
st.set_page_config(page_title="Geomorfología Avanzada", page_icon="🏔️", layout="wide")

st.title("🏔️ Análisis Geomorfológico y Terreno 3D")
st.markdown("""
Esta herramienta utiliza el **Modelo Digital de Elevación (DEM)** para modelar el terreno, 
calcular pendientes y definir redes de drenaje.
""")

# --- 1. BARRA LATERAL (SELECTOR) ---
# Reutilizamos tu selector robusto que ya filtra por Región/Cuenca/Municipio
ids, nombre_zona, alt_ref, gdf_zona_seleccionada = selectors.render_selector_espacial()

# --- 2. CARGA DEL DEM (RASTER) ---
# Ruta del archivo (Ajusta la ruta si está en una subcarpeta 'data' o 'rasters')
DEM_PATH = os.path.join("data", "DemAntioquia_EPSG3116.tif")

@st.cache_data(show_spinner="Cortando DEM...")
def cargar_y_cortar_dem(ruta_dem, _gdf_corte):
    """
    Corta el DEM grande usando la geometría seleccionada.
    """
    if _gdf_corte is None or _gdf_corte.empty:
        return None, None, None

    try:
        if not os.path.exists(ruta_dem):
            return None, None, None

        with rasterio.open(ruta_dem) as src:
            crs_dem = src.crs
            # Usamos el argumento con guion bajo
            gdf_proyectado = _gdf_corte.to_crs(crs_dem)
            geoms = gdf_proyectado.geometry.values
            
            out_image, out_transform = mask(src, geoms, crop=True)
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            dem_array = out_image[0]
            dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
            dem_array = np.where(dem_array < -100, np.nan, dem_array)

            return dem_array, out_meta, out_transform

    except Exception as e:
        st.error(f"Error técnico procesando el DEM: {e}")
        return None, None, None
        
# --- 3. LÓGICA PRINCIPAL ---

@st.cache_data
def generar_mapa_3d(arr_elev, transform):
    """
    Genera una nube de puntos simplificada para visualización 3D en PyDeck.
    """
    # Submuestreo para rendimiento (max 100x100 puntos para fluidez)
    h, w = arr_elev.shape
    factor = max(1, int(max(h, w) / 100))
    
    # Crear malla de coordenadas
    rows, cols = np.indices(arr_elev.shape)
    elevs = arr_elev
    
    # Aplicar submuestreo
    rows = rows[::factor, ::factor].flatten()
    cols = cols[::factor, ::factor].flatten()
    elevs = elevs[::factor, ::factor].flatten()
    
    # Filtrar NaNs
    mask_valid = ~np.isnan(elevs)
    rows = rows[mask_valid]
    cols = cols[mask_valid]
    elevs = elevs[mask_valid]
    
    # Convertir índices pixel a coordenadas reales (EPSG:3116 -> Lat/Lon)
    # Nota: PyDeck necesita Lat/Lon. Aquí haremos una aproximación o reproyección.
    # Para simplificar hoy, usaremos un mapa de calor 3D sobre el mapa base.
    
    # ESTRATEGIA: Usar 'TerrainLayer' de PyDeck es complejo sin un servidor de teselas.
    # Usaremos 'ColumnLayer' (Hexágonos) que es más robusto para datos locales.
    
    # Transformación afín para obtener X, Y (Metros)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    df_3d = pd.DataFrame({
        "x": xs,
        "y": ys,
        "elev": elevs
    })
    
    # Convertir a Lat/Lon (Necesitamos pyproj)
    # Si no tienes pyproj instalado, esto fallará. 
    # ¿Tienes pyproj en requirements.txt? Si no, usaremos Plotly 3D que es más fácil.
    return df_3d

if gdf_zona_seleccionada is not None:
    # Verificación de archivo
    if not os.path.exists(DEM_PATH):
        st.error(f"⚠️ No encuentro el archivo DEM en: {DEM_PATH}")
        st.info("Por favor verifica que el archivo 'DemAntioquia_EPSG3116.tif' esté en la carpeta 'data'.")
    else:
        # Procesar DEM
        arr_elevacion, meta, transform = cargar_y_cortar_dem(DEM_PATH, gdf_zona_seleccionada)
        
        if arr_elevacion is not None:
            # Estadísticas Básicas
            min_el = np.nanmin(arr_elevacion)
            max_el = np.nanmax(arr_elevacion)
            mean_el = np.nanmean(arr_elevacion)
            
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Elevación Mínima", f"{min_el:.0f} m.s.n.m")
            c2.metric("Elevación Máxima", f"{max_el:.0f} m.s.n.m")
            c3.metric("Elevación Media", f"{mean_el:.0f} m.s.n.m")
            c4.metric("Rango Altitudinal", f"{max_el - min_el:.0f} m")

            # --- PESTAÑAS DE ANÁLISIS ---
            tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Elevación", "📈 Hipsometría", "🌊 Red de Drenaje (Beta)"])
            
            with tab1:
                st.subheader(f"Modelo Digital de Elevación 3D: {nombre_zona}")
                
                # --- VISUALIZACIÓN 3D INTERACTIVA (PLOTLY) ---
                import plotly.graph_objects as go

                # 1. Submuestreo inteligente (Downsampling)
                # Esto es vital: si intentamos graficar 1 millón de puntos, el navegador colapsa.
                # Calculamos un factor para tener aprox. una malla de 150x150 puntos, que se ve HD y es rápida.
                h, w = arr_elevacion.shape
                factor = max(1, int(max(h, w) / 150))
                
                # Creamos la versión ligera para el gráfico
                arr_3d = arr_elevacion[::factor, ::factor]
                
                # 2. Crear el Gráfico de Superficie (Surface Plot)
                fig_surf = go.Figure(data=[go.Surface(z=arr_3d, colorscale='Earth')])
                
                fig_surf.update_layout(
                    title=f"Topografía 3D - {nombre_zona}",
                    autosize=True,
                    width=800, 
                    height=600,
                    scene=dict(
                        xaxis_title='Oeste - Este',
                        yaxis_title='Sur - Norte',
                        zaxis_title='Altitud (m)',
                        # aspectmode='auto' ajusta la caja para que se vea bien visualmente
                        aspectmode='auto' 
                    ),
                    margin=dict(l=65, r=50, b=65, t=90)
                )
                
                st.plotly_chart(fig_surf, use_container_width=True)
                
                st.info(f"💡 Usa el mouse para rotar, acercar y explorar el relieve. (Factor de optimización: 1 píxel de cada {factor})")
                
            with tab2:
                st.subheader(f"📈 Análisis Hipsométrico: {nombre_zona}")
                
                if arr_elevacion is not None:
                    # 1. Preparación de Datos (Usando lo que ya tenemos en memoria)
                    elevs_valid = arr_elevacion[~np.isnan(arr_elevacion)].flatten()
                    
                    if len(elevs_valid) > 0:
                        # Ordenamos de Mayor a Menor (Descendente) para la curva estándar
                        elevs_sorted = np.sort(elevs_valid)[::-1]
                        
                        n_pixels = len(elevs_sorted)
                        # Eje X: Porcentaje del Área Acumulada (0% a 100%)
                        area_percent = np.arange(1, n_pixels + 1) / n_pixels * 100
                        
                        # --- 2. OPTIMIZACIÓN (Lógica de tu analysis.py) ---
                        # Si hay demasiados puntos, reducimos a 200 para que el gráfico vuele
                        if n_pixels > 200:
                            indices = np.linspace(0, n_pixels - 1, 200, dtype=int)
                            elevations_plot = elevs_sorted[indices]
                            area_plot = area_percent[indices]
                        else:
                            elevations_plot = elevs_sorted
                            area_plot = area_percent

                        # --- 3. MODELO MATEMÁTICO (Tu joya de código) ---
                        eq_str = "N/A"
                        try:
                            # Ajuste polinómico de grado 3
                            coeffs = np.polyfit(area_plot, elevations_plot, 3)
                            
                            # Formateo elegante de la ecuación
                            eq_str = (
                                f"H = {coeffs[0]:.2e}A³ "
                                f"{'+' if coeffs[1]>=0 else '-'} {abs(coeffs[1]):.2e}A² "
                                f"{'+' if coeffs[2]>=0 else '-'} {abs(coeffs[2]):.2e}A "
                                f"{'+' if coeffs[3]>=0 else '-'} {abs(coeffs[3]):.2f}"
                            )
                        except Exception:
                            pass

                        # --- 4. CÁLCULO DE INTEGRAL HIPSOMÉTRICA (HI) ---
                        min_h = np.min(elevs_valid)
                        max_h = np.max(elevs_valid)
                        mean_h = np.mean(elevs_valid)
                        median_h = np.median(elevs_valid)
                        
                        hi = (mean_h - min_h) / (max_h - min_h)
                        
                        # Interpretación Geomorfológica
                        estado_cuenca = "Madura (Equilibrio)"
                        icono_estado = "🏞️"
                        interpretacion = "La cuenca ha alcanzado un equilibrio entre erosión y sedimentación."
                        
                        if hi > 0.60: 
                            estado_cuenca = "Joven (Fase Activa)"
                            icono_estado = "🌋"
                            interpretacion = "Altas tasas de erosión y laderas inestables. Potencial torrencial."
                        elif hi < 0.35: 
                            estado_cuenca = "Vieja (Senil)"
                            icono_estado = "🏝️"
                            interpretacion = "Dominio de la sedimentación. Relieve muy desgastado."

                        # --- 5. VISUALIZACIÓN ---
                        
                        # Métricas
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Integral Hipsométrica (HI)", f"{hi:.3f}")
                        c2.metric("Estado", estado_cuenca)
                        c3.metric("Altitud Mediana", f"{median_h:.0f} m")
                        
                        st.markdown(f"**📐 Ecuación del Relieve:** `$ {eq_str} $`")

                        # Gráfico Plotly
                        import plotly.graph_objects as go
                        fig_hypso = go.Figure()
                        
                        fig_hypso.add_trace(go.Scatter(
                            x=area_plot, 
                            y=elevations_plot, 
                            mode='lines', 
                            name='Curva Real',
                            line=dict(color='#2E86C1', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(46, 134, 193, 0.2)'
                        ))
                        
                        # Referencias
                        fig_hypso.add_hline(y=mean_h, line_dash="dash", line_color="green", annotation_text="Media")
                        fig_hypso.add_hline(y=median_h, line_dash="dash", line_color="orange", annotation_text="Mediana")

                        fig_hypso.update_layout(
                            title=f"Curva Hipsométrica - {nombre_zona}",
                            xaxis_title="% Área Acumulada (A)",
                            yaxis_title="Altitud H (m.s.n.m)",
                            template="plotly_white",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig_hypso, use_container_width=True)
                        
                        st.info(f"{icono_estado} **Diagnóstico:** {interpretacion}")

                    else:
                        st.warning("Datos insuficientes para calcular la curva.")
                
            with tab3:
                st.info("Aquí procesaremos el DEM con PySheds para obtener ríos.")

else:
    st.info("👈 Por favor selecciona una Cuenca o Municipio en la barra lateral para iniciar el análisis.")
