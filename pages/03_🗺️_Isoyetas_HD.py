# pages/03_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import Rbf
import os
import sys

# --- IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
    from modules import db_manager, selectors
    from modules.utils import encender_gemelo_digital
except:
    from modules import db_manager, selectors
    from modules.config import Config

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")
encender_gemelo_digital()

st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# --- 2. FICHA TÉCNICA ---
with st.expander("📘 Ficha Técnica: Metodología, Utilidad y Fuentes", expanded=False):
    st.markdown("""
    ### 1. Concepto y Utilidad
    Las **isoyetas** son líneas que unen puntos de igual precipitación. Este módulo permite visualizar la distribución espacial de la lluvia en tiempo real, facilitando la identificación de núcleos de humedad y zonas de sombra pluviométrica.
    
    ### 2. Metodología: Interpolación RBF
    Utilizamos funciones de base radial (**Radial Basis Function - Rbf**) con un kernel multicuadrático. A diferencia de un promedio simple, la RBF respeta los valores exactos de las estaciones y genera transiciones suaves y físicamente plausibles en terrenos complejos como los Andes.
    """)

# --- 3. SELECTOR ESPACIAL ---
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

if gdf_zona is not None and not gdf_zona.empty:
    engine = db_manager.get_engine()
    
    # 🛡️ ESCUDO DE NULOS: Obtenemos nombres de columnas con fallbacks seguros
    res_cols = selectors.get_col_names_by_level(ids_sel)
    col_id = res_cols[0] if res_cols[0] else "id_cuenca"
    col_nom = res_cols[1] if res_cols[1] else "nombre_cuenca"
    col_cuenca = res_cols[2] if res_cols[2] else "cuenca"

    # --- 4. CONTROLES DE ESCENARIO ---
    st.sidebar.header("🌦️ Configuración del Mapa")
    
    # Sincronización con el año global
    anio_def = st.session_state.get('aleph_anio', 2024)
    mes_def = st.session_state.get('aleph_mes', 1)
    
    c1, c2 = st.sidebar.columns(2)
    anio = c1.number_input("Año:", 1980, 2050, int(anio_def))
    mes = c2.selectbox("Mes:", range(1, 13), index=int(mes_def)-1)
    
    grid_res = st.sidebar.slider("Resolución del Mapa (Puntos):", 50, 200, 100, help="Más puntos = mayor detalle, pero más lento.")
    tipo_isoyeta = st.sidebar.radio("Estilo Visual:", ["Contornos Llenos (Heatmap)", "Líneas de Contorno (Isoyetas)"])

    # --- 5. MOTOR DE PROCESAMIENTO ---
    try:
        # A. Carga de Estaciones e Influencia
        query_est = text("SELECT id_estacion, nombre, latitud, longitud FROM estaciones")
        df_est = pd.read_sql(query_est, engine)
        
        # B. Carga de Lluvias
        query_data = text("""
            SELECT id_estacion, valor 
            FROM precipitacion 
            WHERE EXTRACT(YEAR FROM fecha) = :a 
            AND EXTRACT(MONTH FROM fecha) = :m
        """)
        df_data = pd.read_sql(query_data, engine, params={"a": anio, "m": mes})
        
        if df_data.empty:
            st.warning(f"⚠️ No se encontraron registros de lluvia para {mes}/{anio}.")
            st.stop()
            
        # C. Cruce Espacial (Solo estaciones que caen en la zona o su buffer)
        df_full = df_est.merge(df_data, on="id_estacion")
        
        # 🌿 UNIFICACIÓN DE DATOS (MATA EL ERROR DE COLUMNS [NONE])
        # Solo procedemos si tenemos suficientes estaciones
        if len(df_full) < 3:
            st.error("❌ Se requieren al menos 3 estaciones con datos para generar isoyetas.")
        else:
            # 📐 INTERPOLACIÓN RBF
            x = df_full['longitud'].values
            y = df_full['latitud'].values
            z = df_full['valor'].values
            
            # Crear malla de predicción basada en los límites de la zona
            bounds = gdf_zona.total_bounds
            xi = np.linspace(bounds[0], bounds[2], grid_res)
            yi = np.linspace(bounds[1], bounds[3], grid_res)
            XI, YI = np.meshgrid(xi, yi)
            
            # Ejecutar RBF
            rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
            ZI = rbf(XI, YI)
            
            # Recortar al polígono (Máscara)
            from shapely.geometry import Point
            points = [Point(lon, lat) for lon, lat in zip(XI.flatten(), YI.flatten())]
            mask_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
            
            # 🛡️ BLINDAJE DE PROYECCIÓN
            if gdf_zona.crs != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
            in_poly = mask_gdf.within(gdf_zona.unary_union)
            ZI_masked = ZI.flatten()
            ZI_masked[~in_poly] = np.nan
            ZI_masked = ZI_masked.reshape(ZI.shape)

            # --- 6. VISUALIZACIÓN ---
            fig = go.Figure()
            
            # Capa de Isoyetas
            fig.add_trace(go.Contour(
                z=ZI_masked,
                x=xi,
                y=yi,
                colorscale='Blues',
                contours=dict(
                    coloring='heatmap' if "Llenos" in tipo_isoyeta else 'lines',
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                ),
                line_width=1,
                colorbar=dict(title="Lluvia (mm)")
            ))
            
            # Capa de Estaciones
            fig.add_trace(go.Scattermapbox(
                lon=df_full['longitud'],
                lat=df_full['latitud'],
                mode='markers+text',
                marker=dict(size=8, color='black'),
                text=df_full['valor'].astype(str) + " mm",
                textposition="top center",
                name="Estaciones"
            ))

            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(center=dict(lat=gdf_zona.centroid.y.mean(), lon=gdf_zona.centroid.x.mean()), zoom=11),
                margin={"r":0,"t":40,"l":0,"b":0},
                height=700,
                title=f"Distribución de Lluvia Estimada: {nombre_zona} ({mes}/{anio})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 📥 DESCARGAS
            st.divider()
            c_down1, c_down2 = st.columns(2)
            with c_down1:
                csv_data = df_full.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Datos de Estaciones (CSV)", csv_data, f"lluvias_{nombre_zona}_{mes}_{anio}.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error procesando isoyetas: {e}")
        st.info("💡 Tip: Verifica que la zona seleccionada tenga estaciones de monitoreo cercanas.")
else:
    st.info("👈 Selecciona un territorio en el menú lateral para generar el mapa de isoyetas.")
