# pages/03_🍃_Biodiversidad.py

import streamlit as st
import sys
import os
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px

# --- IMPORTACIÓN DE MÓDULOS ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector, carbon_calculator
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")
st.title("🍃 Biodiversidad y Servicios Ecosistémicos")

# 2. SELECTOR ESPACIAL
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error en selector: {e}")
    st.stop()

def save_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(ttl=3600)
def load_layer_cached(layer_name):
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson",
        "Predios": "PrediosEjecutados.geojson"
    }
    if layer_name in file_map:
        try:
            # Ajuste de ruta robusto
            file_path = os.path.join(config.Config.DATA_DIR, file_map[layer_name])
            if not os.path.exists(file_path):
                # Fallback por si DATA_DIR no resuelve bien en cloud
                file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_map[layer_name]))
            
            if os.path.exists(file_path):
                gdf = gpd.read_file(file_path)
                if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                return gdf
        except: return None
    return None

# --- DEFINICIÓN DE TABS PRINCIPALES ---
tab_mapa, tab_tax, tab_carbon = st.tabs(["🗺️ Mapa & GBIF", "📊 Taxonomía", "🌳 Calculadora Carbono"])

# Variable global para datos de biodiversidad
gdf_bio = pd.DataFrame()
threatened = pd.DataFrame()
n_threat = 0

# --- PROCESAMIENTO PREVIO (Solo si hay zona) ---
if gdf_zona is not None:
    with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)
        
    if not gdf_bio.empty and 'Amenaza IUCN' in gdf_bio.columns:
        threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
        n_threat = threatened['Nombre Científico'].nunique()

# ==============================================================================
# TAB 1: MAPA Y MÉTRICAS
# ==============================================================================
with tab_mapa:
    if gdf_zona is not None:
        # 1. Métricas Principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros GBIF", f"{len(gdf_bio):,.0f}")
        c2.metric("Especies", f"{gdf_bio['Nombre Científico'].nunique():,.0f}" if not gdf_bio.empty else "0")
        c3.metric("Familias", f"{gdf_bio['Familia'].nunique():,.0f}" if not gdf_bio.empty and 'Familia' in gdf_bio.columns else "0")
        c4.metric("Amenazadas (IUCN)", f"{n_threat}")

        # 2. Visor Territorial
        st.markdown("##### Visor Territorial")
        
        fig = go.Figure()

        # A. CENTRO DEL MAPA
        try:
            center = gdf_zona.to_crs("+proj=cea").centroid.to_crs("EPSG:4326").iloc[0]
            center_lat, center_lon = center.y, center.x
        except: center_lat, center_lon = 6.5, -75.5

        # B. CAPA ZONA (ROJO)
        for idx, row in gdf_zona.iterrows():
            if row.geometry:
                polys = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms) if row.geometry.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=3, color='red'), name='Zona Selección', hoverinfo='skip'))

        # C. CAPAS DE CONTEXTO (Municipios, Cuencas, Predios)
        layers_to_show = [("Municipios", "gray", 1), ("Cuencas", "blue", 1.5), ("Predios", "orange", 1)]
        
        for lyr_name, color, width in layers_to_show:
            gdf_lyr = load_layer_cached(lyr_name)
            if gdf_lyr is not None:
                # Recorte espacial para optimizar (solo predios)
                if lyr_name == "Predios":
                    try:
                        roi_buf = gdf_zona.to_crs("EPSG:3116").buffer(1000).to_crs("EPSG:4326")
                        gdf_lyr = gpd.clip(gdf_lyr, roi_buf)
                    except: pass
                
                if not gdf_lyr.empty:
                    # Dibujamos solo el primer polígono con leyenda, el resto oculto en grupo
                    for idx, row in gdf_lyr.iterrows():
                        if row.geometry:
                            polys = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms) if row.geometry.geom_type == 'MultiPolygon' else []
                            for i, poly in enumerate(polys):
                                x, y = poly.exterior.xy
                                show_leg = True if idx == 0 and i == 0 else False
                                visible_opt = 'legendonly' if lyr_name == "Predios" else True
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines', 
                                    line=dict(width=width, color=color), 
                                    name=lyr_name, legendgroup=lyr_name, 
                                    showlegend=show_leg, hoverinfo='skip', visible=visible_opt
                                ))

        # D. PUNTOS DE BIODIVERSIDAD (VERDE)
        if not gdf_bio.empty:
            fig.add_trace(go.Scattermapbox(
                lon=gdf_bio['lon'], lat=gdf_bio['lat'], 
                mode='markers', marker=dict(size=7, color='rgb(0, 200, 100)'), 
                text=gdf_bio['Nombre Común'], name='Biodiversidad'
            ))

        fig.update_layout(
            mapbox_style="carto-positron", 
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10), 
            margin={"r":0,"t":0,"l":0,"b":0}, height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if not gdf_bio.empty:
            st.download_button("💾 Descargar Datos (CSV)", save_to_csv(gdf_bio.drop(columns='geometry', errors='ignore')), f"biodiv_{nombre_seleccion}.csv", "text/csv")

    else:
        st.info("👈 Seleccione una zona en el menú lateral para visualizar el mapa.")

# ==============================================================================
# TAB 2: TAXONOMÍA
# ==============================================================================
with tab_tax:
    if not gdf_bio.empty:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("##### Estructura Taxonómica")
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                df_chart = gdf_bio.fillna("Sin Dato")
                fig_sun = px.sunburst(df_chart, path=['Reino', 'Clase', 'Orden', 'Familia'], height=600)
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Datos taxonómicos insuficientes.")
        
        with c2:
            st.markdown("##### Especies Amenazadas")
            if not threatened.empty:
                st.warning(f"⚠️ {n_threat} especies en riesgo.")
                st.dataframe(threatened[['Nombre Científico', 'Nombre Común', 'Amenaza IUCN']].drop_duplicates(), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No se detectaron especies en categorías críticas (CR, EN, VU) en esta zona.")
        
        st.markdown("##### Detalle de Registros")
        st.dataframe(gdf_bio.drop(columns='geometry', errors='ignore'), use_container_width=True)
    else:
        st.info("No hay datos de biodiversidad para mostrar estadísticas.")

# ==============================================================================
# TAB 3: CALCULADORA DE CARBONO
# ==============================================================================
with tab_carbon:
    st.header("🌳 Estimación de Captura de Carbono")
    st.markdown("""
    Herramienta alineada con metodologías **MDL (AR-TOOL14)** y coeficientes nacionales **(Álvarez et al. 2012)**.
    Permite estimar el potencial de mitigación climática de proyectos de restauración.
    """)

    modo_calc = st.radio("Selecciona el tipo de análisis:", 
                         ["🔮 Proyección (Restauración Futura)", "📏 Inventario (Medición en Campo)"], 
                         horizontal=True)
    
    st.divider()

    # --- MODO 1: PROYECCIÓN (VON BERTALANFFY) ---
    if "Proyección" in modo_calc:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Parámetros del Proyecto")
            area_ha = st.number_input("Área a restaurar (Ha):", min_value=0.1, value=1.0, step=0.1)
            anios_proj = st.slider("Horizonte de proyección (años):", 5, 50, 20)
            
            tipo_bosque = st.selectbox("Modelo de Crecimiento:", 
                                       ["Bosque Húmedo Tropical (Restauración)", "Bosque Seco Tropical (Teórico)"])
            
            if st.button("🚀 Calcular Proyección", type="primary"):
                # Llamada al motor de cálculo
                df_proj = carbon_calculator.calcular_proyeccion_captura(area_ha, anios_proj)
                st.session_state['df_carbon_proj'] = df_proj
        
        with c2:
            if 'df_carbon_proj' in st.session_state:
                df = st.session_state['df_carbon_proj']
                
                # KPIs Rápidos
                total_captura = df['Proyecto_tCO2e_Acumulado'].iloc[-1]
                tasa_media = df['Proyecto_tCO2e_Anual'].mean()
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Captura Total (20 años)", f"{total_captura:,.0f} tCO2e")
                k2.metric("Tasa Promedio", f"{tasa_media:.1f} tCO2e/año")
                k3.metric("Bono Carbono Est.", f"${total_captura * 5:,.0f} USD", help="Estimado a 5 USD/ton")
                
                # Gráfico de Área Acumulada
                fig = px.area(df, x='Año', y='Proyecto_tCO2e_Acumulado', 
                              title=f"Curva de Captura Acumulada ({area_ha} Ha)",
                              labels={'Proyecto_tCO2e_Acumulado': 'Toneladas CO2e'},
                              color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla Detallada
                with st.expander("Ver tabla año a año"):
                    st.dataframe(df.style.format("{:.2f}"))

    # --- MODO 2: INVENTARIO (ALOMÉTRICO) ---
    else:
        st.subheader("📏 Calculadora de Stock Actual (Inventario)")
        st.info("Sube un archivo Excel/CSV con las columnas: `DAP` (cm) y `Altura` (m). Opcional: `Densidad`.")
        
        up_file = st.file_uploader("Cargar Inventario Forestal", type=['csv', 'xlsx'])
        zona_vida = st.selectbox("Zona de Vida (Holdridge):", 
                                 ["bh-MB", "bh-PM", "bh-T", "bmh-M", "bp-PM"], 
                                 index=2, help="Determina los coeficientes de la ecuación alométrica.")
        
        if up_file:
            try:
                if up_file.name.endswith('.csv'):
                    df_inv = pd.read_csv(up_file, sep=';' if ';' in up_file.getvalue().decode('latin1') else ',')
                else:
                    df_inv = pd.read_excel(up_file)
                
                if st.button("🧮 Calcular Biomasa y Carbono"):
                    df_res, msg = carbon_calculator.calcular_inventario_forestal(df_inv, zona_vida)
                    
                    if df_res is not None:
                        st.success("✅ Cálculo exitoso.")
                        
                        # Resultados
                        tot_arboles = len(df_res)
                        tot_co2 = df_res['CO2e_Total_tCO2e'].sum()
                        avg_dap = df_res['DAP'].mean()
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Árboles Evaluados", f"{tot_arboles}")
                        m2.metric("DAP Promedio", f"{avg_dap:.1f} cm")
                        m3.metric("Stock Total", f"{tot_co2:,.2f} tCO2e")
                        
                        st.dataframe(df_res.head())
                        
                        # Descarga
                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Descargar Resultados Detallados", csv, "inventario_calculado.csv", "text/csv")
                    else:
                        st.error(f"Error: {msg}")
            except Exception as e:
                st.error(f"Error leyendo archivo: {e}")
