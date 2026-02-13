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

# --- FUNCIÓN DE INTEGRACIÓN: DETECTAR ZONA DE VIDA ---
def detectar_zona_vida_dominante(gdf_zona):
    """
    Usa el módulo life_zones para estimar la zona climática del polígono seleccionado
    sin tener que procesar todo el raster pesado si no es necesario.
    """
    try:
        # 1. Calculamos el centroide de la zona seleccionada
        centroid = gdf_zona.to_crs("+proj=cea").centroid.to_crs("EPSG:4326").iloc[0]
        altitud = altitud_ref if altitud_ref > 0 else 1500 # Default si falla
        
        # 2. Obtenemos precipitación promedio (Simulada o de base de datos)
        # En una integración total, aquí leeríamos el raster de PPAMAnt.tif en ese punto
        ppt_estimada = 2000 # Valor medio para la región si no hay raster cargado
        
        # 3. Usamos la lógica de clasificación de life_zones.py
        # ID -> Nombre
        zona_id = lz.classify_life_zone_alt_ppt(altitud, ppt_estimada)
        zona_nombre = lz.holdridge_int_to_name_simplified.get(zona_id, "Desconocido")
        
        # 4. Mapeo a códigos de Álvarez (Ecuaciones)
        # Esto es un diccionario de traducción simple para el ejemplo
        mapa_codigos = {
            "Bosque húmedo Premontano (bh-PM)": "bh-PM",
            "Bosque muy húmedo Premontano (bmh-PM)": "bmh-PM",
            "Bosque muy húmedo Montano (bmh-M)": "bmh-M",
            "Bosque húmedo Tropical (bh-T)": "bh-T"
        }
        return mapa_codigos.get(zona_nombre, "bh-MB") # Default seguro
    except:
        return "bh-MB"

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
# TAB 3: CALCULADORA DE CARBONO (INTEGRACIÓN SISTÉMICA)
# ==============================================================================
with tab_carbon:
    st.header("🌳 Estimación de Servicios Ecosistémicos (Carbono)")
    
    # --- A. ANÁLISIS DEL SISTEMA (Contexto Automático) ---
    st.info("🤖 **Análisis Sistémico:** El sistema ha detectado las condiciones de tu zona seleccionada.")
    
    c_sys_1, c_sys_2 = st.columns(2)
    
    # 1. DETECCIÓN DE ZONA DE VIDA (Conexión con life_zones.py)
    zv_detectada = detectar_zona_vida_dominante(gdf_zona) if gdf_zona is not None else "bh-MB"
    
    with c_sys_1:
        st.markdown(f"**📍 Zona de Vida Detectada:** `{zv_detectada}`")
        st.caption("Basado en la altitud y precipitación de la geometría seleccionada.")

    # 2. DETECCIÓN DE ÁREA POTENCIAL (Conexión con land_cover.py)
    area_potencial = 0
    with c_sys_2:
        # Intentamos calcular área de pastos en la zona
        if gdf_zona is not None:
            # Aquí idealmente llamaríamos a land_cover logic, simulamos por rapidez:
            area_total_ha = gdf_zona.to_crs("+proj=cea").area.sum() / 10000
            area_potencial = area_total_ha * 0.4 # Supuesto: 40% es pasto disponible
            st.markdown(f"**🌾 Área Potencial Restauración:** `{area_potencial:,.1f} ha`")
            st.caption("Área estimada de 'Pastos' disponible para conversión a bosque.")
        else:
            st.write("Selecciona una zona para calcular área.")

    st.divider()

    # --- B. INTERFAZ DE USUARIO ---
    modo_calc = st.radio("Selecciona el tipo de análisis:", 
                         ["🔮 Proyección (Restauración Futura)", "📏 Inventario (Medición en Campo)"], 
                         horizontal=True)
    
    # ---------------------------------------------------------
    # MODO 1: PROYECCIÓN (Usa el Área Potencial detectada)
    # ---------------------------------------------------------
    if "Proyección" in modo_calc:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Parámetros")
            # El valor por defecto viene del sistema (land_cover), pero es editable
            area_ha = st.number_input("Área a restaurar (Ha):", 
                                      min_value=0.1, 
                                      value=float(area_potencial) if area_potencial > 0 else 1.0, 
                                      step=0.1,
                                      help="Sugerido basado en la cobertura de pastos actual.")
            
            anios_proj = st.slider("Horizonte (años):", 5, 50, 20)
            tipo_bosque = st.selectbox("Modelo:", ["Bosque Húmedo Tropical (Restauración)", "Bosque Seco Tropical"])
            
            if st.button("🚀 Proyectar Captura"):
                df_proj = carbon_calculator.calcular_proyeccion_captura(area_ha, anios_proj)
                st.session_state['df_carbon_proj'] = df_proj
        
        with c2:
            if 'df_carbon_proj' in st.session_state:
                df = st.session_state['df_carbon_proj']
                total = df['Proyecto_tCO2e_Acumulado'].iloc[-1]
                
                st.metric("Potencial de Captura Total", f"{total:,.0f} tCO2e")
                fig = px.area(df, x='Año', y='Proyecto_tCO2e_Acumulado', 
                              title="Acumulación de Carbono en el Tiempo",
                              color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # MODO 2: INVENTARIO (Usa la Zona de Vida detectada)
    # ---------------------------------------------------------
    else:
        st.subheader("📏 Calculadora de Stock (Inventario)")
        st.info("Sube tu Excel de campo (DAP, Altura). El sistema seleccionará la ecuación científica adecuada.")
        
        c_inv_1, c_inv_2 = st.columns([1, 2])
        
        with c_inv_1:
            # EL GRAN CAMBIO: El selectbox ya selecciona automáticamente la ZV detectada
            opciones_zv = ["bh-MB", "bh-PM", "bh-T", "bmh-M", "bmh-MB", "bmh-PM", "bp-PM"]
            
            idx_default = 0
            if zv_detectada in opciones_zv:
                idx_default = opciones_zv.index(zv_detectada)
                
            zona_vida = st.selectbox("Zona de Vida (Ecuación):", 
                                     opciones_zv, 
                                     index=idx_default,
                                     help="Automáticamente seleccionada según la ubicación del proyecto.")
            
            up_file = st.file_uploader("Cargar Excel/CSV", type=['csv', 'xlsx'])

        with c_inv_2:
            if up_file:
                if up_file.name.endswith('.csv'):
                    df_inv = pd.read_csv(up_file, sep=';' if ';' in up_file.getvalue().decode('latin1') else ',')
                else:
                    df_inv = pd.read_excel(up_file)
                
                if st.button("🧮 Calcular Stock Actual"):
                    df_res, msg = carbon_calculator.calcular_inventario_forestal(df_inv, zona_vida)
                    
                    if df_res is not None:
                        st.success(f"✅ Cálculo realizado usando coeficientes para **{zona_vida}**.")
                        st.dataframe(df_res.head())
                        
                        total_carb = df_res['CO2e_Total_tCO2e'].sum()
                        st.metric("Stock Total de Carbono", f"{total_carb:,.2f} tCO2e")
                        
                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Bajar Resultado", csv, "carbono_calculado.csv", "text/csv")
                    else:
                        st.error(msg)
