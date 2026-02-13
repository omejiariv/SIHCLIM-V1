# pages/03_🍃_Biodiversidad.py

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
import plotly.graph_objects as go
import plotly.express as px

# --- IMPORTACIÓN DE MÓDULOS DEL SISTEMA ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector, carbon_calculator
    from modules import life_zones as lz 
    from modules import land_cover as lc
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")
st.title("🍃 Biodiversidad y Servicios Ecosistémicos Integrados")

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

# --- MOTOR DE INTEGRACIÓN: ÁLGEBRA DE MAPAS ---
@st.cache_data(show_spinner=False)
def analizar_coberturas_por_zona_vida(_gdf_zona, zone_key):
    """
    Realiza la intersección raster entre Zonas de Vida y Coberturas.
    Retorna un DataFrame con hectáreas por combinación.
    """
    try:
        # 1. Rutas de Archivos Base
        dem_path = st.session_state.get('dem_path', config.Config.DEM_FILE_PATH)
        ppt_path = st.session_state.get('ppt_path', config.Config.PRECIP_RASTER_PATH)
        cov_path = st.session_state.get('cov_path', config.Config.LAND_COVER_RASTER_PATH)

        # 2. Generar Mapa de Zonas de Vida (Raster 1)
        # Usamos _gdf_zona (el objeto original)
        lz_arr, lz_profile, lz_names, _ = lz.generate_life_zone_map(
            dem_path, ppt_path, mask_geometry=_gdf_zona, downscale_factor=4
        )
        
        if lz_arr is None: return None

        # 3. Alinear Mapa de Coberturas (Raster 2) al Raster 1
        with rasterio.open(cov_path) as src_cov:
            cov_aligned = np.zeros_like(lz_arr, dtype=np.uint8)
            
            reproject(
                source=rasterio.band(src_cov, 1),
                destination=cov_aligned,
                src_transform=src_cov.transform,
                src_crs=src_cov.crs,
                dst_transform=lz_profile['transform'],
                dst_crs=lz_profile['crs'],
                resampling=Resampling.nearest
            )

        # 4. Cálculo de Áreas (Hectáreas)
        try:
            # Estimación simple basada en la resolución del transform
            res_x_deg = lz_profile['transform'][0]
            meters_per_deg = 111132.0 
            pixel_area_m2 = (res_x_deg * meters_per_deg) ** 2
            pixel_area_ha = pixel_area_m2 / 10000.0
        except:
            pixel_area_ha = 1.0

        # 5. Cruce (Crosstab)
        df_cross = pd.DataFrame({
            'ZV_ID': lz_arr.flatten(),
            'COV_ID': cov_aligned.flatten()
        })
        
        df_cross = df_cross[(df_cross['ZV_ID'] > 0) & (df_cross['COV_ID'] > 0)]
        
        resumen = df_cross.groupby(['ZV_ID', 'COV_ID']).size().reset_index(name='Pixeles')
        resumen['Hectareas'] = resumen['Pixeles'] * pixel_area_ha
        
        # 6. Enriquecer con Nombres
        resumen['Zona_Vida'] = resumen['ZV_ID'].map(lambda x: lz_names.get(x, f"ZV {x}"))
        resumen['Cobertura'] = resumen['COV_ID'].map(lambda x: lc.LAND_COVER_LEGEND.get(x, f"Clase {x}"))
        
        return resumen

    except Exception as e:
        # st.warning(f"Detalle técnico cruce: {e}") # Descomentar para debug
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
# TAB 3: CALCULADORA DE CARBONO (SISTÉMICA)
# ==============================================================================
with tab_carbon:
    st.header("🌳 Estimación de Servicios Ecosistémicos (Carbono)")
    
    if gdf_zona is None:
        st.warning("👈 Por favor selecciona una zona en el menú lateral para iniciar.")
        st.stop()

    # --- A. DIAGNÓSTICO TERRITORIAL ---
    st.subheader("🔍 Diagnóstico Territorial Integrado")
    
    with st.spinner("🔄 Cruzando mapas de Clima (Holdridge) y Cobertura (Land Cover)..."):
        # LLAMADA CORREGIDA: Pasamos _gdf_zona (el objeto) y nombre_seleccion (la clave de cache)
        df_diagnostico = analizar_coberturas_por_zona_vida(gdf_zona, nombre_seleccion)

    if df_diagnostico is not None and not df_diagnostico.empty:
        # IDs de Pastos/Degradados (Ajustar según land_cover.py)
        # 7: Pastos, 3: Zonas degradadas, 11: Áreas abiertas
        target_ids = [7, 3, 11] 
        df_potencial = df_diagnostico[df_diagnostico['COV_ID'].isin(target_ids)].copy()
        
        total_potencial = df_potencial['Hectareas'].sum()
        
        c_kpi1, c_kpi2 = st.columns(2)
        area_tot = gdf_zona.to_crs('+proj=cea').area.sum()/10000
        c_kpi1.metric("Área Total Seleccionada", f"{area_tot:,.1f} ha")
        c_kpi2.metric("Potencial Restauración", f"{total_potencial:,.1f} ha", 
                      help="Suma de Pastos y Áreas abiertas en todas las zonas de vida.")

        fig_bar = px.bar(df_potencial, x='Hectareas', y='Zona_Vida', color='Cobertura', 
                         orientation='h', title="Áreas Restaurables por Clima",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.warning("No se pudo calcular el detalle espacial (Faltan rasters en BD). Se usarán valores estimados.")
        df_potencial = pd.DataFrame()
        total_potencial = 0

    st.divider()

    # --- B. HERRAMIENTAS DE CÁLCULO ---
    modo_calc = st.radio("Selecciona el enfoque:", 
                         ["🔮 Proyección (Planificación)", "📏 Inventario (Línea Base)"], 
                         horizontal=True)

    # MODO 1: PROYECCIÓN
    if "Proyección" in modo_calc:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Configuración**")
            opcion_area = st.radio("Área a restaurar:", ["Manual", "Todo el Potencial"], index=1)
            
            if opcion_area == "Manual":
                area_ha = st.number_input("Área (Ha):", min_value=0.1, value=1.0)
            else:
                area_ha = st.number_input("Área (Ha):", min_value=0.1, 
                                          value=float(total_potencial) if total_potencial > 0 else 1.0,
                                          disabled=True)

            anios_proj = st.slider("Horizonte (años):", 5, 50, 20)
            tipo_bosque = st.selectbox("Modelo:", ["Bosque Húmedo Tropical (General)", "Bosque Seco Tropical"])
            
            if st.button("🚀 Calcular Escenario", type="primary"):
                df_proj = carbon_calculator.calcular_proyeccion_captura(area_ha, anios_proj)
                st.session_state['df_carbon_proj'] = df_proj

        with c2:
            if 'df_carbon_proj' in st.session_state:
                df = st.session_state['df_carbon_proj']
                total_c = df['Proyecto_tCO2e_Acumulado'].iloc[-1]
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Total CO2e", f"{total_c:,.0f} t")
                k2.metric("Promedio Anual", f"{(total_c/anios_proj):,.0f} t/año")
                k3.metric("Valor Potencial", f"${(total_c*5):,.0f} USD")

                fig = px.area(df, x='Año', y='Proyecto_tCO2e_Acumulado',
                              title=f"Curva de Acumulación ({area_ha:.1f} ha)",
                              color_discrete_sequence=['#27ae60'])
                st.plotly_chart(fig, use_container_width=True)

    # MODO 2: INVENTARIO
    else:
        c_inv_1, c_inv_2 = st.columns([1, 2])
        with c_inv_1:
            st.info("Sube Excel con columnas: DAP, Altura")
            up_file = st.file_uploader("Cargar Archivo", type=['csv', 'xlsx'])
            
            zv_sugerida = "bh-MB"
            if df_diagnostico is not None and not df_diagnostico.empty:
                top_zv = df_diagnostico.groupby('Zona_Vida')['Hectareas'].sum().idxmax()
                mapa_nombres = {"Bosque húmedo Premontano (bh-PM)": "bh-PM", "Bosque muy húmedo Montano Bajo (bmh-MB)": "bmh-MB"}
                zv_sugerida = mapa_nombres.get(top_zv, "bh-MB")

            opciones_zv = ["bh-MB", "bh-PM", "bh-T", "bmh-M", "bmh-MB", "bmh-PM", "bp-PM", "bs-T", "me-T"]
            idx_def = opciones_zv.index(zv_sugerida) if zv_sugerida in opciones_zv else 0
            
            zona_vida = st.selectbox("Ecuación (Zona de Vida):", opciones_zv, index=idx_def, help=f"Sugerido: {zv_sugerida}")

        with c_inv_2:
            if up_file and st.button("🧮 Procesar"):
                try:
                    if up_file.name.endswith('.csv'):
                        df_inv = pd.read_csv(up_file, sep=';' if ';' in up_file.getvalue().decode('latin1') else ',')
                    else:
                        df_inv = pd.read_excel(up_file)
                    
                    df_res, msg = carbon_calculator.calcular_inventario_forestal(df_inv, zona_vida)
                    
                    if df_res is not None:
                        st.success("✅ Procesado.")
                        tot_carb = df_res['CO2e_Total_tCO2e'].sum()
                        st.metric("Stock Total", f"{tot_carb:,.2f} tCO2e")
                        st.dataframe(df_res.head())
                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Descargar", csv, "inventario.csv", "text/csv")
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Error: {e}")
