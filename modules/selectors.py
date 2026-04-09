# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from shapely.geometry import box
from modules import db_manager
from modules.config import Config

# ====================================================================
# --- NUEVA FUNCIÓN: MENÚ DE NAVEGACIÓN EXPANDIBLE ---
# ====================================================================
def renderizar_menu_navegacion(pagina_actual):

    """
    Genera un menú expandible en el sidebar indicando la página actual.
    Reemplaza la navegación nativa de Streamlit para ahorrar espacio.
    """
    titulo_expander = f"📂 Navegación | Actual: {pagina_actual}"

    with st.sidebar.expander(titulo_expander, expanded=False):
        st.page_link("app.py", label="Inicio", icon="🏠")
        st.page_link("pages/01_🌦️_Clima_e_Hidrologia.py", label="Clima e Hidrología", icon="🌦️")
        st.page_link("pages/02_💧_Aguas_Subterraneas.py", label="Aguas Subterráneas", icon="💧")
        st.page_link("pages/03_🗺️_Isoyetas_HD.py", label="Isoyetas HD", icon="🗺️")
        st.page_link("pages/04_🍃_Biodiversidad.py", label="Biodiversidad", icon="🌱")
        st.page_link("pages/05_🏔️_Geomorfologia.py", label="Geomorfología", icon="⛰️")
        st.page_link("pages/06_🐄_Modelo_Pecuario.py", label="Modelo Pecuario", icon="🐄")
        st.page_link("pages/06_📈_Modelo_Demografico.py", label="Modelo Demográfico", icon="👥")
        st.page_link("pages/07_💧_Calidad_y_Vertimientos.py", label="Calidad y Vertimientos", icon="🧪")
        st.page_link("pages/08_🔗_Sistemas_Hidricos_Territoriales.py", label="Sistemas Hídricos", icon="🌊")
        st.page_link("pages/09_📊_Toma_de_Decisiones.py", label="Toma de Decisiones", icon="⚖️")
        st.page_link("pages/10_👑_Panel_Administracion.py", label="Panel Administración", icon="⚙️")
        st.page_link("pages/11_⚙️_Generador.py", label="Generador", icon="✨")
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="Ayuda y Docs", icon="📚")
        st.page_link("pages/13_🕵️_Detective.py", label="Detective", icon="🕵️")
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="Satélite en Vivo", icon="🛰️")
# ====================================================================

# 🔥 OPTIMIZACIÓN: Guardamos los mapas pesados en RAM para no colapsar la base de datos
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_cuencas():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_municipios():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")

def render_selector_espacial():
    """
    Selector espacial de alta velocidad y libre de bloqueos.
    """
    engine = db_manager.get_engine()
    
    # 🛠️ ENVOLVEMOS TODO EN UN EXPANDER PARA LIMPIAR EL SIDEBAR
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        
        # 1. MODO DE AGREGACIÓN
        modo = st.radio(
            "Nivel de Agregación:",
            ["Por Cuenca", "Por Municipio", "Por Región", "Departamento (Antioquia)"],
            index=0
        )
        
        gdf_zona = None
        nombre_zona = "Antioquia"
        altitud_ref = 1500
        
        try:
            # ==========================================
            # --- A. POR CUENCA (Enrutador Dinámico) ---
            # ==========================================
            if modo == "Por Cuenca":
                try:
                    gdf_cuencas = cargar_mapa_cuencas() # ⚡ Carga instantánea
                    
                    # 1. Selector de Ruta de Búsqueda
                    ruta_busqueda = st.selectbox(
                        "🛤️ Seleccione la Ruta de Búsqueda:",
                        ["💧 Jerarquía Hidrológica", "🗺️ División Regional", "🏢 Autoridad Ambiental (CAR)"],
                        index=0
                    )
                    
                    gdf_filtrado_base = None # Variable para guardar el resultado del embudo
                    
                    # --- RUTA 1: HIDROLÓGICA ---
                    if ruta_busqueda == "💧 Jerarquía Hidrológica":
                        if 'nomah' in gdf_cuencas.columns:
                            ah_disp = sorted(gdf_cuencas['nomah'].dropna().unique())
                            ah_sel = st.selectbox("🌊 1. Área Hidrográfica (AH):", ["-- Seleccione --"] + ah_disp)
                            
                            if ah_sel != "-- Seleccione --":
                                gdf_ah = gdf_cuencas[gdf_cuencas['nomah'] == ah_sel]
                                zh_disp = sorted(gdf_ah['nomzh'].dropna().unique())
                                zh_sel = st.selectbox("💧 2. Zona Hidrológica (ZH):", ["-- Seleccione --"] + zh_disp)
                                
                                if zh_sel != "-- Seleccione --":
                                    gdf_zh = gdf_ah[gdf_ah['nomzh'] == zh_sel]
                                    szh_disp = sorted(gdf_zh['nom_szh'].dropna().unique())
                                    szh_sel = st.selectbox("💦 3. Subzona Hidrográfica (SZH):", ["-- Seleccione --"] + szh_disp)
                                    
                                    if szh_sel != "-- Seleccione --":
                                        gdf_filtrado_base = gdf_zh[gdf_zh['nom_szh'] == szh_sel]
                        else:
                            st.warning("Faltan las columnas de Área Hidrográfica (nomah) en la base de datos.")

                    # --- RUTA 2: REGIONAL ---
                    elif ruta_busqueda == "🗺️ División Regional":
                        if 'depto_regi' in gdf_cuencas.columns:
                            reg_disp = sorted(gdf_cuencas['depto_regi'].dropna().unique())
                            reg_sel = st.selectbox("📍 1. Región (Depto-Región):", ["-- Seleccione --"] + reg_disp)
                            
                            if reg_sel != "-- Seleccione --":
                                gdf_reg = gdf_cuencas[gdf_cuencas['depto_regi'] == reg_sel]
                                zona_disp = sorted(gdf_reg['zona'].dropna().unique())
                                zona_sel = st.selectbox("🌍 2. Subregión (Zona):", ["-- Seleccione --"] + zona_disp)
                                
                                if zona_sel != "-- Seleccione --":
                                    gdf_filtrado_base = gdf_reg[gdf_reg['zona'] == zona_sel]
                        else:
                            st.warning("Faltan las columnas regionales (depto_regi) en la base de datos.")

                    # --- RUTA 3: AUTORIDAD AMBIENTAL ---
                    elif ruta_busqueda == "🏢 Autoridad Ambiental (CAR)":
                        if 'corpoamb' in gdf_cuencas.columns:
                            car_disp = sorted(gdf_cuencas['corpoamb'].dropna().unique())
                            car_sel = st.selectbox("🏛️ 1. Autoridad Ambiental:", ["-- Seleccione --"] + car_disp)
                            
                            if car_sel != "-- Seleccione --":
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas['corpoamb'] == car_sel]
                        else:
                            st.warning("Falta la columna 'corpoamb' en la base de datos.")

                    # ==========================================
                    # --- NODO COMÚN: SELECCIÓN DEL NIVEL NSS ---
                    # ==========================================
                    if gdf_filtrado_base is not None and not gdf_filtrado_base.empty:
                        st.markdown("---")
                        nivel_nss = st.radio(
                            "🔎 Resolución de la Cuenca:",
                            ["NSS1 (Macro)", "NSS2 (Intermedia)", "NSS3 (Microcuenca)"],
                            horizontal=True
                        )
                        
                        mapa_cols_nss = {
                            "NSS1 (Macro)": "nom_nss1",
                            "NSS2 (Intermedia)": "nom_nss2",
                            "NSS3 (Microcuenca)": "nom_nss3"
                        }
                        col_objetivo = mapa_cols_nss[nivel_nss]
                        
                        if col_objetivo in gdf_filtrado_base.columns:
                            territorios_disp = sorted(gdf_filtrado_base[col_objetivo].dropna().unique())
                            sel_final = st.selectbox(f"🎯 Seleccione el Territorio ({nivel_nss}):", territorios_disp)
                            
                            if sel_final:
                                nombre_zona = sel_final
                                gdf_zona = gdf_filtrado_base[gdf_filtrado_base[col_objetivo] == sel_final]
                                
                                # Consolidar geometrías fragmentadas en un solo polígono maestro
                                if len(gdf_zona) > 1:
                                    gdf_zona = gpd.GeoDataFrame({'geometry': [gdf_zona.unary_union]}, crs=gdf_zona.crs)
                        else:
                            st.warning(f"La columna {col_objetivo} no existe en la base de datos.")

                except Exception as e:
                    st.warning(f"Error cargando el embudo de cuencas: {e}")
                    
            # ==========================================
            # --- B. POR REGIÓN (FIX: Disolución de Municipios) ---
            # ==========================================
            elif modo == "Por Región":
                try:
                    gdf_mun = cargar_mapa_municipios()
                    # Buscamos la columna de subregión (usualmente 'subregion', 'region' o 'depto_regi')
                    cols_mun_lower = [c.lower() for c in gdf_mun.columns]
                    col_reg_name = next((c for c in gdf_mun.columns if c.lower() in ['subregion', 'region', 'depto_regi']), None)
                    
                    if col_reg_name:
                        lista_reg = sorted(gdf_mun[col_reg_name].dropna().astype(str).unique().tolist())
                        sel = st.selectbox("📍 Seleccione Región:", lista_reg)
                        
                        if sel:
                            nombre_zona = f"Región {sel}"
                            gdf_reg_filt = gdf_mun[gdf_mun[col_reg_name] == sel]
                            
                            # Fusión Topológica de todos los municipios de la región
                            region_geom = gdf_reg_filt.unary_union
                            # Asignamos la columna 'nombre' para el Tooltip
                            gdf_zona = gpd.GeoDataFrame({'nombre': [nombre_zona]}, geometry=[region_geom], crs=gdf_mun.crs)
                    else:
                        st.warning("No se encontró la columna de región en la capa de municipios.")
                except Exception as e:
                    st.warning(f"Error cargando regiones: {e}")        

            # ==========================================
            # --- C. POR MUNICIPIO ---
            # ==========================================
            elif modo == "Por Municipio":
                try:
                    gdf_mun = cargar_mapa_municipios() 
                    
                    cols_texto = [c for c in gdf_mun.columns if c not in ['geometry', 'gid']]
                    default_idx = 0
                    col_name = next((c for c in gdf_mun.columns if c.lower() in ['mpio_cnmbr', 'nombre_mpio', 'municipio']), cols_texto[0] if cols_texto else None)
                    
                    if col_name:
                        lista = sorted(gdf_mun[col_name].dropna().astype(str).unique().tolist())
                        sel = st.selectbox("🏢 Seleccione Municipio:", lista)
                        if sel:
                            nombre_zona = sel
                            gdf_zona = gdf_mun[gdf_mun[col_name] == sel].copy()
                            # Asignamos la columna 'nombre' estandarizada para el Tooltip
                            gdf_zona['nombre'] = sel
                except Exception as e:
                    st.warning(f"Error en tabla municipios: {e}")

            # ==========================================
            # --- D. DEPARTAMENTO (FIX: POLÍGONO REAL) ---
            # ==========================================
            else:
                nombre_zona = "Antioquia"
                try:
                    # 🔥 FIX ESTRUCTURAL: Fusión Topológica (Dissolve)
                    # Traemos todos los municipios y los fundimos en un solo mega-polígono.
                    # Esto garantiza el contorno perfecto del departamento para áreas y balances.
                    gdf_mun = cargar_mapa_municipios()
                    
                    if gdf_mun.crs is None or gdf_mun.crs.to_string() != "EPSG:4326":
                        gdf_mun = gdf_mun.to_crs("EPSG:4326")
                        
                    antioquia_geom = gdf_mun.unary_union
                    gdf_zona = gpd.GeoDataFrame({'nombre': ['Antioquia']}, geometry=[antioquia_geom], crs="EPSG:4326")
                except Exception as e:
                    # Fallback de emergencia
                    gdf_zona = gpd.GeoDataFrame(
                        {'nombre': ['Antioquia']}, 
                        geometry=[box(-77.5, 5.0, -73.5, 9.0)], 
                        crs="EPSG:4326"
                    )
                    
            # =========================================================================
            # --- 2. FILTRAR ESTACIONES (Algoritmo de Alta Velocidad) ---
            # =========================================================================
            ids_estaciones = []
            if gdf_zona is not None and not gdf_zona.empty:
                
                if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                     gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
                # 🔥 EL CAMBIO MAESTRO: st.session_state se adueña del valor (Single Source of Truth)
                buff_km = st.slider("Radio Buffer (Área de Influencia en km):", min_value=0.0, max_value=50.0, value=15.0, step=1.0, key="buffer_global_km")
                buff_deg = buff_km / 111.0 
                
                minx, miny, maxx, maxy = gdf_zona.total_bounds
                
                q_est = text(f"""
                    SELECT id_estacion, nombre, latitud, longitud, altitud 
                    FROM estaciones 
                    WHERE longitud BETWEEN {minx - buff_deg} AND {maxx + buff_deg} 
                    AND latitud BETWEEN {miny - buff_deg} AND {maxy + buff_deg}
                """)
                
                df_est = pd.read_sql(q_est, engine)
                
                if not df_est.empty:
                    gdf_ptos = gpd.GeoDataFrame(
                        df_est, 
                        geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), 
                        crs="EPSG:4326"
                    )
                    
                    zona_buffered = gdf_zona.copy()
                    if buff_deg > 0:
                        zona_buffered['geometry'] = zona_buffered.geometry.buffer(buff_deg)
                    
                    # Intersección de alta velocidad
                    est_in = gpd.sjoin(gdf_ptos, zona_buffered, how="inner", predicate="intersects")
                    est_in = est_in.drop_duplicates(subset=['id_estacion'])
                    
                    if not est_in.empty:
                        ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                        altitud_ref = est_in['altitud'].mean()
                        st.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                    else:
                        st.warning("0 estaciones en el área exacta.")
                else:
                    st.warning("0 estaciones en el cuadrante.")

        except Exception as e:
            st.error(f"Error crítico en selector: {e}")
            
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
