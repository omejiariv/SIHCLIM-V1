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
                    
                    # --- RUTA 1: HIDROLÓGICA (Filtro Progresivo) ---
                    if ruta_busqueda == "💧 Jerarquía Hidrológica":
                        if 'nomah' in gdf_cuencas.columns:
                            ah_disp = sorted(gdf_cuencas['nomah'].dropna().unique())
                            ah_sel = st.selectbox("🌊 1. Área Hidrográfica (AH):", ["-- Seleccione --"] + ah_disp)
                            
                            if ah_sel != "-- Seleccione --":
                                # Asignamos la base al Nivel 1
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas['nomah'] == ah_sel]
                                
                                zh_disp = sorted(gdf_filtrado_base['nomzh'].dropna().unique())
                                zh_sel = st.selectbox("💧 2. Zona Hidrológica (ZH):", ["-- TODAS --"] + zh_disp)
                                
                                if zh_sel != "-- TODAS --":
                                    # Filtramos al Nivel 2
                                    gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base['nomzh'] == zh_sel]
                                    
                                    szh_disp = sorted(gdf_filtrado_base['nom_szh'].dropna().unique())
                                    szh_sel = st.selectbox("💦 3. Subzona Hidrográfica (SZH):", ["-- TODAS --"] + szh_disp)
                                    
                                    if szh_sel != "-- TODAS --":
                                        # Filtramos al Nivel 3
                                        gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base['nom_szh'] == szh_sel]
                        else:
                            st.warning("Faltan las columnas de Área Hidrográfica (nomah) en la base de datos.")

                    # --- RUTA 2: REGIONAL ---
                    elif ruta_busqueda == "🗺️ División Regional":
                        # Buscamos nombres de columna probables dinámicamente
                        col_reg = next((c for c in gdf_cuencas.columns if c.lower() in ['depto_regi', 'region', 'macroregion', 'subregion']), None)
                        
                        if col_reg:
                            reg_disp = sorted(gdf_cuencas[col_reg].dropna().unique())
                            reg_sel = st.selectbox("📍 1. Región:", ["-- Seleccione --"] + reg_disp)
                            
                            if reg_sel != "-- Seleccione --":
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas[col_reg] == reg_sel]
                                
                                col_zona = next((c for c in gdf_cuencas.columns if c.lower() in ['zona', 'subzona']), None)
                                if col_zona:
                                    zona_disp = sorted(gdf_filtrado_base[col_zona].dropna().unique())
                                    zona_sel = st.selectbox("🌍 2. Subregión:", ["-- TODAS --"] + zona_disp)
                                    if zona_sel != "-- TODAS --":
                                        gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base[col_zona] == zona_sel]
                        else:
                            st.warning("⚠️ Las Cuencas son fronteras naturales. El mapa de cuencas en la BD no tiene clasificaciones políticas (Regiones).")

                    # --- RUTA 3: AUTORIDAD AMBIENTAL ---
                    elif ruta_busqueda == "🏢 Autoridad Ambiental (CAR)":
                        col_car = next((c for c in gdf_cuencas.columns if c.lower() in ['corpoamb', 'car', 'autoridad']), None)
                        
                        if col_car:
                            car_disp = sorted(gdf_cuencas[col_car].dropna().unique())
                            car_sel = st.selectbox("🏛️ 1. Autoridad Ambiental:", ["-- Seleccione --"] + car_disp)
                            
                            if car_sel != "-- Seleccione --":
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas[col_car] == car_sel]
                                st.info("💡 **Nota:** La cuenca se recorta a los límites de la jurisdicción elegida. (Ej: Río Aburrá en Corantioquia excluye las áreas urbanas del AMVA).")
                        else:
                            st.warning("Falta la columna de Autoridad Ambiental ('corpoamb') en la base de datos.")

                    # ==========================================
                    # --- NODO COMÚN: SELECCIÓN DEL NIVEL NSS ---
                    # ==========================================
                    if gdf_filtrado_base is not None and not gdf_filtrado_base.empty:
                        st.markdown("---")
                        nivel_nss = st.radio(
                            "🔎 Resolución de visualización en el Mapa:",
                            ["NSS1 (Macro)", "NSS2 (Intermedia)", "NSS3 (Micro)"],
                            horizontal=True
                        )
                        
                        mapa_cols_nss = {
                            "NSS1 (Macro)": "nom_nss1",
                            "NSS2 (Intermedia)": "nom_nss2",
                            "NSS3 (Micro)": "nom_nss3"
                        }
                        col_objetivo = mapa_cols_nss[nivel_nss]
                        
                        if col_objetivo in gdf_filtrado_base.columns:
                            territorios_disp = sorted(gdf_filtrado_base[col_objetivo].dropna().unique())
                            # Añadimos la opción de graficar TODO el bloque seleccionado
                            sel_final = st.selectbox(f"🎯 Seleccione el Territorio ({nivel_nss}):", ["-- GRAFICAR TODO EL BLOQUE --"] + territorios_disp)
                            
                            if sel_final != "-- GRAFICAR TODO EL BLOQUE --":
                                nombre_zona = sel_final
                                gdf_zona = gdf_filtrado_base[gdf_filtrado_base[col_objetivo] == sel_final]
                            else:
                                # Si elige graficar todo, tomamos el nombre del filtro anterior
                                if ruta_busqueda == "💧 Jerarquía Hidrológica":
                                    nombre_zona = szh_sel if szh_sel != "-- TODAS --" else (zh_sel if zh_sel != "-- TODAS --" else ah_sel)
                                elif ruta_busqueda == "🏢 Autoridad Ambiental (CAR)":
                                    nombre_zona = car_sel
                                else:
                                    nombre_zona = "Bloque Regional"
                                gdf_zona = gdf_filtrado_base
                                
                            # Consolidar geometrías fragmentadas en un solo polígono maestro
                            if len(gdf_zona) > 1:
                                gdf_zona = gpd.GeoDataFrame({'geometry': [gdf_zona.unary_union]}, crs=gdf_zona.crs)
                        else:
                            st.warning(f"La columna {col_objetivo} no existe en la base de datos.")

                except Exception as e:
                    st.warning(f"Error cargando el embudo de cuencas: {e}")
                    
            # ==========================================
            # --- B. POR REGIÓN (FIX: Supabase + Doble Filtro) ---
            # ==========================================
            elif modo == "Por Región":
                try:
                    # 1. Leer el archivo maestro desde Supabase (URL directa .xlsx)
                    url_maestro = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
                    
                    try:
                        # pd y gpd ya están importados al inicio del archivo, no los re-importamos aquí.
                        df_maestro = pd.read_excel(url_maestro)
                        
                        # Estandarizar nombres de columnas a minúsculas
                        df_maestro.columns = [c.lower() for c in df_maestro.columns]
                        
                        if 'region' in df_maestro.columns and 'dp_mp' in df_maestro.columns and 'municipio' in df_maestro.columns:
                            # Sacamos las regiones únicas
                            lista_reg = sorted([str(r).title() for r in df_maestro['region'].dropna().unique() if str(r).strip() != ""])
                            sel_reg = st.selectbox("📍 Seleccione Región:", ["-- Seleccione --"] + lista_reg)
                            
                            if sel_reg != "-- Seleccione --":
                                nombre_zona = f"Región {sel_reg}"
                                
                                # Extraer los códigos DANE y los NOMBRES de municipio del Excel
                                df_region_filt = df_maestro[df_maestro['region'].str.lower() == sel_reg.lower()]
                                mpios_codigos = df_region_filt['dp_mp'].astype(str).str.zfill(5).tolist()
                                
                                # Normalizamos nombres para hacer un cruce a prueba de tildes y mayúsculas
                                import unicodedata
                                def limpiar_texto(t):
                                    if pd.isna(t): return ""
                                    t = str(t).upper().strip()
                                    return ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
                                
                                mpios_nombres = [limpiar_texto(m) for m in df_region_filt['municipio'].tolist()]
                                
                                # 2. Cargar geometrías del mapa
                                gdf_mun = cargar_mapa_municipios()
                                
                                # Buscar la columna de código y de nombre en el mapa (PostgreSQL)
                                col_cod = next((c for c in gdf_mun.columns if c.lower() in ['cod_mpio', 'mpio_ccdgo', 'dp_mp', 'id']), None)
                                col_nom = next((c for c in gdf_mun.columns if c.lower() in ['nombre_municipio', 'mpio_nombr', 'mpio_cnmbr', 'nombre_mpio', 'municipio']), None)
                                
                                # Filtrar: O coincide el código, O coincide el nombre exacto
                                if col_cod or col_nom:
                                    mask_cod = pd.Series([False] * len(gdf_mun), index=gdf_mun.index)
                                    mask_nom = pd.Series([False] * len(gdf_mun), index=gdf_mun.index)
                                    
                                    if col_cod:
                                        # Si el BD guardó '145' en vez de '05145', aquí lo corregimos al vuelo
                                        cod_bd_limpio = gdf_mun[col_cod].astype(str).str.zfill(5)
                                        cod_bd_sin_depto = gdf_mun[col_cod].astype(str).str.zfill(3) 
                                        mask_cod = cod_bd_limpio.isin(mpios_codigos) | ("05" + cod_bd_sin_depto).isin(mpios_codigos)
                                        
                                    if col_nom:
                                        nom_bd_limpio = gdf_mun[col_nom].apply(limpiar_texto)
                                        mask_nom = nom_bd_limpio.isin(mpios_nombres)
                                        
                                    gdf_reg_filt = gdf_mun[mask_cod | mask_nom]
                                    
                                    if not gdf_reg_filt.empty:
                                        # ¡Fusión Topológica! Unir todos los municipios en la megaregión
                                        region_geom = gdf_reg_filt.unary_union
                                        gdf_zona = gpd.GeoDataFrame({'nombre': [nombre_zona]}, geometry=[region_geom], crs=gdf_mun.crs)
                                    else:
                                        st.warning("⚠️ No se encontraron las geometrías en la BD para los municipios de esta región.")
                                else:
                                    st.error("⚠️ La capa de municipios no tiene columna de código ni de nombre reconocible.")
                        else:
                            st.warning("El archivo maestro no tiene las columnas requeridas ('region', 'dp_mp', 'municipio').")
                    except Exception as e_read:
                        st.error(f"⚠️ Error leyendo el archivo desde Supabase: {e_read}")

                except Exception as e:
                    st.warning(f"Error procesando la región: {e}")

            # ==========================================
            # --- C. POR MUNICIPIO (FIX: nombre_municipio) ---
            # ==========================================
            elif modo == "Por Municipio":
                try:
                    gdf_mun = cargar_mapa_municipios() 
                    
                    # Identificar dinámicamente las columnas correctas (Añadido 'nombre_municipio')
                    col_mpio = next((c for c in gdf_mun.columns if c.lower() in ['nombre_municipio', 'mpio_nombr', 'mpio_cnmbr', 'nombre_mpio', 'municipio']), None)
                    col_depto = next((c for c in gdf_mun.columns if c.lower() in ['dpto_cnmbr', 'dpto_nombr', 'nombre_dpto', 'departamento']), None)
                    
                    if col_mpio:
                        # Estandarizar capitalización (ej. "Medellín")
                        gdf_mun[col_mpio] = gdf_mun[col_mpio].astype(str).str.title()
                        
                        # Concatenar el Departamento si existe (ej. "Titiribí - Antioquia")
                        if col_depto:
                            gdf_mun[col_depto] = gdf_mun[col_depto].astype(str).str.title()
                            gdf_mun['display_name'] = gdf_mun[col_mpio] + " - " + gdf_mun[col_depto]
                        else:
                            gdf_mun['display_name'] = gdf_mun[col_mpio]
                            
                        lista = sorted(gdf_mun['display_name'].dropna().unique().tolist())
                        sel = st.selectbox("🏢 Seleccione Municipio:", lista)
                        
                        if sel:
                            # Rescatamos solo el nombre del municipio limpio para los balances y gráficas
                            nombre_zona = sel.split(" - ")[0] 
                            gdf_zona = gdf_mun[gdf_mun['display_name'] == sel].copy()
                            gdf_zona['nombre'] = nombre_zona # Esta es la columna mágica para los Tooltips
                    else:
                        st.warning("⚠️ No se encontró la columna de municipio (ej. nombre_municipio) en la base de datos.")
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
