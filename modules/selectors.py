# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from shapely.geometry import box
import json
import time
import unicodedata

from modules import db_manager
from modules.config import Config

# ====================================================================
# --- REPARADOR DE TILDES (MOJIBAKE) ---
# ====================================================================
def decodificar_tildes(texto):
    """Corrige errores de codificación en BD como 'Abriaquã' -> 'Abriaquí'"""
    if not isinstance(texto, str): return texto
    try:
        if 'Ã' in texto or 'ã' in texto or '\x8d' in texto:
            return texto.encode('latin1').decode('utf-8')
    except: pass
    return texto

# ====================================================================
# 📂 NAVEGACIÓN GLOBAL
# ====================================================================
def renderizar_menu_navegacion(pagina_actual):
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
# ☁️ CONEXIÓN A SUPABASE (Para guardar los JSON)
# ====================================================================
@st.cache_resource
def get_supabase_client():
    try:
        from supabase import create_client
        
        # 1. Buscamos primero si están sueltas (sin corchetes)
        url_sb = st.secrets.get("SUPABASE_URL")
        key_sb = st.secrets.get("SUPABASE_KEY")
        
        # 2. Si no están sueltas, buscamos dentro de [supabase] con mayúsculas
        if not url_sb:
            url_sb = st.secrets.get("supabase", {}).get("SUPABASE_URL")
        if not key_sb:
            key_sb = st.secrets.get("supabase", {}).get("SUPABASE_KEY")
            
        # 3. Por si acaso, buscamos dentro de [supabase] con minúsculas (url / key)
        if not url_sb:
            url_sb = st.secrets.get("supabase", {}).get("url")
        if not key_sb:
            key_sb = st.secrets.get("supabase", {}).get("key")

        # Intentamos conectar si encontramos ambas
        if url_sb and key_sb:
            return create_client(url_sb, key_sb)
        else:
            return "NO_SECRETS"
            
    except ImportError:
        return "NO_LIBRARY"
    except Exception as e:
        return str(e)

# ====================================================================
# 💾 GESTOR DE SNAPSHOTS (GUARDAR, CARGAR Y ELIMINAR ESCENARIOS)
# ====================================================================
def renderizar_gestor_escenarios(nombre_zona_actual):
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Gestor de Escenarios (Snapshots)", expanded=False):
        supabase = get_supabase_client()
        
        if supabase == "NO_SECRETS" or supabase == "NO_LIBRARY" or isinstance(supabase, str) or not supabase:
            st.error("Error de conexión a Supabase. Revisa tus credenciales.")
            return

        tab_guardar, tab_cargar = st.tabs(["Guardar", "Cargar / Eliminar"])
        
        # --- TAB GUARDAR ---
        with tab_guardar:
            nombre_escenario = st.text_input("Nombre del Proyecto/Escenario:", placeholder="Ej: Río Buey - Plan 2030")
            if st.button("💾 Guardar Sesión", use_container_width=True):
                if nombre_escenario:
                    with st.spinner("Empaquetando sesión en JSON..."):
                        estado_limpio = {k: v for k, v in st.session_state.items() if isinstance(v, (int, float, str, bool, list, dict)) and not k.startswith('FormSubmitter')}
                        try:
                            supabase.table("escenarios_guardados").insert({
                                "nombre_escenario": nombre_escenario,
                                "territorio": nombre_zona_actual,
                                "estado_json": estado_limpio
                            }).execute()
                            st.success("✅ ¡Escenario guardado en la Nube!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error guardando en tabla: {e}")
                else:
                    st.warning("⚠️ Debes darle un nombre al escenario.")

        # --- TAB CARGAR / ELIMINAR ---
        with tab_cargar:
            try:
                res = supabase.table("escenarios_guardados").select("id, nombre_escenario, territorio, fecha_creacion").order("fecha_creacion", desc=True).execute()
                escenarios = res.data
                if escenarios:
                    opciones = {f"{e['nombre_escenario']} ({e['territorio']})": e['id'] for e in escenarios}
                    seleccion = st.selectbox("Selecciona un proyecto:", list(opciones.keys()))
                    
                    # 🔥 NUEVO: Dos botones paralelos para Cargar o Borrar
                    col_c, col_d = st.columns(2)
                    
                    with col_c:
                        if st.button("📂 Cargar", type="primary", use_container_width=True):
                            id_sel = opciones[seleccion]
                            res_json = supabase.table("escenarios_guardados").select("estado_json").eq("id", id_sel).execute()
                            estado_recuperado = res_json.data[0]['estado_json']
                            with st.spinner("Inyectando variables..."):
                                for k, v in estado_recuperado.items():
                                    st.session_state[k] = v
                                st.success("✅ Interfaz restaurada.")
                                time.sleep(1)
                                st.rerun()
                                
                    with col_d:
                        if st.button("🗑️ Borrar", type="secondary", use_container_width=True):
                            id_del = opciones[seleccion]
                            with st.spinner("Eliminando de la nube..."):
                                supabase.table("escenarios_guardados").delete().eq("id", id_del).execute()
                                st.warning("🗑️ Proyecto eliminado.")
                                time.sleep(1)
                                st.rerun()
                else:
                    st.info("No hay escenarios guardados aún.")
            except Exception as e:
                st.error(f"Error consultando base: {e}")

# ====================================================================
# 🌍 SELECTOR ESPACIAL MAESTRO (RESTAURO 100% ORIGINAL)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_cuencas():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_municipios():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")

def render_selector_espacial():
    # Inicialización blindada (Código Original)
    ids_estaciones = []
    nombre_zona = "Antioquia"
    altitud_ref = 1500.0
    gdf_zona = None
    
    try:
        engine = db_manager.get_engine()
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona 
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento (Antioquia)"], index=0)
        
        try:
            # --- A. POR CUENCA ---
            if modo == "Por Cuenca":
                try:
                    gdf_cuencas = cargar_mapa_cuencas()
                    ruta_busqueda = st.selectbox("🛤️ Seleccione la Ruta de Búsqueda:", ["💧 Jerarquía Hidrológica", "🗺️ División Regional", "🏢 Autoridad Ambiental (CAR)"], index=0)
                    gdf_filtrado_base = None
                    
                    ah_sel = zh_sel = szh_sel = "-- TODAS --"
                    
                    if ruta_busqueda == "💧 Jerarquía Hidrológica":
                        if 'nomah' in gdf_cuencas.columns:
                            ah_disp = sorted(gdf_cuencas['nomah'].dropna().unique())
                            ah_sel = st.selectbox("🌊 1. Área Hidrográfica (AH):", ["-- Seleccione --"] + ah_disp)
                            if ah_sel != "-- Seleccione --":
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas['nomah'] == ah_sel]
                                zh_disp = sorted(gdf_filtrado_base['nomzh'].dropna().unique())
                                zh_sel = st.selectbox("💧 2. Zona Hidrológica (ZH):", ["-- TODAS --"] + zh_disp)
                                if zh_sel != "-- TODAS --":
                                    gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base['nomzh'] == zh_sel]
                                    szh_disp = sorted(gdf_filtrado_base['nom_szh'].dropna().unique())
                                    szh_sel = st.selectbox("💦 3. Subzona Hidrográfica (SZH):", ["-- TODAS --"] + szh_disp)
                                    if szh_sel != "-- TODAS --":
                                        gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base['nom_szh'] == szh_sel]

                    elif ruta_busqueda == "🗺️ División Regional":
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
                                    if zona_sel != "-- TODAS --": gdf_filtrado_base = gdf_filtrado_base[gdf_filtrado_base[col_zona] == zona_sel]
                            else: st.warning("⚠️ El mapa de cuencas en la BD no tiene clasificaciones políticas (Regiones).")

                    elif ruta_busqueda == "🏢 Autoridad Ambiental (CAR)":
                        col_car = next((c for c in gdf_cuencas.columns if c.lower() in ['corpoamb', 'car', 'autoridad']), None)
                        if col_car:
                            car_disp = sorted(gdf_cuencas[col_car].dropna().unique())
                            car_sel = st.selectbox("🏛️ 1. Autoridad Ambiental:", ["-- Seleccione --"] + car_disp)
                            if car_sel != "-- Seleccione --":
                                gdf_filtrado_base = gdf_cuencas[gdf_cuencas[col_car] == car_sel]
                            else: st.warning("Falta la columna de Autoridad Ambiental ('corpoamb') en la base de datos.")

                    if gdf_filtrado_base is not None and not gdf_filtrado_base.empty:
                        st.markdown("---")
                        nivel_nss = st.radio("🔎 Resolución de visualización en el Mapa:", ["NSS1 (Macro)", "NSS2 (Intermedia)", "NSS3 (Micro)"], horizontal=True)
                        mapa_cols_nss = {"NSS1 (Macro)": "nom_nss1", "NSS2 (Intermedia)": "nom_nss2", "NSS3 (Micro)": "nom_nss3"}
                        col_objetivo = mapa_cols_nss[nivel_nss]
                        
                        if col_objetivo in gdf_filtrado_base.columns:
                            territorios_disp = sorted(gdf_filtrado_base[col_objetivo].dropna().unique())
                            sel_final = st.selectbox(f"🎯 Seleccione el Territorio ({nivel_nss}):", ["-- GRAFICAR TODO EL BLOQUE --"] + territorios_disp)
                            
                            if sel_final != "-- GRAFICAR TODO EL BLOQUE --":
                                nombre_zona = sel_final
                                gdf_zona = gdf_filtrado_base[gdf_filtrado_base[col_objetivo] == sel_final]
                            else:
                                if ruta_busqueda == "💧 Jerarquía Hidrológica": nombre_zona = szh_sel if szh_sel != "-- TODAS --" else (zh_sel if zh_sel != "-- TODAS --" else ah_sel)
                                elif ruta_busqueda == "🏢 Autoridad Ambiental (CAR)": nombre_zona = car_sel
                                else: nombre_zona = "Bloque Regional"
                                gdf_zona = gdf_filtrado_base
                        else: st.warning(f"La columna {col_objetivo} no existe en la base de datos.")
                except Exception as e: st.warning(f"Error cargando cuencas: {e}")

            # --- B. POR REGIÓN ---
            elif modo == "Por Región":
                try:
                    url_maestro = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
                    df_maestro = pd.read_excel(url_maestro)
                    df_maestro.columns = [c.lower() for c in df_maestro.columns]
                    
                    if 'region' in df_maestro.columns and 'dp_mp' in df_maestro.columns:
                        lista_reg = sorted([str(r).title() for r in df_maestro['region'].dropna().unique() if str(r).strip() != ""])
                        sel_reg = st.selectbox("📍 Seleccione Región:", ["-- Seleccione --"] + lista_reg)
                        
                        if sel_reg != "-- Seleccione --":
                            nombre_zona = f"Región {sel_reg}"
                            df_region_filt = df_maestro[df_maestro['region'].str.lower() == sel_reg.lower()]
                            mpios_codigos = df_region_filt['dp_mp'].astype(str).str.zfill(5).tolist()
                            
                            def limpiar_t(t):
                                if pd.isna(t): return ""
                                return ''.join(c for c in unicodedata.normalize('NFD', str(t).upper().strip()) if unicodedata.category(c) != 'Mn')
                            
                            mpios_nombres = [limpiar_t(m) for m in df_region_filt['municipio'].tolist()]
                            gdf_mun = cargar_mapa_municipios()
                            
                            col_cod = next((c for c in gdf_mun.columns if c.lower() in ['cod_mpio', 'mpio_ccdgo', 'dp_mp', 'id']), None)
                            col_nom = next((c for c in gdf_mun.columns if c.lower() in ['nombre_municipio', 'mpio_nombr', 'mpio_cnmbr', 'nombre_mpio', 'municipio']), None)
                            
                            if col_cod or col_nom:
                                mask_cod = pd.Series([False] * len(gdf_mun), index=gdf_mun.index)
                                mask_nom = pd.Series([False] * len(gdf_mun), index=gdf_mun.index)
                                if col_cod:
                                    cod_limpio = gdf_mun[col_cod].astype(str).str.zfill(5)
                                    cod_sin_depto = gdf_mun[col_cod].astype(str).str.zfill(3) 
                                    mask_cod = cod_limpio.isin(mpios_codigos) | ("05" + cod_sin_depto).isin(mpios_codigos)
                                if col_nom:
                                    nom_limpio = gdf_mun[col_nom].apply(limpiar_t)
                                    mask_nom = nom_limpio.isin(mpios_nombres)
                                    
                                gdf_reg_filt = gdf_mun[mask_cod | mask_nom]
                                if not gdf_reg_filt.empty:
                                    gdf_zona = gpd.GeoDataFrame({'nombre': [nombre_zona]}, geometry=[gdf_reg_filt.unary_union], crs=gdf_mun.crs)
                    else: st.warning("El archivo maestro no tiene las columnas requeridas.")
                except Exception as e: st.warning(f"Error procesando la región: {e}")

            # --- C. POR MUNICIPIO ---
            elif modo == "Por Municipio":
                gdf_mun = cargar_mapa_municipios() 
                col_mpio = next((c for c in gdf_mun.columns if c.lower() in ['nombre_municipio', 'mpio_nombr', 'mpio_cnmbr', 'nombre_mpio', 'municipio']), None)
                col_depto = next((c for c in gdf_mun.columns if c.lower() in ['dpto_cnmbr', 'dpto_nombr', 'nombre_dpto', 'departamento']), None)
                
                if col_mpio:
                    gdf_mun[col_mpio] = gdf_mun[col_mpio].apply(decodificar_tildes).astype(str).str.title()
                    if col_depto:
                        gdf_mun[col_depto] = gdf_mun[col_depto].apply(decodificar_tildes).astype(str).str.title()
                        gdf_mun['display_name'] = gdf_mun[col_mpio] + " - " + gdf_mun[col_depto]
                    else:
                        gdf_mun['display_name'] = gdf_mun[col_mpio]
                        
                    lista = sorted(gdf_mun['display_name'].dropna().unique().tolist())
                    sel = st.selectbox("🏢 Seleccione Municipio:", lista)
                    
                    if sel:
                        nombre_zona = sel.split(" - ")[0] 
                        gdf_zona = gdf_mun[gdf_mun['display_name'] == sel].copy()
                        gdf_zona['nombre'] = nombre_zona 
                else: st.warning("⚠️ No se encontró la columna de municipio en la base de datos.")
                    
            # --- D. DEPARTAMENTO ---
            else:
                nombre_zona = "Antioquia"
                try:
                    gdf_mun = cargar_mapa_municipios()
                    if gdf_mun.crs is None or gdf_mun.crs.to_string() != "EPSG:4326": gdf_mun = gdf_mun.to_crs("EPSG:4326")
                    gdf_zona = gpd.GeoDataFrame({'nombre': ['Antioquia']}, geometry=[gdf_mun.unary_union], crs="EPSG:4326")
                except:
                    gdf_zona = gpd.GeoDataFrame({'nombre': ['Antioquia']}, geometry=[box(-77.5, 5.0, -73.5, 9.0)], crs="EPSG:4326")

            # --- FILTRAR ESTACIONES (Original) ---
            if gdf_zona is not None and not gdf_zona.empty:
                if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326": gdf_zona = gdf_zona.to_crs("EPSG:4326")
                buff_km = st.slider("Radio Buffer (Área de Influencia en km):", min_value=0.0, max_value=50.0, value=15.0, step=1.0, key="buffer_global_km")
                buff_deg = buff_km / 111.0 
                minx, miny, maxx, maxy = gdf_zona.total_bounds
                
                q_est = text(f"SELECT id_estacion, nombre, latitud, longitud, altitud FROM estaciones WHERE longitud BETWEEN {minx - buff_deg} AND {maxx + buff_deg} AND latitud BETWEEN {miny - buff_deg} AND {maxy + buff_deg}")
                df_est = pd.read_sql(q_est, engine)
                
                if not df_est.empty:
                    gdf_ptos = gpd.GeoDataFrame(df_est, geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), crs="EPSG:4326")
                    zona_buffered = gdf_zona.copy()
                    if buff_deg > 0: zona_buffered['geometry'] = zona_buffered.geometry.buffer(buff_deg)
                    est_in = gpd.sjoin(gdf_ptos, zona_buffered, how="inner", predicate="intersects").drop_duplicates(subset=['id_estacion'])
                    
                    if not est_in.empty:
                        ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                        altitud_ref = est_in['altitud'].mean()
                        st.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                    else: st.warning("0 estaciones en el área exacta.")
                else: st.warning("0 estaciones en el cuadrante.")

        except Exception as e:
            st.error(f"Error crítico en selector: {e}")
            
    # ====================================================================
    # 🧠 ORQUESTADOR SILENCIOSO (BLINDADO)
    # ====================================================================
    # Evitamos que busque demografía para zonas abstractas o nulas
    zonas_ignoradas = ["Antioquia", "Bloque Regional", "-- TODAS --", "-- Seleccione --", ""]
    
    zona_activa = st.session_state.get('zona_activa_global')
    
    # 🛑 FIX: Solo ejecutamos si es un Municipio, Subcuenca o Región válida
    if nombre_zona not in zonas_ignoradas and "Región" not in nombre_zona and nombre_zona != zona_activa:
        st.session_state['zona_activa_global'] = nombre_zona
        
        claves_a_borrar = ['pob_hum_calc_met', 'ica_bovinos_calc_met', 'ica_porcinos_calc_met', 'ica_aves_calc_met', 'demanda_total_m3s', 'carga_dbo_total_ton']
        for k in claves_a_borrar:
            st.session_state.pop(k, None)
            
        try:
            from modules.utils import obtener_metabolismo_exacto
            datos_precargados = obtener_metabolismo_exacto(nombre_zona, 2025)
            if datos_precargados:
                st.session_state['aleph_pob_total'] = datos_precargados.get('pob_total', 0)
                st.session_state['ica_bovinos_calc_met'] = datos_precargados.get('bovinos', 0)
                st.session_state['ica_porcinos_calc_met'] = datos_precargados.get('porcinos', 0)
                st.session_state['ica_aves_calc_met'] = datos_precargados.get('aves', 0)
                st.session_state['aleph_lugar'] = nombre_zona
        except Exception:
            pass 
            
    # ====================================================================
    # 💾 RENDERIZAR EL GESTOR DE ESCENARIOS
    # ====================================================================
    renderizar_gestor_escenarios(nombre_zona)
            
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
