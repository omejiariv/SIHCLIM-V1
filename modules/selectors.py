# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
import io
import requests

from modules import db_manager
from modules.config import Config

# ====================================================================
# --- HERRAMIENTAS BASE ---
# ====================================================================
def decodificar_tildes(texto):
    if not isinstance(texto, str): return texto
    try:
        if 'Ã' in texto or 'ã' in texto or '\x8d' in texto:
            return texto.encode('latin1').decode('utf-8')
    except: pass
    return texto

def renderizar_telemetria_aleph():
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧠 Telemetría del Aleph", expanded=True):
        pob_viva = st.session_state.get('aleph_pob_total', 0)
        st.metric("Demografía Conectada", f"{pob_viva:,.0f} habs")
        if st.sidebar.button("🧹 Purgar Memoria y Caché", key="btn_purgar_aleph_unico"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

# ====================================================================
# --- MOTORES DE CARGA (DIRECTO DESDE POSTGRESQL CON GEOMETRÍA) ---
# ====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_cuencas():
    try:
        engine = db_manager.get_engine()
        return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry").to_crs("EPSG:3116")
    except Exception as e:
        st.error(f"Error cargando cuencas: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_municipios():
    try:
        engine = db_manager.get_engine()
        gdf = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry").to_crs("EPSG:3116")
        # Escudo de nombres para asegurar compatibilidad
        col_map = {}
        for col in gdf.columns:
            if col.lower() == 'mpio_cnmbr' or col.lower() == 'municipio': col_map[col] = 'MPIO_CNMBR'
            if col.lower() == 'dpto_cnmbr': col_map[col] = 'DPTO_CNMBR'
        if col_map:
            gdf = gdf.rename(columns=col_map)
        return gdf
    except Exception as e:
        st.error(f"Error cargando municipios: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_estaciones_geometria():
    try:
        engine = db_manager.get_engine()
        q = text("SELECT id_estacion, altitud, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
        return gpd.read_postgis(q, engine, geom_col="geometry").to_crs("EPSG:3116")
    except: return None

def encontrar_estaciones_en_mapa(gdf_zona, buffer_km):
    """🌍 El nuevo motor: Busca estaciones cruzando coordenadas, no textos."""
    gdf_est = cargar_estaciones_geometria()
    if gdf_zona is None or gdf_zona.empty or gdf_est is None or gdf_est.empty:
        return [], 1500
    
    # Creamos el área de influencia (Buffer)
    area_busqueda = gdf_zona.geometry.unary_union.buffer(buffer_km * 1000)
    estaciones_finales = gdf_est[gdf_est.geometry.within(area_busqueda)]
    
    ids = estaciones_finales['id_estacion'].astype(str).tolist()
    alt = estaciones_finales['altitud'].mean() if not estaciones_finales.empty else 1500
    return ids, alt

# ====================================================================
# --- SELECTOR ESPACIAL ---
# ====================================================================
def render_selector_espacial():
    renderizar_telemetria_aleph()
    st.sidebar.markdown("### 📍 Filtros Geográficos Principales")

    def selectbox_seguro(label, opciones, clave):
        lista = list(opciones)
        if not lista: lista = ["-- NO HAY DATOS --"]
        idx = 0
        if clave in st.session_state and st.session_state[clave] in lista:
            idx = lista.index(st.session_state[clave])
        return st.sidebar.selectbox(label, lista, index=idx, key=clave)

    opciones_agr = ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"]
    idx_agr = opciones_agr.index(st.session_state.get('mem_nivel_agregacion', "Por Cuenca")) if st.session_state.get('mem_nivel_agregacion') in opciones_agr else 0
    nivel_agregacion = st.sidebar.radio("Nivel de Agregación:", opciones_agr, index=idx_agr, key="radio_agregacion_mem")
    st.session_state['mem_nivel_agregacion'] = nivel_agregacion

    nombre_zona = "Sin Selección"
    gdf_zona = None
    
    # --- LÓGICA POR CUENCAS ---
    if nivel_agregacion == "Por Cuenca":
        df_cuencas = cargar_maestro_cuencas()
        if df_cuencas is not None and not df_cuencas.empty:
            ruta_busqueda = selectbox_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta_busqueda")
            
            if ruta_busqueda == "Hidrología":
                nivel_evaluar = selectbox_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nivel_evaluar_hidro")
                col_busqueda = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3", "SZH": "nom_szh", "ZH": "nomzh", "AH": "nomah"}.get(nivel_evaluar, "nom_nss1")
                
                lista_territorios = sorted(df_cuencas[col_busqueda].dropna().unique().tolist())
                territorio_sel = selectbox_seguro(f"🎯 2. Territorio ({nivel_evaluar}):", lista_territorios, "mem_terr_exacto_hidro")
                
                if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(territorio_sel)
                    gdf_zona = df_cuencas[df_cuencas[col_busqueda] == territorio_sel]

            elif ruta_busqueda == "Administrativo":
                nivel_evaluar = selectbox_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión", "Departamento"], "mem_nivel_evaluar_admin")
                col_busqueda = {"CAR": "CorpoAmb", "Subregión": "depto_regi", "Departamento": "departamen"}.get(nivel_evaluar, "CorpoAmb")
                
                lista_territorios = sorted(df_cuencas[col_busqueda].dropna().unique().tolist())
                territorio_sel = selectbox_seguro(f"🎯 2. Territorio ({nivel_evaluar}):", lista_territorios, "mem_terr_exacto_admin")
                
                if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(territorio_sel)
                    gdf_zona = df_cuencas[df_cuencas[col_busqueda] == territorio_sel]

    # --- LÓGICA POR MUNICIPIO / REGION / DEPTO ---
    else:
        df_mun = cargar_maestro_municipios()
        if df_mun is not None and not df_mun.empty:
            
            if nivel_agregacion == "Por Municipio":
                lista_mun = sorted(df_mun['MPIO_CNMBR'].dropna().unique().tolist())
                mun_sel = selectbox_seguro("Municipio:", lista_mun, "mem_municipio_sel")
                if mun_sel and mun_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(mun_sel)
                    gdf_zona = df_mun[df_mun['MPIO_CNMBR'] == mun_sel]
                    
            elif nivel_agregacion == "Por Región" and 'subregion' in df_mun.columns:
                lista_reg = sorted(df_mun['subregion'].dropna().unique().tolist())
                reg_sel = selectbox_seguro("Región:", lista_reg, "mem_region_sel")
                if reg_sel and reg_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(reg_sel)
                    gdf_zona = df_mun[df_mun['subregion'] == reg_sel]
                    
            elif nivel_agregacion == "Departamento" and 'DPTO_CNMBR' in df_mun.columns:
                lista_dep = sorted(df_mun['DPTO_CNMBR'].dropna().unique().tolist())
                dep_sel = selectbox_seguro("Departamento:", lista_dep, "mem_depto_sel")
                if dep_sel and dep_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(dep_sel)
                    gdf_zona = df_mun[df_mun['DPTO_CNMBR'] == dep_sel]

    st.sidebar.markdown("---")
    
    # 🧠 MEMORIA DEL SLIDER DE BUFFER
    buffer_def = float(st.session_state.get('buffer_global_km', 25.0))
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, buffer_def, 5.0, key="slider_buffer_mem")
    st.session_state['buffer_global_km'] = buffer_km
    st.session_state['aleph_lugar'] = nombre_zona

    # 🚀 EJECUCIÓN DEL NUEVO MOTOR ESPACIAL
    ids_estaciones, altitud_ref = encontrar_estaciones_en_mapa(gdf_zona, buffer_km)

    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona

def renderizar_menu_navegacion(pagina_actual=""): pass
