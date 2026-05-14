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
# --- HERRAMIENTAS BASE Y TELEMETRÍA ---
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
# --- MOTORES DE CARGA (GEOMETRÍA Y ESTACIONES) ---
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
        col_map = {}
        for col in gdf.columns:
            if col.lower() in ['mpio_cnmbr', 'municipio']: col_map[col] = 'MPIO_CNMBR'
            if col.lower() == 'dpto_cnmbr': col_map[col] = 'DPTO_CNMBR'
        if col_map: gdf = gdf.rename(columns=col_map)
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
    gdf_est = cargar_estaciones_geometria()
    if gdf_zona is None or gdf_zona.empty or gdf_est is None or gdf_est.empty:
        return [], 1500
    area_busqueda = gdf_zona.geometry.unary_union.buffer(buffer_km * 1000)
    est_finales = gdf_est[gdf_est.geometry.within(area_busqueda)]
    ids = est_finales['id_estacion'].astype(str).tolist()
    alt = est_finales['altitud'].mean() if not est_finales.empty else 1500
    return ids, alt

# ====================================================================
# --- SELECTOR ESPACIAL (CON MEMORIA) ---
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
    
    if nivel_agregacion == "Por Cuenca":
        df_c = cargar_maestro_cuencas()
        if df_c is not None and not df_c.empty:
            ruta = selectbox_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta_busqueda")
            if ruta == "Hidrología":
                nivel = selectbox_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nivel_hidro")
                col = {"NSS1":"nom_nss1", "NSS2":"nom_nss2", "NSS3":"nom_nss3", "SZH":"nom_szh", "ZH":"nomzh", "AH":"nomah"}.get(nivel)
                territorio = selectbox_seguro(f"🎯 Territorio ({nivel}):", sorted(df_c[col].dropna().unique()), "mem_terr_hidro")
                gdf_zona = df_c[df_c[col] == territorio]
                nombre_zona = territorio
            else:
                nivel = selectbox_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión"], "mem_nivel_admin")
                col = "CorpoAmb" if nivel == "CAR" else "depto_regi"
                territorio = selectbox_seguro(f"🎯 Territorio ({nivel}):", sorted(df_c[col].dropna().unique()), "mem_terr_admin")
                gdf_zona = df_c[df_c[col] == territorio]
                nombre_zona = territorio

    else:
        df_m = cargar_maestro_municipios()
        if df_m is not None and not df_m.empty:
            if nivel_agregacion == "Por Municipio":
                mun = selectbox_seguro("Municipio:", sorted(df_m['MPIO_CNMBR'].unique()), "mem_mun_sel")
                gdf_zona = df_m[df_m['MPIO_CNMBR'] == mun]
                nombre_zona = mun
            elif nivel_agregacion == "Por Región":
                reg = selectbox_seguro("Región:", sorted(df_m['subregion'].unique()), "mem_reg_sel")
                gdf_zona = df_m[df_m['subregion'] == reg]
                nombre_zona = reg
            elif nivel_agregacion == "Departamento":
                dep = selectbox_seguro("Departamento:", sorted(df_m['DPTO_CNMBR'].unique()), "mem_dep_sel")
                gdf_zona = df_m[df_m['DPTO_CNMBR'] == dep]
                nombre_zona = dep

    st.sidebar.markdown("---")
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, float(st.session_state.get('buffer_global_km', 25.0)), 5.0, key="slider_buffer_mem")
    st.session_state['buffer_global_km'] = buffer_km
    st.session_state['aleph_lugar'] = decodificar_tildes(nombre_zona)

    ids_estaciones, alt_ref = encontrar_estaciones_en_mapa(gdf_zona, buffer_km)
    return ids_estaciones, decodificar_tildes(nombre_zona), alt_ref, gdf_zona

# ====================================================================
# --- 🧭 MENÚ DE NAVEGACIÓN (RESTAURADO) ---
# ====================================================================
def renderizar_menu_navegacion(pagina_actual=""):
    """Dibuja los botones de navegación por categorías."""
    st.sidebar.markdown("### 🧭 Navegación del Sistema")
    
    categorias = {
        "🌊 Hidrología": [
            ("Inicio", "🏠"),
            ("Clima", "☁️"),
            ("Aguas Subterráneas", "💧"),
            ("Balance Hídrico", "⚖️"),
            ("Caudales Regionales", "📏")
        ],
        "🧪 Calidad y Salud": [
            ("Calidad del Agua", "🔬"),
            ("Salud Ecosistémica", "🌳")
        ],
        "🧠 Análisis Avanzado": [
            ("Satélite en Vivo", "🛰️"),
            ("Escenarios Climáticos", "🎭"),
            ("Detective", "🕵️")
        ]
    }

    for cat, pags in categorias.items():
        with st.sidebar.expander(cat, expanded=(cat == "🌊 Hidrología")):
            for pag, icon in pags:
                # El botón cambia el estado y recarga la página
                if st.button(f"{icon} {pag}", key=f"nav_{pag}", use_container_width=True):
                    # Aquí la lógica de navegación según tu estructura de archivos
                    pass 

    st.sidebar.markdown("---")
