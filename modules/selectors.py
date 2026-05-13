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
    """Panel de control universal (Blindado contra duplicados)"""
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧠 Telemetría del Aleph", expanded=True):
        pob_viva = st.session_state.get('aleph_pob_total', 0)
        st.metric("Demografía Conectada", f"{pob_viva:,.0f} habs")
        if st.button("🧹 Purgar Memoria y Caché", key="btn_purgar_aleph_unico"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

# ====================================================================
# --- MOTORES DE CARGA: LA FUSIÓN DEL MAPA Y EL EXCEL ---
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
        # 1. Bajar geometría de PostgreSQL
        gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry").to_crs("EPSG:3116")
        
        # 2. Bajar Diccionario Administrativo (Excel de Supabase)
        import io, requests
        url = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False)
        df_excel = pd.read_excel(io.BytesIO(res.content))
        
        # 3. Limpiar códigos para que coincidan (DANE)
        df_excel['dp_mp'] = df_excel['dp_mp'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
        
        col_id = next((c for c in ['mpio_cdpmp', 'MPIO_CDPMP', 'dp_mp', 'MPIO_CDP'] if c in gdf_mun.columns), None)
        if col_id:
            gdf_mun['dp_mp_join'] = gdf_mun[col_id].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
            # 4. LA FUSIÓN: Ahora el mapa tiene subregión y departamento
            gdf_mun = gdf_mun.merge(df_excel[['dp_mp', 'depto_nom', 'region', 'subregion', 'car']], 
                                    left_on='dp_mp_join', right_on='dp_mp', how='left')
        return gdf_mun
    except Exception as e:
        st.error(f"Error cargando municipios: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_estaciones_geo():
    try:
        engine = db_manager.get_engine()
        q = text("SELECT id_estacion, altitud, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
        return gpd.read_postgis(q, engine, geom_col="geometry").to_crs("EPSG:3116")
    except: return None

# ====================================================================
# --- MOTOR ESPACIAL 4x4 (REEMPLAZA A 'estaciones_id') ---
# ====================================================================
def calcular_estaciones_en_buffer(gdf_zona, gdf_estaciones, buffer_km):
    if gdf_zona is None or gdf_zona.empty or gdf_estaciones is None or gdf_estaciones.empty:
        return [], 1500
    
    zona_unida = gdf_zona.geometry.unary_union
    zona_buffer = zona_unida.buffer(buffer_km * 1000)
    estaciones_in = gdf_estaciones[gdf_estaciones.geometry.within(zona_buffer)]
    
    ids = estaciones_in['id_estacion'].astype(str).tolist()
    alt = estaciones_in['altitud'].mean() if not estaciones_in['altitud'].isnull().all() else 1500
    return ids, alt

# ====================================================================
# --- SELECTOR PRINCIPAL (CON MEMORIA) ---
# ====================================================================
def render_selector_espacial():
    renderizar_telemetria_aleph()
    st.sidebar.markdown("### 📍 Filtros Geográficos Principales")

    def selector_seguro(label, opciones, clave_memoria):
        opciones_lista = list(opciones)
        if not opciones_lista: opciones_lista = ["-- NO HAY DATOS --"]
        idx_defecto = 0
        if clave_memoria in st.session_state and st.session_state[clave_memoria] in opciones_lista:
            idx_defecto = opciones_lista.index(st.session_state[clave_memoria])
        seleccion = st.sidebar.selectbox(label, opciones_lista, index=idx_defecto, key=clave_memoria)
        return seleccion

    opciones_agregacion = ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"]
    idx_agr = opciones_agregacion.index(st.session_state['mem_nivel_agregacion']) if 'mem_nivel_agregacion' in st.session_state else 0
    nivel_agregacion = st.sidebar.radio("Nivel de Agregación:", opciones_agregacion, index=idx_agr, key="radio_nivel_agregacion")
    st.session_state['mem_nivel_agregacion'] = nivel_agregacion

    nombre_zona = "Sin Selección"
    gdf_zona = None
    
    # --- LÓGICA POR CUENCAS ---
    if nivel_agregacion == "Por Cuenca":
        df_cue = cargar_maestro_cuencas()
        if df_cue is not None and not df_cue.empty:
            ruta_busqueda = selector_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta_busqueda")
            
            if ruta_busqueda == "Hidrología":
                nivel_evaluar = selector_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nivel_evaluar")
                col_busqueda = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3", "SZH": "nom_szh", "ZH": "nomzh", "AH": "nomah"}.get(nivel_evaluar, "nom_nss1")
                
                if col_busqueda in df_cue.columns:
                    lista_territorios = sorted(df_cue[col_busqueda].dropna().unique().tolist())
                    territorio_sel = selector_seguro(f"🎯 2. Territorio Exacto ({nivel_evaluar}):", lista_territorios, "mem_territorio_sel")
                    
                    if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                        nombre_zona = decodificar_tildes(territorio_sel)
                        gdf_zona = df_cue[df_cue[col_busqueda] == territorio_sel]
            
            elif ruta_busqueda == "Administrativo":
                nivel_evaluar = selector_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión", "Departamento"], "mem_nivel_evaluar_admin")
                col_busqueda = {"CAR": "CorpoAmb", "Subregión": "depto_regi", "Departamento": "departamen"}.get(nivel_evaluar, "CorpoAmb")
                
                if col_busqueda in df_cue.columns:
                    lista_territorios = sorted(df_cue[col_busqueda].dropna().unique().tolist())
                    territorio_sel = selector_seguro(f"🎯 2. Territorio Exacto ({nivel_evaluar}):", lista_territorios, "mem_territorio_admin")
                    
                    if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                        nombre_zona = decodificar_tildes(territorio_sel)
                        gdf_zona = df_cue[df_cue[col_busqueda] == territorio_sel]

    # --- LÓGICA POR MUNICIPIO, REGIÓN, DEPARTAMENTO ---
    else:
        df_mun = cargar_maestro_municipios()
        if df_mun is not None and not df_mun.empty:
            col_mun = next((c for c in ['MPIO_CNMBR', 'mpio_cnmbr'] if c in df_mun.columns), None)
            
            if nivel_agregacion == "Por Municipio" and col_mun:
                lista = sorted(df_mun[col_mun].dropna().unique().tolist())
                sel = selector_seguro("Municipio:", lista, "mem_municipio_sel")
                if sel and sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(sel)
                    gdf_zona = df_mun[df_mun[col_mun] == sel]

            elif nivel_agregacion == "Por Región" and 'subregion' in df_mun.columns:
                lista = sorted(df_mun['subregion'].dropna().unique().tolist())
                sel = selector_seguro("Región:", lista, "mem_region_sel")
                if sel and sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(sel)
                    gdf_zona = df_mun[df_mun['subregion'] == sel]

            elif nivel_agregacion == "Departamento" and 'depto_nom' in df_mun.columns:
                lista = sorted(df_mun['depto_nom'].dropna().unique().tolist())
                sel = selector_seguro("Departamento:", lista, "mem_depto_sel")
                if sel and sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(sel)
                    gdf_zona = df_mun[df_mun['depto_nom'] == sel]

    st.sidebar.markdown("---")
    
    # Memoria del Slider y Cálculo Espacial
    buffer_def = float(st.session_state.get('buffer_global_km', 25.0))
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, buffer_def, 5.0, key="slider_buffer_km")
    st.session_state['buffer_global_km'] = buffer_km
    st.session_state['aleph_lugar'] = nombre_zona

    # 🚜 LA TRACCIÓN 4x4 EN ACCIÓN
    gdf_est = cargar_estaciones_geo()
    ids_estaciones, altitud_ref = calcular_estaciones_en_buffer(gdf_zona, gdf_est, buffer_km)

    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona

def renderizar_menu_navegacion(pagina_actual=""): pass
