# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
import io
import requests
import numpy as np

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

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_entorno_maestro():
    """Fusión maestra: Polígonos de BD + Diccionario Administrativo de Excel"""
    try:
        engine = db_manager.get_engine()
        # 1. Geometría desde SQL
        gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry").to_crs("EPSG:3116")
        gdf_cue = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry").to_crs("EPSG:3116")
        
        # 2. Diccionario desde Excel (Supabase Storage)
        url_excel = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
        res = requests.get(url_excel, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=15)
        df_adm = pd.read_excel(io.BytesIO(res.content))
        df_adm['dp_mp'] = df_adm['dp_mp'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)

        # 3. Cruzamos el mapa con el Excel para recuperar 'subregion', 'depto_nom' y 'car'
        col_id = next((c for c in ['mpio_cdpmp', 'MPIO_CDPMP', 'dp_mp'] if c in gdf_mun.columns), None)
        if col_id:
            gdf_mun['join_id'] = gdf_mun[col_id].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
            gdf_mun = gdf_mun.merge(df_adm[['dp_mp', 'depto_nom', 'subregion', 'car', 'region']], 
                                    left_on='join_id', right_on='dp_mp', how='left')
        
        # Renombramos para compatibilidad con tu código histórico
        gdf_mun = gdf_mun.rename(columns={'depto_nom': 'DPTO_CNMBR', 'municipio': 'MPIO_CNMBR'})
        
        # 4. Cargamos estaciones para el motor espacial
        q_est = text("SELECT id_estacion, altitud, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
        gdf_est = gpd.read_postgis(q_est, engine, geom_col="geometry").to_crs("EPSG:3116")
        
        return gdf_mun, gdf_cue, gdf_est
    except Exception as e:
        st.error(f"Error en el motor espacial: {e}")
        return None, None, None

def calcular_estaciones_en_zona(gdf_zona, gdf_estaciones, buffer_km):
    """Tracción 4x4: Encuentra estaciones por geometría, sin depender de columnas de texto"""
    if gdf_zona is None or gdf_zona.empty or gdf_estaciones is None: return [], 1500
    zona_buffer = gdf_zona.geometry.unary_union.buffer(buffer_km * 1000)
    est_in = gdf_estaciones[gdf_estaciones.geometry.within(zona_buffer)]
    return est_in['id_estacion'].astype(str).tolist(), est_in['altitud'].mean() if not est_in.empty else 1500

def renderizar_telemetria_aleph():
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧠 Telemetría del Aleph", expanded=True):
        pob = st.session_state.get('aleph_pob_total', 0)
        st.metric("Demografía Conectada", f"{pob:,.0f} habs")
        if st.sidebar.button("🧹 Purgar Memoria y Caché", key="btn_purgar_aleph_vFinal"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

# ====================================================================
# --- SELECTOR ESPACIAL (VERSIÓN DEFINITIVA CON MEMORIA) ---
# ====================================================================
def render_selector_espacial():
    renderizar_telemetria_aleph()
    st.sidebar.markdown("### 📍 Filtros Geográficos Principales")

    def selector_seguro(label, opciones, clave_memoria):
        opciones_lista = list(opciones)
        if not opciones_lista: opciones_lista = ["-- NO HAY DATOS --"]
        idx = 0
        if clave_memoria in st.session_state and st.session_state[clave_memoria] in opciones_lista:
            idx = opciones_lista.index(st.session_state[clave_memoria])
        sel = st.sidebar.selectbox(label, opciones_lista, index=idx, key=clave_memoria)
        return sel

    # 1. Carga de datos
    gdf_mun, gdf_cue, gdf_est = cargar_entorno_maestro()
    if gdf_mun is None: return [], "Error de Carga", 1500, None

    # 2. Nivel de Agregación
    opciones_agregacion = ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"]
    nivel_agregacion = st.sidebar.radio("Nivel de Agregación:", opciones_agregacion, 
                                        index=opciones_agregacion.index(st.session_state.get('mem_agregacion', "Por Municipio")),
                                        key="mem_agregacion")

    nombre_zona = "Sin Selección"
    gdf_seleccionado = None

    # --- LÓGICA DE FILTRADO ---
    if nivel_agregacion == "Por Cuenca":
        ruta = selector_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta")
        if ruta == "Hidrología":
            nivel = selector_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nss")
            col = {"NSS1":"nom_nss1", "NSS2":"nom_nss2", "NSS3":"nom_nss3", "SZH":"nom_szh", "ZH":"nomzh", "AH":"nomah"}.get(nivel)
            territorio = selector_seguro(f"🎯 Territorio Exacto ({nivel}):", sorted(gdf_cue[col].dropna().unique()), "mem_terr_cue")
            gdf_seleccionado = gdf_cue[gdf_cue[col] == territorio]
            nombre_zona = territorio
        else:
            nivel = selector_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión"], "mem_adm_nss")
            col = "CorpoAmb" if nivel == "CAR" else "depto_regi"
            territorio = selector_seguro(f"🎯 Territorio Exacto ({nivel}):", sorted(gdf_cue[col].dropna().unique()), "mem_terr_adm")
            gdf_seleccionado = gdf_cue[gdf_cue[col] == territorio]
            nombre_zona = territorio

    elif nivel_agregacion == "Por Municipio":
        col_m = next((c for c in ['MPIO_CNMBR', 'mpio_cnmbr', 'municipio'] if c in gdf_mun.columns), 'municipio')
        muni = selector_seguro("Municipio:", sorted(gdf_mun[col_m].dropna().unique()), "mem_muni")
        gdf_seleccionado = gdf_mun[gdf_mun[col_m] == muni]
        nombre_zona = muni

    elif nivel_agregacion == "Por Región":
        reg = selector_seguro("Región:", sorted(gdf_mun['subregion'].dropna().unique()), "mem_reg")
        gdf_seleccionado = gdf_mun[gdf_mun['subregion'] == reg]
        nombre_zona = reg

    elif nivel_agregacion == "Departamento":
        dep = selector_seguro("Departamento:", sorted(gdf_mun['DPTO_CNMBR'].dropna().unique()), "mem_dep")
        gdf_seleccionado = gdf_mun[gdf_mun['DPTO_CNMBR'] == dep]
        nombre_zona = dep

    # 3. Buffer y Cálculo de Estaciones
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, float(st.session_state.get('buffer_global_km', 25.0)), 5.0, key="buffer_global_km")
    ids_estaciones, altitud_ref = calcular_estaciones_en_zona(gdf_seleccionado, gdf_est, buffer_km)
    
    st.session_state['aleph_lugar'] = decodificar_tildes(nombre_zona)
    return ids_estaciones, decodificar_tildes(nombre_zona), altitud_ref, gdf_seleccionado

def renderizar_menu_navegacion(pagina_actual=""): pass
