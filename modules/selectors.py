# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from shapely.geometry import box
import json
import time
import unicodedata
import re

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

def normalizar_texto_busqueda(t):
    """Estandariza texto para comparaciones espaciales sin tildes."""
    if pd.isna(t): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(t).upper().strip()) if unicodedata.category(c) != 'Mn')

# ====================================================================
# 📂 NAVEGACIÓN GLOBAL SINCRONIZADA
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
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="Satélite en Vivo", icon="🛰️")

# ====================================================================
# ☁️ CONEXIÓN A SUPABASE (GESTIÓN DE ESCENARIOS)
# ====================================================================
@st.cache_resource
def get_supabase_client():
    from supabase import create_client
    url_sb = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    key_sb = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    if url_sb and key_sb:
        return create_client(url_sb, key_sb)
    return None

def renderizar_gestor_escenarios(nombre_zona_actual):
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Gestor de Escenarios (Snapshots)", expanded=False):
        supabase = get_supabase_client()
        if not supabase:
            st.error("No hay conexión a Supabase.")
            return

        t_save, t_load = st.tabs(["Guardar", "Cargar"])
        
        with t_save:
            nombre_esc = st.text_input("Nombre del Escenario:", placeholder="Ej: Plan Maestro Buey 2030")
            if st.button("💾 Guardar Sesión Actual", use_container_width=True):
                if nombre_esc:
                    # Serialización segura del estado
                    estado_json = {k: v for k, v in st.session_state.items() 
                                  if isinstance(v, (int, float, str, bool, list, dict)) 
                                  and not k.startswith('FormSubmitter')}
                    try:
                        supabase.table("escenarios_guardados").insert({
                            "nombre_escenario": nombre_esc,
                            "territorio": nombre_zona_actual,
                            "estado_json": estado_json
                        }).execute()
                        st.success("✅ ¡Escenario guardado!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Escribe un nombre.")

        with t_load:
            try:
                res = supabase.table("escenarios_guardados").select("*").order("fecha_creacion", desc=True).execute()
                if res.data:
                    opcs = {f"{e['nombre_escenario']} ({e['territorio']})": e['id'] for e in res.data}
                    sel = st.selectbox("Restaurar proyecto:", list(opcs.keys()))
                    if st.button("📂 Cargar", type="primary", use_container_width=True):
                        idx = opcs[sel]
                        data_rec = supabase.table("escenarios_guardados").select("estado_json").eq("id", idx).execute()
                        for k, v in data_rec.data[0]['estado_json'].items():
                            st.session_state[k] = v
                        st.success("✅ Memoria restaurada.")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.info("Sin escenarios guardados.")
            except: pass

# ====================================================================
# 🌍 SELECTOR ESPACIAL MAESTRO (VERSION INTEGRAL)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_cuencas():
    return gpd.read_postgis("SELECT * FROM cuencas", db_manager.get_engine(), geom_col="geometry")

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_municipios():
    return gpd.read_postgis("SELECT * FROM municipios", db_manager.get_engine(), geom_col="geometry")

def render_selector_espacial():
    ids_estaciones, nombre_zona, altitud_ref, gdf_zona = [], "Antioquia", 1500.0, None
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"], index=0)
        
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            gdf_c = cargar_mapa_cuencas()
            ruta = st.selectbox("Ruta de Búsqueda:", ["Hidrología", "CAR"], index=0)
            if ruta == "Hidrología":
                ah = st.selectbox("AH:", ["-- Seleccione --"] + sorted(gdf_c['nomah'].dropna().unique()))
                if ah != "-- Seleccione --":
                    zh = st.selectbox("ZH:", ["-- TODAS --"] + sorted(gdf_c[gdf_c['nomah']==ah]['nomzh'].dropna().unique()))
                    df_f = gdf_c[gdf_c['nomah']==ah] if zh=="-- TODAS --" else gdf_c[gdf_c['nomzh']==zh]
                    szh = st.selectbox("SZH:", ["-- TODAS --"] + sorted(df_f['nom_szh'].dropna().unique()))
                    gdf_filtrado_base = df_f if szh=="-- TODAS --" else df_f[df_f['nom_szh']==szh]
                    
                    nivel = st.radio("Resolución:", ["NSS1", "NSS2", "NSS3"], horizontal=True)
                    col_obj = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                    sel_fin = st.selectbox("🎯 Territorio:", ["-- BLOQUE --"] + sorted(gdf_filtrado_base[col_obj].dropna().unique()))
                    if sel_fin != "-- BLOQUE --":
                        nombre_zona, gdf_zona = sel_fin, gdf_filtrado_base[gdf_filtrado_base[col_obj]==sel_fin]
                    else:
                        nombre_zona, gdf_zona = "Bloque Hidro", gdf_filtrado_base

        # --- B. POR REGIÓN (Restaurado) ---
        elif modo == "Por Región":
            try:
                df_m = pd.read_excel("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx")
                df_m.columns = [c.lower() for c in df_m.columns]
                lista_reg = sorted([str(r).title() for r in df_m['region'].dropna().unique()])
                sel_reg = st.selectbox("📍 Región:", ["-- Seleccione --"] + lista_reg)
                if sel_reg != "-- Seleccione --":
                    nombre_zona = f"Región {sel_reg}"
                    cods = df_m[df_m['region'].str.lower()==sel_reg.lower()]['dp_mp'].astype(str).str.zfill(5).tolist()
                    gdf_mun = cargar_mapa_municipios()
                    gdf_zona = gdf_mun[gdf_mun['mpio_ccdgo'].astype(str).str.zfill(5).isin(cods)]
            except: pass

        # --- C. POR MUNICIPIO (Restaurado) ---
        elif modo == "Por Municipio":
            gdf_mun = cargar_mapa_municipios()
            gdf_mun['display'] = gdf_mun['mpio_cnmbr'].apply(decodificar_tildes).str.title()
            sel_mpio = st.selectbox("🏢 Municipio:", sorted(gdf_mun['display'].unique()))
            nombre_zona, gdf_zona = sel_mpio, gdf_mun[gdf_mun['display']==sel_mpio]

        # --- D. DEPARTAMENTO ---
        else:
            gdf_mun = cargar_mapa_municipios()
            nombre_zona, gdf_zona = "Antioquia", gpd.GeoDataFrame({'nombre':['Antioquia']}, geometry=[gdf_mun.unary_union], crs=gdf_mun.crs)

        # --- FILTRO DE ESTACIONES ---
        if gdf_zona is not None and not gdf_zona.empty:
            buff = st.slider("Buffer (km):", 0.0, 50.0, 15.0)
            minx, miny, maxx, maxy = gdf_zona.to_crs(4326).total_bounds
            q = text(f"SELECT id_estacion, altitud FROM estaciones WHERE longitud BETWEEN {minx-0.2} AND {maxx+0.2} AND latitud BETWEEN {miny-0.2} AND {maxy+0.2}")
            df_est = pd.read_sql(q, db_manager.get_engine())
            ids_estaciones = df_est['id_estacion'].astype(str).tolist()
            altitud_ref = df_est['altitud'].mean()

    # ====================================================================
    # 🧠 ORQUESTADOR SILENCIOSO (Sincronización Transversal)
    # ====================================================================
    if nombre_zona != st.session_state.get('zona_activa_global'):
        st.session_state['zona_activa_global'] = nombre_zona
        # Limpieza de memoria vieja
        for k in ['pob_hum_calc_met', 'ica_bovinos_calc_met', 'carga_dbo_total_ton']:
            st.session_state.pop(k, None)
        # Precarga silenciosa
        from modules.utils import obtener_metabolismo_exacto
        meta = obtener_metabolismo_exacto(nombre_zona, 2025)
        if meta:
            st.session_state['aleph_pob_total'] = meta.get('pob_total', 0)
            st.session_state['ica_bovinos_calc_met'] = meta.get('bovinos', 0)
            st.session_state['ica_porcinos_calc_met'] = meta.get('porcinos', 0)
            st.session_state['ica_aves_calc_met'] = meta.get('aves', 0)
            st.session_state['aleph_lugar'] = nombre_zona

    renderizar_gestor_escenarios(nombre_zona)
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
