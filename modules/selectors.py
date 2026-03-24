# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from shapely.geometry import box
from modules import db_manager
from modules.config import Config

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
    
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. MODO DE AGREGACIÓN
    modo = st.sidebar.radio(
        "Nivel de Agregación:",
        ["Por Cuenca", "Por Municipio", "Por Región", "Departamento (Antioquia)"],
        index=0
    )
    
    gdf_zona = None
    nombre_zona = "Antioquia"
    altitud_ref = 1500
    
    try:
        # ==========================================
        # --- A. POR CUENCA ---
        # ==========================================
        if modo == "Por Cuenca":
            try:
                gdf_cuencas = cargar_mapa_cuencas() # ⚡ Carga instantánea desde Caché
                
                # --- FILTRO BLINDADO DE COLUMNAS ---
                columnas_permitidas = ['AH', 'ZH', 'SZH', 'Zona', 'N_NSS1', 'SUBC_LBL', 'N-NSS3', 'COD']
                permitidas_lower = [c.lower() for c in columnas_permitidas]
                
                # Rescatamos las columnas ignorando mayúsculas/minúsculas
                columnas_reales = [col for col in gdf_cuencas.columns if col.lower() in permitidas_lower]
                
                # FALLBACK: Si no hay coincidencias, traemos las de texto
                if not columnas_reales:
                    columnas_reales = [c for c in gdf_cuencas.columns if c.lower() not in ['geometry', 'gid', 'objectid', 'shape_length', 'shape_area']]
                
                mapa_nombres = {c.lower(): c.upper() for c in columnas_reales}
                
                # Índice por defecto seguro
                default_idx = 0
                cols_lower = [c.lower() for c in columnas_reales]
                if 'subc_lbl' in cols_lower: default_idx = cols_lower.index('subc_lbl')
                elif 'zona' in cols_lower: default_idx = cols_lower.index('zona')
                
                if len(columnas_reales) == 0:
                    st.sidebar.error("⚠️ La tabla de cuencas no tiene columnas de texto.")
                else:
                    col_nom = st.sidebar.selectbox(
                        "📂 Columna de Nombres:", 
                        options=columnas_reales, 
                        index=min(default_idx, max(0, len(columnas_reales)-1)),
                        format_func=lambda x: mapa_nombres.get(x.lower(), x) if x else "",
                        help="Seleccione el nivel de jerarquía hidrográfica."
                    )
                    
                    if col_nom:
                        # Limpiamos nulos y organizamos alfabéticamente
                        valores_brutos = gdf_cuencas[col_nom].dropna().astype(str).unique().tolist()
                        lista_limpia = sorted([v.strip() for v in valores_brutos if v.strip() != ""])
                        
                        if len(lista_limpia) > 0:
                            sel = st.sidebar.selectbox("🌊 Seleccione Territorio:", lista_limpia)
                            if sel:
                                nombre_zona = sel
                                gdf_zona = gdf_cuencas[gdf_cuencas[col_nom].astype(str).str.strip() == sel]
                        else:
                            st.sidebar.warning("La columna seleccionada no contiene datos.")

            except Exception as e:
                st.sidebar.warning(f"Error cargando cuencas: {e}")

        # ==========================================
        # --- B. POR REGIÓN ---
        # ==========================================
        elif modo == "Por Región":
            try:
                df_reg = pd.read_sql("SELECT DISTINCT subregion FROM estaciones WHERE subregion IS NOT NULL ORDER BY subregion", engine)
                lista_reg = df_reg['subregion'].astype(str).unique().tolist()
                
                sel = st.sidebar.selectbox("Seleccione Región:", lista_reg)
                
                if sel:
                    nombre_zona = f"Región {sel}"
                    q_geo = text(f"SELECT * FROM estaciones WHERE subregion = '{sel}'")
                    df_pts = pd.read_sql(q_geo, engine)
                    
                    if not df_pts.empty:
                        gdf_zona = gpd.GeoDataFrame(
                            df_pts, 
                            geometry=gpd.points_from_xy(df_pts.longitud, df_pts.latitud),
                            crs="EPSG:4326"
                        )
                    else:
                        st.sidebar.warning(f"No hay estaciones en {sel}")
            except Exception as e:
                st.sidebar.warning(f"Error cargando regiones: {e}")        

        # ==========================================
        # --- C. POR MUNICIPIO ---
        # ==========================================
        elif modo == "Por Municipio":
            try:
                gdf_mun = cargar_mapa_municipios() # ⚡ Carga instantánea
                
                cols_texto = [c for c in gdf_mun.columns if c not in ['geometry', 'gid']]
                default_idx = 0
                if 'mpio_cnmbr' in cols_texto: default_idx = cols_texto.index('mpio_cnmbr')
                
                col_nom = st.sidebar.selectbox("📂 Columna de Nombres:", cols_texto, index=default_idx)
                
                if col_nom:
                    lista = sorted(gdf_mun[col_nom].astype(str).unique().tolist())
                    sel = st.sidebar.selectbox("Seleccione Municipio:", lista)
                    if sel:
                        nombre_zona = sel
                        gdf_mun = gdf_mun.to_crs("EPSG:4326")
                        gdf_zona = gdf_mun[gdf_mun[col_nom] == sel]
            except Exception as e:
                st.sidebar.warning(f"Error en tabla municipios: {e}")

        # ==========================================
        # --- D. DEPARTAMENTO ---
        # ==========================================
        else:
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
            
            buff_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 5)
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
                
                # 🔥 EL CAMBIO MAESTRO: sjoin es 100 veces más rápido que unary_union
                est_in = gpd.sjoin(gdf_ptos, zona_buffered, how="inner", predicate="intersects")
                # Removemos duplicados por si un punto toca el borde de dos polígonos
                est_in = est_in.drop_duplicates(subset=['id_estacion'])
                
                if not est_in.empty:
                    ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                    altitud_ref = est_in['altitud'].mean()
                    st.sidebar.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                else:
                    st.sidebar.warning("0 estaciones en el área exacta.")
            else:
                st.sidebar.warning("0 estaciones en el cuadrante.")

    except Exception as e:
        st.sidebar.error(f"Error crítico en selector: {e}")
        
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
