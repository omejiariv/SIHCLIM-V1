# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules import db_manager
from modules.config import Config

def render_selector_espacial():
    """
    Selector espacial con opción manual de columna de nombres.
    """
    engine = db_manager.get_engine()
    
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. MODO DE AGREGACIÓN
    modo = st.sidebar.radio(
        "Nivel de Agregación:",
        # 👇 Agregamos "Por Región" aquí
        ["Por Cuenca", "Por Municipio", "Por Región", "Departamento (Antioquia)"],
        index=0
    )
    
    gdf_zona = None
    nombre_zona = "Antioquia"
    altitud_ref = 1500
    
    try:
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            try:
                gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
                
                # --- FILTRO ESTRICTO DE COLUMNAS (El Radar Geográfico) ---
                columnas_permitidas = ['AH', 'ZH', 'SZH', 'Zona', 'N_NSS1', 'SUBC_LBL', 'N-NSS3', 'COD']
                permitidas_lower = [c.lower() for c in columnas_permitidas]
                
                # Filtramos para usar solo las columnas que existen en la base de datos
                columnas_reales = [col for col in gdf_cuencas.columns if col.lower() in permitidas_lower]
                
                # Diccionario para que en la interfaz se vean en Mayúsculas (estético)
                mapa_nombres = {c.lower(): c.upper() for c in columnas_reales}
                
                # Definir índice por defecto (priorizamos subc_lbl si existe)
                default_idx = 0
                if 'subc_lbl' in columnas_reales: 
                    default_idx = columnas_reales.index('subc_lbl')
                elif 'zona' in columnas_reales:
                    default_idx = columnas_reales.index('zona')
                
                # Renderizamos el selector limpio
                col_nom = st.sidebar.selectbox(
                    "📂 Columna de Nombres:", 
                    options=columnas_reales, 
                    index=default_idx,
                    format_func=lambda x: mapa_nombres.get(x.lower(), x),
                    help="Seleccione el nivel de jerarquía hidrográfica."
                )
                
                if col_nom:
                    # Extraemos valores únicos, eliminando nulos y textos vacíos
                    valores_brutos = gdf_cuencas[col_nom].dropna().astype(str).unique().tolist()
                    lista_limpia = sorted([v.strip() for v in valores_brutos if v.strip() != ""])
                    
                    sel = st.sidebar.selectbox("🌊 Seleccione Territorio:", lista_limpia)
                    
                    if sel:
                        nombre_zona = sel
                        # Filtramos el GeoDataFrame exactamente por la selección
                        gdf_zona = gdf_cuencas[gdf_cuencas[col_nom].astype(str).str.strip() == sel]

            except Exception as e:
                st.sidebar.warning(f"Error cargando cuencas: {e}")
                return [], "", 0, None
                
        # --- NUEVO: POR REGIÓN ---
        elif modo == "Por Región":
            try:
                # Consultamos las regiones únicas disponibles en la tabla de estaciones
                df_reg = pd.read_sql("SELECT DISTINCT subregion FROM estaciones WHERE subregion IS NOT NULL ORDER BY subregion", engine)
                lista_reg = df_reg['subregion'].astype(str).unique().tolist()
                
                sel = st.sidebar.selectbox("Seleccione Región:", lista_reg)
                
                if sel:
                    nombre_zona = f"Región {sel}"
                    # Al no tener un mapa de polígonos de regiones, traemos los puntos de las estaciones
                    # de esa región para definir la ubicación espacial.
                    q_geo = text(f"SELECT * FROM estaciones WHERE subregion = '{sel}'")
                    df_pts = pd.read_sql(q_geo, engine)
                    
                    if not df_pts.empty:
                        # Convertimos a GeoDataFrame para que sea compatible con el resto del código
                        gdf_zona = gpd.GeoDataFrame(
                            df_pts, 
                            geometry=gpd.points_from_xy(df_pts.longitud, df_pts.latitud),
                            crs="EPSG:4326"
                        )
                    else:
                        st.sidebar.warning(f"No hay estaciones en {sel}")

            except Exception as e:
                st.sidebar.warning(f"Error cargando regiones: {e}")        

        # --- B. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            try:
                gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
                
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
            except:
                st.sidebar.warning("Error en tabla municipios.")

        # --- C. DEPARTAMENTO ---
        else:
            from shapely.geometry import box
            gdf_zona = gpd.GeoDataFrame(
                {'nombre': ['Antioquia']}, 
                geometry=[box(-77.5, 5.0, -73.5, 9.0)], 
                crs="EPSG:4326"
            )

        # --- 2. FILTRAR ESTACIONES ---
        ids_estaciones = []
        if gdf_zona is not None and not gdf_zona.empty:
            
            # Auto-corrección de CRS (Metros a Grados)
            if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                 gdf_zona = gdf_zona.to_crs("EPSG:4326")
            
            # Slider de Buffer
            buff_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 5)
            buff_deg = buff_km / 111.0 
            
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            
            # Consulta SQL optimizada
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
                
                # Spatial Join
                zona_buffered = gdf_zona.copy()
                if buff_deg > 0:
                    zona_buffered['geometry'] = zona_buffered.geometry.buffer(buff_deg)
                
                est_in = gdf_ptos[gdf_ptos.geometry.within(zona_buffered.unary_union)]
                
                if not est_in.empty:
                    ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                    altitud_ref = est_in['altitud'].mean()
                    st.sidebar.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                else:
                    st.sidebar.warning("0 estaciones en el área exacta.")
            else:
                st.sidebar.warning("0 estaciones en el cuadrante.")

    except Exception as e:
        st.sidebar.error(f"Error selector: {e}")
        

    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
