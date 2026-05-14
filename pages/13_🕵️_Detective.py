# pages/13_🕵️_Detective.py

import os
import sys
import json
import re

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
from sqlalchemy import create_engine, text

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Centro de Diagnóstico", page_icon="🕵️", layout="wide")

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.db_manager import get_engine
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    try:
        from modules.db_manager import get_engine
    except ImportError:
        # Último recurso si falla la base de datos
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])

st.subheader("🕵️ Radiografía de Matrices (Borrar después de usar)")

try:
    engine_sql = get_engine()
    
    q = text("""
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name IN ('matriz_maestra_demografica', 'matriz_maestra_pecuaria', 'matriz_hidrologica_maestra')
    """)
    
    df_esquema = pd.read_sql(q, engine_sql)
    st.dataframe(df_esquema, use_container_width=True)
    
except Exception as e:
    st.error(f"Error conectando a la BD: {e}")

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Detective")

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "Agua2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "AdminPoter"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo ANTES de mostrar el contenido
muro_de_acceso_beta()

# ==============================================================================
# --- CONTENIDO DE LA PÁGINA (SOLO VISIBLE SI PASAN EL MURO) ---
# ==============================================================================
st.title("🕵️ Centro de Diagnóstico y Detective")
st.markdown("Herramientas forenses para administrador: Evaluación de coordenadas, proyecciones espaciales y auditoría de la base de datos.")
st.divider()

engine = get_engine()

# --- PESTAÑAS PARA ORGANIZAR TODO EL SISTEMA FORENSE ---
tab_coord, tab_dem, tab_bd = st.tabs([
    "🏥 Salud de Coordenadas (Estaciones)", 
    "⛰️ Diagnóstico DEM vs Cuencas", 
    "🔍 Explorador de Tablas (BD)"
])

# ==============================================================================
# TAB 1: BÚSQUEDA DE COORDENADAS (ANTIGUA PÁGINA 13)
# ==============================================================================
with tab_coord:
    st.header("🏥 Análisis de Integridad Espacial de Estaciones")
    
    st.subheader("1. Conteo de Salud")
    try:
        # Contamos cuántas tienen coordenadas y cuántas no
        df_count = pd.read_sql("""
            SELECT 
                COUNT(*) as total,
                COUNT(latitud) as con_latitud,
                COUNT(longitud) as con_longitud
            FROM estaciones
        """, engine)
        
        total = df_count.iloc[0]['total']
        validas = df_count.iloc[0]['con_latitud']
        
        c1, c2 = st.columns(2)
        c1.metric("Total Estaciones en BD", total)
        c2.metric("Con Coordenadas Válidas", validas)
        
        if validas == 0:
            st.error("🚨 ¡CERO! Ninguna estación tiene coordenadas en las columnas 'latitud'/'longitud'.")
        elif validas < total:
            st.warning(f"⚠️ Hay {total - validas} estaciones huérfanas sin coordenadas.")
        else:
            st.success(f"✅ Excelente. Las {validas} estaciones tienen coordenadas.")

    except Exception as e:
        st.error(str(e))

    st.subheader("2. Inspección de Columnas (Vista Cruda)")
    try:
        # Traemos las primeras 5 filas COMPLETAS
        df_all = pd.read_sql("SELECT * FROM estaciones LIMIT 5", engine)
        st.write("Verifica si tus coordenadas están ocultas en alguna columna con otro nombre:")
        st.dataframe(df_all)
        
        cols = df_all.columns.tolist()
        st.caption(f"**Columnas detectadas:** {cols}")

    except Exception as e:
        st.error(str(e))

# ==============================================================================
# TAB 2: DIAGNÓSTICO DEM vs CUENCAS
# ==============================================================================
with tab_dem:
    st.header("🗺️ Detective Espacial: Conflicto de Proyecciones")
    st.info("Verifica si el archivo DEM y las Cuencas en BD están 'viviendo' en el mismo sistema de coordenadas para evitar mapas en blanco.")
    
    PATH_DEM = "data/DemAntioquia_EPSG3116.tif"

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Análisis del DEM (Raster)")
        try:
            if not rasterio:
                st.error("Librería rasterio no instalada.")
            else:
                try:
                    with rasterio.open(PATH_DEM) as src:
                        st.success(f"✅ DEM Cargado: {PATH_DEM}")
                        dem_crs = src.crs
                        dem_bounds = src.bounds
                        st.code(f"CRS DEM:\n{dem_crs}\n\nLímites:\nIzquierda: {dem_bounds.left:,.0f}\nAbajo:     {dem_bounds.bottom:,.0f}\nDerecha:   {dem_bounds.right:,.0f}\nArriba:    {dem_bounds.top:,.0f}")
                        
                        # Diagnóstico de Origen
                        if dem_bounds.left > 4000000:
                            st.info("ℹ️ TIPO: MAGNA ORIGEN NACIONAL (CTM12)")
                        elif dem_bounds.left > 800000:
                            st.info("ℹ️ TIPO: MAGNA BOGOTÁ (EPSG:3116)")
                        else:
                            st.info("ℹ️ TIPO: Probablemente Grados (WGS84)")
                except FileNotFoundError:
                    st.error(f"❌ No se encontró el archivo: {PATH_DEM}")
        except Exception as e:
            st.error(f"Error analizando DEM: {e}")

    with c2:
        st.subheader("2. Análisis de Cuenca (Vectorial)")
        try:
            gdf_test = gpd.read_postgis("SELECT * FROM cuencas LIMIT 1", engine, geom_col="geometry")
            
            if not gdf_test.empty:
                st.success(f"✅ Cuenca cargada: {gdf_test.iloc[0].get('nombre_cuenca', gdf_test.iloc[0].get('subc_lbl', 'Sin Nombre'))}")
                st.write(f"**CRS Original en BD:** {gdf_test.crs}")
                
                if 'dem_crs' in locals():
                    try:
                        gdf_reproj = gdf_test.to_crs(dem_crs)
                        poly_bounds = gdf_reproj.total_bounds
                        st.code(f"Límites Cuenca (Reproyectada al CRS del DEM):\nIzquierda: {poly_bounds[0]:,.0f}\nAbajo:     {poly_bounds[1]:,.0f}\nDerecha:   {poly_bounds[2]:,.0f}\nArriba:    {poly_bounds[3]:,.0f}")
                        
                        dem_box = box(*dem_bounds)
                        cuenca_box = box(*poly_bounds)
                        
                        if dem_box.intersects(cuenca_box):
                            st.success("🎉 ¡HAY INTERSECCIÓN! Los datos se tocan físicamente.")
                        else:
                            st.error("❌ NO SE TOCAN. Están en lugares diferentes.")
                            
                            dist_x = abs(dem_bounds.left - poly_bounds[0])
                            st.write(f"Distancia en X entre ellos: {dist_x:,.0f} metros")
                            if 3500000 < dist_x < 4500000:
                                st.error("⚠️ La diferencia es ~4,000,000m. Conflicto Origen Nacional vs Bogotá confirmado.")
                    except Exception as e:
                        st.error(f"Error reproyectando: {e}")
            else:
                st.warning("La tabla 'cuencas' está vacía.")

        except Exception as e:
            st.error(f"Error consultando Cuenca: {e}")

# ==============================================================================
# TAB 3: EXPLORADOR BD (MODO FORENSE)
# ==============================================================================
with tab_bd:
    st.header("🔍 Explorador de Tablas de la Base de Datos")
    
    with st.container():
        try:
            with engine.connect() as conn:
                tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", conn)
                
                if not tables.empty:
                    table_list = tables['table_name'].tolist()
                    selected_table = st.selectbox("Selecciona la tabla a investigar:", table_list)
                else:
                    st.error("No se encontraron tablas en la base de datos.")
                    selected_table = None
        except Exception as e:
            st.error(f"Error conectando a BD: {e}")
            selected_table = None

    if selected_table:
        st.markdown(f"### 🔬 Analizando: `{selected_table}`")
        
        try:
            with engine.connect() as conn:
                count = pd.read_sql(text(f"SELECT count(*) as total FROM {selected_table}"), conn).iloc[0]['total']
                cols_df = pd.read_sql(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{selected_table}'"), conn)
                
                c1, c2 = st.columns(2)
                c1.metric("Filas Totales", count)
                with c2:
                    with st.expander("Ver Columnas y Tipos de Dato"):
                        st.dataframe(cols_df, hide_index=True)

                st.markdown("#### 🌍 Auditoría de Geometría")
                geom_col = "geom" 
                if "geometry" in cols_df['column_name'].values: geom_col = "geometry"
                
                if geom_col in cols_df['column_name'].values:
                    try:
                        q_geo = text(f"""
                            SELECT 
                                ST_SRID({geom_col}) as srid_detectado, 
                                ST_AsText({geom_col}) as ejemplo_coordenada,
                                ST_IsValid({geom_col}) as es_valido
                            FROM {selected_table} 
                            WHERE {geom_col} IS NOT NULL LIMIT 1
                        """)
                        geo_sample = pd.read_sql(q_geo, conn)
                        
                        if not geo_sample.empty:
                            geo_sample = geo_sample.iloc[0]
                            st.write("**Sistema de Referencia (SRID) en BD:**", f"`{geo_sample['srid_detectado']}`")
                            st.write("**Ejemplo de Coordenada:**", f"`{geo_sample['ejemplo_coordenada']}`")
                            
                            coord_text = str(geo_sample['ejemplo_coordenada'])
                            if "POINT" in coord_text or "POLYGON" in coord_text:
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", coord_text)
                                if nums:
                                    first_num = float(nums[0])
                                    if abs(first_num) <= 180:
                                        st.success("✅ **DIAGNÓSTICO:** Las coordenadas parecen ser **GRADOS (Lat/Lon - WGS84)**.")
                                    else:
                                        st.error(f"🚨 **DIAGNÓSTICO:** Las coordenadas parecen ser **METROS** (Ej. {first_num:,.0f}).")
                        else:
                            st.warning("La columna geométrica está vacía en todos los registros.")
                    except Exception as e:
                        st.warning(f"No se pudo analizar la geometría: {e}")
                else:
                    st.info("Esta tabla no parece tener columna espacial.")

                st.markdown("#### 📄 Vista Previa de Datos Crudos (Primeras 7 filas)")
                cols_safe = [c for c in cols_df['column_name'] if c != geom_col]
                cols_query = ", ".join([f'"{c}"' for c in cols_safe])
                
                try:
                    df_preview = pd.read_sql(text(f"SELECT {cols_query} FROM {selected_table} LIMIT 7"), conn)
                    st.dataframe(df_preview)
                except Exception as e:
                    st.error(f"Error cargando vista previa: {e}")
        except Exception as e:
             st.error(f"Error en análisis de tabla: {e}")

    st.markdown("---")
    st.subheader("🛠️ Consola SQL Manual")
    query = st.text_area("Ejecutar SQL personalizado:", f"SELECT * FROM {selected_table if selected_table else 'cuencas'} LIMIT 10")
    if st.button("Ejecutar Query"):
        with engine.connect() as conn:
            try:
                res = pd.read_sql(text(query), conn)
                st.dataframe(res)
            except Exception as e:
                st.error(f"Error SQL: {e}")
