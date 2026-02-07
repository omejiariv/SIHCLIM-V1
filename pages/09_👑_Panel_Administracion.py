# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import time
import os
import sys
import tempfile
import zipfile
import geopandas as gpd
from sqlalchemy import text
import shutil

# --- 1. CONFIGURACIÓN E IMPORTACIONES ---
st.set_page_config(page_title="Panel de Administración", page_icon="👑", layout="wide")

try:
    from modules.db_manager import get_engine
except ImportError:
    st.error("⚠️ Error: No se encontró el módulo db_manager.")
    st.stop()

# Importación opcional de utilidades de nube
try:
    from modules.admin_utils import get_raster_list, upload_raster_to_storage, delete_raster_from_storage
except ImportError:
    get_raster_list, upload_raster_to_storage, delete_raster_from_storage = None, None, None

# --- 2. GESTIÓN DE CONEXIÓN ---
def get_connection():
    engine = get_engine()
    try: engine.dispose()
    except: pass
    return engine

engine = get_connection()

# --- 3. FUNCIONES AUXILIARES DE LIMPIEZA Y CARGA ---

def limpiar_estaciones(df):
    """Normaliza el CSV de estaciones."""
    df.columns = df.columns.str.lower().str.strip()
    mapa = {
        'id_estacio': 'id_estacion', 'codigo': 'id_estacion', 'code': 'id_estacion',
        'nom_est': 'nombre', 'station': 'nombre', 'name': 'nombre',
        'longitud_geo': 'longitud', 'lon': 'longitud', 'lng': 'longitud',
        'latitud_geo': 'latitud', 'lat': 'latitud',
        'alt_est': 'altitud', 'elev': 'altitud'
    }
    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})
    
    req = ['id_estacion', 'latitud', 'longitud']
    if not all(c in df.columns for c in req):
        return None, f"Faltan columnas obligatorias: {req}"
    
    df['id_estacion'] = df['id_estacion'].astype(str).str.strip()
    for c in ['latitud', 'longitud', 'altitud']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
    
    return df.dropna(subset=['latitud', 'longitud']), "OK"

def cargar_capa_gis_robusta(uploaded_file, nombre_tabla, engine):
    """Carga archivos GIS (ZIP/GeoJSON) a PostGIS."""
    if uploaded_file is None: return
    status = st.status(f"🚀 Procesando {nombre_tabla}...", expanded=True)
    try:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        gdf = None
        if suffix == '.zip':
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.endswith(".shp"):
                            gdf = gpd.read_file(os.path.join(root, file))
                            break
        else:
            gdf = gpd.read_file(tmp_path)
            
        if gdf is None:
            status.error("No se pudo leer el archivo geográfico.")
            return

        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            status.write("🔄 Reproyectando a WGS84...")
            gdf = gdf.to_crs("EPSG:4326")
        
        gdf.columns = [c.lower() for c in gdf.columns]
        
        # Mapeos específicos
        if 'bocatomas' in nombre_tabla and 'nombre' in gdf.columns: gdf = gdf.rename(columns={'nombre': 'nom_bocatoma'})
        
        status.write("📤 Subiendo a Base de Datos...")
        gdf.to_postgis(nombre_tabla, engine, if_exists='replace', index=False)
        
        status.update(label="¡Carga Exitosa!", state="complete", expanded=False)
        st.success(f"✅ {nombre_tabla}: {len(gdf)} registros cargados.")
        if len(gdf) > 0: st.balloons()
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error crítico: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def visor_tabla_simple(nombre_tabla):
    """Muestra una tabla editable básica."""
    try:
        # Excluir geometría para que sea rápido
        q = text(f"SELECT * FROM {nombre_tabla} LIMIT 1000")
        try:
            df = pd.read_sql(q, engine)
            if 'geom' in df.columns: df = df.drop(columns=['geom'])
            if 'geometry' in df.columns: df = df.drop(columns=['geometry'])
            
            st.info(f"Mostrando primeros 1.000 registros de **{nombre_tabla}**.")
            st.dataframe(df, use_container_width=True)
        except:
            st.warning(f"La tabla '{nombre_tabla}' está vacía o no existe.")
    except Exception as e:
        st.error(f"Error leyendo tabla: {e}")

# --- 4. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración Integral")
st.markdown("---")

tabs = st.tabs([
    "📡 Estaciones", "🌧️ Lluvia", "📊 Índices", 
    "🏠 Predios", "🌊 Cuencas", "🏙️ Municipios", "☁️ Coberturas", 
    "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", 
    "🛠️ SQL", "📚 Inventario", "🔥 ZONA PELIGRO"
])

# ==============================================================================
# TAB 0: ESTACIONES (CORREGIDO - GUARDA METADATOS)
# ==============================================================================
with tabs[0]:
    st.header("📍 Catálogo de Estaciones")
    t1, t2 = st.tabs(["👁️ Ver Datos", "🚀 Cargar (Reparador)"])
    
    with t1:
        if st.button("🔄 Refrescar Tabla", key="ref_est"):
            st.rerun()
        visor_tabla_simple("estaciones")
            
    with t2:
        st.info("Sube `mapaCVENSO.csv`. El sistema guardará: Municipio, Departamento, Subregión, etc.")
        up = st.file_uploader("CSV Estaciones", type=["csv"], key="up_est_final")
        
        if up:
            try:
                # 1. Lectura
                try: df_raw = pd.read_csv(up, sep=';')
                except: up.seek(0); df_raw = pd.read_csv(up, sep=',')
                
                # 2. Limpieza y Mapeo Extendido
                df_raw.columns = df_raw.columns.str.lower().str.strip()
                
                # Mapeo de columnas esenciales y metadatos
                mapa = {
                    'id_estacio': 'id_estacion', 'codigo': 'id_estacion',
                    'nom_est': 'nombre', 'station': 'nombre',
                    'longitud_geo': 'longitud', 'lon': 'longitud',
                    'latitud_geo': 'latitud', 'lat': 'latitud',
                    'alt_est': 'altitud', 'elev': 'altitud',
                    # Metadatos
                    'mun_nomb': 'municipio', 'municipio': 'municipio',
                    'depto': 'departamento', 'departamento': 'departamento',
                    'subregion': 'subregion', 'corriente': 'corriente'
                }
                df_clean = df_raw.rename(columns={k: v for k, v in mapa.items() if k in df_raw.columns})
                
                # 3. Validación de Geometría
                req = ['id_estacion', 'latitud', 'longitud']
                if not all(c in df_clean.columns for c in req):
                    st.error(f"Faltan columnas obligatorias: {req}")
                else:
                    # Limpieza numérica
                    df_clean['id_estacion'] = df_clean['id_estacion'].astype(str).str.strip()
                    for c in ['latitud', 'longitud', 'altitud']:
                        if c in df_clean.columns:
                            df_clean[c] = pd.to_numeric(df_clean[c].astype(str).str.replace(',', '.'), errors='coerce')
                    
                    df_clean = df_clean.dropna(subset=['latitud', 'longitud'])
                    
                    # Rellenar columnas faltantes con texto vacío para que SQL no falle
                    for col in ['municipio', 'departamento', 'subregion', 'corriente']:
                        if col not in df_clean.columns:
                            df_clean[col] = None

                    st.success(f"✅ Datos válidos: {len(df_clean)} estaciones.")
                    st.dataframe(df_clean.head(), use_container_width=True)
                    
                    if st.button("💾 CONFIRMAR CARGA COMPLETA", type="primary", key="btn_conf_est_full"):
                        with st.status("Guardando con metadatos...", expanded=True):
                            with engine.connect() as conn:
                                trans = conn.begin()
                                try:
                                    # A. Cargar a temporal
                                    # Filtramos solo las columnas que nos interesan para no subir basura
                                    cols_to_db = ['id_estacion', 'nombre', 'latitud', 'longitud', 'altitud', 'municipio', 'departamento', 'subregion', 'corriente']
                                    # Aseguramos que existan en el df (aunque sean None)
                                    df_final = df_clean[[c for c in cols_to_db if c in df_clean.columns]]
                                    
                                    df_final.to_sql('temp_est', conn, if_exists='replace', index=False)
                                    
                                    # B. Insertar/Actualizar con TODOS los campos
                                    conn.execute(text("""
                                        INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente)
                                        SELECT id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente 
                                        FROM temp_est
                                        ON CONFLICT (id_estacion) DO UPDATE SET
                                            nombre = EXCLUDED.nombre, 
                                            latitud = EXCLUDED.latitud,
                                            longitud = EXCLUDED.longitud, 
                                            altitud = EXCLUDED.altitud,
                                            municipio = EXCLUDED.municipio,
                                            departamento = EXCLUDED.departamento,
                                            subregion = EXCLUDED.subregion,
                                            corriente = EXCLUDED.corriente;
                                    """))
                                    
                                    # C. Geometría
                                    conn.execute(text("ALTER TABLE estaciones ADD COLUMN IF NOT EXISTS geom GEOMETRY(POINT, 4326);"))
                                    conn.execute(text("UPDATE estaciones SET geom = ST_SetSRID(ST_MakePoint(longitud, latitud), 4326) WHERE longitud IS NOT NULL"))
                                    
                                    conn.execute(text("DROP TABLE IF EXISTS temp_est"))
                                    trans.commit()
                                    st.success("✅ Estaciones guardadas con todos sus detalles.")
                                    st.balloons()
                                    time.sleep(2)
                                    st.rerun()
                                except Exception as e:
                                    trans.rollback()
                                    st.error(f"Error SQL: {e}")
            except Exception as e: st.error(f"Error archivo: {e}")

# ==============================================================================
# TAB 1: LLUVIA (RECUPERADO EL VISOR DE DATOS)
# ==============================================================================
with tabs[1]:
    st.header("🌧️ Gestión de Lluvia")
    t1, t2 = st.tabs(["🔍 Explorador de Datos", "🚀 Carga Masiva"])
    
    with t1:
        st.info("Consulta los datos cargados por Estación y Año.")
        try:
            # Selector de Estación
            estaciones = pd.read_sql("SELECT id_estacion, nombre FROM estaciones ORDER BY id_estacion", engine)
            if not estaciones.empty:
                opciones = estaciones.apply(lambda x: f"{x['id_estacion']} - {x['nombre']}", axis=1)
                sel_est_full = st.selectbox("Seleccionar Estación:", opciones)
                id_sel = sel_est_full.split(" - ")[0]
                
                # Selector de Año (basado en datos reales)
                anios = pd.read_sql(f"SELECT DISTINCT EXTRACT(YEAR FROM fecha)::int as anio FROM precipitacion WHERE id_estacion='{id_sel}' ORDER BY anio DESC", engine)
                
                if not anios.empty:
                    anio_sel = st.selectbox("Seleccionar Año:", anios['anio'])
                    
                    # Mostrar Datos
                    q_data = f"SELECT fecha, valor FROM precipitacion WHERE id_estacion='{id_sel}' AND EXTRACT(YEAR FROM fecha)={anio_sel} ORDER BY fecha"
                    df_data = pd.read_sql(q_data, engine)
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.dataframe(df_data, use_container_width=True, height=400)
                    with c2:
                        if not df_data.empty:
                            st.line_chart(df_data.set_index('fecha'))
                else:
                    st.warning("Esta estación no tiene datos de lluvia cargados.")
            else:
                st.warning("Carga primero las estaciones.")
        except Exception as e: st.error(f"Error en explorador: {e}")

    with t2:
        st.info("Carga la matriz `DatosPptnmes_ENSO.csv`.")
        up_rain = st.file_uploader("Matriz de Lluvia", type=["csv"], key="up_rain_final")
        
        if up_rain and st.button("🚀 Procesar Lluvia", key="btn_rain_proc"):
            status = st.status("Procesando...", expanded=True)
            try:
                df = pd.read_csv(up_rain, sep=';', decimal=',')
                if 'Fecha' in df.columns: df = df.rename(columns={'Fecha': 'fecha'})
                
                df_long = df.melt(id_vars=['fecha'], var_name='id_estacion', value_name='valor')
                df_long['fecha'] = pd.to_datetime(df_long['fecha'], errors='coerce')
                df_long['valor'] = pd.to_numeric(df_long['valor'], errors='coerce')
                df_long = df_long.dropna(subset=['fecha', 'valor'])
                df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
                
                status.write(f"📦 Cargando {len(df_long)} registros...")
                
                chunk = 10000
                with engine.connect() as conn:
                    # Crear estaciones faltantes si las hay
                    ids = pd.DataFrame({'id_estacion': df_long['id_estacion'].unique()})
                    ids['nombre'] = 'Estación ' + ids['id_estacion']
                    ids.to_sql('temp_ids', conn, if_exists='replace', index=False)
                    conn.execute(text("INSERT INTO estaciones (id_estacion, nombre) SELECT id_estacion, nombre FROM temp_ids ON CONFLICT (id_estacion) DO NOTHING"))
                    conn.commit()
                    
                    for i in range(0, len(df_long), chunk):
                        df_long.iloc[i:i+chunk].to_sql('precipitacion', conn, if_exists='append', index=False, method='multi')
                
                status.update(label="✅ Carga Completa", state="complete")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: ÍNDICES (CON VISOR)
# ==============================================================================
with tabs[2]:
    st.header("📊 Índices Climáticos")
    t1, t2 = st.tabs(["👁️ Ver Datos", "🚀 Cargar CSV"])
    
    with t1:
        if st.button("🔄 Refrescar", key="ref_ind"): st.rerun()
        visor_tabla_simple("indices_climaticos")
        
    with t2:
        up_ind = st.file_uploader("`Indices_Globales.csv`", type=["csv"], key="up_ind_final")
        if up_ind and st.button("Cargar Índices", key="btn_ind"):
            try:
                df = pd.read_csv(up_ind, sep=None, engine='python', encoding='utf-8-sig')
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                st.success("✅ Índices cargados.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TABS GIS (ESTANDARIZADOS CON VISOR)
# ==============================================================================
def render_gis_tab(title, table_name, file_key, btn_key):
    st.header(title)
    t1, t2 = st.tabs(["👁️ Ver Tabla", "📂 Cargar Archivo"])
    with t1:
        visor_tabla_simple(table_name)
    with t2:
        f = st.file_uploader("GeoJSON/ZIP", key=file_key)
        if st.button("Cargar Capa", key=btn_key): cargar_capa_gis_robusta(f, table_name, engine)

with tabs[3]: render_gis_tab("🏠 Predios", "predios", "up_pred", "btn_pred")
with tabs[4]: render_gis_tab("🌊 Cuencas", "cuencas", "up_cuen", "btn_cuen")
with tabs[5]: render_gis_tab("🏙️ Municipios", "municipios", "up_mun", "btn_mun")

# ==============================================================================
# ==============================================================================
# TAB 6: COBERTURAS (RASTERS) - MEJORADA CON LISTA
# ==============================================================================
with tabs[6]:
    st.header("☁️ Rasters & Coberturas")
    st.info("Aquí se almacenan las imágenes satelitales (GeoTIFF) usadas para los modelos hidrológicos.")
    
    c1, c2 = st.columns(2)
    
    # --- COLUMNA IZQUIERDA: LISTA DE ARCHIVOS ---
    with c1:
        st.subheader("📂 Archivos en la Nube")
        if get_raster_list:
            with st.spinner("Consultando inventario..."):
                try:
                    files = get_raster_list() # Devuelve lista de diccionarios
                    if files:
                        # Convertimos a DataFrame para verlo bonito
                        df_files = pd.DataFrame(files)
                        
                        # Seleccionamos y renombramos columnas útiles si existen
                        cols_show = []
                        if 'name' in df_files.columns: cols_show.append('name')
                        if 'created_at' in df_files.columns: cols_show.append('created_at')
                        if 'metadata' in df_files.columns: # A veces el tamaño viene aquí
                            df_files['size_mb'] = df_files['metadata'].apply(lambda x: round(x.get('size', 0)/1024/1024, 2) if x else 0)
                            cols_show.append('size_mb')
                        
                        st.dataframe(
                            df_files[cols_show], 
                            column_config={
                                "name": "Nombre Archivo",
                                "created_at": st.column_config.DatetimeColumn("Fecha Carga", format="D MMM YYYY, HH:mm"),
                                "size_mb": st.column_config.NumberColumn("Tamaño (MB)")
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("⚠️ El bucket de almacenamiento está vacío.")
                except Exception as e:
                    st.error(f"Error conectando al almacenamiento: {e}")
        else:
            st.error("Módulo de conexión a Storage no disponible.")
    
    # --- COLUMNA DERECHA: SUBIDA ---
    with c2:
        st.subheader("⬆️ Subir Nuevo Raster")
        st.markdown("""
        **Archivos Requeridos:**
        * ✅ `Cob25m_WGS84.tif` (Coberturas)
        * ✅ `DemAntioquia_EPSG3116.tif` (Elevación)
        * ✅ `PPAM.tif` (Lluvia Media - Opcional)
        """)
        
        f = st.file_uploader("Seleccionar Archivo .tif", type=["tif", "tiff"], key="up_raster_main")
        
        if f:
            # Botón con Key única
            if st.button(f"Subir {f.name}", key="btn_up_raster_cloud") and upload_raster_to_storage:
                with st.spinner(f"Subiendo {f.name} a la nube..."):
                    bytes_data = f.getvalue()
                    ok, msg = upload_raster_to_storage(bytes_data, f.name)
                    if ok: 
                        st.success(f"✅ {msg}")
                        time.sleep(2)
                        st.rerun() # Recargar para ver el archivo en la lista de la izquierda
                    else: 
                        st.error(f"❌ {msg}")

# ==============================================================================
# OTRAS CAPAS (CONSERVADAS)
# ==============================================================================
with tabs[7]: render_gis_tab("💧 Bocatomas", "bocatomas", "up_boca", "btn_boca")
with tabs[8]: render_gis_tab("⛰️ Hidrogeología", "zonas_hidrogeologicas", "up_hidro", "btn_hidro")
with tabs[9]: render_gis_tab("🌱 Suelos", "suelos", "up_suelo", "btn_suelo")

# ==============================================================================
# TAB 11: INVENTARIO (COMPLETO)
# ==============================================================================
with tabs[11]:
    st.header("📚 Inventario de Archivos")
    st.markdown("Lista maestra de todos los insumos requeridos por el sistema.")
    
    data = [
        {"Archivo": "mapaCVENSO.csv", "Tipo": "CSV", "Uso": "Catálogo maestro de estaciones (Ubicación)."},
        {"Archivo": "DatosPptnmes_ENSO.csv", "Tipo": "CSV", "Uso": "Matriz histórica de lluvias (Filas=Fechas, Cols=Estaciones)."},
        {"Archivo": "Indices_Globales.csv", "Tipo": "CSV", "Uso": "Índices ONI, SOI para pronósticos climáticos."},
        {"Archivo": "PrediosEjecutados.geojson", "Tipo": "Vector", "Uso": "Polígonos de predios para consulta local."},
        {"Archivo": "SubcuencasAinfluencia.geojson", "Tipo": "Vector", "Uso": "Límites de cuencas hidrográficas."},
        {"Archivo": "Municipios.geojson", "Tipo": "Vector", "Uso": "División política administrativa."},
        {"Archivo": "Cob25m_WGS84.tif", "Tipo": "Raster", "Uso": "Mapa de coberturas del suelo (Bosque, Cultivos, etc.)."},
        {"Archivo": "DemAntioquia_EPSG3116.tif", "Tipo": "Raster", "Uso": "Modelo Digital de Elevación (Altitud/Pendiente)."},
        {"Archivo": "Bocatomas.zip", "Tipo": "Shapefile", "Uso": "Puntos de captación de agua."},
        {"Archivo": "Hidrogeologia.zip", "Tipo": "Shapefile", "Uso": "Unidades de acuíferos y permeabilidad."},
        {"Archivo": "Suelos.zip", "Tipo": "Shapefile", "Uso": "Tipos de suelo y texturas."}
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True)

# ==============================================================================
# TAB 12: ZONA DE PELIGRO (EXPLICADA)
# ==============================================================================
with tabs[12]:
    st.header("🔥 Zona de Peligro")
    st.error("""
    **¿QUÉ ES ESTO?**
    Esta sección contiene controles administrativos de alto nivel que afectan la estructura misma de la base de datos.
    
    **¿POR QUÉ ES PELIGROSA?**
    El botón de abajo ejecuta un `DROP CASCADE`. Esto significa que borra físicamente las tablas de la base de datos y las crea desde cero (vacías).
    
    **¿CUÁNDO USARLA?**
    * Solo cuando la base de datos esté corrupta.
    * Cuando quieras empezar el proyecto totalmente de cero.
    * Si cambiaste la estructura de columnas y necesitas regenerar todo.
    """)
    
    with st.expander("💣 MOSTRAR BOTÓN DE RESET"):
        if st.button("EJECUTAR REINICIO DE FÁBRICA", key="btn_nuke", type="primary"):
            try:
                with engine.connect() as conn:
                    try: conn.rollback()
                    except: pass
                    
                    conn.execute(text("DROP TABLE IF EXISTS precipitacion CASCADE"))
                    conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE"))
                    conn.execute(text("DROP TABLE IF EXISTS indices_climaticos CASCADE"))
                    conn.commit()
                    
                    # Recrear Tablas
                    conn.execute(text("""
                        CREATE TABLE estaciones (
                            id_estacion TEXT PRIMARY KEY,
                            nombre TEXT, latitud FLOAT, longitud FLOAT, altitud FLOAT,
                            municipio TEXT, geom GEOMETRY(POINT, 4326)
                        )
                    """))
                    conn.execute(text("""
                        CREATE TABLE precipitacion (
                            fecha DATE, id_estacion TEXT, valor FLOAT,
                            PRIMARY KEY (fecha, id_estacion)
                        )
                    """))
                    conn.execute(text("""
                        CREATE TABLE indices_climaticos (
                            fecha DATE PRIMARY KEY, anomalia_oni FLOAT, soi FLOAT, iod FLOAT
                        )
                    """))
                    conn.commit()
                st.success("✅ Base de datos reiniciada. Ahora está vacía.")
                st.balloons()
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 10: SQL (HERRAMIENTA)
# ==============================================================================
with tabs[10]:
    st.header("🛠️ Consola SQL")
    st.info("Ejecuta consultas directas a la base de datos para diagnóstico.")
    q = st.text_area("Query:", value="SELECT count(*) FROM estaciones")
    if st.button("Ejecutar", key="btn_sql"):
        try:
            with engine.connect() as conn:
                if q.strip().lower().startswith("select"):
                    st.dataframe(pd.read_sql(text(q), conn))
                else:
                    conn.execute(text(q))
                    conn.commit()
                    st.success("Comando ejecutado.")
        except Exception as e: st.error(str(e))
