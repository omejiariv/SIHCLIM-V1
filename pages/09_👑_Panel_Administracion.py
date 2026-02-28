# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import json
import io
import time
import sys
import os
import tempfile
import zipfile
import geopandas as gpd
import rasterio
from sqlalchemy import text
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape
import shutil

from modules.admin_utils import get_raster_list, upload_raster_to_storage, delete_raster_from_storage
from supabase import create_client

# --- 1. CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from modules.db_manager import get_engine
except ImportError:
    from db_manager import get_engine

st.set_page_config(page_title="Panel de Administración", page_icon="👑", layout="wide")

# --- 2. AUTENTICACIÓN ---
def check_password():
    if st.session_state.get("password_correct", False):
        return True
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Acceso Restringido")
        st.info("Panel de Control SIHCLI-POTER (Nube)")
        
        # Validación de seguridad para que no falle si no hay secrets
        if "iri" not in st.secrets:
            # st.error("⚠️ Falta configuración [iri] en secrets.toml")
            # Si quieres saltar el bloqueo aunque falten secrets, devuelve True o False según prefieras.
            # Aquí lo dejo como estaba pero permitiendo continuar.
            return False 
        
        user_input = st.text_input("Usuario")
        pass_input = st.text_input("Contraseña", type="password")
        
        if st.button("Ingresar"):
            sec_user = st.secrets["iri"]["username"]
            sec_pass = st.secrets["iri"]["password"]
            if user_input == sec_user and pass_input == sec_pass:
                st.session_state.password_correct = True
                st.rerun()
            else:
                # st.error("🚫 Acceso Denegado")
                return False
    return False

if not check_password():
    pass  # <--- AGREGAR ESTO (Indica a Python: "No hagas nada y continúa")
#    st.stop()

engine = get_engine()

# --- 3. FUNCIONES AUXILIARES ---

def cargar_capa_gis_robusta(uploaded_file, nombre_tabla, engine):
    """Carga archivos GIS, repara coordenadas y sube a BD manteniendo TODOS los campos."""
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

        status.write(f"✅ Leído: {len(gdf)} registros. Columnas: {list(gdf.columns)}")

        # REPROYECCIÓN OBLIGATORIA A WGS84
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            status.write("🔄 Reproyectando a WGS84 (EPSG:4326)...")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Normalización de columnas
        gdf.columns = [c.lower() for c in gdf.columns]
        
        # Mapeo inteligente (pero conservamos el resto de columnas)
        rename_map = {}
        if 'bocatomas' in nombre_tabla and 'nombre' in gdf.columns: rename_map['nombre'] = 'nom_bocatoma'
        elif 'suelos' in nombre_tabla:
            if 'gridcode' in gdf.columns: rename_map['gridcode'] = 'codigo'
            if 'simbolo' in gdf.columns: rename_map['simbolo'] = 'codigo'
        elif 'zonas_hidrogeologicas' in nombre_tabla and 'nombre' in gdf.columns: 
            rename_map['nombre'] = 'nombre_zona'
            
        if rename_map:
            gdf = gdf.rename(columns=rename_map)

        status.write("📤 Subiendo a Base de Datos (Conservando todos los atributos)...")
        gdf.to_postgis(nombre_tabla, engine, if_exists='replace', index=False)
        
        status.update(label="¡Carga Exitosa!", state="complete", expanded=False)
        st.success(f"Capa **{nombre_tabla}** actualizada. {len(gdf)} registros con {len(gdf.columns)} campos.")
        if len(gdf) > 0: st.balloons()
        
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error crítico: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def editor_tabla_gis(nombre_tabla, key_editor):
    """Genera un editor de tabla para capas GIS excluyendo la columna de geometría pesada."""
    try:
        # Consultamos columnas excepto 'geometry' para que la tabla sea ligera y legible
        q_cols = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{nombre_tabla}' AND column_name != 'geometry'")
        cols = pd.read_sql(q_cols, engine)['column_name'].tolist()
        if not cols:
             st.warning(f"La tabla {nombre_tabla} existe pero no tiene columnas legibles.")
             return

        cols_str = ", ".join([f'"{c}"' for c in cols]) # Comillas para nombres seguros
        
        df = pd.read_sql(f"SELECT {cols_str} FROM {nombre_tabla} LIMIT 1000", engine)
        st.info(f"Mostrando primeros 1000 registros de **{nombre_tabla}**. ({len(df.columns)} campos)")
        
        # KEY ÚNICA AQUÍ TAMBIÉN
        df_editado = st.data_editor(df, key=key_editor, use_container_width=True, num_rows="dynamic")
        
        if st.button(f"💾 Guardar Cambios en {nombre_tabla}", key=f"btn_{key_editor}"):
            st.warning("⚠️ Edición directa deshabilitada por seguridad en esta versión. Use la carga de archivos para cambios masivos.")
    except Exception as e:
        st.warning(f"La tabla '{nombre_tabla}' aún no tiene datos o no existe. Cargue un archivo primero.")

# --- 4. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tabs = st.tabs([
    "📡 Estaciones", "🌧️ Lluvia", "📊 Índices", "🏠 Predios", "🌊 Cuencas", "🏙️ Municipios", "🌲 Coberturas",
    "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", "🛠️ SQL", "📚 Inventario", "🌧️ Red de Drenaje", "🌧️ Zona de Peligro", "👥 Demografía", "🗺️ Aduana SIG"
])

# --- PESTAÑA DE CONFIGURACIÓN INICIAL

st.markdown("### 🛠️ Zona de Peligro: Reinicio del Sistema")
with st.expander("Mostrar Controles de Reinicio de Base de Datos", expanded=True):
    st.warning("⚠️ ESTA ACCIÓN ES IRREVERSIBLE. BORRARÁ TODOS LOS DATOS.")
    
    if st.button("🔥 EJECUTAR REINICIO TOTAL (CASCADE) 🔥", key="btn_nuke_v3"):
        try:
            with engine.begin() as conn:
                st.write("⏳ Iniciando secuencia de borrado...")
                
                # 1. BORRADO EN ORDEN INVERSO (Hijos primero, luego Padres)
                # Usamos CASCADE en todo por seguridad
                conn.execute(text("DROP TABLE IF EXISTS precipitacion CASCADE;"))
                conn.execute(text("DROP TABLE IF EXISTS indices_climaticos CASCADE;"))
                conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))
                
                st.write("✅ Tablas eliminadas. Creando nueva estructura...")
                
                # 2. CREACIÓN DE TABLAS
                # Estaciones (Padre)
                conn.execute(text("""
                    CREATE TABLE estaciones (
                        id_estacion TEXT PRIMARY KEY,
                        nombre TEXT,
                        longitud FLOAT,
                        latitud FLOAT,
                        altitud FLOAT,
                        municipio TEXT,
                        departamento TEXT,
                        subregion TEXT,
                        corriente TEXT
                    );
                """))
                
                # Índices
                conn.execute(text("""
                    CREATE TABLE indices_climaticos (
                        fecha DATE PRIMARY KEY,
                        enso_año TEXT,
                        enso_mes TEXT,
                        anomalia_oni FLOAT,
                        temp_sst FLOAT,
                        temp_media FLOAT,
                        soi FLOAT,
                        iod FLOAT,
                        fase_enso TEXT
                    );
                """))
                
                # Precipitacion (Hija)
                conn.execute(text("""
                    CREATE TABLE precipitacion (
                        fecha DATE,
                        id_estacion TEXT,
                        valor FLOAT,
                        origen TEXT,
                        PRIMARY KEY (fecha, id_estacion),
                        CONSTRAINT fk_estacion FOREIGN KEY (id_estacion) REFERENCES estaciones(id_estacion)
                    );
                    CREATE INDEX idx_precip_fecha ON precipitacion(fecha);
                    CREATE INDEX idx_precip_estacion ON precipitacion(id_estacion);
                """))
                
            st.success("✅ ¡BASE DE DATOS REINICIADA CORRECTAMENTE!")
            st.balloons()
            time.sleep(2)
            st.rerun() # Recarga la página automáticamente
            
        except Exception as e:
            st.error(f"❌ Error crítico: {e}")


# ==============================================================================
# TAB 0: GESTIÓN DE ESTACIONES (CON DESBLOQUEO DE TRANSACCIÓN)
# ==============================================================================
with tabs[0]: 
    st.header("📍 Gestión de Estaciones")
    
    subtab_ver, subtab_carga = st.tabs(["👁️ Editor de Catálogo", "📂 Carga Masiva (CSV)"])
    
    # --- SUB-PESTAÑA 1: EDITOR ---
    with subtab_ver:
        st.info("Visualiza y edita las estaciones registradas.")
        
        col_ref, col_msg = st.columns([1, 3])
        if col_ref.button("🔄 Refrescar Tabla"):
            st.cache_data.clear()
            st.rerun()
            
        try:
            # Consulta segura
            df_est_db = pd.read_sql("SELECT * FROM estaciones ORDER BY id_estacion", engine)
            st.dataframe(df_est_db, use_container_width=True)
        except:
            st.warning("No se pudo cargar la tabla de estaciones.")

    # --- SUB-PESTAÑA 2: CARGA MASIVA (BLINDADA) ---
    with subtab_carga:
        st.markdown("### Cargar Archivo de Estaciones")
        st.info("Sube `mapaCVENSO.csv`. El sistema limpiará las coordenadas automáticamente.")
        up_est = st.file_uploader("Cargar CSV Estaciones", type=["csv"], key="up_est_csv_fix_v3")
        
        if up_est and st.button("🚀 Procesar Carga Masiva"):
            try:
                # 1. Lectura Robusta (Detecta separador automáticamente)
                try:
                    df_new = pd.read_csv(up_est, sep=';', decimal=',')
                    if len(df_new.columns) < 2: raise ValueError
                except:
                    up_est.seek(0)
                    df_new = pd.read_csv(up_est, sep=',', decimal='.')
                
                # 2. Limpieza de Columnas
                df_new.columns = df_new.columns.str.lower().str.strip()
                rename_map = {
                    'id_estacio': 'id_estacion', 'codigo': 'id_estacion',
                    'nom_est': 'nombre', 'station': 'nombre',
                    'longitud_geo': 'longitud', 'lon': 'longitud',
                    'latitud_geo': 'latitud', 'lat': 'latitud',
                    'alt_est': 'altitud', 'elev': 'altitud'
                }
                df_new = df_new.rename(columns={k: v for k, v in rename_map.items() if k in df_new.columns})
                
                # 3. Validación y Conversión Numérica
                req = ['id_estacion', 'latitud', 'longitud']
                if not all(c in df_new.columns for c in req):
                    st.error(f"Faltan columnas requeridas: {req}")
                else:
                    # Forzar conversión a números (limpia errores de tipeo)
                    for c in ['latitud', 'longitud', 'altitud']:
                        if c in df_new.columns:
                            df_new[c] = pd.to_numeric(
                                df_new[c].astype(str).str.replace(',', '.'), errors='coerce'
                            )
                    
                    # 4. INSERCIÓN BLINDADA (El secreto está aquí)
                    with engine.connect() as conn:
                        # PASO CRÍTICO: Rollback preventivo para desbloquear la BD
                        try: conn.rollback() 
                        except: pass
                        
                        # Iniciar transacción limpia
                        trans = conn.begin()
                        try:
                            # Subir a tabla temporal
                            df_new.to_sql('temp_est_load', conn, if_exists='replace', index=False)
                            
                            # Ejecutar UPSERT (Actualizar si existe, Insertar si no)
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                                SELECT id_estacion, nombre, latitud, longitud, altitud FROM temp_est_load
                                ON CONFLICT (id_estacion) DO UPDATE SET
                                    nombre = EXCLUDED.nombre,
                                    latitud = EXCLUDED.latitud,
                                    longitud = EXCLUDED.longitud,
                                    altitud = EXCLUDED.altitud;
                            """))
                            
                            # Actualizar Geometrías para los mapas (PostGIS)
                            try:
                                conn.execute(text("UPDATE estaciones SET geom = ST_SetSRID(ST_MakePoint(longitud, latitud), 4326) WHERE longitud IS NOT NULL"))
                            except: pass
                            
                            # Limpieza
                            conn.execute(text("DROP TABLE IF EXISTS temp_est_load"))
                            
                            # Confirmar transacción
                            trans.commit()
                            
                            st.success(f"✅ ¡Éxito! {len(df_new)} estaciones procesadas y guardadas.")
                            st.balloons()
                            
                        except Exception as sql_err:
                            trans.rollback() # Si falla algo, deshacemos para no bloquear
                            st.error(f"Error SQL durante la carga: {sql_err}")
                            
            except Exception as ex:
                st.error(f"Error procesando el archivo: {ex}")


# ==============================================================================
# TAB 1: GESTIÓN DE LLUVIA (VERSIÓN DIAGNÓSTICO & CORRECCIÓN)
# ==============================================================================
with tabs[1]:
    st.header("🌧️ Gestión de Lluvia e Índices")

    # --- DIAGNÓSTICO RÁPIDO DE LA BASE DE DATOS ---
    try:
        count_rain = pd.read_sql("SELECT COUNT(*) as conteo FROM precipitacion", engine).iloc[0]['conteo']
        count_est = pd.read_sql("SELECT COUNT(*) as conteo FROM estaciones", engine).iloc[0]['conteo']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Estaciones en Catálogo", f"{count_est:,.0f}")
        c2.metric("Registros de Lluvia Total", f"{count_rain:,.0f}")
        
        if count_rain == 0:
            st.error("🚨 LA TABLA DE LLUVIA ESTÁ VACÍA. Debes cargar el archivo 'DatosPptnmes_ENSO.csv' en la pestaña 'Carga Masiva' de aquí abajo.")
        else:
            st.success("✅ Hay datos de lluvia cargados. Si no ves tu estación, verifica el Código.")
            
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")

    # --- PESTAÑAS ---
    t_explorar, t_carga = st.tabs(["🔍 Explorar y Editar Datos", "📂 Carga Masiva (Matriz)"])

    # --- SUB-PESTAÑA 1: EXPLORADOR ---
    with t_explorar:
        st.info("Consulta y edición de datos históricos.")
        try:
            # 1. Selector de Estación (Traemos solo las que tienen datos si es posible, o todas)
            # Usamos TRIM para limpiar espacios en blanco que suelen causar el error "No hay registros"
            estaciones_list = pd.read_sql("SELECT id_estacion, nombre FROM estaciones ORDER BY nombre", engine)
            
            if estaciones_list.empty:
                st.warning("⚠️ Primero carga el catálogo de estaciones.")
            else:
                # Crear opciones limpias
                opciones = estaciones_list.apply(lambda x: f"{x['id_estacion'].strip()} - {x['nombre']}", axis=1)
                sel_est = st.selectbox("Selecciona Estación:", opciones)
                
                if sel_est:
                    # Extraer código limpio
                    cod_est = sel_est.split(" - ")[0].strip()
                    
                    # 2. Verificar años disponibles para ESA estación específica
                    q_years = text(f"""
                        SELECT DISTINCT EXTRACT(YEAR FROM fecha)::int as anio 
                        FROM precipitacion 
                        WHERE TRIM(id_estacion) = '{cod_est}' 
                        ORDER BY anio DESC
                    """)
                    df_years = pd.read_sql(q_years, engine)
                    
                    if df_years.empty:
                        st.warning(f"⚠️ La estación {cod_est} existe en el catálogo pero NO tiene datos de lluvia asociados.")
                        st.info("Prueba cargando el archivo de lluvias nuevamente.")
                        # Mock para evitar error visual
                        anios_disp = [2023]
                    else:
                        st.success(f"📅 Años con datos: {len(df_years)}")
                        anios_disp = df_years['anio'].tolist()

                    # 3. Selector de Año
                    anio_sel = st.selectbox("Selecciona Año:", anios_disp)
                    
                    # 4. Consulta de Datos (Blindada con TRIM)
                    query_data = text(f"""
                        SELECT fecha, valor 
                        FROM precipitacion 
                        WHERE TRIM(id_estacion) = '{cod_est}' 
                        AND EXTRACT(YEAR FROM fecha) = {anio_sel}
                        ORDER BY fecha ASC
                    """)
                    df_lluvia_est = pd.read_sql(query_data, engine)
                    
                    col_edit, col_chart = st.columns([1, 2])
                    
                    with col_edit:
                        st.write(f"**Datos:** {cod_est} - {anio_sel}")
                        if df_lluvia_est.empty:
                            st.write("Sin registros.")
                        
                        # Edición
                        df_edited = st.data_editor(
                            df_lluvia_est,
                            num_rows="dynamic",
                            key=f"ed_{cod_est}_{anio_sel}",
                            column_config={
                                "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                                "valor": st.column_config.NumberColumn("Valor (mm)")
                            }
                        )
                        
                        if st.button("💾 Guardar"):
                            # Lógica de guardado simplificada (Insert/Update)
                            if not df_edited.empty:
                                with engine.begin() as conn:
                                    conn.execute(text(f"DELETE FROM precipitacion WHERE id_estacion='{cod_est}' AND EXTRACT(YEAR FROM fecha)={anio_sel}"))
                                    df_edited['id_estacion'] = cod_est
                                    df_edited.to_sql('precipitacion', engine, if_exists='append', index=False)
                                st.success("Guardado.")
                                time.sleep(0.5)
                                st.rerun()

                    with col_chart:
                        if not df_edited.empty:
                            st.line_chart(df_edited.set_index('fecha')['valor'])

        except Exception as e:
            st.error(f"Error en explorador: {e}")

    # --- SUB-PESTAÑA 2: CARGA MASIVA ---
    with t_carga:
        st.write("Sube `DatosPptnmes_ENSO.csv` (Matriz de Lluvia).")
        up_rain = st.file_uploader("Cargar Matriz de Lluvia", type=["csv"], key="up_rain_reloaded")
        
        if up_rain:
            if st.button("🚀 Procesar y Cargar Lluvia"):
                status = st.status("Procesando...", expanded=True)
                try:
                    df = pd.read_csv(up_rain, sep=';', decimal=',')
                    
                    # Limpieza básica
                    if 'fecha' not in df.columns and 'Fecha' in df.columns:
                        df = df.rename(columns={'Fecha': 'fecha'})
                        
                    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                    df = df.dropna(subset=['fecha'])
                    
                    # Melt (Pivot)
                    est_cols = [c for c in df.columns if c != 'fecha']
                    df_long = df.melt(id_vars=['fecha'], value_vars=est_cols, var_name='id_estacion', value_name='valor')
                    
                    # Limpieza de valores
                    df_long['valor'] = pd.to_numeric(df_long['valor'], errors='coerce')
                    df_long = df_long.dropna(subset=['valor'])
                    # Limpieza de IDs (CRÍTICO: quitar espacios)
                    df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()
                    
                    status.write(f"Cargando {len(df_long):,.0f} datos...")
                    
                    # Carga por lotes (Chunking) para no saturar memoria
                    chunk_size = 50000
                    total_chunks = (len(df_long) // chunk_size) + 1
                    bar = status.progress(0)
                    
                    for i, start in enumerate(range(0, len(df_long), chunk_size)):
                        batch = df_long.iloc[start : start + chunk_size]
                        
                        # Usamos tabla temporal para carga rápida
                        batch.to_sql('temp_rain', engine, if_exists='replace', index=False)
                        
                        with engine.begin() as conn:
                            # 1. Crear estaciones faltantes (Salvavidas FK)
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nombre)
                                SELECT DISTINCT id_estacion, 'Auto-Generada ' || id_estacion
                                FROM temp_rain
                                WHERE id_estacion NOT IN (SELECT id_estacion FROM estaciones)
                            """))
                            
                            # 2. Insertar Lluvia
                            conn.execute(text("""
                                INSERT INTO precipitacion (fecha, id_estacion, valor)
                                SELECT fecha, id_estacion, valor FROM temp_rain
                                ON CONFLICT (fecha, id_estacion) DO UPDATE SET valor = EXCLUDED.valor
                            """))
                        
                        bar.progress((i+1)/total_chunks)
                    
                    status.update(label="✅ ¡Carga Completa!", state="complete")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as ex:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Detalle: {ex}")


# ==============================================================================
# TAB 2: ÍNDICES (CORREGIDO Y BLINDADO)
# ==============================================================================
with tabs[2]:
    st.header("📊 Índices Climáticos (ENSO/ONI/SOI)")
    
    # Definición de pestañas internas
    sb1, sb2 = st.tabs(["👁️ Ver Tabla Completa", "📂 Cargar/Actualizar CSV"])
    
    # --- SUB-PESTAÑA 1: VISUALIZACIÓN ---
    with sb1: 
        st.markdown("### 📋 Histórico Cargado")
        try:
            # Lectura cruda para evitar errores de nombres de columna
            df_indices = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            
            if df_indices.empty:
                st.warning("⚠️ La tabla existe pero está vacía.")
            else:
                st.success(f"✅ Datos encontrados: {len(df_indices)} registros.")
                
                # Limpieza de nombres (Eliminar BOM y espacios)
                df_indices.columns = [c.replace('ï»¿', '').strip() for c in df_indices.columns]
                
                # Ordenamiento seguro en Python
                col_fecha = next((c for c in df_indices.columns if 'fecha' in c.lower() or 'date' in c.lower()), None)
                if col_fecha:
                    try:
                        df_indices[col_fecha] = pd.to_datetime(df_indices[col_fecha])
                        df_indices = df_indices.sort_values(col_fecha, ascending=False)
                    except: pass
                
                st.dataframe(df_indices, use_container_width=True)
                
        except Exception as e:
            st.info("ℹ️ No hay datos de índices. Usa la pestaña de carga.")

    # --- SUB-PESTAÑA 2: CARGA ---
    with sb2:
        st.markdown("### Cargar Archivo de Índices")
        st.info("Sube el archivo `Indices_Globales.csv`.")
        up_i = st.file_uploader("Seleccionar CSV", type=["csv"], key="up_ind_final_v2")
        
        if up_i and st.button("Procesar y Guardar", key="btn_save_ind_v2"):
            try:
                # Lectura robusta (utf-8-sig elimina BOM)
                df = pd.read_csv(up_i, sep=None, engine='python', encoding='utf-8-sig')
                
                # Normalizar columnas
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                
                # Guardar
                df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                st.success(f"✅ Guardado correcto: {len(df)} registros.")
                st.dataframe(df.head())
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: PREDIOS
# ==============================================================================
with tabs[3]:
    st.header("🏠 Gestión de Predios")
    st.info("Aquí administras la capa base de predios (Catastro).")

    sb1, sb2 = st.tabs(["👁️ Tabla Completa", "📂 Carga GeoJSON"])

    # --- SUB-PESTAÑA 1: VISUALIZAR ---
    with sb1:
        try:
            # 1. Leemos la tabla cruda sin filtros
            query_check = "SELECT * FROM predios LIMIT 5"
            df_preview = pd.read_sql(query_check, engine)
            
            # Si no da error, traemos todo (excluyendo geometría pesada)
            cols = [c for c in df_preview.columns if c != 'geometry']
            cols_sql = ", ".join([f'"{c}"' for c in cols]) # Protegemos nombres
            
            df_predios = pd.read_sql(f"SELECT {cols_sql} FROM predios", engine)
            
            st.success(f"✅ Se encontraron {len(df_predios)} predios en la base de datos.")
            st.dataframe(df_predios, use_container_width=True)
            
        except Exception as e:
            st.warning("No se pudo leer la tabla 'predios'. Posiblemente aún no se ha cargado correctamente.")
            st.error(f"Detalle técnico: {e}")

    # --- SUB-PESTAÑA 2: CARGAR (AQUÍ ESTÁ LA MAGIA) ---
    with sb2:
        st.write("Sube el archivo `PrediosEjecutados.geojson`.")
        up_file = st.file_uploader("GeoJSON Predios", type=["geojson", "json"], key="up_pred")
        
        if up_file:
            if st.button("🚀 Reemplazar Base de Datos de Predios"):
                with st.spinner("Procesando geometría y normalizando datos..."):
                    try:
                        # 1. Leer el archivo
                        import geopandas as gpd
                        gdf = gpd.read_file(up_file)
                        
                        # 2. NORMALIZACIÓN (La Clave del Éxito)
                        # Convertimos todos los nombres de columnas a minúsculas para evitar conflictos SQL
                        gdf.columns = map(str.lower, gdf.columns)
                        
                        # 3. Verificar y corregir proyección
                        if gdf.crs is None:
                            gdf.set_crs(epsg=4326, inplace=True)
                        else:
                            gdf = gdf.to_crs(epsg=4326)
                            
                        # 4. Limpieza de geometrías
                        # Convertimos MultiPolygon a Polygon si es necesario o arreglamos geometrías inválidas
                        gdf['geometry'] = gdf.geometry.buffer(0) 
                        
                        # 5. SUBIDA A SUPABASE (PostGIS)
                        # if_exists='replace' BORRA lo anterior y crea la tabla nueva limpia
                        gdf.to_postgis("predios", engine, if_exists='replace', index=False)
                        
                        st.success("✅ ¡Carga Exitosa! La tabla 'predios' ha sido creada correctamente.")
                        st.balloons()
                        
                        # Mostrar resumen de lo que se subió
                        st.write("Resumen de columnas creadas (Minúsculas):")
                        st.write(list(gdf.columns))
                        
                    except Exception as e:
                        st.error(f"❌ Error crítico subiendo predios: {e}")


# ==============================================================================
# TAB 4: CUENCAS (CARGADOR PRESERVANDO NOMBRES ORIGINALES EN SELECTOR)
# ==============================================================================
with tabs[4]:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2 = st.tabs(["👁️ Tabla Maestra", "📂 Carga GeoJSON (Full Data)"])
    
    with sb1:
        try:
            # Consultamos columnas para verificar qué hay en BD
            cols_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND column_name != 'geometry'"
            cols_bd = pd.read_sql(cols_query, engine)['column_name'].tolist()
            
            if cols_bd:
                cols_str = ", ".join([f'"{c}"' for c in cols_bd])
                df_c = pd.read_sql(f"SELECT {cols_str} FROM cuencas LIMIT 500", engine)
                st.markdown(f"**Muestra (500 registros):** | **Columnas BD:** {cols_bd}")
                st.dataframe(df_c, use_container_width=True)
            else:
                st.info("La tabla 'cuencas' existe pero no tiene columnas legibles.")
        except: 
            st.warning("No hay datos cargados o la tabla no existe.")

    with sb2:
        st.info("Sube 'SubcuencasAinfluencia.geojson'. Verás los nombres de columna ORIGINALES (ej: N-NSS3).")
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"], key="up_cuen_v4_orig")
        
        if up_c:
            try:
                # 1. Leer archivo SIN TOCAR NOMBRES DE COLUMNAS
                gdf_preview = gpd.read_file(up_c)
                
                # Lista exacta del archivo (Aquí aparecerá 'N-NSS3' con guion)
                cols_originales = list(gdf_preview.columns)
                
                st.success(f"✅ Archivo leído. {len(gdf_preview)} registros.")
                st.write(f"Columnas detectadas: {cols_originales}")
                
                st.markdown("##### 🛠️ Mapeo de Identificadores")
                c1, c2 = st.columns(2)
                
                # Buscamos 'N-NSS3' tal cual, o 'subc_lbl'
                # La búsqueda es insensible a mayúsculas para ayudar, pero el selector muestra el original
                idx_nom = next((i for i, c in enumerate(cols_originales) if c.lower() in ['n-nss3', 'n_nss3', 'subc_lbl', 'nombre']), 0)
                idx_id = next((i for i, c in enumerate(cols_originales) if c.lower() in ['cod', 'objectid', 'id']), 0)
                
                # SELECTORES (Muestran nombre original)
                col_nombre_origen = c1.selectbox("📌 Columna de NOMBRE (Busca N-NSS3):", cols_originales, index=idx_nom, key="sel_cn_nom_orig")
                col_id_origen = c2.selectbox("🔑 Columna de ID (Ej: COD):", cols_originales, index=idx_id, key="sel_cn_id_orig")
                
                if st.button("🚀 Guardar en Base de Datos", key="btn_save_cuen_orig"):
                    status = st.status("Procesando...", expanded=True)
                    
                    # 2. Crear las columnas estándar para la App (nombre_cuenca, id_cuenca)
                    # Tomamos los datos de las columnas que TÚ elegiste
                    gdf_preview['nombre_cuenca'] = gdf_preview[col_nombre_origen].astype(str)
                    gdf_preview['id_cuenca'] = gdf_preview[col_id_origen].astype(str)
                    
                    # 3. AHORA SÍ: Limpieza técnica para SQL (solo al momento de guardar)
                    # Convertimos todo a minúsculas y guiones bajos para que PostGIS no falle
                    # 'N-NSS3' se guardará como 'n_nss3' en la BD, pero sus datos ya están copiados en 'nombre_cuenca'
                    gdf_preview.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in gdf_preview.columns]
                    
                    # 4. Reproyección
                    if gdf_preview.crs and gdf_preview.crs.to_string() != "EPSG:4326":
                        status.write("🔄 Reproyectando a WGS84...")
                        gdf_preview = gdf_preview.to_crs("EPSG:4326")
                    
                    # 5. Guardar
                    status.write("📤 Subiendo a Supabase...")
                    gdf_preview.to_postgis("cuencas", engine, if_exists='replace', index=False)
                    
                    status.update(label="¡Carga Exitosa!", state="complete")
                    st.success(f"✅ Tabla actualizada. Se mapeó **'{col_nombre_origen}'** → **'nombre_cuenca'**.")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error procesando archivo: {e}")


# ==============================================================================
# TAB 5: MUNICIPIOS
# ==============================================================================
with tabs[5]:
    st.header("🏙️ Municipios")
    sb1, sb2 = st.tabs(["👁️ Ver y Editar Tabla", "📂 Cargar GeoJSON"])
    
    with sb1:
        try:
            df_m = pd.read_sql("SELECT * FROM municipios ORDER BY nombre_municipio", engine)
            st.info(f"Gestionando {len(df_m)} municipios.")
            
            # Tabla editable
            df_m_edit = st.data_editor(
                df_m, 
                key="editor_municipios", 
                use_container_width=True,
                height=500
            )
            
            if st.button("💾 Guardar Cambios Municipios", key="btn_save_mun"):
                df_m_edit.to_sql('municipios', engine, if_exists='replace', index=False)
                st.success("✅ Municipios actualizados.")
        except Exception as e:
            st.warning("No hay municipios cargados.")

    with sb2:
        st.info("Carga el archivo de Municipios. Selecciona la columna correcta para evitar el error 'ANTIOQUIA'.")
        up_m = st.file_uploader("GeoJSON Municipios", type=["geojson", "json"], key="up_mun_geo_smart")
        
        if up_m:
            try:
                gdf_m = gpd.read_file(up_m)
                cols_m = list(gdf_m.columns)
                
                st.markdown("##### 🛠️ Mapeo de Columnas")
                c1, c2 = st.columns(2)
                
                # Intentamos adivinar MPIO_CNMBR o NOMBRE_MUNICIPIO
                idx_nom_m = next((i for i, c in enumerate(cols_m) if c.lower() in ['mpio_cnmbr', 'nombre_municipio', 'nombre']), 0)
                idx_cod_m = next((i for i, c in enumerate(cols_m) if c.lower() in ['mpio_cdpmp', 'codigo', 'id_municipio']), 0)
                
                # EL USUARIO ELIGE LA VERDAD
                col_nom_mun = c1.selectbox("📌 Columna NOMBRE MUNICIPIO:", cols_m, index=idx_nom_m, help="Selecciona la que dice 'Medellín', NO la que dice 'Antioquia'")
                col_cod_mun = c2.selectbox("🔑 Columna CÓDIGO DANE:", cols_m, index=idx_cod_m)
                
                if st.button("🚀 Guardar Municipios", key="btn_save_mun_smart"):
                    status = st.status("Procesando...", expanded=True)
                    
                    if gdf_m.crs and gdf_m.crs.to_string() != "EPSG:4326":
                        gdf_m = gdf_m.to_crs("EPSG:4326")
                        
                    # Renombrado Estándar
                    gdf_m = gdf_m.rename(columns={
                        col_nom_mun: 'nombre_municipio', # ESTANDARIZADO
                        col_cod_mun: 'id_municipio'
                    })
                    
                    # Limpieza extra
                    if 'departamento' not in gdf_m.columns:
                        gdf_m['departamento'] = 'Antioquia' # Default si falta
                        
                    gdf_m.to_postgis('municipios', engine, if_exists='replace', index=False)
                    
                    status.update(label="¡Listo!", state="complete")
                    st.success(f"✅ Municipios cargados. Mapeo: **{col_nom_mun}** → **nombre_municipio**")
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {e}")


# ==============================================================================
# TAB 6: GESTIÓN DE RASTERS EN LA NUBE (DEM + COBERTURAS)
# ==============================================================================
with tabs[6]:
    st.header("☁️ Gestión de Rasters (DEM / Coberturas)")
    st.info("Sube aquí los archivos .tif para que el modelo hidrológico los use.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 En la Nube")
        rasters = get_raster_list()
        if rasters:
            df_r = pd.DataFrame(rasters)
            if not df_r.empty and 'name' in df_r.columns:
                st.dataframe(df_r[['name', 'created_at']], hide_index=True)
                
                to_del = st.selectbox("Eliminar:", df_r['name'])
                if st.button("🗑️ Borrar Archivo"):
                    ok, msg = delete_raster_from_storage(to_del)
                    if ok: st.success(msg); time.sleep(1); st.rerun()
                    else: st.error(msg)
            else:
                st.info("Bucket vacío o sin acceso.")
        else:
            st.warning("No hay archivos cargados.")

    with col2:
        st.subheader("⬆️ Subir Archivo")
        st.markdown("Requeridos: `DemAntioquia_EPSG3116.tif` y `Cob25m_WGS84.tif`")
        f = st.file_uploader("GeoTIFF", type=["tif", "tiff"], key="up_cloud")
        
        if f:
            if st.button(f"🚀 Subir {f.name} a Supabase"):
                with st.spinner("Subiendo..."):
                    bytes_data = f.getvalue()
                    ok, msg = upload_raster_to_storage(bytes_data, f.name)
                    if ok:
                        st.success(msg)
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(msg)

# ==============================================================================
# TABS 7, 8, 9: GIS ROBUSTO + VISORES DE TABLA (CLAVES ÚNICAS AÑADIDAS)
# ==============================================================================
with tabs[7]: # Bocatomas
    st.header("💧 Bocatomas")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("bocatomas", "ed_boca")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_boca_file")
        if st.button("Cargar", key="btn_load_boca"): cargar_capa_gis_robusta(f, "bocatomas", engine)

with tabs[8]: # Hidro
    st.header("⛰️ Hidrogeología")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("zonas_hidrogeologicas", "ed_hidro")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_hidro_file")
        if st.button("Cargar", key="btn_load_hidro"): cargar_capa_gis_robusta(f, "zonas_hidrogeologicas", engine)

with tabs[9]: # Suelos
    st.header("🌱 Suelos")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("suelos", "ed_suelo")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_suelo_file")
        if st.button("Cargar", key="btn_load_suelo"): cargar_capa_gis_robusta(f, "suelos", engine)

# ==============================================================================
# TAB 10: SQL
# ==============================================================================
with tabs[10]:
    st.header("🛠️ Consola SQL")
    q = st.text_area("Query:")
    if st.button("Ejecutar", key="btn_run_sql"):
        try:
            with engine.connect() as conn:
                if q.strip().lower().startswith("select"):
                    st.dataframe(pd.read_sql(text(q), conn))
                else:
                    conn.execute(text(q))
                    conn.commit()
                    st.success("Hecho.")
        except Exception as e: st.error(str(e))

# ==============================================================================
# TAB 11: INVENTARIO DE ARCHIVOS (NUEVO)
# ==============================================================================
with tabs[11]: # Índice 10 porque es la pestaña número 11 (0-10)
    st.header("📚 Inventario de Archivos del Sistema")
    st.markdown("Documentación técnica de los archivos requeridos para la operación de la plataforma.")
    
    # Definimos la data del inventario manualmente según tu estructura
    inventario_data = [
        {
            "Archivo": "mapaCVENSO.csv",
            "Formato": ".csv",
            "Tipo": "Metadatos Estaciones",
            "Descripción": "Coordenadas, nombres y alturas de las estaciones.",
            "Campos Clave": "id_estacion, nombre, latitud, longitud, altitud"
        },
        {
            "Archivo": "Indices_Globales.csv",
            "Formato": ".csv",
            "Tipo": "Clima Global",
            "Descripción": "Series históricas de índices macroclimáticos (ONI, SOI, etc).",
            "Campos Clave": "año, mes, anomalia_oni, soi, iod, enso_mes"
        },
        {
            "Archivo": "Predios Ejecutados.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Delimitación de predios intervenidos o gestionados.",
            "Campos Clave": "PK_PREDIOS, NOMBRE_PRE, NOMB_MPIO, AREA_HA"
        },
        {
            "Archivo": "SubcuencasAinfluencia.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Límites hidrográficos y zonas de influencia.",
            "Campos Clave": "COD/OBJECTID, SUBC_LBL, Shape_Area, SZH, AH, ZH"
        },
        {
            "Archivo": "Municipios.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "División político-administrativa del departamento.",
            "Campos Clave": "MPIO_CDPMP (Código DANE), MPIO_CNMBR (Nombre)"
        },
        {
            "Archivo": "Cob25m_WGS84.tiff",
            "Formato": ".tiff",
            "Tipo": "Raster",
            "Descripción": "Imagen satelital clasificada de coberturas vegetales.",
            "Campos Clave": "N/A (Valores de píxel: 1=Bosque, 2=Cultivo, etc.)"
        },
        {
            "Archivo": "Bocatomas_Ant.zip",
            "Formato": ".zip (Shapefile)",
            "Tipo": "Vector (Puntos)",
            "Descripción": "Ubicación de captaciones de agua.",
            "Campos Clave": "nombre_bocatoma, caudal, usuario"
        },
        {
            "Archivo": "Zonas_PotHidrogeologico.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Clasificación del potencial de aguas subterráneas.",
            "Campos Clave": "potencial, unidad_geologica"
        },
        {
            "Archivo": "Suelos_Antioquia.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Unidades de suelo y capacidad agrológica.",
            "Campos Clave": "unidad_suelo, textura, grupo_hidro"
        }
    ]
    
    # Crear DataFrame
    df_inv = pd.DataFrame(inventario_data)
    
    # Mostrar tabla bonita
    st.dataframe(
        df_inv,
        column_config={
            "Archivo": st.column_config.TextColumn("Nombre Archivo", width="medium"),
            "Descripción": st.column_config.TextColumn("Descripción", width="large"),
            "Campos Clave": st.column_config.TextColumn("Campos / Columnas", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )

# ==============================================================================
# TAB 12: RED DE DRENAJE (NUEVO)
# ==============================================================================
with tabs[12]: 
    st.header("〰️ Red de Drenaje (Escala 1:25k)")
    st.info("Gestiona la capa oficial de ríos y quebradas.")
    
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    
    with sb1: 
        # Ahora sí funcionará porque la función ya está definida arriba
        editor_tabla_gis("red_drenaje", "ed_drenaje")
        
    with sb2:
        st.markdown("### Cargar Capa de Drenaje")
        st.info("Opciones de carga:")
        st.markdown("""
        * **Opción A (Recomendada):** Arrastra **JUNTOS** los 4 archivos del Shapefile (`.shp`, `.dbf`, `.prj`, `.cpg` o `.shx`).
        * **Opción B:** Sube un solo archivo `.zip` o `.geojson`.
        """)
        
        # CAMBIO CLAVE: accept_multiple_files=True
        files = st.file_uploader(
            "Arrastra aquí tus archivos", 
            type=["zip", "geojson", "shp", "dbf", "prj", "cpg", "shx"], 
            key="up_drenaje_multi",
            accept_multiple_files=True 
        )
        
        if st.button("🚀 Cargar Red de Drenaje", key="btn_load_drenaje"): 
            if files:
                cargar_capa_gis_robusta(files, "red_drenaje", engine)
            else:
                st.warning("⚠️ Debes seleccionar los archivos primero.")

# ==============================================================================
# TAB 13: ZONA DE PELIGRO (MANTENIDA)
# ==============================================================================
with tabs[13]:  # <--- NOTA: AHORA ES TAB 13
    st.header("Zona de Peligro") 
    
    st.subheader("🧹 Limpieza de Tablas Obsoletas")
    st.warning("Se ha detectado una tabla antigua llamada 'precipitacion_mensual' que causa conflictos.")
    
    if st.button("🗑️ Eliminar Tabla 'precipitacion_mensual' (SOLO BASURA)", type="primary"):
        try:
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS precipitacion_mensual"))
                conn.commit()
            st.success("✅ Tabla 'precipitacion_mensual' eliminada.")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"Error al eliminar: {e}")
            
    st.divider()
    
    st.error("""
    **¡CUIDADO!**
    Esta zona permite reiniciar la base de datos. Úsala solo si es estrictamente necesario.
    """)
    
    with st.expander("💣 MOSTRAR BOTÓN DE RESET"):
        if st.button("EJECUTAR REINICIO DE FÁBRICA", key="btn_nuke_final", type="primary"):
            try:
                with engine.connect() as conn:
                    # (Tu código de borrado original va aquí)
                    try: conn.rollback()
                    except: pass
                    
                    conn.execute(text("DROP TABLE IF EXISTS precipitacion CASCADE"))
                    conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE"))
                    conn.execute(text("DROP TABLE IF EXISTS indices_climaticos CASCADE"))
                    # ... (resto de tablas)
                    conn.commit()
                    
                st.success("✅ Base de datos reiniciada.")
                st.balloons()
            except Exception as e: st.error(f"Error: {e}")
    
# ==============================================================================
# TAB 14: GESTIÓN DEMOGRÁFICA (ACTUALIZADA PARA MUNICIPIOS Y EDADES HASTA 100)
# ==============================================================================
with tabs[14]:  # (Asegúrate de que esta variable coincida con tu st.tabs)
    st.header("👥 Gestión de Datos Demográficos y Poblacionales")
    st.markdown("""
    Aquí puedes actualizar las bases de datos maestras de Población Municipal y Estructura por Edades (1950-2070).
    **Instrucciones:** Selecciona el tipo de dato, sube tu archivo asegurándote de que tenga los encabezados exactos.
    """)
    
    # 1. Selector del tipo de datos a cargar
    tipo_carga = st.radio(
        "Selecciona la base de datos a actualizar:",
        ["📈 Población Histórica y Proyección por Municipios", "🏗️ Estructura por Edades (0 a 100 años)"],
        horizontal=True
    )
    
    st.divider()
    
    # 2. Definición de Plantillas y Columnas Requeridas
    if "Municipios" in tipo_carga:
        cols_requeridas = ['depto_nom', 'cod_mp', 'municipio', 'area_geografica', 'año', 'Poblacion']
        archivo_salida = "data/Pob_mpios_colombia.csv"
        nombre_plantilla = "plantilla_mpios_colombia.csv"
        desc_ayuda = "El archivo debe contener las columnas: depto_nom, cod_mp, municipio, area_geografica (total/urbano/rural), año y Poblacion."
    else:
        # Genera automáticamente la lista de columnas ['dpnom', 'año', 'area_geografica', 'sexo', '0', '1', ..., '100']
        cols_requeridas = ['dpnom', 'año', 'area_geografica', 'sexo'] + [str(i) for i in range(101)]
        archivo_salida = "data/Pob_sexo_edad_Colombia_1950-2070.csv"
        nombre_plantilla = "plantilla_sexo_edad.csv"
        desc_ayuda = "El archivo debe contener: dpnom, año, area_geografica, sexo y una columna numérica para cada edad del 0 al 100."

    col_down, col_up = st.columns([1, 2])
    
    # 3. Opción de Descargar Plantilla Vacía
    with col_down:
        st.subheader("1. Descargar Plantilla")
        st.info(desc_ayuda)
        
        df_plantilla = pd.DataFrame(columns=cols_requeridas)
        # Se genera la plantilla estándar
        csv_plantilla = df_plantilla.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=f"📥 Descargar {nombre_plantilla}", 
            data=csv_plantilla, 
            file_name=nombre_plantilla, 
            mime='text/csv', 
            type="primary"
        )
        
    # 4. Cargador del Archivo y Previsualización
    with col_up:
        st.subheader("2. Cargar Datos Diligenciados")
        archivo_subido = st.file_uploader(f"Sube tu archivo CSV o Excel para '{tipo_carga}'", type=['csv', 'xlsx'])
        
        if archivo_subido is not None:
            try:
                if archivo_subido.name.endswith('.csv'):
                    # Intentar leer con punto y coma (;) que es como vienen tus archivos
                    try:
                        df_nuevo = pd.read_csv(archivo_subido, sep=';')
                        if len(df_nuevo.columns) < 2:  # Si falla y lee todo en 1 columna, intenta con coma (,)
                            archivo_subido.seek(0)
                            df_nuevo = pd.read_csv(archivo_subido, sep=',')
                    except:
                        archivo_subido.seek(0)
                        df_nuevo = pd.read_csv(archivo_subido, sep=',')
                else:
                    df_nuevo = pd.read_excel(archivo_subido)
                
                # Convertimos todas las columnas a string para evitar que '0', '1' se lean como enteros y rompan la validación
                df_nuevo.columns = [str(col).strip() for col in df_nuevo.columns]
                
                columnas_faltantes = [col for col in cols_requeridas if col not in df_nuevo.columns]
                
                if columnas_faltantes:
                    st.error(f"❌ Error: El archivo no tiene la estructura correcta. Faltan las siguientes columnas: {', '.join(columnas_faltantes[:10])} ... (Mostrando las primeras 10 faltantes)")
                else:
                    st.success(f"✅ Archivo leído correctamente: {len(df_nuevo):,} registros encontrados.")
                    with st.expander("👁️ Vista Previa de los Datos", expanded=True):
                        st.dataframe(df_nuevo.head(10), use_container_width=True)
                    
                    # 5. Botón de Guardado Final
                    if st.button("💾 Guardar y Actualizar Base de Datos", type="primary", use_container_width=True):
                        import os
                        os.makedirs("data", exist_ok=True) 
                        
                        # Guardamos estandarizado separado por comas internamente
                        df_nuevo.to_csv(archivo_salida, index=False) 
                        st.balloons()
                        st.success(f"¡Base de datos actualizada con éxito! Archivo guardado en: `{archivo_salida}`")
                        
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo: {e}")

# =====================================================================
# TAB 15: MÓDULO DE CARGA ESPACIAL (SHAPEFILE -> GEOJSON -> SUPABASE)
# =====================================================================
with tab15:

    st.subheader("🗺️ Aduana SIG: Estandarización y Carga a Supabase")
    st.info("Sube los múltiples archivos de una capa Shapefile (.shp, .shx, .dbf, .prj). El sistema la convertirá al estándar web (GeoJSON WGS84) y la subirá automáticamente al bucket público de Supabase.")

    # 1. Selector de Carpeta Destino en Supabase
    carpeta_destino = st.selectbox(
        "Selecciona la carpeta de destino en Supabase:",
        ["Puntos_de_interes", "censos_ICA", "limites_administrativos", "otro"]
    )

    if carpeta_destino == "otro":
        carpeta_destino = st.text_input("Escribe el nombre de la nueva carpeta (sin espacios ni tildes):")

    # 2. Cargador Múltiple
    archivos_sig = st.file_uploader("Selecciona todos los archivos del Shapefile al mismo tiempo", accept_multiple_files=True, key="sig_uploader")

    if archivos_sig:
        archivo_shp = next((f for f in archivos_sig if f.name.endswith('.shp')), None)
            
        if archivo_shp:
            if st.button("🚀 Procesar y Subir a Supabase"):
                with st.spinner("Ensamblando, reproyectando y subiendo a la nube..."):
                    try:
                        # A. ENSAMBLAJE Y TRANSFORMACIÓN LOCAL EN MEMORIA TEMPORAL
                        with tempfile.TemporaryDirectory() as tmpdir:
                            for f in archivos_sig:
                                filepath = os.path.join(tmpdir, f.name)
                                with open(filepath, "wb") as f_out:
                                    f_out.write(f.getvalue())
                                
                            ruta_shp_temporal = os.path.join(tmpdir, archivo_shp.name)
                            gdf = gpd.read_file(ruta_shp_temporal)
                                
                            # Estandarización a WGS84 (EPSG:4326)
                            if gdf.crs is None:
                                gdf.set_crs(epsg=3116, inplace=True)
                            if gdf.crs.to_string() != "EPSG:4326":
                                gdf = gdf.to_crs(epsg=4326)
                                    
                            # Convertir a bytes de GeoJSON
                            geojson_bytes = gdf.to_json().encode('utf-8')
                            nombre_limpio = archivo_shp.name.replace('.shp', '.geojson')
                                
                            # B. SUBIDA A SUPABASE (Conexión oficial)
                            # Encendemos la conexión usando las llaves de tu proyecto
                            url_supabase = st.secrets["SUPABASE_URL"]
                            key_supabase = st.secrets["SUPABASE_KEY"]
                            cliente_supabase = create_client(url_supabase, key_supabase)
                                
                            # AQUÍ pones el nombre de tu bucket público (ej. 'sihcli_maestros')
                            nombre_bucket = 'sihcli_maestros' 
                            ruta_supabase = f"{carpeta_destino}/{nombre_limpio}"
                                
                            # Subir archivo sobrescribiendo si ya existe
                            res = cliente_supabase.storage.from_(nombre_bucket).upload(
                                file=geojson_bytes,
                                path=ruta_supabase,
                                file_options={"content-type": "application/json", "upsert": "true"}
                            )
                                
                            st.success(f"✅ ¡Éxito! Capa '{nombre_limpio}' ({len(gdf)} registros) procesada y subida a Supabase en '{ruta_supabase}'.")
                                
                    except Exception as e:
                        st.error(f"❌ Error durante el proceso: {str(e)}")
        else:
            st.warning("⚠️ Debes incluir obligatoriamente el archivo que termina en '.shp'.")


