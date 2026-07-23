# modules/data_processor.py

import os
import unicodedata

import geopandas as gpd
import pandas as pd
from shapely import wkt
from sqlalchemy import create_engine, text

import streamlit as st
from modules.config import Config

# 🔥 IMPORTAMOS LA APLANADORA MAESTRA (EL CEREBRO LINGÜÍSTICO)
from modules.utils import normalizar_texto_maestro, cargar_capa_espacial_cache

# Función auxiliar robusta para fechas en español (ene-70 -> datetime)
def parse_spanish_date_robust(x):
    if isinstance(x, pd.Timestamp):
        return x
    if pd.isna(x) or x == "":
        return pd.NaT
    x = str(x).lower().strip()
    trans = {
        "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
        "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
        "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec",
    }
    for es, en in trans.items():
        if es in x:
            x = x.replace(es, en)
            break
    try:
        return pd.to_datetime(x, format="%b-%y")
    except:
        try:
            return pd.to_datetime(x)
        except:
            return pd.NaT

# =====================================================================
# 🚀 MOTOR ESPACIAL OPTIMIZADO (Lectura Directa de PostGIS)
# =====================================================================
@st.cache_data(show_spinner="Descargando Gemelo Digital de Supabase...", ttl=600)
def load_and_process_all_data():
    gdf_stations = gpd.GeoDataFrame()
    gdf_municipios = gpd.GeoDataFrame()
    gdf_subcuencas = gpd.GeoDataFrame()
    gdf_predios = gpd.GeoDataFrame()
    df_long = pd.DataFrame()
    df_enso = pd.DataFrame()

    try:
        # 🚀 FIX QUIRÚRGICO: Usar el motor blindado de db_manager en lugar de crear uno crudo
        from modules.db_manager import get_engine
        engine = get_engine()
        
        if engine is None:
            st.error("No se pudo establecer la conexión segura con la base de datos.")
            return None, None, None, None, None, None

        # --- 1. ESTACIONES (Ultra-rápido usando geom_col) ---
        try:
            sql_est = """
                SELECT 
                    id_estacion, 
                    nombre AS nom_est, 
                    altitud AS alt_est, 
                    municipio, 
                    departamento AS depto_region, 
                    geom as geometry
                FROM estaciones
            """
            # Leemos directamente la geometría sin pasar por WKT
            gdf_stations = gpd.read_postgis(text(sql_est), engine, geom_col="geometry")
            
            # Aseguramos el CRS Web Global
            if gdf_stations.crs is None:
                gdf_stations.set_crs(epsg=4326, inplace=True)

            # Etiqueta única
            def create_lbl(row):
                n = str(row["nom_est"]).strip() if "nom_est" in row else "Estacion"
                c = str(row["id_estacion"]).strip()
                return n if c in n else f"{n} [{c}]"

            if not gdf_stations.empty:
                gdf_stations["station_label"] = gdf_stations.apply(create_lbl, axis=1)

                if "nom_est" in gdf_stations.columns:
                    gdf_stations = gdf_stations.drop(columns=["nom_est"])

                gdf_stations = gdf_stations.rename(
                    columns={
                        "station_label": Config.STATION_NAME_COL,
                        "alt_est": Config.ALTITUDE_COL,
                        "municipio": Config.MUNICIPALITY_COL,
                        "depto_region": Config.REGION_COL,
                    }
                )
                
                # Extraer Lat/Lon rápidamente
                gdf_stations["latitude"] = gdf_stations.geometry.y
                gdf_stations["longitude"] = gdf_stations.geometry.x
                
        except Exception as e:
            st.warning(f"⚠️ Error cargando Estaciones: {e}")

        # --- 2. PRECIPITACIÓN ---
        try:
            # 🚀 FIX ARQUITECTÓNICO: Apagamos la descarga masiva global.
            df_long = pd.DataFrame()
        except Exception as e:
            st.error(f"⚠️ Error en Precipitación: {e}")
            df_long = pd.DataFrame()
            
        # --- 3. GEOMETRÍAS (Usando Caché Robusta) ---
        try:
            # 🚀 FIX JINETE 2: Usamos la función blindada que creamos en utils.py
            # Esto evitará el "Statement Timeout" y protegerá el disco de Supabase
            gdf_municipios = cargar_capa_espacial_cache("SELECT * FROM municipios")
            gdf_subcuencas = cargar_capa_espacial_cache("SELECT * FROM cuencas")
            
            try:
                gdf_predios = cargar_capa_espacial_cache("SELECT * FROM predios")
            except:
                gdf_predios = gpd.GeoDataFrame() # Fallback seguro
                
            # Validamos el sistema de coordenadas de todas las capas
            for capa in [gdf_municipios, gdf_subcuencas, gdf_predios]:
                if capa is not None and not capa.empty and capa.crs is None:
                    capa.set_crs(epsg=4326, inplace=True)
                    
        except Exception as e:
            st.warning(f"⚠️ Error en lectura de polígonos: {e}")
            
        # --- 4. ENSO ---
        try:
            df_enso = pd.read_sql(text("SELECT * FROM indices_climaticos"), engine)
            df_enso.columns = [c.lower() for c in df_enso.columns]
            col_fecha = "fecha" if "fecha" in df_enso.columns else "fecha_mes_año"
            if col_fecha in df_enso.columns:
                df_enso[Config.DATE_COL] = df_enso[col_fecha].apply(parse_spanish_date_robust)
                df_enso = df_enso.dropna(subset=[Config.DATE_COL])
            if "oni" in df_enso.columns:
                df_enso = df_enso.rename(columns={"oni": Config.ENSO_ONI_COL})
            elif "anomalia_oni" in df_enso.columns:
                df_enso = df_enso.rename(columns={"anomalia_oni": Config.ENSO_ONI_COL})
        except:
            pass

        return gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios

    except Exception as e:
        st.error(f"Error Crítico BD: {e}")
        return None, None, None, None, None, None

# =====================================================================
# 🚀 NUEVO ESCUDO PARA HISTORIAL DE PRECIPITACIÓN
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False) # Caché de 1 hora
def load_precipitation_for_stations_cached(lista_estaciones):
    """
    Descarga el historial de lluvia SOLO para las estaciones seleccionadas
    y guarda el resultado en caché para evitar el 'SSL connection closed'.
    """
    if not lista_estaciones:
        return pd.DataFrame()
        
    try:
        from modules.db_manager import get_engine
        engine = get_engine()
        
        # Formateamos la lista para la consulta SQL IN ('A', 'B')
        estaciones_str = "','".join([str(e).strip() for e in lista_estaciones])
        query = f"SELECT id_estacion, fecha, valor FROM precipitacion WHERE id_estacion IN ('{estaciones_str}')"
        
        with engine.begin() as conn:
            # Aumentamos el tiempo de espera por si son muchos datos
            conn.execute(text("SET statement_timeout = '600000';"))
            df = pd.read_sql(text(query), conn)
            
        return df
    except Exception as e:
        import logging
        logging.error(f"Error descargando historial de lluvia: {e}")
        return pd.DataFrame()

def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method="linear", limit_direction="both")
    return df

def leer_csv_robusto(ruta):
    """
    Lee archivos CSV forzando la codificación UTF-8 para proteger 
    las tildes y eñes (Ej: Abriaquí, Acacías).
    """
    try:
        # 1. Intentamos forzar UTF-8 (Corrige los caracteres extraños)
        df = pd.read_csv(ruta, sep=';', low_memory=False, encoding='utf-8')
        if len(df.columns) < 5: 
            df = pd.read_csv(ruta, sep=',', low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        # 2. Si el archivo es muy antiguo, aplicamos el Fallback a Latin-1
        try:
            df = pd.read_csv(ruta, sep=';', low_memory=False, encoding='latin1')
            if len(df.columns) < 5: 
                df = pd.read_csv(ruta, sep=',', low_memory=False, encoding='latin1')
        except Exception: 
            return pd.DataFrame()
    except Exception: 
        return pd.DataFrame()
    
    if not df.empty:
        # Limpiamos caracteres fantasma (BOM) que Excel suele dejar
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
    return df

# =====================================================================
# CARGADORES CENTRALIZADOS DEL ICA (EL ALEPH DE DATOS)
# =====================================================================
@st.cache_data
def cargar_censo_ica(tipo):
    """
    Carga el censo oficial. 'tipo' puede ser: 'bovino', 'porcino', 'aviar'
    """
    archivos = {
        'bovino': "data/censos_ICA/Censo_ICA_Bovinos_2023",
        'porcino': "data/censos_ICA/Censo_ICA_Porcinos_2023",
        'aviar': "data/censos_ICA/Censo_ICA_Aves_2025"
    }
    
    if tipo not in archivos: return pd.DataFrame()
    
    ruta_base = archivos[tipo]
    df = pd.DataFrame()
    
    if os.path.exists(ruta_base + ".xlsx"): df = pd.read_excel(ruta_base + ".xlsx")
    elif os.path.exists(ruta_base + ".csv"): df = leer_csv_robusto(ruta_base + ".csv")
    
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        if 'MUNICIPIO' in df.columns:
            # 🔥 USAMOS EL CEREBRO LINGÜÍSTICO AQUÍ
            df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto_maestro)
    return df

@st.cache_data
def cargar_territorio_maestro():
    """Carga la base de datos maestra con soporte dual (Local y Nube), purga de nulos y corrección de tildes."""
    import os
    import pandas as pd
    import requests
    import io
    
    df = pd.DataFrame()
    rutas = ["data/territorio_maestro.xlsx", "data/territorio_maestro.csv", "data/depto_region_car_territ_mpios.xlsx"]
    
    for ruta in rutas:
        if os.path.exists(ruta):
            if ruta.endswith('.xlsx'):
                df = pd.read_excel(ruta)
            else:
                # 🚀 FIX: Forzamos la lectura correcta de tildes colombianas (Latin-1 vs UTF-8)
                try:
                    df = pd.read_csv(ruta, encoding='latin-1') 
                except:
                    df = pd.read_csv(ruta, encoding='utf-8')
            break
            
    if df.empty:
        try:
            url = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=15)
            if res.status_code == 200: df = pd.read_excel(io.BytesIO(res.content))
        except: pass
            
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        
        # 🔥 PURGA FORENSE: Eliminamos filas basura y evitamos el "Nan"
        df = df.dropna(subset=['dp_mp'])
        df['dp_mp'] = df['dp_mp'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
        
        from modules.utils import normalizar_texto_maestro
        if 'municipio' in df.columns:
            df['municipio_norm'] = df['municipio'].astype(str).apply(normalizar_texto_maestro)
            
        if 'subregion' in df.columns:
            df['subregion'] = df['subregion'].fillna('Sin Subregion').astype(str).str.title()
            # Esta línea destruye a "Nan" para siempre
            df['subregion_norm'] = df['subregion'].astype(str).apply(normalizar_texto_maestro)
            
    return df
