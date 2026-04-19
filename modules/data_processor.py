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
from modules.utils import normalizar_texto_maestro

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
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL en secrets.toml.")
            return None, None, None, None, None, None

        engine = create_engine(st.secrets["DATABASE_URL"])

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
            gdf_stations = gpd.read_postgis(sql_est, engine, geom_col="geometry")
            
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
            sql_ppt = text("SELECT id_estacion, fecha, valor FROM precipitacion ORDER BY fecha")
            df_ppt = pd.read_sql(sql_ppt, engine)

            df_ppt = df_ppt.rename(columns={'fecha': Config.DATE_COL})
            df_ppt[Config.DATE_COL] = pd.to_datetime(df_ppt[Config.DATE_COL])
            df_ppt['valor'] = pd.to_numeric(df_ppt['valor'], errors='coerce')
            df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str).str.strip()
            df_ppt = df_ppt.dropna(subset=[Config.DATE_COL])

            if not gdf_stations.empty:
                gdf_stations['id_estacion'] = gdf_stations['id_estacion'].astype(str).str.strip()
                df_long = pd.merge(
                    df_ppt,
                    gdf_stations[["id_estacion", Config.STATION_NAME_COL]],
                    on="id_estacion",
                    how="inner",
                )
                df_long = df_long.rename(columns={"valor": Config.PRECIPITATION_COL})
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception as e:
            st.error(f"⚠️ Error cargando Precipitación: {e}")
            df_long = pd.DataFrame() 
            
        # --- 3. GEOMETRÍAS (Adiós a los archivos locales pesados) ---
        try:
            # Ahora la app va directo a tus tablas súper indexadas
            gdf_municipios = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
            gdf_subcuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
            
            try:
                gdf_predios = gpd.read_postgis("SELECT * FROM predios", engine, geom_col="geometry")
            except:
                gdf_predios = gpd.GeoDataFrame() # Fallback seguro
                
            # Validamos el sistema de coordenadas de todas las capas
            for capa in [gdf_municipios, gdf_subcuencas, gdf_predios]:
                if not capa.empty and capa.crs is None:
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

def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method="linear", limit_direction="both")
    return df

def leer_csv_robusto(ruta):
    try:
        df = pd.read_csv(ruta, sep=';', low_memory=False)
        if len(df.columns) < 5: df = pd.read_csv(ruta, sep=',', low_memory=False)
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception: return pd.DataFrame()

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
    """
    Carga la base de datos maestra que relaciona municipios con regiones y CARs.
    """
    import os
    import pandas as pd
    
    rutas = [
        "data/territorio_maestro.xlsx", 
        "data/territorio_maestro.csv",
        "data/depto_region_car_territ_mpios.xlsx",
        "data/depto_region_car_territ_mpios.csv"
    ]
    
    df = pd.DataFrame()
    for ruta in rutas:
        if os.path.exists(ruta):
            if ruta.endswith('.xlsx'): 
                df = pd.read_excel(ruta)
            else: 
                try:
                    df = leer_csv_robusto(ruta) 
                except:
                    df = pd.read_csv(ruta)
            break
            
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        
        if 'municipio' in df.columns:
            # 🔥 USAMOS EL CEREBRO LINGÜÍSTICO AQUÍ
            df['municipio_norm'] = df['municipio'].astype(str).apply(normalizar_texto_maestro)
        if 'car' in df.columns:
            df['car'] = df['car'].astype(str).str.upper()
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.title()
            # 🔥 USAMOS EL CEREBRO LINGÜÍSTICO AQUÍ
            df['region_norm'] = df['region'].astype(str).apply(normalizar_texto_maestro)
            
    return df
