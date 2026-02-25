# modules/data_processor.py

import geopandas as gpd
import pandas as pd
import streamlit as st
import os
import unicodedata
from shapely import wkt
from sqlalchemy import create_engine, text

from modules.config import Config

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

@st.cache_data(show_spinner="Procesando datos...", ttl=600)
def load_and_process_all_data():
    gdf_stations = pd.DataFrame()
    gdf_municipios = pd.DataFrame()
    gdf_subcuencas = pd.DataFrame()
    gdf_predios = pd.DataFrame()
    df_long = pd.DataFrame()
    df_enso = pd.DataFrame()

    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("Falta DATABASE_URL en secrets.toml.")
            return None, None, None, None, None, None

        engine = create_engine(st.secrets["DATABASE_URL"])

        # 1. ESTACIONES
        try:
            # --- TRUCO AQUÍ: 'nombre AS nom_est' ---
            # Esto engaña al código para que crea que la columna se llama 'nom_est'
            # He añadido también alias para altitud y departamento por si acaso (viendo tu imagen)
            sql_est = text("""
                SELECT 
                    id_estacion, 
                    nombre AS nom_est, 
                    altitud AS alt_est, 
                    municipio, 
                    departamento AS depto_region, 
                    ST_AsText(geom) as wkt 
                FROM estaciones
            """)
            
            df_est = pd.read_sql(sql_est, engine)

            if "wkt" in df_est.columns:
                df_est["geometry"] = df_est["wkt"].apply(
                    lambda x: wkt.loads(x) if x else None
                )
                gdf_stations = gpd.GeoDataFrame(
                    df_est, geometry="geometry", crs="EPSG:4326"
                )
            else:
                gdf_stations = df_est.copy()

            # Etiqueta única
            def create_lbl(row):
                # El código sigue buscando 'nom_est' y funcionará porque usamos el Alias arriba
                n = str(row["nom_est"]).strip() if "nom_est" in row else "Estacion"
                c = str(row["id_estacion"]).strip()
                return n if c in n else f"{n} [{c}]"

            gdf_stations["station_label"] = gdf_stations.apply(create_lbl, axis=1)

            # Dropear 'nom_est' para evitar duplicados al renombrar
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

            if "geometry" in gdf_stations.columns:
                gdf_stations = gdf_stations.dropna(subset=["geometry"])
                gdf_stations["latitude"] = gdf_stations.geometry.y
                gdf_stations["longitude"] = gdf_stations.geometry.x
        except Exception as e:
            st.warning(f"⚠️ Error cargando Estaciones: {e}")

        # 2. PRECIPITACIÓN
        try:
            # ✅ CORRECCIÓN: Consulta a la tabla limpia 'precipitacion'
            sql_ppt = text("SELECT id_estacion, fecha, valor FROM precipitacion ORDER BY fecha")
            df_ppt = pd.read_sql(sql_ppt, engine)

            # LIMPIEZA DE DATOS (Ya vienen limpios de BD, pero aseguramos tipos)
            # Mapeamos 'fecha' a lo que diga tu Config (usualmente 'fecha')
            df_ppt = df_ppt.rename(columns={'fecha': Config.DATE_COL})
            df_ppt[Config.DATE_COL] = pd.to_datetime(df_ppt[Config.DATE_COL])
            df_ppt['valor'] = pd.to_numeric(df_ppt['valor'], errors='coerce')
            df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str).str.strip()
            
            df_ppt = df_ppt.dropna(subset=[Config.DATE_COL])

            if not gdf_stations.empty:
                # Aseguramos que el ID de estaciones también esté limpio para el cruce
                gdf_stations['id_estacion'] = gdf_stations['id_estacion'].astype(str).str.strip()
                
                df_long = pd.merge(
                    df_ppt,
                    gdf_stations[["id_estacion", Config.STATION_NAME_COL]],
                    on="id_estacion", # ✅ CORRECCIÓN: Usamos id_estacion en ambos lados
                    how="inner",
                )
                
                # ✅ CORRECCIÓN: La tabla nueva tiene 'valor', renombramos a Config
                df_long = df_long.rename(columns={"valor": Config.PRECIPITATION_COL})
                
                df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
        except Exception as e:
            st.error(f"Error cargando Precipitación: {e}")
            df_long = pd.DataFrame() # Evita que la app colapse si falla
            
        # 3. GEOMETRÍAS y 4. ENSO (Sin cambios)
        try:
            sql_geo = text("SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias")
            df_geo = pd.read_sql(sql_geo, engine)
            if not df_geo.empty:
                df_geo["geometry"] = df_geo["wkt"].apply(lambda x: wkt.loads(x) if x else None)
                gdf_all = gpd.GeoDataFrame(df_geo, geometry="geometry", crs="EPSG:4326")
                gdf_municipios = gdf_all[gdf_all["tipo_geometria"] == "municipio"]
                gdf_subcuencas = gdf_all[gdf_all["tipo_geometria"].isin(["subcuenca", "cuenca"])]
                gdf_predios = gdf_all[gdf_all["tipo_geometria"] == "predio"]
        except:
            pass

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
        st.error(f"Error General DB: {e}")
        return None, None, None, None, None, None

def complete_series(df):
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method="linear", limit_direction="both")

    return df


def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

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
            df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_territorio_maestro():
    """
    Carga la base de datos maestra que relaciona municipios con regiones y CARs.
    """
    # Nombres comunes para este archivo en tu proyecto
    rutas = [
        "data/territorio_maestro.xlsx", 
        "data/territorio_maestro.csv",
        "data/divipola.xlsx",
        "data/divipola.csv"
    ]
    
    df = pd.DataFrame()
    for ruta in rutas:
        if os.path.exists(ruta):
            if ruta.endswith('.xlsx'): 
                df = pd.read_excel(ruta)
            else: 
                df = leer_csv_robusto(ruta)
            break
            
    if not df.empty:
        # Normalizamos las columnas para evitar errores de mayúsculas/espacios
        df.columns = df.columns.str.lower().str.strip()
        if 'municipio' in df.columns:
            df['municipio_norm'] = df['municipio'].astype(str).apply(normalizar_texto)
            
    return df
