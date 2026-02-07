# modules/admin_utils.py


import os
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import tempfile

from modules.utils import standardize_numeric_column

def parsear_fechas_espanol(series_fechas):
    """
    Convierte fechas en español a objetos datetime válidos.
    Funciona con: 'Ene-2021', '01-ago-1999', 'diciembre 2020', etc.
    """
    import pandas as pd
    
    # 1. Validación rápida
    if series_fechas is None or series_fechas.empty:
        return series_fechas

    # 2. Convertir a string, minúsculas y quitar espacios
    s = series_fechas.astype(str).str.lower().str.strip()

    # 3. Diccionario de traducción (Español -> Inglés estándar)
    traduccion = {
        'ene': 'jan', 'abr': 'apr', 'ago': 'aug', 'dic': 'dec',
        'enero': 'january', 'abril': 'april', 'agosto': 'august', 'diciembre': 'december'
    }

    # 4. Reemplazo masivo
    for es, en in traduccion.items():
        s = s.str.replace(es, en, regex=False)

    # 5. Conversión final con Pandas (él es el experto infiriendo formatos)
    return pd.to_datetime(s, errors='coerce')

# --- FUNCIONES DE GESTIÓN DE STORAGE (RASTERS) ---

# Inicializar cliente Supabase (Singleton)
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def get_raster_list(bucket_name="rasters"):
    """Lista archivos en el bucket de Supabase."""
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).list()
        return res # Retorna lista de objetos
    except Exception as e:
        print(f"Error listando bucket: {e}")
        return []

def upload_raster_to_storage(file_bytes, file_name, bucket_name="rasters"):
    """Sube un archivo raster al bucket."""
    try:
        supabase = init_supabase()
        # file_options param is crucial for overwriting
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": "image/tiff", "upsert": "true"}
        )
        return True, f"✅ '{file_name}' subido a la nube exitosamente."
    except Exception as e:
        return False, f"❌ Error subiendo: {str(e)}"

def delete_raster_from_storage(file_name, bucket_name="rasters"):
    """Elimina un archivo del bucket."""
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).remove([file_name])
        return True, f"🗑️ '{file_name}' eliminado."
    except Exception as e:
        return False, f"❌ Error eliminando: {str(e)}"



@st.cache_resource
def init_supabase():
    # Ahora que corregiste la carpeta .streamlit, esto funcionará
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def get_raster_list(bucket_name="rasters"):
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).list()
        return res
    except Exception as e:
        return []

def upload_raster_to_storage(file_bytes, file_name, bucket_name="rasters"):
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": "image/tiff", "upsert": "true"}
        )
        return True, f"✅ Carga exitosa: {file_name}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def delete_raster_from_storage(file_name, bucket_name="rasters"):
    try:
        supabase = init_supabase()
        supabase.storage.from_(bucket_name).remove([file_name])
        return True, f"🗑️ Eliminado: {file_name}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def download_raster_to_temp(file_name, bucket_name="rasters"):
    """
    Descarga un archivo de Supabase y devuelve la ruta temporal local.
    """
    try:
        supabase = init_supabase()
        # Descargamos los bytes
        data = supabase.storage.from_(bucket_name).download(file_name)
        
        # Guardamos en un archivo temporal que el sistema pueda leer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(data)
            return tmp.name # Retornamos la ruta (ej: /tmp/tmpxyz.tif)
    except Exception as e:
        print(f"Error descargando {file_name}: {e}")

        return None

# =============================================================================
# 🛠️ FUNCIONES DE LIMPIEZA Y ESTANDARIZACIÓN UNIVERSAL (NUEVO)
# =============================================================================

def limpiar_encabezados_bom(df):
    """
    Elimina caracteres fantasma (BOM) y espacios de los nombres de columnas.
    Ej: 'ï»¿id_estacion ' -> 'id_estacion'
    """
    if df is None: return None
    # Elimina BOM utf-8, BOM excel y espacios
    df.columns = df.columns.str.replace('ï»¿', '')\
                           .str.replace('\ufeff', '')\
                           .str.strip()
    return df

def estandarizar_id_estacion(df, posibles_nombres=None):
    """
    Busca la columna de ID (entre candidatos), la renombra a 'id_estacion'
    y la convierte a texto limpio para asegurar cruces.
    """
    if df is None: return None
    df = limpiar_encabezados_bom(df) # Paso 1: Limpiar encabezados
    
    if posibles_nombres is None:
        posibles_nombres = ['id_estacion', 'Id_estacio', 'Codigo', 'ID', 'CODIGO', 'estacion']
    
    # 1. Buscar columna candidata
    col_encontrada = next((c for c in posibles_nombres if c in df.columns), None)
    
    # Si está en el índice, sacarla
    if not col_encontrada and df.index.name in posibles_nombres:
        df = df.reset_index()
        col_encontrada = df.index.name or 'id_estacion'

    # 2. Renombrar y Castear
    if col_encontrada:
        df.rename(columns={col_encontrada: 'id_estacion'}, inplace=True)
        # Convertir a string, quitar decimales (.0) si vienen de excel y espacios
        df['id_estacion'] = df['id_estacion'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df
    else:
        # Si no se encuentra, retornamos el df tal cual (o podríamos lanzar error)
        return df

def asegurar_geometria_estaciones(df):
    """
    Recibe un DataFrame de estaciones y garantiza que salga como GeoDataFrame
    con coordenadas válidas WGS84.
    """
    import geopandas as gpd
    import pandas as pd
    
    if df is None: return None
    df = limpiar_encabezados_bom(df)
    
    # Si ya es GeoDataFrame válido, retornar
    if isinstance(df, gpd.GeoDataFrame) and getattr(df, 'crs', None) is not None:
        return df.to_crs("EPSG:4326")

    # Si no, intentar reconstruir desde lat/lon
    candidatos_lon = ['longitud', 'Longitud_geo', 'lon', 'LONGITUD']
    candidatos_lat = ['latitud', 'Latitud_geo', 'lat', 'LATITUD']
    
    c_lon = next((c for c in candidatos_lon if c in df.columns), None)
    c_lat = next((c for c in candidatos_lat if c in df.columns), None)
    
    if c_lon and c_lat:
        try:
            # Limpieza agresiva de coordenadas (comas por puntos, forzar numérico)
            df[c_lon] = pd.to_numeric(df[c_lon].astype(str).str.replace(',', '.'), errors='coerce')
            df[c_lat] = pd.to_numeric(df[c_lat].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Eliminar vacíos
            df = df.dropna(subset=[c_lon, c_lat])
            
            # Crear geometría
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df[c_lon], df[c_lat]),
                crs="EPSG:4326"
            )
            return gdf
        except Exception:
            return df # Retorna el original si falla
            
    return df


