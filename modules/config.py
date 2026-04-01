# modules/config.py
import os
import streamlit as st

class Config:
    """
    Configuración centralizada para SIHCLI-POTER.
    Ajustada a la NUEVA estructura de Base de Datos PostgreSQL y Activos 100% en Nube (Supabase).
    """

    APP_TITLE = "SIHCLI-POTER"

    # --- MAPEO EXACTO CON BASE DE DATOS NUEVA ---
    DATE_COL = "fecha"              # Antes: fecha_mes_año
    PRECIPITATION_COL = "valor"     # Antes: precipitation
    STATION_NAME_COL = "nombre"     # Antes: nom_est
    ALTITUDE_COL = "altitud"        # Antes: alt_est
    MUNICIPALITY_COL = "municipio"
    REGION_COL = "departamento"     # Antes: depto_region
    
    # Columnas geográficas
    LATITUDE_COL = "latitud"
    LONGITUDE_COL = "longitud"
    
    # Columnas generadas internamente
    YEAR_COL = "año"
    MONTH_COL = "mes"

    # Índices Climáticos
    ENSO_ONI_COL = "anomalia_oni"
    SOI_COL = "soi"
    IOD_COL = "iod"

    # ==============================================================================
    # ☁️ RUTAS DE ACTIVOS EN LA NUBE (SUPABASE STORAGE) - CERO ARCHIVOS LOCALES
    # ==============================================================================
    # Base URL pública de tus buckets en Supabase
    BASE_STORAGE_URL = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public"
    
    # Buckets
    BUCKET_RASTERS = f"{BASE_STORAGE_URL}/rasters"
    BUCKET_MAESTROS = f"{BASE_STORAGE_URL}/sihcli_maestros"

    # URLs exactas de los Rasters (Reemplazan las viejas rutas locales de la carpeta data/)
    DEM_FILE_PATH = f"{BUCKET_RASTERS}/DemAntioquia_EPSG3116.tif"
    PRECIP_RASTER_PATH = f"{BUCKET_RASTERS}/PPAMAnt.tif"
    LAND_COVER_RASTER_PATH = f"{BUCKET_RASTERS}/Cob25m_WGS84.tif"

    # URLs exactas de los Archivos Maestros pesados
    POBLACION_MAESTRA_URL = f"{BUCKET_MAESTROS}/Poblacion_Colombia_Maestra.parquet"
    CONCESIONES_MAESTRAS_URL = f"{BUCKET_MAESTROS}/Metabolismo_Hidrico_Antioquia_Maestro.geojson"

    # ==============================================================================
    # 🖼️ RUTAS DE INTERFAZ LOCAL (Solo para UI, imágenes y logos)
    # ==============================================================================
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, ".."))
    ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

    LOGO_PATH = os.path.join(ASSETS_DIR, "CuencaVerde_Logo.jpg")
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, "chaac.png")

    # --- TEXTOS ---
    WELCOME_TEXT = """
    **Sistema de Información Hidroclimática del Norte de la Región Andina**
    Esta plataforma integra datos históricos, análisis estadísticos y modelación espacial.
    """
    QUOTE_TEXT = "El agua es la fuerza motriz de toda la naturaleza."
    QUOTE_AUTHOR = "Leonardo da Vinci"
    CHAAC_STORY = "Chaac es la deidad maya de la lluvia."

    # --- GESTIÓN DE SESIÓN (MEMORIA DEL GEMELO DIGITAL) ---
    @staticmethod
    def initialize_session_state():
        keys = [
            "data_loaded", "apply_interpolation", "gdf_stations", "df_long",
            "df_enso", "gdf_municipios", "gdf_subcuencas", "gdf_predios",
            "unified_basin_gdf", "basin_results", "sarima_res", 
            "prophet_res", "res_cuenca", "current_coverage_stats",
            "dem_in_memory", "ppt_in_memory" 
        ]
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None
