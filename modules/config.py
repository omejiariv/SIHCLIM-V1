# modules/config.py
import os
import streamlit as st

class Config:
    """
    Configuración centralizada para SIHCLI-POTER.
    Ajustada a la NUEVA estructura de Base de Datos PostgreSQL y Activos en Nube.
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

    # --- CLAVES DE ACTIVOS EN NUBE (SUPABASE) ---
    # Estos son los nombres exactos (Keys) que busca la función cargar_raster_db
    DEM_FILENAME = "DemAntioquia_EPSG3116.tif"
    PRECIP_FILENAME = "PPAMAnt.tif"
    LAND_COVER_FILENAME = "Cob25m_WGS84.tif"

    # --- RUTAS DE DIRECTORIOS (Para temporales o fallback) ---
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, ".."))

    ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data") # Útil para descargas temporales

    # --- RUTAS DE REFERENCIA (Compatibilidad hacia atrás) ---
    # Aunque leamos de la nube, mantenemos estas rutas por si algún módulo
    # antiguo intenta verificar os.path.exists (aunque fallará en la nube pura)
    LOGO_PATH = os.path.join(ASSETS_DIR, "CuencaVerde_Logo.jpg")
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, "chaac.png")
    
    # Rutas locales teóricas (se reemplazan por los FILENAME en la lógica nueva)
    LAND_COVER_RASTER_PATH = os.path.join(DATA_DIR, LAND_COVER_FILENAME)
    DEM_FILE_PATH = os.path.join(DATA_DIR, DEM_FILENAME)
    PRECIP_RASTER_PATH = os.path.join(DATA_DIR, PRECIP_FILENAME)

    # --- TEXTOS ---
    WELCOME_TEXT = """
    **Sistema de Información Hidroclimática del Norte de la Región Andina**
    Esta plataforma integra datos históricos, análisis estadísticos y modelación espacial.
    """
    QUOTE_TEXT = "El agua es la fuerza motriz de toda la naturaleza."
    QUOTE_AUTHOR = "Leonardo da Vinci"
    CHAAC_STORY = "Chaac es la deidad maya de la lluvia."

    # --- GESTIÓN DE SESIÓN ---
    @staticmethod
    def initialize_session_state():
        keys = [
            "data_loaded", "apply_interpolation", "gdf_stations", "df_long",
            "df_enso", "gdf_municipios", "gdf_subcuencas", "gdf_predios",
            "unified_basin_gdf", "basin_results", "sarima_res", 
            "prophet_res", "res_cuenca", "current_coverage_stats",
            "dem_in_memory", "ppt_in_memory" # Agregados para caché de mapas
        ]
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None
