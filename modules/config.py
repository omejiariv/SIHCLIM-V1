import streamlit as st
import pandas as pd
import os

# Define la ruta base del proyecto de forma robusta
# Asume que los archivos de datos están en 'data' y los módulos en 'modules'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Nombres de Columnas de Datos
    STATION_NAME_COL = 'nom_est'
    PRECIPITATION_COL = 'precipitation'
    LATITUDE_COL = 'latitud_geo'
    LONGITUDE_COL = 'longitud_geo'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    DATE_COL = 'fecha_mes_año'
    ENSO_ONI_COL = 'anomalia_oni'
    ORIGIN_COL = 'origen'
    ALTITUDE_COL = 'alt_est'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    CELL_COL = 'celda_xy'
    # Índices climáticos
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    # Rutas de Archivos (usando la ruta absoluta)
    LOGO_PATH = os.path.join(BASE_DIR, "data", "Cuenca Verde Logo_V1.JPG")
    LOGO_DROP_PATH = os.path.join(BASE_DIR, "data", "Cuenca Verde Gotica Logo.JPG")
    GIF_PATH = os.path.join(BASE_DIR, "data", "PPAM.gif")
    
    # Mensajes de la UI
    APP_TITLE = "Sistema de información de las lluvias y el Clima en el norte de la región Andina"
    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su
    relación con el fenómeno ENSO en el norte de la región Andina.
    **¿Cómo empezar?**
    1. **Cargue sus archivos**: Si es la primera vez que usa la aplicación, el panel de la izquierda le
    solicitará cargar los archivos de estaciones,
    precipitación y el shapefile de municipios. La aplicación recordará estos archivos en su sesión.
    2. **Filtre los datos**: Una vez cargados los datos, utilice el **Panel de Control** en la barra
    lateral para filtrar las estaciones por ubicación (región, municipio), altitud,
    porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
    3. **Explore las pestañas**: Cada pestaña ofrece una perspectiva diferente de los datos.
    Navegue a través de ellas para descubrir:
    - **Distribución Espacial**: Mapas interactivos de las estaciones.
    - **Gráficos**: Series de tiempo anuales, mensuales, comparaciones y distribuciones.
    - **Mapas Avanzados**: Animaciones y mapas de interpolación.
    - **Análisis de Anomalías**: Desviaciones de la precipitación respecto a la media histórica.
    - **Tendencias y Pronósticos**: Análisis de tendencias a largo plazo y modelos de pronóstico.
    Utilice el botón **/ Limpiar Filtros** en el panel lateral para reiniciar su selección en cualquier
    momento.
    """

    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables necesarias en el estado de la sesión de Streamlit."""
        state_defaults = {
            'data_loaded': False,
            'analysis_mode': "Usar datos originales",
            'select_all_stations_state': False,
            'df_monthly_processed': pd.DataFrame(),
            'gdf_stations': None,
            'df_precip_anual': None,
            'gdf_municipios': None,
            'df_long': None,
            'df_enso': None,
            'min_data_perc_slider': 0,
            'altitude_multiselect': [],
            'regions_multiselect': [],
            'municipios_multiselect': [],
            'celdas_multiselect': [],
            'station_multiselect': [],
            'exclude_na': False,
            'exclude_zeros': False,
            'uploaded_forecast': None,
            'sarima_forecast': None,
            'prophet_forecast': None
        }
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
