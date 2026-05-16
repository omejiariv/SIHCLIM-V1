# modules/sync_hydro.py

import pandas as pd
import requests
import datetime
import streamlit as st
from modules import db_manager

def obtener_datos_satelitales_era5(lat, lon, fecha_inicio, fecha_fin):
    """
    Se conecta a la Agencia Espacial Europea (Copernicus) vía Open-Meteo
    para obtener datos climáticos históricos diarios y los agrupa por mes.
    Variables: Precipitación, ETR, Radiación Solar, Temperatura.
    """
    # API gratuita, sin necesidad de API Keys complejas, ultra estable
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "daily": ["precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_mean", "shortwave_radiation_sum"],
        "timezone": "America/Bogota"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Construimos el DataFrame diario
        df_daily = pd.DataFrame({
            "fecha": pd.to_datetime(data["daily"]["time"]),
            "precipitacion_mm": data["daily"]["precipitation_sum"],
            "etr_mm": data["daily"]["et0_fao_evapotranspiration"],
            "temp_media_c": data["daily"]["temperature_2m_mean"],
            "rad_solar_mj_m2": data["daily"]["shortwave_radiation_sum"]
        })
        
        # Agrupamos a nivel mensual (para empatar con tu archivo histórico)
        df_daily['mes_año'] = df_daily['fecha'].dt.to_period('M')
        df_monthly = df_daily.groupby('mes_año').agg({
            "precipitacion_mm": "sum", # Lluvia se suma
            "etr_mm": "sum",           # ETR se suma
            "temp_media_c": "mean",    # Temperatura se promedia
            "rad_solar_mj_m2": "sum"   # Radiación se suma
        }).reset_index()
        
        df_monthly['fecha'] = df_monthly['mes_año'].dt.to_timestamp()
        return df_monthly
        
    except Exception as e:
        return None

def actualizar_estaciones_bache_reciente():
    """
    Robot automatizado: 
    1. Busca las coordenadas de las estaciones en tu base de datos.
    2. Descarga el "bache" de información (2021 a HOY).
    """
    engine = db_manager.get_engine()
    
    # 1. Obtener lista de estaciones y sus coordenadas (ejemplo: tomamos 5 para no saturar en pruebas)
    try:
        df_estaciones = pd.read_sql("SELECT id_estacion, latitud, longitud FROM estaciones WHERE latitud IS NOT NULL LIMIT 5", engine)
    except:
        return False, "No se pudo conectar a la tabla de estaciones."
        
    fecha_inicio = "2021-01-01"
    fecha_fin = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d') # Datos hasta la semana pasada
    
    progreso = st.progress(0)
    estado = st.empty()
    
    resultados_lluvia = {}
    
    for i, row in df_estaciones.iterrows():
        id_est = str(row['id_estacion'])
        lat = row['latitud']
        lon = row['longitud']
        
        estado.text(f"📡 Satélite apuntando a estación {id_est} (Lat: {lat}, Lon: {lon})...")
        
        df_mes = obtener_datos_satelitales_era5(lat, lon, fecha_inicio, fecha_fin)
        
        if df_mes is not None and not df_mes.empty:
            resultados_lluvia[id_est] = df_mes[['fecha', 'precipitacion_mm']].set_index('fecha')['precipitacion_mm']
            
        progreso.progress((i + 1) / len(df_estaciones))
        
    estado.text("✅ Descarga satelital completada.")
    
    if resultados_lluvia:
        # Construimos un DataFrame con la misma estructura ancha que tu CSV actual
        df_nuevo_tramo = pd.DataFrame(resultados_lluvia)
        df_nuevo_tramo.reset_index(inplace=True)
        return True, df_nuevo_tramo
    else:
        return False, "No se obtuvieron datos."
