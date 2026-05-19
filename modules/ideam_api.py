# modules/ideam_api.py

import pandas as pd
import requests
import streamlit as st

def extraer_datos_ideam(estaciones_ids, fecha_inicio):
    """
    Se conecta a la API de Datos Abiertos Colombia (IDEAM).
    Dataset de Precipitación: s54a-sgyg
    """
    # Endpoint oficial de precipitación del IDEAM
    url = "https://www.datos.gov.co/resource/s54a-sgyg.json"
    
    if not estaciones_ids:
        return False, "No se proporcionaron estaciones."

    # Convertir IDs a strings para la consulta
    estaciones_str = [str(e) for e in estaciones_ids]
    
    # Procesar en lotes de 50 estaciones para no saturar la URL
    BATCH_SIZE = 50
    all_data = []
    
    progreso = st.progress(0, text="📡 Conectando con servidores del IDEAM (Datos Abiertos Colombia)...")

    for i in range(0, len(estaciones_str), BATCH_SIZE):
        lote = estaciones_str[i:i + BATCH_SIZE]
        lista_sql = ",".join([f"'{e}'" for e in lote])
        
        # Consulta SoQL (Socrata Query Language)
        # Filtramos por código de estación y fecha
        query = {
            "$where": f"codigoestacion in ({lista_sql}) AND fechaobservacion >= '{fecha_inicio}T00:00:00.000'",
            "$limit": 50000, # Límite alto para asegurar que lleguen todos los días
            "$select": "codigoestacion, fechaobservacion, valorobservado"
        }
        
        try:
            response = requests.get(url, params=query, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data:
                    all_data.extend(data)
        except Exception as e:
            print(f"Error consultando IDEAM en lote {i}: {e}")
            continue
            
        progreso.progress(min((i + BATCH_SIZE) / len(estaciones_str), 1.0))

    progreso.empty()

    if not all_data:
        return False, "El IDEAM no retornó datos nuevos para estas estaciones y fechas."

    # 1. Convertir JSON a DataFrame
    df = pd.DataFrame(all_data)
    
    # 2. Limpieza de datos
    df['fechaobservacion'] = pd.to_datetime(df['fechaobservacion'], errors='coerce')
    df['valorobservado'] = pd.to_numeric(df['valorobservado'], errors='coerce')
    df = df.dropna(subset=['fechaobservacion', 'valorobservado'])

    # 3. Agrupar la lluvia diaria a total MENSUAL
    df['mes_año'] = df['fechaobservacion'].dt.to_period('M')
    df_mensual = df.groupby(['mes_año', 'codigoestacion'])['valorobservado'].sum().reset_index()
    df_mensual['fecha'] = df_mensual['mes_año'].dt.to_timestamp()

    # 4. Pivotear a Matriz Ancha (Fechas en filas, Estaciones en columnas)
    df_ancha = df_mensual.pivot(index='fecha', columns='codigoestacion', values='valorobservado').reset_index()
    df_ancha['fecha'] = df_ancha['fecha'].dt.strftime('%Y-%m-%d')

    return True, df_ancha
