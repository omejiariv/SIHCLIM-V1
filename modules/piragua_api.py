# modules/piragua_api.py

import requests
import pandas as pd
import streamlit as st

def extraer_datos_piragua(url_oculta, params=None):
    """
    Se conecta al backend oculto del Geoportal de Piragua.
    Transforma la respuesta JSON en una matriz hidrológica mensual.
    """
    try:
        # Hacemos la petición fingiendo ser un navegador normal
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        response = requests.get(url_oculta, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        datos_json = response.json()
        
        # 1. Dependiendo de si usan ArcGIS o una API propia, la ruta del JSON cambia.
        # Intentamos detectar la estructura de la lista de datos:
        if 'features' in datos_json: # Estructura típica de ArcGIS REST
            registros = [feat['attributes'] for feat in datos_json['features']]
        elif 'data' in datos_json: # Estructura típica de APIs modernas
            registros = datos_json['data']
        elif isinstance(datos_json, list): # Si es una lista plana
            registros = datos_json
        else:
            return False, "Estructura JSON no reconocida. Revisa el Endpoint."

        if not registros:
            return False, "La API respondió, pero no hay datos para estas fechas."

        # 2. Convertimos a DataFrame
        df = pd.DataFrame(registros)
        
        # NOTA PARA EL CAPITÁN: Aquí debemos poner los nombres exactos de las columnas 
        # que Piragua devuelve en su JSON. Por ahora usamos nombres genéricos.
        col_fecha = 'fecha' if 'fecha' in df.columns else df.columns[0] # Asumimos la primera
        col_estacion = 'id_estacion' if 'id_estacion' in df.columns else df.columns[1]
        col_lluvia = 'precipitacion' if 'precipitacion' in df.columns else df.columns[2]

        # 3. Limpieza y Agrupación Mensual (Idéntico a Open-Meteo)
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        df[col_lluvia] = pd.to_numeric(df[col_lluvia], errors='coerce')
        df = df.dropna(subset=[col_fecha, col_lluvia])

        # Agrupar por mes y estación sumando la lluvia
        df['mes_año'] = df[col_fecha].dt.to_period('M')
        df_mensual = df.groupby(['mes_año', col_estacion])[col_lluvia].sum().reset_index()
        df_mensual['fecha'] = df_mensual['mes_año'].dt.to_timestamp()

        # 4. Pivoteo (Matriz Ancha) para que encaje en el Generador
        df_ancha = df_mensual.pivot(index='fecha', columns=col_estacion, values=col_lluvia).reset_index()
        df_ancha['fecha'] = df_ancha['fecha'].dt.strftime('%Y-%m-%d')

        return True, df_ancha

    except Exception as e:
        return False, f"Error de conexión: {str(e)}"
