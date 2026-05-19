# modules/ideam_api.py

import pandas as pd
import requests
import streamlit as st

def extraer_datos_ideam(estaciones_ids, fecha_inicio):
    """
    Se conecta a la API de Datos Abiertos Colombia (IDEAM).
    Dataset de Precipitación: s54a-sgyg
    Usa SoQL para agrupar matemáticamente millones de registros de 10 min en totales mensuales.
    """
    url = "https://www.datos.gov.co/resource/s54a-sgyg.json"
    
    if not estaciones_ids:
        return False, "No se proporcionaron estaciones."

    # 🧹 LIMPIEZA FORENSE: Formato exacto
    estaciones_str = [str(e).replace(".0", "").split(".")[0].strip() for e in estaciones_ids]
    
    BATCH_SIZE = 30
    all_data = []
    
    progreso = st.progress(0, text="📡 IDEAM: Sumando millones de registros de 10 minutos en los servidores del Estado...")

    # Reemplaza la sección del bucle for en modules/ideam_api.py con esto:

    for i in range(0, len(estaciones_str), BATCH_SIZE):
        lote = estaciones_str[i:i + BATCH_SIZE]
        lista_sql = ",".join([f"'{e}'" for e in lote])
        
        query = {
            "$select": "codigoestacion, date_trunc_ym(fechaobservacion) as mes_ano, sum(valorobservado) as valor_mensual",
            "$where": f"codigoestacion in ({lista_sql}) AND fechaobservacion >= '{fecha_inicio}T00:00:00.000'",
            "$group": "codigoestacion, date_trunc_ym(fechaobservacion)",
            "$limit": 50000
        }
        
        # 🕵️‍♂️ MODO DIAGNÓSTICO: Construimos la URL visible para que el Capitán audite
        req = requests.Request('GET', url, params=query).prepare()
        if i == 0: # Solo mostramos la URL del primer lote para no inundar la pantalla
            st.info(f"🔗 **URL de Prueba IDEAM (Lote 1):**\n[Haz clic aquí para ver la respuesta cruda del Estado]({req.url})")
        
        try:
            response = requests.get(url, params=query, timeout=45)
            if response.status_code == 200:
                data = response.json()
                if data:
                    all_data.extend(data)
            else:
                print(f"Error IDEAM: {response.status_code}")
        except Exception as e:
            print(f"Fallo de red: {e}")
            
        progreso.progress(min((i + BATCH_SIZE) / len(estaciones_str), 1.0))

    progreso.empty()

    if not all_data:
        return False, "El IDEAM no retornó datos consolidados para estas fechas."

    # 1. Convertir JSON a DataFrame
    df = pd.DataFrame(all_data)
    
    # 2. Limpieza de columnas agrupadas
    df['fecha'] = pd.to_datetime(df['mes_ano'], errors='coerce')
    df['valor_mensual'] = pd.to_numeric(df['valor_mensual'], errors='coerce')
    df = df.dropna(subset=['fecha', 'valor_mensual'])

    # 3. Pivotear a Matriz Ancha
    df_ancha = df.pivot(index='fecha', columns='codigoestacion', values='valor_mensual').reset_index()
    df_ancha['fecha'] = df_ancha['fecha'].dt.strftime('%Y-%m-%d')

    return True, df_ancha
