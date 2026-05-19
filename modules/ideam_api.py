# modules/ideam_api.py

import pandas as pd
import requests
import streamlit as st

def extraer_datos_ideam(estaciones_ids, fecha_inicio):
    """
    Se conecta a la API de Datos Abiertos Colombia (IDEAM).
    Dataset de Precipitación: s54a-sgyg
    """
    url = "https://www.datos.gov.co/resource/s54a-sgyg.json"
    
    if not estaciones_ids:
        return False, "No se proporcionaron estaciones."

    # 🧹 LIMPIEZA FORENSE: Quitamos cualquier decimal '.0' y espacios extra
    estaciones_str = [str(e).replace(".0", "").split(".")[0].strip() for e in estaciones_ids]
    
    BATCH_SIZE = 40
    all_data = []
    
    progreso = st.progress(0, text="📡 Conectando con servidores del IDEAM (Datos Abiertos Colombia)...")

    for i in range(0, len(estaciones_str), BATCH_SIZE):
        lote = estaciones_str[i:i + BATCH_SIZE]
        lista_sql = ",".join([f"'{e}'" for e in lote])
        
        # Consulta SoQL (Socrata Query Language)
        query = {
            "$where": f"codigoestacion in ({lista_sql}) AND fechaobservacion >= '{fecha_inicio}T00:00:00.000'",
            "$limit": 50000
        }
        
        try:
            response = requests.get(url, params=query, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data:
                    all_data.extend(data)
            else:
                print(f"Rechazo del IDEAM: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error de red con IDEAM en lote {i}: {e}")
            continue
            
        progreso.progress(min((i + BATCH_SIZE) / len(estaciones_str), 1.0))

    progreso.empty()

    if not all_data:
        return False, "El IDEAM no retornó datos (Causa probable: Retraso de actualización institucional para años recientes)."

    # 1. Convertir JSON a DataFrame
    df = pd.DataFrame(all_data)
    
    # 🛡️ PROTECCIÓN ANTI-CAMBIOS: Si el IDEAM cambia el nombre de sus columnas
    col_fecha = 'fechaobservacion' if 'fechaobservacion' in df.columns else 'fecha'
    col_valor = 'valorobservado' if 'valorobservado' in df.columns else 'valor'
    col_est = 'codigoestacion' if 'codigoestacion' in df.columns else 'codigo_estacion'
    
    if col_fecha not in df.columns or col_valor not in df.columns:
        return False, f"La API del Estado cambió. Columnas recibidas: {list(df.columns)}"

    # 2. Limpieza de datos
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
    df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
    df = df.dropna(subset=[col_fecha, col_valor])

    # 3. Agrupar la lluvia diaria a total MENSUAL
    df['mes_año'] = df[col_fecha].dt.to_period('M')
    df_mensual = df.groupby(['mes_año', col_est])[col_valor].sum().reset_index()
    df_mensual['fecha'] = df_mensual['mes_año'].dt.to_timestamp()

    # 4. Pivotear a Matriz Ancha (Fechas en filas, Estaciones en columnas)
    df_ancha = df_mensual.pivot(index='fecha', columns=col_est, values=col_valor).reset_index()
    df_ancha['fecha'] = df_ancha['fecha'].dt.strftime('%Y-%m-%d')

    return True, df_ancha
