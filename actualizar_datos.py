# actualizar_datos.py

import pandas as pd
import requests
import os
import time
import numpy as np

def descargar_y_actualizar_ideam():
    print("Iniciando la descarga automatizada con enriquecimiento geográfico y control OMM...")
    inicio_total = time.time()
    
    DATASET_ID = "s54a-sgyg"
    URL_API = f"https://datos.gov.co/resource/{DATASET_ID}.json"
    ARCHIVO_SALIDA = "data/lluvia_mensual_consolidado.csv"
    ARCHIVO_ESTACIONES_NUEVAS = "data/estaciones_automaticas_catalogo.csv"
    
    departamentos = [
        "ANTIOQUIA", "CHOCO", "CHOCÓ", "CORDOBA", "CÓRDOBA", 'SUCRE', 'QUINDÍO', 
        "CALDAS", "TOLIMA", "BOYACA", "BOYACÁ", "SANTANDER", "RISARALDA", 'CUNDINAMARCA'
    ]
    
    deptos_query = ", ".join([f"'{d}'" for d in departamentos])
    
    select_clause = "codigoestacion, nombreestacion, departamento, municipio, latitud, longitud, fechaobservacion, valorobservado"
    where_clause = f"upper(departamento) IN ({deptos_query}) AND unidadmedida = 'mm'"
    
    limit = 100000  
    offset = 0
    resultados_parciales = []
    catalogo_estaciones = {}
    
    while True:
        query_url = f"{URL_API}?$select={select_clause}&$where={where_clause}&$limit={limit}&$offset={offset}"
        
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error al conectar con la API: {e}")
            break
            
        if not data:
            break 
            
        chunk_df = pd.DataFrame(data)
        
        # 1. 🕒 Tratamiento Temporal de Alta Precisión
        chunk_df['fechaobservacion'] = pd.to_datetime(chunk_df['fechaobservacion'], errors='coerce')
        chunk_df = chunk_df.dropna(subset=['fechaobservacion'])
        chunk_df['periodo_mensual'] = chunk_df['fechaobservacion'].dt.to_period('M')
        chunk_df['dia_exacto'] = chunk_df['fechaobservacion'].dt.date  # Extraemos el día para el control WMO
        
        # 2. 🛡️ Purga de anomalías a nivel de sensor (Lecturas individuales)
        chunk_df['valorobservado'] = pd.to_numeric(chunk_df['valorobservado'], errors='coerce')
        # Filtramos negativos (códigos de error) y picos absurdos en 1 lectura (ej. > 150mm en 10 min)
        chunk_df.loc[(chunk_df['valorobservado'] > 150) | (chunk_df['valorobservado'] < 0), 'valorobservado'] = np.nan
        
        # 3. 📍 Captura del Catálogo Geográfico
        for _, fila in chunk_df.dropna(subset=['latitud', 'longitud']).iterrows():
            cod = str(fila['codigoestacion'])
            if cod not in catalogo_estaciones:
                catalogo_estaciones[cod] = {
                    'codigo_estacion': cod,
                    'nombre_estacion': str(fila['nombreestacion']).strip(),
                    'municipio': str(fila['municipio']).strip().title(),
                    'departamento': str(fila['departamento']).strip().upper(),
                    'latitud': fila['latitud'],
                    'longitud': fila['longitud']
                }
        
        # 4. 🧮 Agrupación intradiaria del bloque
        # Agrupamos por DÍA exacto dentro de este chunk para no perder el rastro de la completitud
        chunk_agrupado = chunk_df.groupby(['codigoestacion', 'periodo_mensual', 'dia_exacto']).agg(
            lluvia_diaria=('valorobservado', 'sum')
        ).reset_index()
        
        resultados_parciales.append(chunk_agrupado)
        offset += limit
        print(f"📦 Descargados y pre-procesados {offset} registros históricos...")
        time.sleep(0.2)

    if resultados_parciales:
        print("Consolidando series, aplicando rigor estadístico y exportando...")
        df_total = pd.concat(resultados_parciales)
        
        # 5. 📊 AGREGACIÓN CIENTÍFICA FINAL
        df_consolidado = df_total.groupby(['codigoestacion', 'periodo_mensual']).agg(
            precipitacion=('lluvia_diaria', 'sum'),
            dias_con_datos=('dia_exacto', 'nunique') # Cuenta los días reales medidos en el mes
        ).reset_index()
        
        # --- 🛡️ FILTROS DE CALIDAD DEFINTIVOS ---
        
        # REGLA A: Completitud (Mínimo 20 días de registro para validar el mes)
        df_consolidado = df_consolidado[df_consolidado['dias_con_datos'] >= 20]
        
        # REGLA B: Purga Climática (Sin ceros falsos mensuales y sin diluvios irreales consolidados)
        df_consolidado = df_consolidado[
            (df_consolidado['precipitacion'] > 0.0) & 
            (df_consolidado['precipitacion'] <= 1500) # Ampliado a 1500 por seguridad extrema en Chocó/Antioquia
        ]
        
        df_consolidado['periodo_mensual'] = df_consolidado['periodo_mensual'].astype(str)
        
        # Preparar DataFrame final
        df_final = df_consolidado[['codigoestacion', 'periodo_mensual', 'precipitacion']].copy()
        df_final.columns = ['codigo_estacion', 'periodo_mensual', 'precipitacion']
        
        os.makedirs("data", exist_ok=True)
        
        # Guardar 1: CSV Principal Limpio
        df_final.to_csv(ARCHIVO_SALIDA, index=False, sep=',')
        
        # Guardar 2: Catálogo Geográfico
        if catalogo_estaciones:
            df_cat = pd.DataFrame(catalogo_estaciones.values())
            df_cat.to_csv(ARCHIVO_ESTACIONES_NUEVAS, index=False, sep=',')
            print(f"✅ Catálogo geográfico actualizado con éxito ({len(df_cat)} estaciones).")
        
        # ==========================================
        # 🔄 TRANSFORMACIÓN MATRICIAL
        # ==========================================
        print("Pivotando datos a formato matricial (Ancho)...")
        df_matriz = df_final.pivot(index='periodo_mensual', columns='codigo_estacion', values='precipitacion')
        
        df_matriz.index = pd.to_datetime(df_matriz.index).strftime('%Y-%m-%d')
        df_matriz.reset_index(inplace=True)
        df_matriz.rename(columns={'periodo_mensual': 'fecha'}, inplace=True)
        
        ARCHIVO_MATRIZ = "data/Pp_Automatica_IDEAM_Matriz.csv"
        df_matriz.to_csv(ARCHIVO_MATRIZ, index=False, sep=';', decimal=',')
        print(f"✅ Matriz ancha guardada lista para fusión en: {ARCHIVO_MATRIZ}")

        print(f"Pipeline finalizado con éxito. Tiempo de ejecución: {round((time.time() - inicio_total)/60, 1)} min.")
    else:
        print("Error: No se recuperaron datos válidos de la API.")

if __name__ == "__main__":
    descargar_y_actualizar_ideam()
