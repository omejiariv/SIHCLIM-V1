# modules/system_dynamics.py

import pandas as pd
import numpy as np

def run_enso_system_dynamics_prophet(
    df_proyeccion, 
    oni_mensual, 
    temp_base,
    area_cuenca_km2,
    poblacion_servida,
    caudal_rurh_m3s=0.0
):
    """
    Motor de Dinámica de Sistemas de 2da Generación.
    Utiliza como Línea Base la proyección estocástica de Prophet y aplica 
    física mecanicista para simular el estrés del ENSO sobre la recarga de acuíferos.
    """
    meses_simulacion = len(df_proyeccion)
    
    # ========================================================
    # 1. CONDICIONES INICIALES Y CONSTANTES
    # ========================================================
    stock_humedad_suelo = 100.0  
    
    # Asumimos una reserva base a partir de la recarga del primer mes
    recarga_inicial_hm3 = (df_proyeccion.iloc[0]['recarga_mm'] / 1000.0) * area_cuenca_km2
    stock_embalse_hm3 = max(recarga_inicial_hm3 * 12, 1.0) # Reserva para un año aprox
    vol_max_embalse = stock_embalse_hm3 * 1.5 
    
    dotacion_lhd = 150 # Litros habitante día
    demanda_urbana_hm3_mes = (poblacion_servida * dotacion_lhd * 30) / 1e9
    demanda_rurh_hm3_mes = caudal_rurh_m3s * 2.592 
    demanda_total_hm3 = demanda_urbana_hm3_mes + demanda_rurh_hm3_mes
    
    resultados = []
    
    # ========================================================
    # 2. BUCLE DE INTEGRACIÓN TEMPORAL (Usando Prophet)
    # ========================================================
    for t in range(meses_simulacion):
        row = df_proyeccion.iloc[t]
        fecha_actual = row['fecha']
        mes_del_anio = fecha_actual.month
        
        oni_actual = oni_mensual[t]
        
        # Leemos la LÍNEA BASE (Pronóstico Prophet sin ENSO)
        precip_prophet_mm = row['p_final']
        recarga_prophet_mm = row['recarga_mm']
        infilt_prophet_mm = row['infiltracion_mm']
        
        # --- SUBSISTEMA 1: ESTRÉS CLIMÁTICO (ENSO) ---
        factor_precip = max(0.1, 1.0 - (oni_actual * 0.30)) 
        factor_temp = 1.0 + (oni_actual * 0.15)
        
        precip_simulada_mm = precip_prophet_mm * factor_precip
        temp_simulada_c = temp_base * factor_temp
        multiplicador_viento = 1.5 if mes_del_anio in [7, 8] else 1.0
        
        # --- SUBSISTEMA 2: HIDROGEOLOGÍA AFECTADA ---
        # Si llueve menos por El Niño, la recarga y la infiltración caen proporcionalmente
        recarga_simulada_mm = recarga_prophet_mm * factor_precip
        infilt_simulada_mm = infilt_prophet_mm * factor_precip
        
        aporte_cuenca_hm3 = (recarga_simulada_mm / 1000.0) * area_cuenca_km2
        
        # Dinámica de Suelos
        evaporacion_suelo = temp_simulada_c * 2.0 * multiplicador_viento
        delta_humedad = infilt_simulada_mm - evaporacion_suelo
        stock_humedad_suelo = max(0.0, min(100.0, stock_humedad_suelo + delta_humedad))
        
        # Dinámica de Reservas (Acuífero/Embalse)
        delta_embalse = aporte_cuenca_hm3 - demanda_total_hm3
        stock_embalse_hm3 = max(0.0, min(vol_max_embalse, stock_embalse_hm3 + delta_embalse))
        
        # --- SUBSISTEMA 3: CASCADA DE IMPACTOS ---
        indice_desabastecimiento = 0.0
        if demanda_total_hm3 > 0:
            umbral_alerta = demanda_total_hm3 * 3 # Alerta si queda agua para < 3 meses
            if stock_embalse_hm3 < umbral_alerta: 
                indice_desabastecimiento = min(100.0, (umbral_alerta - stock_embalse_hm3) / umbral_alerta * 100)
            
        # 🚀 FIX FÍSICO: Incendios más sensibles a caídas de humedad (empieza a alertar < 80%)
        indice_incendios = max(0, (80.0 - stock_humedad_suelo)) * 2.5
        indice_incendios = indice_incendios * (temp_simulada_c / temp_base) * multiplicador_viento
        indice_incendios = min(100.0, indice_incendios)
        
        # 🚀 FIX FÍSICO: Estrés Urbano Relativo (Calidad del aire y ola de calor)
        # 1. Estrés por falta de lluvia (no hay lavado de la atmósfera)
        estres_lluvia = max(0, 100.0 - (precip_simulada_mm / (precip_prophet_mm + 0.1) * 100.0))
        # 2. Estrés térmico (anomalía de temperatura: +1°C sobre lo normal suma 20 puntos de estrés)
        estres_termico = max(0, (temp_simulada_c - temp_base) * 20.0) 
        
        indice_estres_urbano = max(0.0, min(100.0, estres_lluvia + estres_termico))
        
        # Acumulación de Déficit de Recarga (La pérdida real de agua subterránea)
        deficit_recarga_mm = recarga_prophet_mm - recarga_simulada_mm
        
        resultados.append({
            "Fecha": fecha_actual, 
            "Mes_Anio": mes_del_anio,
            "ONI": float(oni_actual), 
            "Precipitación (mm)": float(precip_simulada_mm),
            "Recarga Acuífero (mm)": float(recarga_simulada_mm),
            "Pérdida Recarga (mm)": float(max(0, deficit_recarga_mm)), 
            "Temperatura (°C)": float(temp_simulada_c), 
            "Aporte Hídrico (Hm3)": float(aporte_cuenca_hm3), # 🚀 FIX: Columna inyectada para el Waterfall
            "Reservas (Hm3)": float(stock_embalse_hm3), 
            "Humedad Suelo (%)": float(stock_humedad_suelo),
            "Riesgo Incendios (0-100)": float(indice_incendios),
            "Estrés Urbano (0-100)": float(indice_estres_urbano),
            "Desabastecimiento (0-100)": float(indice_desabastecimiento)
        })
        
    return pd.DataFrame(resultados)
