# modules/system_dynamics.py

import pandas as pd
import numpy as np

def run_enso_system_dynamics(
    meses_simulacion, 
    oni_mensual, 
    precip_base_mensual, 
    temp_base,
    area_cuenca_km2,
    poblacion_servida,
    caudal_rurh_m3s=0.0
):
    """
    Motor de Dinámica de Sistemas para simular los efectos en cascada del ENSO
    sobre la hidrología, el riesgo de incendios y la vulnerabilidad urbana.
    
    Parámetros:
    - meses_simulacion: int, número de meses a proyectar.
    - oni_mensual: list o array con los valores del ONI para cada mes.
    - precip_base_mensual: list o array con la precipitación histórica promedio por mes (mm).
    - temp_base: float, temperatura media del territorio (°C).
    - area_cuenca_km2: float, área del territorio activo.
    - poblacion_servida: int, habitantes en el territorio.
    - caudal_rurh_m3s: float, extracciones antrópicas activas.
    """
    
    # 1. Variables de Estado (Stocks) Iniciales
    stock_humedad_suelo = 100.0  # Porcentaje (100% = saturado)
    stock_embalse_hm3 = (precip_base_mensual[0] / 1000.0) * area_cuenca_km2 * 0.3 # Estimación inicial
    vol_max_embalse = stock_embalse_hm3 * 1.5
    
    # Constantes físicas y antrópicas
    dotacion_lhd = 150 # Litros habitante día
    demanda_urbana_hm3_mes = (poblacion_servida * dotacion_lhd * 30) / 1e9
    demanda_rurh_hm3_mes = caudal_rurh_m3s * 2.592 # m3/s a Hm3/mes
    demanda_total_hm3 = demanda_urbana_hm3_mes + demanda_rurh_hm3_mes
    
    resultados = []
    
    for t in range(meses_simulacion):
        # Mes actual (1 a 12 ciclico)
        mes_del_anio = (t % 12) + 1 
        
        # Inputs del mes
        oni_actual = oni_mensual[t]
        precip_historica = precip_base_mensual[t]
        
        # ========================================================
        # SUBSISTEMA 1: FORZAMIENTO CLIMÁTICO (Efecto ENSO)
        # ========================================================
        # Relación empírica: El Niño (+ ONI) seca y calienta. La Niña (- ONI) moja y enfría.
        factor_precip = max(0.1, 1.0 - (oni_actual * 0.30)) # Caída del 30% por cada grado ONI
        factor_temp = 1.0 + (oni_actual * 0.15)
        
        precip_simulada_mm = precip_historica * factor_precip
        temp_simulada_c = temp_base * factor_temp
        
        # Efecto Vientos de Agosto
        multiplicador_viento = 1.5 if mes_del_anio in [7, 8] else 1.0
        
        # ========================================================
        # SUBSISTEMA 2: HIDROLOGÍA Y HUMEDAD
        # ========================================================
        # Fórmula de Turc simplificada para ET
        L_t = 300 + 25 * temp_simulada_c + 0.05 * (temp_simulada_c ** 3)
        etr_mm = precip_simulada_mm / np.sqrt(0.9 + (precip_simulada_mm / L_t) ** 2) if L_t > 0 else 0
        etr_mm = min(etr_mm, precip_simulada_mm) * multiplicador_viento
        
        escorrentia_mm = max(0, precip_simulada_mm - etr_mm)
        aporte_cuenca_hm3 = (escorrentia_mm / 1000.0) * area_cuenca_km2
        
        # Dinámica del Suelo (Integración del Stock)
        infiltracion = precip_simulada_mm * 0.2
        evaporacion_suelo = temp_simulada_c * 2.0 * multiplicador_viento
        delta_humedad = infiltracion - evaporacion_suelo
        stock_humedad_suelo = max(0.0, min(100.0, stock_humedad_suelo + delta_humedad))
        
        # Dinámica del Embalse/Reserva (Integración del Stock)
        delta_embalse = aporte_cuenca_hm3 - demanda_total_hm3
        stock_embalse_hm3 = max(0.0, min(vol_max_embalse, stock_embalse_hm3 + delta_embalse))
        
        # ========================================================
        # SUBSISTEMA 3: IMPACTOS EN CASCADA (Vulnerabilidades)
        # ========================================================
        # A. Déficit de Abastecimiento
        indice_desabastecimiento = 0.0
        if stock_embalse_hm3 < (demanda_total_hm3 * 2): # Alerta si queda agua para < 2 meses
            indice_desabastecimiento = min(100.0, ((demanda_total_hm3 * 2) - stock_embalse_hm3) / (demanda_total_hm3 * 2) * 100)
            
        # B. Riesgo de Incendios Forestales
        # Aumenta drásticamente con baja humedad, alta temperatura y vientos (Agosto)
        indice_incendios = max(0, (40.0 - stock_humedad_suelo)) * 2.5
        indice_incendios = indice_incendios * (temp_simulada_c / temp_base) * multiplicador_viento
        indice_incendios = min(100.0, indice_incendios)
        
        # C. Calidad del Agua y del Aire (Estrés Urbano)
        # Menos lluvia = menor lavado atmosférico y menor dilución en ríos
        indice_estres_urbano = 100.0 - (precip_simulada_mm / (precip_historica + 1) * 100.0)
        if temp_simulada_c > 25:
             indice_estres_urbano += (temp_simulada_c - 25) * 5
        indice_estres_urbano = min(100.0, indice_estres_urbano)
        
        # Guardar resultados del paso t
        resultados.append({
            "Mes": t + 1,
            "Mes_Anio": mes_del_anio,
            "ONI": oni_actual,
            "Precipitación (mm)": precip_simulada_mm,
            "Temperatura (°C)": temp_simulada_c,
            "Aporte Hídrico (Hm3)": aporte_cuenca_hm3,
            "Reservas (Hm3)": stock_embalse_hm3,
            "Humedad Suelo (%)": stock_humedad_suelo,
            "Riesgo Incendios (0-100)": indice_incendios,
            "Estrés Urbano/Calidad (0-100)": indice_estres_urbano,
            "Desabastecimiento (0-100)": indice_desabastecimiento
        })
        
    return pd.DataFrame(resultados)
