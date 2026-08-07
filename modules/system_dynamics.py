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
    """
    
    # ========================================================
    # 1. CONDICIONES INICIALES Y CONSTANTES DEL SISTEMA
    # ========================================================
    stock_humedad_suelo = 100.0  # Porcentaje (100% = saturado)
    
    # Estimación del volumen inicial del embalse/acuífero (30% de la lluvia del primer mes sobre el área)
    stock_embalse_hm3 = (precip_base_mensual[0] / 1000.0) * area_cuenca_km2 * 0.3 
    
    # Capacidad máxima del sistema hídrico (blindado contra ceros absolutos)
    vol_max_embalse = max(stock_embalse_hm3 * 1.5, 1.0) 
    
    # Demandas antrópicas fijas (Constantes)
    dotacion_lhd = 150 # Litros habitante día
    demanda_urbana_hm3_mes = (poblacion_servida * dotacion_lhd * 30) / 1e9
    demanda_rurh_hm3_mes = caudal_rurh_m3s * 2.592 # m3/s a Hm3/mes
    demanda_total_hm3 = demanda_urbana_hm3_mes + demanda_rurh_hm3_mes
    
    resultados = []
    
    # ========================================================
    # 2. BUCLE DE INTEGRACIÓN TEMPORAL (Paso a Paso)
    # ========================================================
    for t in range(meses_simulacion):
        mes_del_anio = (t % 12) + 1 
        
        # Inputs del mes
        oni_actual = oni_mensual[t]
        precip_historica = precip_base_mensual[t]
        
        # --- SUBSISTEMA 1: FORZAMIENTO CLIMÁTICO (Efecto ENSO) ---
        # Relación empírica: El Niño (+ ONI) seca y calienta. La Niña (- ONI) moja y enfría.
        factor_precip = max(0.1, 1.0 - (oni_actual * 0.30)) # Caída máx del 30% por cada grado ONI
        factor_temp = 1.0 + (oni_actual * 0.15)
        
        precip_simulada_mm = precip_historica * factor_precip
        temp_simulada_c = temp_base * factor_temp
        
        # Efecto Vientos de Agosto (Acelera propagación y secamiento)
        multiplicador_viento = 1.5 if mes_del_anio in [7, 8] else 1.0
        
        # --- SUBSISTEMA 2: HIDROLOGÍA Y BALANCE DE MASA ---
        # Fórmula de Turc simplificada para Evapotranspiración Real (ETR)
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
        
        # --- SUBSISTEMA 3: IMPACTOS EN CASCADA (Vulnerabilidades) ---
        
        # A. Déficit de Abastecimiento Humano
        indice_desabastecimiento = 0.0
        if demanda_total_hm3 > 0:
            umbral_alerta = demanda_total_hm3 * 2 # Alerta: Queda agua para < 2 meses
            if stock_embalse_hm3 < umbral_alerta: 
                indice_desabastecimiento = min(100.0, (umbral_alerta - stock_embalse_hm3) / umbral_alerta * 100)
            
        # B. Riesgo de Incendios Forestales y Biodiversidad
        # Aumenta drásticamente con baja humedad, alta temperatura y vientos (Agosto)
        indice_incendios = max(0, (40.0 - stock_humedad_suelo)) * 2.5
        indice_incendios = indice_incendios * (temp_simulada_c / temp_base) * multiplicador_viento
        indice_incendios = min(100.0, indice_incendios)
        
        # C. Calidad del Aire y Agua (Estrés Urbano / Inversión Térmica)
        # Menos lluvia = menor lavado atmosférico y menor dilución en ríos
        indice_estres_urbano = 100.0 - (precip_simulada_mm / (precip_historica + 1) * 100.0)
        if temp_simulada_c > 25: 
            indice_estres_urbano += (temp_simulada_c - 25) * 5
        indice_estres_urbano = max(0.0, min(100.0, indice_estres_urbano))
        
        # Guardar resultados del paso t (Usando float() puro para evitar errores de Plotly/JSON)
        resultados.append({
            "Mes": t + 1,
            "Mes_Anio": mes_del_anio,
            "ONI": float(oni_actual),
            "Precipitación (mm)": float(precip_simulada_mm),
            "Temperatura (°C)": float(temp_simulada_c),
            "Aporte Hídrico (Hm3)": float(aporte_cuenca_hm3),
            "Reservas (Hm3)": float(stock_embalse_hm3),
            "Humedad Suelo (%)": float(stock_humedad_suelo),
            "Riesgo Incendios (0-100)": float(indice_incendios),
            "Estrés Urbano/Calidad (0-100)": float(indice_estres_urbano),
            "Desabastecimiento (0-100)": float(indice_desabastecimiento)
        })
        
    return pd.DataFrame(resultados)
