# modules/water_quality,py

import pandas as pd
import numpy as np

# ==============================================================================
# PARÁMETROS BASE DE CARGAS CONTAMINANTES (g / unidad / día)
# ==============================================================================

# 1. POBLACIÓN HUMANA (Reglamento Técnico del Sector Agua - RAS)
DBO_HAB_URBANO = 40.0   # gramos de DBO5 por habitante al día
SST_HAB_URBANO = 45.0   # gramos de Sólidos Suspendidos Totales
DBO_HAB_RURAL = 35.0    # Menor consumo de agua, carga ligeramente menor
SST_HAB_RURAL = 40.0

# 2. AGROINDUSTRIA PECUARIA (Valores típicos literatura sanitaria)
DBO_SUERO_LACTEO = 35000.0 # g DBO / m3 (35g por Litro de suero crudo)
SST_SUERO_LACTEO = 15000.0 # g SST / m3

DBO_CERDO_CONFINADO = 150.0 # g DBO / cerdo / día (incluye lavado de porqueriza)
SST_CERDO_CONFINADO = 200.0 

DBO_VACA_ORDENO = 80.0      # g DBO / vaca / día (solo efluente de sala de ordeño)
SST_VACA_ORDENO = 120.0

# 3. CARGAS DIFUSAS AGRÍCOLAS (Tasa de lavado superficial en kg/ha/año transformado a g/día)
# Asumiendo escorrentía promedio en zona andina
DBO_CULTIVO_LIMPIO = 15.0  # g DBO / ha / día (Papa, Hortalizas)
DBO_PASTO_FERTILIZADO = 5.0 # g DBO / ha / día
NUTRIENTES_N_PAPA = 12.0   # g Nitrógeno / ha / día
NUTRIENTES_P_PAPA = 3.0    # g Fósforo / ha / día

# ==============================================================================
# MOTORES DE CÁLCULO
# ==============================================================================

def calcular_cargas_organicas(pob_urb, pob_rur, ptar_cob, vol_suero_ldia, cerdos, vacas_ordeno, ha_papa, ha_pastos):
    """
    Calcula el inventario masivo de cargas orgánicas (DBO y SST) en kg/día.
    """
    # 1. Humanos (Aplicando reducción por PTAR a la urbana)
    remocion_ptar = ptar_cob / 100.0
    eficiencia_ptar_dbo = 0.85 # Una PTAR típica remueve 85% de DBO
    
    carga_urb_dbo = (pob_urb * DBO_HAB_URBANO * (1 - (remocion_ptar * eficiencia_ptar_dbo))) / 1000.0
    carga_rur_dbo = (pob_rur * DBO_HAB_RURAL) / 1000.0 # Asume in situ / descarga directa
    
    # 2. Pecuaria / Industrial
    carga_suero_dbo = (vol_suero_ldia * (DBO_SUERO_LACTEO / 1000.0)) / 1000.0 # Litros a g a kg
    carga_cerdos_dbo = (cerdos * DBO_CERDO_CONFINADO) / 1000.0
    carga_vacas_dbo = (vacas_ordeno * DBO_VACA_ORDENO) / 1000.0
    
    # 3. Agrícola Difusa
    carga_agri_dbo = ((ha_papa * DBO_CULTIVO_LIMPIO) + (ha_pastos * DBO_PASTO_FERTILIZADO)) / 1000.0
    
    # Estructuración de resultados
    df_cargas = pd.DataFrame({
        "Sector": ["Población Urbana", "Población Rural", "Industria Láctea", "Porcicultura", "Ganadería (Ordeño)", "Escorrentía Agrícola"],
        "Categoría": ["Doméstico", "Doméstico", "Industrial", "Agropecuario", "Agropecuario", "Difusa"],
        "DBO5_kg_dia": [carga_urb_dbo, carga_rur_dbo, carga_suero_dbo, carga_cerdos_dbo, carga_vacas_dbo, carga_agri_dbo]
    })
    
    return df_cargas

def calcular_streeter_phelps(L0, D0, T_agua, v_ms, H_m, dist_max_km=50, paso_km=0.5):
    """
    Simula la Curva de Caída de Oxígeno (Sag Curve) mediante el modelo Streeter-Phelps.
    
    Parámetros:
    - L0: DBO última de la mezcla en el punto de vertimiento (mg/L)
    - D0: Déficit inicial de oxígeno de la mezcla (mg/L)
    - T_agua: Temperatura del agua (°C)
    - v_ms: Velocidad media del río (m/s)
    - H_m: Profundidad media del río (m)
    - dist_max_km: Distancia aguas abajo a simular (km)
    
    Retorna:
    - DataFrame con la simulación espacial del Oxígeno Disuelto.
    """
    # 1. Constantes y Corrección por Temperatura
    # k1 (Tasa de Desoxigenación): Típicamente 0.23 d^-1 a 20°C para vertimientos mixtos
    k1_20 = 0.23 
    k1_T = k1_20 * (1.047 ** (T_agua - 20))
    
    # k2 (Tasa de Reaireación): Fórmula empírica de O'Connor-Dobbins
    # k2_20 = 3.93 * v^0.5 / H^1.5 (en base e, días^-1)
    k2_20 = (3.93 * (v_ms ** 0.5)) / (H_m ** 1.5) if H_m > 0 else 0.1
    k2_T = k2_20 * (1.024 ** (T_agua - 20))
    
    # Evitar singularidad matemática si k1 == k2
    if abs(k1_T - k2_T) < 0.001: 
        k2_T += 0.001
        
    # 2. Oxígeno Disuelto de Saturación (Fórmula empírica a nivel del mar)
    OD_sat = 14.652 - 0.41022 * T_agua + 0.007991 * (T_agua ** 2) - 0.000077774 * (T_agua ** 3)
    
    # 3. Vectorización Espacial y Temporal
    distancias_km = np.arange(0, dist_max_km + paso_km, paso_km)
    
    # Tiempo de viaje en días = (Distancia en metros) / (Metros recorridos por día)
    tiempos_dias = (distancias_km * 1000) / (v_ms * 86400) if v_ms > 0 else distancias_km * 0
    
    # 4. Ecuación Maestra de Streeter-Phelps
    # D(t) = (k1*L0 / (k2-k1)) * (e^(-k1*t) - e^(-k2*t)) + D0 * e^(-k2*t)
    term1 = (k1_T * L0) / (k2_T - k1_T)
    term2 = np.exp(-k1_T * tiempos_dias) - np.exp(-k2_T * tiempos_dias)
    term3 = D0 * np.exp(-k2_T * tiempos_dias)
    
    deficit_t = term1 * term2 + term3
    od_t = OD_sat - deficit_t
    
    # Blindaje ecológico: El oxígeno no puede ser negativo
    od_t = np.maximum(od_t, 0)
    
    # Ensamblar tabla de salida
    df_sag = pd.DataFrame({
        'Distancia_km': distancias_km,
        'Tiempo_dias': tiempos_dias,
        'Oxigeno_Disuelto_mgL': od_t,
        'Deficit_OD_mgL': deficit_t,
        'Limite_Critico': 4.0,  # 4 mg/L es el umbral para conservación de fauna acuática
        'OD_Saturacion': OD_sat
    })
    
    return df_sag
