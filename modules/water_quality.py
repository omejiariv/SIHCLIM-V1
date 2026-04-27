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

def calcular_streeter_phelps_multipunto(q_rio, t_rio, dbo_rio, od_rio, 
                                        v_ms, H_m, dist_max_km, paso_km,
                                        df_descargas, eq_hipso_res=None):
    """
    Simulación Avanzada Streeter-Phelps por Tramos (Piecewise).
    Calcula la caída de oxígeno asumiendo múltiples vertimientos a diferentes altitudes.
    """
    if df_descargas is None or df_descargas.empty:
        df_descargas = pd.DataFrame([{
            'Altitud (m)': 0, 'Caudal (m3/s)': 0, 'DBO (mg/L)': 0, 'Temp (°C)': t_rio
        }])

    # 1. Parámetros Cinéticos Base
    k1_20 = 0.35
    k2_20 = (3.9 * (v_ms ** 0.5)) / (H_m ** 1.5) if H_m > 0 else 0.1
    
    # 2. Vectorización Espacial
    distancias_km = np.arange(0, dist_max_km + paso_km, paso_km)
    
    q_actual = q_rio
    t_actual = t_rio
    od_actual = od_rio
    dbo_remanente_actual = dbo_rio
    
    # Listas de almacenamiento
    resultados_od = []
    resultados_dbo = []
    resultados_od_sat = []  # 🚀 FIX: Lista para guardar la saturación ideal
    
    km_descargas = {}
    km_actual_asignado = 0
    for idx, row in df_descargas.iterrows():
        km_descargas[km_actual_asignado] = row
        km_actual_asignado += max(paso_km, 5) 
        
    for dist in distancias_km:
        descarga_activa = None
        for km_vert, datos in km_descargas.items():
            if abs(dist - km_vert) < paso_km/2:
                descarga_activa = datos
                break
                
        if descarga_activa is not None:
            q_vert = descarga_activa['Caudal (m3/s)']
            t_vert = descarga_activa['Temp (°C)']
            dbo_vert = descarga_activa['DBO (mg/L)']
            od_vert = 0.0 
            
            q_mezcla = q_actual + q_vert
            if q_mezcla > 0:
                t_mezcla = ((q_actual * t_actual) + (q_vert * t_vert)) / q_mezcla
                dbo_mezcla = ((q_actual * dbo_remanente_actual) + (q_vert * dbo_vert)) / q_mezcla
                od_mezcla = ((q_actual * od_actual) + (q_vert * od_vert)) / q_mezcla
            else:
                t_mezcla, dbo_mezcla, od_mezcla = t_actual, dbo_remanente_actual, od_actual
                
            q_actual = q_mezcla
            t_actual = t_mezcla
            od_actual = od_mezcla
            L0 = dbo_mezcla
            
            # Cálculo exacto del OD de saturación por temperatura
            OD_sat_local = 14.652 - 0.41022 * t_actual + 0.007991 * (t_actual ** 2) - 0.000077774 * (t_actual ** 3)
            D0 = OD_sat_local - od_actual
            
            k1_T = k1_20 * (1.047 ** (t_actual - 20))
            k2_T = k2_20 * (1.024 ** (t_actual - 20))
            if abs(k1_T - k2_T) < 0.001: k2_T += 0.001
            
            tiempo_base = (dist * 1000) / (v_ms * 86400) if v_ms > 0 else 0
        else:
            # Aunque no haya descarga, actualizamos el OD local
            OD_sat_local = 14.652 - 0.41022 * t_actual + 0.007991 * (t_actual ** 2) - 0.000077774 * (t_actual ** 3)
        
        t_dias = (dist * 1000) / (v_ms * 86400) if v_ms > 0 else 0
        t_relativo = t_dias - tiempo_base 
        
        D_t = (k1_T * L0 / (k2_T - k1_T)) * (np.exp(-k1_T * t_relativo) - np.exp(-k2_T * t_relativo)) + D0 * np.exp(-k2_T * t_relativo)
        L_t = L0 * np.exp(-k1_T * t_relativo)
        
        OD_t = max(0, OD_sat_local - D_t)
        
        resultados_od.append(OD_t)
        resultados_dbo.append(L_t)
        resultados_od_sat.append(OD_sat_local) # 🚀 FIX: Guardamos el OD de Saturación
        
        dbo_remanente_actual = L_t
        od_actual_simulado = OD_t
        
    df_resultados = pd.DataFrame({
        'Distancia_km': distancias_km,
        'Oxigeno_Disuelto_mgL': resultados_od,
        'DBO_Remanente_mgL': resultados_dbo,
        'OD_Saturacion': resultados_od_sat, # 🚀 FIX: Lo empaquetamos y exportamos
        'Limite_Normativo': np.full_like(distancias_km, 4.0)
    })
    
    return df_resultados
