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
