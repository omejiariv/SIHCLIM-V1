# modules/carbon_calculator.py

import pandas as pd
import numpy as np
import streamlit as st

# --- PARÁMETROS CIENTÍFICOS (Fuente: Excel 'Modelo_RN' y 'Stand_I') ---
# Modelo Von Bertalanffy: B_t = A * (1 - exp(-k * t)) ^ (1 / (1 - m))

ESCENARIOS_CRECIMIENTO = {
    "ACTIVA_ALTA": {
        "nombre": "Restauración Activa (Alta Densidad)",
        "densidad": 1667, # árb/ha
        "A": 130.57,      # Asíntota (Biomasa máx t/ha) - Bosque Maduro
        "k": 0.15,        # Tasa de crecimiento (Rápida por intervención humana)
        "m": 0.6666,      # Constante alométrica (2/3)
        "desc": "Siembra densa (>1500 arb/ha). Cierre de dosel rápido."
    },
    "ACTIVA_MEDIA": {
        "nombre": "Restauración Activa (Enriquecimiento)",
        "densidad": 1000,
        "A": 130.57,
        "k": 0.091,       # Tasa base del documento (Stand II)
        "m": 0.6666,
        "desc": "Siembra media (~1000 arb/ha) o enriquecimiento."
    },
    "PASIVA": {
        "nombre": "Regeneración Natural (Sucesión)",
        "densidad": 0,    # No se siembra
        "A": 130.57,
        "k": 0.05,        # Tasa lenta (depende de dispersión natural)
        "m": 0.6666,
        "desc": "Proceso de sucesión natural. Bajo costo, inicio lento."
    }
}

FACTOR_C_CO2 = 3.666667
DSOC_SUELO = 0.7050  # Captura suelo tC/ha/año (hasta año 20)

def calcular_proyeccion_captura(hectareas, anios=30, escenario_key="ACTIVA_MEDIA"):
    """
    Proyecta captura considerando la estrategia de siembra/regeneración.
    """
    # 1. Obtener parámetros del escenario seleccionado
    params = ESCENARIOS_CRECIMIENTO.get(escenario_key, ESCENARIOS_CRECIMIENTO["ACTIVA_MEDIA"])
    
    A = params['A']
    k = params['k']
    m = params['m']
    exponente = 1 / (1 - m)

    # 2. Generar serie de tiempo
    years = np.arange(0, anios + 1)
    
    # 3. Calcular Stock Biomasa (Modelo Von Bertalanffy)
    # B(t) = A * [1 - exp(-k*t)] ^ (1/(1-m))
    stock_biomasa_c_ha = A * np.power((1 - np.exp(-k * years)), exponente)
    
    # 4. Calcular Incrementos (Captura Anual)
    delta_biomasa_c_ha = np.diff(stock_biomasa_c_ha, prepend=0)
    
    # 5. Calcular Suelo (Solo primeros 20 años)
    delta_suelo_c_ha = np.where(years <= 20, DSOC_SUELO, 0)
    delta_suelo_c_ha[0] = 0
    
    # 6. Consolidar
    df = pd.DataFrame({
        'Año': years,
        'Stock_Acumulado_tC_ha': stock_biomasa_c_ha,
        'Captura_Anual_Biomasa_tC_ha': delta_biomasa_c_ha,
        'Captura_Anual_Suelo_tC_ha': delta_suelo_c_ha
    })
    
    # Totales CO2e
    df['Captura_Total_tC_ha'] = df['Captura_Anual_Biomasa_tC_ha'] + df['Captura_Anual_Suelo_tC_ha']
    df['Captura_Total_tCO2e_ha'] = df['Captura_Total_tC_ha'] * FACTOR_C_CO2
    
    # Escalado al Proyecto
    df['Proyecto_tCO2e_Anual'] = df['Captura_Total_tCO2e_ha'] * hectareas
    df['Proyecto_tCO2e_Acumulado'] = df['Proyecto_tCO2e_Anual'].cumsum()
    
    return df

# --- 2. MODELO DE INVENTARIO (Álvarez et al. 2012) ---
def calcular_inventario_forestal(df, zona_vida_code='bh-MB'):
    """
    Calcula carbono actual basado en inventario (DAP, Altura).
    Ecuación: ln(BA) = a + c + ln(p * H * D^2)
    """
    # 1. Obtener coeficientes de BD
    try:
        engine = get_engine()
        with engine.connect() as conn:
            q = text("SELECT coefficient_a, coefficient_c FROM carbon_allometric_models WHERE life_zone_code = :z")
            res = conn.execute(q, {"z": zona_vida_code}).fetchone()
            if res:
                a, c = res
            else:
                a, c = -2.231, 0.933 # Default bh-MB
    except:
        a, c = -2.231, 0.933

    # 2. Validar columnas requeridas
    req_cols = ['DAP', 'Altura'] # DAP en cm, Altura en m
    if not all(col in df.columns for col in req_cols):
        return None, "El archivo debe tener columnas: 'DAP' (cm) y 'Altura' (m)."

    # 3. Densidad de madera (p)
    # Si no viene en el excel, usamos 0.6 g/cm3 (promedio latifoliadas suramérica)
    rho = df['Densidad'] if 'Densidad' in df.columns else 0.6
    
    # 4. Cálculo Biomasa Aérea (BA) en Toneladas
    # La ecuación original suele dar resultado en kg o ton dependiendo de los coeficientes.
    # Álvarez et al 2012 con estos coeficientes da BA en kg.
    
    # ln(BA_kg) = a + c + ln(rho * H * DAP^2)
    # BA_kg = exp(...)
    
    term_var = np.log(rho * df['Altura'] * (df['DAP']**2))
    ln_ba = a + c + term_var
    ba_kg = np.exp(ln_ba)
    
    # Conversión a Toneladas
    df['Biomasa_Aerea_ton'] = ba_kg / 1000
    
    # 5. Biomasa Subterránea (Raíces)
    # Usamos factor R (0.24) por defecto o ecuación si se prefiere.
    df['Biomasa_Raices_ton'] = df['Biomasa_Aerea_ton'] * FACTOR_RAIZ_R
    
    # 6. Carbono y CO2e
    df['Biomasa_Total_ton'] = df['Biomasa_Aerea_ton'] + df['Biomasa_Raices_ton']
    df['Carbono_Total_tC'] = df['Biomasa_Total_ton'] * FRACCION_CARBONO
    df['CO2e_Total_tCO2e'] = df['Carbono_Total_tC'] * FACTOR_C_CO2
    
    return df, "OK"
