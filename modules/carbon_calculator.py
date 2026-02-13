# modules/carbon_calculator.py

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text
from modules.db_manager import get_engine

# --- CONSTANTES GLOBALES (Fuente: Valores_defecto.csv) ---
FACTOR_C_CO2 = 3.666667  # Ratio molecular 44/12
FRACCION_CARBONO = 0.47  # Fracción de carbono en biomasa
FACTOR_RAIZ_R = 0.24     # Relación Raíz/Vástago (Modelo_RN.csv)

@st.cache_data(ttl=3600)
def get_growth_params(scenario_id='Bosque_Humedo_Tropical'):
    """Obtiene parámetros A, b, m para Von Bertalanffy desde BD."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            query = text("SELECT * FROM carbon_growth_params WHERE scenario_id = :id")
            result = pd.read_sql(query, conn, params={"id": scenario_id})
            if not result.empty:
                return result.iloc[0]
    except Exception:
        pass
    # Fallback a valores del Excel "Modelo_RN.csv" si falla la BD
    return pd.Series({'param_a': 130.57, 'param_b': 0.091, 'param_m': 0.6666})

@st.cache_data(ttl=3600)
def get_soil_factor():
    """Obtiene la tasa anual de captura en suelo (dSOC) desde BD."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            query = text("SELECT value FROM carbon_soil_factors WHERE parameter = 'dSOC'")
            res = conn.execute(query).fetchone()
            if res: return res[0]
    except Exception:
        pass
    return 0.7050  # Valor por defecto del documento (tC/ha/año)

# --- 1. MODELO DE PROYECCIÓN (Von Bertalanffy) ---
def calcular_proyeccion_captura(hectareas, anios=30):
    """
    Proyecta la captura de carbono a futuro.
    Fórmula: Ct = A * [1 - exp(-b * t)] ^ (1 / (1-m))
    """
    # 1. Obtener parámetros
    params = get_growth_params()
    A = params['param_a']
    b = params['param_b']
    m = params['param_m'] # Usualmente 2/3, lo que hace que el exponente sea 3
    exponente = 1 / (1 - m)
    
    dSOC = get_soil_factor() # Captura suelo tC/ha/año

    # 2. Generar serie de tiempo
    years = np.arange(0, anios + 1)
    
    # 3. Calcular Stock de Biomasa (tC/ha) acumulado al año t
    # Nota: El modelo predice el STOCK acumulado, no el incremento anual directo
    stock_biomasa_c_ha = A * np.power((1 - np.exp(-b * years)), exponente)
    
    # 4. Calcular Incrementos Anuales (Delta)
    delta_biomasa_c_ha = np.diff(stock_biomasa_c_ha, prepend=0)
    
    # 5. Calcular Suelo (Lineal por 20 años según metodología MDL)
    delta_suelo_c_ha = np.where(years <= 20, dSOC, 0)
    delta_suelo_c_ha[0] = 0 # Año 0 no captura
    
    # 6. Consolidar DataFrame
    df = pd.DataFrame({
        'Año': years,
        'Stock_Biomasa_tC_ha': stock_biomasa_c_ha,
        'Captura_Anual_Biomasa_tC_ha': delta_biomasa_c_ha,
        'Captura_Anual_Suelo_tC_ha': delta_suelo_c_ha
    })
    
    # Conversiones a CO2e y Totales
    df['Captura_Total_tC_ha'] = df['Captura_Anual_Biomasa_tC_ha'] + df['Captura_Anual_Suelo_tC_ha']
    df['Captura_Total_tCO2e_ha'] = df['Captura_Total_tC_ha'] * FACTOR_C_CO2
    
    # Escalado al área del proyecto
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
