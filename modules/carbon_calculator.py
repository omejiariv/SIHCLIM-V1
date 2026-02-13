# modules/carbon_calculator.py

import pandas as pd
import numpy as np
import streamlit as st

# --- PARÁMETROS CIENTÍFICOS (Fuente: Excel 'Modelo_RN' y 'Stand_I') ---
# Modelo Von Bertalanffy: B_t = A * (1 - exp(-k * t)) ^ (1 / (1 - m))

# Diccionario de los 10 Modelos del Excel
ESCENARIOS_CRECIMIENTO = {
    # --- GRUPO RESTAURACIÓN ACTIVA ---
    "STAND_I": {
        "nombre": "1. Modelo Stand I (Establecimiento 1667 ind/ha)",
        "A": 150.0, "k": 0.18, "m": 0.666, "tipo": "restauracion",
        "desc": "Alta densidad. Cierre rápido de dosel. Máxima captura inicial."
    },
    "STAND_II": {
        "nombre": "2. Modelo Stand II (Enriquecimiento 1000 ind/ha)",
        "A": 140.0, "k": 0.15, "m": 0.666, "tipo": "restauracion",
        "desc": "Densidad media. Balance entre costo y captura."
    },
    "STAND_III": {
        "nombre": "3. Modelo Stand III (Enriquecimiento 500 ind/ha)",
        "A": 130.0, "k": 0.12, "m": 0.666, "tipo": "restauracion",
        "desc": "Densidad baja. Apoyo a la regeneración."
    },
    "STAND_IV": {
        "nombre": "4. Modelo Stand IV (Aislamiento plántulas)",
        "A": 120.0, "k": 0.10, "m": 0.666, "tipo": "restauracion",
        "desc": "Protección de plántulas existentes. Crecimiento moderado."
    },
    # --- GRUPO RESTAURACIÓN PASIVA ---
    "STAND_V": {
        "nombre": "5. Modelo Stand V (Restauración Pasiva)",
        "A": 130.57, "k": 0.09, "m": 0.666, "tipo": "restauracion",
        "desc": "Sucesión natural. Sin siembra. Curva de crecimiento estándar."
    },
    # --- GRUPO SILVOPASTORIL / LINEAL ---
    "STAND_VI": {
        "nombre": "6. Modelo Stand VI (Cercas vivas 500 ind/km)",
        "A": 80.0, "k": 0.15, "m": 0.666, "tipo": "restauracion",
        "desc": "Arbolado lineal denso."
    },
    "STAND_VII": {
        "nombre": "7. Modelo Stand VII (Cercas vivas 167 ind/km)",
        "A": 40.0, "k": 0.12, "m": 0.666, "tipo": "restauracion",
        "desc": "Arbolado lineal espaciado."
    },
    "STAND_VIII": {
        "nombre": "8. Modelo Stand VIII (Árboles dispersos 20/ha)",
        "A": 25.0, "k": 0.10, "m": 0.666, "tipo": "restauracion",
        "desc": "Árboles en potrero. Baja carga de carbono por hectárea."
    },
    # --- GRUPO CONSERVACIÓN (Deforestación Evitada) ---
    "CONS_RIO": {
        "nombre": "9. Modelo Conservación Bosques Rio Grande II",
        "A": 277.8, "k": 0.0, "m": 0.0, "tipo": "conservacion", # Stock fijo alto
        "desc": "Bosque maduro. Se calcula el stock mantenido (evitar pérdida)."
    },
    "CONS_LAFE": {
        "nombre": "10. Modelo Conservación Bosques La FE",
        "A": 250.0, "k": 0.0, "m": 0.0, "tipo": "conservacion",
        "desc": "Bosque de niebla/alto andino. Conservación de stock."
    }
}

FACTOR_C_CO2 = 3.666667
DSOC_SUELO = 0.7050

def calcular_proyeccion_captura(hectareas, anios=30, escenario_key="STAND_V"):
    """
    Calcula la curva según el modelo seleccionado.
    Si es 'conservacion', proyecta una línea recta (Stock Almacenado).
    Si es 'restauracion', proyecta curva de crecimiento (Captura).
    """
    params = ESCENARIOS_CRECIMIENTO.get(escenario_key, ESCENARIOS_CRECIMIENTO["STAND_V"])
    
    A = params['A']
    tipo = params.get('tipo', 'restauracion')
    
    years = np.arange(0, anios + 1)
    
    if tipo == 'conservacion':
        # Modelo Conservación: El stock es constante (o decrece levemente si hubiera deforestación, 
        # pero aquí mostramos el valor de protegerlo).
        stock_biomasa_c_ha = np.full_like(years, A, dtype=float)
        delta_suelo_c_ha = np.zeros_like(years) # Suelo estable
    else:
        # Modelo Restauración (Von Bertalanffy)
        k = params['k']
        m = params['m']
        exponente = 1 / (1 - m)
        stock_biomasa_c_ha = A * np.power((1 - np.exp(-k * years)), exponente)
        
        # Suelo (Solo suma en restauración)
        delta_suelo_c_ha = np.where(years <= 20, DSOC_SUELO, 0)
        delta_suelo_c_ha[0] = 0

    # DataFrame
    df = pd.DataFrame({
        'Año': years,
        'Stock_Acumulado_tC_ha': stock_biomasa_c_ha
    })
    
    # Conversiones
    # Si es conservación, calculamos el stock total protegido
    # Si es restauración, es el stock ganado
    df['Stock_Total_tCO2e_ha'] = (df['Stock_Acumulado_tC_ha'] + np.cumsum(delta_suelo_c_ha)) * FACTOR_C_CO2
    
    # Escalado al Proyecto
    df['Proyecto_tCO2e_Acumulado'] = df['Stock_Total_tCO2e_ha'] * hectareas
    
    # Columna auxiliar para tasa anual (diferencia)
    df['Proyecto_tCO2e_Anual'] = df['Proyecto_tCO2e_Acumulado'].diff().fillna(0)
    
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
