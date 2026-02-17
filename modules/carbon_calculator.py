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

# ==============================================================================
# MÓDULO AFOLU: GANADERÍA Y PASTURAS (IPCC Nivel 1 - América Latina)
# ==============================================================================

# --- 1. PARÁMETROS GLOBALES IPCC ---
# Potenciales de Calentamiento Global (AR5)
GWP_CH4 = 28   # 1 tonelada de Metano = 28 tCO2e
GWP_N2O = 265  # 1 tonelada de Óxido Nitroso = 265 tCO2e

# Factores de Emisión: Fermentación Entérica (kg CH4 / cabeza / año)
# Fuente: IPCC 2006, Vol 4, Cap 10, Tabla 10.11 (América Latina)
# Factores de Emisión (kg CH4 / cabeza / año) - IPCC 2006 América Latina
EF_ENTERIC_LECHE = 72.0  # Vacas lecheras 
EF_ENTERIC_CARNE = 56.0  # Otro tipo de ganado (cría, levante, ceba, doble propósito)
EF_ENTERIC_CERDOS = 1.5  # Fermentación entérica baja en monogástricos
EF_ENTERIC_AVES = 0.0    # Despreciable

# Factores de Emisión: Gestión de Estiércol en Pasturas (kg CH4 / cabeza / año)
# Fuente: IPCC 2006, Vol 4, Cap 10, Tabla 10.14 (Clima cálido/templado)
# Gestión de Estiércol en Pasturas/Corrales (kg CH4 / cabeza / año)
EF_ESTIERCOL_CH4_LECHE = 2.0  
EF_ESTIERCOL_CH4_CARNE = 1.0
EF_ESTIERCOL_CH4_CERDOS = 4.0 # Alto si hay lagunas de oxidación / pozos
EF_ESTIERCOL_CH4_AVES = 0.02

# Óxido Nitroso (N2O) por Orina/Estiércol (kg N2O / cabeza / año)
EF_ESTIERCOL_N2O_LECHE = 1.5 
EF_ESTIERCOL_N2O_CARNE = 1.2
EF_ESTIERCOL_N2O_CERDOS = 0.2
EF_ESTIERCOL_N2O_AVES = 0.001

# --- 2. ESCENARIOS DE PASTURAS Y SUELOS ---
# Captura o pérdida de Carbono Orgánico del Suelo (COS) en tC/ha/año
ESCENARIOS_PASTURAS = {
    "PASTO_DEGRADADO": {
        "nombre": "1. Pasto Degradado (Línea Base)",
        "tasa_c_ha_anio": -0.5, # Emisor: Pierde carbono anualmente por erosión/compactación
        "desc": "Pasturas sobrepastoreadas. Emite carbono del suelo."
    },
    "PASTO_MANEJADO": {
        "nombre": "2. Pasto Mejorado (Manejo Rotacional)",
        "tasa_c_ha_anio": 0.8,  # Sumidero: Gana carbono
        "desc": "Pasturas con buen manejo, descanso adecuado y sin sobrecarga."
    },
    "SSP_BAJO": {
        "nombre": "3. Silvopastoril (Baja Densidad)",
        "tasa_c_ha_anio": 1.2,  # Sumidero mayor por raíces profundas
        "desc": "Arreglo silvopastoril con árboles dispersos en potrero."
    },
    "SSP_INTENSIVO": {
        "nombre": "4. Silvopastoril Intensivo (SSPi)",
        "tasa_c_ha_anio": 2.5,  # Sumidero alto (Suelo + Arbustos forrajeros)
        "desc": "SSPi con alta densidad de arbustos (ej. Leucaena, Botón de Oro) y maderables."
    }
}

# --- 3. MOTORES DE CÁLCULO ---

def calcular_emisiones_ganaderia(vacas_leche, vacas_carne, cerdos=0, aves=0, anios=20):
    """
    Calcula las emisiones acumuladas de Gases de Efecto Invernadero (CH4 y N2O)
    generadas por bovinos, porcinos y aves.
    """
    # 1. Metano (CH4) en kg
    ch4_leche = vacas_leche * (EF_ENTERIC_LECHE + EF_ESTIERCOL_CH4_LECHE)
    ch4_carne = vacas_carne * (EF_ENTERIC_CARNE + EF_ESTIERCOL_CH4_CARNE)
    ch4_cerdos = cerdos * (EF_ENTERIC_CERDOS + EF_ESTIERCOL_CH4_CERDOS)
    ch4_aves = aves * (EF_ENTERIC_AVES + EF_ESTIERCOL_CH4_AVES)
    total_ch4_kg_anio = ch4_leche + ch4_carne + ch4_cerdos + ch4_aves
    
    # 2. Óxido Nitroso (N2O) en kg
    n2o_leche = vacas_leche * EF_ESTIERCOL_N2O_LECHE
    n2o_carne = vacas_carne * EF_ESTIERCOL_N2O_CARNE
    n2o_cerdos = cerdos * EF_ESTIERCOL_N2O_CERDOS
    n2o_aves = aves * EF_ESTIERCOL_N2O_AVES
    total_n2o_kg_anio = n2o_leche + n2o_carne + n2o_cerdos + n2o_aves
    
    # 3. Conversión a tCO2e
    emision_ch4_tco2e = (total_ch4_kg_anio / 1000) * GWP_CH4
    emision_n2o_tco2e = (total_n2o_kg_anio / 1000) * GWP_N2O
    emision_total_tco2e_anio = emision_ch4_tco2e + emision_n2o_tco2e
    
    # 4. Proyección
    years = np.arange(0, anios + 1)
    emisiones_anuales = np.full_like(years, emision_total_tco2e_anio, dtype=float)
    emisiones_anuales[0] = 0 
    
    df_emisiones = pd.DataFrame({
        'Año': years,
        'Emisiones_CH4_tCO2e': (ch4_leche + ch4_carne + ch4_cerdos + ch4_aves) / 1000 * GWP_CH4,
        'Emisiones_N2O_tCO2e': (n2o_leche + n2o_carne + n2o_cerdos + n2o_aves) / 1000 * GWP_N2O,
        'Emision_Anual_tCO2e': emisiones_anuales,
        'Emision_Acumulada_tCO2e': np.cumsum(emisiones_anuales)
    })
    return df_emisiones

def calcular_captura_pasturas(hectareas, anios=20, escenario_key="PASTO_MANEJADO"):
    """
    Calcula la captura (o pérdida) de carbono en el suelo y biomasa menor
    según el tipo de manejo de la pastura.
    """
    params = ESCENARIOS_PASTURAS.get(escenario_key, ESCENARIOS_PASTURAS["PASTO_MANEJADO"])
    tasa_c = params['tasa_c_ha_anio']
    
    years = np.arange(0, anios + 1)
    
    # El IPCC asume que los cambios en el carbono del suelo (SOC) ocurren
    # linealmente durante un periodo de transición de 20 años.
    # Después del año 20, el suelo alcanza un nuevo equilibrio y la tasa es 0.
    tasa_activa = np.where(years <= 20, tasa_c, 0)
    tasa_activa[0] = 0 # Año 0 no acumula
    
    # tC a tCO2e
    tasa_co2e_ha_anio = tasa_activa * FACTOR_C_CO2
    captura_anual_proyecto = tasa_co2e_ha_anio * hectareas
    
    df_pastos = pd.DataFrame({
        'Año': years,
        'Pastura_tCO2e_Anual': captura_anual_proyecto,
        'Pastura_tCO2e_Acumulado': np.cumsum(captura_anual_proyecto)
    })
    
    return df_pastos

def calcular_balance_afolu(df_forestal, df_pastos, df_emisiones):
    """
    Cruza los tres mundos (Bosque, Pastos y Vacas) para obtener el Balance Neto.
    Si el balance es positivo, el predio es Sumidero (captura más de lo que emite).
    Si es negativo, el predio es Emisor Neto.
    """
    df_balance = pd.DataFrame({'Año': df_forestal['Año']})
    
    # Sumideros (Positivos)
    df_balance['Captura_Bosque'] = df_forestal['Proyecto_tCO2e_Acumulado']
    df_balance['Captura_Pastos'] = df_pastos['Pastura_tCO2e_Acumulado']
    
    # Fuentes (Negativas) - Restamos las emisiones de la ganadería
    df_balance['Emisiones_Ganado'] = -df_emisiones['Emision_Acumulada_tCO2e']
    
    # Balance Neto
    df_balance['Balance_Neto_tCO2e'] = (
        df_balance['Captura_Bosque'] + 
        df_balance['Captura_Pastos'] + 
        df_balance['Emisiones_Ganado']
    )
    
    return df_balance
