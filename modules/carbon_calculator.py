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
# MÓDULO AFOLU: GANADERÍA, PASTURAS, HUMANOS Y DEFORESTACIÓN
# ==============================================================================

# --- 1. PARÁMETROS GLOBALES IPCC ---
GWP_CH4 = 28   
GWP_N2O = 265  

# Factores de Emisión CH4 (kg/cabeza/año)
EF_ENTERIC_LECHE, EF_ENTERIC_CARNE = 72.0, 56.0  
EF_ENTERIC_CERDOS, EF_ENTERIC_AVES = 1.5, 0.0    
EF_ENTERIC_HUMANOS = 0.0

# Gestión de Estiércol / Aguas Residuales (kg CH4/cabeza/año)
EF_ESTIERCOL_CH4_LECHE, EF_ESTIERCOL_CH4_CARNE = 2.0, 1.0
EF_ESTIERCOL_CH4_CERDOS, EF_ESTIERCOL_CH4_AVES = 4.0, 0.02
EF_ESTIERCOL_CH4_HUMANOS = 1.5 # Fosas sépticas / aguas residuales rurales

# Óxido Nitroso (kg N2O/cabeza/año)
EF_ESTIERCOL_N2O_LECHE, EF_ESTIERCOL_N2O_CARNE = 1.5, 1.2
EF_ESTIERCOL_N2O_CERDOS, EF_ESTIERCOL_N2O_AVES = 0.2, 0.001
EF_ESTIERCOL_N2O_HUMANOS = 0.1

ESCENARIOS_PASTURAS = {
    "PASTO_DEGRADADO": {"nombre": "Pasto Degradado (Línea Base)", "tasa_c_ha_anio": -0.5},
    "PASTO_MANEJADO": {"nombre": "Pasto Mejorado (Manejo Rotacional)", "tasa_c_ha_anio": 0.8},
    "SSP_BAJO": {"nombre": "Silvopastoril (Baja Densidad)", "tasa_c_ha_anio": 1.2},
    "SSP_INTENSIVO": {"nombre": "Silvopastoril Intensivo (SSPi)", "tasa_c_ha_anio": 2.5}
}

# Parámetros de Pérdida/Deforestación (tC/ha almacenado)
STOCKS_SUCESION = {
    "PASTO": 15.0, "RASTROJO_BAJO": 30.0, "RASTROJO_MEDIO": 60.0,
    "RASTROJO_ALTO": 90.0, "BOSQUE_SECUNDARIO": 120.0, "BOSQUE_MADURO": 150.0
}
CAUSAS_PERDIDA = {
    "INCENDIO": {"frac": 0.9, "gei": "Altas emisiones de CH4 y N2O por combustión."},
    "AGRICOLA": {"frac": 0.8, "gei": "Oxidación rápida por remoción de biomasa."},
    "PLAGAS": {"frac": 0.5, "gei": "Descomposición lenta (CO2 primario)."}
}

# --- 3. MOTORES DE CÁLCULO ---

def calcular_emisiones_fuentes_detallado(vacas_l=0, vacas_c=0, cerdos=0, aves=0, humanos=0, anios=20):
    """Calcula emisiones separadas por cada fuente para graficar curvas independientes."""
    years = np.arange(0, anios + 1)
    
    # Cálculos anuales en tCO2e para cada fuente
    co2e_v = (((vacas_l * (EF_ENTERIC_LECHE + EF_ESTIERCOL_CH4_LECHE) + vacas_c * (EF_ENTERIC_CARNE + EF_ESTIERCOL_CH4_CARNE)) / 1000) * GWP_CH4) + \
             (((vacas_l * EF_ESTIERCOL_N2O_LECHE + vacas_c * EF_ESTIERCOL_N2O_CARNE) / 1000) * GWP_N2O)
             
    co2e_p = ((cerdos * (EF_ENTERIC_CERDOS + EF_ESTIERCOL_CH4_CERDOS) / 1000) * GWP_CH4) + \
             ((cerdos * EF_ESTIERCOL_N2O_CERDOS / 1000) * GWP_N2O)
             
    co2e_a = ((aves * (EF_ENTERIC_AVES + EF_ESTIERCOL_CH4_AVES) / 1000) * GWP_CH4) + \
             ((aves * EF_ESTIERCOL_N2O_AVES / 1000) * GWP_N2O)
             
    co2e_h = ((humanos * (EF_ENTERIC_HUMANOS + EF_ESTIERCOL_CH4_HUMANOS) / 1000) * GWP_CH4) + \
             ((humanos * EF_ESTIERCOL_N2O_HUMANOS / 1000) * GWP_N2O)
             
    # Arrays de acumulación (el año 0 es 0)
    df = pd.DataFrame({'Año': years})
    df['Bovinos_tCO2e'] = np.cumsum(np.where(years > 0, co2e_v, 0))
    df['Porcinos_tCO2e'] = np.cumsum(np.where(years > 0, co2e_p, 0))
    df['Aves_tCO2e'] = np.cumsum(np.where(years > 0, co2e_a, 0))
    df['Humanos_tCO2e'] = np.cumsum(np.where(years > 0, co2e_h, 0))
    
    # Para el gráfico de pastel
    df['CH4_Total_Anual'] = co2e_v + co2e_p + co2e_a + co2e_h # Simplificado para validación
    
    return df

def calcular_evento_cambio(hectareas, tipo="PERDIDA", estado="BOSQUE_SECUNDARIO", causa="AGRICOLA", anio_evento=1, anios=20):
    """Permite simular tanto pérdida (pulso de emisión) como ganancia (crecimiento) en un año específico."""
    years = np.arange(0, anios + 1)
    em_acumulada = np.zeros_like(years, dtype=float)
    stock_c = STOCKS_SUCESION.get(estado, 100.0)
    
    if hectareas > 0 and anio_evento <= anios:
        if tipo == "PERDIDA":
            fraccion = CAUSAS_PERDIDA.get(causa, {}).get("frac", 0.8)
            pulso = (stock_c * fraccion * 3.666667) * hectareas
            em_acumulada = np.where(years >= anio_evento, pulso, 0)
        else: # GANANCIA (Restauración de otra zona)
            # Asume que llega al stock del estado seleccionado en 10 años
            tasa_anual = (stock_c * 3.666667 * hectareas) / 10.0 
            for i in range(len(years)):
                if years[i] >= anio_evento:
                    anios_crecimiento = min(years[i] - anio_evento + 1, 10)
                    em_acumulada[i] = anios_crecimiento * tasa_anual

    col_name = 'Perdida_tCO2e' if tipo == "PERDIDA" else 'Ganancia_tCO2e'
    return pd.DataFrame({'Año': years, col_name: em_acumulada})

def calcular_balance_territorial(df_bosque, df_pastos, df_fuentes, df_evento):
    """Une todas las capas separadas para el balance total."""
    df_bal = pd.DataFrame({'Año': df_bosque['Año']})
    
    # Sumideros (+)
    df_bal['Captura_Bosque'] = df_bosque['Proyecto_tCO2e_Acumulado']
    df_bal['Captura_Pastos'] = df_pastos['Pastura_tCO2e_Acumulado']
    df_bal['Evento_Ganancia'] = df_evento.get('Ganancia_tCO2e', 0)
    
    # Fuentes (-)
    df_bal['Emision_Bovinos'] = -df_fuentes['Bovinos_tCO2e']
    df_bal['Emision_Porcinos'] = -df_fuentes['Porcinos_tCO2e']
    df_bal['Emision_Aves'] = -df_fuentes['Aves_tCO2e']
    df_bal['Emision_Humanos'] = -df_fuentes['Humanos_tCO2e']
    df_bal['Evento_Perdida'] = -df_evento.get('Perdida_tCO2e', 0)
    
    df_bal['Balance_Neto_tCO2e'] = (
        df_bal['Captura_Bosque'] + df_bal['Captura_Pastos'] + df_bal['Evento_Ganancia'] +
        df_bal['Emision_Bovinos'] + df_bal['Emision_Porcinos'] + 
        df_bal['Emision_Aves'] + df_bal['Emision_Humanos'] + df_bal['Evento_Perdida']
    )
    return df_bal

def calcular_captura_pasturas(hectareas, anios=20, escenario_key="PASTO_MANEJADO"):
    tasa_c = ESCENARIOS_PASTURAS.get(escenario_key, ESCENARIOS_PASTURAS["PASTO_MANEJADO"])['tasa_c_ha_anio']
    years = np.arange(0, anios + 1)
    tasa_activa = np.where(years <= 20, tasa_c, 0)
    tasa_activa[0] = 0 
    
    captura_anual = tasa_activa * 3.666667 * hectareas
    return pd.DataFrame({'Año': years, 'Pastura_tCO2e_Anual': captura_anual, 'Pastura_tCO2e_Acumulado': np.cumsum(captura_anual)})

