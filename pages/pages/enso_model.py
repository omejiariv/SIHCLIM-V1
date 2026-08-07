import streamlit as st
import pandas as pd
import numpy as np

# Configuración inicial de la página
st.set_page_config(page_title="Modelo ENSO - Dinámica de Sistemas", layout="wide")
st.title("🌊 Simulación de Seguridad Hídrica: Efecto ENSO")
st.markdown("Modelo de dinámica de sistemas para evaluar el impacto del ONI en el almacenamiento de embalses y caudales.")

# 1. Panel de Control (Sidebar)
st.sidebar.header("Parámetros del Escenario")
# El ONI va de -2.0 (La Niña extrema) a +2.0 (El Niño extremo)
oni_index = st.sidebar.slider("Índice Oceánico Niño (ONI)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# Parámetros iniciales del sistema (pueden ser luego consultados desde Supabase/PostGIS)
volumen_inicial = st.sidebar.number_input("Volumen Inicial del Embalse (Hm3)", value=100.0)
demanda_urbana = st.sidebar.number_input("Demanda Humana/Ecológica (Hm3/mes)", value=10.0)
precip_base = st.sidebar.number_input("Precipitación Base Histórica (mm/mes)", value=150.0)

# 2. Funciones de Flujo (Impacto del ONI)
def calcular_anomalia_climatica(oni):
    """
    Traduce el índice ONI en anomalías de precipitación y temperatura.
    (Estos coeficientes se pueden calibrar luego con los registros históricos del IDEAM).
    """
    # Si ONI > 0 (El Niño), llueve menos y hace más calor.
    # Si ONI < 0 (La Niña), llueve más y hace menos calor.
    factor_precip = 1.0 - (oni * 0.25) # Un ONI de +2.0 reduce la lluvia a la mitad
    factor_temp_evap = 1.0 + (oni * 0.15) # Un ONI de +2.0 aumenta la evaporación en un 30%
    
    # Evitar lluvias negativas
    factor_precip = max(0.0, factor_precip)
    
    return factor_precip, factor_temp_evap

# 3. Motor de Simulación (Integración tipo Euler)
def simular_embalse(meses, vol_ini, demanda, precip_base, oni):
    factor_precip, factor_evap = calcular_anomalia_climatica(oni)
    
    # Estructura de datos para almacenar la simulación
    resultados = []
    vol_actual = vol_ini
    
    for mes in range(1, meses + 1):
        # FLUJOS DE ENTRADA
        # Simplificación: El aporte al embalse depende de la lluvia modificada por el ENSO
        entrada_escorrentia = (precip_base * factor_precip) * 0.1 # 0.1 es un coeficiente de escorrentía hipotético
        
        # FLUJOS DE SALIDA
        # La evaporación base (ej. 2 Hm3) multiplicada por el factor ENSO
        salida_evaporacion = 2.0 * factor_evap 
        salida_total = demanda + salida_evaporacion
        
        # ECUACIÓN DE NIVEL (STOCK)
        delta_volumen = entrada_escorrentia - salida_total
        vol_actual = vol_actual + delta_volumen
        
        # Evitar volúmenes negativos
        vol_actual = max(0.0, vol_actual)
        
        resultados.append({
            "Mes": mes,
            "Precipitación Modificada": precip_base * factor_precip,
            "Entrada (Hm3)": entrada_escorrentia,
            "Salida (Hm3)": salida_total,
            "Volumen Embalse (Hm3)": vol_actual
        })
        
    return pd.DataFrame(resultados)

# 4. Ejecución y Visualización
meses_simulacion = 12
df_simulacion = simular_embalse(meses_simulacion, volumen_inicial, demanda_urbana, precip_base, oni_index)

# KPIs principales
col1, col2, col3 = st.columns(3)
col1.metric("Volumen Final (Mes 12)", f"{df_simulacion['Volumen Embalse (Hm3)'].iloc[-1]:.2f} Hm3")
col2.metric("Precipitación Promedio", f"{df_simulacion['Precipitación Modificada'].mean():.2f} mm")
col3.metric("Evaporación + Demanda", f"{df_simulacion['Salida (Hm3)'].mean():.2f} Hm3/mes")

# Gráficas
st.subheader("Dinámica del Almacenamiento")
st.line_chart(df_simulacion.set_index("Mes")["Volumen Embalse (Hm3)"])

st.subheader("Balance de Flujos (Entradas vs Salidas)")
st.line_chart(df_simulacion.set_index("Mes")[["Entrada (Hm3)", "Salida (Hm3)"]])

# Vista de datos tabulares (ideal para auditar el modelo PyArrow/Pandas)
with st.expander("Ver tabla de datos de la simulación"):
    st.dataframe(df_simulacion)
