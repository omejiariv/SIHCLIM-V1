import streamlit as st
import pandas as pd
import numpy as np
from modules import db_manager, analysis

st.title("🧪 Prueba de Laboratorio: Estadísticas")

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "Agua2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "Agua2026"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo
muro_de_acceso_beta()
# ==============================================================================

engine = db_manager.get_engine()

if st.button("🔬 Probar Cálculo con 1 Estación"):
    try:
        # 1. Cargar datos de lluvia de una estación cualquiera
        st.info("Cargando datos de prueba...")
        df = pd.read_sql("SELECT fecha_mes_año, precipitation FROM precipitacion_mensual LIMIT 200", engine)
        
        if df.empty:
            st.error("❌ La tabla 'precipitacion_mensual' está vacía.")
            st.stop()
            
        # 2. Preparar Serie
        df['fecha'] = pd.to_datetime(df['fecha_mes_año'])
        serie = df.set_index('fecha')['precipitation']
        
        st.write(f"✅ Datos cargados: {len(serie)} meses.")
        st.line_chart(serie)
        
        # 3. Probar Función
        st.info("Ejecutando `calculate_hydrological_statistics`...")
        
        # Parámetros simulados (Area 100km2, Coeff 0.5)
        resultados = analysis.calculate_hydrological_statistics(serie, 0.5, 100)
        
        st.write("### 📊 Resultados:")
        st.json(resultados)
        
        # 4. Diagnóstico de Error
        if "Error" in resultados:
            st.error(f"⚠️ La función retornó error: {resultados['Error']}")
        elif resultados.get("Q_Max_100a") == -1:
            st.error("❌ FALTA LIBRERÍA SCIPY. No se pueden calcular retornos.")
            st.warning("Solución: Agrega `scipy` a tu archivo requirements.txt")
        elif resultados.get("Q_Max_100a") == 0:
            st.warning("⚠️ El cálculo dio 0 (Posiblemente pocos datos anuales).")
        else:
            st.success("🎉 ¡Cálculo Exitoso! El módulo funciona bien.")
            
    except Exception as e:
        st.error(f"🔥 Error Crítico en la prueba: {e}")
