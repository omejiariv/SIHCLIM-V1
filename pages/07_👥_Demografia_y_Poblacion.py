import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="Demografía y Población", page_icon="👥", layout="wide")

st.title("👥 Dinámica Demográfica y Modelación Poblacional")
st.markdown("""
Plataforma de modelación avanzada para el análisis de crecimiento poblacional, estructura por edades, 
parámetros vitales (natalidad, mortalidad, migración) y proyecciones anidadas (Global a Local).
""")
st.divider()

# ESTRUCTURA MAESTRA DE PESTAÑAS (Respondiendo a los 6 puntos requeridos)
tab_datos, tab_modelos, tab_piramides, tab_anidados = st.tabs([
    "📊 1. Censos y Datos Base", 
    "📈 2. Modelos Evolutivos", 
    "🏗️ 3. Estructura y Parámetros Vitales",
    "🌍 4. Modelos Anidados (Jerárquicos)"
])

# ------------------------------------------------------------------------------
# TAB 1: CENSOS (Puntos 1)
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("Gestión de Información Censal")
    st.info("Módulo para importar, cruzar y limpiar datos de censos reales (DANE, Banco Mundial) o crear poblaciones hipotéticas.")
    st.selectbox("Nivel Espacial de Análisis:", ["Mundial", "Continental", "Nacional (Colombia)", "Departamental (Antioquia)", "Local/Municipal", "Hipotético"])
    # Aquí irá a futuro la carga de archivos CSV de censos.

# ------------------------------------------------------------------------------
# TAB 2: MODELOS DE CRECIMIENTO (Puntos 2 y 3) - ¡FUNCIONAL!
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("Análisis de Modelos Evolutivos de Población")
    st.markdown("Compara cómo diferentes ecuaciones matemáticas proyectan el futuro de una población.")
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.subheader("Datos de Calibración")
        p0 = st.number_input("Población Inicial (P0):", value=10000, step=1000)
        tasa_r = st.number_input("Tasa intrínseca de crecimiento (r):", value=0.025, step=0.005, format="%.3f")
        k_cap = st.number_input("Capacidad de Carga del Territorio (K):", value=50000, step=5000)
        t_max = st.slider("Años a proyectar (t):", 10, 200, 100)
        
        modelos_sel = st.multiselect(
            "Modelos a Comparar:", 
            ["Exponencial (Malthus)", "Logístico (Verhulst)", "Gompertz"],
            default=["Exponencial (Malthus)", "Logístico (Verhulst)"]
        )

    with col_m2:
        # Generar vector de tiempo
        t = np.arange(0, t_max + 1)
        df_modelos = pd.DataFrame({"Año": t})
        
        # 1. Modelo Exponencial: P(t) = P0 * e^(rt)
        if "Exponencial (Malthus)" in modelos_sel:
            df_modelos["Exponencial"] = p0 * np.exp(tasa_r * t)
            
        # 2. Modelo Logístico: P(t) = K / (1 + ((K-P0)/P0) * e^(-rt))
        if "Logístico (Verhulst)" in modelos_sel:
            c = (k_cap - p0) / p0
            df_modelos["Logístico"] = k_cap / (1 + c * np.exp(-tasa_r * t))
            
        # 3. Modelo Gompertz: P(t) = K * e^(ln(P0/K) * e^(-rt))
        if "Gompertz" in modelos_sel:
            ln_p0_k = np.log(p0 / k_cap)
            df_modelos["Gompertz"] = k_cap * np.exp(ln_p0_k * np.exp(-tasa_r * t))

        # Graficar
        fig_mod = go.Figure()
        colores = {"Exponencial": "#e74c3c", "Logístico": "#2ecc71", "Gompertz": "#f39c12"}
        
        for mod in df_modelos.columns[1:]:
            fig_mod.add_trace(go.Scatter(x=df_modelos["Año"], y=df_modelos[mod], mode='lines', name=mod, line=dict(width=3, color=colores.get(mod, 'blue'))))
            
        # Línea de capacidad de carga
        fig_mod.add_trace(go.Scatter(x=[0, t_max], y=[k_cap, k_cap], mode='lines', name='Capacidad de Carga (K)', line=dict(color='black', dash='dash')))
        
        fig_mod.update_layout(title="Comparativa de Modelos de Crecimiento Poblacional", xaxis_title="Años Proyectados (t)", yaxis_title="Habitantes", hovermode="x unified", height=500)
        st.plotly_chart(fig_mod, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ESTRUCTURAS Y PARÁMETROS VITALES (Puntos 4 y 5)
# ------------------------------------------------------------------------------
with tab_piramides:
    st.header("Dinámica de Cohortes (Método de los Componentes)")
    st.info("Aquí modelaremos la evolución de la población por grupos de edad (Pirámides) aplicando tasas de natalidad, mortalidad y vectores de migración.")
    
    st.markdown("### Parámetros Vitales (Simulación)")
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Tasa Bruta Natalidad (TBN)", "14.5 x 1000 hab")
    cp2.metric("Tasa Bruta Mortalidad (TBM)", "6.2 x 1000 hab")
    cp3.metric("Saldo Migratorio", "-1.2 x 1000 hab", delta="Emigración neta", delta_color="inverse")
    
    st.image("https://images.unsplash.com/photo-1518063319518-4c622ccf0a56?auto=format&fit=crop&w=1200&q=80", caption="Próximamente: Gráficos interactivos de Pirámides Poblacionales dinámicas.", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MODELOS ANIDADOS (Punto 6)
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("Downscaling Demográfico (Modelos Jerárquicos)")
    st.info("Conociendo el modelo de crecimiento macro (Ej. Nacional), forzaremos las proyecciones locales (Municipal) para que la suma de las partes coincida con el límite superior.")
