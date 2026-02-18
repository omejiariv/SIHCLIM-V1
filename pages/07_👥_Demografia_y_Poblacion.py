import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import warnings
import os
warnings.filterwarnings("ignore") # Para evitar avisos de overflow en Scipy

st.set_page_config(page_title="Demografía y Población", page_icon="👥", layout="wide")

st.title("👥 Dinámica Demográfica y Modelación Poblacional")
st.markdown("""
Motor de Inferencia Demográfica: Análisis histórico multiescalar, ajuste paramétrico automático (Curve Fitting), 
proyecciones polinómicas/exponenciales y modelos anidados de Downscaling.
""")
st.divider()

# --- 1. SIMULACIÓN DE DATOS (Para la UI) ---
# En un entorno real, esto se lee de tus Parquets unificados de Informacion Demografica SP_2026.xlsx
# Simulamos una base unificada multiescalar para que puedas probar la funcionalidad hoy
@st.cache_data
def generar_datos_prueba():
    anios = np.arange(1980, 2025)
    df = pd.DataFrame({"Año": anios})
    df["Global"] = 4.4e9 * np.exp(0.012 * (anios - 1980))
    df["Suramerica"] = 2.4e8 * np.exp(0.015 * (anios - 1980))
    df["Colombia"] = 2.7e7 * np.exp(0.014 * (anios - 1980))
    df["Antioquia"] = 3.5e6 * np.exp(0.016 * (anios - 1980))
    df["Municipios Antioquia (Medellín)"] = 1.4e6 * np.exp(0.018 * (anios - 1980))
    df["Veredas Antioquia (Promedio)"] = 5000 * np.exp(0.005 * (anios - 1980))
    return df

df_multiescalar = generar_datos_prueba()
escala_opciones = ["Global", "Suramerica", "Colombia", "Antioquia", "Municipios Antioquia (Medellín)", "Veredas Antioquia (Promedio)"]

# ==============================================================================
# ESTRUCTURA MAESTRA DE PESTAÑAS
# ==============================================================================
tab_datos, tab_modelos, tab_piramides, tab_anidados = st.tabs([
    "📊 1. Censos Multiescalar", 
    "📈 2. Modelos Evolutivos & Optimización", 
    "🏗️ 3. Estructura y Cohortes",
    "🌍 4. Modelos Anidados (Downscaling)"
])

# ------------------------------------------------------------------------------
# TAB 1: CENSOS (Puntos 1 y 2)
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial Multiescalar")
    sel_escala = st.multiselect("Selecciona la(s) Escala(s) a visualizar:", escala_opciones, default=["Colombia", "Antioquia"])
    
    if sel_escala:
        df_plot = df_multiescalar[["Año"] + sel_escala].melt(id_vars="Año", var_name="Zona", value_name="Población")
        fig1 = px.line(df_plot, x="Año", y="Población", color="Zona", title="Crecimiento Histórico", markers=True)
        if st.checkbox("Usar Escala Logarítmica (Recomendado para comparar Global vs Local)"):
            fig1.update_layout(yaxis_type="log")
        st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MODELOS Y OPTIMIZACIÓN (Puntos 2 y 3) - ¡EL MOTOR MATEMÁTICO!
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("📈 Ajuste de Modelos Evolutivos y Optimización Paramétrica")
    
    col_opt1, col_opt2 = st.columns([1, 2.5])
    
    with col_opt1:
        st.subheader("Configuración")
        fuente_datos = st.radio("Fuente de Análisis:", ["Región Hipotética (Manual)", "Datos Reales (Multiescalar)"])
        
        if fuente_datos == "Datos Reales (Multiescalar)":
            zona_sel = st.selectbox("Selecciona la Zona Real:", escala_opciones)
            # Extraer datos reales (Normalizamos t para evitar overflow en Scipy)
            t_data = df_multiescalar["Año"].values - df_multiescalar["Año"].min()
            p_data = df_multiescalar[zona_sel].values
            p0_val = p_data[0]
            st.success(f"Datos cargados: {zona_sel} ({len(t_data)} registros)")
        else:
            p0_val = st.number_input("P0 (Pob. Inicial):", value=10000)
            
        t_max = st.slider("Años a proyectar (Horizonte):", 10, 150, 80)
        
        st.markdown("---")
        st.subheader("Selección de Modelos")
        modelos_sel = st.multiselect(
            "Curvas a evaluar:", 
            ["Exponencial", "Logístico", "Gompertz", "Geométrico", "Polinómico (Grado 2)", "Polinómico (Grado 3)", "Polinómico (Grado 4)"],
            default=["Exponencial", "Logístico"]
        )
        
        opt_auto = False
        if fuente_datos == "Datos Reales (Multiescalar)":
            opt_auto = st.button("✨ Optimizar Parámetros Automáticamente", type="primary", use_container_width=True)
            if opt_auto: st.toast("Calculando mejores ajustes mediante Mínimos Cuadrados No Lineales...")

        # Parámetros manuales (se sobrescriben si opt_auto es True)
        st.caption("Parámetros Manuales (Si no se optimiza):")
        r_man = st.number_input("Tasa (r):", value=0.02, format="%.4f")
        k_man = st.number_input("Capacidad (K):", value=p0_val*5, step=1000.0)

    with col_opt2:
        # FUNCIONES MATEMÁTICAS
        def f_exp(t, p0, r): return p0 * np.exp(r * t)
        def f_log(t, k, p0, r): return k / (1 + ((k-p0)/p0) * np.exp(-r * t))
        def f_gomp(t, k, p0, r): return k * np.exp(np.log(p0/k) * np.exp(-r * t))
        def f_geom(t, p0, r): return p0 * (1 + r)**t
        def f_poly2(t, a, b, c): return a*t**2 + b*t + c
        def f_poly3(t, a, b, c, d): return a*t**3 + b*t**2 + c*t + d
        def f_poly4(t, a, b, c, d, e): return a*t**4 + b*t**3 + c*t**2 + d*t + e

        # Generar vector de tiempo futuro
        t_futuro = np.arange(0, t_max + 1)
        anios_futuro = t_futuro + (df_multiescalar["Año"].min() if fuente_datos == "Datos Reales (Multiescalar)" else 2024)
        
        fig2 = go.Figure()
        
        # Si hay datos reales, graficarlos como puntos
        if fuente_datos == "Datos Reales (Multiescalar)":
            fig2.add_trace(go.Scatter(x=df_multiescalar["Año"], y=p_data, mode='markers', name='Datos Históricos Reales', marker=dict(color='black', size=8)))

        # LÓGICA DE OPTIMIZACIÓN Y PROYECCIÓN
        res_text = []
        for mod in modelos_sel:
            y_pred = np.zeros_like(t_futuro, dtype=float)
            try:
                if mod == "Exponencial":
                    if opt_auto:
                        popt, _ = curve_fit(f_exp, t_data, p_data, p0=[p0_val, 0.01])
                        y_pred = f_exp(t_futuro, *popt)
                        res_text.append(f"**Exponencial**: r={popt[1]:.4f}")
                    else: y_pred = f_exp(t_futuro, p0_val, r_man)

                elif mod == "Logístico":
                    if opt_auto:
                        popt, _ = curve_fit(f_log, t_data, p_data, p0=[max(p_data)*2, p0_val, 0.01], maxfev=10000)
                        y_pred = f_log(t_futuro, *popt)
                        res_text.append(f"**Logístico**: K={popt[0]:,.0f}, r={popt[2]:.4f}")
                    else: y_pred = f_log(t_futuro, k_man, p0_val, r_man)

                elif mod == "Gompertz":
                    if opt_auto:
                        popt, _ = curve_fit(f_gomp, t_data, p_data, p0=[max(p_data)*2, p0_val, 0.01], maxfev=10000)
                        y_pred = f_gomp(t_futuro, *popt)
                    else: y_pred = f_gomp(t_futuro, k_man, p0_val, r_man)

                elif mod == "Geométrico":
                    if opt_auto:
                        popt, _ = curve_fit(f_geom, t_data, p_data, p0=[p0_val, 0.01])
                        y_pred = f_geom(t_futuro, *popt)
                    else: y_pred = f_geom(t_futuro, p0_val, r_man)

                elif mod == "Polinómico (Grado 2)":
                    if opt_auto:
                        popt, _ = curve_fit(f_poly2, t_data, p_data)
                        y_pred = f_poly2(t_futuro, *popt)
                    else: y_pred = f_poly2(t_futuro, 10, 50, p0_val) # Manual dummy

                elif mod == "Polinómico (Grado 3)":
                    if opt_auto:
                        popt, _ = curve_fit(f_poly3, t_data, p_data)
                        y_pred = f_poly3(t_futuro, *popt)
                    else: y_pred = f_poly3(t_futuro, 1, 10, 50, p0_val)

                elif mod == "Polinómico (Grado 4)":
                    if opt_auto:
                        popt, _ = curve_fit(f_poly4, t_data, p_data)
                        y_pred = f_poly4(t_futuro, *popt)
                    else: y_pred = f_poly4(t_futuro, 0.1, 1, 10, 50, p0_val)

                fig2.add_trace(go.Scatter(x=anios_futuro, y=y_pred, mode='lines', name=mod, line=dict(width=3, dash='dot' if opt_auto else 'solid')))
            except Exception as e:
                st.warning(f"No se pudo optimizar {mod}. Intenta con parámetros manuales más cercanos.")

        fig2.update_layout(title="Proyección y Ajuste de Modelos", xaxis_title="Año", yaxis_title="Población", hovermode="x unified", height=550)
        st.plotly_chart(fig2, use_container_width=True)
        
        if opt_auto and res_text:
            st.success("✅ **Parámetros Óptimos Encontrados:** " + " | ".join(res_text))

# ------------------------------------------------------------------------------
# TAB 3: ESTRUCTURAS Y PIRÁMIDES (Punto 4)
# ------------------------------------------------------------------------------
with tab_piramides:
    st.header("🏗️ Pirámides Poblacionales por Zona")
    st.info("Visualiza la estructura de cohortes. (Requiere subir el archivo de Edades con la columna 'Zona')")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        # Selector de Zona agregado
        zona_piramide = st.selectbox("Selecciona la Zona de Análisis:", ["Global", "Suramérica", "Colombia", "Antioquia", "Municipios Antioquia", "Veredas Antioquia"])
        anio_sel = st.slider("Selecciona el Año de la Pirámide:", 1985, 2050, 2024)
        st.markdown(f"**Zona actual:** {zona_piramide}")
        st.caption("Aquí se conectará tu archivo `Evolucion edades.csv` filtrado por la zona seleccionada.")
        
    with col_p2:
        # Generación de pirámide simulada reactiva para la UI
        edades = np.arange(0, 100, 5)
        hombres = np.random.normal(5000 - (edades*30), 500).astype(int)
        mujeres = np.random.normal(5200 - (edades*28), 500).astype(int)
        
        fig_pir = go.Figure()
        fig_pir.add_trace(go.Bar(y=edades, x=hombres * -1, name='Hombres', orientation='h', marker=dict(color='#3498db')))
        fig_pir.add_trace(go.Bar(y=edades, x=mujeres, name='Mujeres', orientation='h', marker=dict(color='#e74c3c')))
        fig_pir.update_layout(title=f"Pirámide Simulada - {zona_piramide} ({anio_sel})", barmode='relative', yaxis_title='Edad', xaxis_title='Población', hovermode="y unified", height=500)
        st.plotly_chart(fig_pir, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MODELOS ANIDADOS (Punto 5) - ¡DOWNSCALING FUNCIONAL!
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("🌍 Modelos Jerárquicos Anidados (Downscaling Demográfico)")
    st.markdown("""
    Este módulo toma una proyección "Macro" (Ej. Colombia) y calcula la proyección "Micro" (Ej. Antioquia) 
    manteniendo la coherencia sistémica mediante la cuota de participación histórica o ajustada.
    """)
    
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("Configuración del Downscaling")
        nivel_macro = st.selectbox("Nivel Macro (Contenedor):", ["Colombia", "Suramerica", "Global"])
        nivel_micro = st.selectbox("Nivel Micro (Anidado):", ["Antioquia", "Municipios Antioquia", "Veredas Antioquia"])
        
        st.caption("Método de Participación:")
        metodo_part = st.radio("Cálculo de Cuota (Share):", ["Constante (Último Censo)", "Tendencial (Cambio proyectado)"])
        
        # Simulamos que Antioquia es el 14% de Colombia
        cuota_base = st.slider(f"% de participación de {nivel_micro} en {nivel_macro}:", 0.1, 50.0, 14.0, step=0.1) / 100.0
        
    with col_a2:
        # Generamos una proyección Macro (Logística dummy)
        t_ani = np.arange(2024, 2060)
        pob_macro = 50e6 / (1 + 0.1 * np.exp(-0.02 * (t_ani - 2024))) # Colombia dummy
        
        if metodo_part == "Constante (Último Censo)":
            pob_micro = pob_macro * cuota_base
        else: # Tendencial (Gana o pierde participación con el tiempo)
            tendencia = np.linspace(cuota_base, cuota_base * 1.15, len(t_ani)) # Gana 15% de participación al 2060
            pob_micro = pob_macro * tendencia
            
        fig_ani = go.Figure()
        fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_macro, mode='lines', fill='tozeroy', name=f'Macro: {nivel_macro}', line=dict(color='#95a5a6', width=1)))
        fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_micro, mode='lines', fill='tozeroy', name=f'Micro Anidado: {nivel_micro}', line=dict(color='#e67e22', width=3)))
        
        fig_ani.update_layout(title="Downscaling: Proyección Anidada Consistente", xaxis_title="Año", yaxis_title="Población", hovermode="x unified")
        st.plotly_chart(fig_ani, use_container_width=True)
        st.info(f"💡 Interpretación: La proyección de **{nivel_micro}** está matemáticamente obligada a no superar la envolvente de **{nivel_macro}**, resolviendo el problema de las proyecciones exponenciales locales infinitas.")
