import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings("ignore") 

st.set_page_config(page_title="Demografía y Población", page_icon="👥", layout="wide")

st.title("👥 Dinámica Demográfica y Modelación Poblacional")
st.markdown("""
Motor de Inferencia Demográfica: Análisis histórico multiescalar (desde 1912), ajuste paramétrico automático (Curve Fitting), 
proyecciones polinómicas/exponenciales y modelos jerárquicos de Downscaling territorial.
""")
st.divider()

# --- 1. LECTURA DE DATOS REALES (Desde el Panel de Administración) ---
@st.cache_data
def cargar_historico_real():
    ruta = "data/poblacion_historica_macro.parquet"
    if os.path.exists(ruta):
        df = pd.read_parquet(ruta)
        # Limpiar nombres de columnas (Pob_Colombia -> Colombia)
        df.columns = [c.replace('Pob_', '') for c in df.columns]
        return df
    return pd.DataFrame()

df_real = cargar_historico_real()

# Si no hay datos, mostramos advertencia y detenemos la ejecución limpia
if df_real.empty:
    st.warning("⚠️ No se encontraron datos históricos reales. Ve al 'Panel de Administración' -> 'Demografía' y carga tu archivo CSV con la historia desde 1912.")
    st.stop()

# Opciones dinámicas basadas en las columnas de tu Excel real
escala_opciones = [col for col in df_real.columns if col != "Año"]

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
# TAB 1: CENSOS HISTÓRICOS (AHORA SÍ DESDE 1912)
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial Multiescalar")
    sel_escala = st.multiselect("Selecciona la(s) Escala(s) a visualizar:", escala_opciones, default=escala_opciones[:2])
    
    if sel_escala:
        df_plot = df_real[["Año"] + sel_escala].melt(id_vars="Año", var_name="Zona", value_name="Población")
        fig1 = px.line(df_plot, x="Año", y="Población", color="Zona", title="Crecimiento Histórico Real (Base de Datos)", markers=True)
        if st.checkbox("Usar Escala Logarítmica (Recomendado para comparar escalas muy distintas)"):
            fig1.update_layout(yaxis_type="log")
        st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MODELOS Y OPTIMIZACIÓN (CON EL ERROR NUMÉRICO CORREGIDO)
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("📈 Ajuste de Modelos Evolutivos y Optimización Paramétrica")
    
    col_opt1, col_opt2 = st.columns([1, 2.5])
    
    with col_opt1:
        st.subheader("Configuración")
        zona_sel = st.selectbox("Selecciona la Zona Real a Modelar:", escala_opciones)
        
        # Extraer datos reales (Normalizamos el año base a t=0 para evitar overflow en Scipy)
        t_data = df_real["Año"].values - df_real["Año"].min()
        p_data = df_real[zona_sel].values
        p0_val = float(p_data[0]) # Forzamos float para evitar el StreamlitMixedNumericTypesError
        
        st.success(f"Datos cargados: {zona_sel} desde {df_real['Año'].min()} ({len(t_data)} registros)")
            
        t_max = st.slider("Años a proyectar (Horizonte desde el último dato):", 10, 150, 50)
        
        st.markdown("---")
        modelos_sel = st.multiselect(
            "Curvas a evaluar:", 
            ["Exponencial", "Logístico", "Gompertz", "Geométrico", "Polinómico (Grado 2)", "Polinómico (Grado 3)", "Polinómico (Grado 4)"],
            default=["Logístico", "Polinómico (Grado 2)"]
        )
        
        opt_auto = st.button("✨ Optimizar Parámetros Automáticamente", type="primary", use_container_width=True)

        st.caption("Parámetros Manuales (Si no se optimiza):")
        r_man = st.number_input("Tasa (r):", value=0.02, format="%.4f")
        # Aquí estaba el error. Forzamos value y step a ser floats explícitamente.
        k_man = st.number_input("Capacidad (K):", value=float(p0_val * 5.0), step=1000.0)

    with col_opt2:
        # Funciones Matemáticas
        def f_exp(t, p0, r): return p0 * np.exp(r * t)
        def f_log(t, k, p0, r): return k / (1 + ((k-p0)/p0) * np.exp(-r * t))
        def f_gomp(t, k, p0, r): return k * np.exp(np.log(p0/k) * np.exp(-r * t))
        def f_geom(t, p0, r): return p0 * (1 + r)**t
        def f_poly2(t, a, b, c): return a*t**2 + b*t + c
        def f_poly3(t, a, b, c, d): return a*t**3 + b*t**2 + c*t + d
        def f_poly4(t, a, b, c, d, e): return a*t**4 + b*t**3 + c*t**2 + d*t + e

        # Vector de tiempo incluyendo el pasado y el futuro proyectado
        t_total = np.arange(0, max(t_data) + t_max + 1)
        anios_totales = t_total + df_real["Año"].min()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_real["Año"], y=p_data, mode='markers', name='Datos Históricos Reales', marker=dict(color='black', size=8)))

        res_text = []
        for mod in modelos_sel:
            y_pred = np.zeros_like(t_total, dtype=float)
            try:
                if mod == "Exponencial":
                    if opt_auto:
                        popt, _ = curve_fit(f_exp, t_data, p_data, p0=[p0_val, 0.01])
                        y_pred = f_exp(t_total, *popt)
                        res_text.append(f"**Exp**: r={popt[1]:.4f}") # ¡Línea corregida!
                    else: 
                        y_pred = f_exp(t_total, p0_val, r_man)

                elif mod == "Logístico":
                    if opt_auto:
                        # Max de los datos * 2 como aproximación inicial de K
                        popt, _ = curve_fit(f_log, t_data, p_data, p0=[max(p_data)*2.0, p0_val, 0.01], maxfev=10000)
                        y_pred = f_log(t_total, *popt)
                        res_text.append(f"**Log**: K={popt[0]:,.0f}, r={popt[2]:.4f}") # ¡Línea corregida!
                    else: 
                        y_pred = f_log(t_total, k_man, p0_val, r_man)

                elif mod == "Gompertz":
                    if opt_auto:
                        popt, _ = curve_fit(f_gomp, t_data, p_data, p0=[max(p_data)*2.0, p0_val, 0.01], maxfev=10000)
                        y_pred = f_gomp(t_total, *popt)
                    else: 
                        y_pred = f_gomp(t_total, k_man, p0_val, r_man)

                elif mod == "Geométrico":
                    if opt_auto:
                        popt, _ = curve_fit(f_geom, t_data, p_data, p0=[p0_val, 0.01])
                        y_pred = f_geom(t_total, *popt)
                    else: 
                        y_pred = f_geom(t_total, p0_val, r_man)

                elif mod == "Polinómico (Grado 2)":
                    if opt_auto:
                        popt, _ = curve_fit(f_poly2, t_data, p_data)
                        y_pred = f_poly2(t_total, *popt)
                    else: 
                        y_pred = f_poly2(t_total, 10, 50, p0_val)

                elif mod == "Polinómico (Grado 3)":
                    if opt_auto: 
                        popt, _ = curve_fit(f_poly3, t_data, p_data)
                        y_pred = f_poly3(t_total, *popt)
                    else: 
                        y_pred = f_poly3(t_total, 1, 10, 50, p0_val)

                elif mod == "Polinómico (Grado 4)":
                    if opt_auto: 
                        popt, _ = curve_fit(f_poly4, t_data, p_data)
                        y_pred = f_poly4(t_total, *popt)
                    else: 
                        y_pred = f_poly4(t_total, 0.1, 1, 10, 50, p0_val)

                fig2.add_trace(go.Scatter(x=anios_totales, y=y_pred, mode='lines', name=mod, line=dict(width=3, dash='dot' if opt_auto else 'solid')))
            except Exception as e:
                pass # Si el modelo no logra ajustarse matemáticamente, lo ignoramos

        fig2.update_layout(title="Proyección y Ajuste de Modelos", xaxis_title="Año", yaxis_title="Población", hovermode="x unified", height=550)
        st.plotly_chart(fig2, use_container_width=True)
        
        if opt_auto and res_text:
            st.success("✅ **Parámetros Óptimos Encontrados:** " + " | ".join(res_text))

# ------------------------------------------------------------------------------
# TAB 3: ESTRUCTURAS Y PIRÁMIDES
# ------------------------------------------------------------------------------
with tab_piramides:
    st.header("🏗️ Pirámides Poblacionales por Zona")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        zona_piramide = st.selectbox("Selecciona la Zona de Análisis:", ["Colombia", "Antioquia", "Medellín"])
        anio_sel = st.slider("Selecciona el Año de la Pirámide:", 1985, 2050, 2024)
        st.warning("⚠️ **Nota UI:** La gráfica actual muestra una pirámide generada matemáticamente (simulada) para evaluar la interfaz. Reemplazaremos esto cuando se estructure el archivo Parquet de Edades con la columna 'Zona'.")
        
    with col_p2:
        # Generación de pirámide simulada reactiva para la UI (No son datos reales)
        edades = np.arange(0, 100, 5)
        hombres = np.random.normal(5000 - (edades*30), 500).astype(int)
        mujeres = np.random.normal(5200 - (edades*28), 500).astype(int)
        
        fig_pir = go.Figure()
        fig_pir.add_trace(go.Bar(y=edades, x=hombres * -1, name='Hombres', orientation='h', marker=dict(color='#3498db')))
        fig_pir.add_trace(go.Bar(y=edades, x=mujeres, name='Mujeres', orientation='h', marker=dict(color='#e74c3c')))
        fig_pir.update_layout(title=f"Pirámide (Maqueta Visual) - {zona_piramide} ({anio_sel})", barmode='relative', yaxis_title='Edad', xaxis_title='Población', hovermode="y unified", height=500)
        st.plotly_chart(fig_pir, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MODELOS ANIDADOS (JERARQUÍA DINÁMICA)
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("🌍 Modelos Jerárquicos Anidados (Downscaling Dinámico)")
    st.markdown("Al seleccionar un Nivel Macro, el sistema filtrará automáticamente los territorios correspondientes al Nivel Micro.")
    
    # DICCIONARIO DE JERARQUÍAS (El corazón dinámico del Downscaling)
    jerarquia = {
        "Global": ["Suramérica", "Norteamérica", "Europa", "Asia", "África", "Oceanía"],
        "Suramérica": ["Colombia", "Brasil", "Argentina", "Perú", "Chile", "Ecuador"],
        "Colombia": ["Antioquia", "Cundinamarca", "Valle del Cauca", "Atlántico"],
        "Antioquia": ["Medellín", "Guarne", "Rionegro", "Bello", "Envigado"],
        "Medellín": ["Comuna 1", "Comuna 2", "Corregimiento Santa Elena"]
    }
    
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("Configuración Espacial")
        
        # 1. Selector Macro
        nivel_macro = st.selectbox("Nivel Macro (Contenedor):", list(jerarquia.keys()))
        
        # 2. Selector Micro (Depende estrictamente del Macro)
        opciones_micro = jerarquia.get(nivel_macro, [])
        nivel_micro = st.selectbox("Nivel Micro (Anidado):", opciones_micro)
        
        st.markdown("---")
        st.caption("Método de Participación (Share):")
        metodo_part = st.radio("Cálculo de Cuota:", ["Constante (Último Censo)", "Tendencial (Cambio proyectado)"])
        cuota_base = st.slider(f"% que representa {nivel_micro} dentro de {nivel_macro}:", 0.1, 100.0, 15.0, step=0.1) / 100.0
        
    with col_a2:
        # Generamos una proyección Macro base matemática para visualizar
        t_ani = np.arange(2024, 2060)
        pob_macro = 50e6 / (1 + 0.1 * np.exp(-0.02 * (t_ani - 2024))) 
        
        if metodo_part == "Constante (Último Censo)":
            pob_micro = pob_macro * cuota_base
        else:
            tendencia = np.linspace(cuota_base, cuota_base * 1.15, len(t_ani)) 
            pob_micro = pob_macro * tendencia
            
        fig_ani = go.Figure()
        fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_macro, mode='lines', fill='tozeroy', name=f'Macro: {nivel_macro}', line=dict(color='#bdc3c7', width=1)))
        fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_micro, mode='lines', fill='tozeroy', name=f'Micro Anidado: {nivel_micro}', line=dict(color='#2ecc71', width=3)))
        
        fig_ani.update_layout(title=f"Downscaling Demográfico: {nivel_macro} ➔ {nivel_micro}", xaxis_title="Año", yaxis_title="Población", hovermode="x unified")
        st.plotly_chart(fig_ani, use_container_width=True)
