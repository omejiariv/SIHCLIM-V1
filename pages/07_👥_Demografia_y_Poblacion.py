import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os

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

import os

# --- FUNCIÓN DE CARGA SEGURA ---
@st.cache_data
def cargar_datos_parquet(ruta_archivo):
    """Carga archivos parquet de forma segura. Retorna DataFrame vacío si no existe."""
    if os.path.exists(ruta_archivo):
        return pd.read_parquet(ruta_archivo)
    return pd.DataFrame()

# Cargar las bases de datos guardadas desde el Panel de Administración
df_macro = cargar_datos_parquet("data/poblacion_historica_macro.parquet")
df_piramide = cargar_datos_parquet("data/poblacion_edades_piramide.parquet")

# ------------------------------------------------------------------------------
# TAB 1: CENSOS (HISTÓRICO MACRO)
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial")
    
    if df_macro.empty:
        st.warning("⚠️ No se encontraron datos históricos. Por favor, sube el archivo desde el 'Panel de Administración'.")
    else:
        st.info("Datos cargados exitosamente. Visualizando la evolución comparativa de la población.")
        
        # Transformar los datos para Plotly (Melt)
        df_melt = df_macro.melt(id_vars=["Año"], var_name="Nivel Territorial", value_name="Población")
        
        # Limpiar los nombres para la leyenda (ej: Pob_Antioquia -> Antioquia)
        df_melt['Nivel Territorial'] = df_melt['Nivel Territorial'].str.replace('Pob_', '')
        
        # Selector para filtrar qué líneas ver
        niveles_disp = df_melt['Nivel Territorial'].unique()
        sel_niveles = st.multiselect("Territorios a visualizar:", niveles_disp, default=list(niveles_disp))
        
        # Filtrar y Graficar
        df_plot = df_melt[df_melt['Nivel Territorial'].isin(sel_niveles)]
        
        fig_historico = px.line(
            df_plot, x="Año", y="Población", color="Nivel Territorial",
            title="Crecimiento Poblacional Comparativo",
            markers=True, line_shape="spline"
        )
        # Escala logarítmica opcional para ver Medellín y Colombia en la misma gráfica sin aplastarse
        usar_log = st.checkbox("Usar escala logarítmica en el Eje Y", value=False)
        if usar_log:
            fig_historico.update_layout(yaxis_type="log")
            
        fig_historico.update_layout(hovermode="x unified", height=500)
        st.plotly_chart(fig_historico, use_container_width=True)
        
        with st.expander("Ver Base de Datos Pura"):
            st.dataframe(df_macro, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MODELOS DE CRECIMIENTO (Lo dejamos intacto por ahora)
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
        t = np.arange(0, t_max + 1)
        df_modelos = pd.DataFrame({"Año": t})
        
        if "Exponencial (Malthus)" in modelos_sel:
            df_modelos["Exponencial"] = p0 * np.exp(tasa_r * t)
        if "Logístico (Verhulst)" in modelos_sel:
            c = (k_cap - p0) / p0
            df_modelos["Logístico"] = k_cap / (1 + c * np.exp(-tasa_r * t))
        if "Gompertz" in modelos_sel:
            ln_p0_k = np.log(p0 / k_cap)
            df_modelos["Gompertz"] = k_cap * np.exp(ln_p0_k * np.exp(-tasa_r * t))

        fig_mod = go.Figure()
        colores = {"Exponencial": "#e74c3c", "Logístico": "#2ecc71", "Gompertz": "#f39c12"}
        for mod in df_modelos.columns[1:]:
            fig_mod.add_trace(go.Scatter(x=df_modelos["Año"], y=df_modelos[mod], mode='lines', name=mod, line=dict(width=3, color=colores.get(mod, 'blue'))))
        fig_mod.add_trace(go.Scatter(x=[0, t_max], y=[k_cap, k_cap], mode='lines', name='Capacidad de Carga (K)', line=dict(color='black', dash='dash')))
        fig_mod.update_layout(title="Comparativa de Modelos de Crecimiento", xaxis_title="Años Proyectados (t)", yaxis_title="Habitantes", hovermode="x unified", height=500)
        st.plotly_chart(fig_mod, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ESTRUCTURAS Y PIRÁMIDES
# ------------------------------------------------------------------------------
with tab_piramides:
    st.header("🏗️ Dinámica de Cohortes y Pirámides Poblacionales")
    
    if df_piramide.empty:
        st.warning("⚠️ No se encontraron datos de edades. Sube el archivo 'Estructura por Edades' en el Panel de Administración.")
    else:
        col_p1, col_p2 = st.columns([1, 3])
        
        with col_p1:
            st.info("Visualización de la estructura poblacional por género y edad simple.")
            # Obtener años disponibles
            anios_disp = sorted(df_piramide['Año'].unique())
            
            # Selector de año (Slider para poder "animarlo" manualmente)
            anio_sel = st.select_slider("Selecciona el Año:", options=anios_disp, value=anios_disp[0] if anios_disp else None)
            
            # Filtrar datos por año seleccionado
            df_filtro = df_piramide[df_piramide['Año'] == anio_sel].copy()
            
            # Métricas rápidas
            if not df_filtro.empty:
                tot_m = df_filtro['Male'].sum()
                tot_f = df_filtro['Female'].sum()
                st.metric("Total Hombres", f"{tot_m:,.0f}")
                st.metric("Total Mujeres", f"{tot_f:,.0f}")
                st.metric("Población Total", f"{(tot_m + tot_f):,.0f}")

        with col_p2:
            if not df_filtro.empty:
                # TRUCO DE LA PIRÁMIDE: Multiplicar hombres por -1 para que vayan a la izquierda
                df_filtro['Male_Plot'] = df_filtro['Male'] * -1
                
                # Crear la figura
                fig_piramide = go.Figure()
                
                # Barra Hombres (Izquierda)
                fig_piramide.add_trace(go.Bar(
                    y=df_filtro['Edad'], x=df_filtro['Male_Plot'],
                    name='Hombres', orientation='h', marker=dict(color='#3498db'),
                    hoverinfo='y+text', text=df_filtro['Male'].apply(lambda x: f"{x:,.0f}") # Texto sin el negativo
                ))
                
                # Barra Mujeres (Derecha)
                fig_piramide.add_trace(go.Bar(
                    y=df_filtro['Edad'], x=df_filtro['Female'],
                    name='Mujeres', orientation='h', marker=dict(color='#e74c3c'),
                    hoverinfo='y+text', text=df_filtro['Female'].apply(lambda x: f"{x:,.0f}")
                ))
                
                # Configurar diseño para que parezca pirámide
                fig_piramide.update_layout(
                    title=f"Pirámide Poblacional - Año {anio_sel}",
                    barmode='relative', # Clave para que se superpongan en el 0
                    bargap=0.1,
                    yaxis=dict(title='Edad Simple', dtick=5),
                    xaxis=dict(title='Población', tickformat=',.0f'),
                    height=600,
                    hovermode="y unified"
                )
                
                st.plotly_chart(fig_piramide, use_container_width=True)
                
# ------------------------------------------------------------------------------
# TAB 4: MODELOS ANIDADOS (Punto 6)
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("Downscaling Demográfico (Modelos Jerárquicos)")
    st.info("Conociendo el modelo de crecimiento macro (Ej. Nacional), forzaremos las proyecciones locales (Municipal) para que la suma de las partes coincida con el límite superior.")
