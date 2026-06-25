# modules/escenarios_weap.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def renderizar_motor_escenarios_weap(territorio="Territorio Global"):
    
    # 1. TÍTULO DINÁMICO
    if territorio and territorio != "-- Seleccione --":
        st.markdown(f"## ⚖️ Simulador de Estrés Hídrico: **{territorio}**")
    else:
        st.markdown("## ⚖️ Simulador de Estrés Hídrico: **Territorio Global**")
        st.warning("⚠️ **Aviso:** Selecciona una Cuenca o Municipio en el panel izquierdo.")

    st.info("Ajuste las variables de clima, población y eficiencia para proyectar el balance hídrico y detectar posibles déficits futuros.")

    # =====================================================================
    # 🚀 2. CAPTURA INTELIGENTE DE DATOS DEL ALEPH
    # =====================================================================
    # Buscamos en múltiples variables por si el nombre cambia en otros módulos
    pob_aleph = st.session_state.get('aleph_pob_total', st.session_state.get('poblacion_servida', 0))
    oferta_aleph = st.session_state.get('aleph_oferta_m3s', 0.0)

    # 🚨 SISTEMA DE ALERTA FORENSE
    if pob_aleph == 0 or oferta_aleph == 0.0:
        st.warning(f"⚠️ **Telemetría Inactiva para {territorio}:** El Aleph no encontró datos en la base. (Asegúrate de haber 'Forjado la Matriz' en el módulo de Hidrología). Activando Modo Demostración para no colapsar la gráfica.")
        pob_base = 150000
        oferta_base_m3s = 12.5
        modo_demo = True
    else:
        pob_base = pob_aleph
        oferta_base_m3s = oferta_aleph
        modo_demo = False

    # Cálculo de demanda base (asumiendo 150L / hab / día)
    demanda_base_m3s = (pob_base * 150) / (1000 * 86400) 
    
    # 📌 MOSTRAR CONTEXTO ACTUAL (Con semáforo de estado)
    estado_txt = "🔴 MODO DEMO" if modo_demo else "🟢 DATOS REALES"
    st.markdown(f"📌 **Base Actual ({estado_txt}):** 👥 Población: `{pob_base:,.0f} hab` | 💧 Oferta Media: `{oferta_base_m3s:,.2f} m³/s`")
    st.markdown("---")

    # =====================================================================
    # 3. PANEL DE CONTROL DE ESCENARIOS (Los "Sliders")
    # =====================================================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌦️ Oferta (Clima)")
        var_clima = st.slider(
            "Impacto ENSO / Cambio Climático", 
            min_value=-50, max_value=50, value=0, step=5,
            help="Simula reducción de caudales por El Niño (negativo) o aumento por La Niña (positivo)."
        )
        
    with col2:
        st.markdown("#### 👥 Demanda (Población)")
        var_pob = st.slider(
            "Crecimiento Poblacional Proyectado", 
            min_value=0, max_value=100, value=15, step=5,
            help="Simula el aumento de la demanda por expansión urbana."
        )
        
    with col3:
        st.markdown("#### ⚙️ Gestión (Eficiencia)")
        var_eficiencia = st.slider(
            "Reducción de Pérdidas / Ahorro", 
            min_value=0, max_value=40, value=0, step=5,
            help="Mejoras en acueductos o campañas de ahorro que reducen la demanda neta."
        )

    # 4. MOTOR MATEMÁTICO DE SIMULACIÓN
    oferta_modificada = oferta_base_m3s * (1 + (var_clima / 100))
    demanda_modificada = demanda_base_m3s * (1 + (var_pob / 100)) * (1 - (var_eficiencia / 100))

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    curva_estacional = np.array([0.7, 0.8, 1.0, 1.2, 1.3, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 0.9])
    
    oferta_mensual = oferta_modificada * curva_estacional
    demanda_mensual = np.full(12, demanda_modificada)

    # 5. GRÁFICA DE BALANCE (Estilo WEAP)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=meses, y=oferta_mensual, mode='lines', name='Oferta Disponible',
        line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'
    ))

    fig.add_trace(go.Scatter(
        x=meses, y=demanda_mensual, mode='lines', name='Demanda Total',
        line=dict(color='#e67e22', width=3, dash='dash')
    ))

    oferta_minima_grafico = np.minimum(oferta_mensual, demanda_mensual)
    
    fig.add_trace(go.Scatter(
        x=meses, y=demanda_mensual, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=meses, y=oferta_minima_grafico, mode='lines', name='Déficit Hídrico (Unmet Demand)',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.5)'
    ))

    fig.update_layout(
        title="Proyección de Cobertura de Demanda (Escenario Simulado)",
        xaxis_title="Meses", yaxis_title="Caudal (m³/s)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. DIAGNÓSTICO ESCRITO AL VUELO
    deficit_anual = np.sum(np.maximum(0, demanda_mensual - oferta_mensual))
    if deficit_anual > 0:
        st.error(f"⚠️ **Alerta de Vulnerabilidad:** Bajo este escenario, el sistema entra en déficit. La demanda no cubierta acumulada es crítica.")
    else:
        st.success("✅ **Sistema Resiliente:** Bajo este escenario, la oferta logra cubrir la demanda proyectada sin entrar en déficit.")
