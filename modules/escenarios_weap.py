import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
from modules.db_manager import get_engine

def renderizar_motor_escenarios_weap(territorio="Territorio Global", gdf_zona=None):
    engine = get_engine()
    
    # =====================================================================
    # 🚀 1. EXTRACCIÓN QUIRÚRGICA DE NOMBRES REALES
    # =====================================================================
    nombres_reales = []
    
    # Estrategia A: Sacar los nombres directamente de los datos del mapa (Infalible)
    if gdf_zona is not None and not gdf_zona.empty:
        columnas_posibles = ['Territorio', 'nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh', 'nomzh', 'nomah', 'MPIO_CNMBR', 'Municipio']
        col_nombre = next((col for col in columnas_posibles if col in gdf_zona.columns), None)
        if col_nombre:
            nombres_reales = gdf_zona[col_nombre].astype(str).dropna().unique().tolist()
            
    # Estrategia B: Si falla el mapa, limpiamos el texto del título a la fuerza
    if not nombres_reales and isinstance(territorio, str):
        clean_str = territorio.replace("Cuencas Seleccionadas: ", "").replace("SZH: ", "").replace("ZH: ", "").replace("AH: ", "").replace("Municipio: ", "")
        partes = [n.strip().split(" - (")[0] for n in clean_str.split("+")]
        nombres_reales = [p for p in partes if p and p != "-- Seleccione --"]
        
    nombre_display = territorio if isinstance(territorio, str) else "Territorio Global"
    st.markdown(f"## ⚖️ Simulador de Estrés Hídrico: **{nombre_display}**")

    # =====================================================================
    # 🚀 2. CAPTURA Y RESCATE SQL INTELIGENTE
    # =====================================================================
    pob_aleph = st.session_state.get('aleph_pob_total', 0)
    oferta_aleph = st.session_state.get('aleph_oferta_m3s', 0.0)

    # RESCATE: Buscamos los nombres limpios (ej: 'Q. La Honda') en la Matriz Maestra
    if (pob_aleph == 0 or oferta_aleph == 0.0) and nombres_reales:
        try:
            placeholders = ", ".join([f":p{i}" for i in range(len(nombres_reales))])
            params = {f"p{i}": val for i, val in enumerate(nombres_reales)}
            
            query = text(f"""
                SELECT 
                    SUM(CAST("Poblacion" AS FLOAT)) as tot_pob, 
                    SUM(CAST("Caudal_Medio_m3s" AS FLOAT)) as tot_q 
                FROM matriz_hidrologica_maestra 
                WHERE "Territorio" IN ({placeholders}) 
            """)
            
            df_rescue = pd.read_sql(query, engine, params=params)
            
            if not df_rescue.empty:
                val_pob = df_rescue.iloc[0]['tot_pob']
                val_caudal = df_rescue.iloc[0]['tot_q']
                if pd.notnull(val_pob) and float(val_pob) > 0: pob_aleph = float(val_pob)
                if pd.notnull(val_caudal) and float(val_caudal) > 0: oferta_aleph = float(val_caudal)
        except Exception:
            pass 

    # =====================================================================
    # 🚨 3. ALERTA FORENSE Y DEMO
    # =====================================================================
    if pob_aleph == 0 or oferta_aleph == 0.0:
        st.warning(f"⚠️ **Telemetría Inactiva:** No se hallaron datos en la Matriz Maestra. Mostrando valores de calibración.")
        pob_base, oferta_base_m3s, modo_demo = 500000, 1.2, True 
    else:
        pob_base, oferta_base_m3s, modo_demo = pob_aleph, oferta_aleph, False

    demanda_base_m3s = (pob_base * 150) / (1000 * 86400) 
    
    estado_txt = "🔴 MODO DEMO (CALIBRACIÓN)" if modo_demo else "🟢 DATOS REALES CONECTADOS"
    st.markdown(f"📌 **Base Actual ({estado_txt}):** 👥 Población: `{pob_base:,.0f} hab` | 💧 Oferta Media: `{oferta_base_m3s:,.2f} m³/s`")
    st.markdown("---")

    # =====================================================================
    # 4. PANEL DE CONTROL Y MOTOR WEAP
    # =====================================================================
    col1, col2, col3 = st.columns(3)
    with col1:
        var_clima = st.slider("🌦️ Oferta (Clima)", -50, 50, 0, step=5, help="Negativo = Sequía (El Niño) | Positivo = Lluvia (La Niña)")
    with col2:
        var_pob = st.slider("👥 Demanda (Población)", 0, 100, 15, step=5, help="Crecimiento poblacional en %")
    with col3:
        var_eficiencia = st.slider("⚙️ Gestión (Eficiencia)", 0, 40, 0, step=5, help="Reducción de consumo en %")

    oferta_modificada = oferta_base_m3s * (1 + (var_clima / 100))
    demanda_modificada = demanda_base_m3s * (1 + (var_pob / 100)) * (1 - (var_eficiencia / 100))

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    curva_estacional = np.array([0.7, 0.8, 1.0, 1.2, 1.3, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 0.9])
    
    oferta_mensual = oferta_modificada * curva_estacional
    demanda_mensual = np.full(12, demanda_modificada)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=meses, y=oferta_mensual, name='Oferta Disponible', line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'))
    fig.add_trace(go.Scatter(x=meses, y=demanda_mensual, name='Demanda Total', line=dict(color='#e67e22', width=3, dash='dash')))
    
    oferta_minima = np.minimum(oferta_mensual, demanda_mensual)
    fig.add_trace(go.Scatter(x=meses, y=demanda_mensual, line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=meses, y=oferta_minima, name='Déficit Hídrico (Unmet Demand)',
        fill='tonexty', fillcolor='rgba(231, 76, 60, 0.5)', line=dict(width=0)
    ))

    fig.update_layout(
        title="Proyección de Cobertura de Demanda (Estilo WEAP)", 
        xaxis_title="Meses", yaxis_title="Caudal (m³/s)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    deficit_anual = np.sum(np.maximum(0, demanda_mensual - oferta_mensual))
    if deficit_anual > 0:
        st.error(f"⚠️ **Alerta de Vulnerabilidad:** El sistema entra en déficit hídrico.")
    else:
        st.success("✅ **Sistema Resiliente:** La oferta cubre la demanda.")
