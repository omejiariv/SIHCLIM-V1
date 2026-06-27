import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
from modules.db_manager import get_engine
import ast

def renderizar_motor_escenarios_weap(territorio="Territorio Global", gdf_zona=None):
    engine = get_engine()
    
    # =====================================================================
    # 🚀 1. IDENTIFICACIÓN INTELIGENTE DEL TERRITORIO
    # =====================================================================
    if isinstance(territorio, list):
        lista_raw = territorio
    elif isinstance(territorio, str):
        if territorio.startswith("["):
            try: lista_raw = ast.literal_eval(territorio)
            except: lista_raw = [territorio]
        else:
            lista_raw = [territorio]
    else:
        lista_raw = ["Territorio Global"]

    nombres_exactos = []
    
    # 🚨 EL TRUCO MAESTRO: Si la lista es de puros números (estaciones), la ignoramos
    # y extraemos el nombre real desde el mapa (gdf_zona)
    es_estaciones = all(str(t).strip().isdigit() for t in lista_raw if str(t).strip())
    
    if not es_estaciones and lista_raw and lista_raw[0] not in ["Territorio Global", "-- Seleccione --"]:
        nombres_exactos = [str(t).replace("Cuencas Seleccionadas: ", "").replace("SZH: ", "").strip() for t in lista_raw]
        
    # 🗺️ RESCATE DESDE EL MAPA (Si recibimos estaciones, leemos las columnas del polígono)
    if not nombres_exactos and isinstance(gdf_zona, pd.DataFrame) and not gdf_zona.empty:
        # Buscamos de la escala más pequeña a la más grande
        columnas_jerarquia = ['nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh', 'nomzh', 'nomah', 'MPIO_CNMBR', 'municipio_clean', 'Territorio']
        for col in columnas_jerarquia:
            if col in gdf_zona.columns:
                valores = gdf_zona[col].dropna().unique()
                if len(valores) > 0 and str(valores[0]).strip() not in ["", "None", "nan"]:
                    nombres_exactos = [str(v).strip() for v in valores]
                    break # Encontramos la columna correcta, salimos del loop
                    
    if not nombres_exactos:
        nombres_exactos = ["Territorio Global"]

    # Generamos la versión CORTA (sin código) para buscar en la Matriz Hidrológica
    nombres_cortos = [n.split(" - (")[0].strip() for n in nombres_exactos if n != "Territorio Global"]

    if nombres_cortos:
        nombre_display = " + ".join(nombres_cortos[:2]) + ("..." if len(nombres_cortos)>2 else "")
        st.markdown(f"## ⚖️ Simulador de Estrés Hídrico: **{nombre_display}**")
    else:
        nombre_display = "Territorio Global"
        st.markdown(f"## ⚖️ Simulador de Estrés Hídrico: **{nombre_display}**")
        st.warning("⚠️ **Aviso:** Selecciona un territorio válido en el panel izquierdo.")

    # =====================================================================
    # 🚀 2. CAPTURA DEL ALEPH Y RESCATE INDEPENDIENTE
    # =====================================================================
    pob_aleph = float(st.session_state.get('aleph_pob_total', 0))
    oferta_aleph = float(st.session_state.get('aleph_oferta_m3s', 0.0))

    # RESCATE DEMOGRÁFICO: Buscamos con el nombre EXACTO (Con código IDEAM)
    if pob_aleph == 0 and nombres_exactos and nombres_exactos[0] != "Territorio Global":
        try:
            placeholders_p = ", ".join([f":p{i}" for i in range(len(nombres_exactos))])
            params_p = {f"p{i}": val for i, val in enumerate(nombres_exactos)}
            query_p = text(f"""
                SELECT SUM(CAST("Pob_Base" AS FLOAT)) as tot_pob 
                FROM matriz_multimodelo_demografica 
                WHERE "Territorio" IN ({placeholders_p})
            """)
            df_p = pd.read_sql(query_p, engine, params=params_p)
            if not df_p.empty and pd.notnull(df_p.iloc[0]['tot_pob']):
                pob_aleph = float(df_p.iloc[0]['tot_pob'])
        except Exception:
            pass

    # RESCATE HÍDRICO: Buscamos con el nombre CORTO en la Matriz Hidrológica
    if oferta_aleph == 0.0 and nombres_cortos:
        try:
            placeholders_h = ", ".join([f":h{i}" for i in range(len(nombres_cortos))])
            params_h = {f"h{i}": val for i, val in enumerate(nombres_cortos)}
            query_h = text(f"""
                SELECT SUM(CAST("Caudal_Medio_m3s" AS FLOAT)) as tot_q 
                FROM matriz_hidrologica_maestra 
                WHERE "Territorio" IN ({placeholders_h})
            """)
            df_h = pd.read_sql(query_h, engine, params=params_h)
            if not df_h.empty and pd.notnull(df_h.iloc[0]['tot_q']):
                oferta_aleph = float(df_h.iloc[0]['tot_q'])
        except Exception:
            pass
            
    # =====================================================================
    # 🚨 3. ENSAMBLE DEL BALANCE Y MODO DEMO PARCIAL
    # =====================================================================
    modo_demo_txt = []
    
    # Validamos Población de forma independiente
    if pob_aleph > 0:
        pob_base = pob_aleph
    else:
        pob_base = 150000  # Paracaídas visual si falla todo
        modo_demo_txt.append("Población")
        
    # Validamos Oferta Hídrica de forma independiente
    if oferta_aleph > 0.0:
        oferta_base_m3s = oferta_aleph
    else:
        oferta_base_m3s = 1.2  # Paracaídas visual si falla todo
        modo_demo_txt.append("Oferta Hídrica")

    # Consumo estimado: 150 Litros / habitante / día (convertido a m3/s)
    demanda_base_m3s = (pob_base * 150) / (1000 * 86400) 
    
    if modo_demo_txt:
        st.warning(f"⚠️ **Telemetría Parcial para '{nombre_display}':** Faltan datos en la BD para: {', '.join(modo_demo_txt)}. Se inyectaron valores de referencia para permitir la simulación visual.")
        estado_txt = "🟡 MODO MIXTO"
    else:
        estado_txt = "🟢 DATOS REALES CONECTADOS"
        
    st.markdown(f"📌 **Base Actual ({estado_txt}):** 👥 Población: `{pob_base:,.0f} hab` | 💧 Oferta Media: `{oferta_base_m3s:,.3f} m³/s`")
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

    # Modificadores Matemáticos
    oferta_modificada = oferta_base_m3s * (1 + (var_clima / 100))
    demanda_modificada = demanda_base_m3s * (1 + (var_pob / 100)) * (1 - (var_eficiencia / 100))

    # Curva estacional base
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    curva_estacional = np.array([0.7, 0.8, 1.0, 1.2, 1.3, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 0.9])
    
    oferta_mensual = oferta_modificada * curva_estacional
    demanda_mensual = np.full(12, demanda_modificada)

    # Gráfica WEAP
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=meses, y=oferta_mensual, name='Oferta Disponible', line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'))
    fig.add_trace(go.Scatter(x=meses, y=demanda_mensual, name='Demanda Total', line=dict(color='#e67e22', width=3, dash='dash')))
    
    # Relleno del déficit (Rojo)
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

    # Diagnóstico
    deficit_anual = np.sum(np.maximum(0, demanda_mensual - oferta_mensual))
    if deficit_anual > 0:
        st.error(f"⚠️ **Alerta de Vulnerabilidad:** El sistema entra en déficit hídrico.")
    else:
        st.success("✅ **Sistema Resiliente:** La oferta cubre la demanda.")
