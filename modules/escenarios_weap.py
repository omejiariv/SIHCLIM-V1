import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
from modules.db_manager import get_engine

def renderizar_motor_escenarios_weap(territorio="Territorio Global", gdf_zona=None):
    engine = get_engine()

    # 1. LIMPIEZA QUIRÚRGICA DE ENTRADA
    # Convertimos cualquier lista o string sucio en una lista de nombres "limpios" (sin códigos IDEAM)
    if isinstance(territorio, list):
        lista_raw = territorio
    elif isinstance(territorio, str):
        # Si es un string que parece lista (ej: "[...]")
        if territorio.startswith("["):
            import ast
            try: lista_raw = ast.literal_eval(territorio)
            except: lista_raw = [territorio]
        else:
            lista_raw = [territorio]
    else:
        lista_raw = [territorio]

    # Quitamos el código IDEAM: "Q. La Honda - (2308-01-04-24)" -> "Q. La Honda"
    nombres_limpios = [str(t).split(" - (")[0].strip() for t in lista_raw if str(t).strip() not in ["", "-- Seleccione --"]]
    
    # 🚨 FILTRO ANTI-ESTACIONES (Si el territorio tiene números largos, lo rechazamos)
    if any(any(char.isdigit() for char in n) and len(n) > 5 for n in nombres_limpios):
        st.warning("⚠️ El sistema detectó códigos de estaciones. Por favor, selecciona una Cuenca o Municipio en el filtro lateral.")
        return    
    
    # =====================================================================
    # 🚀 1. TRADUCTOR MAESTRO (De la Interfaz a la Base de Datos)
    # =====================================================================
    # A. Normalizamos la entrada (detectamos si es lista o texto)
    if isinstance(territorio, list):
        lista_raw = territorio
    elif isinstance(territorio, str):
        if territorio.startswith("[") and territorio.endswith("]"):
            import ast
            try:
                lista_raw = ast.literal_eval(territorio)
            except:
                lista_raw = [territorio]
        else:
            lista_raw = [territorio]
    else:
        lista_raw = ["Territorio Global"]

    # B. Limpiamos el código IDEAM (Traducción Inversa para los CSVs)
    # Convierte "Q. La Honda - (2308-01-04-24)" -> "Q. La Honda"
    nombres_limpios = []
    for t in lista_raw:
        t_str = str(t).replace("Cuencas Seleccionadas: ", "").replace("SZH: ", "")
        # Extraemos solo lo que está antes de " - ("
        nombre_real = t_str.split(" - (")[0].strip() if " - (" in t_str else t_str.strip()
        
        if nombre_real and nombre_real not in ["-- Seleccione --", "Territorio Global", "Ninguna"]:
            nombres_limpios.append(nombre_real)
            
    # C. Título Dinámico para la pantalla
    if nombres_limpios:
        nombre_display = " + ".join(nombres_limpios[:2]) + ("..." if len(nombres_limpios)>2 else "")
    else:
        nombre_display = "Territorio Global"
        st.warning("⚠️ **Aviso:** Selecciona un territorio válido en el panel izquierdo.")
        
    st.markdown(f"## ⚖️ Simulador de Estrés Hídrico: **{nombre_display}**")

    # =====================================================================
    # 🚀 2. CAPTURA DEL ALEPH Y RESCATE INDEPENDIENTE
    # =====================================================================
    # Tomamos lo que el Aleph ya haya calculado (¡Protegemos tus 7,563 habitantes!)
    pob_aleph = float(st.session_state.get('aleph_pob_total', 0))
    oferta_aleph = float(st.session_state.get('aleph_oferta_m3s', 0.0))

    # RESCATE HÍDRICO: Si el Aleph no tiene el caudal, lo buscamos limpiamente en la Matriz
    if oferta_aleph == 0.0 and nombres_limpios:
        try:
            placeholders = ", ".join([f":p{i}" for i in range(len(nombres_limpios))])
            params = {f"p{i}": val for i, val in enumerate(nombres_limpios)}
            # Sumamos los caudales de las cuencas seleccionadas
            query_h = text(f"""
                SELECT SUM(CAST("Caudal_Medio_m3s" AS FLOAT)) as tot_q 
                FROM matriz_hidrologica_maestra 
                WHERE "Territorio" IN ({placeholders})
            """)
            df_h = pd.read_sql(query_h, engine, params=params)
            if not df_h.empty and pd.notnull(df_h.iloc[0]['tot_q']):
                oferta_aleph = float(df_h.iloc[0]['tot_q'])
        except Exception as e:
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
