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
    if isinstance(territorio, list): lista_raw = territorio
    elif isinstance(territorio, str):
        if territorio.startswith("["):
            try: lista_raw = ast.literal_eval(territorio)
            except: lista_raw = [territorio]
        else: lista_raw = [territorio]
    else: lista_raw = ["Territorio Global"]

    nombres_exactos = []
    es_estaciones = all(str(t).strip().isdigit() for t in lista_raw if str(t).strip())
    
    if not es_estaciones and lista_raw and lista_raw[0] not in ["Territorio Global", "-- Seleccione --"]:
        nombres_exactos = [str(t).replace("Cuencas Seleccionadas: ", "").replace("SZH: ", "").strip() for t in lista_raw]
        
    if not nombres_exactos and isinstance(gdf_zona, pd.DataFrame) and not gdf_zona.empty:
        columnas_jerarquia = ['nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh', 'nomzh', 'nomah', 'MPIO_CNMBR', 'municipio_clean', 'Territorio']
        for col in columnas_jerarquia:
            if col in gdf_zona.columns:
                valores = gdf_zona[col].dropna().unique()
                if len(valores) > 0 and str(valores[0]).strip() not in ["", "None", "nan"]:
                    nombres_exactos = [str(v).strip() for v in valores]
                    break 
                    
    if not nombres_exactos: nombres_exactos = ["Territorio Global"]
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

    if pob_aleph == 0 and nombres_exactos and nombres_exactos[0] != "Territorio Global":
        try:
            placeholders_p = ", ".join([f":p{i}" for i in range(len(nombres_exactos))])
            params_p = {f"p{i}": val for i, val in enumerate(nombres_exactos)}
            df_p = pd.read_sql(text(f'SELECT SUM(CAST("Pob_Base" AS FLOAT)) as tot_pob FROM matriz_multimodelo_demografica WHERE "Territorio" IN ({placeholders_p})'), engine, params=params_p)
            if not df_p.empty and pd.notnull(df_p.iloc[0]['tot_pob']): pob_aleph = float(df_p.iloc[0]['tot_pob'])
        except Exception: pass

    if oferta_aleph == 0.0 and nombres_cortos:
        try:
            placeholders_h = ", ".join([f":h{i}" for i in range(len(nombres_cortos))])
            params_h = {f"h{i}": val for i, val in enumerate(nombres_cortos)}
            df_h = pd.read_sql(text(f'SELECT SUM(CAST("Caudal_Medio_m3s" AS FLOAT)) as tot_q FROM matriz_hidrologica_maestra WHERE "Territorio" IN ({placeholders_h})'), engine, params=params_h)
            if not df_h.empty and pd.notnull(df_h.iloc[0]['tot_q']): oferta_aleph = float(df_h.iloc[0]['tot_q'])
        except Exception: pass

    # =====================================================================
    # 🧬 3. FORJA DE LA CURVA ESTACIONAL REAL (Caché en Aleph)
    # =====================================================================
    clave_curva = f"curva_{nombres_exactos[0]}" if nombres_exactos else "curva_global"
    
    if clave_curva not in st.session_state:
        curva_estacional = np.array([0.7, 0.8, 1.0, 1.2, 1.3, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 0.9])
        estacion_usada = "Curva Sintética Estándar"
        
        if isinstance(gdf_zona, pd.DataFrame) and not gdf_zona.empty:
            try:
                import geopandas as gpd
                df_estaciones = pd.read_sql("SELECT id_estacion, nombre, latitud, longitud FROM estaciones", engine)
                df_estaciones['longitud'] = pd.to_numeric(df_estaciones['longitud'], errors='coerce')
                df_estaciones['latitud'] = pd.to_numeric(df_estaciones['latitud'], errors='coerce')
                df_estaciones = df_estaciones.dropna(subset=['longitud', 'latitud'])
                
                gdf_est = gpd.GeoDataFrame(df_estaciones, geometry=gpd.points_from_xy(df_estaciones.longitud, df_estaciones.latitud), crs="EPSG:4326")
                
                if gdf_zona.crs is None: gdf_zona.set_crs("EPSG:4326", inplace=True)
                if gdf_est.crs != gdf_zona.crs: gdf_est = gdf_est.to_crs(gdf_zona.crs)
                
                cruce = gpd.sjoin_nearest(gdf_zona, gdf_est, how="left", distance_col="dist")
                
                if not cruce.empty and 'id_estacion' in cruce.columns:
                    id_est_cercana = str(cruce.iloc[0]['id_estacion'])
                    nombre_est_cercana = str(cruce.iloc[0]['nombre'])
                    
                    q_precip = text("SELECT fecha, valor FROM precipitacion WHERE id_estacion = :id_est")
                    df_precip = pd.read_sql(q_precip, engine, params={"id_est": id_est_cercana})
                    
                    if not df_precip.empty:
                        df_precip['mes'] = pd.to_datetime(df_precip['fecha']).dt.month
                        promedios = df_precip.groupby('mes')['valor'].mean()
                        mean_val = df_precip['valor'].mean()
                        prom_array = np.array([promedios.get(m, mean_val) for m in range(1, 13)])
                        
                        if prom_array.mean() > 0:
                            curva_estacional = prom_array / prom_array.mean()
                            estacion_usada = f"Estación {nombre_est_cercana} ({id_est_cercana})"
            except Exception as e: pass
            
        st.session_state[clave_curva] = curva_estacional
        st.session_state[f"est_{clave_curva}"] = estacion_usada
        
    curva_dinamica = st.session_state[clave_curva]
    fuente_curva = st.session_state.get(f"est_{clave_curva}", "Curva Sintética Estándar")

    # =====================================================================
    # 🚨 4. ENSAMBLE DEL BALANCE MULTIDIMENSIONAL (RURH + CALIDAD)
    # =====================================================================
    pob_base = pob_aleph if pob_aleph > 0 else 150000
    oferta_base_m3s = oferta_aleph if oferta_aleph > 0.0 else 1.2

    demanda_humana_m3s = (pob_base * 150) / (1000 * 86400)
    caudal_ecologico_m3s = oferta_base_m3s * 0.25 
    demanda_agro_base_m3s = float(st.session_state.get('aleph_concesiones_m3s', 0.0))

    st.markdown(f"📌 **Oferta Neta Disponible:** `{oferta_base_m3s:,.3f} m³/s` | 👥 **Población:** `{pob_base:,.0f} hab`")
    st.info(f"🔍 **Anatomía de las Presiones:** 💧 Humano: `{demanda_humana_m3s:,.3f} m³/s` | 🌿 Ecológico: `{caudal_ecologico_m3s:,.3f} m³/s` | 🏭 RURH: `{demanda_agro_base_m3s:,.3f} m³/s`")
    st.caption(f"🌧️ **Dinámica Climatológica Basada en:** `{fuente_curva}`")
    st.markdown("---")

    # =====================================================================
    # 5. PANEL DE CONTROL AVANZADO (Escenarios RURH y Vertimientos)
    # =====================================================================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌦️ Escenarios de Oferta**")
        var_clima = st.slider("Clima (El Niño/La Niña) %", -50, 50, 0, step=5)
        var_calidad = st.slider("☣️ Castigo por Vertimientos %", 0, 80, 0, step=5, help="Simula caudal inutilizable.")
    with col2:
        st.markdown("**👥 Presiones de Demanda**")
        var_pob = st.slider("Crecimiento Poblacional %", 0, 100, 15, step=5)
        var_rurh = st.slider("🏭 Carga Agroindustrial RURH (m³/s)", 0.0, float(oferta_base_m3s * 2), demanda_agro_base_m3s, step=0.05)
    with col3:
        st.markdown("**⚙️ Medidas de Mitigación**")
        var_eficiencia = st.slider("Eficiencia Acueductos %", 0, 40, 0, step=5)
        var_reuso = st.slider("🔄 Reuso Agroindustrial %", 0, 50, 0, step=5)

    # 🧮 MOTOR DE RECÁLCULO
    oferta_efectiva = (oferta_base_m3s * (1 + (var_clima / 100))) * (1 - (var_calidad / 100))
    d_hum_mod = demanda_humana_m3s * (1 + (var_pob / 100)) * (1 - (var_eficiencia / 100))
    d_agro_mod = var_rurh * (1 - (var_reuso / 100))
    demanda_total_fija = d_hum_mod + d_agro_mod + caudal_ecologico_m3s

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    # Aplicamos la curva física extraída de la base de datos
    oferta_mensual = oferta_efectiva * curva_dinamica
    demanda_mensual = np.full(12, demanda_total_fija)

    # =====================================================================
    # 6. GRÁFICA WEAP 2.0
    # =====================================================================
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=meses, y=oferta_mensual, name='Oferta Neta', line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.1)'))
    
    y_eco = np.full(12, caudal_ecologico_m3s)
    fig.add_trace(go.Scatter(x=meses, y=y_eco, name='Caudal Ecológico', line=dict(color='#27ae60', width=1, dash='dot'), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.2)'))
    
    y_hum = y_eco + d_hum_mod
    fig.add_trace(go.Scatter(x=meses, y=y_hum, name='Demanda Humana', line=dict(color='#f1c40f', width=1), fill='tonexty', fillcolor='rgba(241, 196, 15, 0.4)'))
    
    fig.add_trace(go.Scatter(x=meses, y=demanda_mensual, name='Presión RURH (Agro)', line=dict(color='#e74c3c', width=2), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.4)'))

    oferta_minima = np.minimum(oferta_mensual, demanda_mensual)
    fig.add_trace(go.Scatter(x=meses, y=demanda_mensual, line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=meses, y=oferta_minima, name='⚠️ DÉFICIT CRÍTICO', fill='tonexty', fillcolor='rgba(0, 0, 0, 0.5)', line=dict(width=0)))

    fig.update_layout(
        title="Balance de Masas - Presiones Apiladas", 
        xaxis_title="Meses", yaxis_title="Caudal (m³/s)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    deficit_anual = np.sum(np.maximum(0, demanda_mensual - oferta_mensual))
    if deficit_anual > 0:
        st.error(f"⚠️ **Colapso Hídrico Detectado:** El sistema no puede sostener la suma del caudal ecológico, poblacional y las concesiones. Revisa los meses críticos.")
    else:
        st.success("✅ **Sistema en Equilibrio:** La oferta actual logra sostener las capas de presión configuradas.")
