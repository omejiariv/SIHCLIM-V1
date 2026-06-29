import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
from modules.db_manager import get_engine
import ast
import geopandas as gpd

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
    # 🚀 2. CAPTURA DEL ALEPH (CONEXIÓN TOTAL SQL)
    # =====================================================================
    pob_aleph = float(st.session_state.get('aleph_pob_total', 0))
    oferta_aleph = float(st.session_state.get('aleph_oferta_m3s', 0.0))
    rurh_aleph = float(st.session_state.get('aleph_concesiones_m3s', 0.0))

    if nombres_exactos and nombres_exactos[0] != "Territorio Global":
        territorio_llave = nombres_exactos[0]
        params = {"t0": territorio_llave}

        # 1. Leer Población (Demografía)
        if pob_aleph == 0:
            try:
                df_p = pd.read_sql(text('SELECT CAST("Pob_Base" AS FLOAT) as tot_pob FROM matriz_multimodelo_demografica WHERE "Territorio" = :t0 LIMIT 1'), engine, params=params)
                if not df_p.empty and pd.notnull(df_p.iloc[0]['tot_pob']): 
                    pob_aleph = float(df_p.iloc[0]['tot_pob'])
                    st.session_state['aleph_pob_total'] = pob_aleph
            except Exception: pass

        # 2. Leer Oferta Bruta (Hidrología)
        if oferta_aleph == 0.0:
            try:
                df_h = pd.read_sql(text('SELECT CAST("Caudal_Medio_m3s" AS FLOAT) as tot_q FROM matriz_hidrologica_maestra WHERE "Territorio" = :t0 LIMIT 1'), engine, params=params)
                if not df_h.empty and pd.notnull(df_h.iloc[0]['tot_q']): 
                    oferta_aleph = float(df_h.iloc[0]['tot_q'])
                    st.session_state['aleph_oferta_m3s'] = oferta_aleph
            except Exception: pass

        # 3. Leer Extracciones (Inyección RURH 65k)
        if rurh_aleph == 0.0:
            try:
                df_r = pd.read_sql(text('SELECT CAST("Presion_Total_RURH_m3s" AS FLOAT) as tot_rurh FROM matriz_presiones_rurh WHERE "Territorio" = :t0 LIMIT 1'), engine, params=params)
                if not df_r.empty and pd.notnull(df_r.iloc[0]['tot_rurh']): 
                    rurh_aleph = float(df_r.iloc[0]['tot_rurh'])
                    st.session_state['aleph_concesiones_m3s'] = rurh_aleph
            except Exception: pass

    # =====================================================================
    # 🧬 3. FORJA DE LA CURVA ESTACIONAL REAL (Autosuficiente)
    # =====================================================================
    clave_curva = f"curva_{nombres_exactos[0]}" if nombres_exactos else "curva_global"
    
    if clave_curva not in st.session_state:
        curva_estacional = np.array([0.7, 0.8, 1.0, 1.2, 1.3, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 0.9])
        estacion_usada = "Curva Sintética Estándar"
        
        # 🛡️ TRUCO MAESTRO ACTUALIZADO: Búsqueda flexible del polígono
        gdf_zona_real = gdf_zona
        if not isinstance(gdf_zona_real, pd.DataFrame) or gdf_zona_real.empty:
            if nombres_exactos and nombres_exactos[0] != "Territorio Global":
                try:
                    termino_busqueda = f"%{nombres_cortos[0]}%"
                    q_geom = text("""
                        SELECT geometry FROM cuencas 
                        WHERE nom_nss3 ILIKE :busqueda OR subc_lbl ILIKE :busqueda LIMIT 1
                    """)
                    gdf_zona_real = gpd.read_postgis(q_geom, engine, params={"busqueda": termino_busqueda}, geom_col="geometry")
                except Exception:
                    pass

        # Con el mapa asegurado, buscamos la estación
        if isinstance(gdf_zona_real, pd.DataFrame) and not gdf_zona_real.empty:
            try:
                df_estaciones = pd.read_sql("SELECT id_estacion, nombre, latitud, longitud FROM estaciones", engine)
                df_estaciones['longitud'] = pd.to_numeric(df_estaciones['longitud'], errors='coerce')
                df_estaciones['latitud'] = pd.to_numeric(df_estaciones['latitud'], errors='coerce')
                df_estaciones = df_estaciones.dropna(subset=['longitud', 'latitud'])
                
                gdf_est = gpd.GeoDataFrame(df_estaciones, geometry=gpd.points_from_xy(df_estaciones.longitud, df_estaciones.latitud), crs="EPSG:4326")
                
                if gdf_zona_real.crs is None: gdf_zona_real.set_crs("EPSG:4326", inplace=True)
                if gdf_est.crs != gdf_zona_real.crs: gdf_est = gdf_est.to_crs(gdf_zona_real.crs)
                
                cruce = gpd.sjoin_nearest(gdf_zona_real, gdf_est, how="left", distance_col="dist")
                
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
    
    # 🔥 FIX 2: Texto dinámico de la curva cuando falla la búsqueda de estación IDEAM
    texto_fallback = f"Patrón Sintético Local ({nombres_cortos[0] if nombres_cortos else 'Territorio'})"
    fuente_curva = st.session_state.get(f"est_{clave_curva}", texto_fallback)
    if "Curva Sintética Estándar" in fuente_curva:
        fuente_curva = texto_fallback

    # =====================================================================
    # 4. BASES MATEMÁTICAS (Preparación)
    # =====================================================================
    
    # 1. Rescatamos prioritariamente la memoria de la sesión (Aleph Telemetría)
    pob_memoria = float(st.session_state.get('aleph_pob_total', 0))
    pob_base = pob_memoria if pob_memoria > 0 else 150000
    
    oferta_memoria = float(st.session_state.get('aleph_oferta_m3s', 0.0))
    oferta_base_m3s = oferta_memoria if oferta_memoria > 0.0 else 1.2
    
    demanda_agro_base_m3s = float(st.session_state.get('aleph_concesiones_m3s', 0.0))

    # 2. Fórmulas de metabolismo base
    demanda_humana_m3s = (pob_base * 150) / (1000 * 86400) # Asume dotación estándar
    caudal_ecologico_m3s = oferta_base_m3s * 0.25 

    st.markdown(f"📌 **Oferta Neta Disponible:** `{oferta_base_m3s:,.3f} m³/s` | 👥 **Población Local:** `{pob_base:,.0f} hab`")

    # =====================================================================
    # 5. PANEL DE CONTROL AVANZADO (Escenarios RURH y Vertimientos)
    # =====================================================================

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌦️ Escenarios de Oferta**")
        var_clima = st.slider("Clima (El Niño/La Niña) %", -50, 50, 0, step=5)
        var_calidad = st.slider("☣️ Castigo por Vertimientos %", 0, 80, 0, step=5, help="Simula caudal inutilizable por alta contaminación.")
        
    with col2:
        st.markdown("**👥 Presiones de Demanda**")
        var_pob = st.slider("Crecimiento Poblacional %", 0, 100, 15, step=5)
        
        # 🔥 BLINDAJE RURH: Límite dinámico. Asegura que el máximo siempre sea mayor que el valor base 
        # para que Streamlit no arroje error "value must be between min and max".
        tope_max_rurh = max(float(oferta_base_m3s * 2), float(demanda_agro_base_m3s * 1.5), 0.1)
        
        # El slider controla el valor en vivo, partiendo de la realidad oficial de PostgreSQL
        var_rurh = st.slider(
            "🏭 Carga Agroindustrial RURH (m³/s)", 
            min_value=0.0, 
            max_value=float(tope_max_rurh), 
            value=float(demanda_agro_base_m3s), 
            step=0.01,
            help="Caudal concesionado total según la matriz RURH (PostGIS)."
        )
        
    with col3:
        st.markdown("**⚙️ Medidas de Mitigación**")
        var_eficiencia = st.slider("Eficiencia Acueductos %", 0, 40, 0, step=5)
        var_reuso = st.slider("🔄 Reuso Agroindustrial %", 0, 50, 0, step=5)

    # 🧮 MOTOR DE RECÁLCULO (En Vivo)
    oferta_efectiva = (oferta_base_m3s * (1 + (var_clima / 100))) * (1 - (var_calidad / 100))
    d_hum_mod = demanda_humana_m3s * (1 + (var_pob / 100)) * (1 - (var_eficiencia / 100))
    d_agro_mod = var_rurh * (1 - (var_reuso / 100))
    demanda_total_fija = d_hum_mod + d_agro_mod + caudal_ecologico_m3s

    # 📊 IMPRESIÓN EN VIVO DE LAS PRESIONES (Ahora sí responde a los sliders)
    st.info(f"🔍 **Anatomía de las Presiones (En Vivo):** 💧 Humano: `{d_hum_mod:,.3f} m³/s` | 🌿 Ecológico: `{caudal_ecologico_m3s:,.3f} m³/s` | 🏭 RURH (Agro/Ind): `{d_agro_mod:,.3f} m³/s`")
    st.caption(f"🌧️ **Dinámica Climatológica Basada en:** `{fuente_curva}`")
    st.markdown("---")

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    oferta_mensual = oferta_efectiva * curva_dinamica
    demanda_mensual = np.full(12, demanda_total_fija)

    # =====================================================================
    # 6. GRÁFICA WEAP 2.0 (ARQUITECTURA DE MONTAÑA Y CUCHILLO)
    # =====================================================================
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # 1. PREPARACIÓN DE LAS CAPAS APILADAS EXACTAS
    y_eco = np.full(12, caudal_ecologico_m3s)
    y_hum = y_eco + d_hum_mod
    y_total_demanda = y_hum + d_agro_mod  # Reemplaza a demanda_mensual
    
    # 2. CONSTRUCCIÓN DE LA MONTAÑA DE DEMANDA (De abajo hacia arriba)
    # Capa Base: Caudal Ecológico (Verde)
    fig.add_trace(go.Scatter(
        x=meses, y=y_eco, name='Caudal Ecológico',
        line=dict(width=0), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.6)'
    ))
    
    # Capa Media: Demanda Humana (Amarillo)
    fig.add_trace(go.Scatter(
        x=meses, y=y_hum, name='Demanda Humana',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(241, 196, 15, 0.6)'
    ))
    
    # Capa Superior: Presión Agroindustrial RURH (Rojo)
    fig.add_trace(go.Scatter(
        x=meses, y=y_total_demanda, name='Presión RURH (Agro/Ind)',
        line=dict(color='#e74c3c', width=2), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.6)'
    ))

    # 3. EL CUCHILLO: Línea de Oferta Neta Dinámica
    fig.add_trace(go.Scatter(
        x=meses, y=oferta_mensual, name='Oferta Neta',
        line=dict(color='#3498db', width=4), mode='lines'
    ))

    # 4. LA SOMBRA DEL DÉFICIT CRÍTICO
    # Truco Plotly: Trazamos una línea invisible en la oferta, y rellenamos 
    # hacia arriba SOLO hasta donde la demanda la supere.
    top_envelope = np.maximum(oferta_mensual, y_total_demanda)
    fig.add_trace(go.Scatter(x=meses, y=oferta_mensual, showlegend=False, hoverinfo='skip', line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=meses, y=top_envelope, name='⚠️ DÉFICIT CRÍTICO',
        fill='tonexty', fillcolor='rgba(0, 0, 0, 0.6)', line=dict(width=0)
    ))

    # 5. RENDERIZADO Y ESTÉTICA
    fig.update_layout(
        title="Balance de Masas - Presiones Apiladas", 
        xaxis_title="Meses", yaxis_title="Caudal (m³/s)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6. MOTOR DE ALARMAS
    deficit_anual = np.sum(np.maximum(0, y_total_demanda - oferta_mensual))
    if deficit_anual > 0:
        st.error("⚠️ **Colapso Hídrico Detectado:** El sistema no puede sostener la suma del caudal ecológico, poblacional y las concesiones. Revisa las franjas oscuras (déficit) en el gráfico.")
    else:
        st.success("✅ **Sistema en Equilibrio:** La oferta actual logra sostener las capas de presión configuradas sin generar déficit crítico.")
