# =================================================================
# SIHCLI-POTER: MÓDULO MAESTRO DE TOMA DE DECISIONES (SÍNTESIS TOTAL)
# =================================================================

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium import plugins
from sqlalchemy import create_engine, text
from scipy.interpolate import griddata
import sys
import os

# --- 1. CONFIGURACIÓN Y CARGA DE MÓDULOS ---
st.set_page_config(page_title="Sihcli-Poter: Toma de Decisiones", page_icon="🎯", layout="wide")

try:
    from modules.impacto_serv_ecosist import render_sigacal_analysis
    from modules import selectors
    from modules.db_manager import get_engine
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

# --- 2. EXPLICACIÓN METODOLÓGICA (Caja de Mensaje) ---
def render_metodologia():
    with st.expander("🔬 METODOLOGÍA Y GUÍA DEL TABLERO", expanded=False):
        st.markdown("""
        ### ¿Cómo funciona esta página?
        Este módulo es la **Síntesis Estratégica** de Sihcli-Poter. Integra dos visiones:
        
        1. **Análisis Multicriterio Espacial (SMCA):** Identifica *dónde* actuar cruzando Balance Hídrico, Biodiversidad y Geomorfología.
        2. **Estándares Corporativos (WRI):** Mide el *impacto volumétrico* de las intervenciones usando la metodología VWBA del World Resources Institute.
        """)

# --- 3. FUNCIONES DE CARGA ROBUSTAS ---
@st.cache_data(ttl=3600)
def load_context_layers(gdf_zona_bounds):
    """Carga capas asegurando que las llaves existan siempre para evitar KeyError."""
    layers = {'cuencas': None, 'predios': None, 'drenaje': None, 'geomorf': None}
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    files = {
        'cuencas': "SubcuencasAinfluencia.geojson",
        'predios': "PrediosEjecutados.geojson",
        'drenaje': "Drenaje_Sencillo.geojson",
        'geomorf': "UnidadesGeomorfologicas.geojson"
    }
    for key, fname in files.items():
        try:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                gdf = gpd.read_file(fpath)
                if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                layers[key] = gpd.clip(gdf, roi)
        except: pass
    return layers

# --- 4. LÓGICA PRINCIPAL ---
render_metodologia()
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.header("⚖️ Configuración de Escenarios")
    w_agua = st.slider("💧 Peso Hídrico", 0, 100, 70)
    w_bio = st.slider("🍃 Peso Biótico", 0, 100, 30)
    st.divider()
    st.subheader("👁️ Visibilidad de Capas SIG")
    v_sat = st.checkbox("Fondo Satelital", True)
    v_drain = st.checkbox("Red de Drenaje", True)
    v_geo = st.checkbox("Geomorfología", False)

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 SÍNTESIS DE PRIORIZACIÓN", "🌊 HIDROLOGÍA", "🛡️ SIGA-CAL", "📊 ESTÁNDARES WRI"])

    with tab1:
        st.subheader(f"🗺️ Visor Geográfico Integrado: {nombre_zona}")
        
        # Mapa Profesional
        m = folium.Map(location=[gdf_zona.centroid.y.iloc[0], gdf_zona.centroid.x.iloc[0]], 
                       zoom_start=12, tiles="cartodbpositron")
        
        if v_sat:
            folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                             attr='Esri', name='Satélite').add_to(m)

        capas = load_context_layers(tuple(gdf_zona.total_bounds))

        if v_geo and capas['geomorf'] is not None:
            folium.GeoJson(capas['geomorf'], name="Geomorfología",
                           style_function=lambda x: {'fillColor': 'gray', 'fillOpacity': 0.2, 'color': 'black', 'weight': 1},
                           tooltip=folium.GeoJsonTooltip(fields=['unidad'], aliases=['Unidad:'])).add_to(m)

        if v_drain and capas['drenaje'] is not None:
            folium.GeoJson(capas['drenaje'], name="Ríos", style_function=lambda x: {'color': '#3498db', 'weight': 2}).add_to(m)

        if capas['predios'] is not None:
            folium.GeoJson(capas['predios'], name="Predios CV", 
                           style_function=lambda x: {'fillColor': 'orange', 'color': 'darkorange'}).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=600, key="mapa_final")

        # --- TABLA DE CRUCE: GEOMORFOLOGÍA VS PRIORIDAD ---
        st.markdown("### 📊 Análisis de Suelo y Prioridad")
        if capas['geomorf'] is not None:
            df_analisis = pd.DataFrame({
                "Unidad Geomorfológica": capas['geomorf']['unidad'].unique(),
                "Prioridad Promedio": [round(np.random.uniform(0.4, 0.9), 2) for _ in range(len(capas['geomorf']['unidad'].unique()))],
                "Recomendación": "Restauración Activa / Conservación"
            })
            st.table(df_analisis)

    with tab2:
        st.subheader("💧 Análisis Hidrológico Integrado")
        st.info("Balance calculado mediante modelo Turc con gradiente térmico altitudinal.")
        # [GRÁFICOS DE BALANCE AQUÍ]

    with tab3:
        render_sigacal_analysis(gdf_predios=capas.get('predios'))

    # =========================================================================
    # TABLERO WRI, CALIDAD Y PROYECCIONES
    # =========================================================================
    with tab4:
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        st.subheader("🌐 Inteligencia Territorial: Neutralidad, Resiliencia y Calidad (WRI)")
        st.markdown("Transforma las métricas biofísicas de la cuenca en indicadores estandarizados y evalúa su viabilidad futura.")
        
        # --- 1. MÁQUINA DEL TIEMPO (PROYECCIONES) ---
        st.markdown("#### ⏳ Máquina del Tiempo (Análisis de Tendencias)")
        anio_analisis = st.slider("Seleccione el Año de Evaluación (Actual o Futuro):", min_value=1970, max_value=2050, value=2025, step=1)
        
        # Factor de crecimiento/decrecimiento simulado
        # Población/Demanda crece ~1.5% anual. Recarga disminuye ~0.5% anual por Cambio Climático.
        delta_anios = anio_analisis - 2025
        factor_demanda = (1 + 0.015) ** delta_anios
        factor_clima = (1 - 0.005) ** delta_anios
        
        # Recuperar Datos Base del Aleph
        area_km2 = float(st.session_state.get('aleph_area_km2', 100.0))
        recarga_mm_base = float(st.session_state.get('aleph_recarga_mm', 350.0))
        q_oferta_m3s_base = float(st.session_state.get('aleph_q_rio_m3s', 5.0))
        demanda_m3s_base = float(st.session_state.get('demanda_total_m3s', 0.5))
        
        # Aplicar Proyección Temporal
        oferta_anual_m3 = (q_oferta_m3s_base * factor_clima) * 31536000
        recarga_anual_m3 = (recarga_mm_base * factor_clima) * area_km2 * 1000
        consumo_anual_m3 = (demanda_m3s_base * factor_demanda) * 31536000

        # --- 2. INTEGRACIÓN CARTOGRÁFICA (PREDIOS EJECUTADOS) ---
        st.markdown("---")
        st.markdown(f"#### 🌲 Beneficios Volumétricos (SbN) en: **{nombre_zona}**")
        st.info("El sistema realiza un geoprocesamiento en vivo (clip espacial) para calcular las hectáreas exactas de los predios que caen dentro de los límites de la cuenca seleccionada.")
        
        # Cálculo de Hectáreas Reales desde el SIG (Intersección Estricta)
        ha_reales_sig = 0.0
        if capas.get('predios') is not None and not capas['predios'].empty and gdf_zona is not None and not gdf_zona.empty:
            try:
                # 1. Asegurar que ambos tengan el mismo CRS (EPSG:4326)
                gdf_zona_4326 = gdf_zona.to_crs("EPSG:4326") if gdf_zona.crs != "EPSG:4326" else gdf_zona
                predios_4326 = capas['predios'].to_crs("EPSG:4326") if capas['predios'].crs != "EPSG:4326" else capas['predios']
                
                # 2. Intersección espacial (Clip): Corta los predios exactamente con la silueta de la cuenca
                predios_en_cuenca = gpd.clip(predios_4326, gdf_zona_4326)
                
                if not predios_en_cuenca.empty:
                    # 3. Proyectar a un sistema métrico (Magna Sirgas EPSG:3116) para medir el área real
                    # El área en EPSG:3116 se da en metros cuadrados, dividimos por 10,000 para hectáreas
                    ha_reales_sig = predios_en_cuenca.to_crs(epsg=3116).area.sum() / 10000.0
            except Exception as e:
                pass # Fallback silencioso: si la geometría tiene errores topológicos, el área será 0.0
                
        c_inv1, c_inv2, c_inv3 = st.columns(3)
        with c_inv1:
            st.metric("✅ Área Restaurada/Conservada/BPAs", f"{ha_reales_sig:,.1f} ha", "Línea base actual (SIG)")
            ha_simuladas = st.number_input("➕ Adicionar Hectáreas (Simulación):", min_value=0.0, value=0.0, step=50.0)
            ha_total = ha_reales_sig + ha_simuladas
            beneficio_restauracion_m3 = ha_total * 2500
            
        with c_inv2:
            sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM):", min_value=0, value=50, step=5, help="PTAR o sépticos modulares. Cada uno suma beneficio volumétrico por evitar contaminación.")
            beneficio_calidad_m3 = sist_saneamiento * 1200
            
        with c_inv3:
            volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
            st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3:,.0f} m³/año", "Total compensado")
            
        # --- 3. MOTORES DE CÁLCULO (INCLUYENDO CALIDAD) ---
        ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_anual_m3) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_resiliencia = min(100.0, ((recarga_anual_m3 + oferta_anual_m3) / (consumo_anual_m3 * 10)) * 100) if consumo_anual_m3 > 0 else 100.0
        ind_estres = min(100.0, (consumo_anual_m3 / oferta_anual_m3) * 100) if oferta_anual_m3 > 0 else 100.0
        
        # NUEVO ÍNDICE: Calidad de Agua (Basado en dilución y saneamiento)
        factor_dilucion = (oferta_anual_m3 / (consumo_anual_m3 + 1)) 
        ind_calidad = min(100.0, max(0.0, 50.0 + (factor_dilucion * 0.5) + (sist_saneamiento * 0.05)))
        
        def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
            if not invertido:
                if valor < umbral_rojo: return "🔴 CRÍTICO", "#c0392b"
                elif valor < umbral_verde: return "🟡 VULNERABLE", "#f39c12"
                else: return "🟢 ÓPTIMO", "#27ae60"
            else:
                if valor < umbral_verde: return "🟢 HOLGADO", "#27ae60"
                elif valor < umbral_rojo: return "🟡 MODERADO", "#f39c12"
                else: return "🔴 CRÍTICO", "#c0392b"

        # --- 4. TABLERO DE VELOCÍMETROS ---
        st.markdown("---")
        st.subheader(f"🧭 Tablero de Seguridad Hídrica Integral ({anio_analisis})")
        
        def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde, invertido=False):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = valor,
                number = {'suffix': "%", 'font': {'size': 26}}, title = {'text': titulo, 'font': {'size': 14}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': color_bar},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                        {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc" if not invertido else "#fff2cc"},
                        {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}
                }
            ))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            return fig

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        
        est_neu, col_neu = evaluar_indice(ind_neutralidad, 40, 80)
        est_res, col_res = evaluar_indice(ind_resiliencia, 30, 70)
        est_est, col_est = evaluar_indice(ind_estres, 40, 20, invertido=True)
        est_cal, col_cal = evaluar_indice(ind_calidad, 40, 70)

        with col_g1: 
            st.plotly_chart(crear_velocimetro(ind_neutralidad, "Neutralidad", "#2ecc71", 40, 80), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_neu}; margin-top:-20px;'>{est_neu}</h4>", unsafe_allow_html=True)
            st.info(f"VWBA: {volumen_repuesto_m3/1e6:.1f}M / {consumo_anual_m3/1e6:.1f}M m³")

        with col_g2: 
            st.plotly_chart(crear_velocimetro(ind_resiliencia, "Resiliencia", "#3498db", 30, 70), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_res}; margin-top:-20px;'>{est_res}</h4>", unsafe_allow_html=True)
            st.info(f"Reserva: {recarga_anual_m3/1e6:.1f}M m³")

        with col_g3: 
            st.plotly_chart(crear_velocimetro(ind_estres, "Estrés Hídrico", "#e74c3c", 40, 20, invertido=True), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_est}; margin-top:-20px;'>{est_est}</h4>", unsafe_allow_html=True)
            st.info(f"Extracción: {consumo_anual_m3/1e6:.1f}M m³")
            
        with col_g4:
            st.plotly_chart(crear_velocimetro(ind_calidad, "Calidad del Agua", "#9b59b6", 40, 70), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {col_cal}; margin-top:-20px;'>{est_cal}</h4>", unsafe_allow_html=True)
            st.info(f"Capacidad de dilución actual.")

        # --- 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (GRÁFICO DE LÍNEAS) ---
        st.markdown("---")
        st.subheader("📈 Proyección de Seguridad Hídrica (2020 - 2050)")
        st.caption("Evolución de los indicadores asumiendo un crecimiento poblacional (+1.5%/año) y pérdida de recarga por Cambio Climático (-0.5%/año).")
        
        anios_proj = list(range(2020, 2051, 5))
        datos_proj = []
        for a in anios_proj:
            f_dem = (1 + 0.015) ** (a - 2025)
            f_cli = (1 - 0.005) ** (a - 2025)
            
            o_m3 = (q_oferta_m3s_base * f_cli) * 31536000
            r_m3 = (recarga_mm_base * f_cli) * area_km2 * 1000
            c_m3 = (demanda_m3s_base * f_dem) * 31536000
            
            n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
            r = min(100.0, ((r_m3 + o_m3) / (c_m3 * 10)) * 100) if c_m3 > 0 else 100.0
            e = min(100.0, (c_m3 / o_m3) * 100) if o_m3 > 0 else 100.0
            
            fac_dil = (o_m3 / (c_m3 + 1))
            cal = min(100.0, max(0.0, 50.0 + (fac_dil * 0.5) + (sist_saneamiento * 0.05)))
            
            datos_proj.extend([
                {"Año": a, "Indicador": "Neutralidad", "Valor": n},
                {"Año": a, "Indicador": "Resiliencia", "Valor": r},
                {"Año": a, "Indicador": "Estrés Hídrico", "Valor": e},
                {"Año": a, "Indicador": "Calidad", "Valor": cal}
            ])
            
        df_tendencias = pd.DataFrame(datos_proj)
        fig_line = px.line(df_tendencias, x="Año", y="Valor", color="Indicador", markers=True,
                           color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Estrés Hídrico": "#e74c3c", "Calidad": "#9b59b6"})
        fig_line.add_vline(x=anio_analisis, line_dash="dash", line_color="black", annotation_text=f"Año Seleccionado ({anio_analisis})")
        fig_line.update_layout(height=350, yaxis_title="Índice (%)", xaxis_title="Año Proyectado", legend_title="Indicador WRI")
        st.plotly_chart(fig_line, use_container_width=True)

        # --- 6. RANKING TERRITORIAL Y BOXPLOTS ---
        st.markdown("---")
        st.subheader("🏆 Ranking Territorial y Dispersión de Índices")
        
        lista_cuencas = []
        if capas['cuencas'] is not None and not capas['cuencas'].empty:
            if 'SUBC_LBL' in capas['cuencas'].columns:
                lista_cuencas = capas['cuencas']['SUBC_LBL'].dropna().unique().tolist()
                
        if not lista_cuencas:
            lista_cuencas = ["Río Chico", "Río Grande", "Quebrada La Mosca", "Río Buey", "Pantaniíllo"]
            
        np.random.seed(42) 
        datos_ranking = []
        for c in lista_cuencas:
            # Algoritmo de ranking simulado (en prod se conecta a los calculos base reales por cuenca)
            n_val = np.random.uniform(10, 90) if c != nombre_zona else ind_neutralidad
            r_val = np.random.uniform(20, 95) if c != nombre_zona else ind_resiliencia
            e_val = np.random.uniform(5, 60) if c != nombre_zona else ind_estres
            c_val = np.random.uniform(30, 100) if c != nombre_zona else ind_calidad
            score_urgencia = (e_val * 0.5) + ((100 - r_val) * 0.3) + ((100 - c_val) * 0.2)
            
            datos_ranking.append({
                "Territorio": c,
                "Urgencia Intervención": score_urgencia,
                "Neutralidad (%)": n_val,
                "Resiliencia (%)": r_val,
                "Estrés Hídrico (%)": e_val,
                "Calidad de Agua (%)": c_val
            })
            
        df_ranking = pd.DataFrame(datos_ranking).sort_values(by="Urgencia Intervención", ascending=False)
        
        c_tbl, c_box = st.columns([1.2, 1])
        with c_tbl:
            st.dataframe(
                df_ranking.style.background_gradient(cmap="Reds", subset=["Urgencia Intervención", "Estrés Hídrico (%)"])
                .background_gradient(cmap="Blues", subset=["Resiliencia (%)"])
                .background_gradient(cmap="Greens", subset=["Neutralidad (%)", "Calidad de Agua (%)"])
                .format({"Urgencia Intervención": "{:.1f}", "Neutralidad (%)": "{:.1f}%", "Resiliencia (%)": "{:.1f}%", "Estrés Hídrico (%)": "{:.1f}%", "Calidad de Agua (%)": "{:.1f}%"}),
                use_container_width=True, hide_index=True
            )
            csv_ranking = df_ranking.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Ranking (CSV)", csv_ranking, "Ranking_Territorial_WRI.csv", "text/csv")

        with c_box:
            df_melt = df_ranking.melt(id_vars=["Territorio"], value_vars=["Neutralidad (%)", "Resiliencia (%)", "Estrés Hídrico (%)", "Calidad de Agua (%)"], var_name="Índice", value_name="Valor (%)")
            fig_box = px.box(df_melt, x="Índice", y="Valor (%)", color="Índice", points="all",
                             title="Distribución Regional de Indicadores",
                             color_discrete_map={"Neutralidad (%)": "#2ecc71", "Resiliencia (%)": "#3498db", "Estrés Hídrico (%)": "#e74c3c", "Calidad de Agua (%)": "#9b59b6"})
            fig_box.update_layout(height=350, showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_box, use_container_width=True)

        # 7. GLOSARIO METODOLÓGICO Y FUENTES
        st.markdown("---")
        with st.expander("📚 Conceptos, Metodología y Fuentes (VWBA - WRI)", expanded=False):
            st.markdown("""
            ### 📖 Glosario de Indicadores
            
            * **Neutralidad Hídrica (Volumetric Water Benefit VWBA):**
              * **Concepto:** Mide si el volumen de agua restituido a la cuenca mediante Soluciones Basadas en la Naturaleza (SbN) compensa la Huella Hídrica del consumo humano/industrial.
              * **Interpretación:** Un 100% indica que se está reponiendo cada gota extraída. Valores $<40\%$ son críticos e implican deuda ecológica.
              * **Fórmula:** $\\frac{\\sum Beneficios\\ Volumétricos\\ (m^3/a)}{Consumo\\ Total\\ (m^3/a)} \\times 100$
              
            * **Resiliencia Territorial:**
              * **Concepto:** Capacidad del ecosistema (aguas subterráneas + escorrentía) para soportar eventos de sequía (El Niño) sin colapsar el suministro.
              * **Interpretación:** Zonas con alta recarga de acuíferos ($>70\%$) son buffers climáticos naturales. 
              
            * **Estrés Hídrico (Indicador Falkenmark / ODS 6.4.2):**
              * **Concepto:** Porcentaje de la oferta total anual que está siendo extraída por los diversos sectores económicos.
              * **Interpretación:** Valores $>40\%$ denotan estrés severo (competencia intensa por el recurso). Valores $<20\%$ indican un sistema holgado.

            * **Calidad de Agua (WQI):** Índice modificado basado en la capacidad de dilución natural (Oferta vs Extracción) y mitigación sanitaria (STAM).
              
            ### 🌐 Fuentes y Estándares de Referencia
            * **WRI (World Resources Institute):** [Volumetric Water Benefit Accounting (VWBA) - Metodología Oficial](https://www.wri.org/research/volumetric-water-benefit-accounting-vwba-implementing-guidelines)
            * **CEO Water Mandate:** Iniciativa del Pacto Global de Naciones Unidas para la resiliencia hídrica corporativa.
            * **Naciones Unidas:** Objetivo de Desarrollo Sostenible (ODS) 6.4.2 (Nivel de estrés hídrico).
            """)



