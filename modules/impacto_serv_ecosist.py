# modules/impacto_serv_ecosist.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium import plugins

def render_sigacal_analysis(gdf_predios=None):
    st.subheader("📊 Análisis de Servicios Ecosistémicos - Modelo SIGA-CAL")
    
    # Carga de datos con manejo de errores
    try:
        df_siga = pd.read_csv('SIGACAL_RioGrande_Omejia_V2.csv', sep=';', decimal=',')
        for c in ['AreaAcu_ha', 'AreaAcuPer', 'S']:
            if c in df_siga.columns:
                df_siga[c] = df_siga[c].astype(str).str.replace('.', '').str.replace(',', '.')
                df_siga[c] = pd.to_numeric(df_siga[c], errors='coerce')
    except Exception as e:
        st.error(f"No se pudo cargar el archivo SIGA-CAL: {e}")
        return

    # --- MÉTRICAS DE IMPACTO ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Máx. Retención Sedimentos", f"{df_siga['Dk_sedimentos_tru_acum'].max()*100:.1f}%")
    m2.metric("Eficiencia Nitrógeno", f"{df_siga['Dk_N_tru_acum'].max()*100:.1f}%")
    m3.metric("Flujo Base (Dk)", f"{df_siga['Dk_flujoBase_tru_acum'].mean():.3f}")

    # --- GRÁFICO DE CURVA DE DESEMPEÑO ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_siga['AreaAcu_ha'], y=df_siga['Dk_sedimentos_tru_acum'], name="Sedimentos", line=dict(color='brown')))
    fig.add_trace(go.Scatter(x=df_siga['AreaAcu_ha'], y=df_siga['Dk_N_tru_acum'], name="Nitrógeno", line=dict(color='green')))
    fig.update_layout(title="Curva de Eficiencia vs Área Acumulada", xaxis_title="Hectáreas", yaxis_title="Índice Dk")
    st.plotly_chart(fig, use_container_width=True)

    # --- MAPA DE LOCALIZACIÓN ---
    st.markdown("### 🗺️ Localización de Intervenciones")
    m = folium.Map(location=[6.59, -75.45], zoom_start=11, tiles="CartoDB positron")
    plugins.Fullscreen().add_to(m)
    
    if gdf_predios is not None and not gdf_predios.empty:
        folium.GeoJson(gdf_predios, name="Predios").add_to(m)
    
    st_folium(m, width="100%", height=400, key="mapa_siga")
