# modules/impacto_serv_ecosist.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium import plugins
from modules.config import Config

def render_sigacal_analysis(gdf_predios=None):
    """
    Renderiza el análisis de impacto basado en los resultados de SIGA-CAL 
    para la cuenca del Río Grande, leyendo directamente desde la nube (Supabase).
    """
    st.subheader("📊 Análisis de Servicios Ecosistémicos - Modelo SIGA-CAL")
    
    # 1. LOCALIZACIÓN DEL ARCHIVO (100% CLOUD NATIVE)
    # Leemos directamente desde el bucket público maestro usando la configuración central
    file_url = f"{Config.BUCKET_MAESTROS}/SIGACAL_RioGrande_om_V2.csv"
    
    # 2. CARGA Y LIMPIEZA DE DATOS (Adaptada a URLs)
    @st.cache_data(ttl=3600)
    def load_and_clean_siga(url):
        try:
            # Pandas lee la URL directamente sin descargar el archivo localmente
            df = pd.read_csv(url, sep=';', decimal=',')
            
            # Limpieza de columnas numéricas (manejo de puntos de miles)
            cols_to_fix = ['AreaAcu_ha', 'AreaAcuPer', 'S']
            for col in cols_to_fix:
                if col in df.columns:
                    # Convertimos a string, quitamos puntos de miles y cambiamos coma por punto
                    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eliminar columnas vacías (Unnamed)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df
        except Exception as e:
            st.error(f"⚠️ No se pudo cargar el modelo SIGA-CAL desde la nube. Verifica que 'SIGACAL_RioGrande_om_V2.csv' esté subido en el bucket 'sihcli_maestros'. Detalle: {e}")
            return None

    df_siga = load_and_clean_siga(file_url)
    
    if df_siga is None or df_siga.empty:
        return

    # 3. INDICADORES CLAVE (Métricas para la Junta de EPM)
    st.markdown("#### Indicadores de Eficiencia Acumulada")
    m1, m2, m3 = st.columns(3)
    
    # Cálculos basados en tus columnas específicas (con protección de nulos)
    max_sed = df_siga['Dk_sedimentos_tru_acum'].max() * 100 if 'Dk_sedimentos_tru_acum' in df_siga.columns else 0.0
    max_n = df_siga['Dk_N_tru_acum'].max() * 100 if 'Dk_N_tru_acum' in df_siga.columns else 0.0
    avg_fb = df_siga['Dk_flujoBase_tru_acum'].mean() if 'Dk_flujoBase_tru_acum' in df_siga.columns else 0.0
    
    m1.metric("Retención Sedimentos (Máx)", f"{max_sed:.1f}%", help="Capacidad máxima de captura de sedimentos")
    m2.metric("Eficiencia Nitrógeno", f"{max_n:.1f}%", help="Remoción de nutrientes (N)")
    m3.metric("Flujo Base (Promedio)", f"{avg_fb:.3f}", help="Estabilidad del flujo de agua")

    # 4. GRÁFICO DE CURVA DE DESEMPEÑO
    # Este gráfico es vital para mostrar dónde es más efectiva la inversión
    fig = go.Figure()
    
    if all(col in df_siga.columns for col in ['AreaAcu_ha', 'Dk_sedimentos_tru_acum']):
        fig.add_trace(go.Scatter(
            x=df_siga['AreaAcu_ha'], 
            y=df_siga['Dk_sedimentos_tru_acum'], 
            name="Sedimentos", 
            line=dict(color='brown', width=3)
        ))
    
    if all(col in df_siga.columns for col in ['AreaAcu_ha', 'Dk_N_tru_acum']):
        fig.add_trace(go.Scatter(
            x=df_siga['AreaAcu_ha'], 
            y=df_siga['Dk_N_tru_acum'], 
            name="Nitrógeno", 
            line=dict(color='green', width=2)
        ))
    
    if all(col in df_siga.columns for col in ['AreaAcu_ha', 'Dk_P_tru_acum']):
        fig.add_trace(go.Scatter(
            x=df_siga['AreaAcu_ha'], 
            y=df_siga['Dk_P_tru_acum'], 
            name="Fósforo", 
            line=dict(color='orange', width=2)
        ))

    fig.update_layout(
        title="<b>Curva de Eficiencia Ambiental vs Área Drenada</b>",
        xaxis_title="Área Acumulada (Hectáreas)",
        yaxis_title="Índice de Eficiencia (Dk)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- ANÁLISIS DE IMPACTO ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig2 = go.Figure()
        if all(col in df_siga.columns for col in ['AreaAcu_ha', 'Dk_sedimentos_tru_acum']):
            fig2.add_trace(go.Scatter(x=df_siga['AreaAcu_ha'], y=df_siga['Dk_sedimentos_tru_acum'], name="Retención Sedimentos", line=dict(color='brown', width=3)))
        fig2.update_layout(title="<b>Curva de Eficiencia: Retención de Sedimentos</b>", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.write("#### 🛡️ Diagnóstico de la Cuenca")
        if 'Dk_sedimentos_tru_acum' in df_siga.columns:
            avg_dk = df_siga['Dk_sedimentos_tru_acum'].mean()
            if avg_dk < 0.4:
                st.error(f"**Prioridad Alta:** La eficiencia media es baja ({avg_dk:.2f}). Se requieren intervenciones en la parte alta.")
            else:
                st.success(f"**Estado Estable:** Eficiencia media de {avg_dk:.2f}.")
        
        # Análisis de Cruce con Predios
        if gdf_predios is not None and not gdf_predios.empty:
            st.write(f"**Intervenciones detectadas:** {len(gdf_predios)} predios.")
            st.info("Estos predios protegen áreas que aportan a la estabilidad del Flujo Base.")
        else:
            st.warning("No se detectan predios de CuencaVerde en el área de este modelo.")

    # 5. MAPA DE LOCALIZACIÓN (Integración con tus predios de CuencaVerde)
    st.markdown("---")
    st.markdown("### 🗺️ Contexto Espacial de Intervenciones")
    
    # Centrado dinámico del mapa basado en los predios
    if gdf_predios is not None and not gdf_predios.empty:
        gdf_4326 = gdf_predios.to_crs("EPSG:4326") if gdf_predios.crs != "EPSG:4326" else gdf_predios
        center_lat = gdf_4326.centroid.y.mean()
        center_lon = gdf_4326.centroid.x.mean()
    else:
        center_lat, center_lon = 6.59, -75.45 # Coordenadas por defecto (Río Grande)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.LocateControl(auto_start=False).add_to(m)
    
    if gdf_predios is not None and not gdf_predios.empty:
        # --- CORRECCIÓN DINÁMICA DE FIELDS Y ALIASES ---
        posibles_campos = ['nombre_pre', 'municipio', 'area_ha', 'vereda']
        fields_existentes = [f for f in posibles_campos if f in gdf_4326.columns]
        
        mapa_alias = {
            'nombre_pre': 'Predio:',
            'municipio': 'Municipio:',
            'area_ha': 'Área (ha):',
            'vereda': 'Vereda:'
        }
        aliases_existentes = [mapa_alias.get(f, f) for f in fields_existentes]

        folium.GeoJson(
            gdf_4326, 
            name="Predios Intervenidos",
            style_function=lambda x: {
                'fillColor': '#e67e22', 
                'color': '#d35400', 
                'weight': 1, 
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields_existentes,
                aliases=aliases_existentes,
                localize=True
            ) if fields_existentes else None 
        ).add_to(m)
    else:
        st.info("💡 No hay predios filtrados para mostrar en esta zona.")

    st_folium(m, width="100%", height=450, key="mapa_sigacal_final")
