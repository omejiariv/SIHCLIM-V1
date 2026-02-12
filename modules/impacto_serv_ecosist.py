# modules/impacto_serv_ecosist.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium import plugins
import os

def render_sigacal_analysis(gdf_predios=None):
    """
    Renderiza el análisis de impacto basado en los resultados de SIGA-CAL 
    para la cuenca del Río Grande.
    """
    st.subheader("📊 Análisis de Servicios Ecosistémicos - Modelo SIGA-CAL")
    
    # 1. LOCALIZACIÓN DEL ARCHIVO (Ruta robusta)
    # Buscamos el archivo en la raíz del proyecto
    file_path = 'SIGACAL_RioGrande_om_V2.csv'
    
    if not os.path.exists(file_path):
        st.error(f"⚠️ Archivo no encontrado: {file_path}. Asegúrate de que esté en la raíz del proyecto.")
        return

    # 2. CARGA Y LIMPIEZA DE DATOS (Ajustada a tu CSV específico)
    @st.cache_data
    def load_and_clean_siga():
        # Tu CSV usa ';' como separador y ',' como decimal
        df = pd.read_csv(file_path, sep=';', decimal=',')
        
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

    try:
        df_siga = load_and_clean_siga()
    except Exception as e:
        st.error(f"❌ Error al procesar los datos de SIGA-CAL: {e}")
        return

    # 3. INDICADORES CLAVE (Métricas para la Junta de EPM)
    st.markdown("#### Indicadores de Eficiencia Acumulada")
    m1, m2, m3 = st.columns(3)
    
    # Cálculos basados en tus columnas específicas
    max_sed = df_siga['Dk_sedimentos_tru_acum'].max() * 100
    max_n = df_siga['Dk_N_tru_acum'].max() * 100
    avg_fb = df_siga['Dk_flujoBase_tru_acum'].mean()
    
    m1.metric("Retención Sedimentos (Máx)", f"{max_sed:.1f}%", help="Capacidad máxima de captura de sedimentos")
    m2.metric("Eficiencia Nitrógeno", f"{max_n:.1f}%", help="Remoción de nutrientes (N)")
    m3.metric("Flujo Base (Promedio)", f"{avg_fb:.3f}", help="Estabilidad del flujo de agua")

    # 4. GRÁFICO DE CURVA DE DESEMPEÑO
    # Este gráfico es vital para mostrar dónde es más efectiva la inversión
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_siga['AreaAcu_ha'], 
        y=df_siga['Dk_sedimentos_tru_acum'], 
        name="Sedimentos", 
        line=dict(color='brown', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_siga['AreaAcu_ha'], 
        y=df_siga['Dk_N_tru_acum'], 
        name="Nitrógeno", 
        line=dict(color='green', width=2)
    ))
    
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

    # 5. MAPA DE LOCALIZACIÓN (Integración con tus predios de CuencaVerde)
    st.markdown("---")
    st.markdown("### 🗺️ Contexto Espacial de Intervenciones")
    
    m = folium.Map(location=[6.59, -75.45], zoom_start=11, tiles="CartoDB positron")
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.LocateControl(auto_start=False).add_to(m)
    
    if gdf_predios is not None and not gdf_predios.empty:
        # --- CORRECCIÓN DINÁMICA DE FIELDS Y ALIASES ---
        # Definimos los campos que queremos mostrar si existen en el GeoJSON
        posibles_campos = ['nombre_pre', 'municipio', 'area_ha', 'vereda']
        # Filtramos solo los que realmente existen en el archivo
        fields_existentes = [f for f in posibles_campos if f in gdf_predios.columns]
        
        # Creamos los alias correspondientes con la misma longitud
        mapa_alias = {
            'nombre_pre': 'Predio:',
            'municipio': 'Municipio:',
            'area_ha': 'Área (ha):',
            'vereda': 'Vereda:'
        }
        aliases_existentes = [mapa_alias[f] for f in fields_existentes]

        folium.GeoJson(
            gdf_predios, 
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
            ) if fields_existentes else None # Si no hay campos, no ponemos tooltip
        ).add_to(m)
    else:
        st.info("💡 No hay predios filtrados para mostrar en esta zona.")

    st_folium(m, width="100%", height=450, key="mapa_sigacal_final")
