# modules/biodiversidad_tools.py
import streamlit as st

def render_motor_ripario(gdf_zona=None):
    """Dibuja un mini-panel para calcular la franja riparia sin salir de la página."""
    
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        # Selector de ancho del buffer (Típico para franjas protectoras: 30m)
        ancho_ripario = st.number_input(
            "Ancho de Protección (Metros):", 
            min_value=10, 
            max_value=100, 
            value=30, 
            step=10,
            help="Distancia desde el eje del río a proteger (Ley 99 de 1993 exige 30m)."
        )
        
    with col_btn2:
        st.write("") # Espaciador
        st.write("") # Espaciador
        
        # Validamos si ya existe la red de ríos en la memoria global
        if 'gdf_rios' not in st.session_state:
            st.error("❌ Faltan los ríos. Debes generar la Red Hídrica primero.")
        else:
            if st.button("🌿 Calcular Franja Riparia Aquí", use_container_width=True):
                with st.spinner(f"Geoprocesando buffer de {ancho_ripario}m sobre la red hídrica..."):
                    try:
                        import geopandas as gpd
                        
                        # Rescatamos los ríos de la memoria
                        gdf_rios = st.session_state['gdf_rios']
                        
                        # --- AQUÍ VA TU LÓGICA DE BIODIVERSIDAD (BUFFER) ---
                        # Ejemplo estándar de Geopandas:
                        # Aseguramos un CRS proyectado en metros (ej. MAGNA-SIRGAS origen nacional EPSG:9377 o 3116)
                        rios_proyectados = gdf_rios.to_crs(epsg=3116) 
                        
                        # Aplicamos el buffer
                        buffer_ripario = rios_proyectados.copy()
                        buffer_ripario['geometry'] = rios_proyectados.buffer(ancho_ripario)
                        
                        # Volvemos a WGS84 para los mapas de Folium
                        buffer_ripario = buffer_ripario.to_crs(epsg=4326)
                        
                        # Unificamos todo en un solo polígono (Opcional, pero recomendado)
                        # riparia_unificada = buffer_ripario.unary_union
                        
                        # 🧠 INYECCIÓN DIRECTA A LA MEMORIA GLOBAL
                        st.session_state['gdf_riparia'] = buffer_ripario
                        st.session_state['ancho_ripario_usado'] = ancho_ripario
                        
                        st.success(f"✅ ¡Franja Riparia de {ancho_ripario}m calculada y guardada en memoria!")
                        st.rerun() # Recarga la página para que Priorización Predial detecte la capa
                    
                    except Exception as e:
                        st.error(f"Error en el geoproceso: {e}")
