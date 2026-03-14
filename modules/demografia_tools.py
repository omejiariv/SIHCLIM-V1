# modules/demografia_tools.py
import streamlit as st

def render_motor_demografico(lugar_defecto="Valle de Aburrá"):
    """Dibuja un mini-panel para calcular la población sin salir de la página."""
    
    st.info("⚠️ La proyección poblacional no está en memoria. Puedes calcularla en el Modelo Demográfico o generarla aquí.")
    
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        # Selector de Año
        anio_proyeccion = st.slider(
            "📅 Año de Proyección:", 
            min_value=2024, 
            max_value=2070, 
            value=st.session_state.get('aleph_anio', 2024), 
            step=1,
            key=f"demo_slider_{lugar_defecto}" # Key única para evitar conflictos
        )
        
    with col_btn2:
        st.write("") # Espaciador
        st.write("") # Espaciador
        if st.button("👥 Generar Proyección Demográfica Aquí", use_container_width=True, key=f"demo_btn_{lugar_defecto}"):
            with st.spinner(f"Encendiendo motor demográfico (DANE) para el año {anio_proyeccion}..."):
                
                # --- AQUÍ VA TU LÓGICA DE DEMOGRAFÍA ---
                # Como ejemplo, pondré un cálculo de crecimiento estándar (1.5% anual)
                # Si tienes una base de datos DANE, aquí harías el query o lectura del CSV
                
                pob_base = 4150000 # Población 2024 base
                anios_dif = anio_proyeccion - 2024
                pob_calculada = pob_base * ((1 + 0.015) ** anios_dif)
                
                # 🧠 INYECCIÓN DIRECTA AL ALEPH
                st.session_state['aleph_pob_total'] = pob_calculada
                st.session_state['aleph_anio'] = anio_proyeccion
                st.session_state['aleph_lugar'] = lugar_defecto
                
                st.success(f"✅ ¡Población de {lugar_defecto} calculada exitosamente para {anio_proyeccion}!")
                st.rerun() # Magia: Recarga la página para que los mapas y paneles lean el nuevo dato
