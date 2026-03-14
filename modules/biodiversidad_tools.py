import streamlit as st

def render_motor_ripario():
    """Motor para definir el ancho del corredor ripario."""
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        ancho_ripario = st.number_input(
            "Ancho de Protección Riparia (Metros):", 
            min_value=10, max_value=100, 
            value=st.session_state.get('buffer_m_ripario', 30), 
            step=5,
            key="rip_num_input"
        )
    with col_btn2:
        st.write("") 
        st.write("") 
        if st.button("🌿 Confirmar Ancho Ripario", use_container_width=True):
            st.session_state['buffer_m_ripario'] = ancho_ripario
            st.success(f"✅ Franja de {ancho_ripario}m configurada para la Cuenca.")
            st.rerun()
