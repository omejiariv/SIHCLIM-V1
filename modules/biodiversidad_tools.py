import streamlit as st

def render_motor_ripario():
    """Motor avanzado para Multi-Anillos Riparios (Multi-Ring Buffers)."""
    st.markdown("**⚙️ Configuración de Escenarios de Restauración**")
    
    col1, col2, col3, col_btn = st.columns([1, 1, 1, 1.5])
    
    # Recuperamos estados previos o usamos defaults (10, 20, 30)
    anillos = st.session_state.get('multi_rings', [10, 20, 30])
    
    with col1:
        v_min = st.number_input("Mínimo (m):", min_value=5, max_value=50, value=anillos[0], step=5, help="Escenario normativo (Ej: 10m)")
    with col2:
        v_med = st.number_input("Ideal (m):", min_value=10, max_value=100, value=anillos[1], step=5, help="Escenario recomendado (Ej: 20m)")
    with col3:
        v_max = st.number_input("Óptimo (m):", min_value=15, max_value=200, value=anillos[2], step=5, help="Escenario ecológico total (Ej: 30m)")
        
    with col_btn:
        st.write("") 
        st.write("") 
        if st.button("🌿 Calcular 3 Escenarios", use_container_width=True, type="primary"):
            if v_min >= v_med or v_med >= v_max:
                st.error("❌ Los valores deben ir de menor a mayor (Ej: 10, 20, 30).")
            else:
                # Guardamos la lista de escenarios
                st.session_state['multi_rings'] = [v_min, v_med, v_max]
                # Guardamos el valor máximo como "buffer_m_ripario" por compatibilidad con otras funciones
                st.session_state['buffer_m_ripario'] = v_max 
                st.success(f"✅ Escenarios listos: {v_min}m | {v_med}m | {v_max}m.")
                st.rerun()
