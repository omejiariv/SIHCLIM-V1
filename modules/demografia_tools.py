# modules/demografia_tools.py

import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sqlalchemy import text
from modules.db_manager import get_engine

# 🔥 Importamos la única Aplanadora Maestra desde utils
from modules.utils import normalizar_texto_maestro

def obtener_poblacion_matriz(nombre_zona, anio_objetivo, area_deseada="Total"):
    """
    Buscador de precisión quirúrgica basado en Llave Universal.
    """
    from modules.db_manager import get_engine
    from modules.utils import normalizar_texto
    import pandas as pd
    import numpy as np
    
    engine = get_engine()
    try:
        # 1. Construimos la búsqueda normalizada
        terr_norm = normalizar_texto(nombre_zona).upper()
        area_norm = area_deseada.upper()
        
        # 2. Consultamos por LIKE en la Llave Universal (es lo más rápido y seguro)
        q = text(f"""
            SELECT * FROM matriz_maestra_demografica 
            WHERE "LLAVE_UNIVERSAL" LIKE :pattern 
            AND "Area" = :area
        """)
        
        # El patrón busca cualquier nivel que contenga el territorio y el área
        df_res = pd.read_sql(q, engine, params={
            "pattern": f"%_{terr_norm}_%",
            "area": area_deseada
        })
        
        if not df_res.empty:
            # Si hay varios (ej. un municipio y una cuenca con mismo nombre), 
            # priorizamos por el nivel activo en el session_state
            nivel_activo = st.session_state.get('nivel_activo_global', 'Municipal')
            fila = df_res[df_res['Nivel'] == nivel_activo]
            if fila.empty: fila = df_res.head(1)
            
            row = fila.iloc[0]
            t_val = anio_objetivo - row['Año_Base']
            mod = row['Modelo_Recomendado']
            
            # Cálculo matemático según modelo...
            if mod == 'Logístico': return row['Log_K'] / (1 + row['Log_a'] * np.exp(-row['Log_r'] * t_val))
            elif mod == 'Exponencial': return row['Exp_a'] * np.exp(row['Exp_b'] * t_val)
            elif mod == 'Lineal': return row['Lin_m'] * t_val + row['Lin_b']
            else: return row['Poly_A']*(t_val**3) + row['Poly_B']*(t_val**2) + row['Poly_C']*t_val + row['Poly_D']
            
    except Exception: pass
    return 0

def render_motor_demografico(lugar_defecto="Antioquia"):
    """Mini-panel actualizado para usar la verdad de la Matriz SQL."""
    st.info(f"🧠 Conectado al Cerebro Demográfico: **{lugar_defecto}**")
    
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        anio_proyeccion = st.slider("📅 Año:", 2024, 2050, st.session_state.get('aleph_anio', 2024), key=f"ds_{lugar_defecto}")
        
    with col_btn2:
        st.write("") 
        if st.button("👥 Sincronizar Población Real", use_container_width=True, key=f"db_{lugar_defecto}"):
            with st.spinner("Consultando Matriz SQL..."):
                pob_calculada = obtener_poblacion_matriz(lugar_defecto, anio_proyeccion)
                
                if pob_calculada > 0:
                    st.session_state['aleph_pob_total'] = pob_calculada
                    st.session_state['aleph_anio'] = anio_proyeccion
                    st.session_state['aleph_lugar'] = lugar_defecto
                    st.success(f"✅ Sincronizado: {int(pob_calculada):,} hab.")
                    st.rerun()
                else:
                    st.warning("⚠️ No se encontró el territorio en la Matriz.")
