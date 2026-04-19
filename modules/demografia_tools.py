# modules/demografia_tools.py

import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sqlalchemy import text
from modules.db_manager import get_engine

# 🔥 Importamos la única Aplanadora Maestra desde utils
from modules.utils import normalizar_texto_maestro

def obtener_poblacion_matriz(nombre_zona, anio_objetivo):
    """
    Función Universal: Busca en la Matriz Maestra SQL, limpia el nombre y 
    calcula la población usando la ecuación matemática ganadora al vuelo.
    """
    engine = get_engine()
    pob_final = 0
    
    try:
        # 1. Traemos la matriz de coeficientes (Solo Totales)
        q = text('SELECT * FROM matriz_maestra_demografica WHERE "Area" IN (\'Total\', \'total\', \'TOTAL\')')
        df_mat = pd.read_sql(q, engine)
        
        if not df_mat.empty:
            # 2. Crear diccionario normalizado agresivo de la base de datos
            territorios_db = df_mat['Territorio'].dropna().unique().tolist()
            nombres_norm = {normalizar_texto_maestro(t): t for t in territorios_db}
            
            # 3. Normalizar lo que el usuario seleccionó en la UI 
            # (normalizar_texto_maestro ya se encarga internamente de quitar los "- NSS", tildes y abreviaturas)
            zona_norm = normalizar_texto_maestro(nombre_zona)
            match_name = None
            
            # 4. Emparejamiento (Fuzzy Match ajustado al 65% para atrapar variaciones difíciles)
            match_name = nombres_norm.get(zona_norm)
            if not match_name:
                matches = difflib.get_close_matches(zona_norm, nombres_norm.keys(), n=1, cutoff=0.65)
                if matches: match_name = nombres_norm[matches[0]]
            
            # 5. Cálculo Matemático Estricto
            if match_name:
                row = df_mat[df_mat['Territorio'] == match_name].iloc[0]
                mod = str(row.get('Modelo_Recomendado', ''))
                t_val = float(anio_objetivo - row.get('Año_Base', 1985))
                
                if 'Logistico' in mod: 
                    pob_final = row['Log_K'] / (1 + row['Log_a'] * np.exp(-row['Log_r'] * t_val))
                elif 'Exponencial' in mod: 
                    pob_final = row['Exp_a'] * np.exp(row['Exp_b'] * t_val)
                elif 'Polinomial' in mod: 
                    pob_final = row['Poly_A']*(t_val**3) + row['Poly_B']*(t_val**2) + row['Poly_C']*t_val + row['Poly_D']
                else: 
                    pob_final = row['Pob_Base']
                    
    except Exception as e:
        pass # Silencioso, el frontend maneja el Cero como advertencia
        
    return round(pob_final, 0)

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
