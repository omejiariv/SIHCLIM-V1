# modules/demografia_tools.py

import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sqlalchemy import text
from modules.db_manager import get_engine

# 🔥 Importamos la única Aplanadora Maestra desde utils
from modules.utils import normalizar_texto_maestro

def obtener_poblacion_matriz(nombre_zona, anio_objetivo, gdf_contexto=None):
    """
    Buscador Francotirador de Matriz Demográfica.
    Usa gdf_contexto para desambiguar colisiones de nombres entre Macro y Micro cuencas.
    """
    from modules.db_manager import get_engine
    from sqlalchemy import text
    import pandas as pd
    import numpy as np
    
    engine = get_engine()
    try:
        from modules.utils import normalizar_texto_maestro
        nombre_norm = normalizar_texto_maestro(nombre_zona)
        
        q = text('SELECT * FROM matriz_maestra_demografica WHERE "Area" IN (\'Total\', \'total\', \'TOTAL\')')
        df_mat = pd.read_sql(q, engine)
        
        if not df_mat.empty:
            df_mat['MATCH_ID'] = df_mat['Territorio'].astype(str).apply(normalizar_texto_maestro)
            fila_ganadora = df_mat[df_mat['MATCH_ID'] == nombre_norm].copy()
            
            if not fila_ganadora.empty:
                # 🔥 RESOLUCIÓN INTELIGENTE DE COLISIONES (El Desambiguador)
                if len(fila_ganadora) > 1:
                    es_macro = True # Por defecto asumimos que es la macrocuenca
                    
                    # Si tenemos el mapa, leemos sus columnas para saber qué escala seleccionó el usuario
                    if gdf_contexto is not None and not gdf_contexto.empty:
                        # Si el nombre coincide con las columnas de micro-escala, sabemos que busca la pequeña
                        if 'nom_nss3' in gdf_contexto.columns and nombre_zona in gdf_contexto['nom_nss3'].values:
                            es_macro = False
                        elif 'nom_nss2' in gdf_contexto.columns and nombre_zona in gdf_contexto['nom_nss2'].values:
                            es_macro = False

                    if es_macro:
                        # Toma el gigante (Ej: ZH Nechí -> 4 Millones)
                        fila_ganadora = fila_ganadora.sort_values(by='Pob_Base', ascending=False) 
                    else:
                        # Toma el pequeño (Ej: NSS3 Nechí -> 29 Mil)
                        fila_ganadora = fila_ganadora.sort_values(by='Pob_Base', ascending=True) 

                row = fila_ganadora.iloc[0]
                
                t_val = float(anio_objetivo - row.get('Año_Base', 1985))
                mod = str(row.get('Modelo_Recomendado', ''))
                
                if 'Logistico' in mod: return row['Log_K'] / (1 + row['Log_a'] * np.exp(-row['Log_r'] * t_val))
                elif 'Exponencial' in mod: return row['Exp_a'] * np.exp(row['Exp_b'] * t_val)
                elif 'Polinomial' in mod: return row['Poly_A']*(t_val**3) + row['Poly_B']*(t_val**2) + row['Poly_C']*t_val + row['Poly_D']
                else: return row['Pob_Base']
    except Exception:
        pass
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
