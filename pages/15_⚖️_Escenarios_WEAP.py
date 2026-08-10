# pages/15_⚖️_Escenarios_WEAP.py

import sys
import os
import streamlit as st
import pandas as pd
from sqlalchemy import text

# 1. RUTA Y MÓDULOS
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from modules.db_manager import get_engine
from modules import selectors, escenarios_weap
from modules.utils import normalizar_texto

# 2. Configuración de página
st.set_page_config(page_title="SIHCLI | Escenarios WEAP", page_icon="⚖️", layout="wide")

# 3. Renderizar menú
selectors.renderizar_menu_navegacion("Escenarios WEAP")

# 4. SELECTOR ESPACIAL
st.sidebar.markdown("---")
ids_estaciones, nombre_zona, altitud_ref, gdf_zona, nivel_jerarquico = selectors.render_selector_espacial(modo_firma="weap")

# 5. GUARDIA Y LÓGICA PRINCIPAL
if nivel_jerarquico == "Estaciones":
    st.error("🛑 **Escala Geográfica Incorrecta**")
    st.warning("El simulador WEAP requiere una unidad territorial que contenga población (como una **Cuenca** o un **Municipio**).")
else:
    territorio_final = nombre_zona
    
    if not territorio_final or territorio_final in [["-- Seleccione --"], "-- Seleccione --"]:
        st.info("👆 Selecciona un territorio en el panel izquierdo para cargar el simulador hidrosocial.")
    else:
        territorio_str = territorio_final[0] if isinstance(territorio_final, list) else territorio_final

        # ENCENDER MOTOR SI HAY TERRITORIO VÁLIDO
        if territorio_str != "Territorio Global":
            try:
                engine = get_engine() 
                nombre_puro = territorio_str.split(" - (")[0].strip() if " - (" in territorio_str else territorio_str.strip()
                nombre_normalizado = normalizar_texto(nombre_puro)
                
                with engine.connect() as conn:
                    # 🛡️ Tolerancia de 120 segundos para evitar colapsos
                    conn.execute(text("SET statement_timeout = '120000'"))
                    
                    # --- 1. CONEXIÓN DEMOGRÁFICA (Aleph SQL - Inteligencia Activa) ---
                    # 🚀 FIX: Leemos directamente de la matriz viva entrenada en la Pág 06
                    q_demo = text('''
                        SELECT "Pob_Base" 
                        FROM matriz_maestra_demografica 
                        WHERE UPPER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("Territorio", 'Á', 'A'), 'É', 'E'), 'Í', 'I'), 'Ó', 'O'), 'Ú', 'U'), ' ', ''))) = :t_norm 
                        AND UPPER(TRIM("Area")) = 'TOTAL'
                        LIMIT 1
                    ''')
                    pob_real = conn.execute(q_demo, {"t_norm": nombre_normalizado}).scalar()
                    
                    if pob_real is not None:
                        st.session_state['aleph_pob_total'] = float(pob_real)
                    else:
                        # Fallback en memoria local por si acaso
                        st.session_state['aleph_pob_total'] = st.session_state.get('pob_hum_calc_met', 0.0)
                        if st.session_state['aleph_pob_total'] == 0.0:
                            st.sidebar.warning(f"⚠️ Demografía no hallada en el Aleph para: {nombre_puro}. Por favor, entrena la matriz en la Pág 06.")

                    # --- 2. CONEXIÓN HIDROLÓGICA/RURH (SQL) ---
                    q_h = text('SELECT "Caudal_Medio_m3s" FROM matriz_hidrologica_maestra WHERE "Territorio" = :t LIMIT 1')
                    oferta_real = conn.execute(q_h, {"t": territorio_str}).scalar()
                    if oferta_real is not None:
                        st.session_state['aleph_oferta_m3s'] = float(oferta_real)
                        
                    q_r = text('SELECT COALESCE(SUM("Presion_Total_RURH_m3s"), 0) FROM matriz_presiones_rurh WHERE "Territorio" = :t')
                    rurh_real = conn.execute(q_r, {"t": territorio_str}).scalar()
                    st.session_state['aleph_concesiones_m3s'] = float(rurh_real)
                    
            except Exception as e:
                st.error(f"Error crítico en sincronización Aleph: {e}")

        # Renderizado final
        escenarios_weap.renderizar_motor_escenarios_weap(territorio_final, gdf_zona)
