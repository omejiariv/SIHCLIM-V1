# pages/15_⚖️_Escenarios_WEAP.py

import sys
import os
import streamlit as st
import pandas as pd
import requests
import io
from sqlalchemy import text

# 1. RUTA Y MÓDULOS (Declarados una sola vez)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from modules.db_manager import get_engine
from modules import selectors, escenarios_weap

# 2. Configuración de página
st.set_page_config(page_title="SIHCLI | Escenarios WEAP", page_icon="⚖️", layout="wide")

# 3. Renderizar menú
selectors.renderizar_menu_navegacion("Escenarios WEAP")

# 4. SELECTOR ESPACIAL
st.sidebar.markdown("---")
# 🚀 FIX APLICADO: Invocamos el selector exigiendo la firma explícita para WEAP
nombre_zona, gdf_zona, nivel_jerarquico, es_busqueda_global = selectors.render_selector_espacial(modo_firma="weap")

# 5. GUARDIA Y LÓGICA PRINCIPAL
if nivel_jerarquico == "Estaciones":
    st.error("🛑 **Escala Geográfica Incorrecta**")
    st.warning("El simulador WEAP requiere una unidad territorial que contenga población (como una **Cuenca** o un **Municipio**).")
else:
    territorio_final = nombre_zona
    
    # 🔥 FIX: Validar si no hay selección para no ejecutar consultas en vano
    if not territorio_final or territorio_final in [["-- Seleccione --"], "-- Seleccione --"]:
        st.info("👆 Selecciona un territorio en el panel izquierdo para cargar el simulador hidrosocial.")
    else:
        # Como la firma está blindada, ya no necesitamos rescatar nombres perdidos
        # Solo garantizamos que lo que evaluemos sea texto
        territorio_str = territorio_final[0] if isinstance(territorio_final, list) else territorio_final

        # ENCENDER MOTOR SI HAY TERRITORIO VÁLIDO
        if territorio_str != "Territorio Global":
            try:
                # --- 1. CONEXIÓN DEMOGRÁFICA (SUPABASE) ---
                url_csv = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Multimodelo_Demografica.csv"
                res_csv = requests.get(url_csv)
                df_dem = pd.read_csv(io.StringIO(res_csv.content.decode('utf-8')))
                
                nombre_puro = territorio_str.split(" - (")[0].strip() if " - (" in territorio_str else territorio_str.strip()
                mascara = (df_dem['Territorio'].astype(str).str.strip().str.lower() == nombre_puro.lower()) & \
                          (df_dem['Area'].astype(str).str.strip().str.lower() == 'total')
                
                row_pob = df_dem[mascara]
                if not row_pob.empty:
                    col_pob = next((c for c in row_pob.columns if 'pob_base' in c.lower()), 'Pob_Base')
                    st.session_state['aleph_pob_total'] = float(row_pob.iloc[0][col_pob])
                else:
                    st.sidebar.warning(f"⚠️ Demografía no hallada en el CSV para: {nombre_puro}")

                # --- 2. CONEXIÓN HIDROLÓGICA/RURH (SQL) ---
                engine = get_engine() 
                with engine.connect() as conn:
                    # 🛡️ Tolerancia de 120 segundos para evitar colapsos
                    conn.execute(text("SET statement_timeout = '120000'"))
                    
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
