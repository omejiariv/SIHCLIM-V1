# pages/15_⚖️_Escenarios_WEAP.py

import sys
import os
import streamlit as st

# 1. Configuración de página
st.set_page_config(page_title="SIHCLI | Escenarios WEAP", page_icon="⚖️", layout="wide")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import selectors, escenarios_weap

# 2. Renderizar el menú principal
selectors.renderizar_menu_navegacion("Escenarios WEAP")

# 3. 🚀 INYECTAR EL SELECTOR ESPACIAL EN EL SIDEBAR
st.sidebar.markdown("---")
# Llamamos a tu selector para elegir Cuenca, Municipio o Región (que ya trae la telemetría incorporada)
nombre_zona, gdf_zona, nivel_jerarquico, es_busqueda_global = selectors.render_selector_espacial()

# 4. ENCENDER EL MOTOR (Pasándole el nombre exacto de la zona)
try:
    escenarios_weap.renderizar_motor_escenarios_weap(nombre_zona)
except Exception as e:
    st.error(f"Error al cargar el simulador: {e}")
