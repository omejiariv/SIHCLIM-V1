# pages/15_⚖️_Escenarios_WEAP.py

import sys
import os
import streamlit as st

# 1. Configuración de página (debe ser la primera línea)
st.set_page_config(page_title="SIHCLI | Escenarios WEAP", page_icon="⚖️", layout="wide")

# 2. Conexión con la carpeta de módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 3. Importar tus herramientas y el nuevo motor
from modules import selectors
from modules import escenarios_weap

# 4. Renderizar el menú lateral estándar de Sihcli-Poter
selectors.renderizar_menu_navegacion("Escenarios WEAP")

# 5. ¡ENCENDER EL MOTOR! (Llamamos a la función que creaste)
try:
    escenarios_weap.renderizar_motor_escenarios_weap()
except Exception as e:
    st.error(f"Error al cargar el simulador: {e}")
