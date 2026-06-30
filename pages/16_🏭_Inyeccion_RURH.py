import sys
import os
import streamlit as st

# FUERZA BRUTA: Añadimos la raíz del proyecto al PATH explícitamente
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from modules import selectors
from modules.Inyeccion_RURH import renderizar_inyeccion_rurh

# 1. Configuración de página
st.set_page_config(page_title="SIHCLI | Inyección RURH", page_icon="🏭", layout="wide")

# 2. Renderizar el menú principal
selectors.renderizar_menu_navegacion("Inyección RURH")

# 3. Encabezado de la página
st.title("Inyección Maestra RURH a PostgreSQL")
st.markdown("---")

# 4. Renderizar el módulo de inyección
renderizar_inyeccion_rurh()
