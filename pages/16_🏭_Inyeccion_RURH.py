# pages/08_🏭_Inyeccion_RURH.py

import sys
import os
import streamlit as st

# Configuración de la página (Siempre primero)
st.set_page_config(page_title="Inyección RURH", page_icon="🏭", layout="wide")

# Rutas para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import selectors
from modules import Inyeccion_RURH  # <-- Importamos tu nuevo módulo

# Renderizar menú de navegación
selectors.renderizar_menu_navegacion("Inyección RURH")

# Interfaz gráfica
st.title("🏭 Motor de Inyección RURH (Concesiones y Vertimientos)")
st.markdown("Este módulo descarga automáticamente los archivos Excel desde Supabase, consolida los caudales concesionados y vertimientos, y los inyecta en la base de datos para alimentar el simulador WEAP.")
st.markdown("---")

# Botón que acciona la lógica en el módulo
if st.button("🚀 Descargar, Procesar e Inyectar a Base de Datos", type="primary"):
    # Llamamos a la función principal del módulo
    Inyeccion_RURH.procesar_e_inyectar_rurh()
