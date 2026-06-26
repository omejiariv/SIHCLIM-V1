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

# 3. 🚀 INYECTAR EL SELECTOR ESPACIAL
st.sidebar.markdown("---")
nombre_zona, gdf_zona, nivel_jerarquico, es_busqueda_global = selectors.render_selector_espacial()

# 4. 🛡️ GUARDIA DE SEGURIDAD PARA WEAP
# Detectamos si 'nombre_zona' es una lista de puros números (códigos de estaciones)
es_lista_estaciones = False
if isinstance(nombre_zona, list):
    # Si al menos el primer elemento es un código numérico (ej. '27015330')
    if len(nombre_zona) > 0 and str(nombre_zona[0]).isdigit():
        es_lista_estaciones = True
elif isinstance(nombre_zona, str) and "[" in nombre_zona and "'" in nombre_zona:
    # Si viene como string de lista (ej. "['27015330', ...]")
    if any(char.isdigit() for char in nombre_zona.split("'")[1:2]):
        es_lista_estaciones = True

if es_lista_estaciones:
    st.error("🛑 **Escala Geográfica Incorrecta**")
    st.warning("El simulador WEAP requiere una unidad territorial que contenga población (como una **Cuenca Hidrográfica** o un **Municipio**).")
    st.info("👉 **Solución:** Ve al panel izquierdo, busca la opción 'Escala de Análisis' y cámbiala a 'Cuencas Hidrográficas'. Luego selecciona la cuenca que deseas analizar.")
else:
    # 5. ENCENDER EL MOTOR WEAP
    try:
        escenarios_weap.renderizar_motor_escenarios_weap(nombre_zona)
    except Exception as e:
        st.error(f"Error al cargar el simulador: {e}")
