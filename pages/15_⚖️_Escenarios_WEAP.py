import sys
import os
import streamlit as st
import pandas as pd

# 1. Configuración de página
st.set_page_config(page_title="SIHCLI | Escenarios WEAP", page_icon="⚖️", layout="wide")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import selectors, escenarios_weap

# 2. Renderizar el menú principal
selectors.renderizar_menu_navegacion("Escenarios WEAP")

# 3. 🚀 INYECTAR EL SELECTOR ESPACIAL
st.sidebar.markdown("---")
nombre_zona, gdf_zona, nivel_jerarquico, es_busqueda_global = selectors.render_selector_espacial()

# 4. 🛡️ GUARDIA DE SEGURIDAD DEFINITIVO
if nivel_jerarquico == "Estaciones":
    st.error("🛑 **Escala Geográfica Incorrecta**")
    st.warning("El simulador WEAP requiere una unidad territorial que contenga población (como una **Cuenca Hidrográfica** o un **Municipio**).")
    st.info("👉 **Solución:** Ve al panel izquierdo, cambia la 'Escala de Análisis' a 'Cuencas Hidrográficas' o 'Municipios'.")
else:
    # 🚀 RESCATE DEL NOMBRE REAL (El Bypass Cosmético)
    territorio_final = nombre_zona
    
    # Si selectors.py nos envió la lista de estaciones, buscamos el nombre de la cuenca en la memoria
    if isinstance(nombre_zona, list) and all(str(t).strip().isdigit() for t in nombre_zona if str(t).strip()):
        nombre_rescatado = None
        # Buscamos en el Aleph de Streamlit cualquier variable que tenga el formato "Nombre - (Código)"
        for key, val in st.session_state.items():
            if isinstance(val, str) and " - (" in val:
                nombre_rescatado = val
                break
        
        if nombre_rescatado:
            territorio_final = [nombre_rescatado]
        else:
            territorio_final = ["Territorio en Memoria Aleph"]

    # 5. ENCENDER EL MOTOR WEAP
    try:
        escenarios_weap.renderizar_motor_escenarios_weap(territorio_final, gdf_zona)
    except Exception as e:
        st.error(f"Error al cargar el simulador: {e}")
