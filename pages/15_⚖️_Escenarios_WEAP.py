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
# Usamos la jerarquía directa del sistema. Si es "Estaciones", bloqueamos.
if nivel_jerarquico == "Estaciones":
    st.error("🛑 **Escala Geográfica Incorrecta**")
    st.warning("El simulador WEAP requiere una unidad territorial que contenga población (como una **Cuenca Hidrográfica** o un **Municipio**).")
    st.info("👉 **Solución:** Ve al panel izquierdo, cambia la 'Escala de Análisis' a 'Cuencas Hidrográficas' o 'Municipios'.")
else:
    # --- SONDA FORENSE ---
    st.info(f"🕵️‍♂️ DETECTIVE 1 (Lo que envía el selector): {nombre_zona}")
    if isinstance(gdf_zona, pd.DataFrame):
        st.info(f"🕵️‍♂️ DETECTIVE 2 (Columnas del mapa): {gdf_zona.columns.tolist()}")
    else:
        st.info(f"🕵️‍♂️ DETECTIVE 2: No llegó ningún mapa (gdf_zona es {type(gdf_zona)})")
    # ---------------------

    # 5. ENCENDER EL MOTOR WEAP
    try:
        # 🚀 TRUCO MAESTRO: Le enviamos el gdf_zona al motor para que pueda extraer los nombres reales
        escenarios_weap.renderizar_motor_escenarios_weap(nombre_zona, gdf_zona)
    except Exception as e:
        st.error(f"Error al cargar el simulador: {e}")
