# pages/17_🛰️_Radar_Meteorologico.py

import os
import sys
import streamlit as st
import streamlit.components.v1 as components

# --- 1. CONFIGURACIÓN DE PÁGINA (PANTALLA COMPLETA) ---
# Usamos layout="wide" para aprovechar cada pixel del monitor
st.set_page_config(page_title="Radar Meteorológico", page_icon="🛰️", layout="wide")

# --- 2. IMPORTACIÓN DE MÓDULOS DEL SISTEMA ---
try:
    from modules import selectors
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors

# --- 3. MENÚ LATERAL ---
selectors.renderizar_menu_navegacion("Radar Meteorológico")

# --- 4. INTERFAZ DEL CENTRO DE COMANDO ---
st.title("🛰️ Centro de Observación Atmosférica (Tiempo Real)")
st.markdown("Monitoreo en vivo de alta disponibilidad. Utiliza el modelo ECMWF para seguimiento de precipitación, vientos y nubosidad en múltiples altitudes.")
st.markdown("---")

# Controles de navegación en la parte superior
c_ctrl1, c_ctrl2 = st.columns([1, 4])

with c_ctrl1:
    st.info("🎛️ **Panel de Control**")
    capa_windy = st.radio(
        "Capa Atmosférica:",
        ["🌧️ Lluvia y Truenos", "💨 Viento (Altitud 3D)", "☁️ Nubes (Altitud 3D)"],
        index=0,
    )
    st.markdown("---")
    st.caption("💡 **Altitud:** El radar de lluvia siempre mide en la superficie. Para explorar las corrientes de aire desde el suelo hasta los 13.5 km (estratosfera), cambia a la capa de **Viento** o **Nubes**.")

with c_ctrl2:
    # Traductor de capas para el motor de Windy
    if capa_windy == "🌧️ Lluvia y Truenos":
        overlay = "rain"
    elif capa_windy == "💨 Viento (Altitud 3D)":
        overlay = "wind"
    else:
        overlay = "clouds"
        
    # iFrame Dinámico: Centrado en Antioquia (Lat 6.24, Lon -75.58)
    windy_iframe = f"""
    <iframe width="100%" height="700" 
        src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=%C2%B0C&metricWind=km/h&zoom=7&overlay={overlay}&product=ecmwf&level=surface&lat=6.24&lon=-75.58" 
        frameborder="0" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </iframe>
    """
    components.html(windy_iframe, height=750)
    
    st.caption("🟢 **Línea de Tiempo:** Dale al botón de 'Play' ▶️ en la esquina inferior izquierda del mapa para proyectar el movimiento del clima.")
