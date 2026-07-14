# pages/17_🛰️_Radar_Meteorologico.py

import os
import sys
import streamlit as st
import streamlit.components.v1 as components

# --- 1. CONFIGURACIÓN DE PÁGINA (PANTALLA COMPLETA) ---
st.set_page_config(page_title="Radar Meteorológico", page_icon="🛰️", layout="wide")

# --- 2. IMPORTACIÓN DE MÓDULOS DEL SISTEMA ---
try:
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.utils import inicializar_torrente_sanguineo

# --- 3. INICIALIZACIÓN Y MENÚ LATERAL ---
try:
    inicializar_torrente_sanguineo()
except:
    pass

selectors.renderizar_menu_navegacion("Radar Meteorológico")

# 🚀 FIX: Traer de vuelta los selectores geográficos al panel lateral
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

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
        
    # 🚀 FIX MAGNÍFICO: Hacer que el radar se centre dinámicamente en el territorio seleccionado
    lat_center = 6.24
    lon_center = -75.58
    zoom_level = 7
    
    # Si el usuario seleccionó un territorio, sacamos su centroide matemático
    if gdf_zona is not None and not gdf_zona.empty:
        try:
            # Forzamos coordenadas GPS estándar por seguridad
            if gdf_zona.crs.to_string() != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
            centroide = gdf_zona.geometry.unary_union.centroid
            lat_center = centroide.y
            lon_center = centroide.x
            zoom_level = 10 # Hacemos zoom más cerca si hay una zona específica
            st.success(f"📍 Radar alineado y centrado automáticamente en: **{nombre_zona}**")
        except Exception as e:
            pass
        
    # iFrame Dinámico con wrapper HTML para Pantalla Completa
    windy_html = f"""
    <div id="map-container" style="position: relative; width: 100%; height: 700px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; background: #e5e9ec;">
        <iframe id="windy-iframe" width="100%" height="100%" 
            src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=%C2%B0C&metricWind=km/h&zoom={zoom_level}&overlay={overlay}&product=ecmwf&level=surface&lat={lat_center}&lon={lon_center}" 
            frameborder="0" allowfullscreen>
        </iframe>
        
        <!-- Botón de Fullscreen inyectado sobre el iframe -->
        <button onclick="toggleFullScreen()" 
                style="position: absolute; top: 15px; right: 15px; z-index: 999; padding: 10px 15px; background: rgba(255,255,255,0.9); border: 2px solid #ccc; border-radius: 6px; cursor: pointer; font-weight: bold; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            🔲 Pantalla Completa
        </button>
    </div>

    <script>
    function toggleFullScreen() {{
        var elem = document.getElementById("map-container");
        if (!document.fullscreenElement) {{
            elem.requestFullscreen().catch(err => {{
                alert(`Error al intentar pantalla completa: ${{err.message}}`);
            }});
        }} else {{
            document.exitFullscreen();
        }}
    }}
    </script>
    """
    
    # Renderizamos el bloque HTML
    components.html(windy_html, height=750)
    
    st.caption("🟢 **Línea de Tiempo:** Dale al botón de 'Play' ▶️ en la esquina inferior izquierda del mapa para proyectar el movimiento del clima.")
