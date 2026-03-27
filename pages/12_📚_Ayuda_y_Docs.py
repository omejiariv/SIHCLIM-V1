# 12_📚_Ayuda_y_Docs.py

import streamlit as st
import os

st.set_page_config(page_title="Ayuda y Documentación", page_icon="📚", layout="wide")

st.title("📚 Centro de Documentación y Ayuda")
st.markdown("---")

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "Agua2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "Agua2026"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo
muro_de_acceso_beta()
# ==============================================================================

tab1, tab2 = st.tabs(["📘 Documentación Técnica", "📖 Manual de Usuario"])

with tab1:
    st.header("Documentación Técnica del Sistema")
    st.info("Este documento detalla la arquitectura, tecnologías y estructura de datos de SIHCLI-POTER.")
    
    try:
        # Busca el archivo en la raíz
        with open("DOCUMENTACION_TECNICA.md", "r", encoding="utf-8") as f:
            doc_content = f.read()
            
        with st.expander("👁️ Ver Documentación en Pantalla", expanded=True):
            st.markdown(doc_content)
            
        st.download_button(
            label="📥 Descargar Documentación Técnica (.md)",
            data=doc_content,
            file_name="SIHCLI_POTER_Docs_Tecnica.md",
            mime="text/markdown"
        )
    except FileNotFoundError:
        st.warning("⚠️ El archivo 'DOCUMENTACION_TECNICA.md' no se encontró en la carpeta raíz.")

with tab2:
    st.header("Guía de Usuario")
    st.write("Bienvenido al manual de operación de SIHCLI-POTER.")
    st.info("🚧 El Manual PDF detallado está en construcción.")
    
    st.subheader("Preguntas Frecuentes")
    with st.expander("¿Cómo subir nuevos datos de lluvia?"):
        st.write("""
        1. Ve al menú lateral: **'👑 Panel Administracion'**.
        2. Ingresa tus credenciales seguras.
        3. Pestaña **'Estaciones & Lluvias'** -> **'Carga Masiva (CSV)'**.
        4. Sube el archivo y sigue el asistente inteligente.
        """)

