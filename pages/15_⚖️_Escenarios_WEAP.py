import sys
import os
import streamlit as st
import pandas as pd
import requests
import io

# Ajuste de rutas para asegurar que los módulos sean siempre visibles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORTACIONES CENTRALES (Aquí radica la solución)
from modules.db_manager import get_engine
from sqlalchemy import text
from modules import selectors, escenarios_weap
from modules.utils import normalizar_texto

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
        import pandas as pd
        import requests
        import io
        from sqlalchemy import text
        from modules.db_manager import get_engine
        
        territorio_str = territorio_final[0] if isinstance(territorio_final, list) else territorio_final
        
        # --- 1. CONEXIÓN DEMOGRÁFICA AUTÓNOMA (VÍA SUPABASE) ---
        url_csv = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Multimodelo_Demografica.csv"
        res_csv = requests.get(url_csv)
        
        # Decodificamos en utf-8 para manejar correctamente tildes y caracteres latinos
        df_dem = pd.read_csv(io.StringIO(res_csv.content.decode('utf-8')))
        
        # 🔥 EL BISTURÍ DE NOMBRES
        # Extraemos "Q. La Honda" de "Q. La Honda - (2308-01-04-24)"
        if " - (" in territorio_str:
            nombre_puro = territorio_str.split(" - (")[0].strip()
        else:
            nombre_puro = territorio_str.strip()
        
        # Filtramos por el nombre exacto Y garantizamos que traiga el Área 'Total'
        # Convertimos todo a minúsculas temporalmente para evitar fallos por mayúsculas
        mascara = (df_dem['Territorio'].astype(str).str.strip().str.lower() == nombre_puro.lower()) & \
                  (df_dem['Area'].astype(str).str.strip().str.lower() == 'total')
        
        row_pob = df_dem[mascara]
        
        if not row_pob.empty:
            # Tomamos la columna Pob_Base (indistintamente de cómo esté escrita)
            col_pob = next((c for c in row_pob.columns if 'pob_base' in c.lower()), 'Pob_Base')
            st.session_state['aleph_pob_total'] = float(row_pob.iloc[0][col_pob])
        else:
            st.sidebar.warning(f"⚠️ Demografía no hallada en el CSV para el nombre puro: {nombre_puro}")

        # --- 2. CONEXIÓN HIDROLÓGICA Y RURH (VÍA POSTGRESQL) ---
        engine = get_engine()
        with engine.connect() as conn:
            # Oferta Hídrica
            query_hidro = text('SELECT "Caudal_Medio_m3s" FROM matriz_hidrologica_maestra WHERE "Territorio" = :t LIMIT 1')
            oferta_real = conn.execute(query_hidro, {"t": territorio_str}).scalar()
            if oferta_real is not None:
                st.session_state['aleph_oferta_m3s'] = float(oferta_real)
                
            # Presiones RURH
            query_rurh = text('SELECT COALESCE(SUM("Presion_Total_RURH_m3s"), 0) FROM matriz_presiones_rurh WHERE "Territorio" = :t')
            rurh_real = conn.execute(query_rurh, {"t": territorio_str}).scalar()
            st.session_state['aleph_concesiones_m3s'] = float(rurh_real)
        
        # --- 3. RENDERIZADO FINAL ---
        escenarios_weap.renderizar_motor_escenarios_weap(territorio_final, gdf_zona)
        
    except Exception as e:
        st.error(f"Error crítico en sincronización Aleph (Demografía/RURH): {e}")
