# modules/utils.py

import io
import pandas as pd
import streamlit as st
import unicodedata
import os

# --- FUNCIÓN PARA CORRECCIÓN NUMÉRICA ---
@st.cache_data
def standardize_numeric_column(series):
    """
    Convierte una serie de Pandas a valores numéricos de manera robusta,
    reemplazando comas por puntos como separador decimal.
    """
    series_clean = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series_clean, errors="coerce")


def display_plotly_download_buttons(fig, file_prefix):
    """Muestra botones de descarga para un gráfico Plotly (HTML y PNG)."""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs="cdn")
        st.download_button(
            label="Descargar Gráfico (HTML)",
            data=html_buffer.getvalue(),
            file_name=f"{file_prefix}.html",
            mime="text/html",
            key=f"dl_html_{file_prefix}",
        )
    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            st.download_button(
                label="Descargar Gráfico (PNG)",
                data=img_bytes,
                file_name=f"{file_prefix}.png",
                mime="image/png",
                key=f"dl_png_{file_prefix}",
            )
        except Exception:
            st.warning(
                "No se pudo generar la imagen PNG. Asegúrate de tener 'kaleido' instalado."
            )


def add_folium_download_button(map_object, file_name):
    """Muestra un botón de descarga para un mapa de Folium (HTML)."""
    st.markdown("---")
    map_buffer = io.BytesIO()
    map_object.save(map_buffer, close_file=False)
    st.download_button(
        label="Descargar Mapa (HTML)",
        data=map_buffer.getvalue(),
        file_name=file_name,
        mime="text/html",
        key=f"dl_map_{file_name.replace('.', '_')}",

    )

# ==============================================================================
# 🧽 FUNCIONES MAESTRAS DE LIMPIEZA Y LECTURA (CENTRALIZADAS)
# ==============================================================================

def normalizar_texto(texto):
    """
    Mata-tildes y espacios: Convierte cualquier texto a minúsculas, 
    sin tildes y sin espacios en blanco a los lados. Ideal para cruzar bases de datos.
    """
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

@st.cache_data
def leer_csv_robusto(ruta):
    """
    Lector blindado: Intenta leer el archivo detectando automáticamente el separador.
    Si falla, intenta con punto y coma (;), comas (,) y diferentes codificaciones.
    """
    if not os.path.exists(ruta):
        return pd.DataFrame()
        
    try:
        # Intento 1: Detección automática (ideal para la mayoría)
        df = pd.read_csv(ruta, sep=None, engine='python')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception:
        try:
            # Intento 2: Forzar separador latinoamericano/europeo (;)
            df = pd.read_csv(ruta, sep=';', low_memory=False)
            if len(df.columns) < 3: # Si leyó todo en una sola columna, el separador era coma
                df = pd.read_csv(ruta, sep=',', low_memory=False)
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            return df
        except Exception as e:
            print(f"Error crítico leyendo {ruta}: {e}")
            return pd.DataFrame()

import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False, ttl=3600)
def descargar_matrices_produccion():
    """Descarga los cerebros pre-entrenados desde Supabase (Se guarda en caché 1 hora)"""
    try:
        url_demo = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Demografica.csv"
        url_pecu = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Pecuaria.csv"
        url_prop = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Proporciones_Veredales_Final.csv" # <-- NUEVA MATRIZ ESPACIAL
        
        df_d = pd.read_csv(url_demo)
        df_p = pd.read_csv(url_pecu)
        df_prop = pd.read_csv(url_prop) # <-- DESCARGA
        return df_d, df_p, df_prop
    except Exception as e:
        print(f"Error descargando cerebros: {e}")
        return None, None, None

def encender_gemelo_digital():
    """Verifica si la memoria está vacía y la llena instantáneamente"""
    # CRÍTICO: Ahora revisamos específicamente si falta la matriz de proporciones
    if 'df_matriz_proporciones' not in st.session_state:
        st.cache_data.clear() # Obligamos a Streamlit a borrar su caché histórico
        df_demo, df_pecu, df_prop = descargar_matrices_produccion()
        
        if df_demo is not None and not df_demo.empty:
            st.session_state['df_matriz_demografica'] = df_demo
            
        if df_pecu is not None and not df_pecu.empty:
            st.session_state['df_matriz_pecuaria'] = df_pecu
            
        if df_prop is not None and not df_prop.empty:
            st.session_state['df_matriz_proporciones'] = df_prop
            print("✅ ¡Matriz Espacial cargada en memoria exitosamente!")
            
import numpy as np
import unicodedata

def normalizar_robusto(texto):
    """Limpiador extremo de texto para evitar fallos de búsqueda"""
    if pd.isna(texto): return ""
    return unicodedata.normalize('NFKD', str(texto).lower().strip()).encode('ascii', 'ignore').decode('utf-8')

def obtener_metabolismo_exacto(nombre_seleccion, anio_destino=None):
    """
    🧠 Cerebro Central V2 (SQL-Ready): Extrae y proyecta la población humana y pecuaria
    directamente desde las Matrices Maestras en PostgreSQL.
    Incluye Traductor Inteligente de Nombres de Cuencas.
    """
    import pandas as pd
    import numpy as np
    
    # 🔌 Conexión a la base de datos
    try:
        from modules.db_manager import get_engine
        engine = get_engine()
    except Exception as e:
        print(f"Error cargando engine: {e}")
        engine = None

    res = {
        'pob_urbana': 0.0, 'pob_rural': 0.0, 'pob_total': 0.0,
        'bovinos': 0.0, 'porcinos': 0.0, 'aves': 0.0,
        'origen_humano': "Sin Datos (0)",
        'origen_pecuario': "Sin Datos (0)"
    }
    
    if engine is None: return res

    # =========================================================================
    # 🛡️ TRADUCTOR INTELIGENTE (El Puente de Nomenclatura)
    # =========================================================================
    zona_limpia = str(nombre_seleccion).strip()
    
    # Arregla abreviaciones espaciales comunes ("R. Chico" -> "Río Chico")
    if zona_limpia.startswith("R. "):
        zona_limpia = zona_limpia.replace("R. ", "Río ")
    elif zona_limpia.startswith("Q. "):
        zona_limpia = zona_limpia.replace("Q. ", "Quebrada ")
        
    zona_limpia_lower = zona_limpia.lower()

    # =========================================================================
    # ⚙️ MOTOR MATEMÁTICO UNIVERSAL (Proyecta años futuros)
    # =========================================================================
    def proyectar(fila, col_base, col_anio_base):
        if anio_destino is None: return float(fila[col_base])
        x_norm = anio_destino - fila[col_anio_base]
        try:
            mod = fila.get('Modelo_Recomendado', 'Polinomial_3')
            if mod == 'Logístico': 
                return max(0.0, fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * x_norm)))
            elif mod == 'Exponencial': 
                return max(0.0, fila['Exp_a'] * np.exp(fila['Exp_b'] * x_norm))
            else: 
                return max(0.0, fila['Poly_A']*(x_norm**3) + fila['Poly_B']*(x_norm**2) + fila['Poly_C']*x_norm + fila['Poly_D'])
        except: 
            return float(fila[col_base])

    # =========================================================================
    # 1. MOTOR DEMOGRÁFICO (Lectura SQL)
    # =========================================================================
    try:
        # Búsqueda exacta ignorando mayúsculas
        q_demo = "SELECT * FROM matriz_maestra_demografica WHERE LOWER(TRIM(\"Territorio\")) = %(z)s"
        df_demo = pd.read_sql(q_demo, engine, params={"z": zona_limpia_lower})
        
        # Búsqueda flexible (Si "Río Chico" está guardado como "Cuenca Río Chico")
        if df_demo.empty:
            q_demo_like = "SELECT * FROM matriz_maestra_demografica WHERE LOWER(\"Territorio\") LIKE %(z)s"
            df_demo = pd.read_sql(q_demo_like, engine, params={"z": f"%{zona_limpia_lower}%"})

        if not df_demo.empty:
            fu = df_demo[df_demo['Area'] == 'Urbana']
            fr = df_demo[df_demo['Area'] == 'Rural']
            
            pob_u = proyectar(fu.iloc[0], 'Pob_Base', 'Año_Base') if not fu.empty else 0.0
            pob_r = proyectar(fr.iloc[0], 'Pob_Base', 'Año_Base') if not fr.empty else 0.0
            
            res['pob_urbana'] = pob_u
            res['pob_rural'] = pob_r
            res['pob_total'] = pob_u + pob_r
            res['origen_humano'] = "Matriz Maestra SQL"
                
    except Exception as e:
        print(f"Error SQL Demográfico: {e}")

    # =========================================================================
    # 2. MOTOR PECUARIO (Lectura SQL)
    # =========================================================================
    try:
        q_pecu = "SELECT * FROM matriz_maestra_pecuaria WHERE LOWER(TRIM(\"Territorio\")) = %(z)s"
        df_pecu = pd.read_sql(q_pecu, engine, params={"z": zona_limpia_lower})
        
        if df_pecu.empty:
            q_pecu_like = "SELECT * FROM matriz_maestra_pecuaria WHERE LOWER(\"Territorio\") LIKE %(z)s"
            df_pecu = pd.read_sql(q_pecu_like, engine, params={"z": f"%{zona_limpia_lower}%"})

        if not df_pecu.empty:
            f_bov = df_pecu[df_pecu['Especie'] == 'Bovinos']
            f_por = df_pecu[df_pecu['Especie'] == 'Porcinos']
            f_ave = df_pecu[df_pecu['Especie'] == 'Aves']
            
            if not f_bov.empty: res['bovinos'] = proyectar(f_bov.iloc[0], 'Poblacion_Base', 'Año_Base')
            if not f_por.empty: res['porcinos'] = proyectar(f_por.iloc[0], 'Poblacion_Base', 'Año_Base')
            if not f_ave.empty: res['aves'] = proyectar(f_ave.iloc[0], 'Poblacion_Base', 'Año_Base')
            
            res['origen_pecuario'] = "Matriz Pecuaria SQL"
            
    except Exception as e:
        print(f"Error SQL Pecuario: {e}")

    # Se eliminaron los fallbacks de números inventados para mantener la pureza de los datos.
    # Si todo dio cero, la plataforma será honesta y pedirá actualizar la matriz.
    return res
