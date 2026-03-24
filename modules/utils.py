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
    if 'df_matriz_demografica' not in st.session_state:
        df_demo, df_pecu, df_prop = descargar_matrices_produccion()
        
        if df_demo is not None and not df_demo.empty:
            st.session_state['df_matriz_demografica'] = df_demo
            
        if df_pecu is not None and not df_pecu.empty:
            st.session_state['df_matriz_pecuaria'] = df_pecu
            
        if df_prop is not None and not df_prop.empty:
            st.session_state['df_matriz_proporciones'] = df_prop # <-- GUARDADO EN MEMORIA
            
import numpy as np
import unicodedata

def normalizar_robusto(texto):
    """Limpiador extremo de texto para evitar fallos de búsqueda"""
    if pd.isna(texto): return ""
    return unicodedata.normalize('NFKD', str(texto).lower().strip()).encode('ascii', 'ignore').decode('utf-8')

def obtener_metabolismo_exacto(nombre_seleccion, anio_destino=None):
    """
    Cerebro Central: Extrae y proyecta la población humana y pecuaria desde el Gemelo Digital.
    """
    nombre_sel_limpio = normalizar_robusto(nombre_seleccion)
    es_cuenca = any(x in nombre_sel_limpio for x in ['rio', 'quebrada', 'alto', 'medio', 'bajo', 'embalse', 'fe', 'q.'])
    
    # 0. Diccionario base
    res = {
        'pob_urbana': 0.0, 'pob_rural': 0.0, 'pob_total': 0.0,
        'bovinos': 0.0, 'porcinos': 0.0, 'aves': 0.0,
        'origen_humano': "Estimación (Fallback)",
        'origen_pecuario': "Estimación (Fallback)"
    }

    # =========================================================================
    # 1. MOTOR HUMANO (Alta Precisión Topológica)
    # =========================================================================
    if 'df_matriz_demografica' in st.session_state:
        df_demo = st.session_state['df_matriz_demografica'].copy()
        
        # Blindaje 1: Normalizamos la base DANE
        if 'Terr_Norm' not in df_demo.columns:
            df_demo['Terr_Norm'] = df_demo['Territorio'].apply(normalizar_robusto)

        def proyectar_pob(fila):
            if anio_destino is None: return float(fila['Pob_Base'])
            x_norm = anio_destino - fila['Año_Base']
            try:
                mod = fila.get('Modelo_Recomendado', 'Polinomial')
                if mod == 'Logístico': return max(0.0, fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * x_norm)))
                elif mod == 'Exponencial': return max(0.0, fila['Exp_a'] * np.exp(fila['Exp_b'] * x_norm))
                else: return max(0.0, fila['Poly_A']*(x_norm**3) + fila['Poly_B']*(x_norm**2) + fila['Poly_C']*x_norm + fila['Poly_D'])
            except: return float(fila['Pob_Base'])

        pob_u, pob_r = 0.0, 0.0

        # --- RUTA A: ES UNA CUENCA (Usa Matriz de Proporciones) ---
        if es_cuenca and 'df_matriz_proporciones' in st.session_state:
            df_prop = st.session_state['df_matriz_proporciones'].copy()
            
            # Blindaje 2: Normalizamos el mapa GIS
            if 'SUBC_Norm' not in df_prop.columns:
                df_prop['SUBC_Norm'] = df_prop['SUBC_LBL'].apply(normalizar_robusto)
                df_prop['CUENCA_Norm'] = df_prop['N_NSS1'].apply(normalizar_robusto)
            
            # Búsqueda exacta primero
            frags = df_prop[(df_prop['SUBC_Norm'] == nombre_sel_limpio) | (df_prop['CUENCA_Norm'] == nombre_sel_limpio)]
            
            # Si falla, búsqueda flexible (Radar)
            if frags.empty:
                pal_clave = nombre_sel_limpio.replace("rio", "").replace("alto", "").replace("q.", "").replace("embalse", "").strip()
                if pal_clave:
                    mask = df_prop['SUBC_Norm'].str.contains(pal_clave, na=False) | df_prop['CUENCA_Norm'].str.contains(pal_clave, na=False)
                    frags = df_prop[mask]

            if not frags.empty:
                for _, frag in frags.iterrows():
                    mpio_limpio = normalizar_robusto(frag['NOMB_MPIO'])
                    ver_limpia = normalizar_robusto(frag['NOMBRE_VER'])
                    pct = min(float(frag['Pct_en_Cuenca']), 1.0) # Vacuna contra el >100%
                    
                    es_urbano = ("cabecera" in ver_limpia) or (mpio_limpio in ver_limpia)
                    tipo = 'Urbana' if es_urbano else 'Rural'
                    
                    # El cruce perfecto (Todo en minúsculas sin tildes)
                    filtro = df_demo[(df_demo['Terr_Norm'] == mpio_limpio) & (df_demo['Area'] == tipo)]
                    if not filtro.empty:
                        aportada = proyectar_pob(filtro.iloc[0]) * pct
                        if es_urbano: pob_u += aportada
                        else: pob_r += aportada
                
                res['origen_humano'] = "Matriz Espacial (Alta Precisión)"

        # --- RUTA B: ES UN MUNICIPIO (Búsqueda Directa DANE) ---
        else:
            fu = df_demo[(df_demo['Terr_Norm'] == nombre_sel_limpio) & (df_demo['Area'] == 'Urbana')]
            fr = df_demo[(df_demo['Terr_Norm'] == nombre_sel_limpio) & (df_demo['Area'] == 'Rural')]
            if not fu.empty: pob_u = proyectar_pob(fu.iloc[0])
            if not fr.empty: pob_r = proyectar_pob(fr.iloc[0])
            res['origen_humano'] = "Matriz Maestra (DANE)"
            
        res['pob_urbana'], res['pob_rural'], res['pob_total'] = pob_u, pob_r, pob_u + pob_r

    # =========================================================================
    # 2. MOTOR PECUARIO (Radar Inteligente)
    # =========================================================================
    if 'df_matriz_pecuaria' in st.session_state:
        df_pec = st.session_state['df_matriz_pecuaria'].copy()
        if 'Terr_Norm' not in df_pec.columns:
            df_pec['Terr_Norm'] = df_pec['Territorio'].apply(normalizar_robusto)
        
        f_b = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Bovinos')]
        f_p = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Porcinos')]
        f_a = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Aves')]
        
        if f_b.empty:
            pal_clave = nombre_sel_limpio.replace("rio", "").replace("alto", "").replace("embalse", "").strip()
            if pal_clave:
                mask = df_pec['Terr_Norm'].str.contains(pal_clave, na=False)
                f_b = df_pec[mask & (df_pec['Especie'] == 'Bovinos')]
                f_p = df_pec[mask & (df_pec['Especie'] == 'Porcinos')]
                f_a = df_pec[mask & (df_pec['Especie'] == 'Aves')]

        def calc_pec(filtro):
            if filtro.empty: return 0.0
            fila = filtro.iloc[0]
            if anio_destino is None: return float(fila['Poblacion_Base'])
            x_norm = anio_destino - fila['Año_Base']
            try:
                mod = fila.get('Modelo_Recomendado', 'Polinomial')
                if mod == 'Logístico': return max(0.0, fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * x_norm)))
                elif mod == 'Exponencial': return max(0.0, fila['Exp_a'] * np.exp(fila['Exp_b'] * x_norm))
                else: return max(0.0, fila['Poly_A']*(x_norm**3) + fila['Poly_B']*(x_norm**2) + fila['Poly_C']*x_norm + fila['Poly_D'])
            except: return float(fila['Poblacion_Base'])

        if not f_b.empty: res['bovinos'] = calc_pec(f_b)
        if not f_p.empty: res['porcinos'] = calc_pec(f_p)
        if not f_a.empty: res['aves'] = calc_pec(f_a)
        if res['bovinos'] > 0: res['origen_pecuario'] = "Matriz Maestra (Sincronizada)"

    # =========================================================================
    # 3. FALLBACKS DE SEGURIDAD EXTREMA
    # =========================================================================
    if res['pob_total'] <= 0:
        res['pob_total'], res['pob_urbana'], res['pob_rural'] = 5000.0, 3500.0, 1500.0
    if res['bovinos'] <= 0: res['bovinos'] = res['pob_total'] * 1.5
    if res['porcinos'] <= 0: res['porcinos'] = res['pob_total'] * 0.8
    if res['aves'] <= 0: res['aves'] = res['pob_total'] * 10.0

    return res
