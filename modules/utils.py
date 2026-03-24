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
    Cerebro Central: Extrae y proyecta la población humana y pecuaria desde el Gemelo Digital.
    Corrige la sobreestimación distribuyendo la población por fracción de área total.
    """
    nombre_sel_limpio = normalizar_robusto(nombre_seleccion)
    
    res = {
        'pob_urbana': 0.0, 'pob_rural': 0.0, 'pob_total': 0.0,
        'bovinos': 0.0, 'porcinos': 0.0, 'aves': 0.0,
        'origen_humano': "Estimación (Fallback)",
        'origen_pecuario': "Estimación (Fallback)"
    }

    # =========================================================================
    # 1. MOTOR HUMANO (Distribución Espacial Perfecta)
    # =========================================================================
    if 'df_matriz_demografica' in st.session_state:
        df_demo = st.session_state['df_matriz_demografica'].copy()
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

        # Inteligencia de Enrutamiento: ¿Está en el DANE? Si sí, es un municipio. Si no, es cuenca.
        es_municipio = not df_demo[df_demo['Terr_Norm'] == nombre_sel_limpio].empty

        if es_municipio:
            # RUTA A: MUNICIPIO DIRECTO
            fu = df_demo[(df_demo['Terr_Norm'] == nombre_sel_limpio) & (df_demo['Area'] == 'Urbana')]
            fr = df_demo[(df_demo['Terr_Norm'] == nombre_sel_limpio) & (df_demo['Area'] == 'Rural')]
            if not fu.empty: pob_u = proyectar_pob(fu.iloc[0])
            if not fr.empty: pob_r = proyectar_pob(fr.iloc[0])
            res['origen_humano'] = "Matriz Maestra (DANE)"
            
        elif 'df_matriz_proporciones' in st.session_state:
            # RUTA B: ES UNA CUENCA O SUBCUENCA
            df_prop = st.session_state['df_matriz_proporciones']
            
            # 1. Preparación y cálculo de Áreas Totales (Ejecutado 1 sola vez en memoria)
            if 'C1_Norm' not in df_prop.columns:  # <-- EL CAMBIO ESTÁ AQUÍ
                df_prop['MPIO_Norm'] = df_prop['NOMB_MPIO'].astype(str).apply(normalizar_robusto)
                df_prop['VER_Norm'] = df_prop['NOMBRE_VER'].astype(str).apply(normalizar_robusto)
                df_prop['SUBC_Norm'] = df_prop['SUBC_LBL'].astype(str).apply(normalizar_robusto)
                df_prop['C1_Norm'] = df_prop['N_NSS1'].astype(str).apply(normalizar_robusto)
                
                df_prop['Tipo_Area'] = 'Rural'
                mask_urb = df_prop['VER_Norm'].str.contains('cabecera') | (df_prop['VER_Norm'] == df_prop['MPIO_Norm'])
                df_prop.loc[mask_urb, 'Tipo_Area'] = 'Urbana'
                
                # Para evitar duplicados, sumamos el área original de las veredas únicas de cada municipio
                df_unicos = df_prop.drop_duplicates(subset=['MPIO_Norm', 'VER_Norm'])
                st.session_state['areas_totales_mpio'] = df_unicos.groupby(['MPIO_Norm', 'Tipo_Area'])['Area_Original_km2'].sum().to_dict()
                st.session_state['df_matriz_proporciones'] = df_prop
                df_prop['VER_Norm'] = df_prop['NOMBRE_VER'].astype(str).apply(normalizar_robusto)
                df_prop['SUBC_Norm'] = df_prop['SUBC_LBL'].astype(str).apply(normalizar_robusto)
                df_prop['C1_Norm'] = df_prop['N_NSS1'].astype(str).apply(normalizar_robusto)
                
                df_prop['Tipo_Area'] = 'Rural'
                mask_urb = df_prop['VER_Norm'].str.contains('cabecera') | (df_prop['VER_Norm'] == df_prop['MPIO_Norm'])
                df_prop.loc[mask_urb, 'Tipo_Area'] = 'Urbana'
                
                # Para evitar duplicados, sumamos el área original de las veredas únicas de cada municipio
                df_unicos = df_prop.drop_duplicates(subset=['MPIO_Norm', 'VER_Norm'])
                st.session_state['areas_totales_mpio'] = df_unicos.groupby(['MPIO_Norm', 'Tipo_Area'])['Area_Original_km2'].sum().to_dict()
                st.session_state['df_matriz_proporciones'] = df_prop

            # 2. Búsqueda Implacable del Polígono
            frags = df_prop[(df_prop['SUBC_Norm'] == nombre_sel_limpio) | (df_prop['C1_Norm'] == nombre_sel_limpio)]
            if frags.empty:
                mask = df_prop['SUBC_Norm'].str.contains(nombre_sel_limpio, na=False) | df_prop['C1_Norm'].str.contains(nombre_sel_limpio, na=False)
                frags = df_prop[mask]
            if frags.empty:
                pal_clave = nombre_sel_limpio.replace("rio ", "").replace("q. ", "").replace("quebrada ", "").strip()
                if len(pal_clave) > 3:
                    mask = df_prop['SUBC_Norm'].str.contains(pal_clave, na=False) | df_prop['C1_Norm'].str.contains(pal_clave, na=False)
                    frags = df_prop[mask]

            # 3. Matemática de Distribución Proporcional
            if not frags.empty:
                areas_totales = st.session_state['areas_totales_mpio']
                
                # Agrupamos cuánta área aportó cada municipio DENTRO de la cuenca
                areas_cuenca = frags.groupby(['MPIO_Norm', 'Tipo_Area'])['Area_Fragmento_km2'].sum().reset_index()
                
                for _, row in areas_cuenca.iterrows():
                    mpio = row['MPIO_Norm']
                    tipo = row['Tipo_Area']
                    area_dentro = row['Area_Fragmento_km2']
                    
                    # Obtenemos el Área Total del municipio
                    area_total_mpio = areas_totales.get((mpio, tipo), 0.001) # Evita división por cero
                    
                    # Calculamos qué % del municipio se "comió" la cuenca
                    fraccion = min(area_dentro / area_total_mpio, 1.0)
                    
                    filtro = df_demo[(df_demo['Terr_Norm'] == mpio) & (df_demo['Area'] == tipo)]
                    if not filtro.empty:
                        aportada = proyectar_pob(filtro.iloc[0]) * fraccion
                        if tipo == 'Urbana': pob_u += aportada
                        else: pob_r += aportada
                        
                res['origen_humano'] = "Matriz Espacial (Fracción de Área)"
            else:
                res['origen_humano'] = "Polígono no encontrado en GIS"

        res['pob_urbana'], res['pob_rural'], res['pob_total'] = pob_u, pob_r, pob_u + pob_r

    # =========================================================================
    # 2. MOTOR PECUARIO (Se mantiene estable)
    # =========================================================================
    if 'df_matriz_pecuaria' in st.session_state:
        df_pec = st.session_state['df_matriz_pecuaria'].copy()
        if 'Terr_Norm' not in df_pec.columns: df_pec['Terr_Norm'] = df_pec['Territorio'].apply(normalizar_robusto)
        
        f_b = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Bovinos')]
        f_p = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Porcinos')]
        f_a = df_pec[(df_pec['Terr_Norm'] == nombre_sel_limpio) & (df_pec['Especie'] == 'Aves')]
        
        if f_b.empty:
            pal_clave = nombre_sel_limpio.replace("rio ", "").replace("q. ", "").replace("quebrada ", "").strip()
            if len(pal_clave) > 3:
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
    # 3. FALLBACKS DE SEGURIDAD
    # =========================================================================
    if res['pob_total'] <= 0: res['pob_total'], res['pob_urbana'], res['pob_rural'] = 5000.0, 3500.0, 1500.0
    if res['bovinos'] <= 0: res['bovinos'] = res['pob_total'] * 1.5
    if res['porcinos'] <= 0: res['porcinos'] = res['pob_total'] * 0.8
    if res['aves'] <= 0: res['aves'] = res['pob_total'] * 10.0

    return res
