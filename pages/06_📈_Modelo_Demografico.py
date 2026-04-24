# pages/06_📈_Modelo_Demografico.py

import os
import sys
import time
import json
import unicodedata
import re
import warnings

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Modelo Demográfico Integral", page_icon="📈", layout="wide")
warnings.filterwarnings('ignore')

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.utils import encender_gemelo_digital, normalizar_texto
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.utils import encender_gemelo_digital, normalizar_texto

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Modelo Demográfico")

# Encendemos el sistema inmunológico
encender_gemelo_digital()

st.title("📈 Modelo Demográfico Integral (Proyección y Dasimetría)")
st.markdown("Ajuste matemático, simulación animada, mapas espaciales y proyección top-down de estructuras poblacionales (1952-2100).")

# --- FUNCION MÁGICA 1: EL ASPIRADOR DE TEXTOS (Match infalible) ---
def normalizar_texto(texto):
    if pd.isna(texto): return ""
    t = str(texto).upper()
    t = re.sub(r'\(.*?\)', '', t)
    
    # --- DESTRUCTOR DE PREFIJOS VEREDALES ---
    t = t.replace("VDA.", "").replace("VDA ", "").replace("VEREDA ", "").replace("SECTOR ", "").replace("CGE.", "")
    
    # 1. HOTFIX: Reparador de Caracteres Mutantes del DANE (UTF-8 a Latin-1)
    reparador = {
        "Ã\x81": "A", "Ã": "A", "Ã\x89": "E", "Ã‰": "E",
        "Ã\x8d": "I", "Ã": "I", "Ã\x93": "O", "Ã“": "O",
        "Ã\x9a": "U", "Ãš": "U", "Ã\x91": "N", "Ã‘": "N",
        "Ãœ": "U", "Ã\x9c": "U", "Ã¡": "A", "Ã©": "E", 
        "Ã­": "I", "Ã³": "O", "Ãº": "U", "Ã±": "N", "Ã¼": "U"
    }
    for malo, bueno in reparador.items():
        t = t.replace(malo, bueno)
        
    # 2. Quitar tildes y dejar solo letras/números
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = re.sub(r'[^A-Z0-9]', '', t)
    
    # 3. Traductor DANE-IGAC Actualizado (Con los 9 nombres pomposos y los rebeldes de las CARs)
    diccionario_rebeldes = {
        "BOGOTADC": "BOGOTA", "SANJOSEDECUCUTA": "CUCUTA", "LAGUAJIRA": "GUAJIRA", 
        "VALLE": "VALLEDELCAUCA", "VILLADESANDIEGODEUBATE": "UBATE", "SANTIAGODETOLU": "TOLU",
        "PUEBLORRICO": "PUEBLORICO", "SANANDRESDECUERQUIA": "SANANDRES",
        "SANPEDRODELOSMILAGROS": "SANPEDRO", "BRICENO": "BRICEN0",
        "PIZARRO": "BAJOBAUDO", "DOCORDO": "ELLITORALDELSANJUAN", "LITORALDELSANJUAN": "ELLITORALDELSANJUAN",
        "BAHIASOLANO": "BAHIASOLANOMUTIS", "TUMACO": "SANANDRESDETUMACO", "PATIA": "PATIAELBORDO",
        "LOPEZDEMICAY": "LOPEZ", "MAGUI": "MAGUIPAYAN", "ROBERTOPAYAN": "ROBERTOPAYANSANJOSE",
        "MALLAMA": "MALLAMAPIEDRAANCHA", "CUASPUD": "CUASPUDCARLOSAMA", "ALTOBAUDO": "ALTOBAUDOPIEDEPATO",
        "OLAYAHERRERA": "OLAYAHERRERABOCASDESATINGA", "SANTACRUZ": "SANTACRUZGUACHAVEZ",
        "LOSANDES": "LOSANDESSOTOMAYOR", "FRANCISCOPIZARRO": "FRANCISCOPIZARROSALAHONDA",
        "MEDIOSANJUAN": "ELLITORALDELSANJUANDOCORDO", "ELCANTONDELSANPABLO": "ELCANTONDESANPABLOMANAGRU",
        "ATRATO": "ATRATOYUTO", "LEGUIZAMO": "PUERTOLEGUIZAMO", "BARRANCOMINAS": "BARRANCOMINA",
        "MAPIRIPANA": "PANAPANA", "MORICHAL": "MORICHALNUEVO", "SANANDRESSOTAVENTO": "SANANDRESDESOTAVENTO",
        "SANLUISDESINCE": "SINCE", "SANVICENTEDECHUCURI": "SANVICENTEDELCHUCURI", "ELCARMENDECHUCURI": "ELCARMEN",
        "ARIGUANI": "ARIGUANIELDIFICIL", "SANMIGUEL": "SANMIGUELLADORADA", "VILLADELEYVA": "VILLADELEIVA",
        "PURACE": "PURACECOCONUCO", "ELTABLONDEGOMEZ": "ELTABLON", "ARMERO": "ARMEROGUAYABAL",
        "COLON": "COLONGENOVA", "SANPEDRODECARTAGO": "SANPEDRODECARTAGOCARTAGO", "CERROSANANTONIO": "CERRODESANANTONIO",
        "ARBOLEDA": "ARBOLEDABERRUECOS", "ENCINO": "ELENCINO", "MACARAVITA": "MARACAVITA",
        "TUNUNGUA": "TUNUNGA", "LAMONTANITA": "MONTANITA", "ELPAUJIL": "PAUJIL", "VILLARICA": "VILLARRICA",
        "GUADALAJARADEBUGA": "BUGA",
        # --- NOMBRES POMPOSOS ---
        "CARTAGENA": "CARTAGENADEINDIAS",
        "PIENDAMO": "PIENDAMOTUNIA",
        "MARIQUITA": "SANSEBASTIANDEMARIQUITA",
        "TOLUVIEJO": "SANJOSEDETOLUVIEJO",
        "SOTARA": "SOTARAPAISPAMBA",
        "PURISIMA": "PURISIMADELACONCEPCION",
        "GUICAN": "GUICANDELASIERRA",
        "PAPUNAUACD": "PAPUNAHUA",
        "PAPUNAUA": "PAPUNAHUA",
        "CHIBOLO": "CHIVOLO", 
        "MANAUREBALCONDELCESAR": "MANAURE",
        # --- 🔥 FIX: HOMOLOGACIÓN CARs (CORNAR Y CORANTIOQUIA) ---
        "ELPENOL": "PENOL", # O si el DANE dice PENOL y el mapa dice ELPENOL, usa "PENOL": "ELPENOL"
        "PENOL": "ELPENOL", # (Cubrimos ambas direcciones por seguridad)
        "ELRETIRO": "RETIRO", 
        "RETIRO": "ELRETIRO",
        "ELSANTUARIO": "SANTUARIO",
        "SANTUARIO": "ELSANTUARIO",
        "ELCARMENDEVIBORAL": "CARMENDEVIBORAL",
        "CARMENDEVIBORAL": "ELCARMENDEVIBORAL",
        "SANVICENTEFERRER": "SANVICENTE", 
        "SANVICENTE": "SANVICENTEFERRER", # Cubrimos ambas direcciones
        "LACEJA": "LACEJADELTAMBO",
        "LACEJADELTAMBO": "LACEJA",
        "CAROLINA": "CAROLINADELPRINCIPE",
        "CAROLINADELPRINCIPE": "CAROLINA",
        "SANTAFEDEANTIOQUIA": "SANTAFE",
        "SANTAFE": "SANTAFEDEANTIOQUIA"
    }
    
    if t in diccionario_rebeldes: 
        t = diccionario_rebeldes[t]

    # A la izquierda Excel, a la derecha el GeoJSON
    diccionario_veredas = {
        "ELVALLANO": "VALLANO",
        "LASPALMAS": "PALMAS",
        "MULATOS": "LOSMULATOS",
        "LAQUIEBRA": "LARAYA", # En Caldas se llama La Raya en el mapa
        "ELROSARIOLOMADELOSZULETA": "LOMADELOSZULETA",
        "ELMELLITO": "ELMELLITOALTO",
        "TABLAZOHATILLO": "ELHATILLO",
        "MANDE": "GUAPANDE", # Urrao
        "PANTANONEGRO": "PANTANOS", # Abejorral
        "ELCHAGUALO": "CHAGUALO", # Abejorral
        "SANNICOLASDELRIO": "SANNICOLAS",
        "ZARZALCURAZAO": "ZARZALCURZAO", # Copacabana (El IGAC lo escribió mal)
        "SADEM": "SADEMGUACAMAYA", # Chigorodó
        "COLORADO": "ELCOLORADO", # Guarne
        "ELESCOBERO": "ESCOBERO", # Envigado
        "LAVERDELAMARIA": "LAMARIA", # Itagüí
        "NUEVOANTIOQUIA": "NUEVAUNION", # Turbo
        "PARAISO": "ELPARAISO", # Barbosa
        "LAAURORA": "LAMADERAAURORA", # El Carmen de V.
        "GUAPA": "GUAPALEON", # Chigorodó
        "SANJOSEDEMULATOS": "ALTODEMULATOS", # Turbo
        "LACIONDOR-X10": "ELCONDOR", # Yondó
        "AGUASCLARAS": "AGUASCLARASSONDORA", # El Carmen de V.
        "POTREROMISERANGA": "POTRERAMISERENGA", # Medellín
        "ELRETIRO": "RETIRO",    # ARGELIA
        "PIÑONAL": "PINONAL",    # BETULIA
        "LOSANTIOQUEÑOS": "LOSANTIOQUENOS",    # CANASGORDAS
        "SANVICENTE-ELKIOSKO": "SANVICENTEELKIOSKO",    # GUADALUPE
        "LAVERDE-LAMARIA": "LAVERDELAMARIA",    # ITAGUI
        "LACABAÑA": "LACABANA",    # ITUANGO
        "SANMIGUEL": "SANMIGUELLADORADA",    # LAUNION
        "POTRERO–MISERANGA": "POTREROMISERANGA",    # MEDELLIN
        "LAVOLCANA-GUAYABAL": "LAVOLCANAGUAYABAL",    # MEDELLIN
        "PATIO-BOLAS": "PATIOBOLAS",    # MEDELLIN
        "BEBARAMEÑO": "BEBARAMENO",    # MURINDO
        "SANANTONIO-ELRIO": "SANANTONIOELRIO",    # REMEDIOS
        "PUERTOGARZA-NARICES": "PUERTOGARZANARICES",    # SANCARLOS
        "NORCASIA-7DEAGOSTO": "NORCASIA7DEAGOSTO",    # SANCARLOS
        "LASPALMAS": "PALMAS",    # SANCARLOS
        "QUEBRADON-20DEJULIO": "QUEBRADON20DEJULIO",    # SANCARLOS
        "CAÑAFISTO": "CANAFISTO",    # SANCARLOS
        "LANUTRIA-CAUNZALES": "LANUTRIACAUNZALES",    # SANFRANCISCO
        "LAHABANA-PALESTINA": "LAHABANAPALESTINA",    # SANLUIS
        "MANIZALES-VILLANUEVA": "MANIZALESVILLANUEVA",    # SANROQUE
        "PEÑASAZULES": "PENASAZULES",    # SANROQUE
        "MORTIÑAL": "MORTINAL",    # SANTAROSADEOSOS
        "SANTAGERTRUDIS-PEÑAS": "SANTAGERTRUDISPENAS",    # SANTODOMINGO
        "ROBLALABAJO-CHIRIMOYO": "ROBLALABAJOCHIRIMOYO",    # SONSON
        "ELLLANO-CAÑAVERAL": "ELLLANOCANAVERAL",    # SONSON
        "SIRGÜITA": "SIRGUITA",    # SONSON
        "BRISAS-CAUNZAL": "BRISASCAUNZAL",    # SONSON
        "SANTAROSA-LADANTA": "SANTAROSALADANTA",    # SONSON
        "SANJOSEMONTAÑITAS": "SANJOSEMONTANITAS",    # URRAO
        "MORRON-SEVILLA": "MORRONSEVILLA",    # VALDIVIA
        "VEREDACABECERAMUNICIPAL": "CABECERAMUNICIPAL",    # YARUMAL
        "LACONDOR-X10": "LACONDORX10",    # YONDO
        "CAÑODONJUAN": "CANODONJUAN",    # YONDO
        "LAROMPIDANO.1": "LAROMPIDANO1",    # YONDO
        "LAROMPIDANO.2": "LAROMPIDANO2",    # YONDO
        "ZONAURBANAVEREDAELDIQUE": "ZONAURBANAELDIQUE"    # YONDO
    }
    
    if t in diccionario_veredas:
        t = diccionario_veredas[t]
        
    return t

@st.cache_data
def cargar_maestro_territorial():
    # URL de tu archivo en Supabase o ruta local
    url_maestro = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.csv"
    try:
        df = pd.read_csv(url_maestro)
        # Normalizamos para que coincida con df_mun
        df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
        return df
    except:
        return pd.DataFrame()

df_maestro = cargar_maestro_territorial()
    
# --- 1. LECTURA DE DATOS LIMPIOS Y VEREDALES ---
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

REGIONES_COL = {
    'Caribe': ['Atlántico', 'Bolívar', 'Cesar', 'Córdoba', 'La Guajira', 'Magdalena', 'Sucre', 'Archipiélago De San Andrés'],
    'Pacífica': ['Cauca', 'Chocó', 'Nariño', 'Valle Del Cauca'],
    'Andina': ['Antioquia', 'Boyacá', 'Caldas', 'Cundinamarca', 'Huila', 'Norte De Santander', 'Quindio', 'Risaralda', 'Santander', 'Tolima', 'Bogotá, D.C.'],
    'Orinoquía': ['Arauca', 'Casanare', 'Meta', 'Vichada'],
    'Amazonía': ['Amazonas', 'Caquetá', 'Guainía', 'Guaviare', 'Putumayo', 'Vaupés']
}

@st.cache_data
def cargar_datos_limpios():
    try:
        # =======================================================
        # 1. EL NUEVO CORAZÓN: Conexión Directa a Supabase
        # =======================================================
        url_parquet = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Poblacion_Colombia_1985_2042_Optimizado.parquet" 
        
        df_master = pd.read_parquet(url_parquet)
        
        columnas_base = {
            'DPNOM': 'depto_nom',
            'DPMP': 'municipio',
            'AÑO': 'año',
            'AREA_GEOGRAFICA': 'area_geografica'
        }
        for col_orig in df_master.columns:
            col_upper = str(col_orig).strip().upper()
            if col_upper in columnas_base:
                df_master = df_master.rename(columns={col_orig: columnas_base[col_upper]})

        # ESTANDARIZACIÓN DE NOMBRES
        df_master['depto_nom'] = df_master['depto_nom'].str.title()
        df_master['municipio'] = df_master['municipio'].str.title()
        
        # ESCUDO DE AÑO: Forzamos que el año sea un número entero y no texto
        df_master['año'] = pd.to_numeric(df_master['año'], errors='coerce').fillna(1985).astype(int)
        
        # --- SOLUCIÓN QUINDÍO Y SAN ANDRÉS ---
        reemplazos_deptos = {
            'Quindio': 'Quindío',
            # Forzamos el nombre corto para que cruce con el GeoJSON
            'Archipiélago De San Andrés, Providencia Y Santa Catalina': 'Archipiélago De San Andrés',
            'Archipielago De San Andres': 'Archipiélago De San Andrés',
            'Bogota D.c.': 'Bogotá, D.C.'
        }
        df_master['depto_nom'] = df_master['depto_nom'].replace(reemplazos_deptos)
        
        # --- TRADUCTOR ROBUSTO DE ÁREAS GEOGRÁFICAS ---
        # 1. Convertimos todo a minúsculas y quitamos espacios a los lados para evitar fallos
        df_master['area_geografica'] = df_master['area_geografica'].astype(str).str.lower().str.strip()
        
        # 2. Escáner inteligente a prueba de balas (Sella la fuga rural del 100%)
        def clasificar_dane(x):
            # 🔥 CORRECCIÓN CRÍTICA: Aquí es donde normalizamos para que la UI no se rompa
            if 'cabecera' in x or 'urban' in x: return 'urbano'
            if 'rural' in x or 'centros' in x or 'resto' in x: return 'rural'
            return 'total'
            
        df_master['area_geografica'] = df_master['area_geografica'].apply(clasificar_dane)
        
        # 🔥 FIX DEFINITIVO PARA LA INTERFAZ (Pirámides):
        # Aseguramos que la primera letra sea mayúscula para que la interfaz se vea elegante
        df_master['area_geografica'] = df_master['area_geografica'].str.title()

        # --- SEPARACIÓN DE HOMBRES Y MUJERES (CON ESCUDO NUMÉRICO) ---
        cols_hombres = [c for c in df_master.columns if 'Hombre' in str(c) and any(char.isdigit() for char in str(c))]
        cols_mujeres = [c for c in df_master.columns if 'Mujer' in str(c) and any(char.isdigit() for char in str(c))]
        cols_poblacion = cols_hombres + cols_mujeres
        
        for col in cols_poblacion:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce').fillna(0)
            
        df_master['Hombres'] = df_master[cols_hombres].sum(axis=1)
        df_master['Mujeres'] = df_master[cols_mujeres].sum(axis=1)
        df_master['Total'] = df_master['Hombres'] + df_master['Mujeres']

        # -------------------------------------------------------
        # A. Crear df_mun (Municipal)
        # -------------------------------------------------------
        df_mun = df_master[['año', 'depto_nom', 'municipio', 'area_geografica', 'Total', 'Hombres', 'Mujeres'] + cols_poblacion].copy()
        
        def asignar_region(fila):
            uraba_caribe = ["TURBO", "NECOCLI", "SAN JUAN DE URABA", "ARBOLETES", "SAN PEDRO DE URABA", "APARTADO", "CAREPA", "CHIGORODO", "MUTATA"]
            if normalizar_texto(fila['municipio']) in [normalizar_texto(m) for m in uraba_caribe]:
                return 'Caribe'
            depto = fila['depto_nom']
            # NOTA: Asegúrate de tener REGIONES_COL y normalizar_texto definidos antes de llamar esta función
            try:
                for region, deptos in REGIONES_COL.items():
                    if depto in deptos: return region
            except: pass
            return "Sin Región"
            
        df_mun['Macroregion'] = df_mun.apply(asignar_region, axis=1)

        # -------------------------------------------------------
        # B. Crear df_nac (Nacional) -> AHORA INCLUYE TODAS LAS EDADES
        # -------------------------------------------------------
        df_nac_temp = df_mun[df_mun['area_geografica'].str.lower() == 'total']
        cols_agrupar_nac = ['Total', 'Hombres', 'Mujeres'] + cols_poblacion
        df_nac = df_mun.groupby(['año', 'area_geografica'])[cols_agrupar_nac].sum().reset_index()

        # -------------------------------------------------------
        # C. Crear df_global (Fusión Dinámica + Histórica)
        # -------------------------------------------------------
        mpios_amva = ['Medellín', 'Bello', 'Itagüí', 'Envigado', 'Sabaneta', 'Copacabana', 'La Estrella', 'Girardota', 'Caldas', 'Barbosa']
        df_ant = df_nac_temp[df_nac_temp['depto_nom'] == 'Antioquia'].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Antioquia'})
        df_amva = df_nac_temp[(df_nac_temp['depto_nom'] == 'Antioquia') & (df_nac_temp['municipio'].str.title().isin(mpios_amva))].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Amva'})
        df_med = df_nac_temp[df_nac_temp['municipio'] == 'Medellín'].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Medellin'})
        
        df_global_dinamico = pd.merge(df_ant, df_amva, on='año', how='outer')
        df_global_dinamico = pd.merge(df_global_dinamico, df_med, on='año', how='outer')
        df_global_dinamico = df_global_dinamico.rename(columns={'año': 'Año'})

        # --- CONEXIÓN AL ARCHIVO HISTÓRICO GLOBAL (SUPABASE) ---
        url_global = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Pob_Col_Ant_Amva_Med.csv"
        df_global = df_global_dinamico.copy()
        
        try:
            # Intentamos leer con coma, si falla probamos punto y coma
            df_global_csv = pd.read_csv(url_global, sep=',', encoding='utf-8')
            if len(df_global_csv.columns) == 1:
                df_global_csv = pd.read_csv(url_global, sep=';', encoding='utf-8')
                
            # Forzamos que la columna de tiempo se llame 'Año'
            if 'año' in df_global_csv.columns:
                df_global_csv = df_global_csv.rename(columns={'año': 'Año'})
                
            # --- ESCUDO EXCLUSIVO PARA LA ESCALA MUNDIAL ---
            for col in df_global_csv.columns:
                if col != 'Año':
                    if df_global_csv[col].dtype == 'O': 
                        df_global_csv[col] = pd.to_numeric(
                            df_global_csv[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False),
                            errors='coerce'
                        )
                
            df_global = pd.merge(df_global_csv, df_global_dinamico, on='Año', how='outer')
        except Exception as e:
            pass
                
        # =======================================================
        # 2. Cargar datos Veredales y MATRIZ (Dinámico desde Supabase)
        # =======================================================
        try:
            from modules.db_manager import get_engine
            engine_db = get_engine()
            
            # 1. Cargamos Veredas
            df_ver = pd.read_sql("SELECT * FROM veredas_poblacion", engine_db)
            
            # 2. 🔥 FIX: CARGAMOS LA MATRIZ FRESCA RECIÉN ENTRENADA
            df_matriz = pd.read_sql("SELECT * FROM matriz_maestra_demografica", engine_db)
            
        except Exception as e:
            df_ver = pd.DataFrame() # Escudo anti-errores
            df_matriz = pd.DataFrame() # Si no hay conexión, queda vacía
            
        # --- SINCRONIZACIÓN DE NOMBRES VEREDAS ---
        if 'Municipio' in df_ver.columns:
            df_ver['Municipio'] = df_ver['Municipio'].astype(str).str.title()
        if 'Poblacion_hab' in df_ver.columns:
            df_ver['Poblacion_hab'] = pd.to_numeric(df_ver['Poblacion_hab'].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').fillna(0)

        # =======================================================
        # 3. 🗺️ CARGA DEL MAESTRO TERRITORIAL (Sincronizado con Supabase)
        # =======================================================
        df_maestro = pd.DataFrame()
        # URL directa al archivo Excel que confirmaste en tu bucket
        url_maestro_xlsx = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx"
        
        try:
            # Leemos directamente de la nube sin pasar por el disco local
            df_maestro = pd.read_excel(url_maestro_xlsx, engine='openpyxl')
        except Exception as e:
            st.error(f"🚨 Error al descargar el Maestro Territorial desde Supabase: {e}")

        # Escudo de seguridad por si la descarga falla o el archivo está mal estructurado
        if df_maestro.empty or 'depto_nom' not in df_maestro.columns:
            st.warning("⚠️ El archivo maestro llegó vacío o con errores. Usando estructura de emergencia.")
            df_maestro = pd.DataFrame(columns=['depto_nom', 'municipio', 'subregion', 'car', 'municipio_norm'])
        else:
            # Normalización automática: El paso más importante para que las Subregiones y CARs funcionen
            try:
                df_maestro['municipio_norm'] = df_maestro['municipio'].astype(str).apply(normalizar_texto)
            except: pass

        # Devolvemos el maestro como el 5to elemento de la tupla
        return df_nac, df_mun, df_ver, df_global, df_maestro, df_matriz
        
    except Exception as e:
        import streamlit as st
        st.error(f"🚨 Error cargando las bases de datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# 🔥 FIX: ACTUALIZAMOS LA LLAMADA PARA RECIBIR LA MATRIZ
df_nac, df_mun, df_ver, df_global, df_maestro, df_matriz = cargar_datos_limpios()

# Si los datos esenciales fallan, detenemos la app para evitar errores en cascada
if df_nac.empty or df_mun.empty: 
    st.error("No se pudieron cargar los datos base del DANE. Verifique la conexión a Supabase.")
    st.stop()
    
# --- 2. MODELOS MATEMÁTICOS ---
def modelo_lineal(x, m, b): return m * x + b
def modelo_exponencial(x, p0, r): return p0 * np.exp(r * (x - 2005))
def modelo_logistico(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))
def modelo_polinomial(x, A, B, C, D): return A*(x**3) + B*(x**2) + C*x + D    

# --- 3. CONFIGURACIÓN Y FILTROS LATERALES ---
st.sidebar.header("⚙️ 1. Selección Territorial")

# 1. LA LISTA MAESTRA (Sincronizada con el Solver y las Nuevas Escalas)
escala_sel = st.sidebar.radio("Nivel de Análisis:", [
    "🌍 Global y Suramérica",
    "🇨🇴 Nacional (Colombia)", 
    "🏛️ Departamental (Colombia)", 
    "🗺️ Subregiones (Antioquia)", # <-- NUEVA ESCALA 1
    "🦅 Autoridades Ambientales (CARs)", # <-- NUEVA ESCALA 2
    "🧩 Regional (Macroregiones)",
    "💧 Cuencas Hidrográficas", 
    "🏢 Municipal (Regiones)", 
    "🏢 Municipal (Departamentos)",
    "🏙️ Escala Urbana (Cabeceras Antioquia)", 
    "🌿 Veredal (Antioquia)",
    "🏘️ Escala Intra-Urbana (Medellín)"
])

# --- SELECTOR GLOBAL DE ÁREA (Afecta Gráficos y Mapas) ---

if escala_sel == "🌿 Veredal (Antioquia)":
    area_global = "Rural"
    st.info("ℹ️ A escala veredal, el motor matemático calcula todo como población rural.")
elif escala_sel == "🏙️ Escala Urbana (Cabeceras Antioquia)":
    area_global = "Urbano" # <-- FIX: Forzamos la matriz urbana
    st.info("ℹ️ A escala de Cabeceras, el motor matemático aísla la población urbana.")
else:
    area_global = st.sidebar.selectbox("Filtro Poblacional:", ["Total", "Urbano", "Rural"])
    
años_hist, pob_hist = [], []
df_mapa_base = pd.DataFrame()

# ==========================================================
# 2. LOS MOTORES LÓGICOS (Uno para cada elemento de la lista)
# ==========================================================

if escala_sel == "🌍 Global y Suramérica":
    opciones_hist = ["Mundo", "Suramérica", "Colombia (DANE)", "Colombia (ONU)", "Antioquia", "AMVA", "Medellín"]
    contexto_sel = st.sidebar.selectbox("Seleccione la región histórica:", opciones_hist)
    
    filtro_zona = contexto_sel
    titulo_terr = contexto_sel
    
    if not df_global.empty:
        diccionario_columnas = {
            "Mundo": "Pob_mundial", 
            "Suramérica": "Pob_suramerica", 
            "Colombia (DANE)": "Pob_Colombia_DANE", 
            "Colombia (ONU)": "Pob_Colombia_ONU",
            "Antioquia": "Pob_Antioquia",
            "AMVA": "Pob_Amva", 
            "Medellín": "Pob_Medellin"
        }
        col_objetivo = diccionario_columnas[contexto_sel]
        
        # ESCUDO: Solo graficar si la columna realmente existe en los datos
        col_anio_glob = 'Año' if 'Año' in df_global.columns else 'año'
        
        if col_objetivo in df_global.columns:
            df_temp = df_global.dropna(subset=[col_anio_glob, col_objetivo])
            años_hist = df_temp[col_anio_glob].values
            pob_hist = df_temp[col_objetivo].values
        else:
            st.warning(f"⚠️ Los datos de '{contexto_sel}' no están en el nuevo archivo maestro.")
            años_hist, pob_hist = np.array([]), np.array([])
    else:
        años_hist, pob_hist = np.array([]), np.array([])
        
    df_mapa_base = pd.DataFrame()

elif escala_sel == "🇨🇴 Nacional (Colombia)":
    df_base = df_nac[df_nac['area_geografica'] == 'total'].copy()
    filtro_zona = "Colombia"
    titulo_terr = "Colombia"
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    
    # ¡CONSERVAMOS TU VALIOSO ESCUDO DE COLUMNAS!
    if 'Total' in df_base.columns: df_base['Pob_Calc'] = df_base['Total']
    elif 'Population' in df_base.columns: df_base['Pob_Calc'] = df_base['Population']
    elif 'Poblacion' in df_base.columns: df_base['Pob_Calc'] = df_base['Poblacion']
    elif 'Hombres' in df_base.columns and 'Mujeres' in df_base.columns:
        df_base['Pob_Calc'] = df_base['Hombres'] + df_base['Mujeres']
    else:
        df_base['Pob_Calc'] = df_base.iloc[:, 1]
        
    df_hist = df_base.groupby(col_anio)['Pob_Calc'].sum().reset_index()
    df_hist = df_hist.sort_values(by=col_anio) 
    
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Pob_Calc'].values
    
    df_mapa_base = df_mun.groupby(['municipio', 'depto_nom', 'año'])['Total'].sum().reset_index()
    df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

# --- NUEVO MOTOR: DEPARTAMENTAL ---
elif escala_sel == "🏛️ Departamental (Colombia)":
    depto_sel = st.sidebar.selectbox("Departamento:", sorted(df_mun['depto_nom'].unique()))
    # 🔥 FIX: Quitamos el candado 'total' para que fluyan las 3 áreas
    df_base = df_mun[df_mun['depto_nom'] == depto_sel]
    
    filtro_zona = depto_sel
    titulo_terr = depto_sel
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    # Aplicamos el filtro del sidebar dinámicamente
    df_hist = df_base[df_base['area_geografica'].str.lower() == area_global.lower()].groupby(col_anio)['Total'].sum().reset_index()
    df_hist = df_hist.sort_values(by=col_anio)
    
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base[df_base['area_geografica'].str.lower() == area_global.lower()].groupby(['municipio', col_anio])['Total'].sum().reset_index()
    df_mapa_base.rename(columns={'municipio': 'Territorio'}, inplace=True)
    df_mapa_base['Padre'] = depto_sel

elif escala_sel in ["🗺️ Subregiones (Antioquia)", "🦅 Autoridades Ambientales (CARs)"]:
    col_agrupadora = 'subregion' if "Subregiones" in escala_sel else 'car'
    
    # --- 🧽 TRADUCTOR UNIVERSAL (Limpieza Absoluta) ---
    def limpiar_nombres(s):
        if pd.isna(s): return ""
        s = str(s).upper().strip() # Todo a mayúscula, sin espacios a los lados
        import unicodedata
        # Quitamos todas las tildes (á -> A)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s

    # 1. Limpiamos el Maestro Territorial
    maestro_ant = df_maestro.copy()
    if 'depto_nom' in maestro_ant.columns:
        maestro_ant['depto_nom_cln'] = maestro_ant['depto_nom'].apply(limpiar_nombres)
        maestro_ant = maestro_ant[maestro_ant['depto_nom_cln'] == 'ANTIOQUIA']
    
    if col_agrupadora in maestro_ant.columns:
        maestro_ant[col_agrupadora] = maestro_ant[col_agrupadora].apply(limpiar_nombres)
        opciones = maestro_ant[col_agrupadora].dropna().unique().tolist()
        
        # 🔥 INYECCIÓN MANUAL DEL AMVA: Lo forzamos en la lista aunque el Excel diga Corantioquia
        if escala_sel == "🦅 Autoridades Ambientales (CARs)" and "AMVA" not in opciones:
            opciones.append("AMVA")
            
        opciones = sorted(opciones)
    else:
        opciones = []
    
    sel_territorio = st.sidebar.selectbox(f"Seleccione {col_agrupadora.title()}:", opciones)
    
    # Mapeamos los municipios base (Garantizando el formato)
    if 'municipio' in maestro_ant.columns:
        maestro_ant['mun_norm_local'] = maestro_ant['municipio'].apply(limpiar_nombres)
        mpios_en_zona = maestro_ant[maestro_ant[col_agrupadora] == sel_territorio]['mun_norm_local'].tolist()
    else:
        mpios_en_zona = []
    
    # 2. Limpiamos el DANE (df_mun) para que hablen el mismo idioma
    df_mun_ant = df_mun.copy()
    if 'depto_nom' in df_mun_ant.columns:
        df_mun_ant['depto_nom_cln'] = df_mun_ant['depto_nom'].apply(limpiar_nombres)
        df_mun_ant = df_mun_ant[df_mun_ant['depto_nom_cln'] == 'ANTIOQUIA']
    
    if 'municipio' in df_mun_ant.columns:
        df_mun_ant['mun_norm_local'] = df_mun_ant['municipio'].apply(limpiar_nombres)
    else:
        df_mun_ant['mun_norm_local'] = ""
    
    # ---------------------------------------------------------
    # ⚖️ RESOLUCIÓN DE JURISDICCIÓN COMPARTIDA (AMVA vs CORANTIOQUIA)
    # ---------------------------------------------------------
    if escala_sel == "🦅 Autoridades Ambientales (CARs)":
        # Nombres limpios, en mayúscula y con espacios correctos
        mpios_amva = ['MEDELLIN', 'BELLO', 'ITAGUI', 'ENVIGADO', 'SABANETA', 'COPACABANA', 'LA ESTRELLA', 'GIRARDOTA', 'CALDAS', 'BARBOSA']
        
        if sel_territorio == 'AMVA':
            df_amva_urb = df_mun_ant[(df_mun_ant['mun_norm_local'].isin(mpios_amva)) & (df_mun_ant['area_geografica'].str.lower() == 'urbano')].copy()
            df_amva_tot = df_amva_urb.copy()
            df_amva_tot['area_geografica'] = 'total'
            
            df_base = pd.concat([df_amva_urb, df_amva_tot]) if not df_amva_urb.empty else pd.DataFrame()
            st.sidebar.info("🏢 **Jurisdicción Urbana:** El AMVA rige únicamente sobre las cabeceras municipales del Valle de Aburrá.")
            
        elif sel_territorio == 'CORANTIOQUIA':
            mpios_propios = [m for m in mpios_en_zona if m not in mpios_amva]
            df_propios = df_mun_ant[df_mun_ant['mun_norm_local'].isin(mpios_propios)].copy()
            
            df_amva_rur = df_mun_ant[(df_mun_ant['mun_norm_local'].isin(mpios_amva)) & (df_mun_ant['area_geografica'].str.lower() == 'rural')].copy()
            df_amva_rur_tot = df_amva_rur.copy()
            df_amva_rur_tot['area_geografica'] = 'total'
            
            df_base = pd.concat([df_propios, df_amva_rur, df_amva_rur_tot]) if not df_propios.empty or not df_amva_rur.empty else pd.DataFrame()
            st.sidebar.info("🌿 **Jurisdicción Mixta:** Incluye sus municipios propios y exclusivamente las áreas rurales del Valle de Aburrá.")
            
        else:
            df_base = df_mun_ant[df_mun_ant['mun_norm_local'].isin(mpios_en_zona)]
    else:
        df_base = df_mun_ant[df_mun_ant['mun_norm_local'].isin(mpios_en_zona)]
    # ---------------------------------------------------------
    
    # 🔥 ESCUDO ANTI-NULOS: Verificamos que haya una selección válida
    if sel_territorio:
        filtro_zona = str(sel_territorio).title()
        titulo_terr = f"{col_agrupadora.upper()}: {filtro_zona}"
    else:
        filtro_zona = "Sin Selección"
        titulo_terr = f"{col_agrupadora.upper()}: No Disponible"
    
   
    # Matemáticas para gráficos
    if not df_base.empty:
        df_hist = df_base[df_base['area_geografica'].str.lower() == area_global.lower()].groupby('año')['Total'].sum().reset_index()
        años_hist = df_hist['año'].values
        pob_hist = df_hist['Total'].values
    else:
        st.sidebar.warning(f"⚠️ No se encontraron municipios para {sel_territorio}")
        años_hist, pob_hist = np.array([]), np.array([])
    
    # Preparación para el mapa
    df_mapa_base = df_base.copy()
    if not df_mapa_base.empty:
        df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

elif escala_sel == "🧩 Regional (Macroregiones)":
    regiones_list = sorted([r for r in df_mun['Macroregion'].unique() if r != "Sin Región"])
    reg_sel = st.sidebar.selectbox("Seleccione la Macroregión:", regiones_list)
    
    filtro_zona = reg_sel
    titulo_terr = f"Región {reg_sel}"
    
    df_terr = df_mun[(df_mun['Macroregion'] == reg_sel) & (df_mun['area_geografica'] == 'total')].groupby('año')['Total'].sum().reset_index()
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values
    
    df_mapa_base = df_mun[df_mun['Macroregion'] == reg_sel].groupby(['municipio', 'depto_nom', 'año'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel == "💧 Cuencas Hidrográficas":
    if not df_matriz.empty and 'Nivel' in df_matriz.columns:
        df_cuencas_solo = df_matriz[df_matriz['Nivel'] == 'Cuenca'].copy()
        
        if not df_cuencas_solo.empty:
            # --- 1. MOTOR DE FILTROS EN CASCADA ---
            @st.cache_data(ttl=3600)
            def cargar_jerarquia_cuencas():
                try:
                    from modules.db_manager import get_engine
                    from sqlalchemy import text
                    # 🔥 FIX KEYERROR: Restauramos la columna 'subc_lbl' que necesitan los Rankings
                    q_hier = text("""
                        SELECT DISTINCT nomah, nomzh, nom_szh, nom_nss1, nom_nss2, nom_nss3,
                        COALESCE(
                            NULLIF(TRIM(nom_nss3), ''), NULLIF(TRIM(nom_nss2), ''), NULLIF(TRIM(nom_nss1), ''), 
                            NULLIF(TRIM(nom_szh), ''), NULLIF(TRIM(nomzh), ''), NULLIF(TRIM(nomah), ''), 'Cuenca Sin Nombre'
                        ) AS subc_lbl
                        FROM cuencas
                    """)
                    return pd.read_sql(q_hier, get_engine())
                except: return pd.DataFrame()
            
            df_hier = cargar_jerarquia_cuencas()
            
            if not df_hier.empty:
                st.sidebar.markdown("### 🌊 Filtros Jerárquicos")
                ah_opts = sorted(df_hier['nomah'].dropna().unique())
                sel_ah = st.sidebar.selectbox("1. Área Hidrográfica (AH):", ["-- Seleccione --"] + ah_opts)
                
                zh_opts = sorted(df_hier[df_hier['nomah'] == sel_ah]['nomzh'].dropna().unique()) if sel_ah != "-- Seleccione --" else []
                sel_zh = st.sidebar.selectbox("2. Zona Hidrológica (ZH):", ["-- Seleccione --"] + zh_opts)
                
                szh_opts = sorted(df_hier[df_hier['nomzh'] == sel_zh]['nom_szh'].dropna().unique()) if sel_zh != "-- Seleccione --" else []
                sel_szh = st.sidebar.selectbox("3. Subzona (SZH):", ["-- Seleccione --"] + szh_opts)
                
                resolucion = st.sidebar.radio("🔎 Resolución de visualización:", ["NSS1 (Macro)", "NSS2 (Intermedia)", "NSS3 (Micro)"])
                col_res = 'nom_nss1' if 'NSS1' in resolucion else ('nom_nss2' if 'NSS2' in resolucion else 'nom_nss3')
                
                if sel_szh != "-- Seleccione --": cuencas_disp = sorted(df_hier[df_hier['nom_szh'] == sel_szh][col_res].dropna().unique())
                elif sel_zh != "-- Seleccione --": cuencas_disp = sorted(df_hier[df_hier['nomzh'] == sel_zh][col_res].dropna().unique())
                elif sel_ah != "-- Seleccione --": cuencas_disp = sorted(df_hier[df_hier['nomah'] == sel_ah][col_res].dropna().unique())
                else: cuencas_disp = sorted(df_hier[col_res].dropna().unique())
            else:
                cuencas_disp = sorted(df_cuencas_solo['Territorio'].dropna().astype(str).unique())

            cuenca_sel = st.sidebar.multiselect("🎯 Seleccione cuencas específicas (Opcional):", cuencas_disp, default=None)
            
            # --- 2. 🎯 DETERMINAR QUÉ LEER DE LA MATRIZ ---
            # REGLA DE ORO: No calcular piezas sueltas. Leer el dato maestro directamente.
            if cuenca_sel:
                filtro_zona = " + ".join(cuenca_sel[:2]) + ("..." if len(cuenca_sel)>2 else "")
                titulo_terr = f"Cuencas Seleccionadas: {filtro_zona}"
                territorios_a_buscar = cuenca_sel
                territorios_para_mapa = cuenca_sel
            else:
                if not df_hier.empty:
                    if sel_szh != "-- Seleccione --": 
                        titulo_terr = f"SZH: {sel_szh}"
                        territorios_a_buscar = [sel_szh]
                        territorios_para_mapa = cuencas_disp # Dibujar las sub-piezas en el mapa
                    elif sel_zh != "-- Seleccione --": 
                        titulo_terr = f"ZH: {sel_zh}"
                        territorios_a_buscar = [sel_zh]
                        territorios_para_mapa = cuencas_disp
                    elif sel_ah != "-- Seleccione --": 
                        titulo_terr = f"AH: {sel_ah}"
                        territorios_a_buscar = [sel_ah]
                        territorios_para_mapa = cuencas_disp
                    else: 
                        titulo_terr = "Todas las Cuencas"
                        # 🔥 FIX 35M: Si son todas, sumamos SOLO las 3 Áreas Hidrográficas Mayores (No a los hijos)
                        territorios_a_buscar = df_hier['nomah'].dropna().unique().tolist()
                        territorios_para_mapa = df_hier['nomzh'].dropna().unique().tolist() # Dibujar ZHs en el mapa
                else:
                    titulo_terr = "Todas las Cuencas"
                    territorios_a_buscar = df_cuencas_solo['Territorio'].unique().tolist()
                    territorios_para_mapa = territorios_a_buscar
                filtro_zona = titulo_terr

            # --- 3. ⚡ LECTURA MATEMÁTICA DIRECTA (Bypass Anti-Amnesia Rural) ---
            años_hist = np.arange(1985, 2043)
            pob_hist_acumulada = np.zeros_like(años_hist, dtype=float)
            
            # Aplanamos los nombres para búsquedas infalibles
            df_cuencas_solo['MATCH_ID'] = df_cuencas_solo['Territorio'].astype(str).apply(normalizar_texto)
            area_buscada = str(area_global).strip().lower()

            def evaluar_curva(fila, anios):
                mod = str(fila.get('Modelo_Recomendado', 'Desconocido'))
                x = anios - 1985 # Offset base para alinear los años
                
                if 'Logistico' in mod or 'Logístico' in mod: 
                    return fila.get('Log_K', 0) / (1 + fila.get('Log_a', 0) * np.exp(-fila.get('Log_r', 0) * x))
                if 'Exponencial' in mod: 
                    return fila.get('Exp_a', 0) * np.exp(fila.get('Exp_b', 0) * x)
                if 'Polinomial' in mod:
                    return fila.get('Poly_A', 0)*(x**3) + fila.get('Poly_B', 0)*(x**2) + fila.get('Poly_C', 0)*x + fila.get('Poly_D', 0)
                if 'Lineal' in mod:
                    return fila.get('Lin_m', 0) * x + fila.get('Lin_b', 0)
                    
                return np.full_like(anios, fila.get('Pob_Base', 0))

            import difflib
            
            # 🔥 FUNCIÓN MAESTRA: Aplica el Bypass Rural tanto a gráficas como al mapa
            def calcular_poblacion_bypass(t_norm, anios_array):
                filas_terr = df_cuencas_solo[df_cuencas_solo['MATCH_ID'] == t_norm]
                if filas_terr.empty:
                    matches = difflib.get_close_matches(t_norm, df_cuencas_solo['MATCH_ID'].tolist(), n=1, cutoff=0.8)
                    if matches: filas_terr = df_cuencas_solo[df_cuencas_solo['MATCH_ID'] == matches[0]]
                
                pob_calculada = np.zeros_like(anios_array, dtype=float)
                if not filas_terr.empty:
                    if area_buscada == 'total':
                        # Sumamos Urbano + Rural para evitar datos faltantes en SQL
                        fila_u = filas_terr[filas_terr['Area'].str.lower().isin(['urbano', 'urbana', 'cabecera'])]
                        fila_r = filas_terr[filas_terr['Area'].str.lower().isin(['rural', 'resto'])]
                        sumo_partes = False
                        
                        if not fila_u.empty: 
                            pob_calculada += evaluar_curva(fila_u.iloc[0], anios_array)
                            sumo_partes = True
                        if not fila_r.empty: 
                            pob_calculada += evaluar_curva(fila_r.iloc[0], anios_array)
                            sumo_partes = True
                            
                        # Fallback si no existen las partes, usa el Total crudo
                        if not sumo_partes:
                            fila_t = filas_terr[filas_terr['Area'].str.lower() == 'total']
                            if not fila_t.empty: pob_calculada += evaluar_curva(fila_t.iloc[0], anios_array)
                    else:
                        # Filtrado directo si elige específicamente Urbano o Rural
                        if 'urb' in area_buscada: fila_esp = filas_terr[filas_terr['Area'].str.lower().isin(['urbano', 'urbana', 'cabecera'])]
                        else: fila_esp = filas_terr[filas_terr['Area'].str.lower().isin(['rural', 'resto'])]
                        if not fila_esp.empty: pob_calculada += evaluar_curva(fila_esp.iloc[0], anios_array)
                return pob_calculada

            # Aplicamos a las curvas históricas
            for t_crudo in territorios_a_buscar:
                pob_hist_acumulada += calcular_poblacion_bypass(normalizar_texto(t_crudo), años_hist)

            pob_hist = pob_hist_acumulada

            # --- 4. 🗺️ DATOS PARA EL MAPA ---
            mapa_data = []
            for t_mapa in territorios_para_mapa:
                val_tot_arr = calcular_poblacion_bypass(normalizar_texto(t_mapa), np.array([2024]))
                val_tot = val_tot_arr[0] if len(val_tot_arr) > 0 else 0
                mapa_data.append({'Territorio': t_mapa, 'Total': val_tot, 'area_geografica': area_global.lower(), 'Padre': titulo_terr})
            
            df_mapa_base = pd.DataFrame(mapa_data)

        else:
            filtro_zona, titulo_terr, años_hist, pob_hist, df_mapa_base = "Ninguna", "Sin Datos", np.array([]), np.array([]), pd.DataFrame()
    else:
        st.sidebar.warning("⚠️ Entrena la matriz de cuencas en la pestaña 4.")
        filtro_zona, titulo_terr, años_hist, pob_hist, df_mapa_base = "Error", "Sin Datos", np.array([]), np.array([]), pd.DataFrame()
        
elif escala_sel in ["🏢 Municipal (Regiones)", "🏢 Municipal (Departamentos)"]:
    if "Regiones" in escala_sel:
        agrupador_col = 'Macroregion'
        agrupador_sel = st.sidebar.selectbox("Macroregión:", sorted([r for r in df_mun['Macroregion'].unique() if r != "Sin Región"]))
    else:
        agrupador_col = 'depto_nom'
        agrupador_sel = st.sidebar.selectbox("Departamento:", sorted(df_mun['depto_nom'].unique()))
        
    mpios_filtrados = df_mun[df_mun[agrupador_col] == agrupador_sel]
    municipio_sel = st.sidebar.selectbox("Municipio:", sorted(mpios_filtrados['municipio'].unique()))
    
    df_base = mpios_filtrados[mpios_filtrados['municipio'] == municipio_sel]
    filtro_zona = municipio_sel
    titulo_terr = f"{municipio_sel} ({agrupador_sel})"
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    # 🔥 FIX: Respetamos la selección (Total/Urbano/Rural)
    df_hist = df_base[df_base['area_geografica'].str.lower() == area_global.lower()].groupby(col_anio)['Total'].sum().reset_index()
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base[df_base['area_geografica'].str.lower() == area_global.lower()].copy()
    df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

# =====================================================================
# 🏘️ ESCALA INTRA-URBANA (MEDELLÍN V7 - NOMBRES REALES)
# =====================================================================
elif escala_sel == "🏘️ Escala Intra-Urbana (Medellín)":
    st.sidebar.markdown("### 🏘️ Explorador de Medellín")
    # 1. Ajuste de Nomenclatura solicitado
    nivel_medellin = st.sidebar.radio("Nivel de Detalle:", ["Barrios y Corregimientos", "Comunas"])
    
    try:
        import geopandas as gpd
        import json
        
        @st.cache_data(ttl=60)
        def cargar_barrios_medellin_v6():
            URL_BARRIOS = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/geojson/PoblacionBarrioCorregimiento_optimizado.geojson"
            return gpd.read_file(URL_BARRIOS).to_crs(epsg=4326)
        
        gdf_med = cargar_barrios_medellin_v6()
        
        # Diccionario Oficial de Comunas y Corregimientos
        dict_comunas = {
            "01": "Popular", "02": "Santa Cruz", "03": "Manrique", "04": "Aranjuez",
            "05": "Castilla", "06": "Doce de Octubre", "07": "Robledo", "08": "Villa Hermosa",
            "09": "Buenos Aires", "10": "La Candelaria", "11": "Laureles - Estadio", "12": "La América",
            "13": "San Javier", "14": "El Poblado", "15": "Guayabal", "16": "Belén",
            "50": "Corregimiento Palmitas", "60": "Corregimiento San Cristóbal", 
            "70": "Corregimiento Altavista", "80": "Corregimiento San Antonio de Prado", 
            "90": "Corregimiento Santa Elena"
        }
        
        if nivel_medellin == "Barrios y Corregimientos":
            lista_barrios = sorted(gdf_med['NombreBarr'].dropna().unique())
            barrio_sel = st.sidebar.selectbox("Seleccione un Barrio/Corregimiento:", ["Todos"] + lista_barrios)
            
            if barrio_sel != "Todos":
                gdf_plot = gdf_med[gdf_med['NombreBarr'] == barrio_sel].copy()
                titulo_terr = f"Barrio/Correg.: {barrio_sel}"
            else:
                gdf_plot = gdf_med.copy()
                titulo_terr = "Todos los Barrios y Corregimientos"
            
            gdf_plot['MATCH_ID'] = gdf_plot['Cod_Barrio'].astype(str).str.zfill(4)
            st.session_state['boveda_mapa_medellin'] = json.loads(gdf_plot.to_json())
            
            df_mapa_base = pd.DataFrame({
                'Territorio': gdf_plot['NombreBarr'],
                'MATCH_ID': gdf_plot['MATCH_ID'],
                'Total': pd.to_numeric(gdf_plot['Pob_Total'], errors='coerce').fillna(0),
                'Padre': 'Medellín',
                'area_geografica': 'total'
            })
            
        else:
            gdf_med['Cod_Comuna'] = gdf_med['Cod_Barrio'].astype(str).str.zfill(4).str[:2]
            gdf_comunas = gdf_med.dissolve(by='Cod_Comuna', aggfunc={'Pob_Total': 'sum'}).reset_index()
            
            # Asignamos los nombres reales usando el diccionario
            gdf_comunas['NombreComuna'] = gdf_comunas['Cod_Comuna'].map(dict_comunas).fillna("Comuna " + gdf_comunas['Cod_Comuna'])
            
            lista_comunas = sorted(gdf_comunas['NombreComuna'].unique())
            comuna_sel = st.sidebar.selectbox("Seleccione una Comuna:", ["Todas"] + lista_comunas)
            
            if comuna_sel != "Todas":
                gdf_plot = gdf_comunas[gdf_comunas['NombreComuna'] == comuna_sel].copy()
                titulo_terr = comuna_sel
            else:
                gdf_plot = gdf_comunas.copy()
                titulo_terr = "Todas las Comunas y Corregimientos"

            gdf_plot['MATCH_ID'] = gdf_plot['Cod_Comuna'].astype(str).str.zfill(2)
            st.session_state['boveda_mapa_medellin'] = json.loads(gdf_plot.to_json())
            
            df_mapa_base = pd.DataFrame({
                'Territorio': gdf_plot['NombreComuna'],
                'MATCH_ID': gdf_plot['MATCH_ID'],
                'Total': pd.to_numeric(gdf_plot['Pob_Total'], errors='coerce').fillna(0),
                'Padre': 'Medellín',
                'area_geografica': 'total'
            })
        
        filtro_zona = titulo_terr
        
        # Matemáticas Históricas
        territorio_busqueda = "MEDELLÍN" 
        col_anio_glob = 'Año' if 'Año' in df_global.columns else 'año'
        if not df_global.empty and 'Pob_Medellin' in df_global.columns:
            df_med_hist = df_global.dropna(subset=[col_anio_glob, 'Pob_Medellin']).sort_values(by=col_anio_glob)
            años_hist = df_med_hist[col_anio_glob].values.astype(float)
            pob_med_macro = df_med_hist['Pob_Medellin'].values.astype(float)
        else:
            años_hist = np.arange(1985, 2040).astype(float)
            pob_med_macro = np.linspace(1500000, 2600000, len(años_hist))
            
        pob_total_shape = pd.to_numeric(gdf_med['Pob_Total'], errors='coerce').fillna(0).sum()
        pob_sel_actual = df_mapa_base['Total'].sum()
        factor_proporcional = (pob_sel_actual / pob_total_shape) if pob_total_shape > 0 else 0
        pob_hist = pob_med_macro * factor_proporcional
        
    except Exception as e:
        st.sidebar.error(f"Error cargando datos intra-urbanos: {e}")

# =====================================================================
# 🏙️ ESCALA URBANA (CABECERAS MUNICIPALES - ANTIOQUIA)
# =====================================================================
elif escala_sel == "🏙️ Escala Urbana (Cabeceras Antioquia)":
    st.sidebar.markdown("### 🏙️ Explorador de Cabeceras")
    
    # 1. Filtramos solo la población urbana de Antioquia desde la base DANE
    df_urbano_ant = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['area_geografica'] == 'urbano')].copy()
    
    # 2. Selector de Municipio para aislar la Cabecera
    lista_mpios = sorted(df_urbano_ant['municipio'].dropna().unique())
    mpio_sel = st.sidebar.selectbox("Municipio (Cabecera de):", ["TODAS (Ver Mapa Completo)"] + lista_mpios)
    
    if mpio_sel != "TODAS (Ver Mapa Completo)":
        df_base = df_urbano_ant[df_urbano_ant['municipio'] == mpio_sel]
        filtro_zona = mpio_sel
        titulo_terr = f"Cabecera Urbana de {mpio_sel}"
        
        # Matemáticas Históricas para las gráficas
        col_anio = 'año' if 'año' in df_base.columns else 'Año'
        df_hist = df_base.groupby(col_anio)['Total'].sum().reset_index()
        df_hist = df_hist.sort_values(by=col_anio)
        
        años_hist = df_hist[col_anio].values
        pob_hist = df_hist['Total'].values
        
        # Preparación para el mapa
        df_mapa_base = df_base.copy()
        df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)
    else:
        # Vista de todo el departamento
        filtro_zona = "Antioquia"
        titulo_terr = "Todas las Cabeceras Urbanas (Antioquia)"
        
        # Matemáticas de la suma total urbana
        col_anio = 'año' if 'año' in df_urbano_ant.columns else 'Año'
        df_hist = df_urbano_ant.groupby(col_anio)['Total'].sum().reset_index()
        años_hist = df_hist[col_anio].values
        pob_hist = df_hist['Total'].values
        
        # Preparación para el mapa global
        df_mapa_base = df_urbano_ant.groupby(['municipio', 'depto_nom', col_anio])['Total'].sum().reset_index()
        df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

# =====================================================================
# 🏘️ ESCALA VEREDAL
# =====================================================================

elif escala_sel == "🌿 Veredal (Antioquia)":

    try:
        import pandas as pd
        from modules.db_manager import get_engine
        from sqlalchemy import inspect
        
        engine_sql = get_engine()
        
        inspector = inspect(engine_sql)
        tablas_existentes = inspector.get_table_names()
        
        nombre_tabla_veredas = next((t for t in tablas_existentes if 'vereda' in t.lower() and 'geo' not in t.lower()), None)
        
        if nombre_tabla_veredas:
            df_veredas = pd.read_sql(f'SELECT * FROM "{nombre_tabla_veredas}"', engine_sql)
        else:
            st.sidebar.error(f"🛑 No encontré la tabla poblacional. Tablas: {tablas_existentes}")
            df_veredas = pd.DataFrame()
        
        if not df_veredas.empty:
            df_veredas.columns = df_veredas.columns.str.strip()
            cols_lower = [c.lower() for c in df_veredas.columns]
            
            # --- FIX: SABUESO DE ALTA PRECISIÓN ---
            # Prioridad 1: Nombres exactos (Para evitar el cruce con id-vereda_mpio)
            if 'vereda' in cols_lower:
                col_ver = df_veredas.columns[cols_lower.index('vereda')]
            else:
                col_ver = next((c for c in df_veredas.columns if 'nom_ver' in c.lower() or 'terr' in c.lower()), df_veredas.columns[2] if len(df_veredas.columns)>2 else df_veredas.columns[0])
                
            if 'municipio' in cols_lower:
                col_mun = df_veredas.columns[cols_lower.index('municipio')]
            else:
                col_mun = next((c for c in df_veredas.columns if 'padre' in c.lower() or 'mpio' in c.lower()), df_veredas.columns[1] if len(df_veredas.columns)>1 else df_veredas.columns[0])
                
            col_pob = next((c for c in df_veredas.columns if 'pob' in c.lower() or 'hab' in c.lower() or 'total' in c.lower()), df_veredas.columns[-1])
            
            # 🔥 ESCUDO NUMÉRICO VEREDAL: Transforma cualquier texto a número puro
            df_veredas[col_pob] = pd.to_numeric(df_veredas[col_pob].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').fillna(0)
            
            # Renombramos las columnas
            df_mapa_base = df_veredas.rename(columns={
                col_ver: 'Territorio',
                col_mun: 'Padre',
                col_pob: 'Total'
            })
            
            # 🔥 INYECCIÓN CRÍTICA PARA QUE EL MAPA FUNCIONE
            df_mapa_base['area_geografica'] = 'rural' 
            
            if 'Padre' not in df_mapa_base.columns:
                df_mapa_base['Padre'] = "Antioquia"
            
            lista_mpios = sorted(df_mapa_base['Padre'].dropna().astype(str).unique())
            mpio_sel = st.sidebar.selectbox("📍 Municipio Padre:", ["TODOS (Ver Mapa Completo)"] + lista_mpios)
            
            if mpio_sel != "TODOS (Ver Mapa Completo)":
                # Filtramos la base temporalmente para dejar solo las veredas de este municipio
                df_mapa_base = df_mapa_base[df_mapa_base['Padre'] == mpio_sel]
                
                # --- 🔥 EL SUB-FILTRO VEREDAL RECUPERADO ---
                lista_veredas = sorted(df_mapa_base['Territorio'].dropna().astype(str).unique())
                vereda_sel = st.sidebar.selectbox("🌾 Vereda:", ["TODAS (Ver Municipio)"] + lista_veredas)
                
                if vereda_sel != "TODAS (Ver Municipio)":
                    # Si elige una vereda específica, filtramos al nivel más profundo
                    df_mapa_base = df_mapa_base[df_mapa_base['Territorio'] == vereda_sel]
                    titulo_terr = f"{vereda_sel} ({mpio_sel})"
                else:
                    titulo_terr = f"Veredas de {mpio_sel}"
            else:
                titulo_terr = "Todas las Veredas (Antioquia)"
                
        else:
            df_mapa_base = pd.DataFrame()
            
    except Exception as e:
        st.sidebar.error(f"❌ Error general: {e}")
        df_mapa_base = pd.DataFrame()
    
# =====================================================================
# --- 4. CÁLCULO DE PROYECCIONES (NUEVO PARADIGMA TOP-DOWN) ---
# =====================================================================

# 1. ESCUDO DEFINITIVO ANTI-NAME ERROR
try: _ = filtro_zona
except NameError:
    try: filtro_zona = titulo_terr
    except NameError: filtro_zona = "Colombia"

territorio_busqueda = str(filtro_zona).upper().strip()
if escala_sel == "🇨🇴 Nacional (Colombia)": territorio_busqueda = "COLOMBIA"

# 2. ESCUDO UNIVERSAL ANTI ZIG-ZAGS (Limpia matemática y gráficas)
if len(años_hist) > 0:
    df_clean = pd.DataFrame({'Año': años_hist, 'Pob': pob_hist})
    if df_clean['Pob'].dtype == object:
        df_clean['Pob'] = pd.to_numeric(df_clean['Pob'].astype(str).str.replace(',', ''), errors='coerce')
    else:
        df_clean['Pob'] = pd.to_numeric(df_clean['Pob'], errors='coerce')
        
    df_clean = df_clean.fillna(0)
    df_clean = df_clean[df_clean['Pob'] > 0] 
    df_clean = df_clean.groupby('Año')['Pob'].max().reset_index().sort_values(by='Año') 
    
    x_hist = df_clean['Año'].values.astype(float)
    y_hist = df_clean['Pob'].values.astype(float)
    años_hist = x_hist
    pob_hist = y_hist
else:
    x_hist, y_hist = np.array([]), np.array([])

# 3. Eje X Futuro
col_anio_nac = 'año' if 'año' in df_nac.columns else 'Año'
año_maximo = int(max(df_nac[col_anio_nac].max() if 'df_nac' in locals() and not df_nac.empty else 2100, 2100))
x_proj = np.arange(1950, año_maximo + 1, 1) 

proyecciones = {'Año': x_proj, 'Real': [np.nan]*len(x_proj)}
for i, año in enumerate(x_proj):
    if año in x_hist: proyecciones['Real'][i] = y_hist[np.where(x_hist == año)[0][0]]

# 4. Cargar Matriz Maestra y aplicar Top-Down
if 'df_matriz_demografica' in st.session_state:
    df_matriz = st.session_state['df_matriz_demografica']
else:
    try: 
        # 🔥 AUTORECUPERACIÓN: Buscamos primero en Supabase
        from modules.db_manager import get_engine
        df_matriz = pd.read_sql("SELECT * FROM matriz_maestra_demografica", get_engine())
        st.session_state['df_matriz_demografica'] = df_matriz
    except: 
        try: df_matriz = pd.read_csv(os.path.join(RUTA_RAIZ, "data", "Matriz_Maestra_Demografica.csv"))
        except: df_matriz = pd.DataFrame()

def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))

row_matriz = pd.DataFrame()
modo_calc = "Cálculo en Vivo (Robusto)"

if not df_matriz.empty:
    if escala_sel == "🌿 Veredal (Antioquia)":
        # --- LA MAGIA TOP-DOWN: Heredar la curva Rural del Municipio Padre ---
        if mpio_sel != "TODOS (Ver Mapa Completo)":
            mpio_padre = normalizar_texto(mpio_sel)
            mask = (df_matriz['Nivel'] == 'Municipal') & (df_matriz['Territorio'].apply(normalizar_texto) == mpio_padre) & (df_matriz['Area'] == 'Rural')
            row_matriz = df_matriz[mask]
            modo_calc = f"Matriz Maestra (Top-Down Rural: {mpio_sel})"
        else:
            mask = (df_matriz['Nivel'] == 'Departamental') & (df_matriz['Territorio'].apply(normalizar_texto) == 'ANTIOQUIA') & (df_matriz['Area'] == 'Rural')
            row_matriz = df_matriz[mask]
            modo_calc = "Matriz Maestra (Top-Down Rural: Antioquia)"
            
    elif escala_sel not in ["🌍 Global y Suramérica", "💧 Cuencas Hidrográficas"]:
        # El gráfico ahora busca en la Matriz respetando si elegiste Total, Urbano o Rural
        mask = (df_matriz['Territorio'].astype(str).str.upper().str.strip() == territorio_busqueda) & (df_matriz['Area'].str.lower() == area_global.lower())
        row_matriz = df_matriz[mask]

# --- 5. INYECTAR RESULTADOS ---
x_train, y_train = x_hist, y_hist

if not row_matriz.empty:
    row = row_matriz.iloc[0]
    k_opt = float(str(row['Log_K']).replace('.', '').replace(',', '.')) if isinstance(row['Log_K'], str) else float(row['Log_K'])
    # ... (el resto del código sigue igual)
    a_opt = float(str(row['Log_a']).replace(',', '.'))
    r_opt = float(str(row['Log_r']).replace(',', '.'))
    anio_base = int(row['Año_Base'])
    
    x_proj_norm = x_proj - anio_base
    proyecciones['Logístico'] = f_log(x_proj_norm, k_opt, a_opt, r_opt)
    
    # Inyectar Polinomial y Exponencial si existen
    if 'Poly_A' in row:
        A, B, C, D = row['Poly_A'], row['Poly_B'], row['Poly_C'], row['Poly_D']
        proyecciones['Lineal'] = A*(x_proj_norm**3) + B*(x_proj_norm**2) + C*x_proj_norm + D
    if 'Exp_a' in row:
        proyecciones['Exponencial'] = row['Exp_a'] * np.exp(row['Exp_b'] * x_proj_norm)
        
    param_K, param_r = k_opt, r_opt
else:
    # MODO 2: FALLBACK ROBUSTO
    x_train, y_train = x_hist, y_hist
    x_offset = x_train[0] if len(x_train) > 0 else 1950
    x_train_norm = x_train - x_offset
    x_proj_norm = x_proj - x_offset
    
    try:
        p0_val = max(1, y_train[0] if len(y_train)>0 else 1)
        max_y = max(y_train) if len(y_train)>0 else p0_val
        es_creciente = (y_train[-1] if len(y_train)>0 else p0_val) >= p0_val
        
        # 🔥 EL ESCUDO: Ajuste para absorber la curva de la ONU sin romperse
        k_guess = max_y * 1.2 if es_creciente else max(1, y_train[-1] * 0.95)
        a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
        a_guess = max(-0.999, a_guess) 
        r_guess = 0.02 if es_creciente else -0.02
        
        k_min = max_y * 0.8 if es_creciente else y_train[-1] * 0.5
        limites = ([k_min, -0.999, -0.2], [max_y * 3.0 if es_creciente else max_y * 1.1, np.inf, 0.3])
        
        popt_log, _ = curve_fit(f_log, x_train_norm, y_train, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
        proyecciones['Logístico'] = f_log(x_proj_norm, *popt_log)
        param_K, param_r = popt_log[0], popt_log[2]
        
    except Exception:
        # Si todo falla, mantenemos tu salvavidas
        val_seguro = np.mean(y_train[-3:]) if len(y_train) >= 3 else (y_train[-1] if len(y_train)>0 else 0)
        proyecciones['Logístico'] = np.full(len(x_proj), val_seguro)
        param_K, param_r = "N/A", 0
        
if 'Exponencial' not in proyecciones: proyecciones['Exponencial'] = proyecciones['Logístico']
if 'Lineal' not in proyecciones: proyecciones['Lineal'] = proyecciones['Logístico']
df_proj = pd.DataFrame(proyecciones)

# --- CONFIGURACIÓN DE PESTAÑAS (TABS) ---
tab_modelos, tab_opt, tab_mapas, tab_matriz, tab_rankings, tab_descargas = st.tabs([
    "📈 Tendencia y Estructura", 
    "⚙️ Optimización (Solver)", 
    "🗺️ Mapa Demográfico", 
    "🧠 4. Matriz Maestra Demográfica",
    "📊 Rankings y Dinámica Histórica", 
    "💾 Descargas"
])

# ==========================================
# PESTAÑA 1: MODELOS Y PIRÁMIDES ANIMADAS
# ==========================================
with tab_modelos:
    col_graf, col_param = st.columns([3, 1])
    with col_graf:
        # --- FIX: ESCUDO SUPREMO ANTI-NAME ERROR ---
        titulo_seguro = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio Seleccionado"))
        
        # 🔥 Aplicamos el Expander envolviendo la gráfica
        with st.expander(f"📈 Curvas de Crecimiento Poblacional - {titulo_seguro}", expanded=True):
            fig_curvas = go.Figure()
            fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Logístico'], mode='lines', name='Mod. Logístico', line=dict(color='#10b981', dash='dash')))
            fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Exponencial'], mode='lines', name='Mod. Exponencial', line=dict(color='#f59e0b', dash='dot')))
            fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Lineal'], mode='lines', name='Mod. Lineal', line=dict(color='#6366f1', dash='dot')))
            fig_curvas.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Reales (Censo)', marker=dict(color='#ef4444', size=8, symbol='diamond')))
            fig_curvas.update_layout(hovermode="x unified", xaxis_title="Año", yaxis_title="Habitantes", template="plotly_white")
            st.plotly_chart(fig_curvas, use_container_width=True)
            
    with col_param:
        with st.expander("🧮 Ecuaciones", expanded=True):
            st.latex(r"Log: P(t) = \frac{K}{1 + a \cdot e^{-r \cdot t}}")
            if param_K != "N/A": st.success(f"**K:** {param_K:,.0f} hab.")
            st.latex(r"Exp: P(t) = a \cdot e^{b \cdot t}")
            st.latex(r"Poly_3: P(t) = At^3 + Bt^2 + Ct + D")
            st.latex(r"Lin: P(t) = m \cdot t + b")

    st.sidebar.header("🎯 2. Viaje en el Tiempo")
    # 🔥 FIX: Agregamos "Polinomial" a la lista de opciones para la pirámide
    modelo_sel = st.sidebar.radio("Base de cálculo para la pirámide:", ["Logístico", "Exponencial", "Polinomial", "Lineal", "Dato Real (Si existe)"])
    columna_modelo = "Real" if modelo_sel == "Dato Real (Si existe)" else modelo_sel
    col_anio_pyr = 'año' if 'año' in df_nac.columns else 'Año'
    años_disp = sorted(df_nac[col_anio_pyr].unique())
    año_sel = st.sidebar.select_slider("Selecciona un Año Estático:", options=años_disp, value=2024)

    st.sidebar.markdown("---")
    st.sidebar.subheader("▶️ Animación Temporal")
    velocidad_animacion = st.sidebar.slider("Velocidad (Segundos por cuadro)", 0.1, 2.0, 0.5)
    iniciar_animacion = st.sidebar.button("▶️ Reproducir Evolución", type="primary", use_container_width=True)

    # ==========================================================================
    # 🏛️ SECCIÓN: PIRÁMIDES POBLACIONALES (EL BYPASS HÍBRIDO - OPCIÓN 2)
    # ==========================================================================
    import uuid
    import re
    import unicodedata
    import plotly.graph_objects as go
    import pandas as pd

    def clean_text(s):
        if pd.isna(s): return ""
        return re.sub(r'[^a-z0-9]', ' ', unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8').lower()).strip()

    # 1. Recuperamos el único contexto que sobrevive: El Título
    titulo_seguro = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio Seleccionado"))
    
    st.markdown("---")
    st.subheader(f"📊 Estructura Poblacional Sintética - {titulo_seguro}")

    # 2. Traductor del Área
    area_bruta = str(globals().get('area_global', locals().get('area_global', 'Total'))).lower()
    
    if 'cab' in area_bruta or 'urb' in area_bruta: area_principal = 'Urbano'
    elif 'rur' in area_bruta or 'rest' in area_bruta: area_principal = 'Rural'
    else: area_principal = 'Total'
    
    opciones_basicas = ["Total", "Urbano", "Rural"]
    opciones_filtradas = [opt for opt in opciones_basicas if opt != area_principal]
    
    key_radio = f"comp_{uuid.uuid4().hex[:6]}"
    
    st.info(f"💡 La pirámide izquierda muestra la selección global: **{area_principal}**.")
    zona_comparacion = st.radio(
        f"Selecciona el área para la pirámide de comparación (Derecha):",
        opciones_filtradas, horizontal=True, key=key_radio
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ph_tit_1, ph_graf_1 = st.empty(), st.empty()
        m_c1, m_c2 = st.columns(2)
        ph_m_pob_1, ph_m_hom_1, ph_m_muj_1, ph_m_ind_1 = m_c1.empty(), m_c1.empty(), m_c2.empty(), m_c2.empty()
            
    with col_p2:
        ph_tit_2, ph_graf_2 = st.empty(), st.empty()
        m_c3, m_c4 = st.columns(2)
        ph_m_pob_2, ph_m_hom_2, ph_m_muj_2, ph_m_ind_2 = m_c3.empty(), m_c3.empty(), m_c4.empty(), m_c4.empty()

    df_piramide_final = pd.DataFrame()

    def generar_figura_piramide(año_obj, zona_str):
        try:
            año_num = int(año_obj)
            
            # --- 1. BUSCAR EL TECHO MATEMÁTICO (SQL) ---
            pob_modelo = 0.0
            df_p = globals().get('df_proj', locals().get('df_proj', pd.DataFrame()))
            col_m = globals().get('columna_modelo', locals().get('columna_modelo', ''))
            
            if not df_p.empty and col_m in df_p.columns:
                val_pob = df_p[df_p['Año'].astype(int) == año_num][col_m].values
                if len(val_pob) > 0 and not pd.isna(val_pob[0]):
                    pob_modelo = float(val_pob[0])

            # --- 2. EL RUTEO INTELIGENTE ---
            df_nac_puro = cargar_datos_limpios()[0]
            df_mun_puro = cargar_datos_limpios()[1]
            col_a = 'año' if 'año' in df_nac_puro.columns else 'Año'
            
            z_lim = 'urbano' if 'urb' in zona_str.lower() else 'rural' if 'rur' in zona_str.lower() else 'total'
            tit_cln = clean_text(titulo_seguro)
            es_bypass = False

            if pob_modelo > 0:
                # RUTA A: TOP-DOWN (Modelo SQL Activo - ej. Caribe, Colombia)
                df_base = df_nac_puro[df_nac_puro[col_a].astype(int) == año_num].copy()
                df_f = df_base[df_base['area_geografica'].str.lower() == z_lim].copy()
                
                if df_f.empty and z_lim == 'total': df_f = pd.DataFrame(df_base.sum(numeric_only=True)).T
                df_fnac_tot = df_base[df_base['area_geografica'].str.lower() == 'total']
                if df_fnac_tot.empty: df_fnac_tot = pd.DataFrame(df_base.sum(numeric_only=True)).T
                
                pob_censo_real = float(df_fnac_tot['Total'].values[0]) if not df_fnac_tot.empty else 1.0
                factor_escala = (pob_modelo / pob_censo_real) if pob_censo_real > 0 else 0.0
                
            else:
                # RUTA B: BYPASS HISTÓRICO (Sin Modelo SQL - ej. Amazonía, Cabeceras)
                es_bypass = True
                df_base = df_mun_puro[df_mun_puro[col_a].astype(int) == año_num].copy()
                
                if "cabeceras" in tit_cln and "antioquia" in tit_cln:
                    df_base['dep_cln'] = df_base['depto_nom'].apply(clean_text)
                    df_base = df_base[df_base['dep_cln'] == 'antioquia']
                    z_lim = 'urbano' # Forzamos a urbano
                elif "region" in tit_cln:
                    macro_n = tit_cln.replace("region", "").strip()
                    df_base['mac_cln'] = df_base['Macroregion'].apply(clean_text)
                    df_base = df_base[df_base['mac_cln'] == macro_n]
                else:
                    df_base['mun_cln'] = df_base['municipio'].apply(clean_text)
                    df_base = df_base[df_base['mun_cln'] == tit_cln]
                
                df_base['area_cln'] = df_base['area_geografica'].apply(clean_text)
                df_f = df_base[df_base['area_cln'] == z_lim].copy()
                
                if df_f.empty and z_lim == 'total': df_f = pd.DataFrame(df_base.sum(numeric_only=True)).T
                factor_escala = 1.0 # Usamos los datos crudos sumados, sin escalar
            
            if df_f.empty: return None, 0, 0, 0, 0, f"Datos DANE no disponibles para {zona_str}.", pd.DataFrame(), False

            # --- 3. EXTRACCIÓN Y CÁLCULO ---
            df_agg = pd.DataFrame(df_f.sum(numeric_only=True)).T
            cols_h = [c for c in df_agg.columns if 'Hombre' in str(c) and any(char.isdigit() for char in str(c))]
            cols_m = [c for c in df_agg.columns if 'Mujer' in str(c) and any(char.isdigit() for char in str(c))]
            
            def get_e(t): 
                nums = re.findall(r'\d+', t)
                return int(nums[0]) if nums else 0

            datos = []
            for ch in cols_h:
                ed = get_e(ch)
                vh = float(df_agg[ch].values[0])
                cm = next((c for c in cols_m if get_e(c) == ed), None)
                vm = float(df_agg[cm].values[0]) if cm else 0.0
                datos.append({'Edad': ed, 'Hombres': vh, 'Mujeres': vm})
            
            df_edades = pd.DataFrame(datos)
            
            df_edades['Hom_Terr'] = df_edades['Hombres'] * factor_escala
            df_edades['Muj_Terr'] = df_edades['Mujeres'] * factor_escala
            
            df_pyr = pd.DataFrame({
                'Rango': pd.cut(df_edades['Edad'], bins=list(range(0, 105, 5)) + [200], 
                                labels=[f"{i}-{i+4}" for i in range(0, 100, 5)] + ["100+"], right=False),
                'Hombres': df_edades['Hom_Terr'] * -1,
                'Mujeres': df_edades['Muj_Terr']
            }).groupby('Rango', observed=True).sum().reset_index()

            # --- 4. DIBUJO ---
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df_pyr['Rango'], x=df_pyr['Hombres'], name='Hombres', orientation='h', marker_color='#3498db'))
            fig.add_trace(go.Bar(y=df_pyr['Rango'], x=df_pyr['Mujeres'], name='Mujeres', orientation='h', marker_color='#e74c3c'))
            
            r_max = max(abs(df_pyr['Hombres'].min()), df_pyr['Mujeres'].max()) if not df_pyr.empty else 100
            if pd.isna(r_max) or r_max == 0: r_max = 100
            
            fig.update_layout(barmode='relative', xaxis=dict(range=[-r_max*1.1, r_max*1.1]), height=400, margin=dict(l=0,r=0,t=20,b=0))
            
            t_h, t_m = df_edades['Hom_Terr'].sum(), df_edades['Muj_Terr'].sum()
            return fig, (t_h + t_m), t_h, t_m, (t_h/t_m*100 if t_m > 0 else 0), None, df_pyr, es_bypass
        except Exception as e: return None, 0, 0, 0, 0, f"Error: {e}", pd.DataFrame(), False

    def safe_fmt(val): return f"{int(float(val)):,}".replace(",", ".")

    def refrescar_piramides(anio):
        global df_piramide_final
        
        f1, tot1, h1, m1, ind1, err1, df_ex1, bypass1 = generar_figura_piramide(anio, area_principal)
        if err1: ph_graf_1.warning(err1)
        else:
            df_piramide_final = df_ex1.copy()
            aviso_bp = " ⚠️ *(Censo)*" if bypass1 else ""
            ph_tit_1.markdown(f"#### 🔵 Estructura {area_principal} ({anio}){aviso_bp}")
            ph_graf_1.plotly_chart(f1, use_container_width=True, key=f"p1_{uuid.uuid4().hex[:6]}")
            ph_m_pob_1.metric("Población", safe_fmt(tot1))
            ph_m_hom_1.metric("Hombres", safe_fmt(h1), f"{(h1/tot1*100):.1f}%" if tot1>0 else "0%")
            ph_m_muj_1.metric("Mujeres", safe_fmt(m1), f"{(m1/tot1*100):.1f}%" if tot1>0 else "0%")
            ph_m_ind_1.metric("Índ. Masc.", f"{ind1:.1f}")

        f2, tot2, h2, m2, ind2, err2, _, bypass2 = generar_figura_piramide(anio, zona_comparacion)
        if err2: ph_graf_2.warning(err2)
        else:
            aviso_bp2 = " ⚠️ *(Censo)*" if bypass2 else ""
            ph_tit_2.markdown(f"#### 🟢 Perfil {zona_comparacion} ({anio}){aviso_bp2}")
            ph_graf_2.plotly_chart(f2, use_container_width=True, key=f"p2_{uuid.uuid4().hex[:6]}")
            ph_m_pob_2.metric("Población", safe_fmt(tot2))
            ph_m_hom_2.metric("Hombres", safe_fmt(h2), f"{(h2/tot2*100):.1f}%" if tot2>0 else "0%")
            ph_m_muj_2.metric("Mujeres", safe_fmt(m2), f"{(m2/tot2*100):.1f}%" if tot2>0 else "0%")
            ph_m_ind_2.metric("Índ. Masc.", f"{ind2:.1f}")

    # --- EJECUCIÓN ---
    try: a_sel = int(año_sel)
    except: a_sel = 2025
    
    if globals().get('iniciar_animacion', False):
        for a in años_disp:
            if a >= 1985:
                refrescar_piramides(a)
                import time
                time.sleep(globals().get('velocidad_animacion', 0.5))
    else: refrescar_piramides(a_sel)
        
# --- 7. MARCO METODOLÓGICO Y CONCEPTUAL ---
with st.expander("📚 Marco Conceptual, Metodológico y Matemático", expanded=False):
    st.markdown("""
    ### 1. Conceptos Teóricos
    La dinámica demográfica es el motor fundamental de la planificación hídrica y territorial. Conocer no solo *cuántos* somos, sino la *estructura por edades*, permite proyectar demandas futuras de acueductos, escenarios de presión sobre el recurso hídrico y necesidades de infraestructura.

    ### 2. Metodología de Mapeo Dasimétrico y Asignación Top-Down
    Ante la falta de censos poblacionales continuos en micro-territorios (como veredas), este modelo utiliza una **Estimación Sintética Anidada**:
    * **Paso 1 (Calibración):** Se utiliza la serie censal DANE a nivel municipal (Urbano/Rural) entre 2005 y 2020.
    * **Paso 2 (Dasimetría Veredal):** Se calcula el peso gravitacional de la población de cada vereda respecto a la población rural total de su municipio, asumiendo proporcionalidad espacial ($P_{vereda} = P_{rural\_mpio} \\times \\left( \\frac{P_{base\_vereda}}{\\sum P_{base\_veredas}} \\right)$).
    * **Paso 3 (Anidación Estructural):** Se aplica la "receta" porcentual de la pirámide de edades nacional del año correspondiente, a la masa poblacional calculada del micro-territorio.

    ### 3. Modelos Matemáticos de Ajuste Histórico
    Para viajar en el tiempo (1950-2100), la serie histórica se somete a regresiones no lineales (`scipy.optimize`):
    * **Logístico:** Ideal para poblaciones que alcanzan un techo por límites físicos o recursos. La población crece hasta encontrar resistencia ambiental, estabilizándose en una *Capacidad de Carga* ($K$). Es el modelo más robusto para planeación a largo plazo.
    * **Exponencial:** Asume recursos infinitos. Útil para modelar cortos períodos de "explosión demográfica" en centros urbanos nuevos. Ideal para poblaciones en crecimiento o decrecimiento libre constante.
    * **Polinomial (Grado 3):** Ideal para poblaciones con fluctuaciones o declives no lineales.
    * **Lineal:** Representa tendencias promedio sin aceleración. crecimiento constante.

    ### 4. Fuentes de Información
    * **Capa 1 (Estructura Nacional):** Proyecciones y retroproyecciones oficiales DANE (1950-2070).
    * **Capa 2 (Masa Municipal):** Series censales DANE conciliadas (2005-2020).
    * **Capa 3 (Filtro Veredal):** Base de datos espaciales y tabulares Gobernación de Antioquia / IGAC.
    """)

# ==============================================================================
#  🧠  TRANSMISIÓN AL CEREBRO GLOBAL (EL ALEPH)
# ==============================================================================
if 'año_sel' in locals() and 'escala_sel' in locals():
    # --- ACTUALIZACIÓN DE SINCRONIZACIÓN Y BALANCE DE MASAS ---
    # 1. Definimos un nombre legible y 100% seguro 
    if 'titulo_terr' in locals() and titulo_terr:
        import re
        # Limpiamos prefijos para que el Aleph tenga el nombre puro
        nombre_contexto = re.sub(r'^(Cabecera Urbana de |Veredas de |Región |SZH: |ZH: |AH: |Cuencas Seleccionadas: )', '', str(titulo_terr)).strip()
    elif 'cuenca_sel' in locals() and cuenca_sel:
        nombre_contexto = cuenca_sel[0] # Tomamos la primera si hay varias
    else:
        nombre_contexto = "Antioquia" # Fallback seguro

    # 2. Inyectamos los datos en el Session State para persistencia
    st.session_state['aleph_lugar'] = nombre_contexto
    st.session_state['aleph_escala'] = escala_sel
    st.session_state['aleph_anio'] = año_sel

    # Aseguramos que la población inyectada sea el valor final calculado
    st.session_state['aleph_pob_total'] = float(pob_hist[-1]) if 'pob_hist' in locals() and len(pob_hist) > 0 else 0.0
    
    #  💉  INYECCIÓN AL TORRENTE SANGUÍNEO (Sincronización con modelos Pág 07, 08 y 09)
    poblacion_referencia = 0.0
    if 'pob_hist' in locals() and 'años_hist' in locals() and len(años_hist) > 0:
        try:
            import numpy as np
            idx = np.abs(np.array(años_hist) - año_sel).argmin()
            poblacion_referencia = float(pob_hist[idx])
        except Exception:
            poblacion_referencia = 0.0
            
    st.session_state['pob_hum_calc_met'] = poblacion_referencia
    st.session_state[f'pob_asig_met'] = poblacion_referencia # Quitamos el nombre dinámico para facilitar la lectura global
    
    # 3. Mensaje de éxito limpio en el Sidebar
    st.sidebar.success(f" 🔗  Contexto demográfico de **{nombre_contexto}** sincronizado con éxito.")
    
# ==============================================================================
# TAB 2: MODELOS Y OPTIMIZACIÓN MATEMÁTICA (SOLVER)
# ==============================================================================
with tab_opt:
    st.header("⚙️ Ajuste de Modelos Evolutivos (Solver)")
    
    if len(x_hist) == 0:
        st.info("👆 Selecciona una escala válida en el panel izquierdo.")
    # --- FIX: EMOJI AÑADIDO PARA QUE EL CONDICIONAL FUNCIONE ---
    elif escala_sel == "🌿 Veredal (Antioquia)":
        st.error("❌ La escala Veredal no posee serie de tiempo continua para modelación matemática pura.")
    else:
        col_opt1, col_opt2 = st.columns([1, 2.5])
        with col_opt1:
            st.subheader("Configuración")
            st.success(f"Modelando: **{filtro_zona if 'filtro_zona' in locals() else titulo_terr}**")
            
            # Usamos x_train y y_train (nuestra data curada post-2018) para no dañar la matemática
            t_data_raw = x_train
            p_data = y_train
            t_data = t_data_raw - t_data_raw.min()
            p0_val = float(p_data[0]) 
                
            t_max = st.slider("Años a proyectar (Horizonte):", 10, 100, 30, key='slider_opt')
            st.markdown("---")
            # 🚀 FIX: AÑADIMOS EL MODELO LINEAL A LA LISTA
            modelos_sel = st.multiselect("Curvas a evaluar:", 
                                         ["Exponencial", "Logístico", "Geométrico", "Polinómico (Grado 2)", "Polinómico (Grado 3)", "Lineal"], 
                                         default=["Logístico", "Polinómico (Grado 3)", "Lineal"])
            opt_auto = st.button("✨ Optimizar Parámetros", type="primary", use_container_width=True)

            st.caption("Ajuste Manual:")
            r_man = st.number_input("Tasa (r):", value=0.02, format="%.4f")
            k_man = st.number_input("Capacidad (K):", value=float(p_data.max() * 2.0), step=1000.0)

        with col_opt2:
            def f_exp(t, p0, r): return p0 * np.exp(r * t)
            # --- FÓRMULA LOGÍSTICA CORREGIDA Y ESTABILIZADA ---
            def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))
            def f_geom(t, p0, r): return p0 * (1 + r)**t
            def f_poly2(t, a, b, c): return a*t**2 + b*t + c
            def f_poly3(t, a, b, c, d): return a*t**3 + b*t**2 + c*t + d
            # 🚀 FIX: AÑADIMOS LA FÓRMULA LINEAL
            def f_lin(t, m, b): return m*t + b

            # Calculadora Universal de R2
            def calcular_r2(y_real, y_prediccion):
                ss_res = np.sum((y_real - y_prediccion) ** 2)
                ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            t_total = np.arange(0, max(t_data) + t_max + 1)
            anios_totales = t_total + t_data_raw.min()
            
            import plotly.graph_objects as go
            fig2 = go.Figure()
            # Mostramos TODO el histórico en puntos negros
            fig2.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Históricos', marker=dict(color='black', size=8)))

            res_text = []
            resultados_modelos = {} # Diccionario para guardar métricas y fórmulas
            
            for mod in modelos_sel:
                y_pred = np.zeros_like(t_total, dtype=float)
                try:
                    from scipy.optimize import curve_fit
                    if mod == "Exponencial":
                        if opt_auto: 
                            popt, _ = curve_fit(f_exp, t_data, p_data, p0=[p0_val, 0.01])
                            y_pred = f_exp(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_exp(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = P_0 \cdot e^{r \cdot t}", "vars": ["P_0", "r"]}
                            res_text.append(f"**Exp**: r={popt[1]:.4f}")
                        else: 
                            y_pred = f_exp(t_total, p0_val, r_man)
                            r2_val = calcular_r2(p_data, f_exp(t_data, p0_val, r_man))
                            resultados_modelos[mod] = {"popt": [p0_val, r_man], "r2": r2_val, "latex": r"P(t) = P_0 \cdot e^{r \cdot t}", "vars": ["P_0", "r"]}
                            
                    elif mod == "Logístico":
                        if opt_auto: 
                            k_guess = max(p_data) * 1.5
                            a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                            r_guess = 0.02
                            limites = ([max(p_data), 0, 0.0001], [max(p_data)*5, np.inf, 0.3])
                            
                            popt, _ = curve_fit(f_log, t_data, p_data, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=25000)
                            y_pred = f_log(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_log(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = \frac{K}{1 + a \cdot e^{-r \cdot t}}", "vars": ["K", "a", "r"]}
                            res_text.append(f"**Log**: K={popt[0]:,.0f}, r={popt[2]:.4f}")
                        else: 
                            a_man = (k_man - p0_val) / p0_val if p0_val > 0 else 1
                            y_pred = f_log(t_total, k_man, a_man, r_man)
                            r2_val = calcular_r2(p_data, f_log(t_data, k_man, a_man, r_man))
                            resultados_modelos[mod] = {"popt": [k_man, a_man, r_man], "r2": r2_val, "latex": r"P(t) = \frac{K}{1 + a \cdot e^{-r \cdot t}}", "vars": ["K", "a", "r"]}
                            
                    elif mod == "Geométrico":
                        if opt_auto: 
                            popt, _ = curve_fit(f_geom, t_data, p_data, p0=[p0_val, 0.01])
                            y_pred = f_geom(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_geom(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = P_0 \cdot (1 + r)^t", "vars": ["P_0", "r"]}
                        else: 
                            y_pred = f_geom(t_total, p0_val, r_man)
                            r2_val = calcular_r2(p_data, f_geom(t_data, p0_val, r_man))
                            resultados_modelos[mod] = {"popt": [p0_val, r_man], "r2": r2_val, "latex": r"P(t) = P_0 \cdot (1 + r)^t", "vars": ["P_0", "r"]}
                            
                    elif mod == "Polinómico (Grado 2)":
                        if opt_auto: 
                            popt, _ = curve_fit(f_poly2, t_data, p_data)
                            y_pred = f_poly2(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_poly2(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = a \cdot t^2 + b \cdot t + c", "vars": ["a", "b", "c"]}
                        else: 
                            y_pred = f_poly2(t_total, 1, 10, p0_val)
                            r2_val = calcular_r2(p_data, f_poly2(t_data, 1, 10, p0_val))
                            resultados_modelos[mod] = {"popt": [1, 10, p0_val], "r2": r2_val, "latex": r"P(t) = a \cdot t^2 + b \cdot t + c", "vars": ["a", "b", "c"]}
                        
                    elif mod == "Polinómico (Grado 3)":
                        if opt_auto: 
                            popt, _ = curve_fit(f_poly3, t_data, p_data)
                            y_pred = f_poly3(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_poly3(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = a \cdot t^3 + b \cdot t^2 + c \cdot t + d", "vars": ["a", "b", "c", "d"]}
                        else: 
                            y_pred = f_poly3(t_total, 1, 10, p0_val, 0)
                            r2_val = calcular_r2(p_data, f_poly3(t_data, 1, 10, p0_val, 0))
                            resultados_modelos[mod] = {"popt": [1, 10, p0_val, 0], "r2": r2_val, "latex": r"P(t) = a \cdot t^3 + b \cdot t^2 + c \cdot t + d", "vars": ["a", "b", "c", "d"]}

                    # 🚀 FIX: AÑADIMOS EL CÁLCULO LINEAL
                    elif mod == "Lineal":
                        if opt_auto: 
                            popt, _ = curve_fit(f_lin, t_data, p_data)
                            y_pred = f_lin(t_total, *popt)
                            r2_val = calcular_r2(p_data, f_lin(t_data, *popt))
                            resultados_modelos[mod] = {"popt": popt, "r2": r2_val, "latex": r"P(t) = m \cdot t + b", "vars": ["m", "b"]}
                        else:
                            # Para modo manual: calcula una pendiente simple (m) entre el primer y último año
                            m_man = (p_data[-1] - p_data[0]) / (t_data[-1] - t_data[0]) if t_data[-1] != t_data[0] else 0
                            b_man = p0_val
                            y_pred = f_lin(t_total, m_man, b_man)
                            r2_val = calcular_r2(p_data, f_lin(t_data, m_man, b_man))
                            resultados_modelos[mod] = {"popt": [m_man, b_man], "r2": r2_val, "latex": r"P(t) = m \cdot t + b", "vars": ["m", "b"]}

                    fig2.add_trace(go.Scatter(x=anios_totales, y=y_pred, mode='lines', name=mod, line=dict(width=3, dash='dot' if opt_auto else 'solid')))
                except Exception as e: 
                    pass # Si un modelo matemático falla, simplemente no lo dibuja

            fig2.update_layout(title="Proyección de Modelos Dinámicos", xaxis_title="Año", yaxis_title="Población", hovermode="x unified", height=500)
            st.plotly_chart(fig2, use_container_width=True)
            
        # ----------------------------------------------------------------------
        # NUEVO PANEL: AUDITORÍA DE ECUACIONES Y PARÁMETROS MATEMÁTICOS
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧮 Auditoría Matemática: Ecuaciones y Precisión (R²)")
        
        if resultados_modelos:
            # Encontrar el mejor modelo (el de mayor R2)
            mejor_mod_nombre = max(resultados_modelos, key=lambda k: resultados_modelos[k]['r2'])
            mejor_r2 = resultados_modelos[mejor_mod_nombre]['r2']
            st.info(f"🏆 **El modelo que mejor se ajusta a estos datos históricos es:** {mejor_mod_nombre} (R² = {mejor_r2:.4f})")
            
            # Mostramos las ecuaciones en columnas (máximo 3 por fila para que se vea ordenado)
            cols_eq = st.columns(3)
            idx_col = 0
            
            for nombre_mod, datos_mod in resultados_modelos.items():
                with cols_eq[idx_col % 3]:
                    st.markdown(f"**{nombre_mod}**")
                    st.latex(datos_mod['latex'])
                    
                    # Construir el bloque de código con los parámetros
                    texto_params = f"R² = {datos_mod['r2']:.4f}\n\nParámetros:\n"
                    for variable, valor in zip(datos_mod['vars'], datos_mod['popt']):
                        # Formateamos K (capacidad) sin decimales porque son personas, los demás con 4 decimales
                        if variable == "K":
                            texto_params += f"{variable} = {valor:,.0f}\n"
                        else:
                            texto_params += f"{variable} = {valor:.6f}\n"
                            
                    st.code(texto_params)
                
                idx_col += 1
        else:
            st.caption("Presiona 'Optimizar Parámetros' o ajusta manualmente para visualizar las ecuaciones.")
            
# ==========================================
# PESTAÑA 3: MAPA DEMOGRÁFICO (GEOVISOR ZERO-CONFIG)
# ==========================================
with tab_mapas:
    # 🛡️ ESCUDO 1: Título dinámico seguro
    titulo_seguro_mapa = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio Seleccionado"))
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_seguro_mapa} ({año_sel})")
    
    # 🛡️ ESCUDO 2: Protección para Escala Global
    if escala_sel == "🌍 Global y Suramérica":
        st.info("🌍 A escala Global/Suramérica la visualización espacial se encuentra desactivada. Los datos consolidados están disponibles en el panel de tendencias.")
        
    else:
        # Mini-menú integrado y estético
        col_m1, col_m2, col_m3 = st.columns([1, 1, 3])
        with col_m1:
            # Sincronizamos con el filtro global definido arriba para evitar confusiones
            if escala_sel == "🌿 Veredal (Antioquia)":
                area_mapa = "Rural"
                st.info("ℹ️ Escala veredal: Población rural.")
            elif escala_sel == "🏙️ Escala Urbana (Cabeceras Antioquia)":
                area_mapa = "Urbano"
                st.info("ℹ️ Escala de Cabeceras: Población urbana.")
            else:
                # El usuario puede cambiar el filtro localmente en el mapa
                area_mapa = st.radio("Filtro Poblacional (Mapa):", ["Total", "Urbano", "Rural"], 
                                     index=["Total", "Urbano", "Rural"].index(area_global), # Sincroniza con el sidebar
                                     key="filtro_pob_mapa")
        with col_m2:
            st.markdown("<br>", unsafe_allow_html=True) # Espaciador
            mostrar_capa_cuencas = st.toggle("🌊 Superponer Cuencas", value=False)
                
        with col_m3:
            st.success("🤖 **Motor Topológico Automático:** Conectando capas con precisión administrativa.")

        # 🛡️ ESCUDO 3: LA CURA AL ValueError (max() arg is an empty sequence)
        # Solo calculamos el año si la tabla no está vacía
        if 'año' in df_mapa_base.columns and not df_mapa_base.empty:
            año_maximo = max(df_mapa_base['año'])
            df_mapa_año = df_mapa_base[df_mapa_base['año'] == min(año_maximo, año_sel)].copy()
        elif not df_mapa_base.empty:
            df_mapa_año = df_mapa_base.copy()
        else:
            # Si el territorio no tiene datos para esa CAR o área, queda vacío pero no crashea
            df_mapa_año = pd.DataFrame()

        # 🔥 MOTOR DE FILTRADO Y AGRUPACIÓN (Anti-Duplicados)
        df_mapa_plot = pd.DataFrame()

        if not df_mapa_año.empty:
            # 1. Filtro riguroso por área (Urbano/Rural/Total)
            if 'area_geografica' in df_mapa_año.columns:
                df_mapa_plot = df_mapa_año[df_mapa_año['area_geografica'].str.lower() == area_mapa.lower()].copy()
            else:
                df_mapa_plot = df_mapa_año.copy()
            
            if not df_mapa_plot.empty:
                # 2. Agrupación por Territorio para evitar duplicar por sub-registros
                cols_agrupar = [c for c in ['Territorio', 'Padre', 'MATCH_ID'] if c in df_mapa_plot.columns]
                if cols_agrupar:
                    df_mapa_plot = df_mapa_plot.groupby(cols_agrupar)['Total'].sum().reset_index()
                
                # 3. Limpieza de filas basura "TOTAL" o "CABECERA" que ensucian el mapa
                if 'Territorio' in df_mapa_plot.columns:
                    df_mapa_plot['Territorio'] = df_mapa_plot['Territorio'].astype(str)
                    df_mapa_plot = df_mapa_plot[~df_mapa_plot['Territorio'].str.upper().isin(['TOTAL', 'ANTIOQUIA', 'AMVA'])]
                    
                    if escala_sel == "💧 Cuencas Hidrográficas":
                        df_mapa_plot = df_mapa_plot[~df_mapa_plot['Territorio'].str.contains('CABECERA', case=False, na=False)]

        # 4. ESTANDARIZACIÓN FINAL DE COLUMNAS PARA EL GEOVISOR
        if not df_mapa_plot.empty:
            if 'Territorio' not in df_mapa_plot.columns:
                col_t = next((c for c in df_mapa_plot.columns if c.lower() in ['municipio', 'cuenca', 'vereda', 'nombre', 'subzona', 'nom_nss3']), df_mapa_plot.columns[0])
                df_mapa_plot = df_mapa_plot.rename(columns={col_t: 'Territorio'})
            
            if 'Padre' not in df_mapa_plot.columns:
                col_p = next((c for c in df_mapa_plot.columns if c.lower() in ['padre', 'depto_nom', 'departamento', 'macroregion', 'zona', 'subregion']), None)
                if col_p: df_mapa_plot = df_mapa_plot.rename(columns={col_p: 'Padre'})
                else: df_mapa_plot['Padre'] = ""
            
            if 'Total' not in df_mapa_plot.columns:
                col_tot = next((c for c in df_mapa_plot.columns if c.lower() in ['total', 'poblacion', 'pob', 'habitantes', 'valor']), df_mapa_plot.columns[-1])
                df_mapa_plot = df_mapa_plot.rename(columns={col_tot: 'Total'})
                
            try:
                import json
                import copy
                import plotly.express as px
                
                # =========================================================
                # 🚀 VÍA RÁPIDA (BYPASS): MEDELLÍN INTRA-URBANO
                # =========================================================
                if escala_sel == "🏘️ Escala Intra-Urbana (Medellín)":
                    datos_para_dibujar = df_mapa_plot.copy()
                    mapa_bruto = st.session_state.get('boveda_mapa_medellin', {})
                    mapa_para_dibujar = copy.deepcopy(mapa_bruto)
                    
                    z_fill_val = 4 if nivel_medellin == "Barrios y Corregimientos" else 2
                    prop_key = 'Cod_Barrio' if nivel_medellin == "Barrios y Corregimientos" else 'Cod_Comuna'
                    
                    if 'MATCH_ID' not in datos_para_dibujar.columns and 'MATCH_ID' in df_mapa_base.columns:
                        datos_para_dibujar = pd.merge(datos_para_dibujar, df_mapa_base[['Territorio', 'MATCH_ID']].drop_duplicates(), on='Territorio', how='left')
                        
                    datos_para_dibujar['MATCH_ID'] = datos_para_dibujar['MATCH_ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(z_fill_val)
                    
                    # Diccionario para inyectar nombres reales si no existen en el GeoJSON original
                    dict_comunas_mapa = {
                        "01": "Popular", "02": "Santa Cruz", "03": "Manrique", "04": "Aranjuez",
                        "05": "Castilla", "06": "Doce de Octubre", "07": "Robledo", "08": "Villa Hermosa",
                        "09": "Buenos Aires", "10": "La Candelaria", "11": "Laureles - Estadio", "12": "La América",
                        "13": "San Javier", "14": "El Poblado", "15": "Guayabal", "16": "Belén",
                        "50": "Corregimiento Palmitas", "60": "Corregimiento San Cristóbal", 
                        "70": "Corregimiento Altavista", "80": "Corregimiento San Antonio de Prado", 
                        "90": "Corregimiento Santa Elena"
                    }

                    nombres_reales = {}
                    for f in mapa_para_dibujar.get('features', []):
                        raw_val = str(f['properties'].get(prop_key, '')).replace('.0', '').strip().zfill(z_fill_val)
                        f['id'] = raw_val 
                        
                        # Asignación inteligente del nombre
                        if nivel_medellin == "Barrios y Corregimientos":
                            nombre = f['properties'].get('NombreBarr', f'Territorio {raw_val}')
                        else:
                            nombre = dict_comunas_mapa.get(raw_val, f'Comuna {raw_val}')
                            
                        nombres_reales[raw_val] = nombre
                        
                    datos_para_dibujar['Territorio'] = datos_para_dibujar['MATCH_ID'].map(nombres_reales).fillna(datos_para_dibujar['Territorio'])
                    
                    safe_center_lat, safe_center_lon = 6.2518, -75.5636
                    if titulo_terr in ["Todos los Barrios y Corregimientos", "Todas las Comunas y Corregimientos"]:
                        safe_zoom = 11.0
                    else:
                        safe_zoom = 13.5
                    
                    llave_geojson = 'id'
                    
                # =========================================================
                # 🌍 VÍA LENTA: POSTGIS (Cuencas, Municipios, Veredas)
                # =========================================================
                else:
                    import geopandas as gpd
                    from sqlalchemy import text
                    from modules.db_manager import get_engine
                    
                    engine_geo = get_engine()
                    if "veredal" in escala_sel.lower(): q_geo = text("SELECT * FROM veredas_geometria")
                    elif "cuencas" in escala_sel.lower(): q_geo = text("SELECT * FROM cuencas")
                    else: q_geo = text("SELECT * FROM municipios")
                        
                    gdf_mapa = gpd.read_postgis(q_geo, engine_geo, geom_col="geometry")
                    
                    # 🔥 FIX: Match ID más permisivo para Cuencas
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot.apply(
                        lambda row: normalizar_texto(row['Territorio']) if "cuencas" in escala_sel.lower() 
                        else (normalizar_texto(row['Territorio']) + "_" + normalizar_texto(row['Padre']) if str(row['Padre']).strip() else normalizar_texto(row['Territorio'])), 
                        axis=1
                    )
                    
                    codigos_dane_deptos = { "05": "ANTIOQUIA", "08": "ATLANTICO", "11": "BOGOTA", "13": "BOLIVAR", "15": "BOYACA", "17": "CALDAS", "18": "CAQUETA", "19": "CAUCA", "20": "CESAR", "23": "CORDOBA", "25": "CUNDINAMARCA", "27": "CHOCO", "41": "HUILA", "44": "GUAJIRA", "47": "MAGDALENA", "50": "META", "52": "NARINO", "54": "NORTEDESANTANDER", "63": "QUINDIO", "66": "RISARALDA", "68": "SANTANDER", "70": "SUCRE", "73": "TOLIMA", "76": "VALLEDELCAUCA", "81": "ARAUCA", "85": "CASANARE", "86": "PUTUMAYO", "88": "ARCHIPIELAGODESANANDRES", "91": "AMAZONAS", "94": "GUAINIA", "95": "GUAVIARE", "97": "VAUPES", "99": "VICHADA" }
                    
                    def generar_id_geojson(row):
                        if "cuencas" in escala_sel.lower():
                            cols_posibles = ['nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh', 'nomzh', 'nomah', 'NOM_NSS3', 'NOM_NSS2', 'NOM_NSS1']
                            # 🔥 FIX: Agregamos el fallback exacto 'Cuenca Sin Nombre' para que Urrao no quede invisible
                            val_terr = next((str(row[c]).strip() for c in cols_posibles if c in row and pd.notnull(row[c]) and str(row[c]).strip() not in ["", "None"]), "Cuenca Sin Nombre")
                            
                            if "-" in val_terr: val_terr = val_terr.split("-")[-1]
                            return normalizar_texto(val_terr)
                        elif "veredal" in escala_sel.lower():
                            val_terr = str(row.get('NOMBRE_VER', row.get('nombre_ver', '')))
                            val_padre = str(row.get('NOMB_MPIO', row.get('nomb_mpio', row.get('MPIO_CNMBR', ''))))
                            if val_padre.zfill(2) in codigos_dane_deptos: val_padre = codigos_dane_deptos[val_padre.zfill(2)]
                            return normalizar_texto(val_terr) + "_" + normalizar_texto(val_padre)
                        else:
                            val_terr = str(row.get('MPIO_CNMBR', row.get('mpio_cnmbr', row.get('nombre', ''))))
                            val_padre = str(row.get('DPTO_CCDGO', row.get('dpto_ccdgo', '')))
                            if val_padre.zfill(2) in codigos_dane_deptos: val_padre = codigos_dane_deptos[val_padre.zfill(2)]
                            if normalizar_texto(val_terr) == "MANAUREBALCONDELCESAR": val_terr = "MANAURE"
                            return normalizar_texto(val_terr) + "_" + normalizar_texto(val_padre)

                    gdf_mapa['MATCH_ID'] = gdf_mapa.apply(generar_id_geojson, axis=1)
                    
                    # --- FIX TOPOLÓGICO: Eliminamos el filtro 'padre_norm' que borraba a Cornare ---
                    # No filtramos gdf_mapa por padre, dejamos que el MATCH_ID haga el trabajo.

                    ids_geojson = set(gdf_mapa['MATCH_ID'].dropna().unique())
                    df_mapa_plot['En_Mapa'] = df_mapa_plot['MATCH_ID'].isin(ids_geojson)
                    
                    faltantes_iniciales = df_mapa_plot[df_mapa_plot['En_Mapa'] == False]
                    if not faltantes_iniciales.empty:
                        import difflib
                        ids_geojson_list = list(ids_geojson)
                        for idx, row in faltantes_iniciales.iterrows():
                            umbral = 0.90 if "cuencas" in escala_sel.lower() else 0.85
                            matches = difflib.get_close_matches(row['MATCH_ID'], ids_geojson_list, n=1, cutoff=umbral)
                            if matches:
                                df_mapa_plot.at[idx, 'MATCH_ID'] = matches[0] 
                                df_mapa_plot.at[idx, 'En_Mapa'] = True
                                
                    ids_curados = df_mapa_plot[df_mapa_plot['En_Mapa'] == True]['MATCH_ID'].unique()
                    gdf_filtrado = gdf_mapa[gdf_mapa['MATCH_ID'].isin(ids_curados)]
                    
                    safe_center_lat, safe_center_lon, safe_zoom = 4.57, -74.29, 5
                    if not gdf_filtrado.empty:
                        gdf_4326 = gdf_filtrado.to_crs(epsg=4326)
                        minx, miny, maxx, maxy = gdf_4326.total_bounds
                        safe_center_lon = (minx + maxx) / 2
                        safe_center_lat = (miny + maxy) / 2
                        max_diff = max(maxx - minx, maxy - miny)
                        if max_diff < 0.1: safe_zoom = 11.5
                        elif max_diff < 0.3: safe_zoom = 10
                        elif max_diff < 0.8: safe_zoom = 8.5
                        elif max_diff < 2.5: safe_zoom = 7
                        elif max_diff < 5.0: safe_zoom = 6
                        
                    mapa_para_dibujar = json.loads(gdf_filtrado.to_json())
                    datos_para_dibujar = df_mapa_plot[df_mapa_plot['En_Mapa'] == True].copy()
                    llave_geojson = 'properties.MATCH_ID'

                # =========================================================
                # 🎨 RENDERIZADO UNIFICADO CON CAPAS MÚLTIPLES
                # =========================================================
                if datos_para_dibujar['Total'].sum() == 0:
                    datos_para_dibujar['Color_Fix'] = 1 
                else:
                    datos_para_dibujar['Color_Fix'] = datos_para_dibujar['Total']
                    
                fig_mapa = px.choropleth_mapbox(
                    datos_para_dibujar, 
                    geojson=mapa_para_dibujar,
                    locations='MATCH_ID',        
                    featureidkey=llave_geojson, 
                    color='Color_Fix', 
                    color_continuous_scale="Viridis",
                    mapbox_style="carto-positron",
                    zoom=safe_zoom,
                    center={"lat": safe_center_lat, "lon": safe_center_lon},
                    opacity=0.8,
                    hover_name='Territorio',
                    hover_data={
                        'Color_Fix': False, 
                        'Total': ':,.0f', 
                        'MATCH_ID': False
                    }
                )
                
                # 🔥 AÑADIR CAPA SECUNDARIA DE CUENCAS INTERACTIVA (Vía Rápida con Jerarquía Completa)
                if mostrar_capa_cuencas:
                    try:
                        from sqlalchemy import text
                        from modules.db_manager import get_engine
                        import geopandas as gpd
                        import json
                        import plotly.graph_objects as go
                        
                        engine_geo = get_engine()
                        
                        # 1. Traemos la jerarquía completa. 
                        # Usamos TRIM y NULLIF para capturar vacíos en archivos de Cornare/Corpourabá.
                        q_cue_overlay = text("""
                            SELECT 
                                nomah, nomzh, nom_szh, nom_nss1, nom_nss2, nom_nss3, 
                                geometry 
                            FROM cuencas
                        """)
                        gdf_cue_overlay = gpd.read_postgis(q_cue_overlay, engine_geo, geom_col="geometry")
                        
                        if not gdf_cue_overlay.empty:
                            gdf_cue_overlay = gdf_cue_overlay.to_crs(epsg=4326)
                            
                            # 2. Generamos un ID único por fila para el mapeo de Plotly
                            gdf_cue_overlay['ID_CUE'] = gdf_cue_overlay.index.astype(str)
                            
                            # 3. Limpieza de nombres para el Tooltip (Sanación de Nulos y Espacios)
                            cols_tooltip = ['nomah', 'nomzh', 'nom_szh', 'nom_nss1', 'nom_nss2', 'nom_nss3']
                            for col in cols_tooltip:
                                gdf_cue_overlay[col] = gdf_cue_overlay[col].apply(
                                    lambda x: str(x).strip() if pd.notnull(x) and str(x).strip() not in ["", "None", "nan"] else "No Aplica"
                                )
                            
                            # 4. Construcción del GeoJSON interactivo
                            geojson_cuencas = json.loads(gdf_cue_overlay.to_json())
                            for i, f in enumerate(geojson_cuencas['features']):
                                f['id'] = str(i)
                            
                            # 5. Inyección de la Traza Invisible (Solo contornos y Tooltips)
                            fig_mapa.add_trace(go.Choroplethmapbox(
                                geojson=geojson_cuencas,
                                locations=gdf_cue_overlay['ID_CUE'],
                                z=[0] * len(gdf_cue_overlay), # Valor base para habilitar el hover
                                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], # Transparente
                                marker_line_color='black',
                                marker_line_width=1.5,
                                showscale=False,
                                customdata=gdf_cue_overlay[cols_tooltip],
                                hovertemplate=(
                                    "<b>Microcuenca (NSS3):</b> %{customdata[5]}<br><br>" +
                                    "<b>AH:</b> %{customdata[0]}<br>" +
                                    "<b>ZH:</b> %{customdata[1]}<br>" +
                                    "<b>SZH:</b> %{customdata[2]}<br>" +
                                    "<b>NSS1:</b> %{customdata[3]}<br>" +
                                    "<b>NSS2:</b> %{customdata[4]}<br>" +
                                    "<extra></extra>"
                                )
                            ))
                            
                    except Exception as e:
                        st.sidebar.warning(f"No se pudo superponer la capa de cuencas interactiva: {e}")

                fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
                st.plotly_chart(fig_mapa, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                
                st.success("✅ MAPA RENDERIZADO. Si los datos están en 0, recuerda procesar la matriz en la pestaña 4.")
                
            except Exception as e:
                st.error(f"❌ Error procesando el mapa o conectando a la Base de Datos: {e}")
                
        else:
            st.warning("⚠️ Esperando datos poblacionales del panel lateral...")

# =====================================================================
# PESTAÑA 4: GENERADOR DE MATRIZ MAESTRA (TOP-DOWN) MULTIMODELO CON R²
# =====================================================================
with tab_matriz:
    st.subheader("📍 FASE 1: Cálculo Espacial de Cuencas (Dasimetría)")
    st.info("Calculando distribución poblacional territorial top-down. Este proceso puede tomar varios minutos.")
    
    # 🚀 NUEVO BOTÓN CON BARRA DE PROGRESO INTELIGENTE
    if st.button("⚙️ Iniciar Entrenamiento Masivo de Matriz (Automático)", type="primary", use_container_width=True):
        st.info("🧠 Iniciando motor de Machine Learning. Por favor, no recargues ni cierres la página.")
        barra_progreso = st.progress(0)
        texto_progreso = st.empty()
        
        try:
            import time
            import numpy as np
            import pandas as pd
            from scipy.optimize import curve_fit
            from modules.db_manager import get_engine
            from sqlalchemy import text
            import unicodedata
            import difflib
            import re
            
            engine_sql = get_engine()
            start_time = time.time()
            
            def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))
            def f_exp(t, a, b): return a * np.exp(b * t)
            def calcular_r2(y_real, y_pred):
                ss_res = np.sum((y_real - y_pred) ** 2)
                ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            matriz_resultados = []
            
            def ajustar_modelos(x, y, nivel, territorio, padre, area):
                if len(x) < 4: return 
                x_offset = x[0]
                x_norm = x - x_offset
                p0_val = max(1, y[0])
                max_y = max(y)
                es_creciente = y[-1] >= p0_val
                
                # 1. LOGÍSTICO
                log_k, log_a, log_r, log_r2 = 0, 0, 0, 0
                try:
                    # 🔥 FIX: Límite estricto de crecimiento (Corsé) para evitar explosiones
                    k_max = max_y * 1.5 if es_creciente else max_y * 1.05
                    k_guess = max_y * 1.1 if es_creciente else max(1, y[-1] * 0.95)
                    
                    a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                    a_guess = max(-0.999, a_guess) 
                    r_guess = 0.02 
                    
                    k_min = max_y * 0.8 if es_creciente else y[-1] * 0.5
                    
                    limites = ([k_min, -0.999, 0.0001], [k_max, np.inf, 0.3])
                    
                    popt_log, _ = curve_fit(f_log, x_norm, y, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=10000)
                    log_k, log_a, log_r = popt_log
                    log_r2 = calcular_r2(y, f_log(x_norm, *popt_log))
                except Exception: pass

                # 2. EXPONENCIAL
                exp_a, exp_b, exp_r2 = 0, 0, 0
                try:
                    r_exp_guess = 0.01 if es_creciente else -0.01 # Pista al solver de que la curva baja
                    popt_exp, _ = curve_fit(f_exp, x_norm, y, p0=[p0_val, r_exp_guess], maxfev=10000)
                    exp_a, exp_b = popt_exp
                    exp_r2 = calcular_r2(y, f_exp(x_norm, *popt_exp))
                except Exception: pass

                # 3. POLINOMIAL (Grado 3)
                poly_A, poly_B, poly_C, poly_D, poly_r2 = 0, 0, 0, 0, 0
                try:
                    coefs = np.polyfit(x_norm, y, 3)
                    poly_A, poly_B, poly_C, poly_D = coefs
                    poly_r2 = calcular_r2(y, np.polyval(coefs, x_norm))
                except Exception: pass

                # 🚀 NUEVO: 4. LINEAL (Grado 1)
                lin_m, lin_b, lin_r2 = 0, 0, 0
                try:
                    coefs_lin = np.polyfit(x_norm, y, 1)
                    lin_m, lin_b = coefs_lin
                    lin_r2 = calcular_r2(y, np.polyval(coefs_lin, x_norm))
                except Exception: pass

                # ⚖️ JUEZ ACTUALIZADO: MEJOR MODELO (Incluye al Lineal)
                dic_modelos = {'Logístico': log_r2, 'Exponencial': exp_r2, 'Polinomial_3': poly_r2, 'Lineal': lin_r2}
                mejor_modelo = max(dic_modelos, key=dic_modelos.get)
                mejor_r2 = dic_modelos[mejor_modelo]

                matriz_resultados.append({
                    'Area': area, 'Nivel': nivel, 'Territorio': territorio, 'Padre': padre,
                    'Año_Base': int(x_offset), 'Pob_Base': round(p0_val, 0),
                    'Log_K': log_k, 'Log_a': log_a, 'Log_r': log_r, 'Log_R2': round(log_r2, 4),
                    'Exp_a': exp_a, 'Exp_b': exp_b, 'Exp_R2': round(exp_r2, 4),
                    'Poly_A': poly_A, 'Poly_B': poly_B, 'Poly_C': poly_C, 'Poly_D': poly_D, 'Poly_R2': round(poly_r2, 4),
                    'Lin_m': lin_m, 'Lin_b': lin_b, 'Lin_R2': round(lin_r2, 4), # <-- INYECCIÓN DE LOS PARÁMETROS LINEALES
                    'Modelo_Recomendado': mejor_modelo, 'Mejor_R2': round(mejor_r2, 4)
                })
                
            df_mun_memoria = df_mun.copy() 
            col_anio = 'año' if 'año' in df_mun_memoria.columns else 'Año'
            
            def clasificar_area(val):
                v = str(val).lower()
                if 'total' in v: return 'Total'
                if 'cabecera' in v or 'urban' in v: return 'Urbana'
                if 'rural' in v or 'centros' in v or 'resto' in v: return 'Rural'
                return 'Desconocido'
                
            df_mun_memoria['Categoria_Area'] = df_mun_memoria['area_geografica'].apply(clasificar_area)
            
            # 🔥 LA CURA A ABEJORRAL INTELIGENTE (Sella la fuga de los 272k):
            mpios_con_total = df_mun_memoria[df_mun_memoria['Categoria_Area'] == 'Total']['municipio'].unique()
            mpios_necesitan_total = df_mun_memoria[~df_mun_memoria['municipio'].isin(mpios_con_total)]['municipio'].unique()
            
            if len(mpios_necesitan_total) > 0:
                df_faltantes = df_mun_memoria[df_mun_memoria['municipio'].isin(mpios_necesitan_total)]
                df_urb_rur = df_faltantes[df_faltantes['Categoria_Area'].isin(['Urbana', 'Rural'])]
                
                if not df_urb_rur.empty:
                    cols_suma = [c for c in df_mun_memoria.columns if c not in ['municipio', 'depto_nom', col_anio, 'Categoria_Area', 'area_geografica', 'Macroregion']]
                    df_totales_calc = df_urb_rur.groupby(['depto_nom', 'municipio', col_anio])[cols_suma].sum().reset_index()
                    df_totales_calc['Categoria_Area'] = 'Total'
                    df_totales_calc['area_geografica'] = 'total'
                    df_mun_memoria = pd.concat([df_mun_memoria, df_totales_calc], ignore_index=True)
                    
            # --- 🕵️‍♂️ RECOLECCIÓN MASIVA DE CUENCAS DE LA BASE DE DATOS ---
            q_todas = text("SELECT DISTINCT nom_nss3 FROM cuencas WHERE nom_nss3 IS NOT NULL")
            lista_todas_cuencas = pd.read_sql(q_todas, engine_sql)['nom_nss3'].tolist()
            
            # --- ⚙️ MOTOR DE PROGRESO UI ---
            mpios = df_mun_memoria['municipio'].dropna().unique()
            deptos = df_mun_memoria['depto_nom'].dropna().unique()
            areas_a_procesar = ['Total', 'Urbana', 'Rural']
            
            total_ops = len(areas_a_procesar) * (1 + len(deptos) + len(mpios) + len(lista_todas_cuencas))
            ops_completadas = 0
            historico_cuencas = [] # 🔥 NUEVO: Recolector maestro de fragmentos para balancear cuencas

            for tipo_area in areas_a_procesar:
                df_area_actual = df_mun_memoria[df_mun_memoria['Categoria_Area'] == tipo_area].copy()
                
                # 🚨 SELLO DE BALANCE MAESTRO: Si un área está vacía, bloqueamos el motor
                # Esto evita el "Fantasma de la Inflación" de 9.1M
                if df_area_actual.empty: 
                    st.error(f"🚨 Error Crítico: No se encontró población en el DANE para la categoría '{tipo_area}'. Revisa la normalización.")
                    st.stop()
                
                # 1. Nacional
                df_nac_temp = df_area_actual.groupby(col_anio)['Total'].sum().reset_index().sort_values(by=col_anio)
                ajustar_modelos(df_nac_temp[col_anio].values, df_nac_temp['Total'].values, 'Nacional', 'Colombia', 'Mundo', tipo_area)
                ops_completadas += 1

                # 2. Departamental
                df_deptos = df_area_actual.groupby(['depto_nom', col_anio])['Total'].sum().reset_index()
                for depto in deptos:
                    df_temp = df_deptos[df_deptos['depto_nom'] == depto].sort_values(by=col_anio)
                    if not df_temp.empty: ajustar_modelos(df_temp[col_anio].values, df_temp['Total'].values, 'Departamental', depto, 'Colombia', tipo_area)
                    ops_completadas += 1

                # 3. Municipal
                df_mpios = df_area_actual.groupby(['municipio', 'depto_nom', col_anio])['Total'].sum().reset_index()
                for mpio in mpios:
                    df_temp = df_mpios[df_mpios['municipio'] == mpio].sort_values(by=col_anio)
                    if not df_temp.empty: ajustar_modelos(df_temp[col_anio].values, df_temp['Total'].values, 'Municipal', mpio, df_temp['depto_nom'].iloc[0], tipo_area)
                    ops_completadas += 1
                    
                    if ops_completadas % 10 == 0:
                        porcentaje = min(ops_completadas / total_ops, 1.0)
                        barra_progreso.progress(porcentaje)
                        elapsed = time.time() - start_time
                        eta = max((elapsed / ops_completadas) * total_ops - elapsed, 0)
                        mins, secs = divmod(int(eta), 60)
                        texto_progreso.markdown(f"**Procesando Base Administrativa:** {mpio} ({tipo_area})... | **ETA:** {mins}m {secs}s")

                # ================================================================
                # 🧠 BISTURÍ ESPACIAL V6: Hiper-Resolución (Barrios + Gravedad)
                # ================================================================
                try:
                    if tipo_area in ['Urbana', 'Rural']:
                        texto_progreso.markdown(f"🗺️ **Fase {tipo_area}:** Descargando mapas y preparando geoprocesamiento (Paciencia, proceso pesado...)")
                        
                        import geopandas as gpd
                        from sqlalchemy import text
                        import unicodedata
                        import difflib
                        import re
                        from modules.db_manager import get_engine
                        import gc # Recolector de basura para liberar RAM
                        
                        engine_geo = get_engine()
                        
                        URL_CABECERAS = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/geojson/CabeceraMunicipal_GisAnt_PT.geojson"
                        URL_CENTROS_POBLADOS = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/geojson/CentrosPoblados_GisAnt_PT.geojson"
                        URL_BARRIOS_MED = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/geojson/PoblacionBarrioCorregimiento_optimizado.geojson"
                        
                        q_cue = text("""
                            SELECT COALESCE(
                                NULLIF(TRIM(nom_nss3), ''), NULLIF(TRIM(nom_nss2), ''), NULLIF(TRIM(nom_nss1), ''), 
                                NULLIF(TRIM(nom_szh), ''), NULLIF(TRIM(nomzh), ''), NULLIF(TRIM(nomah), ''), 'Cuenca Sin Nombre'
                            ) AS subc_lbl, geometry 
                            FROM cuencas
                        """)
                        gdf_cue = gpd.read_postgis(q_cue, engine_geo, geom_col="geometry").to_crs(epsg=3116)
                        
                        # 🔥 OPTIMIZACIÓN RAM 1: Dejamos solo lo esencial de las cuencas
                        gdf_cue_limpio = gdf_cue[['subc_lbl', 'geometry']].copy()
                        gdf_cue_limpio['geometry'] = gdf_cue_limpio.geometry.buffer(0)
                        
                        def cargar_y_proyectar(url):
                            temp_gdf = gpd.read_file(url)
                            if temp_gdf.crs is None: temp_gdf = temp_gdf.set_crs(epsg=4326)
                            return temp_gdf.to_crs(epsg=3116)

                        def clean_v6(t):
                            if not t or pd.isna(t): return ""
                            t = str(t).lower().strip()
                            t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
                            return re.sub(r'[^a-z0-9]', '', t)

                        # --- 1. MEDELLÍN (Se cruza en ambas fases) ---
                        texto_progreso.markdown(f"🗺️ **Fase {tipo_area}:** Cruzando Barrios de Medellín con Cuencas...")
                        gdf_barrios = cargar_y_proyectar(URL_BARRIOS_MED)
                        gdf_barrios['Pob_Total'] = pd.to_numeric(gdf_barrios['Pob_Total'], errors='coerce').fillna(0)
                        gdf_barrios['geometry'] = gdf_barrios.geometry.buffer(0)
                        gdf_barrios = gdf_barrios[['Cod_Barrio', 'Pob_Total', 'geometry']] # Saver RAM
                        
                        inter_b = gpd.overlay(gdf_barrios, gdf_cue_limpio, how='intersection')
                        if not inter_b.empty:
                            inter_b['area_inter'] = inter_b.geometry.area
                            inter_b = inter_b[inter_b['area_inter'] > 100].copy()
                            suma_areas_b = inter_b.groupby('Cod_Barrio')['area_inter'].transform('sum')
                            inter_b['pct_area'] = inter_b['area_inter'] / suma_areas_b
                            inter_b['pob_frag'] = inter_b['Pob_Total'] * inter_b['pct_area']
                        else:
                            inter_b = pd.DataFrame(columns=['Cod_Barrio', 'subc_lbl', 'pob_frag'])
                            
                        barrios_in = inter_b['Cod_Barrio'].unique() if not inter_b.empty else []
                        barrios_out = gdf_barrios[~gdf_barrios['Cod_Barrio'].isin(barrios_in)].copy()
                        if not barrios_out.empty:
                            barrios_out['geometry'] = barrios_out.geometry.centroid
                            rescate_b = gpd.sjoin_nearest(barrios_out, gdf_cue_limpio, how='inner')
                            if not rescate_b.empty:
                                rescate_b = rescate_b.drop_duplicates(subset=['Cod_Barrio']) 
                                rescate_b['pob_frag'] = rescate_b['Pob_Total'] 
                                inter_b = pd.concat([inter_b, rescate_b[['Cod_Barrio', 'subc_lbl', 'pob_frag']]], ignore_index=True)

                        pesos_med_pct = {}
                        if not inter_b.empty:
                            pesos_med = inter_b.groupby('subc_lbl')['pob_frag'].sum()
                            pesos_med_pct = pesos_med / pesos_med.sum() if pesos_med.sum() > 0 else {}

                        # Limpieza
                        del gdf_barrios; gc.collect()

                        # --- 2. URBANO Y RURAL SELECTIVO (La cura contra la asfixia de memoria) ---
                        inter_urbana = pd.DataFrame()
                        cp_en_cuenca = pd.DataFrame()
                        inter_dispersa = pd.DataFrame()

                        if tipo_area == 'Urbana':
                            texto_progreso.markdown(f"🗺️ **Fase Urbana:** Cruzando Cabeceras Municipales con Cuencas...")
                            gdf_cab = cargar_y_proyectar(URL_CABECERAS)
                            col_cab = 'MPIO_NOMBR' if 'MPIO_NOMBR' in gdf_cab.columns else 'mpio_nombr'
                            gdf_cab['mun_norm'] = gdf_cab[col_cab].apply(clean_v6)
                            gdf_cab['geometry'] = gdf_cab.geometry.buffer(0)
                            gdf_cab = gdf_cab[['mun_norm', 'geometry']] # Saver RAM
                            
                            inter_u = gpd.overlay(gdf_cab, gdf_cue_limpio, how='intersection')
                            if not inter_u.empty:
                                inter_u['area_inter'] = inter_u.geometry.area
                                inter_u = inter_u[inter_u['area_inter'] > 1000].copy()
                                suma_areas_u = inter_u.groupby('mun_norm')['area_inter'].transform('sum')
                                inter_u['pct_area_urb'] = inter_u['area_inter'] / suma_areas_u
                            else:
                                inter_u = pd.DataFrame(columns=['mun_norm', 'subc_lbl', 'pct_area_urb'])
                                
                            mpios_u_in = inter_u['mun_norm'].unique() if not inter_u.empty else []
                            mpios_u_out = gdf_cab[~gdf_cab['mun_norm'].isin(mpios_u_in)].copy()
                            if not mpios_u_out.empty:
                                mpios_u_out['geometry'] = mpios_u_out.geometry.centroid
                                rescate_u = gpd.sjoin_nearest(mpios_u_out, gdf_cue_limpio, how='inner')
                                if not rescate_u.empty:
                                    rescate_u = rescate_u.drop_duplicates(subset=['mun_norm'])
                                    rescate_u['pct_area_urb'] = 1.0 
                                    inter_u = pd.concat([inter_u, rescate_u[['mun_norm', 'subc_lbl', 'pct_area_urb']]], ignore_index=True)
                            inter_urbana = inter_u 
                            del gdf_cab; gc.collect()

                        if tipo_area == 'Rural':
                            texto_progreso.markdown(f"🗺️ **Fase Rural 1/2:** Cruzando Centros Poblados...")
                            gdf_cp = cargar_y_proyectar(URL_CENTROS_POBLADOS)
                            col_cp = 'NOMBRE_MPI' if 'NOMBRE_MPI' in gdf_cp.columns else 'nombre_mpi'
                            gdf_cp['mun_norm'] = gdf_cp[col_cp].apply(clean_v6)
                            gdf_cp['geometry'] = gdf_cp.geometry.buffer(0)
                            gdf_cp['id_unico_cp'] = gdf_cp.index.astype(str)
                            gdf_cp = gdf_cp[['mun_norm', 'id_unico_cp', 'geometry']] # Saver RAM
                            
                            inter_cp = gpd.overlay(gdf_cp, gdf_cue_limpio, how='intersection')
                            if not inter_cp.empty:
                                inter_cp['area_inter'] = inter_cp.geometry.area
                                inter_cp = inter_cp[inter_cp['area_inter'] > 500].copy()
                                inter_cp = inter_cp.sort_values('area_inter', ascending=False).drop_duplicates(subset=['id_unico_cp'])
                            else:
                                inter_cp = pd.DataFrame(columns=['mun_norm', 'id_unico_cp', 'subc_lbl'])

                            cp_in = inter_cp['id_unico_cp'].unique() if not inter_cp.empty else []
                            cp_out = gdf_cp[~gdf_cp['id_unico_cp'].isin(cp_in)].copy()
                            if not cp_out.empty:
                                cp_out['geometry'] = cp_out.geometry.centroid
                                rescate_cp = gpd.sjoin_nearest(cp_out, gdf_cue_limpio, how='inner')
                                if not rescate_cp.empty:
                                    rescate_cp = rescate_cp.drop_duplicates(subset=['id_unico_cp'])
                                    inter_cp = pd.concat([inter_cp, rescate_cp[['mun_norm', 'id_unico_cp', 'subc_lbl']]], ignore_index=True)
                            cp_en_cuenca = inter_cp 
                            del gdf_cp; gc.collect()

                            texto_progreso.markdown(f"🗺️ **Fase Rural 2/2:** Intersección Masiva de Polígonos Municipales (La operación más pesada ⚠️)...")
                            gdf_mun = gpd.read_postgis(text("SELECT * FROM municipios"), engine_geo, geom_col="geometry")
                            col_dpto = 'dpto_ccdgo' if 'dpto_ccdgo' in gdf_mun.columns else 'DPTO_CCDGO'
                            if col_dpto in gdf_mun.columns: gdf_mun = gdf_mun[gdf_mun[col_dpto] == '05'].copy()
                            gdf_mun = gdf_mun.to_crs(epsg=3116)
                            col_mun = 'mpio_cnmbr' if 'mpio_cnmbr' in gdf_mun.columns else 'MPIO_CNMBR'
                            gdf_mun['mun_norm'] = gdf_mun[col_mun].apply(clean_v6)
                            gdf_mun['geometry'] = gdf_mun.geometry.buffer(0)
                            gdf_mun = gdf_mun[['mun_norm', 'geometry']] # Saver RAM Absoluto
                            
                            inter_r = gpd.overlay(gdf_mun, gdf_cue_limpio, how='intersection')
                            if not inter_r.empty:
                                inter_r['area_inter'] = inter_r.geometry.area
                                inter_r = inter_r[inter_r['area_inter'] > 10000].copy() 
                                suma_areas_r = inter_r.groupby('mun_norm')['area_inter'].transform('sum')
                                inter_r['pct_area_rur'] = inter_r['area_inter'] / suma_areas_r
                            else:
                                inter_r = pd.DataFrame(columns=['mun_norm', 'subc_lbl', 'pct_area_rur'])
                                
                            mpios_r_in = inter_r['mun_norm'].unique() if not inter_r.empty else []
                            mpios_r_out = gdf_mun[~gdf_mun['mun_norm'].isin(mpios_r_in)].copy()
                            if not mpios_r_out.empty:
                                mpios_r_out['geometry'] = mpios_r_out.geometry.centroid
                                rescate_r = gpd.sjoin_nearest(mpios_r_out, gdf_cue_limpio, how='inner')
                                if not rescate_r.empty:
                                    rescate_r = rescate_r.drop_duplicates(subset=['mun_norm'])
                                    rescate_r['pct_area_rur'] = 1.0 
                                    inter_r = pd.concat([inter_r, rescate_r[['mun_norm', 'subc_lbl', 'pct_area_rur']]], ignore_index=True)
                            inter_dispersa = inter_r 
                            del gdf_mun; gc.collect()

                        # --- 3. DEDUCCIÓN DE FRAGMENTOS MATEMÁTICA (Bypass RAM 2.0) ---
                        texto_progreso.markdown(f"🧮 **Fase {tipo_area}:** Extrayendo fracciones poblacionales (Optimizando Memoria)...")
                        df_area_v6 = df_area_actual[df_area_actual['depto_nom'].str.upper() == 'ANTIOQUIA'].copy()
                        df_area_v6['mun_norm_dane'] = df_area_v6['municipio'].apply(clean_v6)
                        
                        agregados_fantasma = ['valledeaburra', 'areametropolitana', 'total', 'antioquia']
                        df_area_v6 = df_area_v6[~df_area_v6['mun_norm_dane'].str.contains('|'.join(agregados_fantasma))]
                        
                        mpios_mapa_lista = []
                        if tipo_area == 'Urbana' and not inter_urbana.empty: mpios_mapa_lista = inter_urbana['mun_norm'].tolist()
                        elif tipo_area == 'Rural' and not cp_en_cuenca.empty: mpios_mapa_lista = cp_en_cuenca['mun_norm'].tolist() 
                            
                        if mpios_mapa_lista:
                            mpios_mapa = set(mpios_mapa_lista)
                            
                            # 🔥 FIX RAM 1: Difflib a velocidad luz. (Calcula 125 veces en lugar de 7000)
                            unicos_dane = df_area_v6['mun_norm_dane'].unique()
                            map_nombres = {}
                            for m in unicos_dane:
                                if m in mpios_mapa: map_nombres[m] = m
                                else:
                                    matches = difflib.get_close_matches(m, mpios_mapa, n=1, cutoff=0.8)
                                    map_nombres[m] = matches[0] if matches else m
                                    
                            df_area_v6['mun_norm_dane'] = df_area_v6['mun_norm_dane'].map(map_nombres)
                        
                        df_area_v6 = df_area_v6.groupby(['mun_norm_dane', col_anio])['Total'].sum().reset_index()

                        nombre_real_aburra = next((c for c in lista_todas_cuencas if 'aburra' in str(c).lower() or 'aburrá' in str(c).lower()), 'Rio Aburra')
                        nombre_real_leon = next((c for c in lista_todas_cuencas if 'leon' in str(c).lower() or 'león' in str(c).lower()), 'Rio Leon')

                        df_final_cuencas = []
                        mpios_amva_rescate = ['medellin', 'bello', 'itagui', 'envigado', 'sabaneta', 'copacabana', 'laestrella', 'girardota', 'caldas', 'barbosa']
                        
                        for mpio in df_area_v6['mun_norm_dane'].unique():
                            # 🔥 FIX RAM 2: Copiamos SOLO Año y Total. Reducción del 95% del peso.
                            pob_mpio = df_area_v6[df_area_v6['mun_norm_dane'] == mpio][[col_anio, 'Total']].copy()
                            
                            if pob_mpio.empty: continue
                            fallback_basin = nombre_real_leon if mpio in ['apartado', 'turbo', 'carepa', 'necocli', 'sanjuan'] else nombre_real_aburra
                            
                            # Mini-función inyectora ligera
                            def agregar_fragmento(df_pob, cuenca_lbl, factor):
                                df_temp = df_pob.copy()
                                df_temp['Total_frag'] = df_temp['Total'] * factor
                                df_temp['subc_lbl'] = cuenca_lbl
                                df_final_cuencas.append(df_temp)
                            
                            if mpio in mpios_amva_rescate:
                                # 🔥 EL FIX DEFINITIVO: Usamos solo len() para evitar la ambigüedad de la Serie
                                if mpio == 'medellin' and len(pesos_med_pct) > 0:
                                    for subc, peso in pesos_med_pct.items():
                                        agregar_fragmento(pob_mpio, subc, float(peso))
                                else:
                                    agregar_fragmento(pob_mpio, nombre_real_aburra, 1.0)
                            else:
                                if tipo_area == 'Urbana':
                                    cuencas_urb = inter_urbana[inter_urbana['mun_norm'] == mpio] if not inter_urbana.empty else pd.DataFrame()
                                    if not cuencas_urb.empty:
                                        sum_u = float(cuencas_urb['pct_area_urb'].sum())
                                        if sum_u > 0:
                                            for _, u_row in cuencas_urb.iterrows():
                                                agregar_fragmento(pob_mpio, u_row['subc_lbl'], float(u_row['pct_area_urb']) / sum_u)
                                        else:
                                            agregar_fragmento(pob_mpio, fallback_basin, 1.0)
                                    else:
                                        agregar_fragmento(pob_mpio, fallback_basin, 1.0)
                                        
                                elif tipo_area == 'Rural':
                                    cuencas_cp = cp_en_cuenca[cp_en_cuenca['mun_norm'] == mpio] if not cp_en_cuenca.empty else pd.DataFrame()
                                    cuencas_area = inter_dispersa[inter_dispersa['mun_norm'] == mpio] if not inter_dispersa.empty else pd.DataFrame()
                                    
                                    if not cuencas_cp.empty and not cuencas_area.empty:
                                        len_cp = len(cuencas_cp)
                                        for _, cp_row in cuencas_cp.iterrows():
                                            agregar_fragmento(pob_mpio, cp_row['subc_lbl'], 0.30 / len_cp)
                                            
                                        sum_r = float(cuencas_area['pct_area_rur'].sum())
                                        for _, a_row in cuencas_area.iterrows():
                                            factor = 0.70 * (float(a_row['pct_area_rur']) / sum_r) if sum_r > 0 else 0.70
                                            agregar_fragmento(pob_mpio, a_row['subc_lbl'], factor)
                                            
                                    elif not cuencas_cp.empty:
                                        len_cp = len(cuencas_cp)
                                        for _, cp_row in cuencas_cp.iterrows():
                                            agregar_fragmento(pob_mpio, cp_row['subc_lbl'], 1.0 / len_cp)
                                            
                                    elif not cuencas_area.empty:
                                        sum_r = float(cuencas_area['pct_area_rur'].sum())
                                        for _, a_row in cuencas_area.iterrows():
                                            factor = (float(a_row['pct_area_rur']) / sum_r) if sum_r > 0 else 1.0
                                            agregar_fragmento(pob_mpio, a_row['subc_lbl'], factor)
                                    else:
                                        agregar_fragmento(pob_mpio, fallback_basin, 1.0)

                        # 5. RECOLECCIÓN DE FRAGMENTOS (Sin entrenar todavía)
                        if df_final_cuencas:
                            df_cuencas_v6 = pd.concat(df_final_cuencas).groupby(['subc_lbl', col_anio])['Total_frag'].sum().reset_index()
                            df_cuencas_v6['Categoria_Area'] = tipo_area
                            historico_cuencas.append(df_cuencas_v6)
                            
                            # 🔥 FIX RAM 3: Purgamos la lista gigante de memoria manualmente
                            del df_final_cuencas; import gc; gc.collect()
                            
                except Exception as e:
                    st.error(f"❌ Error en Motor V6: {e}")

            # =====================================================================
            # 🔥 FASE 2 AUTOMÁTICA: ENTRENAMIENTO MULTIESCALA DE CUENCAS (HIDRO)
            # =====================================================================
            if historico_cuencas:
                texto_progreso.markdown("⚖️ **Fase 2: Ensamblando balance de masas jerárquico para Cuencas...**")
                
                # 1. Recuperamos y limpiamos fragmentos base
                df_hist_base = pd.concat(historico_cuencas)
                df_hist_base['Total_frag'] = df_hist_base['Total_frag'].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 2. Generamos totales 'Total' para las Microcuencas Base (NSS3)
                df_hist_totales = df_hist_base.groupby(['subc_lbl', col_anio])['Total_frag'].sum().reset_index()
                df_hist_totales['Categoria_Area'] = 'Total'
                df_hist_micro = pd.concat([df_hist_base, df_hist_totales], ignore_index=True)
                
                # 3. Descargamos el árbol genealógico completo de la DB
                try:
                    q_jerarquia = text("""
                        SELECT DISTINCT 
                            nomah, nomzh, nom_szh, nom_nss1, nom_nss2, nom_nss3,
                            COALESCE(
                                NULLIF(TRIM(nom_nss3), ''), NULLIF(TRIM(nom_nss2), ''), NULLIF(TRIM(nom_nss1), ''), 
                                NULLIF(TRIM(nom_szh), ''), NULLIF(TRIM(nomzh), ''), NULLIF(TRIM(nomah), ''), 'Cuenca Sin Nombre'
                            ) AS subc_lbl
                        FROM cuencas
                    """)
                    df_arbol = pd.read_sql(q_jerarquia, engine_sql)
                    for c in df_arbol.columns:
                        df_arbol[c] = df_arbol[c].astype(str).str.strip()
                except Exception as e:
                    st.error(f"Error cargando árbol hídrico: {e}")
                    df_arbol = pd.DataFrame(columns=['subc_lbl', 'nomah', 'nomzh', 'nom_szh', 'nom_nss1', 'nom_nss2', 'nom_nss3'])
                
                # 4. FUSIÓN MAESTRA (Match con Aplanadora de Texto)
                # Sincronizamos nombres para que 'Río Aburrá' (mapa) coincida con 'Rio Aburra' (censo)
                df_hist_micro['MATCH_ID'] = df_hist_micro['subc_lbl'].apply(normalizar_texto)
                df_arbol['MATCH_ID'] = df_arbol['subc_lbl'].apply(normalizar_texto)
                
                df_arbol_unico = df_arbol.drop_duplicates(subset=['MATCH_ID']).copy()
                df_hist_completo = pd.merge(df_hist_micro, df_arbol_unico, on='MATCH_ID', how='left')
                
                # 5. Entrenamiento de modelos Top-Down (Cascada Hidrológica)
                niveles_hidrologicos = {
                    'nomah': 'AH', 'nomzh': 'ZH', 'nom_szh': 'SZH',
                    'nom_nss1': 'NSS1', 'nom_nss2': 'NSS2', 'nom_nss3': 'NSS3', 'subc_lbl_x': 'Microcuenca'
                }
                
                texto_progreso.markdown("🧠 **Fase 3: Entrenando Modelos Matemáticos para Cuencas...**")
                entrenados_log = set()
                
                for col_nivel, etiqueta_padre in niveles_hidrologicos.items():
                    if col_nivel in df_hist_completo.columns:
                        # Agrupamos población por el nivel hídrico específico, área y año
                        df_nivel = df_hist_completo.dropna(subset=[col_nivel]).groupby([col_nivel, 'Categoria_Area', col_anio])['Total_frag'].sum().reset_index()
                        
                        for area_c in ['Total', 'Urbana', 'Rural']:
                            df_area_c = df_nivel[df_nivel['Categoria_Area'] == area_c]
                            for cuenca in df_area_c[col_nivel].unique():
                                # Filtro de seguridad para nombres nulos
                                if not cuenca or str(cuenca).lower() in ['nan', 'none', 'cuenca sin nombre']: continue

                                # 🔥 ESTRATEGIA DE SINCRONIZACIÓN:
                                # El 'Nivel' en la matriz SQL será siempre 'Cuenca' para facilitar el match global
                                # El 'Padre' conservará la jerarquía técnica (ej. AH, ZH, NSS1)
                                id_control = f"Cuenca_{cuenca}_{area_c}".upper()
                                
                                if id_control not in entrenados_log:
                                    df_t = df_area_c[df_area_c[col_nivel] == cuenca].sort_values(by=col_anio)
                                    if not df_t.empty and df_t['Total_frag'].sum() > 0:
                                        try:
                                            # ajustar_modelos(x, y, nivel, territorio, padre, area)
                                            ajustar_modelos(
                                                df_t[col_anio].values, 
                                                df_t['Total_frag'].values, 
                                                'Cuenca',       # Nivel para búsqueda en Toma de Decisiones
                                                cuenca,         # Nombre del territorio
                                                etiqueta_padre, # Jerarquía original como metadato
                                                area_c          # Segmento (Total/Urbano/Rural)
                                            )
                                            entrenados_log.add(id_control)
                                        except Exception: pass
                                            
            ===========================================================
            # 🏢 FASE 4: ENTRENAMIENTO ADMINISTRATIVO Y MACROREGIONAL
            # =========================================================
            texto_progreso.markdown("🏢 **Fase 4: Entrenando Modelos Administrativos (Municipios, Regiones, Macroregiones)...**")
            
            if 'df_mun' in locals() or 'df_mun' in globals():
                df_admin = df_mun.copy()
                # Asegurar columna Categoria_Area
                if 'Categoria_Area' not in df_admin.columns and 'area_geografica' in df_admin.columns:
                    df_admin['Categoria_Area'] = df_admin['area_geografica'].apply(clasificar_area)
                
                # A. MUNICIPIOS
                for mpio in df_admin['municipio'].dropna().unique():
                    for cat in ['Total', 'Urbana', 'Rural']:
                        df_f = df_admin[(df_admin['municipio'] == mpio) & (df_admin['Categoria_Area'] == cat)].sort_values(by=col_anio)
                        if len(df_f) >= 3 and df_f['Total'].sum() > 0:
                            ajustar_modelos(df_f[col_anio].values, df_f['Total'].values, 'Municipal', mpio, df_f['depto_nom'].iloc[0] if not df_f.empty else 'Antioquia', cat)
                
                # B. DEPARTAMENTO
                for cat in ['Total', 'Urbana', 'Rural']:
                    df_f = df_admin[df_admin['Categoria_Area'] == cat].groupby(col_anio)['Total'].sum().reset_index()
                    if len(df_f) >= 3 and df_f['Total'].sum() > 0:
                        ajustar_modelos(df_f[col_anio].values, df_f['Total'].values, 'Departamental', 'Antioquia', 'Colombia', cat)
                        
                # C. MACROREGIONES (Pacífica, Amazonía, etc.)
                if 'Macroregion' in df_admin.columns:
                    for macro in df_admin['Macroregion'].dropna().unique():
                        for cat in ['Total', 'Urbana', 'Rural']:
                            df_f = df_admin[(df_admin['Macroregion'] == macro) & (df_admin['Categoria_Area'] == cat)].groupby(col_anio)['Total'].sum().reset_index()
                            if len(df_f) >= 3 and df_f['Total'].sum() > 0:
                                ajustar_modelos(df_f[col_anio].values, df_f['Total'].values, 'MACROREGION', macro, 'Colombia', cat)

                # D. SUBREGIONES DE ANTIOQUIA (Oriente, Bajo Cauca, etc.)
                col_reg = 'subregion' if 'subregion' in df_admin.columns else ('Region' if 'Region' in df_admin.columns else None)
                if col_reg:
                    for reg in df_admin[col_reg].dropna().unique():
                        for cat in ['Total', 'Urbana', 'Rural']:
                            df_f = df_admin[(df_admin[col_reg] == reg) & (df_admin['Categoria_Area'] == cat)].groupby(col_anio)['Total'].sum().reset_index()
                            if len(df_f) >= 3 and df_f['Total'].sum() > 0:
                                ajustar_modelos(df_f[col_anio].values, df_f['Total'].values, 'REGION', reg, 'Antioquia', cat)
                                
                # E. ESCALA URBANA (Todas las Cabeceras)
                df_f = df_admin[df_admin['Categoria_Area'] == 'Urbana'].groupby(col_anio)['Total'].sum().reset_index()
                if len(df_f) >= 3 and df_f['Total'].sum() > 0:
                    ajustar_modelos(df_f[col_anio].values, df_f['Total'].values, 'ESCALA_URBANA', 'Todas las Cabeceras', 'Antioquia', 'Urbana')            

            # =====================================================================
            # 🔥 6. FORJA DE LLAVES UNIVERSALES Y CARGA A MEMORIA
            # =====================================================================
            if matriz_resultados:
                df_masivo = pd.DataFrame(matriz_resultados)
                
                # 🔑 LA MAGIA: Aplicamos la Llave Universal a todo el DataFrame
                def forjar_llave(row):
                    jerarquia = str(row.get('Nivel', '')).upper()
                    if jerarquia == "MUNICIPAL": jerarquia = "MUNICIPIO"
                    elif jerarquia == "DEPARTAMENTAL": jerarquia = "DEPARTAMENTO"
                    elif jerarquia == "CUENCA": jerarquia = "NSS3" 
                    terr = str(row.get('Territorio', '')).replace(" ", "_").upper()
                    cat = str(row.get('Area', '')).upper()
                    return f"{jerarquia}_{terr}_{cat}"
                    
                df_masivo['LLAVE_UNIVERSAL'] = df_masivo.apply(forjar_llave, axis=1)
                
                barra_progreso.progress(1.0)
                texto_progreso.success(f"✅ ¡Entrenamiento Completado! {len(df_masivo)} modelos forjados con Llaves Universales.")
                
                # Guardamos en RAM para que el Validador Visual y la Pestaña de Descargas lo detecten
                st.session_state['df_matriz_demografica'] = df_masivo
                st.info("💡 Ve al panel inferior 'Exportar Cerebro Demográfico' para Inyectar estos resultados a PostgreSQL de forma permanente.")
            else:
                texto_progreso.warning("⚠️ No se generaron resultados para procesar.")

        # 🛑 SALVAVIDAS DE PYTHON: Cierra el 'try' principal del botón
        except Exception as e:
            st.error(f"❌ Error crítico durante el entrenamiento masivo: {e}")

    # =====================================================================
    # 🔬 VALIDADOR VISUAL COMPARATIVO (ACTUALIZADO A LLAVE UNIVERSAL)
    # =====================================================================
    # 🔥 FIX 2: Ya no busca 'Area' sino 'Categoria' para evitar el próximo error
    if 'df_matriz_demografica' in st.session_state and 'Categoria' in st.session_state['df_matriz_demografica'].columns:
        st.divider()
        st.subheader("🔬 Validador Visual Comparativo (Urbano vs Rural vs Total)")
        
        df_mat = st.session_state['df_matriz_demografica']
        
        c_nav1, c_nav2, c_nav3 = st.columns([1, 1.5, 1])
        with c_nav1:
            niveles_disp = list(df_mat['Jerarquia'].unique())
            idx_mun = niveles_disp.index('MUNICIPIO') if 'MUNICIPIO' in niveles_disp else 0
            nivel_val = st.selectbox("1. Nivel de Análisis:", niveles_disp, index=idx_mun)
        with c_nav2:
            territorios_disp = sorted(df_mat[df_mat['Jerarquia'] == nivel_val]['Territorio'].unique())
            idx_terr = 0
            terr_val = st.selectbox("2. Territorio (Municipio/Cuenca):", territorios_disp, index=idx_terr)
        with c_nav3:
            anio_futuro = st.slider("3. Proyectar hasta el año:", min_value=2025, max_value=2100, value=2050, step=5)
            
        st.markdown("---")
        
        def renderizar_panel(area_sel, key_suffix):
            import numpy as np
            import plotly.graph_objects as go
            
            df_filtrado = df_mat[(df_mat['Jerarquia'] == nivel_val) & (df_mat['Territorio'] == terr_val) & (df_mat['Categoria'] == area_sel)]
            if df_filtrado.empty:
                st.warning(f"No hay datos proyectados para la categoría '{area_sel}' en {terr_val}.")
                return
                
            fila_terr = df_filtrado.iloc[0]
            mejor_modelo = fila_terr.get('Modelo_Recomendado', 'Logístico')
            
            x_hist, y_hist = [], [] # Simplificado para evitar colisiones con tu df_mun original
            x_offset = fila_terr.get('Año_Base', 2018)
            x_pred = np.arange(x_offset, anio_futuro + 1)
            x_norm_pred = x_pred - x_offset
            
            fig = go.Figure()
            
            def config_linea(nombre_mod, color):
                es_ganador = mejor_modelo == nombre_mod
                return dict(color=color, width=4 if es_ganador else 2, dash='solid' if es_ganador else 'dash'), 1.0 if es_ganador else 0.4
                
            if 'Log_K' in fila_terr:
                y_log = fila_terr['Log_K'] / (1 + fila_terr['Log_a'] * np.exp(-fila_terr['Log_r'] * x_norm_pred))
                line_log, op_log = config_linea('Logístico', '#2980b9')
                fig.add_trace(go.Scatter(x=x_pred, y=y_log, mode='lines', name=f"Logístico", line=line_log, opacity=op_log))
                
            if 'Exp_a' in fila_terr and not pd.isna(fila_terr['Exp_a']):
                y_exp = fila_terr['Exp_a'] * np.exp(fila_terr['Exp_b'] * x_norm_pred)
                line_exp, op_exp = config_linea('Exponencial', '#e67e22')
                fig.add_trace(go.Scatter(x=x_pred, y=y_exp, mode='lines', name=f"Exponencial", line=line_exp, opacity=op_exp))
                
            if 'Poly_A' in fila_terr and not pd.isna(fila_terr['Poly_A']):
                y_poly = fila_terr['Poly_A']*(x_norm_pred**3) + fila_terr['Poly_B']*(x_norm_pred**2) + fila_terr['Poly_C']*x_norm_pred + fila_terr['Poly_D']
                line_poly, op_poly = config_linea('Polinomial_3', '#27ae60')
                fig.add_trace(go.Scatter(x=x_pred, y=y_poly, mode='lines', name=f"Polinomial", line=line_poly, opacity=op_poly))
                
            fig.update_layout(
                title=f"Proyección {area_sel} - {terr_val}", 
                xaxis_title="Año", yaxis_title="Habitantes", hovermode="x unified", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander(f"🔑 Ver Llave Universal del Modelo", expanded=False):
                st.code(fila_terr.get('LLAVE_UNIVERSAL', 'No generada'), language="bash")

        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1:
            area_1 = st.selectbox("Área (Panel Izquierdo):", ["Total", "Urbana", "Rural"], index=0, key="sel_a1")
            renderizar_panel(area_1, "g1")
        with col_graf_2:
            area_2 = st.selectbox("Área (Panel Derecho):", ["Total", "Urbana", "Rural"], index=1, key="sel_a2")
            renderizar_panel(area_2, "g2")
            
# ==========================================
# PESTAÑA 5: RANKINGS Y DINÁMICA HISTÓRICA (VERSIÓN ROBUSTA FINAL)
# ==========================================
with tab_rankings:
    zona_actual = "Total"
    if not df_mapa_base.empty and 'area_geografica' in df_mapa_base.columns:
        zona_actual = df_mapa_base['area_geografica'].iloc[0].title()
        
    st.subheader(f"📊 Análisis Comparativo y Trayectorias Poblacionales ({zona_actual})")
    zona_q = zona_actual.lower()
    
    df_rank = pd.DataFrame()
    titulo_ranking = ""
    escala_str = escala_sel.lower()

    # --- 🧠 MOTOR DE RANKING INFALIBLE (Bypass de Hermanos Territoriales) ---
    if "global" in escala_str or "nacional" in escala_str:
        df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q)].groupby('depto_nom')['Total'].sum().reset_index()
        df_rank.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
        df_rank['Padre'] = 'Colombia'
        titulo_ranking = "Departamentos de Colombia"
        
    elif "municipal" in escala_str:
        # 🔥 EL RESCATE MUNICIPAL: Buscamos a los "hermanos" en la base DANE
        padre_dep = locals().get('depto_sel', locals().get('agrupador_sel', "ANTIOQUIA"))
        if "regiones" in escala_str:
            df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q) & (df_mun['Macroregion'] == padre_dep)].copy()
            titulo_ranking = f"Municipios de la Región {padre_dep}"
        else:
            df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q) & (df_mun['depto_nom'] == padre_dep)].copy()
            titulo_ranking = f"Municipios de {padre_dep}"
            
        df_rank = df_rank.groupby('municipio')['Total'].sum().reset_index().rename(columns={'municipio': 'Territorio'})
        df_rank['Padre'] = padre_dep
        
    elif "urbana" in escala_str:
        # 🔥 EL RESCATE URBANO: Forzamos a rankear todas las cabeceras de Antioquia
        df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == 'urbano') & (df_mun['depto_nom'] == 'Antioquia')].copy()
        df_rank = df_rank.groupby('municipio')['Total'].sum().reset_index().rename(columns={'municipio': 'Territorio'})
        df_rank['Padre'] = 'Antioquia'
        titulo_ranking = "Cabeceras Municipales (Antioquia)"
        
    else:
        # Para Cuencas, Veredas, Medellín y Regiones, el mapa sí trae datos múltiples
        if not df_mapa_base.empty:
            df_rk_base = df_mapa_base.copy()
            col_t = 'año' if 'año' in df_rk_base.columns else ('Año' if 'Año' in df_rk_base.columns else None)
            if col_t: df_rk_base = df_rk_base[df_rk_base[col_t] == año_sel]
            
            if not df_rk_base.empty:
                cols_group = ['Territorio', 'Padre'] if 'Padre' in df_rk_base.columns else ['Territorio']
                df_rank = df_rk_base.groupby(cols_group)['Total'].sum().reset_index()
                
                if 'Padre' not in df_rank.columns: df_rank['Padre'] = 'Zona Analizada'
                
                if "cuencas" in escala_str: titulo_ranking = "Cuencas / Microcuencas"
                elif "intra-urbana" in escala_str: titulo_ranking = "Comunas / Barrios de Medellín"
                elif "veredal" in escala_str: titulo_ranking = "Veredas"
                else: titulo_ranking = f"División Territorial de {df_rank['Padre'].iloc[0]}"

    # Limpieza numérica vital
    if not df_rank.empty:
        df_rank['Total'] = pd.to_numeric(df_rank['Total'], errors='coerce').fillna(0)
        df_rank = df_rank[df_rank['Total'] > 0]

    # DIBUJO DE LOS GRÁFICOS TOP/BOTTOM
    if not df_rank.empty and len(df_rank) > 1:
        st.markdown(f"### 🏆 Extremos Poblacionales ({año_sel}) - {titulo_ranking}")
        col_rank_izq, col_rank_der = st.columns(2)
        
        with col_rank_izq:
            df_plot_top = df_rank.nlargest(15, 'Total')
            fig_top = px.bar(df_plot_top, x='Total', y='Territorio', orientation='h', color='Total', color_continuous_scale='Viridis', hover_data={'Padre': True} if 'Padre' in df_plot_top.columns else None, title="📈 Top 15: Mayor Población")
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_rank_der:
            df_plot_bot = df_rank.nsmallest(15, 'Total')
            fig_bot = px.bar(df_plot_bot, x='Total', y='Territorio', orientation='h', color='Total', color_continuous_scale='Plasma', hover_data={'Padre': True} if 'Padre' in df_plot_bot.columns else None, title="📉 Bottom 15: Menor Población")
            fig_bot.update_layout(yaxis={'categoryorder':'total descending'}, height=450)
            st.plotly_chart(fig_bot, use_container_width=True)
            
        # ==========================================================
        # --- 📈 SECCIÓN DE CURVAS HISTÓRICAS (TOP 10) ---
        # ==========================================================
        st.markdown("---")
        st.markdown(f"### 📈 Dinámica Poblacional - Trayectorias, Evolución y Proyección - Top 10 ({zona_actual})")
        
        # Obtenemos los nombres de los 10 más grandes para la curva
        top_10_nombres = df_rank.nlargest(10, 'Total')['Territorio'].tolist()
        df_line = pd.DataFrame()
        
        # Tu bloque de código revisado y asegurado:
        padre_seguro = locals().get('agrupador_sel', locals().get('sel_territorio', locals().get('reg_sel', "Selección")))

        es_escala_medellin = "Intra-Urbana" in escala_sel
        es_escala_espacial = any(e in escala_sel for e in ["Veredal", "Cuencas"])

        if es_escala_medellin:
            # 🔥 MOTOR C: Medellín (Lógica Proporcional Pura)
            if 'df_global' in locals() and not df_global.empty and 'Pob_Medellin' in df_global.columns:
                col_anio_glob = 'Año' if 'Año' in df_global.columns else 'año'
                df_med_hist = df_global.dropna(subset=[col_anio_glob, 'Pob_Medellin']).sort_values(by=col_anio_glob)
                años_plot = df_med_hist[col_anio_glob].values.astype(float)
                pob_med_macro = df_med_hist['Pob_Medellin'].values.astype(float)
                
                pob_total_shape = df_mapa_base['Total'].sum() if 'df_mapa_base' in locals() else df_rank['Total'].sum()
                
                records = []
                for t in top_10_nombres:
                    pob_t_actual = df_rank[df_rank['Territorio'] == t]['Total'].sum()
                    factor = (pob_t_actual / pob_total_shape) if pob_total_shape > 0 else 0
                    y_vals = pob_med_macro * factor
                    for a, y in zip(años_plot, y_vals):
                        records.append({'año': a, 'Territorio': t, 'Total': y})
                df_line = pd.DataFrame(records)

        elif not es_escala_espacial:
            # 🔥 MOTOR A: Escaleras Administrativas (DANE Puro - Revisado y Blindado)
            df_base_historica = df_mun[df_mun['area_geografica'] == zona_q].copy() if 'df_mun' in locals() else pd.DataFrame()
            
            if not df_base_historica.empty:
                if "global" in escala_str or "nacional" in escala_str:
                    df_line = df_base_historica.groupby(['año', 'depto_nom'])['Total'].sum().reset_index()
                    df_line.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
                    
                elif "departamental" in escala_str or "municipal (departamentos)" in escala_str:
                    padre_dep = locals().get('depto_sel', locals().get('agrupador_sel', "Antioquia"))
                    df_line = df_base_historica[df_base_historica['depto_nom'] == padre_dep].groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)

                elif "regiones" in escala_str or "macroregiones" in escala_str or "autoridades" in escala_str or "subregiones" in escala_str:
                    col_agrup = 'Macroregion'
                    if "subregion" in escala_str: col_agrup = 'subregion'
                    elif "autoridad" in escala_str or "car" in escala_str: col_agrup = 'car'
                    
                    if col_agrup in df_base_historica.columns:
                        df_line = df_base_historica[df_base_historica[col_agrup] == padre_seguro].groupby(['año', 'municipio'])['Total'].sum().reset_index()
                        df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)
                        
                elif "urbana" in escala_str:
                    df_line = df_base_historica[df_base_historica['depto_nom'] == 'Antioquia'].groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)
                
                else:
                    df_line = df_base_historica.groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)

            if not df_line.empty:
                df_line = df_line[df_line['Territorio'].isin(top_10_nombres)]
                
                # Interpolación para limpiar baches del DANE (Años 2006-2019)
                def conciliacion_censal(group):
                    group['año'] = pd.to_numeric(group['año'], errors='coerce')
                    group = group.sort_values('año')
                    mask = (group['año'] >= 2006) & (group['año'] <= 2019)
                    group.loc[mask, 'Total'] = np.nan
                    group['Total'] = group['Total'].interpolate(method='linear').ffill().bfill()
                    return group
                df_line = df_line.groupby('Territorio', group_keys=False).apply(conciliacion_censal)

        else:
            # 🔥 MOTOR B: Cuencas y Veredas (Matriz Top-Down)
            if 'df_matriz_demografica' in st.session_state:
                df_mat = st.session_state['df_matriz_demografica']
                area_matriz = 'Urbana' if zona_actual.lower() in ['urbano', 'urbana'] else zona_actual.capitalize()
                filas_top = df_mat[(df_mat['Territorio'].isin(top_10_nombres)) & (df_mat['Area'] == area_matriz)]
                
                if not filas_top.empty:
                    records = []
                    años_plot = np.arange(1985, 2041)
                    for _, row in filas_top.iterrows():
                        terr = row['Territorio']
                        x_offset = row['Año_Base']
                        x_norm = años_plot - x_offset
                        modelo = str(row.get('Modelo_Recomendado', 'Desconocido'))
                        
                        y_vals = np.zeros_like(años_plot, dtype=float)
                        if 'Logistico' in modelo: y_vals = row.get('Log_K', 0) / (1 + row.get('Log_a', 0) * np.exp(-row.get('Log_r', 0) * x_norm))
                        elif 'Exponencial' in modelo: y_vals = row.get('Exp_a', 0) * np.exp(row.get('Exp_b', 0) * x_norm)
                        elif 'Polinomial' in modelo: y_vals = row.get('Poly_A', 0)*(x_norm**3) + row.get('Poly_B', 0)*(x_norm**2) + row.get('Poly_C', 0)*x_norm + row.get('Poly_D', 0)
                        else: y_vals = np.full_like(años_plot, row.get('Pob_Base', 0))
                            
                        for a, y in zip(años_plot, y_vals):
                            records.append({'año': a, 'Territorio': terr, 'Total': y})
                    df_line = pd.DataFrame(records)

        # RENDERIZADO FINAL DE LAS CURVAS (Motor Universal)
        if not df_line.empty:
            fig_line = px.line(df_line, x='año', y='Total', color='Territorio', markers=True,
                               title=f"Trayectorias Demográficas: Top 10 Territorios ({zona_actual})",
                               labels={'año': 'Año', 'Total': 'Habitantes'})
            fig_line.update_layout(hovermode="x unified", height=500, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_line, use_container_width=True)
            
    else:
        st.info("💡 Por favor, selecciona un nivel de análisis superior (ej. Departamento completo) para ver comparativas.")
        
# ==========================================
# PESTAÑA 6: DESCARGAS Y EXPORTACIÓN
# ==========================================
with tab_descargas:
    # --- FIX: ESCUDO SUPREMO EN DESCARGAS ---
    titulo_seguro_descarga = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio_Seleccionado"))
    
    st.subheader("💾 Exportación de Resultados y Series de Tiempo")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("### 📈 Curvas de Proyección (1950-2100)")
        csv_proj = df_proj.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Descargar Proyecciones (CSV)", data=csv_proj, file_name=f"Proyecciones_{titulo_seguro_descarga}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(df_proj.dropna(subset=['Real']).head(5), use_container_width=True)

    with col_d2:
        st.markdown(f"### 📊 Pirámide Sintética ({año_sel})")
        df_descarga_pir = df_piramide_final.copy()
        if not df_descarga_pir.empty:
            # Volvemos los Hombres a números positivos para el Excel
            if 'Hombres' in df_descarga_pir.columns:
                df_descarga_pir['Hombres'] = df_descarga_pir['Hombres'].abs()
                
            csv_pir = df_descarga_pir.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"⬇️ Descargar Pirámide {año_sel} (CSV)", data=csv_pir, file_name=f"Piramide_{año_sel}_{titulo_seguro_descarga}.csv", mime="text/csv", use_container_width=True)
            st.dataframe(df_descarga_pir.head(5), use_container_width=True)

# ==============================================================================
# 💾 EXPORTACIÓN AUTOMÁTICA A SQL Y DESCARGA (PRODUCCIÓN)
# ==============================================================================

if 'df_matriz_demografica' in st.session_state:
    st.markdown("---")
    st.subheader("💾 Exportar Cerebro Demográfico (Para Producción)")
    st.info("💡 Tu matriz ya está en memoria. Puedes inyectarla directamente a la base de datos o descargarla como archivo de respaldo.")
    
    # Rescatamos la matriz de la memoria
    df_matriz_demo = st.session_state['df_matriz_demografica']
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        # 🔒 MURO DE SEGURIDAD PARA INYECCIÓN DB
        with st.expander("🔐 Administración: Inyectar a SQL"):
            st.warning("⚠️ Acción segura: Actualizará los registros manteniendo intactos los permisos (RLS) de la tabla en Supabase.")
            pwd = st.text_input("Clave de Administrador:", type="password", key="pwd_sql_demo")
            
            if st.button("🚀 Confirmar Inyección", type="primary", use_container_width=True):
                # Usa la misma clave que en el Generador Beta (AdminPoter)
                if pwd == st.secrets.get("admin_password", "AdminPoter"):
                    with st.spinner("Conectando con PostgreSQL (Supabase)..."):
                        try:
                            from modules.db_manager import get_engine
                            from sqlalchemy import text
                            engine_sql = get_engine()
                            
                            # 🔥 FIX DEFINITIVO: PROTOCOLO DE AUTO-MANTENIMIENTO AMPLIADO (LINEAL)
                            with engine_sql.begin() as conn:
                                # 1. Creamos las columnas necesarias si no existen (Llave y parámetros de la recta)
                                conn.execute(text('''
                                    ALTER TABLE matriz_maestra_demografica 
                                    ADD COLUMN IF NOT EXISTS "LLAVE_UNIVERSAL" TEXT,
                                    ADD COLUMN IF NOT EXISTS "Lin_m" FLOAT,
                                    ADD COLUMN IF NOT EXISTS "Lin_b" FLOAT,
                                    ADD COLUMN IF NOT EXISTS "Lin_R2" FLOAT;
                                '''))
                                # 2. Borramos las filas viejas manteniendo los permisos RLS intactos
                                conn.execute(text("DELETE FROM matriz_maestra_demografica;"))
                                
                            # 3. Inyectamos la nueva matriz (que ahora incluye Lin_m, Lin_b y Lin_R2)
                            df_matriz_demo.to_sql('matriz_maestra_demografica', engine_sql, if_exists='append', index=False)
                            st.cache_data.clear()
                            
                            st.success(f"✅ ¡Inyección Exitosa! {len(df_matriz_demo)} registros actualizados en PostgreSQL.")
                            st.info("🔄 La caché ha sido limpiada. Los nuevos datos se cargarán al recargar la página.")
                        except Exception as e:
                            st.error(f"🚨 Error SQL: {e}")
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
                    
    with col_btn2:
        # El botón de descarga siempre estará visible mientras la matriz exista en memoria
        csv_matriz = df_matriz_demo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Matriz (CSV de Respaldo)", 
            data=csv_matriz, 
            file_name="Matriz_Multimodelo_Demografica.csv", 
            mime='text/csv',
            use_container_width=True
        )
