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
st.divider()

# --- FUNCION MÁGICA 1: EL ASPIRADOR DE TEXTOS (Match infalible) ---
def normalizar_texto(texto):
    if pd.isna(texto): return ""
    t = str(texto).upper()
    t = re.sub(r'\(.*?\)', '', t)
    
    # --- NUEVO: DESTRUCTOR DE PREFIJOS VEREDALES ---
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
    
    # 3. Traductor DANE-IGAC Actualizado (Con los 9 nombres pomposos)
    diccionario_rebeldes = {
        "BOGOTADC": "BOGOTA", "SANJOSEDECUCUTA": "CUCUTA", "LAGUAJIRA": "GUAJIRA", 
        "VALLEDELCAUCA": "VALLE", "VILLADESANDIEGODEUBATE": "UBATE", "SANTIAGODETOLU": "TOLU",
        "ELPENOL": "PENOL", "ELRETIRO": "RETIRO", "ELSANTUARIO": "SANTUARIO",
        "ELCARMENDEVIBORAL": "CARMENDEVIBORAL", "SANVICENTEFERRER": "SANVICENTE",
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
        # --- LOS 9 NUEVOS REBELDES (Nombres pomposos del nuevo mapa) ---
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
        "MANAUREBALCONDELCESAR": "MANAURE"
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
        
        # 2. Diccionario a prueba de balas
        reemplazos_area = {
            'cabecera': 'urbano',
            'cabecera municipal': 'urbano',
            'centros poblados y rural disperso': 'rural',
            'centro poblado y rural disperso': 'rural'
        }
        df_master['area_geografica'] = df_master['area_geografica'].replace(reemplazos_area)

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
            for region, deptos in REGIONES_COL.items():
                if depto in deptos: return region
            return "Sin Región"
            
        df_mun['Macroregion'] = df_mun.apply(asignar_region, axis=1)

        # -------------------------------------------------------
        # B. Crear df_nac (Nacional) -> AHORA INCLUYE TODAS LAS EDADES
        # -------------------------------------------------------
        df_nac_temp = df_mun[df_mun['area_geografica'] == 'total']
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
            # Limpiamos los puntos (miles) antes de que el sistema los confunda con decimales
            for col in df_global_csv.columns:
                if col != 'Año':
                    df_global_csv[col] = pd.to_numeric(
                        df_global_csv[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False),
                        errors='coerce'
                    )
                
            df_global = pd.merge(df_global_csv, df_global_dinamico, on='Año', how='outer')
        except Exception as e:
            pass
                
        # =======================================================
        # 2. Cargar datos Veredales (Dinámico desde Supabase)
        # =======================================================
        try:
            from modules.db_manager import get_engine
            engine_db = get_engine()
            df_ver = pd.read_sql("SELECT * FROM veredas_poblacion", engine_db)
        except Exception as e:
            df_ver = pd.DataFrame() # Escudo anti-errores si falla BD
            
            # --- SINCRONIZACIÓN DE NOMBRES ---
            # Hacemos que las veredas hablen el mismo idioma que el DANE (Mayúscula Inicial)
            if 'Municipio' in df_ver.columns:
                df_ver['Municipio'] = df_ver['Municipio'].astype(str).str.title()
                
            # EL ESCUDO DEFINITIVO
            if 'Poblacion_hab' in df_ver.columns:
                df_ver['Poblacion_hab'] = pd.to_numeric(df_ver['Poblacion_hab'].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').fillna(0)

        return df_nac, df_mun, df_ver, df_global
        
    except Exception as e:
        import streamlit as st
        st.error(f"🚨 Error cargando las bases de datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Llamada principal a la función y escudo de seguridad para detener la app si no hay datos
df_nac, df_mun, df_ver, df_global = cargar_datos_limpios()
if df_nac.empty or df_mun.empty: st.stop()

# --- 2. MODELOS MATEMÁTICOS ---
def modelo_lineal(x, m, b): return m * x + b
def modelo_exponencial(x, p0, r): return p0 * np.exp(r * (x - 2005))
def modelo_logistico(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))

# --- 3. CONFIGURACIÓN Y FILTROS LATERALES ---
st.sidebar.header("⚙️ 1. Selección Territorial")

# 1. LA LISTA MAESTRA (Sincronizada con el Solver)
escala_sel = st.sidebar.radio("Nivel de Análisis:", [
    "🌍 Global y Suramérica",
    "🇨🇴 Nacional (Colombia)", 
    "🏛️ Departamental (Colombia)", 
    "🧩 Regional (Macroregiones)",
    "💧 Cuencas Hidrográficas", 
    "🏢 Municipal (Regiones)", 
    "🏢 Municipal (Departamentos)", 
    "🌿 Veredal (Antioquia)"
])

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
    df_base = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['area_geografica'] == 'total')]
    
    filtro_zona = depto_sel
    titulo_terr = depto_sel
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    df_hist = df_base.groupby(col_anio)['Total'].sum().reset_index()
    df_hist = df_hist.sort_values(by=col_anio)
    
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base.groupby(['municipio', col_anio])['Total'].sum().reset_index()
    df_mapa_base.rename(columns={'municipio': 'Territorio'}, inplace=True)
    df_mapa_base['Padre'] = depto_sel

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
    if 'df_matriz_demografica' in st.session_state:
        df_matriz = st.session_state['df_matriz_demografica']
    else:
        try: df_matriz = pd.read_csv("Matriz_Multimodelo_Demografica (1).csv") 
        except: df_matriz = pd.DataFrame()
            
    if not df_matriz.empty and 'Nivel' in df_matriz.columns:
        df_cuencas_solo = df_matriz[df_matriz['Nivel'] == 'Cuenca']
        
        if not df_cuencas_solo.empty:
            # --- MOTOR EN CASCADA PARA CUENCAS ---
            try:
                from modules.db_manager import get_engine
                df_hier = pd.read_sql("SELECT DISTINCT nomah, nomzh, nom_szh, nom_nss1, nom_nss2, nom_nss3 FROM cuencas", get_engine())
            except:
                df_hier = pd.DataFrame()
            
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
            
            # --- PROTECCIÓN ANTI-NAME ERROR ---
            cuencas_a_graficar = cuenca_sel if cuenca_sel else cuencas_disp
            
            if cuencas_a_graficar:
                if cuenca_sel:
                    filtro_zona = " + ".join(cuenca_sel[:2]) + ("..." if len(cuenca_sel)>2 else "")
                    titulo_terr = f"Cuencas Seleccionadas: {filtro_zona}"
                else:
                    if not df_hier.empty:
                        if sel_szh != "-- Seleccione --": titulo_terr = f"SZH: {sel_szh}"
                        elif sel_zh != "-- Seleccione --": titulo_terr = f"ZH: {sel_zh}"
                        elif sel_ah != "-- Seleccione --": titulo_terr = f"AH: {sel_ah}"
                        else: titulo_terr = "Todas las Cuencas"
                    else:
                        titulo_terr = "Todas las Cuencas"
                    filtro_zona = titulo_terr

                años_hist = np.arange(1985, 2043)
                pob_hist_acumulada = np.zeros_like(años_hist, dtype=float)
                mapa_data = []
                
                # 2. MOTOR DE AGREGACIÓN MULTICAPA (CON FILTRO ANTI-DUPLICADOS Y DEPURADOR)
                df_cuencas_solo = df_cuencas_solo.copy()
                if 'MATCH_ID' not in df_cuencas_solo.columns:
                    df_cuencas_solo['MATCH_ID'] = df_cuencas_solo['Territorio'].astype(str).apply(normalizar_texto)

                ids_matriz = df_cuencas_solo['MATCH_ID'].dropna().unique().tolist()
                import difflib

                cuencas_cruzadas = 0
                log_cruces = []

                for c in cuencas_a_graficar:
                    # 🔥 EXTRACCIÓN TOPOLÓGICA BLINDADA: Usamos el nivel de detalle máximo disponible
                    if col_res in df_hier.columns:
                        cols_lbl = [col for col in ['nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh'] if col in df_hier.columns]
                        df_hier['subc_lbl'] = df_hier[cols_lbl].bfill(axis=1).iloc[:, 0]
                        
                        c_norm = normalizar_texto(c)
                        # Filtramos hijos donde el padre coincida exactamente o en versión normalizada
                        hijos = df_hier[(df_hier[col_res] == c) | (df_hier[col_res].astype(str).apply(normalizar_texto) == c_norm)]['subc_lbl'].dropna().unique()
                        micro_cuencas = hijos if len(hijos) > 0 else [c]
                    else:
                        micro_cuencas = [c]

                    c_pob_temp_hist = np.zeros_like(años_hist, dtype=float)
                    matrix_ids_sumados = set() 

                    for micro in micro_cuencas:
                        micro_norm = normalizar_texto(micro)
                        match_val = None

                        if micro_norm in ids_matriz:
                            match_val = micro_norm
                        else:
                            # Tolerancia alta (0.85) para ignorar pelusas topológicas
                            matches = difflib.get_close_matches(micro_norm, ids_matriz, n=1, cutoff=0.85)
                            if matches: match_val = matches[0]

                        if match_val:
                            log_cruces.append({"Micro-cuenca en Mapa": micro, "ID Matriz": match_val, "Estado": "✅ Encontrado"})
                            
                            cuenca_data = df_cuencas_solo[df_cuencas_solo['MATCH_ID'] == match_val]
                            if not cuenca_data.empty:
                                c_total = cuenca_data[cuenca_data['Area'] == 'Total']
                                fila_tot = c_total.iloc[0] if not c_total.empty else cuenca_data.iloc[0]
                                
                                v_t = float(fila_tot.get('Pob_Base', 0))
                                
                                c_urb = cuenca_data[cuenca_data['Area'] == 'Urbano']
                                v_u = float(c_urb.iloc[0].get('Pob_Base', 0)) if not c_urb.empty else 0
                                
                                c_rur = cuenca_data[cuenca_data['Area'] == 'Rural']
                                v_r = float(c_rur.iloc[0].get('Pob_Base', 0)) if not c_rur.empty else 0

                                # 🗺️ MAPA: Pintamos polígonos correctos
                                mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': v_t, 'area_geografica': 'total'})
                                mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': v_u, 'area_geografica': 'urbano'})
                                mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': v_r, 'area_geografica': 'rural'})

                                # 📊 MATEMÁTICA: Sumar solo una vez
                                if match_val not in matrix_ids_sumados:
                                    matrix_ids_sumados.add(match_val)
                                    cuencas_cruzadas += 1

                                    modelo_ganador = str(fila_tot.get('Modelo_Recomendado', 'Desconocido'))
                                    pob_temp = np.zeros_like(años_hist, dtype=float)

                                    if 'Logistico' in modelo_ganador:
                                        pob_temp = fila_tot.get('Log_K', 0) / (1 + fila_tot.get('Log_a', 0) * np.exp(-fila_tot.get('Log_r', 0) * (años_hist - 1985)))
                                    elif 'Exponencial' in modelo_ganador:
                                        pob_temp = fila_tot.get('Exp_a', 0) * np.exp(fila_tot.get('Exp_b', 0) * (años_hist - 1985))
                                    elif 'Polinomial' in modelo_ganador:
                                        x_norm = años_hist - 1985
                                        pob_temp = fila_tot.get('Poly_A', 0)*x_norm**3 + fila_tot.get('Poly_B', 0)*x_norm**2 + fila_tot.get('Poly_C', 0)*x_norm + fila_tot.get('Poly_D', 0)

                                    c_pob_temp_hist += pob_temp
                        else:
                            log_cruces.append({"Micro-cuenca en Mapa": micro, "ID Matriz": "Ninguno", "Estado": "❌ Faltante"})
                            # 🔥 FIX HUECOS: Pintar vacíos con 0 habitantes para cerrar la cuenca
                            mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': 0, 'area_geografica': 'total'})
                            mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': 0, 'area_geografica': 'urbano'})
                            mapa_data.append({'Territorio': micro, 'Padre': c, 'Total': 0, 'area_geografica': 'rural'})

                    pob_hist_acumulada += c_pob_temp_hist

                # --- 🔍 DEPURADOR FORENSE INYECTADO AL PANEL ---
                if log_cruces:
                    df_log = pd.DataFrame(log_cruces)
                    faltantes = len(df_log[df_log['Estado'] == '❌ Faltante'])
                    if faltantes > 0:
                        st.sidebar.warning(f"⚠️ {faltantes} micro-cuencas del mapa no tienen datos en la Matriz Demográfica.")
                    with st.sidebar.expander("🔍 Ver Depurador de Cuencas"):
                        st.dataframe(df_log, use_container_width=True)

                pob_hist = pob_hist_acumulada
                df_mapa_base = pd.DataFrame(mapa_data)
            else:
                filtro_zona, titulo_terr, años_hist, pob_hist, df_mapa_base = "Ninguna", "Sin Datos", np.array([]), np.array([]), pd.DataFrame()
        else:
            st.sidebar.warning("⚠️ Entrena la matriz de cuencas en la pestaña 4.")
            filtro_zona, titulo_terr, años_hist, pob_hist, df_mapa_base = "Error", "Sin Datos", np.array([]), np.array([]), pd.DataFrame()
    else:
        st.sidebar.error("🚨 Matriz Maestra no encontrada.")
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
    df_hist = df_base[df_base['area_geografica'] == 'total'].groupby(col_anio)['Total'].sum().reset_index()
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base.copy()
    df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

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
            
            df_mapa_base = df_veredas.rename(columns={
                col_ver: 'Territorio',
                col_mun: 'Padre',
                col_pob: 'Total'
            })
            
            if 'Padre' not in df_mapa_base.columns:
                df_mapa_base['Padre'] = "Antioquia"
            
            lista_mpios = sorted(df_mapa_base['Padre'].dropna().astype(str).unique())
            mpio_sel = st.sidebar.selectbox("Municipio:", ["TODOS (Ver Mapa Completo)"] + lista_mpios)
            
            if mpio_sel != "TODOS (Ver Mapa Completo)":
                df_mapa_base = df_mapa_base[df_mapa_base['Padre'] == mpio_sel]
                titulo_terr = f"Veredas de {mpio_sel}"
            else:
                titulo_terr = "Todas las Veredas (Antioquia)"
        else:
            df_mapa_base = pd.DataFrame()
            
    except Exception as e:
        st.sidebar.error(f"❌ Error general: {e}")
        df_mapa_base = pd.DataFrame()
    
# --- SELECTOR GLOBAL DE ÁREA (Afecta Gráficos y Mapas) ---
st.markdown("---")
if escala_sel != "🌿 Veredal (Antioquia)":
    area_global = st.radio("Filtro Poblacional Global:", ["Total", "Urbano", "Rural"], horizontal=True)
else:
    area_global = "Rural"
    st.info("ℹ️ A escala veredal, el motor matemático calcula todo como población rural.")

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
    try: df_matriz = pd.read_csv(os.path.join(RUTA_RAIZ, "data", "Matriz_Maestra_Demografica.csv"), sep=';' if ';' in open(os.path.join(RUTA_RAIZ, "data", "Matriz_Maestra_Demografica.csv")).readline() else ',')
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
        k_guess = max_y * 1.2 if es_creciente else max_y
        a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
        r_guess = 0.02 if es_creciente else -0.02
        
        limites = ([max_y * 0.8, 0, -0.2], [max_y * 3.0 if es_creciente else max_y * 1.1, np.inf, 0.3])
        popt_log, _ = curve_fit(f_log, x_train_norm, y_train, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
        proyecciones['Logístico'] = f_log(x_proj_norm, *popt_log)
        param_K, param_r = popt_log[0], popt_log[2]
    except Exception:
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
        
        st.subheader(f"📈 Curvas de Crecimiento Poblacional - {titulo_seguro}")
        fig_curvas = go.Figure()
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Logístico'], mode='lines', name='Mod. Logístico', line=dict(color='#10b981', dash='dash')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Exponencial'], mode='lines', name='Mod. Exponencial', line=dict(color='#f59e0b', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Lineal'], mode='lines', name='Mod. Lineal', line=dict(color='#6366f1', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Reales (Censo)', marker=dict(color='#ef4444', size=8, symbol='diamond')))
        fig_curvas.update_layout(hovermode="x unified", xaxis_title="Año", yaxis_title="Habitantes", template="plotly_white")
        st.plotly_chart(fig_curvas, use_container_width=True)

    with col_param:
        st.subheader("🧮 Ecuaciones")
        st.latex(r"Log: P(t) = \frac{K}{1 + e^{-r(t - t_0)}}")
        if param_K != "N/A": st.success(f"**K:** {param_K:,.0f} hab.")
        st.latex(r"Exp: P(t) = P_0 \cdot e^{r(t - t_0)}")
        st.latex(r"Lin: P(t) = m \cdot t + b")

    st.divider()

    st.sidebar.header("🎯 2. Viaje en el Tiempo")
    modelo_sel = st.sidebar.radio("Base de cálculo para la pirámide:", ["Logístico", "Exponencial", "Lineal", "Dato Real (Si existe)"])
    columna_modelo = "Real" if modelo_sel == "Dato Real (Si existe)" else modelo_sel

    col_anio_pyr = 'año' if 'año' in df_nac.columns else 'Año'
    años_disp = sorted(df_nac[col_anio_pyr].unique())
    año_sel = st.sidebar.select_slider("Selecciona un Año Estático:", options=años_disp, value=2024)

    st.sidebar.markdown("---")
    st.sidebar.subheader("▶️ Animación Temporal")
    velocidad_animacion = st.sidebar.slider("Velocidad (Segundos por cuadro)", 0.1, 2.0, 0.5)
    iniciar_animacion = st.sidebar.button("▶️ Reproducir Evolución", type="primary", use_container_width=True)

    # --- FIX: ESCUDO SUPREMO ANTI-NAME ERROR (PARA LA PIRÁMIDE) ---
    titulo_seguro = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio Seleccionado"))
    
    st.subheader(f"Estructura Poblacional Sintética - {titulo_seguro}")
    
    # Selector de comparación en la interfaz principal
    zona_comparacion = st.radio("Selecciona el área para comparar con la Estructura Total:", ["Urbano", "Rural"], horizontal=True)

    # Creamos dos columnas maestras para poner las pirámides lado a lado
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        ph_tit_1 = st.empty()
        ph_graf_1 = st.empty()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            ph_m_pob_1 = st.empty()
            ph_m_hom_1 = st.empty()
        with col_m2:
            ph_m_muj_1 = st.empty()
            ph_m_ind_1 = st.empty()
            
    with col_p2:
        ph_tit_2 = st.empty()
        ph_graf_2 = st.empty()
        col_m3, col_m4 = st.columns(2)
        with col_m3:
            ph_m_pob_2 = st.empty()
            ph_m_hom_2 = st.empty()
        with col_m4:
            ph_m_muj_2 = st.empty()
            ph_m_ind_2 = st.empty()

    # --- EL MOTOR MATEMÁTICO DE LAS PIRÁMIDES (Totalmente aislado del diseño) ---
    df_piramide_final = pd.DataFrame()

    # --- EL MOTOR MATEMÁTICO DE LAS PIRÁMIDES ---
    def generar_figura_piramide(año_obj, zona_str):
        try:
            pob_modelo_total = df_proj[df_proj['Año'] == año_obj][columna_modelo].values[0]
        except:
            return None, 0, 0, 0, 0, f"No hay proyección para el año {año_obj}.", pd.DataFrame()

        col_anio_pyr2 = 'año' if 'año' in df_nac.columns else 'Año'
        df_fnac_zona = df_nac[(df_nac[col_anio_pyr2] == año_obj) & (df_nac['area_geografica'] == zona_str.lower())].copy()
        df_fnac_total = df_nac[(df_nac[col_anio_pyr2] == año_obj) & (df_nac['area_geografica'] == 'total')].copy()
        
        if df_fnac_zona.empty or df_fnac_total.empty:
            return None, 0, 0, 0, 0, "No hay datos base nacionales para este año/zona.", pd.DataFrame()

        import re
        cols_h = [c for c in df_fnac_zona.columns if 'Hombre' in str(c) and any(char.isdigit() for char in str(c))]
        cols_m = [c for c in df_fnac_zona.columns if 'Mujer' in str(c) and any(char.isdigit() for char in str(c))]
        
        def extraer_edad(texto):
            nums = re.findall(r'\d+', texto)
            return int(nums[0]) if nums else 0

        datos_edades = []
        for col in cols_h:
            edad = extraer_edad(col)
            val_h = df_fnac_zona[col].values[0]
            col_mujer = next((c for c in cols_m if extraer_edad(c) == edad), None)
            val_m = df_fnac_zona[col_mujer].values[0] if col_mujer else 0
            datos_edades.append({'Edad': edad, 'Hombres': val_h, 'Mujeres': val_m})
            
        df_edades = pd.DataFrame(datos_edades)
        
        # ESCALADO
        pob_nac_total_real = df_fnac_total['Total'].values[0]
        factor_escala = (pob_modelo_total / pob_nac_total_real) if pob_nac_total_real > 0 else 0
            
        df_edades['Hom_Terr'] = df_edades['Hombres'] * factor_escala
        df_edades['Muj_Terr'] = df_edades['Mujeres'] * factor_escala
        
        df_pir = pd.DataFrame({'Edad': df_edades['Edad'], 'Hombres': df_edades['Hom_Terr'] * -1, 'Mujeres': df_edades['Muj_Terr']})
        
        cortes = list(range(0, 105, 5)) + [200]
        etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)] + ["100+"]
        df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
        df_pir_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()

        fig_pir = go.Figure()
        fig_pir.add_trace(go.Bar(y=df_pir_agrupado['Rango'], x=df_pir_agrupado['Hombres'], name='Hombres', orientation='h', marker_color='#3498db'))
        fig_pir.add_trace(go.Bar(y=df_pir_agrupado['Rango'], x=df_pir_agrupado['Mujeres'], name='Mujeres', orientation='h', marker_color='#e74c3c'))
        
        rango_max = max(abs(df_pir_agrupado['Hombres'].min()), df_pir_agrupado['Mujeres'].max()) if not df_pir_agrupado.empty else 100
        
        fig_pir.update_layout(
            barmode='relative', yaxis_title='Rango de Edad', xaxis_title='Población',
            xaxis=dict(range=[-rango_max*1.1, rango_max*1.1], tickvals=[-rango_max, 0, rango_max], ticktext=[f"{int(rango_max):,}", "0", f"{int(rango_max):,}"]),
            margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        total_hom = df_edades['Hom_Terr'].sum()
        total_muj = df_edades['Muj_Terr'].sum()
        total_zona = total_hom + total_muj
        ind_masculinidad = (total_hom / total_muj * 100) if total_muj > 0 else 0
        
        # Agregamos la tabla df_pir al final del return
        return fig_pir, total_zona, total_hom, total_muj, ind_masculinidad, None, df_pir

    # --- EL PINTOR DE LA INTERFAZ ---
    def actualizar_ui_piramides(año):
        global df_piramide_final
        
        fig1, t1, h1, m1, ind1, err1, df_export = generar_figura_piramide(año, "Total")
        if err1:
            ph_graf_1.warning(err1)
        else:
            df_piramide_final = df_export.copy()
            ph_tit_1.markdown(f"#### 🔵 Estructura Total ({año})")
            # SOLUCIÓN ID: Le damos un "key" único basado en el año
            ph_graf_1.plotly_chart(fig1, use_container_width=True, key=f"pir_1_{año}")
            ph_m_pob_1.metric("Pob. Total", f"{int(t1):,}".replace(",", "."))
            if t1 > 0:
                ph_m_hom_1.metric("Hombres", f"{int(h1):,}".replace(",", "."), f"{(h1/t1)*100:.1f}%")
                ph_m_muj_1.metric("Mujeres", f"{int(m1):,}".replace(",", "."), f"{(m1/t1)*100:.1f}%")
            ph_m_ind_1.metric("Índ. Masc.", f"{ind1:.1f}")

        fig2, t2, h2, m2, ind2, err2, _ = generar_figura_piramide(año, zona_comparacion)
        if err2:
            ph_graf_2.warning(err2)
        else:
            ph_tit_2.markdown(f"#### 🟢 Perfil {zona_comparacion} ({año})")
            # SOLUCIÓN ID: Le damos un "key" único a la segunda gráfica también
            ph_graf_2.plotly_chart(fig2, use_container_width=True, key=f"pir_2_{año}_{zona_comparacion}")
            ph_m_pob_2.metric(f"Pob. {zona_comparacion}", f"{int(t2):,}".replace(",", "."))
            if t2 > 0:
                ph_m_hom_2.metric("Hombres", f"{int(h2):,}".replace(",", "."), f"{(h2/t2)*100:.1f}%")
                ph_m_muj_2.metric("Mujeres", f"{int(m2):,}".replace(",", "."), f"{(m2/t2)*100:.1f}%")
            ph_m_ind_2.metric("Índ. Masc.", f"{ind2:.1f}")
            
    # === EJECUCIÓN ===
    if iniciar_animacion:
        for a in años_disp:
            if a >= 1985:
                actualizar_ui_piramides(a)
                time.sleep(velocidad_animacion)
    else:
        actualizar_ui_piramides(año_sel)
        
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
    * **Logístico:** Modela ecosistemas limitados. La población crece hasta encontrar resistencia ambiental, estabilizándose en una *Capacidad de Carga* ($K$). Es el modelo más robusto para planeación a largo plazo.
    * **Exponencial:** Asume recursos infinitos. Útil para modelar cortos períodos de "explosión demográfica" en centros urbanos nuevos.
    * **Lineal:** Representa tendencias promedio sin aceleración.

    ### 4. Fuentes de Información
    * **Capa 1 (Estructura Nacional):** Proyecciones y retroproyecciones oficiales DANE (1950-2070).
    * **Capa 2 (Masa Municipal):** Series censales DANE conciliadas (2005-2020).
    * **Capa 3 (Filtro Veredal):** Base de datos espaciales y tabulares Gobernación de Antioquia / IGAC.
    """)

# ==============================================================================
# 🧠 TRANSMISIÓN AL CEREBRO GLOBAL (EL ALEPH)
# ==============================================================================
if 'año_sel' in locals() and 'escala_sel' in locals():
    # Determinamos el lugar seleccionado según la escala
    lugar_activo = "Colombia"
    if "Departamental" in escala_sel: lugar_activo = depto_sel if 'depto_sel' in locals() else "Antioquia"
    elif "Municipal" in escala_sel: lugar_activo = municipio_sel if 'municipio_sel' in locals() else "Medellín"
    elif "Regional" in escala_sel: lugar_activo = region_sel if 'region_sel' in locals() else "Andina"
    elif "Cuencas" in escala_sel: lugar_activo = cuenca_sel if 'cuenca_sel' in locals() else "Río Grande"
    elif "Veredal" in escala_sel: lugar_activo = vereda_sel if 'vereda_sel' in locals() else "Centro"

    # Captura matemática infalible (Buscamos directamente en los vectores de la gráfica)
    pob_total_aleph = 0
    try:
        if 'años_hist' in locals() and 'pob_hist' in locals() and len(años_hist) > 0:
            import numpy as np
            # Buscamos la población que coincide con el año seleccionado
            idx = np.abs(np.array(años_hist) - año_sel).argmin()
            pob_total_aleph = pob_hist[idx]
    except Exception:
        pass

    # Inyectamos los datos en el Session State
    st.session_state['aleph_lugar'] = lugar_activo
    st.session_state['aleph_escala'] = escala_sel
    st.session_state['aleph_anio'] = año_sel
    st.session_state['aleph_pob_total'] = float(pob_total_aleph)

# 💉 INYECCIÓN AL TORRENTE SANGUÍNEO (Nuevas llaves para Pág 07 y 08)
    st.session_state['pob_hum_calc_met'] = float(pob_total_aleph)
    st.session_state[f'pob_asig_{lugar_activo}_met'] = float(pob_total_aleph)
    
    st.sidebar.success(f"🔗 Contexto demográfico de {lugar_activo} sincronizado.")
    
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
            modelos_sel = st.multiselect("Curvas a evaluar:", 
                                         ["Exponencial", "Logístico", "Geométrico", "Polinómico (Grado 2)", "Polinómico (Grado 3)"], 
                                         default=["Logístico", "Polinómico (Grado 2)"])
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
    titulo_seguro_mapa = locals().get('titulo_terr', globals().get('titulo_terr', "Territorio Seleccionado"))
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_seguro_mapa} ({año_sel})")
    
    # --- ESCUDO INFRANQUEABLE PARA ESCALA GLOBAL ---
    if escala_sel == "🌍 Global y Suramérica":
        st.info("🌍 A escala Global/Suramérica la visualización espacial se encuentra desactivada. Los datos consolidados están disponibles en el panel de tendencias.")
        # La ejecución del mapa muere aquí para la escala global (Evita el KeyError)
    
    # --- LÓGICA ESPACIAL PARA EL RESTO DE ESCALAS ---
    else:
        # Mini-menú integrado y estético
        col_m1, col_m2 = st.columns([1, 4])
        with col_m1:
            if escala_sel != "🌿 Veredal (Antioquia)":
                area_mapa = st.radio("Filtro Poblacional:", ["Total", "Urbano", "Rural"])
            else:
                area_mapa = "Rural"
                st.info("ℹ️ A escala veredal, toda la población se modela como rural.")
        with col_m2:
            st.success("🤖 **Motor Topológico Automático:** Conectando capas espaciales (GeoJSON) con matrices demográficas.")

        if 'año' in df_mapa_base.columns:
            df_mapa_año = df_mapa_base[df_mapa_base['año'] == min(max(df_mapa_base['año']), año_sel)].copy()
        else:
            df_mapa_año = df_mapa_base.copy()

        if df_mapa_año.empty:
            df_mapa_plot = pd.DataFrame()
        else:
            if area_mapa == "Total":
                cols_agrupar = [c for c in ['Territorio', 'Padre'] if c in df_mapa_año.columns]
                if cols_agrupar:
                    df_mapa_plot = df_mapa_año.groupby(cols_agrupar)['Total'].sum().reset_index()
                else:
                    df_mapa_plot = df_mapa_año.copy()
            else:
                if 'area_geografica' in df_mapa_año.columns:
                    df_mapa_plot = df_mapa_año[df_mapa_año['area_geografica'] == area_mapa.lower()].copy()
                else:
                    df_mapa_plot = df_mapa_año.copy()
                    
            if 'Territorio' in df_mapa_plot.columns:
                df_mapa_plot = df_mapa_plot[df_mapa_plot['Territorio'].astype(str).str.upper() != 'TOTAL']
                if escala_sel == "💧 Cuencas Hidrográficas" and not df_mapa_plot.empty:
                    df_mapa_plot = df_mapa_plot[~df_mapa_plot['Territorio'].astype(str).str.contains('CABECERA', case=False, na=False)]

        if not df_mapa_plot.empty:
            # Estandarización de columnas a prueba de balas
            if 'Territorio' not in df_mapa_plot.columns:
                col_t = next((c for c in df_mapa_plot.columns if c.lower() in ['municipio', 'cuenca', 'vereda', 'nombre', 'subzona', 'nom_nss3']), df_mapa_plot.columns[0])
                df_mapa_plot = df_mapa_plot.rename(columns={col_t: 'Territorio'})
            if 'Padre' not in df_mapa_plot.columns:
                col_p = next((c for c in df_mapa_plot.columns if c.lower() in ['padre', 'depto_nom', 'departamento', 'macroregion', 'zona']), None)
                if col_p: df_mapa_plot = df_mapa_plot.rename(columns={col_p: 'Padre'})
                else: df_mapa_plot['Padre'] = ""
            if 'Total' not in df_mapa_plot.columns:
                col_tot = next((c for c in df_mapa_plot.columns if c.lower() in ['total', 'poblacion', 'pob', 'habitantes', 'valor']), df_mapa_plot.columns[-1])
                df_mapa_plot = df_mapa_plot.rename(columns={col_tot: 'Total'})

            try:
                import json
                import geopandas as gpd
                from sqlalchemy import text
                from modules.db_manager import get_engine
                
                engine_geo = get_engine()
                
                # --- LÓGICA ESPACIAL AUTOMÁTICA ---
                if "veredal" in escala_sel.lower(): 
                    q_geo = text("SELECT * FROM veredas_geometria")
                elif "cuencas" in escala_sel.lower(): 
                    q_geo = text("SELECT * FROM cuencas")
                else: 
                    q_geo = text("SELECT * FROM municipios")
                    
                gdf_mapa = gpd.read_postgis(q_geo, engine_geo, geom_col="geometry")
                
                # 🔥 FIX POLÍGONOS INTRUSOS: Filtramos el GeoJSON por jerarquía topológica estricta
                if "cuencas" in escala_sel.lower() and not df_mapa_plot.empty:
                    # Extraemos los NOMBRES y los PADRES para un cruce topológico perfecto
                    df_validos = df_mapa_plot[['Territorio', 'Padre']].drop_duplicates()
                    
                    if col_res in gdf_mapa.columns:
                        # Cortamos el mapa: Solo dibujamos polígonos cuyo "Padre" (ej. Magdalena Cauca) sea el que seleccionaste
                        padres_seleccionados = df_validos['Padre'].unique().tolist()
                        
                        # Manejo agresivo de tildes para no perder a Cornare ni al Caribe
                        gdf_mapa['padre_norm'] = gdf_mapa[col_res].astype(str).apply(normalizar_texto)
                        padres_norm = [normalizar_texto(str(p)) for p in padres_seleccionados]
                        
                        gdf_mapa = gdf_mapa[gdf_mapa['padre_norm'].isin(padres_norm)]
                        
                        # Limpiamos las columnas auxiliares
                        gdf_mapa = gdf_mapa.drop(columns=['padre_norm'], errors='ignore')

                if not gdf_mapa.empty:
                    # 1. MATCH_ID Dinámico en Pandas
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot.apply(
                        lambda row: normalizar_texto(row['Territorio']) + "_" + normalizar_texto(row['Padre']) 
                        if str(row['Padre']).strip() and "cuencas" not in escala_sel.lower() 
                        else normalizar_texto(row['Territorio']), axis=1
                    )
                    
                    codigos_dane_deptos = {
                        "05": "ANTIOQUIA", "08": "ATLANTICO", "11": "BOGOTA", "13": "BOLIVAR", 
                        "15": "BOYACA", "17": "CALDAS", "18": "CAQUETA", "19": "CAUCA", 
                        "20": "CESAR", "23": "CORDOBA", "25": "CUNDINAMARCA", "27": "CHOCO", 
                        "41": "HUILA", "44": "GUAJIRA", "47": "MAGDALENA", "50": "META", 
                        "52": "NARINO", "54": "NORTEDESANTANDER", "63": "QUINDIO", "66": "RISARALDA", 
                        "68": "SANTANDER", "70": "SUCRE", "73": "TOLIMA", "76": "VALLEDELCAUCA",
                        "81": "ARAUCA", "85": "CASANARE", "86": "PUTUMAYO", "88": "ARCHIPIELAGODESANANDRES",
                        "91": "AMAZONAS", "94": "GUAINIA", "95": "GUAVIARE", "97": "VAUPES", "99": "VICHADA"
                    }
                    
                    # 2. MATCH_ID Dinámico en el GeoDataFrame
                    def generar_id_geojson(row):
                        if "cuencas" in escala_sel.lower():
                            cols_posibles = ['nom_nss3', 'nom_nss2', 'nom_nss1', 'nom_szh', 'nomzh', 'nomah', 'NOM_NSS3', 'NOM_NSS2', 'NOM_NSS1']
                            val_terr = next((str(row[c]) for c in cols_posibles if c in row and pd.notnull(row[c])), "")
                            
                            # 🛡️ FIX CUENCAS: Si el polígono tiene un guión ("Río Aburrá - Q. La Iguaná"), extraemos solo la parte final
                            if "-" in str(val_terr):
                                val_terr = str(val_terr).split("-")[-1]
                                
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
                    
                    # --- 3. AUTO-SANADOR Y ENCUADRE MILIMÉTRICO (Bounding Box) ---
                    ids_geojson = gdf_mapa['MATCH_ID'].dropna().unique().tolist()
                    df_mapa_plot['En_Mapa'] = df_mapa_plot['MATCH_ID'].isin(ids_geojson)
                    
                    # 🛠️ MAGIA IA: Reparación automática de nombres usando similitud (Fuzzy Matching)
                    faltantes_iniciales = df_mapa_plot[df_mapa_plot['En_Mapa'] == False]
                    if not faltantes_iniciales.empty:
                        import difflib
                        for idx, row in faltantes_iniciales.iterrows():
                            # Busca la coincidencia más cercana en el GeoJSON (Ej: LAHONDA vs LAONDA)
                            matches = difflib.get_close_matches(row['MATCH_ID'], ids_geojson, n=1, cutoff=0.85)
                            if matches:
                                df_mapa_plot.at[idx, 'MATCH_ID'] = matches[0] # ¡Curado!
                                df_mapa_plot.at[idx, 'En_Mapa'] = True
                    
                    # Volvemos a extraer los IDs ya curados
                    ids_curados = df_mapa_plot[df_mapa_plot['En_Mapa'] == True]['MATCH_ID'].unique()
                    gdf_filtrado = gdf_mapa[gdf_mapa['MATCH_ID'].isin(ids_curados)]
                    
                    center_lat, center_lon, zoom_level = 4.57, -74.29, 5
                    if not gdf_filtrado.empty:
                        gdf_4326 = gdf_filtrado.to_crs(epsg=4326)
                        minx, miny, maxx, maxy = gdf_4326.total_bounds
                        center_lon = (minx + maxx) / 2
                        center_lat = (miny + maxy) / 2
                        
                        max_diff = max(maxx - minx, maxy - miny)
                        if max_diff < 0.1: zoom_level = 11.5
                        elif max_diff < 0.3: zoom_level = 10
                        elif max_diff < 0.8: zoom_level = 8.5
                        elif max_diff < 2.5: zoom_level = 7
                        elif max_diff < 5.0: zoom_level = 6
                        else: zoom_level = 5

                    geo_data = json.loads(gdf_mapa.to_json())
                    q_val = 0.85 if area_mapa == "Total" else 0.90
                    
                    # Usamos el dataframe final ya curado para el máximo color
                    df_mapa_plot_final = df_mapa_plot[df_mapa_plot['En_Mapa'] == True].copy()
                    max_color = df_mapa_plot_final['Total'].quantile(q_val) if len(df_mapa_plot_final) > 10 else (df_mapa_plot_final['Total'].max() if not df_mapa_plot_final.empty else 1)
                    
                    # 4. RENDERIZADO DEL MAPA
                    import plotly.express as px
                    fig_mapa = px.choropleth_mapbox(
                        df_mapa_plot, # Usamos el df con los IDs actualizados y curados
                        geojson=geo_data,
                        locations='MATCH_ID',        
                        featureidkey='properties.MATCH_ID', 
                        color='Total',
                        color_continuous_scale="Viridis",
                        range_color=[0, max_color],  
                        mapbox_style="carto-positron",
                        zoom=zoom_level, 
                        center={"lat": center_lat, "lon": center_lon},
                        opacity=0.8,
                        labels={'Total': f'Población {area_mapa}'},
                        hover_data={'Total': ':,.0f', 'MATCH_ID': False, 'Territorio': True, 'Padre': True}
                    )
                    
                    fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
                    st.plotly_chart(fig_mapa, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                    
                    # --- DEPURADOR FORENSE INYECTADO (SOLO MUESTRA LOS INCURABLES) ---
                    faltantes_finales = df_mapa_plot[df_mapa_plot['En_Mapa'] == False]
                    
                    if not faltantes_finales.empty:
                        st.warning(f"⚠️ {len(faltantes_finales)} territorios no cruzaron incluso después de la sanación automática.")
                        with st.expander("🔍 ABRIR DEPURADOR FORENSE (Para resolver por qué no cruzó)"):
                            st.markdown("El mapa solo se dibuja si el ID de la Tabla es idéntico al ID del Polígono. Compara ambas listas:")
                            col_dbg1, col_dbg2 = st.columns(2)
                            with col_dbg1:
                                st.write("🔴 **Lo que la Tabla está buscando:**")
                                st.dataframe(faltantes_finales[['Territorio', 'MATCH_ID']], use_container_width=True)
                            with col_dbg2:
                                st.write("🟢 **Lo que el Mapa (GeoJSON) tiene disponible:**")
                                ids_disponibles = sorted(ids_geojson)
                                st.dataframe(pd.DataFrame({"IDs en PostGIS": ids_disponibles}), use_container_width=True)
                else:
                    st.warning("⚠️ No se encontraron geometrías en la base de datos para dibujar.")
            except Exception as e:
                st.error(f"❌ Error conectando a PostGIS o procesando el mapa: {e}")
        else:
            st.warning("⚠️ Esperando datos poblacionales del panel lateral...")
            
# =====================================================================
# PESTAÑA 4: GENERADOR DE MATRIZ MAESTRA (TOP-DOWN) MULTIMODELO CON R²
# =====================================================================
with tab_matriz:
    st.subheader("🧠 Motor Generador de Matriz Maestra Demográfica (Total, Urbano y Rural)")
    st.markdown("""
    Este motor entrena simultáneamente tres modelos matemáticos predictivos y recomienda el de mejor ajuste:
    * **Logístico:** Ideal para poblaciones que alcanzan un techo por límites físicos o recursos.
    * **Exponencial:** Ideal para poblaciones en crecimiento o decrecimiento libre constante.
    * **Polinomial (Grado 3):** Ideal para poblaciones con fluctuaciones o declives no lineales.
    """)
    
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
                    k_max = max_y * 3.0 if es_creciente else max_y * 1.1
                    k_guess = max_y * 1.2 if es_creciente else (y[-1] * 0.9 if y[-1] > 0 else max_y)
                    a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                    r_guess = 0.02 if es_creciente else -0.02
                    k_min = max_y * 0.8 if es_creciente else max_y * 0.1
                    limites = ([k_min, 0, -0.2], [k_max, np.inf, 0.3])
                    
                    popt_log, _ = curve_fit(f_log, x_norm, y, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
                    log_k, log_a, log_r = popt_log
                    log_r2 = calcular_r2(y, f_log(x_norm, *popt_log))
                except Exception: pass

                # 2. EXPONENCIAL
                exp_a, exp_b, exp_r2 = 0, 0, 0
                try:
                    popt_exp, _ = curve_fit(f_exp, x_norm, y, p0=[p0_val, 0.01], maxfev=50000)
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

                dic_modelos = {'Logístico': log_r2, 'Exponencial': exp_r2, 'Polinomial_3': poly_r2}
                mejor_modelo = max(dic_modelos, key=dic_modelos.get)
                mejor_r2 = dic_modelos[mejor_modelo]

                matriz_resultados.append({
                    'Area': area, 'Nivel': nivel, 'Territorio': territorio, 'Padre': padre,
                    'Año_Base': int(x_offset), 'Pob_Base': round(p0_val, 0),
                    'Log_K': log_k, 'Log_a': log_a, 'Log_r': log_r, 'Log_R2': round(log_r2, 4),
                    'Exp_a': exp_a, 'Exp_b': exp_b, 'Exp_R2': round(exp_r2, 4),
                    'Poly_A': poly_A, 'Poly_B': poly_B, 'Poly_C': poly_C, 'Poly_D': poly_D, 'Poly_R2': round(poly_r2, 4),
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

            # --- 🕵️‍♂️ RECOLECCIÓN MASIVA DE CUENCAS DE LA BASE DE DATOS ---
            q_todas = text("SELECT DISTINCT nom_nss3 FROM cuencas WHERE nom_nss3 IS NOT NULL")
            lista_todas_cuencas = pd.read_sql(q_todas, engine_sql)['nom_nss3'].tolist()
            
            # --- ⚙️ MOTOR DE PROGRESO UI ---
            mpios = df_mun_memoria['municipio'].dropna().unique()
            deptos = df_mun_memoria['depto_nom'].dropna().unique()
            areas_a_procesar = ['Total', 'Urbana', 'Rural']
            
            total_ops = len(areas_a_procesar) * (1 + len(deptos) + len(mpios) + len(lista_todas_cuencas))
            ops_completadas = 0

            for tipo_area in areas_a_procesar:
                df_area_actual = df_mun_memoria[df_mun_memoria['Categoria_Area'] == tipo_area]
                if df_area_actual.empty: 
                    ops_completadas += (1 + len(deptos) + len(mpios) + len(lista_todas_cuencas))
                    continue
                
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
                # 🧠 BISTURÍ ESPACIAL V3: Alta Precisión Dasimétrica y Sanación Fuzzy
                # ================================================================
                try:
                    import geopandas as gpd
                    from sqlalchemy import text
                    import unicodedata
                    import difflib
                    import re
                    from modules.db_manager import get_engine
                    
                    engine_geo = get_engine()
                    
                    q_cue = text("""
                        SELECT COALESCE(nom_nss3, nom_nss2, nom_nss1, nom_szh) AS subc_lbl, 
                               geometry 
                        FROM cuencas 
                        WHERE COALESCE(nom_nss3, nom_nss2, nom_nss1, nom_szh) IS NOT NULL
                    """)
                    gdf_cue = gpd.read_postgis(q_cue, engine_geo, geom_col="geometry").to_crs(epsg=3116)
                    gdf_cue['geometry'] = gdf_cue.geometry.buffer(0)
                    
                    if tipo_area == 'Urbana': q_esp = text("SELECT * FROM cabeceras_municipales")
                    else: q_esp = text("SELECT * FROM municipios")

                    gdf_esp = gpd.read_postgis(q_esp, engine_geo, geom_col="geometry")
                    col_mpio_das = next((c for c in gdf_esp.columns if c.lower() in ['mpio_cnmbr', 'nombre_mpi', 'mpio_nombr', 'nombre_municipio', 'municipio', 'nomb_mpio', 'mun_name']), None)
                    if col_mpio_das: gdf_esp = gdf_esp.rename(columns={col_mpio_das: 'mun_name'})
                    
                    gdf_esp = gdf_esp.to_crs(epsg=3116)
                    gdf_esp['geometry'] = gdf_esp.geometry.buffer(0)
                    
                    def clean_das(t):
                        if not t or pd.isna(t): return ""
                        t = str(t).lower().strip()
                        t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
                        return re.sub(r'[^a-z0-9]', '', t)
                    
                    df_area_actual_esp = df_area_actual.copy()
                    df_area_actual_esp['mun_norm_dane'] = df_area_actual_esp['municipio'].apply(clean_das)
                    lista_nombres_dane_clean = df_area_actual_esp['mun_norm_dane'].dropna().unique().tolist()

                    def sanar_nombre_espacial(nombre_gis):
                        n_clean = clean_das(nombre_gis)
                        if n_clean in lista_nombres_dane_clean: return n_clean
                        matches = difflib.get_close_matches(n_clean, lista_nombres_dane_clean, n=1, cutoff=0.7)
                        return matches[0] if matches else n_clean

                    gdf_esp['mun_norm'] = gdf_esp['mun_name'].apply(sanar_nombre_espacial)
                    
                    gdf_esp['area_poly'] = gdf_esp.geometry.area
                    esp_areas = gdf_esp.groupby('mun_norm')['area_poly'].sum().reset_index().rename(columns={'area_poly': 'area_esponja'})
                    
                    inter = gpd.overlay(gdf_esp, gdf_cue, how='intersection')
                    inter['area_frag'] = inter.geometry.area
                    
                    inter_grouped = inter.groupby(['mun_norm', 'subc_lbl'])['area_frag'].sum().reset_index()
                    inter_final = inter_grouped.merge(esp_areas, on='mun_norm')
                    inter_final['proporcion'] = (inter_final['area_frag'] / inter_final['area_esponja']).clip(upper=1.0)
                    
                    df_inter = inter_final.merge(df_area_actual_esp, left_on='mun_norm', right_on='mun_norm_dane', how='inner')
                    df_inter['Total_frag'] = df_inter['Total'] * df_inter['proporcion']
                    
                    df_cuencas = df_inter.groupby(['subc_lbl', col_anio])['Total_frag'].sum().reset_index()
                    
                    for cuenca in lista_todas_cuencas:
                        df_temp = df_cuencas[df_cuencas['subc_lbl'] == cuenca].sort_values(by=col_anio)
                        if not df_temp.empty and df_temp['Total_frag'].sum() > 0:
                            ajustar_modelos(df_temp[col_anio].values, df_temp['Total_frag'].values, 'Cuenca', cuenca, 'Antioquia', tipo_area)
                        
                        ops_completadas += 1
                        
                        if ops_completadas % 5 == 0:
                            porcentaje = min(ops_completadas / total_ops, 1.0)
                            barra_progreso.progress(porcentaje)
                            elapsed = time.time() - start_time
                            eta = max((elapsed / ops_completadas) * total_ops - elapsed, 0)
                            mins, secs = divmod(int(eta), 60)
                            texto_progreso.markdown(f"**Dasimetría Espacial:** {cuenca} ({tipo_area})... | **ETA:** {mins}m {secs}s")

                except Exception as e:
                    st.warning(f"⚠️ Nota en proceso dasimétrico ({tipo_area}): {e}")

            # 4. Finalización y Carga en Sesión
            if matriz_resultados:
                df_masivo = pd.DataFrame(matriz_resultados)
                barra_progreso.progress(1.0)
                texto_progreso.success(f"✅ ¡Entrenamiento Masivo Completado! {len(df_masivo)} modelos generados con éxito.")
                st.session_state['df_matriz_demografica'] = df_masivo
                
                st.info("💡 Ve a la pestaña **💾 Descargas** o usa el panel de **Administración: Inyectar a SQL** para hacer los cambios permanentes en Supabase.")
            else:
                texto_progreso.warning("⚠️ No se generaron resultados. Verifica la conexión a la base de datos.")

        # 🛑 AQUÍ ESTÁ EL CIERRE QUE FALTABA (SALVAVIDAS DE PYTHON)
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento masivo: {e}")
            
    # =====================================================================
    # 🔬 VALIDADOR VISUAL COMPARATIVO (DOBLE VENTANA)
    # =====================================================================
                    
    # =====================================================================
    # 🔬 VALIDADOR VISUAL COMPARATIVO (DOBLE VENTANA)
    # =====================================================================
    if 'df_matriz_demografica' in st.session_state and 'Area' in st.session_state['df_matriz_demografica'].columns:
        st.divider()
        st.subheader("🔬 Validador Visual Comparativo (Urbano vs Rural vs Total)")
        
        df_mat = st.session_state['df_matriz_demografica']
        
        c_nav1, c_nav2, c_nav3 = st.columns([1, 1.5, 1])
        with c_nav1:
            niveles_disp = list(df_mat['Nivel'].unique())
            idx_mun = niveles_disp.index('Municipal') if 'Municipal' in niveles_disp else 0
            nivel_val = st.selectbox("1. Nivel de Análisis:", niveles_disp, index=idx_mun)
        with c_nav2:
            territorios_disp = sorted(df_mat[df_mat['Nivel'] == nivel_val]['Territorio'].unique())
            idx_terr = territorios_disp.index('BELMIRA') if 'BELMIRA' in territorios_disp else 0
            terr_val = st.selectbox("2. Territorio (Municipio/Depto):", territorios_disp, index=idx_terr)
        with c_nav3:
            anio_futuro = st.slider("3. Proyectar hasta el año:", min_value=2025, max_value=2100, value=2050, step=5)
            
        st.markdown("---")
        
        def renderizar_panel(area_sel, key_suffix):
            import numpy as np
            import plotly.graph_objects as go
            
            df_filtrado = df_mat[(df_mat['Nivel'] == nivel_val) & (df_mat['Territorio'] == terr_val) & (df_mat['Area'] == area_sel)]
            if df_filtrado.empty:
                st.warning(f"No hay datos procesados para el área {area_sel} en {terr_val}.")
                return
                
            fila_terr = df_filtrado.iloc[0]
            mejor_modelo = fila_terr['Modelo_Recomendado']
            
            df_mun_memoria = df_mun.copy() 
            col_anio = 'año' if 'año' in df_mun_memoria.columns else 'Año'
            
            def clasificar_area(val):
                v = str(val).lower()
                if 'total' in v: return 'Total'
                if 'cabecera' in v or 'urban' in v: return 'Urbana'
                if 'rural' in v or 'centros' in v or 'resto' in v: return 'Rural'
                return 'Desconocido'
                
            df_mun_memoria['Categoria_Area'] = df_mun_memoria['area_geografica'].apply(clasificar_area)
            
            # 🔥 Seleccionamos el área puramente
            df_hist_base = df_mun_memoria[df_mun_memoria['Categoria_Area'] == area_sel]
            
            if nivel_val == 'Nacional': df_hist = df_hist_base.groupby(col_anio)['Total'].sum().reset_index()
            elif nivel_val == 'Departamental': df_hist = df_hist_base[df_hist_base['depto_nom'] == terr_val].groupby(col_anio)['Total'].sum().reset_index()
            else: df_hist = df_hist_base[df_hist_base['municipio'] == terr_val].groupby(col_anio)['Total'].sum().reset_index()
                
            df_hist = df_hist.sort_values(by=col_anio)
            x_hist = df_hist[col_anio].values
            y_hist = df_hist['Total'].values
            
            x_offset = fila_terr['Año_Base']
            x_pred = np.arange(x_offset, anio_futuro + 1)
            x_norm_pred = x_pred - x_offset
            
            y_log = fila_terr['Log_K'] / (1 + fila_terr['Log_a'] * np.exp(-fila_terr['Log_r'] * x_norm_pred))
            y_exp = fila_terr['Exp_a'] * np.exp(fila_terr['Exp_b'] * x_norm_pred)
            y_poly = fila_terr['Poly_A']*(x_norm_pred**3) + fila_terr['Poly_B']*(x_norm_pred**2) + fila_terr['Poly_C']*x_norm_pred + fila_terr['Poly_D']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Histórico DANE', marker=dict(color='black', size=8, symbol='diamond')))
            
            def config_linea(nombre_mod, color):
                es_ganador = mejor_modelo == nombre_mod
                return dict(color=color, width=4 if es_ganador else 2, dash='solid' if es_ganador else 'dash'), 1.0 if es_ganador else 0.4
                
            line_log, op_log = config_linea('Logístico', '#2980b9')
            fig.add_trace(go.Scatter(x=x_pred, y=y_log, mode='lines', name=f"Logístico (R²: {fila_terr['Log_R2']})", line=line_log, opacity=op_log))
            
            line_exp, op_exp = config_linea('Exponencial', '#e67e22')
            fig.add_trace(go.Scatter(x=x_pred, y=y_exp, mode='lines', name=f"Exponencial (R²: {fila_terr['Exp_R2']})", line=line_exp, opacity=op_exp))
            
            line_poly, op_poly = config_linea('Polinomial_3', '#27ae60')
            fig.add_trace(go.Scatter(x=x_pred, y=y_poly, mode='lines', name=f"Polinomial 3 (R²: {fila_terr['Poly_R2']})", line=line_poly, opacity=op_poly))
            
            fig.update_layout(
                title=f"Proyección {area_sel} (Ganador: {mejor_modelo})", 
                xaxis_title="Año", 
                yaxis_title="Habitantes", 
                hovermode="x unified", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander(f"📐 Parámetros y Ecuaciones del Modelo {area_sel}", expanded=True):
                st.markdown(f"**Donde la variable tiempo es:** $t = Año\_Proyectado - {fila_terr['Año_Base']}$")
                st.latex(r"Log\text{\'{i}}stico: P(t) = \frac{K}{1 + a \cdot e^{-r \cdot t}}")
                st.latex(r"Exponencial: P(t) = a \cdot e^{b \cdot t}")
                st.latex(r"Polinomial: P(t) = A \cdot t^3 + B \cdot t^2 + C \cdot t + D")
                
                df_coefs = pd.DataFrame([
                    {"Modelo": "Logístico", "R²": f"{fila_terr['Log_R2']:.4f}", "Parámetros": f"K={fila_terr['Log_K']:.0f}, a={fila_terr['Log_a']:.4f}, r={fila_terr['Log_r']:.4f}"},
                    {"Modelo": "Exponencial", "R²": f"{fila_terr['Exp_R2']:.4f}", "Parámetros": f"a={fila_terr['Exp_a']:.0f}, b={fila_terr['Exp_b']:.4f}"},
                    {"Modelo": "Polinomial 3", "R²": f"{fila_terr['Poly_R2']:.4f}", "Parámetros": f"A={fila_terr['Poly_A']:.4e}, B={fila_terr['Poly_B']:.4e}, C={fila_terr['Poly_C']:.4f}, D={fila_terr['Poly_D']:.0f}"}
                ])
                def highlight_winner(row): return ['background-color: #d4edda' if row['Modelo'] == mejor_modelo else '' for _ in row]
                st.dataframe(df_coefs.style.apply(highlight_winner, axis=1), use_container_width=True)

        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1:
            area_1 = st.selectbox("Área de Análisis (Panel Izquierdo):", ["Total", "Urbana", "Rural"], index=0, key="sel_a1")
            renderizar_panel(area_1, "g1")
        with col_graf_2:
            area_2 = st.selectbox("Área de Análisis (Panel Derecho):", ["Total", "Urbana", "Rural"], index=1, key="sel_a2")
            renderizar_panel(area_2, "g2")
            
# ==========================================
# PESTAÑA 5: RANKINGS Y DINÁMICA HISTÓRICA (Top 15 y 2005-2035)
# ==========================================
with tab_rankings:
    # Extraemos la zona actual (Total, Urbano o Rural)
    zona_actual = "Total"
    if not df_mapa_base.empty and 'area_geografica' in df_mapa_base.columns:
        zona_actual = df_mapa_base['area_geografica'].iloc[0].title()
        
    st.subheader(f"📊 Análisis Comparativo y Trayectorias Poblacionales ({zona_actual})")
    
    zona_q = zona_actual.lower()
    
    # --- LA MAGIA: CONSTRUIMOS EL RANKING DE FORMA INDEPENDIENTE AL MAPA ---
    df_rank = pd.DataFrame()
    titulo_ranking = ""
    
    # --- FIX DEFINITIVO: RANKINGS DINÁMICOS CON ESCUDO DE VARIABLES ---
    if "Global" in escala_sel or "Nacional" in escala_sel:
        df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q)].groupby('depto_nom')['Total'].sum().reset_index()
        df_rank.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
        titulo_ranking = "Departamentos"
        
    elif "Departamental" in escala_sel or "Municipal (Departamentos)" in escala_sel:
        # Escudo: busca el nuevo nombre (agrupador_sel) o usa ANTIOQUIA por defecto
        padre_seguro = locals().get('agrupador_sel', globals().get('agrupador_sel', "ANTIOQUIA"))
        df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q) & (df_mun['depto_nom'] == padre_seguro)].groupby('municipio')['Total'].sum().reset_index()
        df_rank.rename(columns={'municipio': 'Territorio'}, inplace=True)
        titulo_ranking = f"Municipios de {padre_seguro}"
        
    elif "Municipal (Regiones)" in escala_sel:
        padre_seguro = locals().get('agrupador_sel', globals().get('agrupador_sel', "Oriente"))
        if 'Macroregion' in df_mun.columns:
            df_rank = df_mun[(df_mun['año'] == año_sel) & (df_mun['area_geografica'] == zona_q) & (df_mun['Macroregion'] == padre_seguro)].groupby('municipio')['Total'].sum().reset_index()
        else:
            df_rank = pd.DataFrame(columns=['municipio', 'Total'])
        df_rank.rename(columns={'municipio': 'Territorio'}, inplace=True)
        titulo_ranking = f"Municipios de {padre_seguro}"
        
    else:
        # Para Veredas o Cuencas, usamos directamente la base del mapa
        if 'df_mapa_base' in locals() and not df_mapa_base.empty and 'Territorio' in df_mapa_base.columns:
            df_rank = df_mapa_base.groupby('Territorio')['Total'].sum().reset_index()
        else:
            df_rank = pd.DataFrame(columns=['Territorio', 'Total'])
        titulo_ranking = "Territorios Seleccionados"
        
    # Escudo Numérico Universal para la Pestaña 5
    if not df_rank.empty and 'Total' in df_rank.columns:
        df_rank['Total'] = pd.to_numeric(df_rank['Total'], errors='coerce').fillna(0)
        df_rank = df_rank[df_rank['Total'] > 0]
        
    # Procedemos solo si hay datos para armar gráficas
    if not df_rank.empty and len(df_rank) > 1:
        
        # -------------------------------------------------------------------
        # SECCIÓN 1: RANKINGS LADO A LADO (Escalera Perfecta a Prueba de Balas)
        # -------------------------------------------------------------------
        st.markdown(f"### 🏆 Extremos Poblacionales ({año_sel}) - {titulo_ranking}")
        
        col_rank_izq, col_rank_der = st.columns(2)
        
        with col_rank_izq:
            # Top 15: Forzamos 'total ascending' para que el mayor quede arriba
            df_plot_top = df_rank.nlargest(15, 'Total')
            fig_top = px.bar(df_plot_top, x='Total', y='Territorio', orientation='h', 
                             color='Total', color_continuous_scale='Viridis',
                             title="📈 Top 15: Mayor Población")
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, height=450, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_top, use_container_width=True, key=f"rank_top_{año_sel}_{zona_actual}")

        with col_rank_der:
            # Bottom 15: Forzamos 'total descending' para que el menor quede arriba
            if len(df_rank) > 5:
                df_plot_bot = df_rank.nsmallest(15, 'Total')
                fig_bot = px.bar(df_plot_bot, x='Total', y='Territorio', orientation='h', 
                                 color='Total', color_continuous_scale='Plasma',
                                 title="📉 Bottom 15: Menor Población")
                fig_bot.update_layout(yaxis={'categoryorder':'total descending'}, height=450, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_bot, use_container_width=True, key=f"rank_bot_{año_sel}_{zona_actual}")

        # -------------------------------------------------------------------
        # SECCIÓN 2: CURVAS HISTÓRICAS (Totalmente Funcionales)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📈 Dinámica Poblacional (2005 - 2035)")
        
        if "Veredal" in escala_sel or "Cuencas" in escala_sel:
            # Muestra el mensaje informativo
            st.info("ℹ️ A escala Veredal/Cuencas, la plataforma utiliza un corte censal oficial estático. Las curvas de proyección dinámica 2005-2035 se activan desde la escala municipal hacia arriba.")
        else:
            # Obtenemos los 10 líderes del ranking independiente que acabamos de crear
            top_10_nombres = df_rank.nlargest(10, 'Total')['Territorio'].tolist()
            
            df_base_historica = df_mun[df_mun['area_geografica'] == zona_q].copy()
            df_line = pd.DataFrame()
            
            if "Nacional" in escala_sel:
                df_line = df_base_historica.groupby(['año', 'depto_nom'])['Total'].sum().reset_index()
                df_line.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
                
            elif "Departamental" in escala_sel or "Municipal (Departamentos)" in escala_sel:
                # Usamos el escudo también aquí para evitar NameError
                padre_seguro = locals().get('agrupador_sel', globals().get('agrupador_sel', "ANTIOQUIA"))
                df_base_historica = df_base_historica[df_base_historica['depto_nom'] == padre_seguro]
                df_line = df_base_historica.groupby(['año', 'municipio'])['Total'].sum().reset_index()
                df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)

            elif "Regional" in escala_sel or "Municipal (Regiones)" in escala_sel:
                padre_seguro = locals().get('agrupador_sel', globals().get('agrupador_sel', "Oriente"))
                if 'Macroregion' in df_base_historica.columns:
                    df_base_historica = df_base_historica[df_base_historica['Macroregion'] == padre_seguro]
                    df_line = df_base_historica.groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)
            
            if not df_line.empty:
                # Filtramos las curvas para que solo muestren a los del Top 10
                df_line = df_line[df_line['Territorio'].isin(top_10_nombres)]
                
                # Suavizador para la gráfica
                def conciliacion_censal(group):
                    group['año'] = pd.to_numeric(group['año'], errors='coerce')
                    group = group.sort_values('año')
                    mask = (group['año'] >= 2006) & (group['año'] <= 2019)
                    group.loc[mask, 'Total'] = np.nan
                    # Usamos solo interpolate de Pandas sobre la columna
                    group['Total'] = group['Total'].interpolate(method='linear').ffill().bfill()
                    return group
                    
                df_line = df_line.groupby('Territorio', group_keys=False).apply(conciliacion_censal)
                
                fig_line = px.line(df_line, x='año', y='Total', color='Territorio', markers=True,
                                   title="Evolución de los 10 Territorios más Poblados",
                                   labels={'año': 'Año', 'Total': 'Habitantes'})
                
                st.plotly_chart(fig_line, use_container_width=True, key=f"line_hist_{zona_actual}")

    else:
        st.info("💡 Selecciona una escala territorial con múltiples divisiones para ver el ranking y las curvas comparativas.")
        
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
# Este bloque debe estar FUERA del if st.button("Entrenar...")
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
            st.warning("⚠️ Acción destructiva: Reemplazará la matriz maestra en Supabase.")
            pwd = st.text_input("Clave de Administrador:", type="password", key="pwd_sql_demo")
            
            if st.button("🚀 Confirmar Inyección", type="primary", use_container_width=True):
                # Usa la misma clave que en el Generador Beta (Agua2026)
                if pwd == st.secrets.get("admin_password", "Agua2026"):
                    with st.spinner("Conectando con PostgreSQL (Supabase)..."):
                        try:
                            from modules.db_manager import get_engine
                            engine_sql = get_engine()
                            df_matriz_demo.to_sql('matriz_maestra_demografica', engine_sql, if_exists='replace', index=False)
                            st.success(f"✅ ¡Inyección Exitosa! {len(df_matriz_demo)} registros actualizados en PostgreSQL.")
                        except Exception as e:
                            st.error(f"Error SQL: {e}")
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
