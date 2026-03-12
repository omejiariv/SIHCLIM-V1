# 08_📈_Modelo_Demografico.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
import time
import json
import unicodedata
import re
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Modelo Demográfico Integral", page_icon="📈", layout="wide")

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
        "PAPUNAUA": "PAPUNAHUA"
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
            'Archipiélago De San Andrés': 'Archipiélago De San Andrés, Providencia Y Santa Catalina',
            'Archipielago De San Andres': 'Archipiélago De San Andrés, Providencia Y Santa Catalina'
        }
        df_master['depto_nom'] = df_master['depto_nom'].replace(reemplazos_deptos)
        
        mapeo_areas = {
            'Cabecera': 'urbano',
            'Cabecera municipal': 'urbano',
            'Centros Poblados y Rural Disperso': 'rural',
            'Total': 'total'
        }
        df_master['area_geografica'] = df_master['area_geografica'].astype(str).str.strip().map(lambda x: mapeo_areas.get(x, x.lower()))

        # Separamos Hombres y Mujeres y Totales
        cols_hombres = [c for c in df_master.columns if 'Hombre' in str(c)]
        cols_mujeres = [c for c in df_master.columns if 'Mujer' in str(c)]
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
        df_nac = df_nac_temp.groupby('año')[cols_agrupar_nac].sum().reset_index()

        # -------------------------------------------------------
        # C. Crear df_global
        # -------------------------------------------------------
        mpios_amva = ['Medellín', 'Bello', 'Itagüí', 'Envigado', 'Sabaneta', 'Copacabana', 'La Estrella', 'Girardota', 'Caldas', 'Barbosa']
        df_ant = df_nac_temp[df_nac_temp['depto_nom'] == 'Antioquia'].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Antioquia'})
        df_amva = df_nac_temp[(df_nac_temp['depto_nom'] == 'Antioquia') & (df_nac_temp['municipio'].str.title().isin(mpios_amva))].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Amva'})
        df_med = df_nac_temp[df_nac_temp['municipio'] == 'Medellín'].groupby('año')['Total'].sum().reset_index().rename(columns={'Total': 'Pob_Medellin'})
        
        df_global = pd.merge(df_ant, df_amva, on='año', how='outer')
        df_global = pd.merge(df_global, df_med, on='año', how='outer')
        df_global = df_global.rename(columns={'año': 'Año'})

        # =======================================================
        # 2. Cargar datos Veredales 
        # =======================================================
        df_ver = pd.DataFrame()
        ruta_ver_1 = "data/veredas_Antioquia.csv"
        if os.path.exists(ruta_ver_1): df_ver = pd.read_csv(ruta_ver_1, sep=';')

        return df_nac, df_mun, df_ver, df_global
        
    except Exception as e:
        import streamlit as st
        st.error(f"🚨 Error cargando las bases de datos desde la nube: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    df_base = df_nac.copy()
    filtro_zona = "Colombia"
    titulo_terr = "Colombia"
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    
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
    
    df_mapa_base = df_mun.groupby(['depto_nom', 'año'])['Total'].sum().reset_index()
    df_mapa_base.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
    df_mapa_base['Padre'] = "Colombia"

# --- NUEVO MOTOR: DEPARTAMENTAL ---
elif escala_sel == "🏛️ Departamental (Colombia)":
    depto_sel = st.sidebar.selectbox("Departamento:", sorted(df_mun['depto_nom'].unique()))
    df_base = df_mun[df_mun['depto_nom'] == depto_sel]
    
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
    
    df_terr = df_mun[df_mun['Macroregion'] == reg_sel].groupby('año')['Total'].sum().reset_index()
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values
    
    df_mapa_base = df_mun[df_mun['Macroregion'] == reg_sel].groupby(['municipio', 'depto_nom', 'año'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel == "💧 Cuencas Hidrográficas":
    ruta_cuencas_ver = os.path.join(RUTA_RAIZ, "data", "cuencas_veredas_proporcion.csv")
    ruta_cuencas_mun = os.path.join(RUTA_RAIZ, "data", "cuencas_mpios_proporcion.csv")
    
    if os.path.exists(ruta_cuencas_ver):
        df_prop = pd.read_csv(ruta_cuencas_ver)
        lista_cuencas = sorted(df_prop['Subcuenca'].dropna().unique())
        cuenca_sel = st.sidebar.selectbox("Seleccione la Subcuenca (Alta Precisión):", lista_cuencas)
        
        df_prop_sel = df_prop[df_prop['Subcuenca'] == cuenca_sel].copy()
        
        def limpiar_texto(columna):
            return columna.astype(str).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.upper().str.strip()
            
        df_prop_sel['Vereda_upper'] = limpiar_texto(df_prop_sel['Vereda'])
        df_prop_sel['Municipio_upper'] = limpiar_texto(df_prop_sel['Municipio'])
        
        # --- BLOQUE CORREGIDO PARA VEREDAS ---
        df_ver_temp = df_ver.copy()
        df_ver_temp['Vereda_upper'] = limpiar_texto(df_ver_temp['Vereda'])
        df_ver_temp['Municipio_upper'] = limpiar_texto(df_ver_temp['Municipio'])
        
        # ESCUDO ANTI-TEXTO: Convertimos la población a números puros obligatoriamente
        df_ver_temp['Poblacion_hab'] = pd.to_numeric(df_ver_temp['Poblacion_hab'].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').fillna(0)
        
        # 1. Cruzar Veredas base con la cuenca
        df_cruce_ver = pd.merge(df_ver_temp, df_prop_sel, on=['Vereda_upper', 'Municipio_upper'], how='inner')
        
        # ESCUDO ANTI-TEXTO: Forzamos a que sean números reales borrando comas o puntos raros
        df_cruce_ver['Poblacion_hab'] = pd.to_numeric(df_cruce_ver['Poblacion_hab'].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').fillna(0)
        df_cruce_ver['Porcentaje'] = pd.to_numeric(df_cruce_ver['Porcentaje'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        df_cruce_ver['Pob_en_cuenca'] = df_cruce_ver['Poblacion_hab'] * (df_cruce_ver['Porcentaje'] / 100.0)
        
        aporte_mpio = df_cruce_ver.groupby('Municipio_upper')['Pob_en_cuenca'].sum().reset_index()
        total_rural_mpio = df_ver_temp.groupby('Municipio_upper')['Poblacion_hab'].sum().reset_index()
        total_rural_mpio.rename(columns={'Poblacion_hab': 'Pob_Total_Rural'}, inplace=True)
        
        ratios_mpio = pd.merge(aporte_mpio, total_rural_mpio, on='Municipio_upper')
        ratios_mpio['Ratio_Cuenca'] = ratios_mpio['Pob_en_cuenca'] / ratios_mpio['Pob_Total_Rural']
        ratios_mpio['Ratio_Cuenca'] = ratios_mpio['Ratio_Cuenca'].fillna(0)
        
        terminos_rurales = ['rural', 'resto', 'centros poblados y rural disperso', 'centro poblado y rural disperso']
        mask_rural = limpiar_texto(df_mun['area_geografica']).str.lower().isin(terminos_rurales)
        
        df_mun_rural = df_mun[mask_rural].copy()
        df_mun_rural['Municipio_upper'] = limpiar_texto(df_mun_rural['municipio'])
        
        df_base = pd.merge(df_mun_rural, ratios_mpio, on='Municipio_upper', how='inner')
        df_base['Total_Cuenca'] = df_base['Total'] * df_base['Ratio_Cuenca']        
        
        filtro_zona = cuenca_sel
        titulo_terr = f"{cuenca_sel}"
        
        col_anio = 'año' if 'año' in df_base.columns else 'Año'
        
        df_hist = df_base.groupby(col_anio)['Total_Cuenca'].sum().reset_index()
        años_hist = df_hist[col_anio].values
        pob_hist = df_hist['Total_Cuenca'].values
        
        df_mapa_base = df_cruce_ver.copy()
        df_mapa_base.rename(columns={
            'Vereda_x': 'Territorio', 
            'Vereda': 'Territorio', 
            'Municipio_x': 'Padre', 
            'Municipio': 'Padre',
            'Pob_en_cuenca': 'Total'
        }, inplace=True)
        df_mapa_base = df_mapa_base.loc[:, ~df_mapa_base.columns.duplicated()]

    elif os.path.exists(ruta_cuencas_mun):
        df_prop = pd.read_csv(ruta_cuencas_mun)
        lista_cuencas = sorted(df_prop['Subcuenca'].dropna().unique())
        cuenca_sel = st.sidebar.selectbox("Seleccione la Subcuenca:", lista_cuencas)
        
        df_prop_sel = df_prop[df_prop['Subcuenca'] == cuenca_sel].copy()
        
        df_mun_puro = df_mun[df_mun['area_geografica'].str.lower() == 'total'].copy()
        if df_mun_puro.empty: df_mun_puro = df_mun.groupby(['municipio', 'año', 'depto_nom'])['Total'].sum().reset_index()

        df_prop_sel['Municipio_merge'] = df_prop_sel['Municipio'].astype(str).str.upper().str.strip()
        df_mun_puro['Municipio_merge'] = df_mun_puro['municipio'].astype(str).str.upper().str.strip()
        
        df_base = pd.merge(df_mun_puro, df_prop_sel, on='Municipio_merge', how='inner')
        df_base['Total_Cuenca'] = df_base['Total'] * (df_base['Porcentaje'] / 100.0)
        
        filtro_zona = cuenca_sel
        titulo_terr = f"{cuenca_sel}"
        
        col_anio = 'año' if 'año' in df_base.columns else 'Año'
        df_hist = df_base.groupby(col_anio)['Total_Cuenca'].sum().reset_index()
        años_hist = df_hist[col_anio].values
        pob_hist = df_hist['Total_Cuenca'].values
        
        df_mapa_base = df_base.copy()
        df_mapa_base.rename(columns={'municipio': 'Territorio'}, inplace=True)
        df_mapa_base['Total'] = df_mapa_base['Total_Cuenca']
        df_mapa_base['Padre'] = cuenca_sel
        
    else:
        st.error("🚨 Ejecuta el cruce espacial en el Generador para ver las Cuencas.")
        años_hist, pob_hist = np.array([]), np.array([])
        df_mapa_base = pd.DataFrame()
        filtro_zona = "Error Cuencas"
    
elif escala_sel == "🏢 Municipal (Regiones)":
    region_sel = st.sidebar.selectbox("Macroregión:", sorted([r for r in df_mun['Macroregion'].unique() if r != "Sin Región"]))
    mpios_reg = df_mun[df_mun['Macroregion'] == region_sel]
    municipio_sel = st.sidebar.selectbox("Municipio:", sorted(mpios_reg['municipio'].unique()))
    
    df_base = mpios_reg[mpios_reg['municipio'] == municipio_sel]
    filtro_zona = municipio_sel
    titulo_terr = f"{municipio_sel} ({region_sel})"
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    df_hist = df_base.groupby(col_anio)['Total'].sum().reset_index()
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base.copy()
    df_mapa_base.rename(columns={'municipio': 'Territorio', 'Macroregion': 'Padre'}, inplace=True)

elif escala_sel == "🏢 Municipal (Departamentos)":
    depto_sel = st.sidebar.selectbox("Departamento:", sorted(df_mun['depto_nom'].unique()))
    mpios_depto = df_mun[df_mun['depto_nom'] == depto_sel]
    municipio_sel = st.sidebar.selectbox("Municipio:", sorted(mpios_depto['municipio'].unique()))
    
    df_base = mpios_depto[mpios_depto['municipio'] == municipio_sel]
    filtro_zona = municipio_sel
    titulo_terr = f"{municipio_sel} ({depto_sel})"
    
    col_anio = 'año' if 'año' in df_base.columns else 'Año'
    df_hist = df_base.groupby(col_anio)['Total'].sum().reset_index()
    años_hist = df_hist[col_anio].values
    pob_hist = df_hist['Total'].values
    
    df_mapa_base = df_base.copy()
    df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'}, inplace=True)

elif escala_sel == "🌿 Veredal (Antioquia)":
    mpios_veredas = ["TODOS (Ver Mapa Completo)"] + sorted(df_ver['Municipio'].dropna().unique())
    mpio_sel = st.sidebar.selectbox("Municipio (Antioquia)", mpios_veredas)
    
    if mpio_sel == "TODOS (Ver Mapa Completo)":
        df_rural_ant = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['area_geografica'] == 'rural')]
        df_hist_rural = df_rural_ant.groupby('año')['Total'].sum().reset_index()

        años_hist = df_hist_rural['año'].values
        pob_hist = df_hist_rural['Total'].values
        
        filtro_zona = "Antioquia"
        titulo_terr = "Todas las Veredas (Región Disponible)"
        
        df_mapa_base = df_ver.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'
        
    else:
        veredas_lista = sorted(df_ver[df_ver['Municipio'] == mpio_sel]['Vereda'].dropna().unique())
        vereda_sel = st.sidebar.selectbox("Vereda", veredas_lista)
        
        df_rural_mpio = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['municipio'] == mpio_sel) & (df_mun['area_geografica'] == 'rural')]
        df_hist_rural = df_rural_mpio.groupby('año')['Total'].sum().reset_index()
        
        df_mpio_veredas = df_ver[df_ver['Municipio'] == mpio_sel]
        pob_total_veredas = df_mpio_veredas['Poblacion_hab'].sum()
        pob_ver_especifica = df_mpio_veredas[df_mpio_veredas['Vereda'] == vereda_sel]['Poblacion_hab'].sum()
        
        ratio_vereda = pob_ver_especifica / pob_total_veredas if pob_total_veredas > 0 else 0
        
        años_hist = df_hist_rural['año'].values
        pob_hist = df_hist_rural['Total'].values * ratio_vereda
        
        filtro_zona = vereda_sel
        titulo_terr = f"Vereda {vereda_sel} ({mpio_sel})"
        
        df_mapa_base = df_mpio_veredas.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'

# Escudo final de seguridad
if 'titulo_terr' not in locals():
    titulo_terr = filtro_zona if 'filtro_zona' in locals() else "Territorio Seleccionado"
    
# =====================================================================
# --- 4. CÁLCULO DE PROYECCIONES (NUEVO PARADIGMA TOP-DOWN) ---
# =====================================================================

# 1. ESCUDO DEFINITIVO ANTI-NAME ERROR
try:
    _ = filtro_zona
except NameError:
    try:
        filtro_zona = titulo_terr
    except NameError:
        filtro_zona = "Colombia"

territorio_busqueda = str(filtro_zona).upper().strip()
if escala_sel == "🇨🇴 Nacional (Colombia)": territorio_busqueda = "COLOMBIA"

# 2. ESCUDO UNIVERSAL ANTI ZIG-ZAGS (Limpia matemática y gráficas)
if len(años_hist) > 0:
    df_clean = pd.DataFrame({'Año': años_hist, 'Pob': pob_hist})
    df_clean = df_clean[df_clean['Pob'] > 0] 
    df_clean = df_clean.groupby('Año')['Pob'].max().reset_index() 
    df_clean = df_clean.sort_values(by='Año') 
    
    x_hist = df_clean['Año'].values.astype(float)
    y_hist = df_clean['Pob'].values.astype(float)
    
    # ¡LA CURA DEL ZIG-ZAG! Obligamos a Plotly a usar los datos limpios y ordenados
    años_hist = x_hist
    pob_hist = y_hist
else:
    x_hist, y_hist = np.array([]), np.array([])

# 3. Eje X Futuro
# Escudo protector para detectar si la columna es 'año' o 'Año'
col_anio_nac = 'año' if 'año' in df_nac.columns else 'Año'
año_maximo = int(max(df_nac[col_anio_nac].max() if 'df_nac' in locals() and not df_nac.empty else 2100, 2100))
x_proj = np.arange(1950, año_maximo + 1, 1) 

proyecciones = {'Año': x_proj, 'Real': [np.nan]*len(x_proj)}
for i, año in enumerate(x_proj):
    if año in x_hist: proyecciones['Real'][i] = y_hist[np.where(x_hist == año)[0][0]]

# 4. Cargar Matriz Maestra
ruta_matriz = os.path.join(RUTA_RAIZ, "data", "Matriz_Maestra_Demografica.csv")
df_matriz = pd.DataFrame()
if os.path.exists(ruta_matriz):
    df_matriz = pd.read_csv(ruta_matriz, sep=';' if ';' in open(ruta_matriz).readline() else ',')

def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))

row_matriz = pd.DataFrame()
if not df_matriz.empty and escala_sel not in ["🌍 Global y Suramérica", "💧 Cuencas Hidrográficas", "Veredal (Antioquia)"]:
    mask = df_matriz['Territorio'].astype(str).str.upper().str.strip() == territorio_busqueda
    row_matriz = df_matriz[mask]

# --- 5. INYECTAR RESULTADOS ---
if not row_matriz.empty:
    # MODO 1: MATRIZ MAESTRA
    row = row_matriz.iloc[0]
    k_opt = float(str(row['K']).replace('.', '').replace(',', '.')) if isinstance(row['K'], str) else float(row['K'])
    a_opt = float(str(row['a']).replace(',', '.'))
    r_opt = float(str(row['r']).replace(',', '.'))
    anio_base = int(row['Año_Base'])
    
    x_proj_norm = x_proj - anio_base
    proyecciones['Logístico'] = f_log(x_proj_norm, k_opt, a_opt, r_opt)
    
    param_K, param_r = k_opt, r_opt
    modo_calc = f"Matriz Maestra (R²: {row.get('R2', 'N/A')})"
    
else:
    # MODO 2: FALLBACK ROBUSTO (Macroregiones, Cuencas, Veredas)
    x_train, y_train = x_hist, y_hist
    x_offset = x_train[0] if len(x_train) > 0 else 1950
    x_train_norm = x_train - x_offset
    x_proj_norm = x_proj - x_offset
    
    try:
        p0_val = max(1, y_train[0] if len(y_train)>0 else 1)
        p_final = y_train[-1] if len(y_train)>0 else p0_val
        max_y = max(y_train)
        
        es_creciente = p_final >= p0_val
        k_guess = max_y * 1.2 if es_creciente else max_y
        a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
        r_guess = 0.02 if es_creciente else -0.02
        
        # Ampliamos los límites para que el modelo nunca se rinda y caiga a cero
        limites = ([max_y * 0.8, 0, -0.2], [max_y * 3.0 if es_creciente else max_y * 1.1, np.inf, 0.3])
        
        popt_log, _ = curve_fit(f_log, x_train_norm, y_train, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
        proyecciones['Logístico'] = f_log(x_proj_norm, *popt_log)
        
        param_K, param_r = popt_log[0], popt_log[2]
        modo_calc = "Cálculo en Vivo (Robusto)"
    except Exception as e:
        # Failsafe extremo: Si las matemáticas fallan, asume el promedio reciente
        val_seguro = np.mean(y_train[-3:]) if len(y_train) >= 3 else (y_train[-1] if len(y_train)>0 else 0)
        proyecciones['Logístico'] = np.full(len(x_proj), val_seguro)
        param_K, param_r = "N/A", 0
        modo_calc = "Ajuste Promedio (Falla Matemática)"

# Uniformizamos
proyecciones['Lineal'] = proyecciones['Logístico'] 
proyecciones['Exponencial'] = proyecciones['Logístico']
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
        st.subheader(f"📈 Curvas de Crecimiento Poblacional - {titulo_terr}")
        fig_curvas = go.Figure()
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Logístico'], mode='lines', name='Mod. Logístico', line=dict(color='#10b981', dash='dash')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Exponencial'], mode='lines', name='Mod. Exponencial', line=dict(color='#f59e0b', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=df_proj['Año'], y=df_proj['Lineal'], mode='lines', name='Mod. Lineal', line=dict(color='#6366f1', dash='dot')))
        fig_curvas.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Reales (Censo)', marker=dict(color='#ef4444', size=8, symbol='diamond')))
        fig_curvas.update_layout(hovermode="x unified", xaxis_title="Año", yaxis_title="Habitantes", template="plotly_white")
        st.plotly_chart(fig_curvas, width='stretch')

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
    velocidad_animacion = st.sidebar.slider("Velocidad (Segundos por año)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    iniciar_animacion = st.sidebar.button("▶️ Reproducir Evolución", type="primary", use_container_width=True)

    st.subheader(f"Estructura Poblacional Sintética - {titulo_terr}")
    ph_titulo_año = st.empty()

    col_pir1, col_pir2 = st.columns([2, 1])
    with col_pir1: ph_grafico_pir = st.empty()
    with col_pir2:
        st.markdown("### Resumen del Perfil")
        ph_metrica_pob = st.empty()
        ph_metrica_hom = st.empty()
        ph_metrica_muj = st.empty()
        ph_metrica_ind = st.empty()

    df_piramide_final = pd.DataFrame() 

def renderizar_piramide(año_obj):
        st.markdown(f"#### 📊 Estructura Poblacional Modelada ({año_obj})")
        
        try:
            pob_modelo = df_proj[df_proj['Año'] == año_obj][columna_modelo].values[0]
        except:
            pob_modelo = np.nan
            
        if pd.isna(pob_modelo) or pob_modelo == 0:
            st.warning(f"No hay datos de población proyectada para el año {año_obj}.")
            return

        col_anio_pyr2 = 'año' if 'año' in df_nac.columns else 'Año'
        df_fnac = df_nac[df_nac[col_anio_pyr2] == año_obj].copy()
        
        if df_fnac.empty:
            st.warning("No hay datos base nacionales para este año.")
            return

        # --- MAGIA DE TRANSFORMACIÓN (De Ancho a Largo) ---
        import re
        # Buscamos las columnas de edades ignorando las de totales
        cols_h = [c for c in df_fnac.columns if 'Hombre' in str(c) and c != 'Hombres']
        cols_m = [c for c in df_fnac.columns if 'Mujer' in str(c) and c != 'Mujeres']
        
        def extraer_edad(texto):
            nums = re.findall(r'\d+', texto)
            return int(nums[0]) if nums else 0

        datos_edades = []
        for col in cols_h:
            edad = extraer_edad(col)
            val_h = df_fnac[col].values[0]
            # Buscamos la columna de mujer correspondiente a la misma edad
            col_mujer = next((c for c in cols_m if extraer_edad(c) == edad), None)
            val_m = df_fnac[col_mujer].values[0] if col_mujer else 0
            datos_edades.append({'Edad': edad, 'Hombres': val_h, 'Mujeres': val_m})
            
        df_edades = pd.DataFrame(datos_edades)
        
        # Calcular proporciones usando la NUEVA tabla df_edades
        pob_nacional_tot = df_edades['Hombres'].sum() + df_edades['Mujeres'].sum()
        df_edades['Prop_H'] = df_edades['Hombres'] / pob_nacional_tot
        df_edades['Prop_M'] = df_edades['Mujeres'] / pob_nacional_tot
        
        # Escalar al territorio objetivo (Antioquia, Medellín, etc.) usando la NUEVA tabla
        df_edades['Hom_Terr'] = df_edades['Prop_H'] * pob_modelo
        df_edades['Muj_Terr'] = df_edades['Prop_M'] * pob_modelo
        
        # Construir df_pir para Plotly usando la NUEVA tabla
        df_pir = pd.DataFrame({
            'Edad': df_edades['Edad'], 
            'Hombres': df_edades['Hom_Terr'] * -1, 
            'Mujeres': df_edades['Muj_Terr']
        })
        
        # Clasificar en grupos de 5 años
        cortes = list(range(0, 105, 5)) + [200]
        etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)] + ["100+"]
        df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
        
        df_pir_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()

        # --- DIBUJAR LA PIRÁMIDE ---
        fig_pir = go.Figure()
        fig_pir.add_trace(go.Bar(y=df_pir_agrupado['Rango'], x=df_pir_agrupado['Hombres'], name='Hombres', orientation='h', marker_color='#3498db'))
        fig_pir.add_trace(go.Bar(y=df_pir_agrupado['Rango'], x=df_pir_agrupado['Mujeres'], name='Mujeres', orientation='h', marker_color='#e74c3c'))
        
        rango_max = max(abs(df_pir_agrupado['Hombres'].min()), df_pir_agrupado['Mujeres'].max()) if not df_pir_agrupado.empty else 100
        
        fig_pir.update_layout(
            barmode='relative',
            yaxis_title='Rango de Edad',
            xaxis_title='Población',
            xaxis=dict(range=[-rango_max*1.1, rango_max*1.1], tickvals=[-rango_max, 0, rango_max], ticktext=[f"{int(rango_max):,}", "0", f"{int(rango_max):,}"]),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pir, use_container_width=True)
        
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
# TAB 2: MODELOS Y OPTIMIZACIÓN MATEMÁTICA (SOLVER)
# ==============================================================================
with tab_opt:
    st.header("⚙️ Ajuste de Modelos Evolutivos (Solver)")
    
    if len(x_hist) == 0:
        st.info("👆 Selecciona una escala válida en el panel izquierdo.")
    elif escala_sel == "Veredal (Antioquia)":
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

            t_total = np.arange(0, max(t_data) + t_max + 1)
            anios_totales = t_total + t_data_raw.min()
            
            fig2 = go.Figure()
            # Mostramos TODO el histórico en puntos negros
            fig2.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Datos Históricos', marker=dict(color='black', size=8)))

            res_text = []
            for mod in modelos_sel:
                y_pred = np.zeros_like(t_total, dtype=float)
                try:
                    if mod == "Exponencial":
                        if opt_auto: 
                            popt, _ = curve_fit(f_exp, t_data, p_data, p0=[p0_val, 0.01])
                            y_pred = f_exp(t_total, *popt)
                            res_text.append(f"**Exp**: r={popt[1]:.4f}")
                        else: 
                            y_pred = f_exp(t_total, p0_val, r_man)
                            
                    elif mod == "Logístico":
                        if opt_auto: 
                            k_guess = max(p_data) * 1.5
                            a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                            r_guess = 0.02
                            limites = ([max(p_data), 0, 0.0001], [max(p_data)*5, np.inf, 0.3])
                            
                            popt, _ = curve_fit(f_log, t_data, p_data, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=25000)
                            y_pred = f_log(t_total, *popt)
                            
                            # Calculamos R2 para mostrarlo
                            ss_res = np.sum((p_data - f_log(t_data, *popt)) ** 2)
                            ss_tot = np.sum((p_data - np.mean(p_data)) ** 2)
                            r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            res_text.append(f"**Log**: K={popt[0]:,.0f}, r={popt[2]:.4f}, R²={r2_val:.3f}")
                        else: 
                            # Ajuste manual perfecto anclado al inicio
                            a_man = (k_man - p0_val) / p0_val if p0_val > 0 else 1
                            y_pred = f_log(t_total, k_man, a_man, r_man)
                            
                    elif mod == "Geométrico":
                        if opt_auto: 
                            popt, _ = curve_fit(f_geom, t_data, p_data, p0=[p0_val, 0.01])
                            y_pred = f_geom(t_total, *popt)
                        else: 
                            y_pred = f_geom(t_total, p0_val, r_man)
                            
                    elif mod == "Polinómico (Grado 2)":
                        if opt_auto: popt, _ = curve_fit(f_poly2, t_data, p_data); y_pred = f_poly2(t_total, *popt)
                        else: y_pred = f_poly2(t_total, 1, 10, p0_val)
                        
                    elif mod == "Polinómico (Grado 3)":
                        if opt_auto: popt, _ = curve_fit(f_poly3, t_data, p_data); y_pred = f_poly3(t_total, *popt)
                        else: y_pred = f_poly3(t_total, 1, 10, p0_val, 0)

                    fig2.add_trace(go.Scatter(x=anios_totales, y=y_pred, mode='lines', name=mod, line=dict(width=3, dash='dot' if opt_auto else 'solid')))
                except: 
                    pass # Si un modelo matemático falla, simplemente no lo dibuja, evitando que colapse la app

            fig2.update_layout(title="Proyección de Modelos Dinámicos", xaxis_title="Año", yaxis_title="Población", hovermode="x unified", height=550)
            st.plotly_chart(fig2, use_container_width=True)
            if opt_auto and res_text: st.success("✅ **Parámetros Óptimos:** " + " | ".join(res_text))
                
# ==========================================
# PESTAÑA 3: MAPA DEMOGRÁFICO (GEOVISOR)
# ==========================================
with tab_mapas:
    # Usamos un pequeño seguro por si titulo_terr no existe en alguna escala
    titulo_tab_mapa = titulo_terr if 'titulo_terr' in locals() else "Territorio Seleccionado"
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_tab_mapa} ({año_sel})")
    
    if escala_sel != "Veredal (Antioquia)":
        area_mapa = st.radio("Filtro de Zona Geográfica:", ["Total", "Urbano", "Rural"], horizontal=True)
    else:
        area_mapa = "Rural"
        st.info("ℹ️ A escala veredal, toda la población es considerada rural.")

    if 'año' in df_mapa_base.columns:
        df_mapa_año = df_mapa_base[df_mapa_base['año'] == min(max(df_mapa_base['año']), año_sel)].copy()
    else:
        df_mapa_año = df_mapa_base.copy()

    # --- ESCUDO CONTRA BASES DE DATOS VACÍAS (Global/Cuencas) ---
    if df_mapa_año.empty:
        df_mapa_plot = pd.DataFrame()
    else:
        if area_mapa == "Total":
            # Agrupamos SOLO por las columnas que realmente existan
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

    col_map1, col_map2 = st.columns([1, 3])
    
    with col_map1:
        st.markdown("**⚙️ Configuración del GeoJSON**")
        if escala_sel == "Veredal (Antioquia)": 
            sugerencia_geo = "Veredas_Antioquia_TOTAL_UrbanoyRural.geojson"
            sugerencia_prop = "properties.NOMBRE_VER"
            sugerencia_padre = "properties.NOMB_MPIO"
        else: 
            sugerencia_geo = "mgn_municipios_optimizado.geojson"
            sugerencia_prop = "properties.MPIO_CNMBR"
            sugerencia_padre = "properties.DPTO_CCDGO"
            
        archivo_geo_input = st.text_input("Archivo en GitHub:", value=sugerencia_geo)
        prop_geo_input = st.text_input("Llave Territorio:", value=sugerencia_prop)
        st.markdown("**🔗 Llave Doble (Anti-Homonimia)**")
        prop_padre_input = st.text_input("Llave Contexto:", value=sugerencia_padre)

    with col_map2:
        ruta_geo = os.path.join(RUTA_RAIZ, "data", archivo_geo_input)
        
        if os.path.exists(ruta_geo) and not df_mapa_plot.empty:
            try:
                # 1. Crear ADN Único en Pandas
                if prop_padre_input.strip() != "":
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot['Territorio'].apply(normalizar_texto) + "_" + df_mapa_plot['Padre'].apply(normalizar_texto)
                else:
                    df_mapa_plot['MATCH_ID'] = df_mapa_plot['Territorio'].apply(normalizar_texto)

                with open(ruta_geo, encoding='utf-8') as f:
                    geo_data = json.load(f)
                
                # --- TRADUCTOR DANE DE CÓDIGOS A NOMBRES ---
                codigos_dane_deptos = {
                    "05": "ANTIOQUIA", "08": "ATLANTICO", "11": "BOGOTA", "13": "BOLIVAR", 
                    "15": "BOYACA", "17": "CALDAS", "18": "CAQUETA", "19": "CAUCA", 
                    "20": "CESAR", "23": "CORDOBA", "25": "CUNDINAMARCA", "27": "CHOCO", 
                    "41": "HUILA", "44": "GUAJIRA", "47": "MAGDALENA", "50": "META", 
                    "52": "NARINO", "54": "NORTEDESANTANDER", "63": "QUINDIO", "66": "RISARALDA", 
                    "68": "SANTANDER", "70": "SUCRE", "73": "TOLIMA", "76": "VALLE", 
                    "81": "ARAUCA", "85": "CASANARE", "86": "PUTUMAYO", "88": "ARCHIPIELAGODESANANDRES", 
                    "91": "AMAZONAS", "94": "GUAINIA", "95": "GUAVIARE", "97": "VAUPES", "99": "VICHADA"
                }
                
                # 2. Crear ADN Único en el GeoJSON
                prop_key = prop_geo_input.replace("properties.", "")
                padre_key = prop_padre_input.replace("properties.", "") if prop_padre_input.strip() else ""
                
                for feature in geo_data['features']:
                    val_terr = feature['properties'].get(prop_key, "")
                    val_padre = str(feature['properties'].get(padre_key, "")) if padre_key else ""
                    
                    # Fix caracteres mutantes desde la fuente
                    val_terr = normalizar_texto(val_terr)
                    
                    if val_padre.zfill(2) in codigos_dane_deptos:
                        val_padre = codigos_dane_deptos[val_padre.zfill(2)]
                    
                    # Fix espacial: Para distinguir los dos Manaures (El del Cesar es Balcón del Cesar)
                    if val_terr == "MANAUREBALCONDELCESAR": val_terr = "MANAURE"
                    
                    if val_padre:
                        feature['properties']['MATCH_ID'] = val_terr + "_" + normalizar_texto(val_padre)
                    else:
                        feature['properties']['MATCH_ID'] = val_terr
                        
                q_val = 0.85 if area_mapa == "Total" else 0.90
                max_color = df_mapa_plot['Total'].quantile(q_val) if len(df_mapa_plot) > 10 else df_mapa_plot['Total'].max()
                
                # 3. Renderizar Mapa
                fig_mapa = px.choropleth_mapbox(
                    df_mapa_plot,
                    geojson=geo_data,
                    locations='MATCH_ID',        
                    featureidkey='properties.MATCH_ID', 
                    color='Total',
                    color_continuous_scale="Viridis",
                    range_color=[0, max_color],  
                    mapbox_style="carto-positron",
                    zoom=5 if escala_sel != "Veredal (Antioquia)" else 9, 
                    center={"lat": 4.57, "lon": -74.29} if escala_sel != "Veredal (Antioquia)" else {"lat": 6.25, "lon": -75.56},
                    opacity=0.8,
                    labels={'Total': f'Población {area_mapa}'},
                    hover_data={'Total': ':,.0f', 'MATCH_ID': False, 'Territorio': True, 'Padre': True}
                )
                fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_mapa, width='stretch')
                
                geo_ids_disp = [f['properties'].get('MATCH_ID', '') for f in geo_data['features']]
                df_mapa_plot['En_Mapa'] = df_mapa_plot['MATCH_ID'].isin(geo_ids_disp)
                faltantes = df_mapa_plot[df_mapa_plot['En_Mapa'] == False]
                
                if not faltantes.empty:
                    st.warning(f"⚠️ {len(faltantes)} territorios de la tabla no cruzaron con el GeoJSON.")
                    
                    # --- NUEVO: REVELADOR DE VEREDAS (RAYOS X) ---
                    if escala_sel == "Veredal (Antioquia)":
                        with st.expander("🔍 Ver nombres exactos dentro del mapa GeoJSON (Para arreglar el Excel)"):
                            municipio_actual = normalizar_texto(df_mapa_plot['Padre'].iloc[0]) if not df_mapa_plot.empty else ""
                            
                            # Extraer las veredas que SÍ están dibujadas en el mapa para este municipio
                            veredas_en_mapa = []
                            for f in geo_data['features']:
                                m_padre = str(f['properties'].get(padre_key, ""))
                                if normalizar_texto(m_padre) == municipio_actual:
                                    veredas_en_mapa.append(f['properties'].get(prop_key, ""))
                                    
                            if veredas_en_mapa:
                                st.write(f"El mapa tiene **{len(veredas_en_mapa)}** polígonos para este municipio. Estos son sus nombres reales:")
                                st.dataframe(pd.DataFrame({"Nombres exactos en el GeoJSON": sorted(veredas_en_mapa)}), use_container_width=True)
                                st.info("💡 **Solución:** Busca tu vereda en esta lista. Si en tu Excel dice 'La Loma' pero aquí dice 'Vda. Loma', simplemente actualiza el nombre en tu Excel o añádelo al diccionario del código.")
                            else:
                                st.error("No se encontró ningún polígono para este municipio en el archivo GeoJSON. Verifica que la 'Llave Contexto' sea correcta.")
                
            except Exception as e:
                st.error(f"❌ Error dibujando mapa: {e}")
        else:
            st.warning(f"⚠️ No se encontró **{archivo_geo_input}** o no hay datos para esta vista.")
            
    # Mostramos solo las columnas que realmente existan en la tabla actual
    cols_existentes = [c for c in ['Territorio', 'Padre', 'Total', 'MATCH_ID', 'En_Mapa'] if c in df_mapa_plot.columns]
    
    # Solo ordenamos si la columna 'Total' existe (evita errores en la vista)
    if 'Total' in cols_existentes and not df_mapa_plot.empty:
        # ESCUDO ANTI-TEXTO: Convertimos todo a números puros antes de ordenar
        df_mapa_plot['Total'] = pd.to_numeric(df_mapa_plot['Total'], errors='coerce').fillna(0)
        df_mostrar = df_mapa_plot[cols_existentes].sort_values('Total', ascending=False)
    else:
        df_mostrar = df_mapa_plot[cols_existentes]
        
    st.dataframe(df_mostrar, use_container_width=True)

# =====================================================================
# PESTAÑA 4: GENERADOR DE MATRIZ MAESTRA (TOP-DOWN) CON R²
# =====================================================================
with tab_matriz:
    st.subheader("🧠 Motor Generador de Matriz Maestra Demográfica")
    st.write("Calcula los coeficientes logísticos óptimos (K, a, r) y su coeficiente de ajuste (R²).")
    
    if st.button("⚙️ Iniciar Entrenamiento de Matriz Maestra", type="primary"):
        with st.spinner("Entrenando modelos matemáticos y calculando R²... Esto tomará unos segundos."):
            import numpy as np
            import pandas as pd
            from scipy.optimize import curve_fit
            
            def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))
            
            # --- FUNCIÓN PARA CALCULAR R² ---
            def calcular_r2(y_real, y_pred):
                ss_res = np.sum((y_real - y_pred) ** 2)
                ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            matriz_resultados = []
            
            # --- FUNCIÓN ENTRENADORA ROBUSTA ---
            def ajustar_logistica(x, y, nivel, territorio, padre):
                x_offset = x[0] if len(x) > 0 else 1950
                x_norm = x - x_offset
                
                try:
                    p0_val = max(1, y[0])
                    p_final = y[-1]
                    max_y = max(y)
                    
                    es_creciente = p_final >= p0_val
                    
                    # ========================================================
                    # EL ESCUDO ANTI-RAGONVALIA (Límites Estrictos de K)
                    # ========================================================
                    # Si crece, el techo máximo es 3 veces su población mayor.
                    # Si decrece, el techo máximo es solo un 10% más de su pico histórico.
                    k_max = max_y * 3.0 if es_creciente else max_y * 1.1
                    
                    k_guess = max_y * 1.2 if es_creciente else max_y
                    a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                    r_guess = 0.02 if es_creciente else -0.02
                    
                    # Obligamos al algoritmo a no salirse de k_max
                    limites = ([max_y, 0, -0.2], [k_max, np.inf, 0.3])
                    
                    popt, _ = curve_fit(f_log, x_norm, y, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
                    k_opt, a_opt, r_opt = popt
                    
                    y_pred = f_log(x_norm, *popt)
                    r2_val = calcular_r2(y, y_pred)
                    
                    matriz_resultados.append({
                        'Nivel': nivel, 'Territorio': territorio, 'Padre': padre,
                        'Año_Base': x_offset, 'Pob_Base': round(p0_val, 0),
                        'K': round(k_opt, 4), 'a': round(a_opt, 6), 'r': round(r_opt, 6), 'R2': round(r2_val, 4)
                    })
                except Exception as e:
                    pass

            # ==========================================
            # 1. CARGA DE BASE DE DATOS ÚNICA (LA SOLUCIÓN AL ERROR)
            # ==========================================
            # Usamos la base que ya está cargada en la memoria de tu app
            df_mun_memoria = df_mun.copy() 
            col_anio = 'año' if 'año' in df_mun_memoria.columns else 'Año'
            
            # Filtramos solo el 'Total' para no duplicar datos urbanos/rurales
            df_mun_puro = df_mun_memoria[df_mun_memoria['area_geografica'].str.lower() == 'total'].copy()
            if df_mun_puro.empty: 
                df_mun_puro = df_mun_memoria.copy()

            # ==========================================
            # 2. ESCALA NACIONAL (Construida sumando municipios)
            # ==========================================
            df_nac_temp = df_mun_puro.groupby(col_anio)['Total'].sum().reset_index()
            df_nac_temp = df_nac_temp.sort_values(by=col_anio)
            ajustar_logistica(df_nac_temp[col_anio].values, df_nac_temp['Total'].values, 'Nacional', 'Colombia', 'Mundo')

            # ==========================================
            # 3. ESCALA DEPARTAMENTAL
            # ==========================================
            df_deptos = df_mun_puro.groupby(['depto_nom', col_anio])['Total'].sum().reset_index()
            for depto in df_deptos['depto_nom'].unique():
                df_temp = df_deptos[df_deptos['depto_nom'] == depto].sort_values(by=col_anio)
                ajustar_logistica(df_temp[col_anio].values, df_temp['Total'].values, 'Departamental', depto, 'Colombia')

            # ==========================================
            # 4. ESCALA MUNICIPAL
            # ==========================================
            df_mpios = df_mun_puro.groupby(['municipio', 'depto_nom', col_anio])['Total'].sum().reset_index()
            for mpio in df_mpios['municipio'].unique():
                df_temp = df_mpios[df_mpios['municipio'] == mpio].sort_values(by=col_anio)
                padre_mpio = df_temp['depto_nom'].iloc[0]
                ajustar_logistica(df_temp[col_anio].values, df_temp['Total'].values, 'Municipal', mpio, padre_mpio)

            # ==========================================
            # EXPORTACIÓN
            # ==========================================
            df_matriz = pd.DataFrame(matriz_resultados)
            st.success(f"✅ Matriz generada con éxito. Total unidades procesadas: {len(df_matriz)}")
            st.dataframe(df_matriz.head(10))
            
            csv_matriz = df_matriz.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Matriz_Maestra_Demografica.csv",
                data=csv_matriz,
                file_name="Matriz_Maestra_Demografica.csv",
                mime='text/csv'
            )
            
# ==========================================
# PESTAÑA 5: RANKINGS Y DINÁMICA HISTÓRICA (Top 15 y 2005-2035)
# ==========================================
with tab_rankings:
    # Extraemos la zona automáticamente de la base para evitar errores
    zona_actual = df_mapa_base['area_geografica'].iloc[0].title() if not df_mapa_base.empty and 'area_geografica' in df_mapa_base.columns else "Total"
    
    st.subheader(f"📊 Análisis Comparativo y Trayectorias Poblacionales ({zona_actual})")
    
    # Solo mostrar si hay datos para comparar
    if not df_mapa_base.empty and len(df_mapa_base) > 1:
        col_r1, col_r2 = st.columns([1, 1.2])
        
        # --- COLUMNA 1: RANKING BARS (TOP 15) ---
        with col_r1:
            st.markdown(f"**Top 15 Territorios ({año_sel})**")
            es_top = st.radio("Ordenar por:", ["Mayor Población", "Menor Población"], horizontal=True)
            
            df_rank = df_mapa_base.copy().dropna(subset=['Total'])
            # ESCUDO ANTI-TEXTO: Convertimos a números antes de filtrar el Top
            df_rank['Total'] = pd.to_numeric(df_rank['Total'], errors='coerce').fillna(0)
            df_rank = df_rank[df_rank['Total'] > 0]
            
            if es_top == "Mayor Población":
                df_plot = df_rank.nlargest(15, 'Total')
            else:
                df_plot = df_rank.nsmallest(15, 'Total')
                
            fig_bar = px.bar(df_plot, x='Total', y='Territorio', orientation='h', 
                             color='Total', color_continuous_scale='Viridis',
                             title=f"Ranking Poblacional ({año_sel})")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending' if es_top == "Mayor Población" else 'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- COLUMNA 2: CURVAS HISTÓRICAS (2005 - 2035) ---
        with col_r2:
            st.markdown("**Dinámica Poblacional (2005 - 2035)**")
            
            if escala_sel == "Veredal (Antioquia)":
                st.info("ℹ️ A escala Veredal, la plataforma utiliza un corte censal oficial estático. Las curvas de proyección dinámica 2005-2035 se activan desde la escala municipal hacia arriba.")
            else:
                # Extraemos los NOMBRES de los 10 territorios más poblados para graficar sus líneas
                top_10_nombres = df_rank.nlargest(10, 'Total')['Territorio'].tolist()
                
                # Preparamos los datos desde nuestra base maestra (df_mun) actualizada al 2035
                zona_q = zona_actual.lower()
                df_base_historica = df_mun[df_mun['area_geografica'] == zona_q]
                df_line = pd.DataFrame()
                
                if escala_sel == "Departamental":
                    if 'region_sel' in locals() and region_sel != "Todas":
                        df_base_historica = df_base_historica[df_base_historica['Macroregion'] == region_sel]
                    df_line = df_base_historica.groupby(['año', 'depto_nom'])['Total'].sum().reset_index()
                    df_line.rename(columns={'depto_nom': 'Territorio'}, inplace=True)
                    
                elif escala_sel == "Municipal (Departamentos)":
                    df_base_historica = df_base_historica[df_base_historica['depto_nom'] == depto_sel]
                    df_line = df_base_historica.groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)
                    
                elif escala_sel == "Municipal (Regiones)":
                    df_base_historica = df_base_historica[df_base_historica['Macroregion'] == region_sel]
                    df_line = df_base_historica.groupby(['año', 'municipio'])['Total'].sum().reset_index()
                    df_line.rename(columns={'municipio': 'Territorio'}, inplace=True)
                
                # Filtramos la serie de tiempo SOLO para el Top 10
                if not df_line.empty:
                    df_line = df_line[df_line['Territorio'].isin(top_10_nombres)]
                    
                    # --- NUEVO: Conciliación Censal forzada ---
                    def conciliacion_censal(group):
                        group['año'] = pd.to_numeric(group['año'], errors='coerce')
                        group = group.sort_values('año')
                        
                        mask = (group['año'] >= 2006) & (group['año'] <= 2019)
                        group.loc[mask, 'Total'] = np.nan
                        group['Total'] = group['Total'].interpolate(method='linear').ffill().bfill()
                        return group
                        
                    df_line = df_line.groupby('Territorio', group_keys=False).apply(conciliacion_censal)
                    # -------------------------------------------------------------
                    
                    fig_line = px.line(df_line, x='año', y='Total', color='Territorio', markers=True,
                                       title=f"Evolución de los Territorios más Poblados",
                                       labels={'año': 'Año', 'Total': 'Habitantes'})
                    
    else:
        st.info("💡 Selecciona una escala territorial con múltiples divisiones (ej. Municipal o Departamental) para ver el ranking y las curvas comparativas.")
        
# ==========================================
# PESTAÑA 6: DESCARGAS Y EXPORTACIÓN
# ==========================================
with tab_descargas:
    st.subheader("💾 Exportación de Resultados y Series de Tiempo")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("### 📈 Curvas de Proyección (1950-2100)")
        csv_proj = df_proj.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Descargar Proyecciones (CSV)", data=csv_proj, file_name=f"Proyecciones_{titulo_terr}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(df_proj.dropna(subset=['Real']).head(5), use_container_width=True)

    with col_d2:
        st.markdown(f"### 📊 Pirámide Sintética ({año_sel})")
        df_descarga_pir = df_piramide_final.copy()
        if not df_descarga_pir.empty:
            df_descarga_pir['Poblacion'] = df_descarga_pir['Poblacion'].abs()
            csv_pir = df_descarga_pir.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"⬇️ Descargar Pirámide {año_sel} (CSV)", data=csv_pir, file_name=f"Piramide_{año_sel}_{titulo_terr}.csv", mime="text/csv", use_container_width=True)
            st.dataframe(df_descarga_pir.head(5), use_container_width=True)
