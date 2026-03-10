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
        "LACIONDORX10": "ELCONDOR", # Yondó
        "AGUASCLARAS": "AGUASCLARASSONDORA", # El Carmen de V.
        "POTREROMISERANGA": "POTRERAMISERENGA", # Medellín
        "NÁPOLES": "NAPOLES",    # Abejorral
        "LANANCUÍ": "LANANCUI",    # Abriaqui
        "GUALÍ": "GUALI",    # Amaga
        "ÁNIMAS": "ANIMAS",    # Amalfi
        "JARDÍN": "JARDIN",    # Amalfi
        "BALDÍO": "BALDIO",    # Amalfi
        "ELCARMÍN": "ELCARMIN",    # Anori
        "VILLAFÁTIMA": "VILLAFATIMA",    # Anori
        "ELCAFÉ": "ELCAFE",    # Argelia
        "ELRETIRO": "RETIRO",    # Argelia
        "PARAÍSO": "PARAISO",    # Barbosa
        "TRAVESÍAS": "TRAVESIAS",    # Betania
        "ÁREASINLEVANTAR": "AREASINLEVANTAR",    # Betania
        "PIÑONAL": "PINONAL",    # Betulia
        "UNTÍ": "UNTI",    # Buritica
        "LAMANÍDELCARDAL": "LAMANIDELCARDAL",    # Caldas
        "CANALÓN": "CANALON",    # Caldas
        "LALEGÍA": "LALEGIA",    # Caldas
        "CUMBARRÁ": "CUMBARRA",    # Cañasgordas
        "LOSANTIOQUEÑOS": "LOSANTIOQUENOS",    # Cañasgordas
        "BOCASDECHIGORODÓ": "BOCASDECHIGORODO",    # Carepa
        "GUAPÁARRIBA": "GUAPAARRIBA",    # Chigorodo
        "ELPLÁTANO": "ELPLATANO",    # Chigorodo
        "SERRANÍADEABIBE": "SERRANIADEABIBE",    # Chigorodo
        "ELJORDÁN": "ELJORDAN",    # Cocorna
        "LAFÁTIMA": "LAFATIMA",    # Concepcion
        "ANCÓN": "ANCON",    # Copacabana
        "ELJORDÁN": "ELJORDAN",    # Dabeiba
        "CHIMURRÓNENDO": "CHIMURRONENDO",    # Dabeiba
        "LAPÍA": "LAPIA",    # Dabeiba
        "CHOROMANDÓ": "CHOROMANDO",    # Dabeiba
        "TUGURIDÓ": "TUGURIDO",    # Dabeiba
        "RESGUARDOINDÍGENAPEGADÓ": "RESGUARDOINDIGENAPEGADO",    # Dabeiba
        "BALDÍOSDELANACIÓN": "BALDIOSDELANACION",    # Dabeiba
        "LLANODESANTABÁRBARA": "LLANODESANTABARBARA",    # Ebejico
        "SANTABÁRBARA": "SANTABARBARA",    # El Bagre
        "GUACHÍ": "GUACHI",    # El Bagre
        "BALDÍOSDELANACIÓN": "BALDIOSDELANACION",    # El Bagre
        "ELCIPRÉS": "ELCIPRES",    # El Carmen De Viboral
        "SANLÁZARO": "SANLAZARO",    # Frontino
        "ATAUSÍ": "ATAUSI",    # Frontino
        "SANVICENTE-ELKIOSKO": "SANVICENTEELKIOSKO",    # Guadalupe
        "MORRÓN": "MORRON",    # Guadalupe
        "LAVERDE-LAMARÍA": "LAVERDELAMARIA",    # Itagui
        "PÍOX": "PIOX",    # Ituango
        "LACABAÑA": "LACABANA",    # Ituango
        "RÓOCLARO": "ROOCLARO",    # Jardin
        "SANMIGUEL": "SANMIGUELLADORADA",    # La Union
        "FÁTIMA": "FATIMA",    # La Union
        "SANLUÍS": "SANLUIS",    # Maceo
        "POTRERO–MISERANGA": "POTREROMISERANGA",    # Medellin
        "LAVOLCANA-GUAYABAL": "LAVOLCANAGUAYABAL",    # Medellin
        "PATIO-BOLAS": "PATIOBOLAS",    # Medellin
        "SANJOSÉDELMANZANILLO": "SANJOSEDELMANZANILLO",    # Medellin
        "BEBARAMEÑO": "BEBARAMENO",    # Murindo
        "BELÉNDEBAJIRÁ": "BELENDEBAJIRA",    # Mutata
        "LAUNIÓN": "LAUNION",    # Necocli
        "LABERRÍO": "LABERRIO",    # Pueblorrico
        "RÍOMAGDALENA": "RIOMAGDALENA",    # Puerto Triunfo
        "SANANTONIO-ELRÍO": "SANANTONIOELRIO",    # Remedios
        "PUERTOGARZA-NARICES": "PUERTOGARZANARICES",    # San Carlos
        "LARÁPIDA": "LARAPIDA",    # San Carlos
        "NORCASIA-7DEAGOSTO": "NORCASIA7DEAGOSTO",    # San Carlos
        "LASPALMAS": "PALMAS",    # San Carlos
        "QUEBRADÓN-20DEJULIO": "QUEBRADON20DEJULIO",    # San Carlos
        "CAÑAFISTO": "CANAFISTO",    # San Carlos
        "LANUTRIA-CAUNZALES": "LANUTRIACAUNZALES",    # San Francisco
        "ELRINCÓN": "ELRINCON",    # San Jeronimo
        "ELGUÁSIMO": "ELGUASIMO",    # San Jeronimo
        "SANNICOLÁSDELRÍO": "SANNICOLASDELRIO",    # San Juan De Uraba
        "LAHABANA-PALESTINA": "LAHABANAPALESTINA",    # San Luis
        "MANIZALES-VILLANUEVA": "MANIZALESVILLANUEVA",    # San Roque
        "PEÑASAZULES": "PENASAZULES",    # San Roque
        "ALTODELOSGÓMEZ": "ALTODELOSGOMEZ",    # Santa Barbara
        "MORTIÑAL": "MORTINAL",    # Santa Rosa De Osos
        "RÍONEGRITO": "RIONEGRITO",    # Santa Rosa De Osos
        "OROBAJOSANTAINÉS": "OROBAJOSANTAINES",    # Santa Rosa De Osos
        "MONTEFRÍO": "MONTEFRIO",    # Santa Rosa De Osos
        "SANTAGERTRUDIS-PEÑAS": "SANTAGERTRUDISPENAS",    # Santo Domingo
        "BALDÍOSDELANACIÓN": "BALDIOSDELANACION",    # Segovia
        "ROBLALABAJO-CHIRIMOYO": "ROBLALABAJOCHIRIMOYO",    # Sonson
        "ELLLANO-CAÑAVERAL": "ELLLANOCANAVERAL",    # Sonson
        "SANJOSÉLASCRUCES": "SANJOSELASCRUCES",    # Sonson
        "ELLIMÓN": "ELLIMON",    # Sonson
        "SIRGÜITA": "SIRGUITA",    # Sonson
        "GUAYABALRÍOARMA": "GUAYABALRIOARMA",    # Sonson
        "BRISAS-CAUNZAL": "BRISASCAUNZAL",    # Sonson
        "JERUSALÉN": "JERUSALEN",    # Sonson
        "SANTAROSA-LADANTA": "SANTAROSALADANTA",    # Sonson
        "RESGUARDOINDÍGENAJAIDUSABI": "RESGUARDOINDIGENAJAIDUSABI",    # Taraza
        "RINCÓNSANTO": "RINCONSANTO",    # Taraza
        "PUEBLITODELOSBOLÍVARES": "PUEBLITODELOSBOLIVARES",    # Titiribi
        "ELCÁNTARO": "ELCANTARO",    # Toledo
        "SANJOSÉDEMULATOS": "SANJOSEDEMULATOS",    # Turbo
        "SANTIAGODEURABÁ": "SANTIAGODEURABA",    # Turbo
        "KILÓMETRO25": "KILOMETRO25",    # Turbo
        "PARAÍSOTULAPA": "PARAISOTULAPA",    # Turbo
        "ELCAIMÁN": "ELCAIMAN",    # Turbo
        "LAUNIÓN": "LAUNION",    # Turbo
        "BOCASDETÍOLÓPEZ": "BOCASDETIOLOPEZ",    # Turbo
        "LAUNIÓN": "LAUNION",    # Turbo
        "ELPALÓN": "ELPALON",    # Uramita
        "SANJOSÉMONTAÑITAS": "SANJOSEMONTANITAS",    # Urrao
        "PUNTADEOCAIDÓ": "PUNTADEOCAIDO",    # Urrao
        "PARQUENACIONALNATURALLASORQUÍDEAS": "PARQUENACIONALNATURALLASORQUIDEAS",    # Urrao
        "ÁREASINLEVANTAR": "AREASINLEVANTAR",    # Urrao
        "SANJOSÉDEGÉNOVA": "SANJOSEDEGENOVA",    # Valdivia
        "CARACOLÍ": "CARACOLI",    # Valdivia
        "MORRÓN-SEVILLA": "MORRONSEVILLA",    # Valdivia
        "LAALEJANDRÍA": "LAALEJANDRIA",    # Vegachi
        "VEREDACABECERAMUNICIPAL": "CABECERAMUNICIPAL",    # Yarumal
        "ALTODEMÉNDEZ": "ALTODEMENDEZ",    # Yolombo
        "LACÓNDOR-X10": "LACONDORX10",    # Yondo
        "CAÑODONJUAN": "CANODONJUAN",    # Yondo
        "LAROMPIDANO.1": "LAROMPIDANO1",    # Yondo
        "LAROMPIDANO.2": "LAROMPIDANO2",    # Yondo
        "ZONAURBANAVEREDAELDIQUE": "ZONAURBANAELDIQUE",    # Yondo
        "LIMÓNADENTRO": "LIMONADENTRO",    # Zaragoza
        "LAPEÑA": "LAPENA",    # Abejorral
        "MONTELORO-ELREPOSO": "MONTELOROELREPOSO",    # Abejorral
        "ELVOLCÁN": "ELVOLCAN",    # Abejorral
        "CAÑAVERAL": "CANAVERAL",    # Abejorral
        "ELMORRÓN": "ELMORRON",    # Abejorral
        "SANLUÍS": "SANLUIS",    # Abejorral
        "PANTANONEGRO": "PANTANOS",    # Abejorral
        "ELCHAGUALO": "CHAGUALO",    # Abejorral
        "SANJOSÉ": "SANJOSE",    # Abejorral
        "ELBUEY-COLMENAS": "ELBUEYCOLMENAS",    # Abejorral
        "ELCEJÉN": "ELCEJEN",    # Abriaqui
        "SANJOSÉ": "SANJOSE",    # Abriaqui
        "SANMIGUEL": "SANMIGUELLADORADA",    # Alejandria
        "SANJOSÉ": "SANJOSE",    # Alejandria
        "ELCARBÓN": "ELCARBON",    # Alejandria
        "LAFERRERÍA": "LAFERRERIA",    # Amaga
        "PUEBLITODELOSSÁNCHEZ": "PUEBLITODELOSSANCHEZ",    # Amaga
        "NECHÍ": "NECHI",    # Amaga
        "TRAVESÍAS": "TRAVESIAS",    # Amaga
        "MANÍDELCARDAL": "MANIDELCARDAL",    # Amaga
        "PUEBLITODESANJOSÉ": "PUEBLITODESANJOSE",    # Amaga
        "LASPEÑAS": "LASPENAS",    # Amaga
        "PUEBLITODELOSBOLÍVARES": "PUEBLITODELOSBOLIVARES",    # Amaga
        "POCORÓ": "POCORO",    # Amalfi
        "BOQUERÓN": "BOQUERON",    # Amalfi
        "MONDRAGÓN": "MONDRAGON",    # Amalfi
        "ELRETIRO": "RETIRO",    # Amalfi
        "ROMAZÓN": "ROMAZON",    # Amalfi
        "LAPICARDÍA": "LAPICARDIA",    # Amalfi
        "SANMIGUEL": "SANMIGUELLADORADA",    # Amalfi
        "LAVÍBORA": "LAVIBORA",    # Amalfi
        "PINTOLIMÓN": "PINTOLIMON",    # Amalfi
        "ELCAÑAL": "ELCANAL",    # Amalfi
        "GUAMOCÓ": "GUAMOCO",    # Amalfi
        "ELRÍO": "ELRIO",    # Amalfi
        "ELCEDRÓN": "ELCEDRON",    # Andes
        "SANAGUSTÍN": "SANAGUSTIN",    # Andes
        "ELLÍBANO": "ELLIBANO",    # Andes
        "VALLEUMBRÍA": "VALLEUMBRIA",    # Andes
        "BAJOCAÑAVERAL": "BAJOCANAVERAL",    # Andes
        "ALTOCAÑAVERAL": "ALTOCANAVERAL",    # Andes
        "LALEGÍA": "LALEGIA",    # Andes
        "LAESPERANZA-HOYOGRANDE": "LAESPERANZAHOYOGRANDE",    # Andes
        "SANMIGUEL": "SANMIGUELLADORADA",    # Andes
        "ALTOSENÓN": "ALTOSENON",    # Andes
        "SANJULIÁN": "SANJULIAN",    # Andes
        "RÍOCLARO": "RIOCLARO",    # Andes
        "LAESTACIÓN": "LAESTACION",    # Angelopolis
        "SANTABÁRBARA": "SANTABARBARA",    # Angelopolis
        "PROMISIÓN": "PROMISION",    # Angelopolis
        "CAÑAVERALABAJO": "CANAVERALABAJO",    # Angostura
        "LLANOSDECUIVÁ": "LLANOSDECUIVA",    # Angostura
        "TENCHEALGODÓN": "TENCHEALGODON",    # Angostura
        "RÍOARRIBA": "RIOARRIBA",    # Angostura
        "MONTAÑITA": "MONTANITA",    # Angostura
        "CAÑAVERALARRIBA": "CANAVERALARRIBA",    # Angostura
        "LAMUÑOZ": "LAMUNOZ",    # Angostura
        "CHOCHORÍO": "CHOCHORIO",    # Angostura
        "PÁCORA": "PACORA",    # Angostura
        "CONCEPCIÓN": "CONCEPCION",    # Angostura
        "LAMONTAÑA": "LAMONTANA",    # Angostura
        "GUÁSIMO": "GUASIMO",    # Angostura
        "ELLIMÓN": "ELLIMON",    # Anori
        "LASÁNIMAS": "LASANIMAS",    # Anori
        "ELRETIRO": "RETIRO",    # Anori
        "BOLÍVAR": "BOLIVAR",    # Anori
        "TRAVESÍAS": "TRAVESIAS",    # Anori
        "MONTEFRÍO": "MONTEFRIO",    # Anori
        "BRISASDELNECHÍ": "BRISASDELNECHI",    # Anori
        "LAQUIEBRA": "LARAYA",    # Anza
        "CHURIDÓ": "CHURIDO",    # Apartado
        "CHURIDÓPUENTE": "CHURIDOPUENTE",    # Apartado
        "SANMARTÍN": "SANMARTIN",    # Apartado
        "CHURIDÓMEDIO": "CHURIDOMEDIO",    # Apartado
        "LAUNIÓN": "LAUNION",    # Apartado
        "CHURIDÓSINAÍ": "CHURIDOSINAI",    # Apartado
        "RODOXALÍ": "RODOXALI",    # Apartado
        "SANJOSÉ": "SANJOSE",    # Arboletes
        "BOCAALREVÉS": "BOCAALREVES",    # Arboletes
        "ELVOLCÁN": "ELVOLCAN",    # Arboletes
        "LAVÉLEZ": "LAVELEZ",    # Arboletes
        "PARAÍSO": "PARAISO",    # Arboletes
        "SANLUÍS": "SANLUIS",    # Argelia
        "LAQUIEBRA": "LARAYA",    # Argelia
        "SANAGUSTÍN": "SANAGUSTIN",    # Argelia
        "ELPERÚ": "ELPERU",    # Argelia
        "SANTAINÉS": "SANTAINES",    # Argelia
        "TRAVESÍAS": "TRAVESIAS",    # Armenia
        "LAQUIEBRA": "LARAYA",    # Armenia
        "CARTAGÜEÑO": "CARTAGUENO",    # Armenia
        "TABLAZOHATILLO": "ELHATILLO",    # Barbosa
        "LAMONTAÑITA": "MONTANITA",    # Barbosa
        "LAGÓMEZ": "LAGOMEZ",    # Barbosa
        "LASPEÑAS": "LASPENAS",    # Barbosa
        "LAQUIEBRA": "LARAYA",    # Barbosa
        "VOLANTÍN": "VOLANTIN",    # Barbosa
        "LAUNIÓN": "LAUNION",    # Bello
        "RÍOARRIBA": "RIOARRIBA",    # Belmira
        "TAPARTÓ": "TAPARTO",    # Betania
        "LASÁNIMAS": "LASANIMAS",    # Betania
        "LACORAZONADA-LAVALDIVIA": "LACORAZONADALAVALDIVIA",    # Betulia
        "LAQUIEBRA": "LARAYA",    # Betulia
        "LASÁNIMAS": "LASANIMAS",    # Betulia
        "ELLEÓN": "ELLEON",    # Betulia
        "LAURRAEÑA": "LAURRAENA",    # Betulia
        "CUCHILLÓN": "CUCHILLON",    # Betulia
        "ELRETIRO": "RETIRO",    # Betulia
        "GURIMÁN": "GURIMAN",    # Briceño
        "LARODRÍGUEZ": "LARODRIGUEZ",    # Briceño
        "TRAVESÍAS": "TRAVESIAS",    # Briceño
        "BERLÍN": "BERLIN",    # Briceño
        "LAVÉLEZ": "LAVELEZ",    # Briceño
        "LAAMÉRICA": "LAAMERICA",    # Briceño
        "OREJÓN": "OREJON",    # Briceño
        "MORRÓN": "MORRON",    # Briceño
        "ELLEÓN": "ELLEON",    # Buritica
        "PUERTOBÉLGICA": "PUERTOBELGICA",    # Caceres
        "RÍOMAN": "RIOMAN",    # Caceres
        "ALTOCACERÍ": "ALTOCACERI",    # Caceres
        "SANJOSÉDELMAN": "SANJOSEDELMAN",    # Caceres
        "ALTOTAMANÁ": "ALTOTAMANA",    # Caceres
        "CAÑOPRIETO": "CANOPRIETO",    # Caceres
        "CORRALES–ELPLAYÓN": "CORRALESELPLAYON",    # Caceres
        "ELJARDÍN": "ELJARDIN",    # Caceres
        "JUANMARTÍN": "JUANMARTIN",    # Caceres
        "LAGARCÍA": "LAGARCIA",    # Caicedo
        "ELPLAYÓN": "ELPLAYON",    # Caicedo
        "LAQUIEBRA": "LARAYA",    # Caldas
        "SINIFANÁ": "SINIFANA",    # Caldas
        "CAÑAVERAL": "CANAVERAL",    # Campamento
        "LATRAVESÍA": "LATRAVESIA",    # Campamento
        "SANJOSÉDELAGLORIA": "SANJOSEDELAGLORIA",    # Campamento
        "ELPIÑAL": "ELPINAL",    # Campamento
        "LAQUIEBRA": "LARAYA",    # Campamento
        "MONTAÑITA": "MONTANITA",    # Campamento
        "ELLIMÓN": "ELLIMON",    # Campamento
        "TIERRAFRÍA": "TIERRAFRIA",    # Campamento
        "RÍOABAJO": "RIOABAJO",    # Campamento
        "LAQUIEBRA": "LARAYA",    # Cañasgordas
        "SANLUÍSDELCAFÉ": "SANLUISDELCAFE",    # Cañasgordas
        "MOROTÓ": "MOROTO",    # Cañasgordas
        "LACUSUTÍ": "LACUSUTI",    # Cañasgordas
        "SANMIGUEL": "SANMIGUELLADORADA",    # Cañasgordas
        "SANTABÁRBARA": "SANTABARBARA",    # Cañasgordas
        "RUBICÓN": "RUBICON",    # Cañasgordas
        "ELRETIRO": "RETIRO",    # Cañasgordas
        "LACAMPIÑA": "LACAMPINA",    # Cañasgordas
        "ELCAFÉ": "ELCAFE",    # Cañasgordas
        "ELLEÓN": "ELLEON",    # Cañasgordas
        "LAUNIÓN": "LAUNION",    # Cañasgordas
        "SANJULIÁN": "SANJULIAN",    # Cañasgordas
        "LOMADELAALEGRÍA": "LOMADELAALEGRIA",    # Cañasgordas
        "SANJOSÉDEJUNTAS": "SANJOSEDEJUNTAS",    # Cañasgordas
        "SANLUÍS": "SANLUIS",    # Cañasgordas
        "LAMARÍA": "LAMARIA",    # Caracoli
        "LASÁGUILAS": "LASAGUILAS",    # Caracoli
        "CAÑAS": "CANAS",    # Caramanta
        "SANJOSÉ": "SANJOSE",    # Caramanta
        "LAUNIÓN": "LAUNION",    # Caramanta
        "LAUNIÓN15": "LAUNION15",    # Carepa
        "LAUNIÓN": "LAUNION",    # Carepa
        "POLINESSANSEBASTIÁN": "POLINESSANSEBASTIAN",    # Carepa
        "CHIRIDÓ": "CHIRIDO",    # Carepa
        "CUTURÚ": "CUTURU",    # Caucasia
        "LAILUSIÓN": "LAILUSION",    # Caucasia
        "CACERÍ": "CACERI",    # Caucasia
        "ELKILÓMETRO18": "ELKILOMETRO18",    # Caucasia
        "JURADÓ": "JURADO",    # Chigorodo
        "MALAGÓN": "MALAGON",    # Chigorodo
        "CHIRIDÓ": "CHIRIDO",    # Chigorodo
        "JURADÓARRIBA": "JURADOARRIBA",    # Chigorodo
        "ELLIMÓN": "ELLIMON",    # Cisneros
        "BELLAFÁTIMA": "BELLAFATIMA",    # Cisneros
        "SANMIGUEL": "SANMIGUELLADORADA",    # Ciudad Bolivar
        "BOLÍVARARRIBA": "BOLIVARARRIBA",    # Ciudad Bolivar
        "SUCIAINDÍGENA": "SUCIAINDIGENA",    # Ciudad Bolivar
        "LAPIÑUELA": "LAPINUELA",    # Cocorna
        "SANTACRUZ": "SANTACRUZGUACHAVEZ",    # Cocorna
        "LAQUIEBRA": "LARAYA",    # Cocorna
        "ELCHOCÓ": "ELCHOCO",    # Cocorna
        "LAPEÑA": "LAPENA",    # Cocorna
        "ELRETIRO": "RETIRO",    # Cocorna
        "MONTAÑITA": "MONTANITA",    # Cocorna
        "SANMIGUEL": "SANMIGUELLADORADA",    # Cocorna
        "SANPEDROPEÑOLPARTEALTA": "SANPEDROPENOLPARTEALTA",    # Concepcion
        "LASFRÍAS": "LASFRIAS",    # Concepcion
        "PELÁEZ": "PELAEZ",    # Concepcion
        "SANBARTOLOMÉ": "SANBARTOLOME",    # Concepcion
        "SANPEDROPEÑOLPARTEBAJA": "SANPEDROPENOLPARTEBAJA",    # Concepcion
        "LASÁNIMAS": "LASANIMAS",    # Concordia
        "MORRÓN": "MORRON",    # Concordia
        "SANLUÍS": "SANLUIS",    # Concordia
        "ELHIGUERÓN": "ELHIGUERON",    # Concordia
        "PEÑOLCITO": "PENOLCITO",    # Copacabana
        "MONTAÑITA": "MONTANITA",    # Copacabana
        "FONTIDUEÑO": "FONTIDUENO",    # Copacabana
        "ELRETIRO": "RETIRO",    # Dabeiba
        "ANTADÓ": "ANTADO",    # Dabeiba
        "CUCHILLÓN": "CUCHILLON",    # Dabeiba
        "BARRANCÓN": "BARRANCON",    # Dabeiba
        "QUIPARADÓ": "QUIPARADO",    # Dabeiba
        "ELPÁRAMO": "ELPARAMO",    # Dabeiba
        "SANAGUSTÍN": "SANAGUSTIN",    # Dabeiba
        "PEGADÓ": "PEGADO",    # Dabeiba
        "CAÑAVERALES": "CANAVERALES",    # Dabeiba
        "VALLESÍ": "VALLESI",    # Dabeiba
        "ELÁGUILA": "ELAGUILA",    # Dabeiba
        "TASCÓN": "TASCON",    # Dabeiba
        "TASIDÓ": "TASIDO",    # Dabeiba
        "LAMONTAÑITA": "MONTANITA",    # Dabeiba
        "CRUCESDETUGURIDÓ": "CRUCESDETUGURIDO",    # Dabeiba
        "LLANÓN": "LLANON",    # Dabeiba
        "JENATURADÓ": "JENATURADO",    # Dabeiba
        "CAÑAVERALESANTADÓ": "CANAVERALESANTADO",    # Dabeiba
        "CHOROMANDÓALTOMEDIO": "CHOROMANDOALTOMEDIO",    # Dabeiba
        "CHUSCALDEMURRÍ": "CHUSCALDEMURRI",    # Dabeiba
        "ELJARDÍN": "ELJARDIN",    # Dabeiba
        "LAMONTAÑITA": "MONTANITA",    # Dabeiba
        "LASÁNIMAS": "LASANIMAS",    # Don Matias
        "COLÓN": "COLONGENOVA",    # Don Matias
        "RIOGRANDE-BELLAVISTA": "RIOGRANDEBELLAVISTA",    # Don Matias
        "PANDEAZÚCAR": "PANDEAZUCAR",    # Don Matias
        "ROMAZÓN": "ROMAZON",    # Don Matias
        "FÁTIMA": "FATIMA",    # Ebejico
        "ELPALÓN": "ELPALON",    # Ebejico
        "NARIÑO": "NARINO",    # Ebejico
        "FILODESANJOSÉ": "FILODESANJOSE",    # Ebejico
        "BOSQUE-NARANJO": "BOSQUENARANJO",    # Ebejico
        "ELRETIRO": "RETIRO",    # Ebejico
        "LAQUIEBRA": "LARAYA",    # Ebejico
        "AMACERÍ": "AMACERI",    # El Bagre
        "VILLAUCURÚ": "VILLAUCURU",    # El Bagre
        "BROJOLÁ": "BROJOLA",    # El Bagre
        "MUQUÍ": "MUQUI",    # El Bagre
        "LUÍSCANO": "LUISCANO",    # El Bagre
        "CHIRITÁ": "CHIRITA",    # El Bagre
        "MEDIOSDEMANICERÍA": "MEDIOSDEMANICERIA",    # El Bagre
        "SABALITOSINAÍ": "SABALITOSINAI",    # El Bagre
        "BOQUERÓN": "BOQUERON",    # El Carmen De Viboral
        "SANTAINÉS": "SANTAINES",    # El Carmen De Viboral
        "DOSQUEBRADAS–QUEBRADONA": "DOSQUEBRADASQUEBRADONA",    # El Carmen De Viboral
        "GUARINÓ": "GUARINO",    # El Carmen De Viboral
        "ELESTÍO": "ELESTIO",    # El Carmen De Viboral
        "ELRETIRO": "RETIRO",    # El Carmen De Viboral
        "SANJOSÉ": "SANJOSE",    # El Carmen De Viboral
        "CHIQUINQUIRÁ": "CHIQUINQUIRA",    # El Peñol
        "LAHÉLIDA": "LAHELIDA",    # El Peñol
        "SANTAINÉS": "SANTAINES",    # El Peñol
        "PUENTEPELÁEZ": "PUENTEPELAEZ",    # El Retiro
        "NORMANDÍA": "NORMANDIA",    # El Retiro
        "VALLEDEMARÍA": "VALLEDEMARIA",    # El Santuario
        "LATENERÍA": "LATENERIA",    # El Santuario
        "LASPALMAS": "PALMAS",    # El Santuario
        "ELRETIRO": "RETIRO",    # El Santuario
        "ELSEÑORCAÍDO": "ELSENORCAIDO",    # El Santuario
        "SANMATÍAS-LATRINIDAD": "SANMATIASLATRINIDAD",    # El Santuario
        "SANMATÍAS": "SANMATIAS",    # El Santuario
        "LASERRANÍA": "LASERRANIA",    # El Santuario
        "ELPEÑOL": "PENOL",    # Entrerrios
        "PÍOXII": "PIOXII",    # Entrerrios
        "RÍOGRANDE": "RIOGRANDE",    # Entrerrios
        "RÍOCHICO": "RIOCHICO",    # Entrerrios
        "ELVALLANO": "VALLANO",    # Envigado
        "LASPALMAS": "PALMAS",    # Envigado
        "HOYOFRÍO": "HOYOFRIO",    # Fredonia
        "LAMARÍA": "LAMARIA",    # Fredonia
        "TRAVESÍAS": "TRAVESIAS",    # Fredonia
        "JONÁS": "JONAS",    # Fredonia
        "ALTODELOSFERNÁNDEZ": "ALTODELOSFERNANDEZ",    # Fredonia
        "LAQUIEBRA": "LARAYA",    # Fredonia
        "MORRÓN": "MORRON",    # Fredonia
        "NOBOGÁ": "NOBOGA",    # Frontino
        "PONTÓN": "PONTON",    # Frontino
        "LACAMPIÑA": "LACAMPINA",    # Frontino
        "MONTAÑÓN": "MONTANON",    # Frontino
        "LACABAÑA": "LACABANA",    # Frontino
        "QUIPARADÓ": "QUIPARADO",    # Frontino
        "SANANDRÉS": "SANANDRES",    # Frontino
        "SANMIGUEL": "SANMIGUELLADORADA",    # Frontino
        "RÍOVERDE": "RIOVERDE",    # Frontino
        "CURBATÁ": "CURBATA",    # Frontino
        "PEGADÓ": "PEGADO",    # Frontino
        "CUAJARÓN": "CUAJARON",    # Giraldo
        "ELÁGUILA": "ELAGUILA",    # Giraldo
        "LACIÉNAGA": "LACIENAGA",    # Giraldo
        "SANANDRÉS": "SANANDRES",    # Girardota
        "JAMUNDÍ": "JAMUNDI",    # Girardota
        "ELPARAÍSO": "ELPARAISO",    # Girardota
        "LAREGIÓN": "LAREGION",    # Gomez Plata
        "CAÑAVERAL": "CANAVERAL",    # Gomez Plata
        "ELTABLÓN": "ELTABLON",    # Gomez Plata
        "GARZÓN": "GARZON",    # Gomez Plata
        "SANMATÍAS": "SANMATIAS",    # Granada
        "ELEDÉN": "ELEDEN",    # Granada
        "LAQUIEBRA": "LARAYA",    # Granada
        "ELJARDÍN": "ELJARDIN",    # Granada
        "LASPALMAS": "PALMAS",    # Granada
        "CRISTALINA-CEBADERO": "CRISTALINACEBADERO",    # Granada
        "CRISTALINA-CRUCES": "CRISTALINACRUCES",    # Granada
        "LAMARIA-ELPROGRESO": "LAMARIAELPROGRESO",    # Granada
        "SANMIGUEL": "SANMIGUELLADORADA",    # Granada
        "SANJULIÁN": "SANJULIAN",    # Guadalupe
        "SANVICENTE-LASUSANA": "SANVICENTELASUSANA",    # Guadalupe
        "SANVICENTE-LOSSAUCES": "SANVICENTELOSSAUCES",    # Guadalupe
        "MONTAÑITA": "MONTANITA",    # Guadalupe
        "SANJOSÉ": "SANJOSE",    # Guarne
        "MONTAÑEZ": "MONTANEZ",    # Guarne
        "LAMEJÍA": "LAMEJIA",    # Guarne
        "LAPEÑA": "LAPENA",    # Guatape
        "LLANODESANJOSÉ": "LLANODESANJOSE",    # Heliconia
        "JOLY-TABLAZO": "JOLYTABLAZO",    # Heliconia
        "LOSGÓMEZ": "LOSGOMEZ",    # Itagui
        "ELROSARIO-LOMADELOSZULETA": "LOMADELOSZULETA",    # Itagui
        "SANLUÍS": "SANLUIS",    # Ituango
        "ELRÍO": "ELRIO",    # Ituango
        "SANLUÍSCHISPA": "SANLUISCHISPA",    # Ituango
        "LACIÉNAGA": "LACIENAGA",    # Ituango
        "ELQUINDÍO": "ELQUINDIO",    # Ituango
        "REVENTÓN": "REVENTON",    # Ituango
        "ELEDÉN": "ELEDEN",    # Ituango
        "LASARAÑAS": "LASARANAS",    # Ituango
        "PEÑA": "PENA",    # Ituango
        "LAMARÍA": "LAMARIA",    # Ituango
        "SANAGUSTÍNDELEONES": "SANAGUSTINDELEONES",    # Ituango
        "TRAVESÍAS": "TRAVESIAS",    # Ituango
        "LASAGÜITAS": "LASAGUITAS",    # Ituango
        "LAAMÉRICA": "LAAMERICA",    # Ituango
        "ALTODESANAGUSTÍN": "ALTODESANAGUSTIN",    # Ituango
        "SANLUÍS": "SANLUIS",    # Ituango
        "SANLUÍS": "SANLUIS",    # Ituango
        "SERRANÍAS": "SERRANIAS",    # Jardin
        "VERDÚN": "VERDUN",    # Jardin
        "CRISTIANÍA": "CRISTIANIA",    # Jardin
        "LAVIÑA": "LAVINA",    # Jerico
        "ELZACATÍN": "ELZACATIN",    # Jerico
        "LACABAÑA": "LACABANA",    # Jerico
        "SANRAMÓN": "SANRAMON",    # Jerico
        "RÍOFRÍO": "RIOFRIO",    # Jerico
        "VOLCÁNCOLORADO": "VOLCANCOLORADO",    # Jerico
        "SANNICOLÁS": "SANNICOLAS",    # La Ceja
        "SANJOSÉ": "SANJOSE",    # La Ceja
        "SANMIGUEL": "SANMIGUELLADORADA",    # La Ceja
        "FÁTIMA": "FATIMA",    # La Ceja
        "ELHIGUERÓN": "ELHIGUERON",    # La Ceja
        "SANMIGUEL": "SANMIGUELLADORADA",    # La Estrella
        "SANJOSÉ": "SANJOSE",    # La Estrella
        "PEÑASBLANCAS": "PENASBLANCAS",    # La Estrella
        "LAALMERÍA": "LAALMERIA",    # La Union
        "LACABAÑA": "LACABANA",    # La Union
        "LAPEÑOLA": "LAPENOLA",    # Liborina
        "CURITÍ": "CURITI",    # Liborina
        "SANMIGUEL": "SANMIGUELLADORADA",    # Liborina
        "PEÑOLES": "PENOLES",    # Liborina
        "CRISTÓBAL": "CRISTOBAL",    # Liborina
        "LAMONTAÑITA": "MONTANITA",    # Liborina
        "LAUNIÓN": "LAUNION",    # Maceo
        "BELÉN": "BELEN",    # Marinilla
        "LAMONTAÑITA": "MONTANITA",    # Marinilla
        "SANJOSÉ": "SANJOSE",    # Marinilla
        "SANTACRUZ": "SANTACRUZGUACHAVEZ",    # Marinilla
        "LAASUNCIÓN": "LAASUNCION",    # Marinilla
        "LAPEÑA": "LAPENA",    # Marinilla
        "BUGA-PATIOBONITO": "BUGAPATIOBONITO",    # Medellin
        "SANJOSÉDEMANZANILLO": "SANJOSEDEMANZANILLO",    # Medellin
        "AGUASFRÍAS": "AGUASFRIAS",    # Medellin
        "SECTORCENTRAL": "CENTRAL",    # Medellin
        "URQUITÁ": "URQUITA",    # Medellin
        "MONTAÑITA": "MONTANITA",    # Medellin
        "ELCORAZÓN-ELMORRO": "ELCORAZONELMORRO",    # Medellin
        "TRAVESÍAS": "TRAVESIAS",    # Medellin
        "LAILUSIÓN": "LAILUSION",    # Medellin
        "SANJOSÉ": "SANJOSE",    # Medellin
        "PIEDRASBLANCAS-MATASANO": "PIEDRASBLANCASMATASANO",    # Medellin
        "ELJARDÍN": "ELJARDIN",    # Medellin
        "LASPALMAS": "PALMAS",    # Medellin
        "SANJOSÉDELAMONTAÑA": "SANJOSEDELAMONTANA",    # Medellin
        "BOQUERÓN": "BOQUERON",    # Medellin
        "ELGAVILÁN": "ELGAVILAN",    # Montebello
        "LAQUIEBRA": "LARAYA",    # Montebello
        "LAPEÑA": "LAPENA",    # Montebello
        "MURINDÓVIEJO": "MURINDOVIEJO",    # Murindo
        "OPOGADÓ": "OPOGADO",    # Murindo
        "SANTAFEDEMURINDÓ": "SANTAFEDEMURINDO",    # Murindo
        "CHAGERADÓ": "CHAGERADO",    # Murindo
        "CHIBUGADÓ": "CHIBUGADO",    # Murindo
        "COREDÓ": "COREDO",    # Murindo
        "ÑARANGUE": "NARANGUE",    # Murindo
        "TURRIQUITADÓALTO": "TURRIQUITADOALTO",    # Murindo
        "TURRIQUITADÓLLANO": "TURRIQUITADOLLANO",    # Murindo
        "MONTERÍALEÓN": "MONTERIALEON",    # Mutata
        "JURADÓ": "JURADO",    # Mutata
        "LEÓNPORROSO": "LEONPORROSO",    # Mutata
        "CHADÓLARAYA": "CHADOLARAYA",    # Mutata
        "MUNGUDÓ": "MUNGUDO",    # Mutata
        "CHADÓCARRETERA": "CHADOCARRETERA",    # Mutata
        "CHADÓARRIBA": "CHADOARRIBA",    # Mutata
        "SANJOSÉDELEÓN": "SANJOSEDELEON",    # Mutata
        "CAÑADUZALES": "CANADUZALES",    # Mutata
        "JURADÓARRIBA": "JURADOARRIBA",    # Mutata
        "MUTATÁ": "MUTATA",    # Mutata
        "SANANDRÉS": "SANANDRES",    # Nariño
        "LAESPAÑOLA": "LAESPANOLA",    # Nariño
        "ELPIÑAL": "ELPINAL",    # Nariño
        "ELCARAÑO": "ELCARANO",    # Nariño
        "ELJAZMÍN": "ELJAZMIN",    # Nariño
        "ELLIMÓN": "ELLIMON",    # Nariño
        "SANMIGUEL": "SANMIGUELLADORADA",    # Nariño
        "NECHÍ": "NECHI",    # Nariño
        "QUIEBRADESANJOSÉ": "QUIEBRADESANJOSE",    # Nariño
        "RÍOARRIBA": "RIOARRIBA",    # Nariño
        "ELCÓNDOR": "ELCONDOR",    # Nariño
        "BERLÍN": "BERLIN",    # Nariño
        "CAÑOPESCADO": "CANOPESCADO",    # Nechi
        "PUERTOGAITÁN": "PUERTOGAITAN",    # Nechi
        "QUEBRADACIÉNAGA": "QUEBRADACIENAGA",    # Nechi
        "SANTAMARÍA": "SANTAMARIA",    # Nechi
        "LACONCEPCIÓN": "LACONCEPCION",    # Nechi
        "CACERÍ": "CACERI",    # Nechi
        "MULATOS": "LOSMULATOS",    # Necocli
        "ELMELLITO": "ELMELLITOALTO",    # Necocli
        "BRISASDELRÍO": "BRISASDELRIO",    # Necocli
        "SANJOAQUÍN": "SANJOAQUIN",    # Necocli
        "ELRETIRO": "RETIRO",    # Necocli
        "SANSEBASTIÁN": "SANSEBASTIAN",    # Necocli
        "ELCOMEJÉN": "ELCOMEJEN",    # Necocli
        "LACAÑA": "LACANA",    # Necocli
        "RÍONECOCLÍ": "RIONECOCLI",    # Necocli
        "ALGODÓNABAJO": "ALGODONABAJO",    # Necocli
        "ALGODÓNARRIBA": "ALGODONARRIBA",    # Necocli
        "GARITÓN": "GARITON",    # Necocli
        "GIGANTÓN": "GIGANTON",    # Necocli
        "CAIMÁNVIEJO": "CAIMANVIEJO",    # Necocli
        "SANJOSÉDEMULATOS": "SANJOSEDEMULATOS",    # Necocli
        "CAIMÁNNUEVO": "CAIMANNUEVO",    # Necocli
        "CABAÑAS": "CABANAS",    # Necocli
        "PIÑONES": "PINONES",    # Olaya
        "COMÚNCOMINAL": "COMUNCOMINAL",    # Olaya
        "ELCHAPÓN": "ELCHAPON",    # Olaya
        "ELPÁRAMO": "ELPARAMO",    # Peque
        "RENEGADO-VALLE": "RENEGADOVALLE",    # Peque
        "SANJULIÁN": "SANJULIAN",    # Peque
        "MONTARRÓN": "MONTARRON",    # Peque
        "GUAYABAL-PENA": "GUAYABALPENA",    # Peque
        "LASFALDASDELCAFÉ": "LASFALDASDELCAFE",    # Peque
        "SANMIGUEL": "SANMIGUELLADORADA",    # Peque
        "SINAÍ": "SINAI",    # Pueblorrico
        "ELCEDRÓN": "ELCEDRON",    # Pueblorrico
        "LAGÓMEZ": "LAGOMEZ",    # Pueblorrico
        "SANTABÁRBARA": "SANTABARBARA",    # Pueblorrico
        "LAUNIÓN": "LAUNION",    # Pueblorrico
        "MORRÓN": "MORRON",    # Pueblorrico
        "CABAÑAS-PALESTINA": "CABANASPALESTINA",    # Puerto Berrio
        "BRASIL-LACARLOTA": "BRASILLACARLOTA",    # Puerto Berrio
        "CALAMAR-ELDORADO": "CALAMARELDORADO",    # Puerto Berrio
        "SABALETAS-BOLÍVAR": "SABALETASBOLIVAR",    # Puerto Berrio
        "SANTACRUZ": "SANTACRUZGUACHAVEZ",    # Puerto Berrio
        "SANJULIÁN": "SANJULIAN",    # Puerto Berrio
        "ELJARDÍN": "ELJARDIN",    # Puerto Berrio
        "LAUNIÓN": "LAUNION",    # Puerto Nare
        "PEÑAFLOR": "PENAFLOR",    # Puerto Nare
        "LAPATIÑO": "LAPATINO",    # Puerto Nare
        "SERRANÍAS": "SERRANIAS",    # Puerto Nare
        "ELPARAÍSO": "ELPARAISO",    # Puerto Nare
        "CAÑOSECO": "CANOSECO",    # Puerto Nare
        "ESTACIÓNCOCORNA": "ESTACIONCOCORNA",    # Puerto Triunfo
        "LAFLORIDA-TRESRANCHOS": "LAFLORIDATRESRANCHOS",    # Puerto Triunfo
        "SANTIAGOBERRÍO": "SANTIAGOBERRIO",    # Puerto Triunfo
        "ESTRELLA-RÍOCLARO": "ESTRELLARIOCLARO",    # Puerto Triunfo
        "OTÚ": "OTU",    # Remedios
        "MARTANÁ": "MARTANA",    # Remedios
        "CAÑAVERAL": "CANAVERAL",    # Remedios
        "SANCRISTÓBAL": "SANCRISTOBAL",    # Remedios
        "BELÉN": "BELEN",    # Remedios
        "COSTEÑAL": "COSTENAL",    # Remedios
        "SANTALUCÍA": "SANTALUCIA",    # Remedios
        "ELRETIRO": "RETIRO",    # Remedios
        "MANÍ-SANTANA": "MANISANTANA",    # Remedios
        "ITÉ": "ITE",    # Remedios
        "RÍOBAGRE": "RIOBAGRE",    # Remedios
        "TÍASLAAURORA": "TIASLAAURORA",    # Remedios
        "CUCHILLASDESANJOSÉ": "CUCHILLASDESANJOSE",    # Rionegro
        "SANTABÁRBARA": "SANTABARBARA",    # Rionegro
        "ELCARMÍN": "ELCARMIN",    # Rionegro
        "SANLUÍS": "SANLUIS",    # Rionegro
        "RÍOABAJO": "RIOABAJO",    # Rionegro
        "PLAYARICA-RANCHERÍA": "PLAYARICARANCHERIA",    # Rionegro
        "LAQUIEBRA": "LARAYA",    # Rionegro
        "ELHIGUERÓN": "ELHIGUERON",    # Rionegro
        "LACONVENCIÓN": "LACONVENCION",    # Rionegro
        "SANCRISTÓBAL-PENÁ": "SANCRISTOBALPENA",    # Sabanalarga
        "REMARTÍN": "REMARTIN",    # Sabanalarga
        "FILODELOSPÉREZ": "FILODELOSPEREZ",    # Sabanalarga
        "MALPASO-BUENOSAIRES": "MALPASOBUENOSAIRES",    # Sabanalarga
        "SANTAMARÍA": "SANTAMARIA",    # Sabanalarga
        "LATRAVESÍA": "LATRAVESIA",    # Sabanalarga
        "PANDEAZÚCAR": "PANDEAZUCAR",    # Sabaneta
        "CAÑAVERALEJO": "CANAVERALEJO",    # Sabaneta
        "SANJOSÉ": "SANJOSE",    # Sabaneta
        "PEÑALISA": "PENALISA",    # Salgar
        "ELLEÓN": "ELLEON",    # Salgar
        "MONTAÑITA": "MONTANITA",    # Salgar
        "LAAMAGACEÑA": "LAAMAGACENA",    # Salgar
        "CAJÓNLARGO": "CAJONLARGO",    # Salgar
        "MONTAÑAADENTRO": "MONTANAADENTRO",    # San Andres De Cuerquia
        "CAÑADUZALES": "CANADUZALES",    # San Andres De Cuerquia
        "TRAVESÍAS": "TRAVESIAS",    # San Andres De Cuerquia
        "ELPEÑOL": "PENOL",    # San Andres De Cuerquia
        "ELCÁNTARO": "ELCANTARO",    # San Andres De Cuerquia
        "SANMIGUEL": "SANMIGUELLADORADA",    # San Andres De Cuerquia
        "LACIÉNAGA": "LACIENAGA",    # San Andres De Cuerquia
        "LALEJÍA": "LALEJIA",    # San Andres De Cuerquia
        "SANJULIÁN": "SANJULIAN",    # San Andres De Cuerquia
        "PÍOXII": "PIOXII",    # San Carlos
        "CAÑAVERAL": "CANAVERAL",    # San Carlos
        "LACIÉNAGA": "LACIENAGA",    # San Carlos
        "LAILUSIÓN": "LAILUSION",    # San Carlos
        "PEÑOLGRANDE": "PENOLGRANDE",    # San Carlos
        "LACABAÑA": "LACABANA",    # San Carlos
        "SANTABÁRBARA": "SANTABARBARA",    # San Carlos
        "PEÑOLES": "PENOLES",    # San Carlos
        "LASFRÍAS": "LASFRIAS",    # San Carlos
        "SANTAINÉS": "SANTAINES",    # San Carlos
        "LAMARÍA": "LAMARIA",    # San Carlos
        "SANJOSÉ": "SANJOSE",    # San Carlos
        "PABELLÓN": "PABELLON",    # San Carlos
        "SAMANÁ": "SAMANA",    # San Carlos
        "ELCHARCÓN": "ELCHARCON",    # San Carlos
        "ELCHOCÓ": "ELCHOCO",    # San Carlos
        "ELPAJUÍ": "ELPAJUI",    # San Francisco
        "ELJARDÍNDEAQUITANIA": "ELJARDINDEAQUITANIA",    # San Francisco
        "ELJARDÍN-BUENOSAIRES": "ELJARDINBUENOSAIRES",    # San Francisco
        "SANAGUSTÍN": "SANAGUSTIN",    # San Francisco
        "ELPORTÓN": "ELPORTON",    # San Francisco
        "CAÑADAHONDA": "CANADAHONDA",    # San Francisco
        "BOQUERÓN": "BOQUERON",    # San Francisco
        "SANPEDRO-BUENOSAIRES": "SANPEDROBUENOSAIRES",    # San Francisco
        "MONTEFRÍO": "MONTEFRIO",    # San Jeronimo
        "SANTABÁRBARA": "SANTABARBARA",    # San Jose De La Montaña
        "LAMARÍA": "LAMARIA",    # San Jose De La Montaña
        "SANTAINÉS": "SANTAINES",    # San Jose De La Montaña
        "BELÉN": "BELEN",    # San Juan De Uraba
        "CAÑABRAVA": "CANABRAVA",    # San Juan De Uraba
        "BOCASDELRÍOSANJUAN": "BOCASDELRIOSANJUAN",    # San Juan De Uraba
        "FILODESANJOSÉ": "FILODESANJOSE",    # San Juan De Uraba
        "VILLAFÁTIMA": "VILLAFATIMA",    # San Juan De Uraba
        "SINAÍ": "SINAI",    # San Juan De Uraba
        "ALTAVISTA-RIOCLARO": "ALTAVISTARIOCLARO",    # San Luis
        "SOPETRÁN": "SOPETRAN",    # San Luis
        "SANTABÁRBARA": "SANTABARBARA",    # San Luis
        "ELJORDÁN": "ELJORDAN",    # San Luis
        "ESPÍRITUSANTO": "ESPIRITUSANTO",    # San Pedro De Los Milagros
        "SANTABÁRBARA": "SANTABARBARA",    # San Pedro De Los Milagros
        "RÍOCHICO": "RIOCHICO",    # San Pedro De Los Milagros
        "ELCAÑO": "ELCANO",    # San Pedro De Uraba
        "CAIMÁNSANPABLO": "CAIMANSANPABLO",    # San Pedro De Uraba
        "CARACOLÍ": "CARACOLI",    # San Pedro De Uraba
        "LACABAÑA": "LACABANA",    # San Pedro De Uraba
        "TÍODOCTO": "TIODOCTO",    # San Pedro De Uraba
        "ELAJÍ": "ELAJI",    # San Pedro De Uraba
        "SANMIGUEL": "SANMIGUELLADORADA",    # San Pedro De Uraba
        "TATOÑO": "TATONO",    # San Pedro De Uraba
        "ELPOZÓN": "ELPOZON",    # San Pedro De Uraba
        "ELCAIMÁN": "ELCAIMAN",    # San Pedro De Uraba
        "TINAJÓN": "TINAJON",    # San Pedro De Uraba
        "ELJAGÜE": "ELJAGUE",    # San Rafael
        "SANJULIÁN": "SANJULIAN",    # San Rafael
        "LARÁPIDA": "LARAPIDA",    # San Rafael
        "SANAGUSTÍN": "SANAGUSTIN",    # San Rafael
        "PEÑOLES": "PENOLES",    # San Rafael
        "BOQUERÓN": "BOQUERON",    # San Rafael
        "SANTACRUZ": "SANTACRUZGUACHAVEZ",    # San Rafael
        "SANMATÍAS": "SANMATIAS",    # San Roque
        "SANJOAQUÍN": "SANJOAQUIN",    # San Roque
        "ELTÁCHIRA": "ELTACHIRA",    # San Roque
        "SANTABÁRBARA": "SANTABARBARA",    # San Roque
        "SANJOSÉDELNARE": "SANJOSEDELNARE",    # San Roque
        "ELJARDÍN": "ELJARDIN",    # San Roque
        "PEÑOLCITO": "PENOLCITO",    # San Vicente
        "SANJOSÉ": "SANJOSE",    # San Vicente
        "SANNICOLÁS": "SANNICOLAS",    # San Vicente
        "LATRAVESÍA": "LATRAVESIA",    # San Vicente
        "ALTOLACOMPAÑÍA": "ALTOLACOMPANIA",    # San Vicente
        "COMPAÑÍAABAJO": "COMPANIAABAJO",    # San Vicente
        "SANANTONIOLACOMPAÑÍA": "SANANTONIOLACOMPANIA",    # San Vicente
        "SANCRISTÓBAL": "SANCRISTOBAL",    # San Vicente
        "LASFRÍAS": "LASFRIAS",    # San Vicente
        "LAPEÑA": "LAPENA",    # San Vicente
        "LACABAÑA": "LACABANA",    # San Vicente
        "SANJOSÉ": "SANJOSE",    # Santa Barbara
        "GUÁSIMO": "GUASIMO",    # Santa Barbara
        "LAUMBRÍA": "LAUMBRIA",    # Santa Barbara
        "NUQUÍ": "NUQUI",    # Santa Fe De Antioquia
        "KILÓMETRO2": "KILOMETRO2",    # Santa Fe De Antioquia
        "OBREGÓN": "OBREGON",    # Santa Fe De Antioquia
        "FÁTIMA": "FATIMA",    # Santa Fe De Antioquia
        "KILÓMETRO14": "KILOMETRO14",    # Santa Fe De Antioquia
        "ELGUÁSIMO": "ELGUASIMO",    # Santa Fe De Antioquia
        "ARAGÓN": "ARAGON",    # Santa Rosa De Osos
        "RÍOGRANDE": "RIOGRANDE",    # Santa Rosa De Osos
        "ELCHÁQUIRO": "ELCHAQUIRO",    # Santa Rosa De Osos
        "LAMUÑOZ": "LAMUNOZ",    # Santa Rosa De Osos
        "MONTAÑITA": "MONTANITA",    # Santa Rosa De Osos
        "LACABAÑA": "LACABANA",    # Santa Rosa De Osos
        "SANJOSÉDELAAHUMADA": "SANJOSEDELAAHUMADA",    # Santa Rosa De Osos
        "LASÁNIMAS": "LASANIMAS",    # Santa Rosa De Osos
        "SANJOSÉ": "SANJOSE",    # Santa Rosa De Osos
        "ELBOTÓN": "ELBOTON",    # Santa Rosa De Osos
        "SANTABÁRBARA": "SANTABARBARA",    # Santa Rosa De Osos
        "SANRAMÓN": "SANRAMON",    # Santa Rosa De Osos
        "ELCHAGUALO": "CHAGUALO",    # Santa Rosa De Osos
        "LAQUIEBRA": "LARAYA",    # Santo Domingo
        "LASBEATRICES-LAM": "LASBEATRICESLAM",    # Santo Domingo
        "LASÁNIMAS": "LASANIMAS",    # Santo Domingo
        "BAJOCANTAYÚS": "BAJOCANTAYUS",    # Santo Domingo
        "SANLUÍS": "SANLUIS",    # Santo Domingo
        "LAPRIMAVERA-CUATROESQUINAS": "LAPRIMAVERACUATROESQUINAS",    # Santo Domingo
        "VAINILLAL–PACHOHONDO": "VAINILLALPACHOHONDO",    # Santo Domingo
        "ELLIMÓN": "ELLIMON",    # Santo Domingo
        "ELROSARIO-REYES": "ELROSARIOREYES",    # Santo Domingo
        "DANTAS-NUSITO": "DANTASNUSITO",    # Santo Domingo
        "SANJOSÉ": "SANJOSE",    # Santo Domingo
        "CUTURÚARRIBA": "CUTURUARRIBA",    # Segovia
        "CUTURÚABAJO": "CUTURUABAJO",    # Segovia
        "SANMIGUEL": "SANMIGUELLADORADA",    # Sonson
        "RÍOARRIBA": "RIOARRIBA",    # Sonson
        "LAMONTAÑITA": "MONTANITA",    # Sonson
        "MEDIACUESTADESANJOSÉ": "MEDIACUESTADESANJOSE",    # Sonson
        "BOQUERÓN": "BOQUERON",    # Sonson
        "SANJERÓNIMO": "SANJERONIMO",    # Sonson
        "NORÍ": "NORI",    # Sonson
        "LACIÉNAGA": "LACIENAGA",    # Sonson
        "LAPAZ-SANFRANCISCO": "LAPAZSANFRANCISCO",    # Sonson
        "LAFLOR-ELTESORO": "LAFLORELTESORO",    # Sonson
        "LLANODEMONTAÑA": "LLANODEMONTANA",    # Sopetran
        "SANTABÁRBARA": "SANTABARBARA",    # Sopetran
        "MORRÓN": "MORRON",    # Sopetran
        "CÓRDOBA": "CORDOBA",    # Sopetran
        "SANNICOLÁS": "SANNICOLAS",    # Sopetran
        "SANLUÍS": "SANLUIS",    # Tamesis
        "ELLÍBANO": "ELLIBANO",    # Tamesis
        "CEDEÑOBAJO": "CEDENOBAJO",    # Tamesis
        "TRAVESÍAS": "TRAVESIAS",    # Tamesis
        "ELTACÓN": "ELTACON",    # Tamesis
        "CEDEÑOALTO": "CEDENOALTO",    # Tamesis
        "SANNICOLÁS": "SANNICOLAS",    # Tamesis
        "ELGUÁIMARO": "ELGUAIMARO",    # Taraza
        "CHUCHUÍ": "CHUCHUI",    # Taraza
        "CAÑÓNDEIGLESIAS": "CANONDEIGLESIAS",    # Taraza
        "PUQUÍ": "PUQUI",    # Taraza
        "PURÍ": "PURI",    # Taraza
        "PÉCORA": "PECORA",    # Taraza
        "MATECAÑA": "MATECANA",    # Taraza
        "LACABAÑA": "LACABANA",    # Taraza
        "TAHAMÍ": "TAHAMI",    # Taraza
        "NERÍ": "NERI",    # Taraza
        "CURUMANÁ": "CURUMANA",    # Taraza
        "QUINTERÓN": "QUINTERON",    # Taraza
        "ANAPARCÍ": "ANAPARCI",    # Taraza
        "ÁREASINLEVANTAR": "AREASINLEVANTAR",    # Taraza
        "LAUNIÓN": "LAUNION",    # Taraza
        "ELCEDRÓN": "ELCEDRON",    # Tarso
        "MORRÓN": "MORRON",    # Tarso
        "ELVOLCÁN": "ELVOLCAN",    # Titiribi
        "LAPEÑA": "LAPENA",    # Titiribi
        "SINIFANÁ": "SINIFANA",    # Titiribi
        "ELMORAL-ELTORO": "ELMORALELTORO",    # Toledo
        "SANTAMARÍA": "SANTAMARIA",    # Toledo
        "MONTEVERDENO.1": "MONTEVERDENO1",    # Turbo
        "MONTEVERDENO.2": "MONTEVERDENO2",    # Turbo
        "SANTAINÉS": "SANTAINES",    # Turbo
        "BOCASDELRÍOTURBO": "BOCASDELRIOTURBO",    # Turbo
        "SINAÍ": "SINAI",    # Turbo
        "CARACOLÍ": "CARACOLI",    # Turbo
        "LIMÓNMEDIO": "LIMONMEDIO",    # Turbo
        "MATADEPLÁTANOARRIBA": "MATADEPLATANOARRIBA",    # Turbo
        "JUANBENÍTEZ": "JUANBENITEZ",    # Turbo
        "TÍOLOPEZALTO": "TIOLOPEZALTO",    # Turbo
        "TÍOLÓPEZMEDIO": "TIOLOPEZMEDIO",    # Turbo
        "ELAZÚCAR": "ELAZUCAR",    # Turbo
        "NUEVAUNIÓN": "NUEVAUNION",    # Turbo
        "LAPIÑA": "LAPINA",    # Turbo
        "ELVOLCÁN": "ELVOLCAN",    # Turbo
        "LOSMANATÍES": "LOSMANATIES",    # Turbo
        "SANANDRÉSDETULAPA": "SANANDRESDETULAPA",    # Turbo
        "GUSTAVOMEJÍA": "GUSTAVOMEJIA",    # Turbo
        "ISAÍAS": "ISAIAS",    # Turbo
        "SANTABÁRBARAARRIBA": "SANTABARBARAARRIBA",    # Turbo
        "RANCHERÍA": "RANCHERIA",    # Turbo
        "RÍOTURBO": "RIOTURBO",    # Turbo
        "SANTABÁRBARAABAJO": "SANTABARBARAABAJO",    # Turbo
        "LASCAÑAS": "LASCANAS",    # Turbo
        "TÍOGIL": "TIOGIL",    # Turbo
        "LAILUSIÓN": "LAILUSION",    # Turbo
        "ELLIMÓN": "ELLIMON",    # Turbo
        "BOCALIMÓN": "BOCALIMON",    # Turbo
        "LEÓNABAJO": "LEONABAJO",    # Turbo
        "ELVOLCÁN": "ELVOLCAN",    # Turbo
        "ISAÍASARRIBA": "ISAIASARRIBA",    # Turbo
        "LAFRÍA": "LAFRIA",    # Turbo
        "ELLIMÓN": "ELLIMON",    # Turbo
        "ELALGODÓN": "ELALGODON",    # Turbo
        "AGUASFRÍAS": "AGUASFRIAS",    # Turbo
        "ALTOCAIMÁN": "ALTOCAIMAN",    # Turbo
        "BOCADEMATADEPLÁTANO": "BOCADEMATADEPLATANO",    # Turbo
        "TUNTÚNARRIBA": "TUNTUNARRIBA",    # Turbo
        "TUNTÚNABAJO": "TUNTUNABAJO",    # Turbo
        "TUMARADÓ": "TUMARADO",    # Turbo
        "TRAVESÍAS": "TRAVESIAS",    # Uramita
        "LIMÓNCHUPADERO": "LIMONCHUPADERO",    # Uramita
        "PEÑASBLANCAS": "PENASBLANCAS",    # Uramita
        "LIMÓNCABUYAL": "LIMONCABUYAL",    # Uramita
        "BALCÓN": "BALCON",    # Uramita
        "RÍOVERDE": "RIOVERDE",    # Uramita
        "CABAÑA": "CABANA",    # Uramita
        "MANDÉ": "GUAPANDE",    # Urrao
        "LALOMA-SANLUÍS-SANVIDAL": "LALOMASANLUISSANVIDAL",    # Urrao
        "ELSALADO-LAHONDA": "ELSALADOLAHONDA",    # Urrao
        "SANMATÍAS": "SANMATIAS",    # Urrao
        "SANJOSÉ": "SANJOSE",    # Urrao
        "ELVOLCÁN": "ELVOLCAN",    # Urrao
        "SANAGUSTÍN": "SANAGUSTIN",    # Urrao
        "SANJOAQUÍN": "SANJOAQUIN",    # Urrao
        "SANJOSÉ(LIMITESINDEFINIR)": "SANJOSE",    # Urrao
        "SANFERMÍN": "SANFERMIN",    # Valdivia
        "SANTAINÉS": "SANTAINES",    # Valdivia
        "SANTABÁRBARA": "SANTABARBARA",    # Valdivia
        "ELHIGUERÓN": "ELHIGUERON",    # Valdivia
        "LAAMÉRICA": "LAAMERICA",    # Valdivia
        "MONTEFRÍO": "MONTEFRIO",    # Valdivia
        "ELLÍBANO": "ELLIBANO",    # Valparaiso
        "SANJOSÉ": "SANJOSE",    # Valparaiso
        "PARCELACIÓNMONTENEGRO": "PARCELACIONMONTENEGRO",    # Valparaiso
        "ELCHURÚ": "ELCHURU",    # Vegachi
        "MONÁ": "MONA",    # Vegachi
        "BÉLGICA": "BELGICA",    # Vegachi
        "ELJABÓN": "ELJABON",    # Vegachi
        "LASOÑADORA": "LASONADORA",    # Vegachi
        "LAUNIÓN": "LAUNION",    # Vegachi
        "ELRINCÓN": "ELRINCON",    # Venecia
        "ELLIMÓN": "ELLIMON",    # Venecia
        "RITAPEÑASAZULES": "RITAPENASAZULES",    # Venecia
        "BUCHADÓ": "BUCHADO",    # Vigia Del Fuerte
        "LOMAMURRÍ": "LOMAMURRI",    # Vigia Del Fuerte
        "SANMIGUEL": "SANMIGUELLADORADA",    # Vigia Del Fuerte
        "SANTAMARÍA": "SANTAMARIA",    # Vigia Del Fuerte
        "PUERTOMEDELLÍN": "PUERTOMEDELLIN",    # Vigia Del Fuerte
        "BRICEÑO": "BRICEN0",    # Vigia Del Fuerte
        "SANMARTÍN": "SANMARTIN",    # Vigia Del Fuerte
        "BELÉN": "BELEN",    # Vigia Del Fuerte
        "PARTADÓ": "PARTADO",    # Vigia Del Fuerte
        "JARAPETÓ": "JARAPETO",    # Vigia Del Fuerte
        "GUAGUANDÓ": "GUAGUANDO",    # Vigia Del Fuerte
        "GENGADÓ": "GENGADO",    # Vigia Del Fuerte
        "ELJARDÍN": "ELJARDIN",    # Yali
        "MONTAÑITA": "MONTANITA",    # Yali
        "BRICEÑO": "BRICEN0",    # Yali
        "LASAGÜITAS": "LASAGUITAS",    # Yali
        "CEDEÑO": "CEDENO",    # Yarumal
        "LLANOSDECUIVÁ": "LLANOSDECUIVA",    # Yarumal
        "ELLLANO-YOLOMBAL": "ELLLANOYOLOMBAL",    # Yarumal
        "OCHALÍ": "OCHALI",    # Yarumal
        "CAÑAVERAL": "CANAVERAL",    # Yarumal
        "TOBÓN": "TOBON",    # Yarumal
        "LACONSPIRACIÓN": "LACONSPIRACION",    # Yarumal
        "ELRETIRO": "RETIRO",    # Yarumal
        "RÍOABAJO": "RIOABAJO",    # Yarumal
        "ELRUBÍ": "ELRUBI",    # Yolombo
        "LASFRÍAS": "LASFRIAS",    # Yolombo
        "BAREÑO": "BARENO",    # Yolombo
        "ELTAPÓN": "ELTAPON",    # Yolombo
        "DOÑAANA": "DONAANA",    # Yolombo
        "LOSANDES": "LOSANDESSOTOMAYOR",    # Yolombo
        "ELJARDÍN": "ELJARDIN",    # Yolombo
        "ESTACIÓNSOFÍA": "ESTACIONSOFIA",    # Yolombo
        "SANAGUSTÍN": "SANAGUSTIN",    # Yolombo
        "MULATOS": "LOSMULATOS",    # Yolombo
        "BÉLGICA": "BELGICA",    # Yolombo
        "LACABAÑA": "LACABANA",    # Yolombo
        "ELPICHÓN": "ELPICHON",    # Yolombo
        "BELLAVISTA-LAJOSEFINA": "BELLAVISTALAJOSEFINA",    # Yolombo
        "ELRUBÍ-LAFLORESTA": "ELRUBILAFLORESTA",    # Yolombo
        "CAÑOBODEGAS": "CANOBODEGAS",    # Yondo
        "KILÓMETROCINCO": "KILOMETROCINCO",    # Yondo
        "CAÑOBONITO": "CANOBONITO",    # Yondo
        "CAÑONEGRO": "CANONEGRO",    # Yondo
        "SANLUÍSBELTRÁN": "SANLUISBELTRAN",    # Yondo
        "CAÑOBLANCO": "CANOBLANCO",    # Yondo
        "LAUNIÓN": "LAUNION",    # Yondo
        "PEÑASBLANCAS": "PENASBLANCAS",    # Yondo
        "LACABAÑA": "LACABANA",    # Yondo
        "LAORQUÍDEA": "LAORQUIDEA",    # Yondo
        "CAÑOHUILA": "CANOHUILA",    # Yondo
        "CAÑOLASCRUCES": "CANOLASCRUCES",    # Yondo
        "ELLIMÓN": "ELLIMON",    # Zaragoza
        "BOCASDECANÁ": "BOCASDECANA",    # Zaragoza
        "ELRETIRO": "RETIRO",    # Zaragoza
        "RÍOVIEJO": "RIOVIEJO",    # Zaragoza
        "AQUÍSI": "AQUISI",    # Zaragoza
        "CIMARRÓN": "CIMARRON",    # Zaragoza
        "CANÁMEDIO": "CANAMEDIO",    # Zaragoza
        "NUEVAILUSIÓN": "NUEVAILUSION",    # Zaragoza
        "CAÑOLAOCHO": "CANOLAOCHO",    # Zaragoza
        "JALA-JALA": "JALAJALA",    # Zaragoza
        "CAÑOLATRES": "CANOLATRES",    # Zaragoza
        "POCUNÉMEDIO": "POCUNEMEDIO",    # Zaragoza
        "BOCASDEPOCUNÉ": "BOCASDEPOCUNE",    # Zaragoza
        "VILLAAMARÁ": "VILLAAMARA",    # Zaragoza
        "POCUNÉABAJO": "POCUNEABAJO",    # Zaragoza
        "VILLASEVERÁ": "VILLASEVERA"    # Zaragoza

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
        ruta_nac = os.path.join(RUTA_RAIZ, "data", "PobCol1912_2100.csv")
        df_nac = pd.read_csv(ruta_nac, sep=',')
        if len(df_nac.columns) < 2: df_nac = pd.read_csv(ruta_nac, sep=';')
        df_nac.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().title() for c in df_nac.columns]
        df_nac = df_nac.rename(columns={'Male': 'Hombres', 'Female': 'Mujeres', 'Ano': 'Año'})
        
        ruta_mun_1 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_Colombia.csv")
        ruta_mun_2 = os.path.join(RUTA_RAIZ, "data", "Pob_mpios_colombia.csv")
        if os.path.exists(ruta_mun_1): df_mun = pd.read_csv(ruta_mun_1, sep=',')
        elif os.path.exists(ruta_mun_2): df_mun = pd.read_csv(ruta_mun_2, sep=',')
        else: raise FileNotFoundError("Archivo municipal no encontrado.")
        if len(df_mun.columns) < 2: df_mun = pd.read_csv(ruta_mun_1 if os.path.exists(ruta_mun_1) else ruta_mun_2, sep=';')
            
        df_mun.columns = [str(c).replace('\ufeff', '').replace('"', '').strip().lower() for c in df_mun.columns]
        df_mun = df_mun.rename(columns={'poblacion': 'Total'})
        df_mun['depto_nom'] = df_mun['depto_nom'].str.title()
        df_mun['municipio'] = df_mun['municipio'].str.title()
        
        # MAGIA 2: Asignar Región con excepciones (Urabá)
        def asignar_region(fila):
            uraba_caribe = ["TURBO", "NECOCLI", "SAN JUAN DE URABA", "ARBOLETES", "SAN PEDRO DE URABA", "APARTADO", "CAREPA", "CHIGORODO", "MUTATA"]
            if normalizar_texto(fila['municipio']) in [normalizar_texto(m) for m in uraba_caribe]:
                return 'Caribe'
                
            depto = fila['depto_nom']
            for region, deptos in REGIONES_COL.items():
                if depto in deptos: return region
            return "Sin Región"
            
        df_mun['Macroregion'] = df_mun.apply(asignar_region, axis=1)
        
        df_ver = pd.DataFrame()
        ruta_ver_1 = os.path.join(RUTA_RAIZ, "data", "veredas_Antioquia.csv")
        ruta_ver_2 = os.path.join(RUTA_RAIZ, "data", "veredas_Antioquia.xlsx")
        
        if os.path.exists(ruta_ver_1): df_ver = pd.read_csv(ruta_ver_1, sep=';')
        elif os.path.exists(ruta_ver_2): df_ver = pd.read_excel(ruta_ver_2)
            
        if not df_ver.empty:
            df_ver.columns = [str(c).replace('\ufeff', '').strip() for c in df_ver.columns]
            if 'Municipio' in df_ver.columns: df_ver['Municipio'] = df_ver['Municipio'].str.title()
            if 'Vereda' in df_ver.columns: df_ver['Vereda'] = df_ver['Vereda'].str.title()
            if 'Poblacion_hab' in df_ver.columns: df_ver['Poblacion_hab'] = pd.to_numeric(df_ver['Poblacion_hab'], errors='coerce').fillna(0)

        return df_nac, df_mun, df_ver
    except Exception as e:
        st.error(f"🚨 Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_nac, df_mun, df_ver = cargar_datos_limpios()
if df_nac.empty or df_mun.empty: st.stop()

# --- 2. MODELOS MATEMÁTICOS ---
def modelo_lineal(x, m, b): return m * x + b
def modelo_exponencial(x, p0, r): return p0 * np.exp(r * (x - 2005))
def modelo_logistico(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))

# --- 3. CONFIGURACIÓN Y FILTROS LATERALES ---
st.sidebar.header("⚙️ 1. Selección Territorial")
escala_sel = st.sidebar.selectbox("Escala Territorial", ["Nacional", "Regional (Macroregiones)", "Departamental", "Municipal", "Veredal (Antioquia)"])

años_hist, pob_hist = [], []
df_mapa_base = pd.DataFrame()

if escala_sel == "Nacional":
    df_agrup_nac = df_nac.groupby('Año')[['Hombres', 'Mujeres']].sum().reset_index()
    años_hist = df_agrup_nac['Año'].values
    pob_hist = (df_agrup_nac['Hombres'] + df_agrup_nac['Mujeres']).values
    titulo_terr = "Colombia (Nacional)"
    df_mapa_base = df_mun.groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel == "Regional (Macroregiones)":
    regiones_list = sorted(df_mun['Macroregion'].unique())
    reg_sel = st.sidebar.selectbox("Seleccione la Macroregión", regiones_list)
    df_terr = df_mun[df_mun['Macroregion'] == reg_sel].groupby('año')['Total'].sum().reset_index()
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values
    titulo_terr = f"Región {reg_sel}"
    df_mapa_base = df_mun[df_mun['Macroregion'] == reg_sel].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
    df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})

elif escala_sel in ["Departamental", "Municipal"]:
    deptos = sorted(df_mun['depto_nom'].dropna().unique())
    depto_sel = st.sidebar.selectbox("Departamento", deptos)
    
    if escala_sel == "Departamental":
        df_terr = df_mun[df_mun['depto_nom'] == depto_sel].groupby('año')['Total'].sum().reset_index()
        titulo_terr = depto_sel
        df_mapa_base = df_mun[df_mun['depto_nom'] == depto_sel].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
        df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})
    else:
        mpios = sorted(df_mun[df_mun['depto_nom'] == depto_sel]['municipio'].dropna().unique())
        mpio_sel = st.sidebar.selectbox("Municipio", mpios)
        df_terr = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby('año')['Total'].sum().reset_index()
        titulo_terr = f"{mpio_sel}, {depto_sel}"
        df_mapa_base = df_mun[(df_mun['depto_nom'] == depto_sel) & (df_mun['municipio'] == mpio_sel)].groupby(['municipio', 'depto_nom', 'año', 'area_geografica'])['Total'].sum().reset_index()
        df_mapa_base = df_mapa_base.rename(columns={'municipio': 'Territorio', 'depto_nom': 'Padre'})
        
    años_hist = df_terr['año'].values
    pob_hist = df_terr['Total'].values

elif escala_sel == "Veredal (Antioquia)":
    # 1. Agregamos la opción "TODOS" al principio de la lista
    mpios_veredas = ["TODOS (Ver Mapa Completo)"] + sorted(df_ver['Municipio'].dropna().unique())
    mpio_sel = st.sidebar.selectbox("Municipio (Antioquia)", mpios_veredas)
    
    if mpio_sel == "TODOS (Ver Mapa Completo)":
        # Sumamos toda la población rural de Antioquia como base histórica general
        df_rural_ant = df_mun[(df_mun['depto_nom'] == 'Antioquia') & (df_mun['area_geografica'] == 'rural')]
        df_hist_rural = df_rural_ant.groupby('año')['Total'].sum().reset_index()
        
        años_hist = df_hist_rural['año'].values
        pob_hist = df_hist_rural['Total'].values
        titulo_terr = "Todas las Veredas (Región Disponible)"
        
        # Inyectamos la tabla completa de veredas al mapa
        df_mapa_base = df_ver.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'
        
    else:
        # El código original para un solo municipio
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
        titulo_terr = f"Vereda {vereda_sel} ({mpio_sel})"
        
        df_mapa_base = df_mpio_veredas.copy()
        df_mapa_base = df_mapa_base.rename(columns={'Vereda': 'Territorio', 'Municipio': 'Padre', 'Poblacion_hab': 'Total'})
        df_mapa_base['año'] = 2020 
        df_mapa_base['area_geografica'] = 'rural'
        
# --- 4. CÁLCULO DE PROYECCIONES ---
x_hist = np.array(años_hist, dtype=float)
y_hist = np.array(pob_hist, dtype=float)

año_maximo = int(max(df_nac['Año'].max(), 2100))
x_proj = np.arange(1950, año_maximo + 1, 1) 

proyecciones = {'Año': x_proj, 'Real': [np.nan]*len(x_proj)}
for i, año in enumerate(x_proj):
    if año in x_hist: proyecciones['Real'][i] = y_hist[np.where(x_hist == año)[0][0]]

try:
    popt_lin, _ = curve_fit(modelo_lineal, x_hist, y_hist)
    proyecciones['Lineal'] = np.maximum(0, modelo_lineal(x_proj, *popt_lin))
except: proyecciones['Lineal'] = [np.nan]*len(x_proj)

try:
    popt_exp, _ = curve_fit(modelo_exponencial, x_hist, y_hist, p0=[max(1, y_hist[0]), 0.01], maxfev=10000)
    proyecciones['Exponencial'] = modelo_exponencial(x_proj, *popt_exp)
except: proyecciones['Exponencial'] = proyecciones['Lineal']

try:
    K_guess = max(1, max(y_hist)) * 1.5
    popt_log, _ = curve_fit(modelo_logistico, x_hist, y_hist, p0=[K_guess, 0.05, 2020], maxfev=10000)
    proyecciones['Logístico'] = modelo_logistico(x_proj, *popt_log)
    param_K = popt_log[0]
except: 
    proyecciones['Logístico'] = proyecciones['Lineal']
    param_K = "N/A"

df_proj = pd.DataFrame(proyecciones)

# --- CONFIGURACIÓN DE PESTAÑAS (TABS) ---
tab_modelos, tab_mapas, tab_descargas = st.tabs(["📊 Pirámides y Modelos", "🌍 Geovisor Espacial", "💾 Descarga de Datos"])

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

    años_disp = sorted(df_nac['Año'].unique())
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
        global df_piramide_final
        try: pob_modelo = df_proj[df_proj['Año'] == año_obj][columna_modelo].values[0]
        except: pob_modelo = np.nan
            
        if pd.isna(pob_modelo): pob_modelo = df_proj[df_proj['Año'] == año_obj]['Logístico'].values[0]
        
        df_fnac = df_nac[df_nac['Año'] == año_obj].copy()
        pob_nacional_tot = df_fnac['Hombres'].sum() + df_fnac['Mujeres'].sum()
        df_fnac['Prop_H'] = df_fnac['Hombres'] / pob_nacional_tot
        df_fnac['Prop_M'] = df_fnac['Mujeres'] / pob_nacional_tot
        
        df_fnac['Hom_Terr'] = df_fnac['Prop_H'] * pob_modelo
        df_fnac['Muj_Terr'] = df_fnac['Prop_M'] * pob_modelo
        
        df_pir = pd.DataFrame({'Edad': df_fnac['Edad'], 'Hombres': df_fnac['Hom_Terr'] * -1, 'Mujeres': df_fnac['Muj_Terr']})
        cortes = list(range(0, 105, 5))
        etiquetas = [f"{i}-{i+4}" for i in range(0, 100, 5)]
        df_pir['Rango'] = pd.cut(df_pir['Edad'], bins=cortes, labels=etiquetas, right=False)
        df_agrupado = df_pir.groupby('Rango', observed=True)[['Hombres', 'Mujeres']].sum().reset_index()
        
        df_melt = pd.melt(df_agrupado, id_vars=['Rango'], value_vars=['Hombres', 'Mujeres'], var_name='Sexo', value_name='Poblacion')
        df_piramide_final = df_melt.copy()
        
        fig_pir = px.bar(df_melt, y='Rango', x='Poblacion', color='Sexo', orientation='h', color_discrete_map={'Hombres': '#2563eb', 'Mujeres': '#db2777'})
        
        max_x = max(abs(df_melt['Poblacion'].min()), df_melt['Poblacion'].max()) * 1.1
        fig_pir.update_layout(barmode='relative', xaxis_title="Habitantes", yaxis_title="Edades", xaxis=dict(range=[-max_x, max_x]))
        fig_pir.update_traces(hovertemplate="%{y}: %{x:,.0f}")
        
        ph_titulo_año.markdown(f"### ⏳ **Año Proyectado: {año_obj}**")
        ph_grafico_pir.plotly_chart(fig_pir, width='stretch')
        
        tot_h, tot_m = df_fnac['Hom_Terr'].sum(), df_fnac['Muj_Terr'].sum()
        ind_m = (tot_h / tot_m) * 100 if tot_m > 0 else 0
        
        ph_metrica_pob.metric("Población Proyectada", f"{pob_modelo:,.0f}")
        ph_metrica_hom.metric("Total Hombres", f"{tot_h:,.0f}")
        ph_metrica_muj.metric("Total Mujeres", f"{tot_m:,.0f}")
        ph_metrica_ind.metric("Índice Masculinidad", f"{ind_m:.1f} H x 100 M")

    if iniciar_animacion:
        for a in años_disp:
            if a >= 1950:
                renderizar_piramide(a)
                time.sleep(velocidad_animacion)
    else:
        renderizar_piramide(año_sel)
        
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

# ==========================================
# PESTAÑA 2: MAPA DEMOGRÁFICO (GEOVISOR)
# ==========================================
with tab_mapas:
    st.subheader(f"🗺️ Geovisor de Distribución Poblacional - {titulo_terr} ({año_sel})")
    
    if escala_sel != "Veredal (Antioquia)":
        area_mapa = st.radio("Filtro de Zona Geográfica:", ["Total", "Urbano", "Rural"], horizontal=True)
    else:
        area_mapa = "Rural"
        st.info("ℹ️ A escala veredal, toda la población es considerada rural.")

    if 'año' in df_mapa_base.columns:
        df_mapa_año = df_mapa_base[df_mapa_base['año'] == min(max(df_mapa_base['año']), año_sel)].copy()
    else:
        df_mapa_año = df_mapa_base.copy()

    if area_mapa == "Total":
        df_mapa_plot = df_mapa_año.groupby(['Territorio', 'Padre'])['Total'].sum().reset_index()
    else:
        df_mapa_plot = df_mapa_año[df_mapa_año['area_geografica'] == area_mapa.lower()].copy()

    col_map1, col_map2 = st.columns([1, 3])
    
    with col_map1:
        st.markdown("**⚙️ Configuración del GeoJSON**")
        if escala_sel == "Veredal (Antioquia)": 
            sugerencia_geo = "Veredas_Antioquia_TOTAL_UrbanoyRural.geojson"
            sugerencia_prop = "properties.NOMBRE_VER"
            sugerencia_padre = "properties.NOMB_MPIO"
        else: 
            sugerencia_geo = "mgn_municipios_optimizado.geojson" # Asegúrate de que este sea tu nombre de archivo
            sugerencia_prop = "properties.MPIO_CNMBR"
            sugerencia_padre = "properties.DPTO_CCDGO" # AHORA USAMOS EL CÓDIGO DANE
            
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
            
    st.dataframe(df_mapa_plot[['Territorio', 'Padre', 'Total', 'MATCH_ID', 'En_Mapa']].sort_values('Total', ascending=False), use_container_width=True)

# ==========================================
# PESTAÑA 3: DESCARGAS Y EXPORTACIÓN
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
