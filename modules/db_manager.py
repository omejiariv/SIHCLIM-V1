# modules/db_manager.py

import json
import streamlit as st
import geopandas as gpd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Obtener la URL de conexión desde secrets.toml
try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except Exception:
    DATABASE_URL = None

# 🚀 FIX DEFINITIVO: Almacenamos el motor en la memoria global del servidor.
# Esto evita que Supabase colapse por exceso de peticiones simultáneas.
@st.cache_resource(show_spinner=False)
def get_engine():
    db_url = DATABASE_URL
    
    if not db_url:
        st.error("Error crítico: No se encontró DATABASE_URL en los secrets.")
        return None

    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,      
            pool_recycle=1800,       # 👈 Reducido a 30 mins para ajustarse a Supabase
            pool_size=5,             # 👈 Límite seguro de conexiones simultáneas
            max_overflow=10,         # 👈 Margen de emergencia
            connect_args={
                'sslmode': 'require', 
                'client_encoding': 'utf8',
                # 🛡️ Parámetros avanzados para mantener la línea viva (Keep-Alive)
                'keepalives': 1,
                'keepalives_idle': 30,
                'keepalives_interval': 10,
                'keepalives_count': 5
            }
        )
        return engine
        
    except Exception as e:
        st.error(f"Error creando engine: {e}")
        return None

def init_db():
    """
    Inicializa la tabla de preferencias en PostgreSQL si no existe.
    """
    engine = get_engine()
    if engine is not None:
        try:
            with engine.connect() as conn:
                # Sintaxis PostgreSQL
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        preference_key TEXT NOT NULL,
                        preference_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                    )
                )
                # Índice para búsquedas rápidas
                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_user_pref ON user_preferences (username, preference_key);
                """
                    )
                )
                conn.commit()
        except SQLAlchemyError as e:
            st.error(f"Error inicializando DB: {e}")


def save_user_preference(username, key, value):
    """
    Guarda o actualiza una preferencia.
    """
    engine = get_engine()
    if engine is not None:
        try:
            # Serializar si es objeto complejo
            if isinstance(value, (dict, list)):
                val_str = json.dumps(value)
            else:
                val_str = str(value)

            with engine.connect() as conn:
                # Lógica UPSERT simple: Borrar e Insertar
                conn.execute(
                    text(
                        """
                    DELETE FROM user_preferences
                    WHERE username = :user AND preference_key = :key
                """
                    ),
                    {"user": username, "key": key},
                )

                conn.execute(
                    text(
                        """
                    INSERT INTO user_preferences (username, preference_key, preference_value)
                    VALUES (:user, :key, :val)
                """
                    ),
                    {"user": username, "key": key, "val": val_str},
                )

                conn.commit()
            return True
        except SQLAlchemyError as e:
            st.error(f"Error guardando preferencia: {e}")
            return False
    return False


def get_user_preference(username, key, default=None):
    """
    Recupera una preferencia específica.
    """
    engine = get_engine()
    if engine is not None:
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT preference_value FROM user_preferences
                    WHERE username = :user AND preference_key = :key
                    LIMIT 1
                """
                    ),
                    {"user": username, "key": key},
                ).fetchone()

                if result:
                    val = result[0]
                    # Intentar deserializar JSON
                    try:
                        return json.loads(val)
                    except:
                        return val
        except SQLAlchemyError:
            # st.error(f"Error leyendo DB: {e}") # Opcional: silenciar en producción
            pass

    return default

import streamlit as st
import geopandas as gpd

@st.cache_data(show_spinner=False, ttl=3600) # Cache de 1 hora para no saturar Supabase
def cargar_concesiones_maestro():
    """
    Descarga y almacena en caché el archivo unificado de concesiones de agua 
    directamente desde el bucket público de Supabase.
    """
    url_concesiones = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Metabolismo_Hidrico_Antioquia_Maestro.geojson"
    
    try:
        # GeoPandas es lo suficientemente inteligente para leer URLs públicas directamente
        gdf_concesiones = gpd.read_file(url_concesiones)
        
        # Opcional: Aseguramos que el Caudal sea numérico por si acaso
        if 'Caudal_Lps' in gdf_concesiones.columns:
            gdf_concesiones['Caudal_Lps'] = pd.to_numeric(gdf_concesiones['Caudal_Lps'], errors='coerce').fillna(0)
            
        return gdf_concesiones
    except Exception as e:
        st.error(f"⚠️ Error conectando con la Nube (Concesiones): {e}")
        return None
