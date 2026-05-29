# modules/exportar_cobertura_2026.py

import os
import time
import ee
import geopandas as gpd

def lanzar_exportacion_ge_to_drive(gdf_zona, nombre_territorio):
    """
    Extrae la cobertura de suelo 2026 desde Google Earth Engine (Dynamic World)
    y la exporta directamente a la raíz de Google Drive del usuario.
    """
    print("🚀 Inicializando Google Earth Engine...")
    try:
        ee.Initialize()
    except Exception as e:
        print("💼 GEE no inicializado. Intentando autenticación...")
        ee.Authenticate()
        ee.Initialize()

    # 1. Convertir la geometría de tu GeoDataFrame a un formato compatible con GEE
    print(f"📦 Procesando geometría para: {nombre_territorio}")
    # Aseguramos que la geometría esté en WGS84 (Grados)
    gdf_wgs84 = gdf_zona.to_crs(epsg=4326)
    geom_unificada = gdf_wgs84.geometry.unary_union
    
    # Pasamos el JSON de la geometría a Earth Engine
    roi_ee = ee.Geometry(geom_unificada.__geo_interface__)

    # 2. Llamar a la colección Dynamic World V1 (Cobertura Terrestre en Tiempo Real)
    print("🌿 Filtrando colección Dynamic World para el año 2026...")
    dw_collection = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                     .filterBounds(roi_ee)
                     .filterDate('2026-01-01', '2026-12-31')) # Ajustado al año actual

    # 3. Obtener la clasificación dominante (Moda) del año para limpiar nubes y sombras
    # Seleccionamos la banda 'label' que contiene los códigos de cobertura (0: Agua, 1: Árboles, etc.)
    imagen_cobertura_2026 = dw_collection.select('label').reduce(ee.Reducer.mode()).clip(roi_ee)

    # 4. Configurar la exportación directa a tu REPOSITORIO de Google Drive
    # Nombre del archivo final sin espacios ni caracteres especiales
    nombre_archivo = f"Cob2026_{nombre_territorio.replace(' ', '_')}"
    
    print(f"📥 Configurando tarea de exportación para Google Drive: '{nombre_archivo}.tif'")
    
    tarea = ee.batch.Export.image.toDrive(
        image=imagen_cobertura_2026,
        description=f"Export_Cob_2026_{nombre_territorio}",
        folder="SIHCLI_Rasters",  # Nombre de la carpeta que se creará automáticamente en tu Drive
        fileNamePrefix=nombre_archivo,
        region=roi_ee,
        scale=10,                 # Resolución nativa de Sentinel-2 (10 metros por píxel)
        crs='EPSG:4326',          # Guardado directamente en el sistema de coordenadas universal
        maxPixels=1e9             # Blindaje para evitar bloqueos en zonas masivas
    )

    # 5. Encender los motores en la nube de Google
    tarea.start()
    print("\n✅ ¡La tarea ha sido enviada con éxito a los servidores de Google!")
    print("----------------------------------------------------------------")
    print(f"📂 Se creará una carpeta llamada 'SIHCLI_Rasters' en tu Google Drive.")
    print(f"⏳ El archivo '{nombre_archivo}.tif' aparecerá allí en unos minutos.")
    print("----------------------------------------------------------------")
    
    # 6. Monitor de estado opcional en consola
    while tarea.active():
        print(f"🔄 Procesando píxeles en la nube... Estado actual: {tarea.status()['state']}")
        time.sleep(15)
        
    print(f"🏁 Tarea finalizada. Resultado: {tarea.status()['state']}")

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Puedes probarlo cargando temporalmente tu archivo de subcuencas o municipios
    # gdf = gpd.read_file("ruta_a_tu_archivo.geojson")
    # lanzar_exportacion_ge_to_drive(gdf, "Rio Grande Chico")
    pass
