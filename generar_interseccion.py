import geopandas as gpd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. Cargando mapas...")
mpios = gpd.read_file('data/MunicipiosAntioquia.geojson')
cuencas = gpd.read_file('data/SubcuencasAinfluencia.geojson')

# 2. Proyectar al sistema MAGNA-SIRGAS (Origen Nacional EPSG:9377) para cálculos precisos de área en metros
print("2. Proyectando mapas para cálculo de áreas...")
mpios = mpios.to_crs(epsg=9377)
cuencas = cuencas.to_crs(epsg=9377)

# 3. Calcular el área original de cada municipio en Hectáreas (Ha)
mpios['area_mpio_ha'] = mpios.geometry.area / 10000

# 4. Cruzar los mapas (Intersección Espacial)
print("3. Cruzando polígonos (esto puede tardar unos segundos)...")
interseccion = gpd.overlay(mpios, cuencas, how='intersection')

# 5. Calcular el área de los fragmentos resultantes
interseccion['area_fragmento_ha'] = interseccion.geometry.area / 10000

# 6. Calcular el porcentaje exacto
interseccion['porcentaje_en_cuenca'] = (interseccion['area_fragmento_ha'] / interseccion['area_mpio_ha']) * 100

# 7. Limpiar datos (Ignorar polígonos astilla que sean menos del 0.1%)
df_final = interseccion[interseccion['porcentaje_en_cuenca'] > 0.1].copy()

# Seleccionamos las columnas útiles basadas en los archivos que subiste
df_export = df_final[['MPIO_CNMBR', 'Zona', 'SUBC_LBL', 'N_NSS1', 'area_fragmento_ha', 'porcentaje_en_cuenca']]
df_export.columns = ['Municipio', 'Zona_Hidrografica', 'Subcuenca', 'Sistema', 'Area_Ha', 'Porcentaje']

# 8. Guardar el archivo mágico
df_export.to_csv('data/cuencas_mpios_proporcion.csv', index=False)
print("✅ Archivo 'cuencas_mpios_proporcion.csv' generado con éxito en la carpeta data/")
