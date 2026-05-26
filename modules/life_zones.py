# modules/life_zones.py

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import io
from contextlib import contextmanager
from rasterio.features import rasterize, shapes
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape, Point

# --- CONSTANTES Y DICCIONARIOS ---
holdridge_zone_map_simplified = {
    "Nival": 1,
    "Tundra pluvial (tp-A)": 2,
    "Tundra húmeda (th-A)": 3,
    "Tundra seca (ts-A)": 4,
    "Páramo pluvial subalpino (pp-SA)": 5,
    "Páramo muy húmedo subalpino (pmh-SA)": 6,
    "Páramo seco subalpino (ps-SA)": 7,
    "Bosque pluvial Montano (bp-M)": 8,
    "Bosque muy húmedo Montano (bmh-M)": 9,
    "Bosque húmedo Montano (bh-M)": 10,
    "Bosque seco Montano (bs-M)": 11,
    "Monte espinoso Montano (me-M)": 12,
    "Bosque pluvial Premontano (bp-PM)": 13,
    "Bosque muy húmedo Premontano (bmh-PM)": 14,
    "Bosque húmedo Premontano (bh-PM)": 15,
    "Bosque seco Premontano (bs-PM)": 16,
    "Monte espinoso Premontano (me-PM)": 17,
    "Bosque pluvial Tropical (bp-T)": 18,
    "Bosque muy húmedo Tropical (bmh-T)": 19,
    "Bosque húmedo Tropical (bh-T)": 20,
    "Bosque seco Tropical (bs-T)": 21,
    "Monte espinoso Tropical (me-T)": 22,
    "Zona Desconocida": 0,
}

holdridge_int_to_name_simplified = {v: k for k, v in holdridge_zone_map_simplified.items()}

holdridge_colors = {
    1: "#FFFFFF", 2: "#B0E0E6", 3: "#87CEEB", 4: "#708090",
    5: "#8A2BE2", 6: "#9370DB", 7: "#D8BFD8", 
    8: "#00008B", 9: "#006400", 10: "#228B22", 11: "#9ACD32", 12: "#F0E68C",
    13: "#0000CD", 14: "#008000", 15: "#32CD32", 16: "#FFFF00", 17: "#DAA500",
    18: "#191970", 19: "#2E8B57", 20: "#7CFC00", 21: "#FFA500", 22: "#FF4500",
    0: "#000000",
}

@contextmanager
def open_raster_source(source):
    if isinstance(source, (io.BytesIO, bytes)) or hasattr(source, 'read'):
        with MemoryFile(source) as memfile:
            with memfile.open() as src:
                yield src
    else:
        with rasterio.open(source) as src:
            yield src

def classify_life_zone_alt_ppt(altitude, ppt):
    if pd.isna(altitude) or pd.isna(ppt) or altitude < 0 or ppt <= 0:
        return 0
    if altitude >= 4500: return 1
    if altitude >= 3800: return 2 if ppt >= 1000 else (3 if ppt >= 500 else 4)
    if altitude >= 3000: return 5 if ppt >= 2000 else (6 if ppt >= 1000 else 7)
    if altitude >= 2000:
        if ppt >= 4000: return 8
        elif ppt >= 2000: return 9
        elif ppt >= 1000: return 10
        elif ppt >= 500: return 11
        else: return 12
    if altitude >= 1000:
        if ppt >= 4000: return 13
        elif ppt >= 2000: return 14
        elif ppt >= 1000: return 15
        elif ppt >= 500: return 16
        else: return 17
    if ppt >= 8000: return 18
    elif ppt >= 4000: return 19
    elif ppt >= 2000: return 20
    elif ppt >= 1000: return 21
    else: return 22

_vectorized_classify = np.vectorize(classify_life_zone_alt_ppt)

def generate_life_zone_map(dem_input, ppt_input, mask_geometry=None, downscale_factor=4, delta_temp=0.0):
    """
    Genera el raster con soporte para Delta de Temperatura (Migración Altitudinal)
    y enmascaramiento estricto por polígono.
    """
    try:
        if downscale_factor is None or downscale_factor <= 0: downscale_factor = 1
        dst_crs = "EPSG:4326"

        with open_raster_source(dem_input) as dem_src:
            dst_width = max(1, dem_src.width // downscale_factor)
            dst_height = max(1, dem_src.height // downscale_factor)

            # Si hay un polígono, ajustamos el bounding box a los límites de la cuenca
            if mask_geometry is not None and not mask_geometry.empty:
                bounds = mask_geometry.to_crs(dst_crs).total_bounds
            else:
                bounds = dem_src.bounds

            dst_transform, dst_width, dst_height = calculate_default_transform(
                dem_src.crs, dst_crs, dem_src.width, dem_src.height, 
                *bounds, dst_width=dst_width, dst_height=dst_height
            )

            dem_resampled = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_resampled,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        with open_raster_source(ppt_input) as ppt_src:
            ppt_resampled = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(ppt_src, 1),
                destination=ppt_resampled,
                src_transform=ppt_src.transform,
                src_crs=ppt_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
            )

        # 🌡️ Inyección de Cambio Climático: 1°C = -154m de altitud equivalente
        shift_metros = delta_temp * (100.0 / 0.65)
        dem_simulado = dem_resampled - shift_metros

        dem_mask = np.isnan(dem_resampled)
        ppt_mask = np.isnan(ppt_resampled)
        valid_mask = (~dem_mask) & (~ppt_mask) & (dem_resampled > -500) & (ppt_resampled >= 0)

        classified_raster = np.zeros((dst_height, dst_width), dtype=np.int16)

        if np.any(valid_mask):
            zone_ints = _vectorized_classify(dem_simulado[valid_mask], ppt_resampled[valid_mask])
            classified_raster[valid_mask] = zone_ints.astype(np.int16)

        # ✂️ Recorte Perfecto (Máscara Vectorial)
        if mask_geometry is not None and not mask_geometry.empty:
            try:
                mask_reproj = mask_geometry.to_crs(dst_crs)
                shapes_list = [(geom, 1) for geom in mask_reproj.geometry]
                mask_raster = rasterize(
                    shapes_list, out_shape=(dst_height, dst_width),
                    transform=dst_transform, fill=0, dtype=np.uint8,
                )
                classified_raster = np.where(mask_raster == 1, classified_raster, 0)
            except Exception as e:
                st.warning(f"Error en recorte vectorial: {e}")

        output_profile = {
            "driver": "GTiff", "dtype": "int16", "nodata": 0,
            "width": dst_width, "height": dst_height, "count": 1,
            "crs": dst_crs, "transform": dst_transform, "compress": "lzw",
        }

        return classified_raster, output_profile, holdridge_int_to_name_simplified, holdridge_colors, dst_transform

    except Exception as e:
        st.error(f"Error generando mapa de Holdridge: {e}")
        return None, None, None, None, None

def vectorize_raster_to_gdf(raster_array, transform, crs):
    try:
        mask = raster_array != 0
        results = ({"properties": {"id_zona": v}, "geometry": s}
                   for i, (s, v) in enumerate(shapes(raster_array, mask=mask, transform=transform)))
        geoms, ids = [], []
        for r in results:
            geoms.append(shape(r["geometry"]))
            ids.append(r["properties"]["id_zona"])

        if not geoms: return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame({"id_zona": ids}, geometry=geoms, crs=crs)
        gdf["zona_vida"] = gdf["id_zona"].map(holdridge_int_to_name_simplified)
        return gdf
    except:
        return gpd.GeoDataFrame()

# 🏔️ NUEVA FUNCIÓN EXTRACTORA DE ALTITUD PARA ESTACIONES
def extract_elevation_from_dem(gdf_points, dem_input):
    """
    Toma un GeoDataFrame de estaciones y les asigna su elevación real pinchando el DEM.
    """
    try:
        # Aseguramos que los puntos estén en WGS84 para cruzar con el DEM
        gdf_points = gdf_points.to_crs("EPSG:4326")
        coords = [(geom.x, geom.y) for geom in gdf_points.geometry]
        
        elevations = []
        with open_raster_source(dem_input) as src:
            for val in src.sample(coords):
                # val[0] es el valor del pixel pinchado
                elevations.append(val[0] if val[0] != src.nodata else np.nan)
                
        gdf_points['altitud_dem'] = elevations
        return gdf_points
    except Exception as e:
        st.error(f"Error extrayendo elevación: {e}")
        return gdf_points
