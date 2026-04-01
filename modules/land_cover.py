# modules/land_cover.py

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import os
import io
import base64
import urllib.request
import tempfile
import streamlit as st
import matplotlib.pyplot as plt

# --- 🚀 CLOUD SMART CACHE ---
@st.cache_data(show_spinner=False)
def get_cached_raster(url):
    """Descarga el raster a un directorio temporal local la primera vez."""
    if not url or not url.startswith("http"): return url
    filename = url.split("/")[-1]
    local_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(local_path):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Error descargando {url}: {e}")
            return url
    return local_path

# --- 1. LEYENDA Y COLORES ---
LAND_COVER_LEGEND = {
    1: "Zonas Urbanas", 2: "Zonas industriales o comerciales", 3: "Zonas degradadas -canteras, escombreras, minas",
    4: "Zonas Verdes artificializadas No Agrícolas", 5: "Cultivos transitorios", 6: "Cultivos permanentes",
    7: "Pastos", 8: "Areas Agrícolas Heterogéneas", 9: "Bosque", 10: "Vegetación Herbácea / Arbustiva",
    11: "Areas abiertas sin o con poca cobertura vegetal", 12: "Humedales", 13: "Agua / Cuerpos de Agua"
}

LAND_COVER_COLORS = {
    1: "#A9A9A9", 2: "#FFFF00", 3: "#FFA500", 4: "#FFD700", 
    5: "#006400", 6: "#32CD32", 7: "#F4A460", 8: "#2E8B57", 
    9: "#228B22", 10: "#9ACD32", 11: "#8B4513", 12: "#00CED1", 13: "#0000FF"
}

# --- 2. FUNCIONES AUXILIARES ---
def get_pixel_area_in_km2(transform, crs, height, width):
    px_w, px_h = abs(transform[0]), abs(transform[4])
    if crs.is_geographic: return (px_w * 110.5) * (px_h * 111.32)
    return (px_w * px_h) / 1_000_000

# --- 3. PROCESAMIENTO ROBUSTO ---
def process_land_cover_raster(raster_path, gdf_mask=None, scale_factor=1):
    safe_path = get_cached_raster(raster_path)
    if not safe_path: return None, None, None, None
    try:
        with rasterio.open(safe_path) as src:
            nodata, crs = src.nodata, src.crs
            if gdf_mask is not None:
                if gdf_mask.crs != src.crs:
                    try: gdf_proj = gdf_mask.to_crs(src.crs)
                    except:
                        gdf_mask.set_crs("EPSG:4326", inplace=True, allow_override=True)
                        gdf_proj = gdf_mask.to_crs(src.crs)
                else: gdf_proj = gdf_mask
                
                try:
                    out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
                    data = out_image[0]
                except ValueError: return None, None, None, None
            else:
                new_height, new_width = int(src.height / scale_factor), int(src.width / scale_factor)
                data = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
                out_transform = src.transform * src.transform.scale((src.width / data.shape[-1]), (src.height / data.shape[-2]))
        return data, out_transform, crs, nodata
    except Exception as e:
        print(f"Error procesando raster: {e}")
        return None, None, None, None

def calculate_land_cover_stats(data, transform, crs, nodata, manual_area_km2=None):
    if data is None: return pd.DataFrame(), 0
    valid_pixels = data[(data != nodata) & (data > 0)]
    if valid_pixels.size == 0: return pd.DataFrame(), 0
    pixel_area_km2 = get_pixel_area_in_km2(transform, crs, data.shape[0], data.shape[1])
    unique, counts = np.unique(valid_pixels, return_counts=True)
    calc_total_area = counts.sum() * pixel_area_km2
    
    final_total_area, factor = calc_total_area, 1.0
    if manual_area_km2 and calc_total_area > 0 and 0.8 < (manual_area_km2 / calc_total_area) < 1.2:
        final_total_area, factor = manual_area_km2, manual_area_km2 / calc_total_area
    
    rows = [{"ID": v, "Cobertura": LAND_COVER_LEGEND.get(v, f"Clase {v}"), "Área (km²)": (c * pixel_area_km2) * factor, "%": ((c * pixel_area_km2 * factor) / final_total_area) * 100, "Color": LAND_COVER_COLORS.get(v, "#808080")} for v, c in zip(unique, counts)]
    return pd.DataFrame(rows).sort_values("%", ascending=False), final_total_area

def get_raster_img_b64(data, nodata):
    if data is None: return ""
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    for val, hex_c in LAND_COVER_COLORS.items():
        if isinstance(hex_c, str):
            hex_c = hex_c.lstrip('#')
            r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            mask_val = (data == val)
            rgba[mask_val, 0], rgba[mask_val, 1], rgba[mask_val, 2], rgba[mask_val, 3] = r, g, b, 200 
    rgba[(data == 0) | (data == nodata), 3] = 0
    image_data = io.BytesIO()
    plt.imsave(image_data, rgba, format='png')
    image_data.seek(0)
    return f"data:image/png;base64,{base64.b64encode(image_data.read()).decode('utf-8')}"

def vectorize_raster_optimized(data, transform, crs, nodata, max_shapes=3000):
    if data is None: return gpd.GeoDataFrame()
    mask_arr = (data != nodata) & (data != 0)
    geoms, values, count = [], [], 0
    for geom, val in shapes(data, mask=mask_arr, transform=transform):
        if count > max_shapes: break 
        s_geom_simp = shape(geom).simplify(tolerance=10, preserve_topology=True) 
        if not s_geom_simp.is_empty:
            geoms.append(s_geom_simp); values.append(val); count += 1
            
    if not geoms: return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame({'ID': values}, geometry=geoms, crs=crs)
    gdf['Cobertura'] = gdf['ID'].map(lambda x: LAND_COVER_LEGEND.get(int(x), f"Clase {int(x)}"))
    gdf['Color'] = gdf['ID'].map(lambda x: LAND_COVER_COLORS.get(int(x), "#808080"))
    
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        try: gdf = gdf.to_crs("EPSG:4326")
        except: pass 
    return gdf

def generate_legend_html():
    html = '<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background-color: white; padding: 10px; border: 2px solid #ccc; border-radius: 5px; font-size: 11px; max-height: 250px; overflow-y: auto;"><b>Leyenda</b><br>'
    for id_cov, name in sorted(LAND_COVER_LEGEND.items()):
        html += f'<div style="display:flex; align-items:center; margin-bottom:2px;"><span style="background:{LAND_COVER_COLORS.get(id_cov, "#808080")}; width:12px; height:12px; display:inline-block; margin-right:5px; border:1px solid #333;"></span>{name}</div>'
    return html + "</div>"

def get_tiff_bytes(data, transform, crs, nodata):
    if data is None: return None
    mem_file = io.BytesIO()
    with rasterio.open(mem_file, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data, 1)
    mem_file.seek(0)
    return mem_file

def calculate_weighted_cn(df_stats, cn_config):
    if df_stats.empty: return 0
    cn_pond, total_pct = 0, 0
    for _, row in df_stats.iterrows():
        cob, pct, val = str(row["Cobertura"]), row["%"], 85 
        if "Bosque" in cob: val = cn_config['bosque']
        elif "Pasto" in cob or "Herbácea" in cob: val = cn_config['pasto']
        elif "Urban" in cob: val = cn_config['urbano']
        elif "Agua" in cob: val = 100
        elif "Suelo" in cob or "Degradada" in cob: val = cn_config['suelo']
        elif "Cultivo" in cob: val = cn_config['cultivo']
        cn_pond += val * pct / 100
        total_pct += pct
    return (cn_pond / total_pct) * 100 if total_pct > 0 else 0

def calculate_scs_runoff(cn, ppt_mm):
    if cn >= 100: return ppt_mm
    if cn <= 0: return 0
    s = (25400 / cn) - 254; ia = 0.2 * s
    return ((ppt_mm - ia) ** 2) / (ppt_mm - ia + s) if ppt_mm > ia else 0

def get_land_cover_at_point(lat, lon, raster_path):
    safe_path = get_cached_raster(raster_path)
    if not safe_path: return "Raster no encontrado"
    try:
        with rasterio.open(safe_path) as src:
            if src.crs.to_string() != "EPSG:4326":
                from pyproj import Transformer
                x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
                coords = [(x, y)]
            else: coords = [(lon, lat)]
            try: return LAND_COVER_LEGEND.get(int(next(src.sample(coords))[0]), "Desconocido")
            except StopIteration: return "Fuera de rango"
    except Exception as e: return f"Error: {str(e)}"

def calculate_cover_stats(gdf_geom, raster_path):
    safe_path = get_cached_raster(raster_path)
    if gdf_geom is None or gdf_geom.empty or not safe_path: return {}
    try:
        with rasterio.open(safe_path) as src:
            if gdf_geom.crs != src.crs: gdf_geom = gdf_geom.to_crs(src.crs)
            out_image, _ = mask(src, gdf_geom.geometry, crop=True, nodata=src.nodata)
            valid_pixels = out_image[0][out_image[0] != src.nodata]
            if valid_pixels.size == 0: return {}
            unique, counts = np.unique(valid_pixels, return_counts=True)
            return {LAND_COVER_LEGEND.get(int(v), f"Clase {int(v)}"): (c / valid_pixels.size) * 100 for v, c in zip(unique, counts)}
    except Exception as e:
        print(f"Error stats: {e}")
        return {}

def get_infiltration_suggestion(stats):
    if not stats: return 0.30, "Sin información (Default)"
    coef_map = {"Bosque": 0.50, "Vegetación": 0.40, "Cultivos": 0.30, "Pastos": 0.25, "Urbano": 0.05, "Agua": 1.00, "Suelo": 0.15}
    weighted_sum, total_pct = 0, 0
    for cover_name, pct in stats.items():
        coef = next((v for k, v in coef_map.items() if k.lower() in cover_name.lower()), 0.20)
        weighted_sum += coef * pct; total_pct += pct
    top_cover = max(stats, key=stats.get)
    return (weighted_sum / total_pct if total_pct > 0 else 0.30), f"Predomina {top_cover} ({stats[top_cover]:.1f}%)"

def calcular_estadisticas_zona(gdf_zona, raster_path):
    return calculate_cover_stats(gdf_zona, raster_path)

def agrupar_coberturas_turc(stats_dict):
    if not stats_dict: return 40.0, 20.0, 30.0, 5.0, 5.0 
    s_bosque = sum(pct for n, pct in stats_dict.items() if "Bosque" in n)
    s_agri = sum(pct for n, pct in stats_dict.items() if any(g in n for g in ["Cultivos", "Agrícolas"]))
    s_pec = sum(pct for n, pct in stats_dict.items() if any(g in n for g in ["Pastos", "Herbácea", "abiertas"]))
    s_agua = sum(pct for n, pct in stats_dict.items() if any(g in n for g in ["Humedales", "Agua"]))
    return s_bosque, s_agri, s_pec, s_agua, max(0, 100 - (s_bosque + s_agri + s_pec + s_agua))

def obtener_imagen_folium_coberturas(gdf_zona, raster_path):
    safe_path = get_cached_raster(raster_path)
    if gdf_zona is None or not safe_path: return None, None
    try:
        with rasterio.open(safe_path) as src:
            if gdf_zona.crs != src.crs: gdf_zona = gdf_zona.to_crs(src.crs)
            out_image, out_transform = mask(src, gdf_zona.geometry, crop=True)
            data = out_image[0]
            rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
            for cat_id, hex_color in LAND_COVER_COLORS.items():
                r, g, b = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                mask_cat = (data == cat_id)
                rgba[mask_cat, 0], rgba[mask_cat, 1], rgba[mask_cat, 2], rgba[mask_cat, 3] = r, g, b, 180 
            rgba[(data == src.nodata) | (data == 0), 3] = 0
            minx, miny, maxx, maxy = gdf_zona.to_crs("EPSG:4326").total_bounds
            return rgba, [[miny, minx], [maxy, maxx]]
    except: return None, None

def obtener_vector_coberturas_ligero(gdf_zona, raster_path, max_poly=1000):
    safe_path = get_cached_raster(raster_path)
    if gdf_zona is None or not safe_path: return None
    try:
        with rasterio.open(safe_path) as src:
            if gdf_zona.crs != src.crs: gdf_zona = gdf_zona.to_crs(src.crs)
            out_image, out_transform = mask(src, gdf_zona.geometry, crop=True)
            data = out_image[0]
            geoms = list({'properties': {'val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(data, mask=(data != src.nodata) & (data > 0), transform=out_transform)) if i < max_poly)
            if not geoms: return None
            gdf_vector = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
            gdf_vector['Cobertura'] = gdf_vector['val'].map(lambda x: LAND_COVER_LEGEND.get(int(x), f"Clase {int(x)}"))
            return gdf_vector.to_crs("EPSG:4326")
    except: return None
