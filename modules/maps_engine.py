# modules/maps_engine.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca.colormap as cm
from scipy.ndimage import gaussian_filter

import folium
from folium import plugins
from folium.plugins import Fullscreen, MeasureControl, MousePosition
from folium.features import DivIcon

# ==============================================================================
# 1. GENERADORES DE POPUPS HTML (Plantillas)
# ==============================================================================
def generar_popup_estacion(row, valor_col='ppt_media'):
    nombre = str(row.get('nombre', 'Estación')).replace("'", "")
    muni = str(row.get('municipio', 'N/A')).replace("'", "")
    altura = float(row.get('altitud', 0))
    valor = float(row.get(valor_col, 0))
    std = float(row.get('ppt_std', 0))
    anios = int(row.get('n_anios', 0))
    
    return f"""
    <div style='font-family:sans-serif; font-size:12px; min-width:160px; line-height:1.4;'>
        <b style='color:#1f77b4; font-size:14px'>{nombre}</b>
        <hr style='margin:4px 0; border-top:1px solid #ddd'>
        📍 <b>Mpio:</b> {muni}<br>
        ⛰️ <b>Altitud:</b> {altura:.0f} msnm<br>
        💧 <b>P. Media:</b> {valor:.0f} mm/año<br>
        📉 <b>Desv. Std:</b> ±{std:.0f} mm<br>
        📅 <b>Registro:</b> {anios} años
    </div>
    """

def generar_popup_bocatoma(row):
    nombre = str(row.get('nombre_acu', 'Bocatoma')).replace("'", "")
    fuente = str(row.get('fuente_aba', 'N/A')).replace("'", "")
    mpio = str(row.get('municipio', '')).strip()
    vereda = str(row.get('veredas', '')).strip()
    ubicacion = f"{mpio} - {vereda}" if vereda else mpio
    tipo = str(row.get('tipo', 'N/A'))
    entidad = str(row.get('entidad_ad', 'N/A'))

    return f"""
    <div style='font-family:sans-serif; font-size:12px; min-width:180px;'>
        <b style='color:#16a085; font-size:14px'>🚰 {nombre}</b>
        <hr style='margin:4px 0; border-top:1px solid #ddd'>
        📍 <b>Ubicación:</b> {ubicacion}<br>
        🌊 <b>Fuente:</b> {fuente}<br>
        ⚙️ <b>Tipo:</b> {tipo}<br>
        🏢 <b>Entidad:</b> {entidad}
    </div>
    """

def generar_popup_predio(row):
    datos_norm = {k.lower(): v for k, v in row.items()}
    def get_seguro(col_key, default='N/A'):
        val = datos_norm.get(col_key.lower(), default)
        if val is None or str(val).lower() in ['none', 'nan', 'null', '']: return default
        return str(val).strip()

    nombre = get_seguro('nombre_pre', 'Predio')
    pk = get_seguro('pk_predios')
    anio = get_seguro('año_acuer', '-')
    mpio = get_seguro('nomb_mpio')
    vereda = get_seguro('nombre_ver')
    ubicacion = f"{mpio} / {vereda}" if (mpio != 'N/A' or vereda != 'N/A') else "N/A"
    embalse = get_seguro('embalse')
    mecanismo = get_seguro('mecanism')
    
    try:
        val_area = float(datos_norm.get('area_ha', 0))
        area_txt = f"{val_area:.2f} ha"
    except: area_txt = "N/A"

    return f"""
    <div style='font-family:sans-serif; font-size:12px; min-width:200px;'>
        <b style='color:#d35400; font-size:14px'>🏡 {nombre}</b>
        <hr style='margin:4px 0; border-top:1px solid #ddd'>
        🔑 <b>PK:</b> {pk}<br>
        📅 <b>Año:</b> {anio}<br>
        📍 <b>Ubicación:</b> {ubicacion}<br>
        💧 <b>Embalse:</b> {embalse}<br>
        📜 <b>Mecanismo:</b> {mecanismo}<br>
        📐 <b>Área:</b> {area_txt}
    </div>
    """

# ==============================================================================
# 2. MOTOR MAESTRO DE MAPAS FOLIUM
# ==============================================================================
def generar_mapa_interactivo(grid_data, bounds, gdf_stations, gdf_zona, gdf_buffer, 
                             gdf_predios=None, gdf_bocatomas=None, gdf_municipios=None,
                             nombre_capa="Variable", cmap_name="Spectral_r", opacidad=0.7):
    """
    Genera el mapa completo con Raster coloreado, Isolíneas limpias y Vectores ricos.
    """
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None, control_scale=True)
    
    # Capas Base
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="🛰️ Satélite", overlay=False).add_to(m)
    folium.TileLayer(tiles="CartoDB positron", name="🗺️ Mapa Claro", overlay=False).add_to(m)

    # 1. RASTER (Imagen de Fondo)
    if grid_data is not None:
        Z = grid_data[0] if isinstance(grid_data, tuple) else grid_data
        Z = Z.astype(float)
        try:
            valid = Z[~np.isnan(Z)]
            vmin, vmax = (np.percentile(valid, 2), np.percentile(valid, 98)) if len(valid) > 0 else (0, 1)
            
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_name)
            rgba_img = cmap(norm(Z))
            rgba_img[..., 3] = np.where(np.isnan(Z), 0, opacidad)
            
            folium.raster_layers.ImageOverlay(
                image=np.flipud(rgba_img),
                bounds=[[miny, minx], [maxy, maxx]], 
                name=f"🎨 {nombre_capa}", opacity=1, mercator_project=True
            ).add_to(m)
            
            # Leyenda
            colors_hex = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, 15)]
            cm.LinearColormap(colors=colors_hex, vmin=vmin, vmax=vmax, caption=nombre_capa).add_to(m)
        except: pass

    # 2. ISOLÍNEAS
    if grid_data is not None:
        fg_iso = folium.FeatureGroup(name="〰️ Isolíneas", overlay=True, show=True)
        try:
            Z_Smooth = gaussian_filter(np.nan_to_num(Z, nan=np.nanmean(Z)), sigma=1.0)
            xi = np.linspace(minx, maxx, Z.shape[1])
            yi = np.linspace(miny, maxy, Z.shape[0])
            grid_x_mesh, grid_y_mesh = np.meshgrid(xi, yi)
            
            fig_iso, ax_iso = plt.subplots()
            contours = ax_iso.contour(grid_x_mesh, grid_y_mesh, Z_Smooth, levels=12)
            plt.close(fig_iso)
            
            for i, level_segs in enumerate(contours.allsegs):
                val = contours.levels[i]
                for segment in level_segs:
                    lat_lon_coords = [[pt[1], pt[0]] for pt in segment]
                    if len(lat_lon_coords) > 10:
                        folium.PolyLine(lat_lon_coords, color='black', weight=0.6, opacity=0.5, tooltip=f"{val:.1f}").add_to(fg_iso)
                        mid_idx = len(lat_lon_coords) // 2
                        mid_point = lat_lon_coords[mid_idx]
                        folium.map.Marker(
                            mid_point,
                            icon=DivIcon(
                                icon_size=(150,36), icon_anchor=(0,0),
                                html=f'<div style="font-size: 9pt; font-weight: bold; color: #333; text-shadow: 1px 1px 0 #fff;">{val:.0f}</div>'
                            )
                        ).add_to(fg_iso)
        except Exception as e: print(f"Iso error: {e}")
        fg_iso.add_to(m)

    # 3. MUNICIPIOS
    if gdf_municipios is not None and not gdf_municipios.empty:
        if 'MPIO_NAREA' in gdf_municipios.columns:
            gdf_municipios['area_ha_fmt'] = (gdf_municipios['MPIO_NAREA'] * 100).apply(lambda x: f"{x:,.1f} ha")
            col_area = 'area_ha_fmt'
        else: col_area = None

        col_name = next((c for c in gdf_municipios.columns if 'MPIO_CNMBR' in c or 'nombre' in c), None)
        fields, aliases = [], []
        if col_name: fields.append(col_name); aliases.append('Municipio:')
        if col_area: fields.append(col_area); aliases.append('Área:')

        folium.GeoJson(
            gdf_municipios, name="🏛️ Municipios",
            style_function=lambda x: {'color': '#7f8c8d', 'weight': 1, 'fill': False, 'dashArray': '4, 4'},
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases) if fields else None
        ).add_to(m)

    # 4. CAPAS ZONA
    if gdf_zona is not None:
        folium.GeoJson(gdf_zona, name="🟦 Cuenca", style_function=lambda x: {'color': 'black', 'weight': 2, 'fill': False}).add_to(m)
    if gdf_buffer is not None:
        folium.GeoJson(gdf_buffer, name="⭕ Buffer", style_function=lambda x: {'color': 'red', 'weight': 1, 'dashArray': '5, 5', 'fill': False}).add_to(m)

    # 5. PREDIOS
    if gdf_predios is not None and not gdf_predios.empty:
        fg_predios = folium.FeatureGroup(name="🏡 Predios", show=True)
        try:
            gdf_viz = gdf_predios.to_crs(epsg=4326) if gdf_predios.crs and gdf_predios.crs.to_string() != "EPSG:4326" else gdf_predios
        except: gdf_viz = gdf_predios
            
        for _, row in gdf_viz.iterrows():
            if row.geometry and not row.geometry.is_empty:
                try: popup_obj = folium.Popup(generar_popup_predio(row), max_width=250)
                except: popup_obj = folium.Popup(str(row.get('nombre_pre', 'Predio')), max_width=200)

                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x: {'color': '#e67e22', 'weight': 1.5, 'fillOpacity': 0.3, 'fillColor': '#f39c12'},
                    popup=popup_obj, tooltip=str(row.get('nombre_pre', 'Predio'))
                ).add_to(fg_predios)
        fg_predios.add_to(m)

    # 6. BOCATOMAS
    if gdf_bocatomas is not None and not gdf_bocatomas.empty:
        fg_bocas = folium.FeatureGroup(name="🚰 Bocatomas", show=True)
        for _, row in gdf_bocatomas.iterrows():
            if row.geometry:
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6, color='white', weight=1, fill=True, fill_color='#16a085', fill_opacity=1,
                    popup=folium.Popup(generar_popup_bocatoma(row), max_width=200),
                    tooltip=str(row.get('nombre_predio', 'Bocatoma'))
                ).add_to(fg_bocas)
        fg_bocas.add_to(m)

    # 7. ESTACIONES
    if gdf_stations is not None and not gdf_stations.empty:
        fg_est = folium.FeatureGroup(name="🌦️ Estaciones")
        for _, row in gdf_stations.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5, color='black', weight=1, fill=True, fill_color='#3498db', fill_opacity=1,
                popup=folium.Popup(generar_popup_estacion(row), max_width=200),
                tooltip=row.get('nombre', 'Estación')
            ).add_to(fg_est)
        fg_est.add_to(m)

    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MousePosition().add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    
    return m
