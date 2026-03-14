import streamlit as st
import os

def render_motor_hidrologico(gdf_zona):
    """Motor de bolsillo para extraer la red hídrica con PySheds."""
    
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        umbral_acc = st.slider(
            "Umbral de Acumulación (Celdas):", 
            min_value=10, max_value=2000, value=100, step=10,
            help="Menor umbral = Más riachuelos. Mayor umbral = Solo ríos principales.",
            key="geom_slider"
        )
        
    with col_btn2:
        st.write("") 
        st.write("") 
        if st.button("🌊 Generar Red Hídrica Aquí", use_container_width=True):
            with st.spinner(f"Encendiendo motor hidrológico (Umbral: {umbral_acc} celdas)..."):
                try:
                    import tempfile
                    import rasterio
                    import numpy as np
                    from rasterio.mask import mask
                    from pysheds.grid import Grid
                    import geopandas as gpd
                    
                    DEM_PATH = os.path.join("data", "DemAntioquia_EPSG3116.tif")
                    if os.path.exists(DEM_PATH) and gdf_zona is not None:
                        geom_valida = gdf_zona.copy()
                        geom_valida['geometry'] = geom_valida.buffer(0)
                        
                        with rasterio.open(DEM_PATH) as src:
                            out_image, out_transform = mask(src, geom_valida.to_crs(src.crs).geometry.values, crop=True)
                            out_meta = src.meta.copy()
                            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform, "count": 1, "nodata": -9999.0})
                            dem_clean = np.where(np.isnan(out_image[0]) | (out_image[0] == src.nodata), -9999.0, out_image[0])
                        
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            with rasterio.open(tmp.name, 'w', **out_meta) as dst: dst.write(dem_clean.astype('float64'), 1)
                            grid = Grid.from_raster(tmp.name)
                            dem_grid = grid.read_raster(tmp.name, nodata=-9999.0)
                            
                            flooded = grid.fill_depressions(dem_grid)
                            resolved = grid.resolve_flats(flooded)
                            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                            fdir = grid.flowdir(resolved, dirmap=dirmap)
                            acc = grid.accumulation(fdir, dirmap=dirmap)
                            
                            branches = grid.extract_river_network(fdir, acc > umbral_acc, dirmap=dirmap)
                            
                            if branches and len(branches["features"]) > 0:
                                r_raw = gpd.GeoDataFrame.from_features(branches["features"], crs=out_meta['crs'])
                                r_clip = gpd.clip(r_raw, gdf_zona.to_crs(out_meta['crs']))
                                if not r_clip.empty:
                                    r_clip_m = r_clip.to_crs(epsg=3116)
                                    r_clip['longitud_km'] = r_clip_m.length / 1000.0
                                    r_clip['Orden_Strahler'] = np.where(
                                        r_clip['longitud_km'] > 2.0, 4, 
                                        np.where(r_clip['longitud_km'] > 1.0, 3, 
                                        np.where(r_clip['longitud_km'] > 0.4, 2, 1))
                                    )
                                    # INYECCIÓN DIRECTA AL ALEPH
                                    st.session_state['gdf_rios'] = r_clip
                                    st.success("✅ Red hídrica calculada exitosamente.")
                                    st.rerun() 
                    else:
                        st.error("❌ No se encontró el archivo base en data/")
                except Exception as e:
                    st.error(f"Error calculando hidrología: {e}")
