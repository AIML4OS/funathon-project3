# %%
# Exercise 1 — Read a Sentinel-2 tile and display it
# Packages: rasterio, numpy, matplotlib.pyplot
# Key functions: rasterio.open(), src.read([4, 3, 2]), np.transpose(), np.percentile(), np.clip()

# %%
# Exercise 2 — Explore raster metadata and create a false-colour composite
# Packages: rasterio, numpy, matplotlib.pyplot
# Key functions: src.profile, src.read([8, 4, 3])

# %%
# Exercise 3 — Compute and display NDVI
# Packages: rasterio, numpy, matplotlib.pyplot
# Key functions: src.read(8), np.where(), ax.imshow(cmap="RdYlGn"), fig.colorbar()

# %%
# Exercise 4 — Geocode a city and build a tile URL
# Packages: requests, geopandas, shapely.geometry
# Key functions: requests.get() (Nominatim API), Point(), gpd.GeoDataFrame(), .to_crs(), gpd.sjoin()

# %%
# Exercise 5 — Find and display the satellite tile for your city
# Packages: pandas, rasterio, numpy, matplotlib.pyplot
# Key functions: pd.read_parquet(), city_point.geometry.iloc[0].x/.y, rasterio.open()

# %%
# Exercise 6 — Build a GeoDataFrame from tile bounds and convert CRS
# Packages: geopandas, shapely.geometry
# Key functions: box(), gpd.GeoDataFrame(), .to_crs(), .total_bounds

# %%
# Exercise 7 — Spatial join with NUTS3 regions
# Packages: geopandas, shapely.geometry
# Key functions: gpd.read_file(), gpd.sjoin(predicate="intersects"), .area

# %%
# Exercise 8 — Display a Sentinel-2 tile on an interactive folium map
# Packages: folium, rasterio.warp
# Key functions: transform_bounds(), folium.Map(), ImageOverlay()

# %%
# Exercise 9 — Load a CLC+ label from S3
# Packages: urllib.request, io, numpy, matplotlib.pyplot, matplotlib.colors
# Key functions: urllib.request.urlopen(), np.load(io.BytesIO()), ListedColormap(), ax.imshow(vmin=1, vmax=10)

# %%
# Exercise 10 — Overlay the satellite image and CLC+ label on an interactive map
# Packages: rasterio, numpy, urllib.request, io, folium, matplotlib.colors
# Key functions: to_rgba(), color_lut[label] (fancy indexing), transform_bounds(), ImageOverlay(), LayerControl()
