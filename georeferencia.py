import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)  # reproyectar a CRS en metros

# Crear grilla de 50 m
xmin, ymin, xmax, ymax = cali.total_bounds
cell_size = 250
cols = np.arange(xmin, xmax, cell_size)
rows = np.arange(ymin, ymax, cell_size)
print(xmin)
print(xmax)
polygons = [box(x, y, x+cell_size, y+cell_size) for x in cols for y in rows]
grid = gpd.GeoDataFrame(geometry=polygons, crs=cali.crs)

df = pd.read_csv(r"C:\Users\Usuario\Documents\Proyecto Investigacion IoT Seguridad Ciudadana\violencia-db\casos_violencia.csv")
geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf_casos = gdf_casos.to_crs(grid.crs)

# Asignar cada caso a una celda
casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
conteo = casos_con_celda.groupby("index_right").size()
grid["casos"] = grid.index.map(conteo).fillna(0)

ax = cali.plot(figsize=(10,10), color="white", edgecolor="black")
grid.plot(ax=ax, column="casos", cmap="Reds", alpha=0.6, edgecolor=None, legend=True)
gdf_casos.plot(ax=ax, color="blue", markersize=5)  # puntos de casos
plt.title("Mapa de calor de violencia en Cali (50m x 50m)")
plt.show()


grid_cali = gpd.overlay(grid, cali, how="intersection")

# --- 4. Graficar ---
fig, ax = plt.subplots(figsize=(10, 10))
cali.plot(ax=ax, color="white", edgecolor="black")
grid_cali.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.2)
ax.set_title("Grilla 50x50m sobre Cali", fontsize=14)
plt.show()