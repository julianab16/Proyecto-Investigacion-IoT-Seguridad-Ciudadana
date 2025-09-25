import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

# Obtener límites de Cali y reproyectar
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)  # reproyectar a CRS en metros

# Crear grilla de 250 m
xmin, ymin, xmax, ymax = cali.total_bounds
cell_size = 250
cols = np.arange(xmin, xmax, cell_size)
rows = np.arange(ymin, ymax, cell_size)
print(f"Límites X: {xmin:.2f} - {xmax:.2f}")
print(f"Límites Y: {ymin:.2f} - {ymax:.2f}")

polygons = [box(x, y, x+cell_size, y+cell_size) for x in cols for y in rows]
grid = gpd.GeoDataFrame(geometry=polygons, crs=cali.crs)

# Cargar casos de violencia
df = pd.read_csv("casos_violencia.csv")
geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf_casos = gdf_casos.to_crs(grid.crs)

# Cargar estaciones de policía (CORREGIDO)
df_estaciones = pd.read_csv("estaciones.csv")  # Sin delimiter=";" 
print("Columnas en estaciones.csv:", df_estaciones.columns.tolist())
print("Primeras filas:")
print(df_estaciones.head())

# Crear geometría usando las columnas correctas (x=latitud, y=longitud según tu archivo)
# Nota: En tu archivo x parece ser latitud y y longitud, así que usamos (y,x) para (lon,lat)
geometry_est = [Point(xy) for xy in zip(df_estaciones["y"], df_estaciones["x"])]
gdf_estaciones = gpd.GeoDataFrame(df_estaciones, geometry=geometry_est, crs="EPSG:4326")
gdf_estaciones = gdf_estaciones.to_crs(grid.crs)

# Verificar que las estaciones estén en el área correcta
print(f"Número de estaciones: {len(gdf_estaciones)}")
print("Bounds de estaciones:", gdf_estaciones.total_bounds)

# Asignar cada caso a una celda
casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
conteo = casos_con_celda.groupby("index_right").size()
grid["casos"] = grid.index.map(conteo).fillna(0)

# Intersección de grilla con Cali
grid_cali = gpd.overlay(grid, cali, how="intersection")

# --- MAPA 1: Mapa de calor con casos y estaciones ---
fig, ax = plt.subplots(figsize=(12, 10))
cali.plot(ax=ax, color="white", edgecolor="black", linewidth=1)
grid_cali.plot(ax=ax, column="casos", cmap="Reds", alpha=0.7, edgecolor=None, legend=True)
gdf_casos.plot(ax=ax, color="blue", markersize=3, alpha=0.6, label="Casos violencia")
gdf_estaciones.plot(ax=ax, color="green", markersize=100, marker="^", 
                   label="Estaciones Policía", edgecolor="black", linewidth=1)
ax.set_title("Mapa de calor de violencia en Cali con estaciones de policía", fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- MAPA 2: Solo grilla sobre Cali ---
fig, ax = plt.subplots(figsize=(10, 10))
cali.plot(ax=ax, color="lightgray", edgecolor="black")
grid_cali.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.3)
gdf_estaciones.plot(ax=ax, color="green", markersize=80, marker="^", 
                   label="Estaciones Policía", edgecolor="black")
ax.set_title("Grilla 250x250m sobre Cali con estaciones", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

# --- ESTADÍSTICAS ---
print(f"\nEstadísticas del análisis:")
print(f"Total de casos de violencia: {len(gdf_casos)}")
print(f"Total de estaciones de policía: {len(gdf_estaciones)}")
print(f"Total de celdas en la grilla: {len(grid_cali)}")
print(f"Celdas con al menos un caso: {(grid_cali['casos'] > 0).sum()}")
print(f"Máximo de casos por celda: {grid_cali['casos'].max()}")

# Verificar si las estaciones están dentro de Cali
estaciones_en_cali = gpd.sjoin(gdf_estaciones, cali, how="inner", predicate="within")
print(f"Estaciones dentro de Cali: {len(estaciones_en_cali)} de {len(gdf_estaciones)}")