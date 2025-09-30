import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

# se usa OSMnx para geocodificar la ciudad de Cali y obtener sus límites 
# administrativos como un GeoDataFram

cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)  # reproyectar a CRS en metros

# Crear grilla de 250 m
xmin, ymin, xmax, ymax = cali.total_bounds

print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax }, ymax: {ymax}")
cell_size = 250
cols = np.arange(xmin, xmax, cell_size)
rows = np.arange(ymin, ymax, cell_size)

# crea celdas usando las coordenadas de esquinas
polygons = [box(x, y, x+cell_size, y+cell_size) for x in cols for y in rows]
print(f"Cantidad de celdas creadas: {len(polygons)}")
# crea un GeoDataFrame a partir de las celdas
grid = gpd.GeoDataFrame(geometry=polygons, crs=cali.crs)

def limpiar_coordenadas_miles(df, lat_col='lat', lon_col='lon'):
    
    def limpiar_numero(numero):
        numero_str = str(numero).strip()
        # Si tiene coma, es separador decimal
        numero_str = numero_str.replace(',', '')
        if numero_str.startswith("-7"):
            numero_str= numero_str[0:3] + '.' + numero_str[3:]
        else:
            numero_str = numero_str[0:1] + '.' + numero_str[1:]
        try:
            return numero_str
        except ValueError:
            return np.nan

    # Aplicar limpieza
    df[lat_col] = df[lat_col].apply(limpiar_numero)
    df[lon_col] = df[lon_col].apply(limpiar_numero)
    return df

df = pd.read_csv(r"C:\Users\Usuario\Documents\Proyecto Investigacion IoT Seguridad Ciudadana\violencia-db\SEMANA.csv")
if 'x' in df.columns and 'y' in df.columns:
    lat_col, lon_col = 'y', 'x'  # y=latitud, x=longitud

df = limpiar_coordenadas_miles(df, lat_col, lon_col)
df = df.dropna(subset=[lat_col,lon_col])


nx, ny = cols, rows
# esos de cada tipo de violencia
pesos = {"DELINCUENCIA": 5, "FEMENICIDIO": 3, "CONVIVENCIA": 1}  # homicidio > violencia de género > hurto

# Función para agrupar tipos de violencia
def agrupar_tipo_violencia(valor):
    valor = str(valor).upper()
    if "DELINCUENCIA" in valor:
        return "DELINCUENCIA"
    elif "FEMENICIDIO" in valor or "FEMENICIDIO" in valor:
        return "FEMENICIDIO"
    elif "CONVIVENCIA" in valor:
        return "CONVIVENCIA"
    else:
        return "OTRO"

# Aplica la función al DataFrame
df["tipo_violencia_agrupado"] = df["tipo_violencia"].apply(agrupar_tipo_violencia)
df["peso"] = df["tipo_violencia_agrupado"].map(pesos).fillna(0)
homicidios = df[df["tipo_violencia_agrupado"] == "DELINCUENCIA"]
vgenero = df[df["tipo_violencia_agrupado"] == "FEMENICIDIO"]
hurtos = df[df["tipo_violencia_agrupado"] == "CONVIVENCIA"]

casos_feminicidio = df[df["feminicidios"] == "S"]

print(f"Total de casos de feminicidio: {len(casos_feminicidio)}")


# crea geometrías puntuales a partir de las coordenadas x/y
geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]

gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
# reproyecta los puntos al mismo sistema de la grilla
gdf_casos = gdf_casos.to_crs(grid.crs)

# realiza una unión espacial entre los casos y la grilla

casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
# cuenta cuántos casos hay en cada celda
conteo = casos_con_celda.groupby("index_right").size()
# mapea el conteo a cada celda de la grilla y rellena con 0 las celdas que no tienen casos
grid["casos"] = grid.index.map(conteo).fillna(0)

# realiza una intersección geométrica entre la grilla y los límites de Cali
grid_cali = gpd.overlay(grid, cali, how="intersection")

# Selecciona los casos que NO están dentro de los límites de Cali
fuera_cali = gdf_casos[~gdf_casos.within(cali.geometry.iloc[0])]
print(fuera_cali[[lat_col, lon_col]].head(10))
print(f"Casos fuera del mapa: {len(fuera_cali)}")
dentro_cali = gdf_casos[gdf_casos.within(cali.geometry.iloc[0])]

ax = cali.plot(figsize=(11,11), color="white", edgecolor="black")
dentro_cali.plot(ax=ax, color="blue", markersize=2)  # puntos de casos
#grid_cali.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.2)
plt.title("Mapa de calor de violencia en Cali (50m x 50m)")
plt.show()



# Puedes usar len(homicidios), len(vgenero), len(hurtos) para estadísticas
print(f"Casos: {len(homicidios)} H, {len(vgenero)} VG, {len(hurtos)} R")
conteo_peso = casos_con_celda.groupby("index_right")["peso"].sum()
grid["peso_total"] = grid.index.map(conteo_peso).fillna(0)

casos_array = grid_cali["casos"].values

matriz_clasificada = np.zeros_like(casos_array)

# calcular los 5 quintiles (20%, 40%, 60%, 80%, 100%)
q1 = np.percentile(casos_array[casos_array > 0], 20)  # Quintil 1
q2 = np.percentile(casos_array[casos_array > 0], 40)  # Quintil 2
q3 = np.percentile(casos_array[casos_array > 0], 60)  # Quintil 3
q4 = np.percentile(casos_array[casos_array > 0], 80)  # Quintil 4
q5 = np.percentile(casos_array[casos_array > 0], 100) # Quintil 5 (máximo)

matriz_clasificada[casos_array == 0] = 0  # Sin datos
matriz_clasificada[(casos_array > 0) & (casos_array <= q1)] = 1  # Q1
matriz_clasificada[(casos_array > q1) & (casos_array <= q2)] = 2  # Q2
matriz_clasificada[(casos_array > q2) & (casos_array <= q3)] = 3  # Q3
matriz_clasificada[(casos_array > q3) & (casos_array <= q4)] = 4  # Q4
matriz_clasificada[casos_array > q4] = 5  # Q5

# Si quieres agregar la clasificación al GeoDataFrame:
grid_cali["quintil"] = matriz_clasificada
print(grid_cali["quintil"].value_counts())
# Contar celdas por quintil
for i in range(1, 6):
    count = np.sum(matriz_clasificada == i)
    pct = (count / len(casos_array)) * 100 if len(casos_array) > 0 else 0
    print(f"Quintil {i}: {count:,} celdas ({pct:.1f}%)")

# Graficar el mapa base de Cali
ax = cali.plot(figsize=(11, 10), color="white", edgecolor="black")
grid_cali.plot(ax=ax, column="quintil", cmap="inferno", legend=True, alpha=0.7, edgecolor=None)
#dentro_cali.plot(ax=ax, color="blue", markersize=5)
plt.title("Mapa de Calor de Seguridad - Clasificación por Quintiles\n(Colores ascendentes: Oscuro=Seguro, Claro=Peligroso)", fontsize=14, pad=20)
plt.xlabel("Coordenada X (metros)", fontsize=12)
plt.ylabel("Coordenada Y (metros)", fontsize=12)
plt.tight_layout()
plt.show()

