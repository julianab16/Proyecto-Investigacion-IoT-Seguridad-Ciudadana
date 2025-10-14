import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

# se usa OSMnx para geocodificar la ciudad de Cali y obtener sus límites 
# administrativos como un GeoDataFram

cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=None)
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

df = pd.read_csv("violencia-db/SEMANA.csv")


if 'x' in df.columns and 'y' in df.columns:
    lat_col, lon_col = 'y', 'x'  # y=latitud, x=longitud
    print(f"Usando columnas: {lat_col} (latitud), {lon_col} (longitud)")
else:
    print("Error: No se encontraron columnas 'x' e 'y'")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    exit()

df = limpiar_coordenadas_miles(df, lat_col, lon_col)
df = df.dropna(subset=[lat_col,lon_col])

# ========== NUEVO SISTEMA DE ÍNDICE DE SEVERIDAD ==========
nx, ny = cols, rows

# 1. Pesos de severidad (W) por tipo de delito
pesos_severidad = {
    "Nivel 1": 1,  # Bajo
    "Nivel 2": 2,  # Moderado  
    "Nivel 3": 4,  # Alto
    "Nivel 4": 6   # Crítico
}

# 2. Función para agrupar tipos de violencia
def agrupar_tipo_violencia(valor):
    valor = str(valor).upper()
    if "NIVEL 4" in valor:
        return "Nivel 4"
    elif "NIVEL 3" in valor:
        return "Nivel 3"
    elif "NIVEL 2" in valor:
        return "Nivel 2"
    elif "NIVEL 1" in valor:
        return "Nivel 1"
    else:
        return "Nivel 1"  # Por defecto

# 3. Aplicar transformaciones al DataFrame
df["tipo_violencia_agrupado"] = df["nivel_violencia"].apply(agrupar_tipo_violencia)
df["peso_severidad"] = df["tipo_violencia_agrupado"].map(pesos_severidad).fillna(1)

# 4. Calcular puntaje por caso individual (sin frecuencia aún)
df["puntaje_individual"] = df["peso_severidad"]
print(f"Tipos de violencia encontrados: {df['tipo_violencia_agrupado'].unique()}")

# Crear variables por nivel para estadísticas
nivel4 = df[df["tipo_violencia_agrupado"] == "Nivel 4"]
nivel3 = df[df["tipo_violencia_agrupado"] == "Nivel 3"]
nivel2 = df[df["tipo_violencia_agrupado"] == "Nivel 2"]
nivel1 = df[df["tipo_violencia_agrupado"] == "Nivel 1"]

print(f"Distribución de casos por nivel:")
print(f"- Nivel 4 (Crítico): {len(nivel4)} casos")
print(f"- Nivel 3 (Alto): {len(nivel3)} casos")
print(f"- Nivel 2 (Moderado): {len(nivel2)} casos")
print(f"- Nivel 1 (Bajo): {len(nivel1)} casos")

# crea geometrías puntuales a partir de las coordenadas x/y
geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]

gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
# reproyecta los puntos al mismo sistema de la grilla
gdf_casos = gdf_casos.to_crs(grid.crs)

# realiza una unión espacial entre los casos y la grilla
casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")

# ========== CÁLCULO DE ÍNDICE POR CELDA ==========

# 1. Calcular frecuencia por celda (F) - cuántos casos hay en cada celda
frecuencia_por_celda = casos_con_celda.groupby("index_right").size()

# 2. Aplicar mapeo de frecuencia a valores numéricos
def mapear_frecuencia_celda(frecuencia):
    """Mapea la frecuencia de casos por celda a valores numéricos"""
    if frecuencia == 0:
        return 0
    elif frecuencia == 1:
        return 1
    elif 2 <= frecuencia <= 5:
        return 3.5
    elif 6 <= frecuencia <= 12:
        return 9
    else:  # > 12
        return 15


# 3. Calcular puntaje bruto total por celda con peso promedio
puntaje_bruto_celda = casos_con_celda.groupby("index_right")["puntaje_individual"].sum()
peso_promedio_celda = casos_con_celda.groupby("index_right")["peso_severidad"].mean()

# 4. Aplicar el mapeo de frecuencia y calcular índice final por celda
indice_final_celda = pd.Series(dtype=float)

for celda_id in frecuencia_por_celda.index:
    frecuencia = frecuencia_por_celda[celda_id]
    frecuencia_mapeada = mapear_frecuencia_celda(frecuencia)
    peso_promedio = peso_promedio_celda[celda_id]
    
    # Índice = Frecuencia_mapeada × Peso_promedio
    indice_final_celda[celda_id] = frecuencia_mapeada * peso_promedio

# 5. Calcular el puntaje máximo posible para normalización
# Max_posible = 15 (máxima frecuencia) × 6 (máximo peso)
max_posible = 15 * puntaje_bruto_celda

# 6. Calcular índice de severidad normalizado (0-100) por celda
indice_severidad_celda = (indice_final_celda / max_posible) * 100

# Mapear a la grilla
grid["frecuencia"] = grid.index.map(frecuencia_por_celda).fillna(0)
grid["frecuencia_mapeada"] = grid["frecuencia"].apply(mapear_frecuencia_celda)
grid["peso_promedio"] = grid.index.map(peso_promedio_celda).fillna(0)
grid["puntaje_bruto"] = grid.index.map(puntaje_bruto_celda).fillna(0)
grid["indice_severidad"] = grid.index.map(indice_severidad_celda).fillna(0)
grid["casos"] = grid["frecuencia"]  # Es lo mismo

print(f"\nPuntaje máximo posible por celda: {max_posible}")
print(f"Estadísticas de frecuencia por celda:")
print(f"- Celdas con 1 caso: {len(grid[grid['frecuencia'] == 1])}")
print(f"- Celdas con 2-5 casos: {len(grid[(grid['frecuencia'] >= 2) & (grid['frecuencia'] <= 5)])}")
print(f"- Celdas con 6-12 casos: {len(grid[(grid['frecuencia'] >= 6) & (grid['frecuencia'] <= 12)])}")
print(f"- Celdas con >12 casos: {len(grid[grid['frecuencia'] > 12])}")

# realiza una intersección geométrica entre la grilla y los límites de Cali
grid_cali = gpd.overlay(grid, cali, how="intersection")

# Filtrar casos dentro y fuera de Cali
fuera_cali = gdf_casos[~gdf_casos.within(cali.geometry.iloc[0])]
print(f"Casos fuera del mapa: {len(fuera_cali)}")
dentro_cali = gdf_casos[gdf_casos.within(cali.geometry.iloc[0])]

# ========== CLASIFICACIÓN POR QUINTILES DEL ÍNDICE DE SEVERIDAD ==========

# Usar el índice de severidad en lugar del conteo simple
indices_array = grid_cali["indice_severidad"].values

matriz_clasificada = np.zeros_like(indices_array)

# calcular los 5 quintiles del índice de severidad
if np.any(indices_array > 0):
    q1 = np.percentile(indices_array[indices_array > 0], 20)  # Quintil 1
    q2 = np.percentile(indices_array[indices_array > 0], 40)  # Quintil 2
    q3 = np.percentile(indices_array[indices_array > 0], 60)  # Quintil 3
    q4 = np.percentile(indices_array[indices_array > 0], 80)  # Quintil 4
    q5 = np.percentile(indices_array[indices_array > 0], 100) # Quintil 5 (máximo)

    matriz_clasificada[indices_array == 0] = 0  # Sin datos
    matriz_clasificada[(indices_array > 0) & (indices_array <= q1)] = 1  # Q1
    matriz_clasificada[(indices_array > q1) & (indices_array <= q2)] = 2  # Q2
    matriz_clasificada[(indices_array > q2) & (indices_array <= q3)] = 3  # Q3
    matriz_clasificada[(indices_array > q3) & (indices_array <= q4)] = 4  # Q4
    matriz_clasificada[indices_array > q4] = 5  # Q5

    print(f"\nQuintiles del Índice de Severidad:")
    print(f"Q1 (0-20%): 0-{q1:.2f}")
    print(f"Q2 (20-40%): {q1:.2f}-{q2:.2f}")
    print(f"Q3 (40-60%): {q2:.2f}-{q3:.2f}")
    print(f"Q4 (60-80%): {q3:.2f}-{q4:.2f}")
    print(f"Q5 (80-100%): {q4:.2f}-{q5:.2f}")

# Agregar clasificación al GeoDataFrame
grid_cali["quintil"] = matriz_clasificada

# Estadísticas de quintiles
print(f"\nDistribución de celdas por quintil:")
for i in range(1, 6):
    count = np.sum(matriz_clasificada == i)
    pct = (count / len(indices_array)) * 100 if len(indices_array) > 0 else 0
    print(f"Quintil {i}: {count:,} celdas ({pct:.1f}%)")

# Graficar el mapa base de Cali
fig, ax = plt.subplots(figsize=(12, 10))

# Graficar el mapa base de Cali
cali.plot(ax=ax, color="white", edgecolor="black")

# Graficar la grilla con cmap="inferno"
im = grid_cali.plot(ax=ax, column="quintil", cmap="inferno", 
                    alpha=0.7, edgecolor=None, vmin=0, vmax=5)

# Crear colorbar usando ScalarMappable para inferno
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=4))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Nivel de Riesgo", shrink=0.8)
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.set_ticklabels(['Sin datos', 'Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4'])

#dentro_cali.plot(ax=ax, color="blue", markersize=5)
plt.title("Mapa de Calor de Seguridad - Clasificación por Quintiles\n(Colores ascendentes: Oscuro=Seguro, Claro=Peligroso)", fontsize=14, pad=20)
plt.xlabel("Coordenada X (metros)", fontsize=12)
plt.ylabel("Coordenada Y (metros)", fontsize=12)
plt.tight_layout()
plt.show()



# ...existing code...

# Reemplaza la sección de descarga de comunas (después del primer plt.show()) con esto:

# Obtener las comunas de Cali usando OSMnx
print("Descargando comunas de Cali...")
try:
    # Obtener el área de Cali
    area_cali = cali.geometry.iloc[0]
    
    # Opción 1: Usar la función correcta según la versión de OSMnx
    try:
        # Para versiones más recientes de OSMnx
        comunas = ox.features_from_polygon(area_cali, tags={'admin_level': '9'})
    except AttributeError:
        # Para versiones más antiguas
        comunas = ox.geometries_from_polygon(area_cali, tags={'admin_level': '9'})
    
    # Si no funciona el nivel 9, intentar con nivel 8
    if len(comunas) == 0:
        try:
            comunas = ox.features_from_polygon(area_cali, tags={'admin_level': '8'})
        except AttributeError:
            comunas = ox.geometries_from_polygon(area_cali, tags={'admin_level': '8'})
    
    # Si aún no hay resultados, intentar con boundary
    if len(comunas) == 0:
        try:
            comunas = ox.features_from_polygon(area_cali, tags={'boundary': 'administrative'})
        except AttributeError:
            comunas = ox.geometries_from_polygon(area_cali, tags={'boundary': 'administrative'})
    
    # Filtrar solo polígonos
    if len(comunas) > 0:
        comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # Reproyectar al mismo CRS
        comunas = comunas.to_crs(cali.crs)
        
        print(f"Comunas descargadas: {len(comunas)}")
        if len(comunas) > 0:
            print("Columnas disponibles:", comunas.columns.tolist())
    else:
        print("No se encontraron comunas")
        comunas = None
    
except Exception as e:
    print(f"Error descargando comunas: {e}")
    print(f"Versión de OSMnx: {ox.__version__}")
    comunas = None

# Alternativa: Usar geometrías de lugar si las anteriores fallan
if comunas is None or len(comunas) == 0:
    print("Intentando método alternativo...")
    try:
        # Buscar directamente por lugar
        comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                        tags={'admin_level': ['8', '9', '10']})
        
        if len(comunas) > 0:
            comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            comunas = comunas.to_crs(cali.crs)
            print(f"Comunas encontradas con método alternativo: {len(comunas)}")
        else:
            comunas = None
            
    except Exception as e:
        print(f"Error con método alternativo: {e}")
        comunas = None

# ========== VISUALIZACIÓN CON COMUNAS ==========
fig, ax = plt.subplots(figsize=(14, 12))

# Graficar el mapa base de Cali
cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2)

# Graficar las comunas si se descargaron
if comunas is not None and len(comunas) > 0:
    comunas.plot(ax=ax, color="none", edgecolor="gray", linewidth=1, alpha=0.8)
    print("✅ Comunas agregadas al mapa")
else:
    print("⚠️ No se pudieron cargar las comunas, mostrando solo el mapa base")

# Graficar la grilla con cmap="inferno"
im = grid_cali.plot(ax=ax, column="quintil", cmap="inferno", 
                    alpha=0.6, edgecolor=None, vmin=0, vmax=5)

# Crear colorbar
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=4))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Nivel de Riesgo", shrink=0.8)
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.set_ticklabels(['Sin datos', 'Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4'])

titulo = "Mapa de Calor de Seguridad - Cali"
if comunas is not None and len(comunas) > 0:
    titulo += "\n(Líneas grises = Límites administrativos)"

plt.title(titulo, fontsize=14, pad=20)
plt.xlabel("Coordenada X (metros)", fontsize=12)
plt.ylabel("Coordenada Y (metros)", fontsize=12)
plt.tight_layout()
plt.show()

# Mostrar versión de OSMnx para debug
print(f"\nVersión de OSMnx: {ox.__version__}")

