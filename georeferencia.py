import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import math
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')
#from sclim import mejorcelda
#print("Mejor celda:", mejorcelda)
# ========== CONFIGURACIÓN ==========
LADO_HEX = 123.45   # lado del hexágono en metros

print("=" * 70)
print(" " * 15 + "MAPA DE CALOR DE SEGURIDAD - CALI")
print("=" * 70)

# ========== CARGA DE DATOS ==========
print("\n[1/6] Cargando mapa base de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=None)
cali = cali.to_crs(3116)
print("✓ Mapa base cargado correctamente")

# ========== CREACIÓN DE GRILLA HEXAGONAL ==========
print("\n[2/6] Generando grilla hexagonal...")
ancho_hex = math.sqrt(3) * LADO_HEX
paso_x = ancho_hex
paso_y = 1.5 * LADO_HEX

xmin, ymin, xmax, ymax = cali.total_bounds
xmin -= ancho_hex
ymin -= LADO_HEX
xmax += ancho_hex
ymax += LADO_HEX

centers = []
x_vals = np.arange(xmin, xmax + paso_x, paso_x)
y_vals = np.arange(ymin, ymax + paso_y, paso_y)

for ix, x in enumerate(x_vals):
    y_offset = (paso_y / 2.0) if (ix % 2 == 1) else 0.0
    for y in y_vals:
        centers.append((x, y + y_offset))

def hexagon(center_x, center_y, s):
    """Crear hexágono regular"""
    angles = [i * math.pi / 3.0 for i in range(6)]
    coords = [(center_x + s * math.cos(a), center_y + s * math.sin(a)) for a in angles]
    return Polygon(coords)

try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

polygons = [hexagon(cx, cy, LADO_HEX) for cx, cy in centers if hexagon(cx, cy, LADO_HEX).intersects(area_cali)]
grid = gpd.GeoDataFrame(geometry=polygons, crs=cali.crs)
print(f"✓ {len(polygons):,} hexágonos generados")

# ========== FUNCIONES ==========
def limpiar_coordenadas(df, lat_col='y', lon_col='x'):
    """Limpia y valida coordenadas"""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    lat_col = lat_col.lower()
    lon_col = lon_col.lower()
    
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] >= 2.5) & (df[lat_col] <= 4.5)]
    df = df[(df[lon_col] >= -77.5) & (df[lon_col] <= -75.5)]
    
    return df

def extraer_nivel_severidad(valor):
    """Extrae el nivel de severidad"""
    if pd.isna(valor) or str(valor).strip() == '':
        return None
    
    valor = str(valor).upper().strip()
    
    if "NIVEL 4" in valor or valor == "4":
        return "Nivel 4"
    elif "NIVEL 3" in valor or valor == "3":
        return "Nivel 3"
    elif "NIVEL 2" in valor or valor == "2":
        return "Nivel 2"
    elif "NIVEL 1" in valor or valor == "1":
        return "Nivel 1"
    return None

def procesar_csv(archivo, nombre_categoria=None):
    """
    Procesa un archivo CSV y retorna DataFrame limpio
    
    Parámetros:
    - archivo: ruta del archivo CSV
    - nombre_categoria: nombre para identificar el tipo de delito
    """
    try:
        print(f"\n  Cargando: {archivo}")
        df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')
        
        if len(df) == 0:
            print(f"    ⚠ Archivo vacío")
            return None
        
        print(f"    • Registros iniciales: {len(df):,}")
        
        # Limpiar coordenadas
        df = limpiar_coordenadas(df, 'y', 'x')
        
        if df is None or len(df) == 0:
            print(f"    ⚠ Sin coordenadas válidas")
            return None
        
        # Normalizar columnas
        df.columns = df.columns.str.strip().str.lower()
        
        # Buscar columna de severidad
        col_severidad = None
        posibles_cols = ['nivel_severidad', 'nivel severidad', 'tipo_violencia']
        
        for col in posibles_cols:
            if col in df.columns:
                col_severidad = col
                break
        
        if col_severidad is None:
            print(f"    ✗ No se encontró columna de severidad")
            print(f"       Columnas disponibles: {', '.join(df.columns.tolist()[:10])}")
            return None
        
        df['nivel_severidad'] = df[col_severidad]
        
        # Identificar tipo de delito
        if nombre_categoria:
            df['categoria'] = nombre_categoria
        else:
            # Intentar inferir del nombre del archivo
            nombre_archivo = archivo.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
            df['categoria'] = nombre_archivo
        
        # Buscar columna con tipo específico de delito
        cols_tipo = ['tipo de hurto', 'dinámica', 'dinamica', 'tipo_delito']
        col_tipo_encontrada = None
        
        for col in cols_tipo:
            if col in df.columns:
                col_tipo_encontrada = col
                break
        
        if col_tipo_encontrada:
            df['tipo_delito'] = df[col_tipo_encontrada].fillna(df['categoria'])
        else:
            df['tipo_delito'] = df['categoria']
        
        df['archivo_fuente'] = archivo.split('/')[-1]
        
        # Mostrar información
        niveles = df['nivel_severidad'].dropna().unique()
        print(f"    • Niveles encontrados: {', '.join(map(str, sorted(niveles)[:5]))}")
        
        tipos_top = df['tipo_delito'].value_counts().head(3)
        if len(tipos_top) > 0:
            print(f"    • Tipos principales:")
            for tipo, count in tipos_top.items():
                print(f"      - {str(tipo)[:55]}: {count:,}")
        
        print(f"    ✓ {len(df):,} registros procesados")
        return df
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return None

# ========== CARGA DE ARCHIVOS CSV ==========
print("\n[3/6] Cargando archivos de delitos...")

datasets = []

# OPCIÓN A: Cargar archivos específicos (más control)
archivos_especificos = [
    ('Hurtos_fiscalia.csv', 'Hurto'),
    ('Homicidios_fiscalia.csv', 'Homicidio'),
    ('Delitos_Sexuales_fiscalia.csv', 'Delito Sexual'),
    ('Extorsion_fiscalia.csv', 'Extorsion'),
    ('Lesiones_fiscalia.csv', 'Lesiones'),
    ('Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
]

print(f"\nBuscando archivos CSV...")
archivos_encontrados = 0

for archivo, categoria in archivos_especificos:
    try:
        df = procesar_csv(archivo, categoria)
        if df is not None:
            datasets.append(df)
            archivos_encontrados += 1
    except FileNotFoundError:
        pass  # Archivo no existe, continuar
    except Exception as e:
        print(f"  ⚠ Error procesando {archivo}: {str(e)}")

if archivos_encontrados == 0:
    print("\n" + "="*70)
    print("✗ ERROR: No se encontró ningún archivo CSV válido")
    print("="*70)
    print("\nAsegúrate de tener archivos CSV con las siguientes columnas:")
    print("  - x, y (coordenadas)")
    print("  - nivel_severidad o tipo_violencia")
    print("\nArchivos esperados:")
    print("  • homicidios_2023.csv")
    print("  • hurtos_2023.csv")
    print("  • delitos_sexuales_2023.csv")
    print("  • extorsion_2023.csv")
    print("  • lesiones_personales_2023.csv")
    print("  • violencia_intrafamiliar_2023.csv")
    print("\nO usa tus archivos actuales:")
    print("  • Hurtos_fiscalia.csv")
    print("  • Dinamicas_delictivas.csv")
    print("  • Dinamicasdelicticas.csv")
    exit()

print(f"\n✓ Total de archivos procesados: {archivos_encontrados}")

# Combinar todos los datasets
df = pd.concat(datasets, ignore_index=True)
print(f"✓ Total de registros combinados: {len(df):,}")

# ========== SISTEMA DE SEVERIDAD ==========
print("\n[4/6] Calculando índices de severidad...")

pesos_severidad = {
    "Nivel 1": 1,   # Bajo
    "Nivel 2": 2,   # Moderado  
    "Nivel 3": 4,   # Alto
    "Nivel 4": 6    # Crítico
}

df["nivel_agrupado"] = df["nivel_severidad"].apply(extraer_nivel_severidad)

# Eliminar casos sin nivel válido
casos_sin_nivel = df["nivel_agrupado"].isna().sum()
if casos_sin_nivel > 0:
    print(f"  ⚠ {casos_sin_nivel:,} casos sin nivel válido (excluidos)")

    df = df[df["nivel_agrupado"].notna()].copy()

    cols_show = ['archivo_fuente', 'categoria', 'tipo_delito', 'nivel_severidad', 'x', 'y']
    cols_show = [c for c in cols_show if c in df.columns]

    print("\n  Ejemplos (hasta 10) de casos sin nivel válido:")
    if len(df) > 0:
        print(df[cols_show].head(10).to_string(index=False))
    else:
        print("  (ninguno)")

if len(df) == 0:
    print("\n✗ ERROR: No hay casos con nivel de severidad válido")
    exit()

df["peso_severidad"] = df["nivel_agrupado"].map(pesos_severidad)
df["puntaje_individual"] = df["peso_severidad"]

# Estadísticas
print(f"\n  Distribución por nivel de severidad:")
for nivel in ["Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4"]:
    count = len(df[df["nivel_agrupado"] == nivel])
    pct = (count / len(df)) * 100 if len(df) > 0 else 0
    print(f"    {nivel}: {count:,} ({pct:.1f}%)")

print(f"\n  Distribución por categoría:")
for cat in sorted(df['categoria'].unique()):
    count = len(df[df['categoria'] == cat])
    pct = (count / len(df)) * 100
    print(f"    {cat}: {count:,} ({pct:.1f}%)")

print(f"\n  Top 10 tipos de delito:")
for delito, count in df['tipo_delito'].value_counts().head(10).items():
    pct = (count / len(df)) * 100
    print(f"    {str(delito)[:50]}: {count:,} ({pct:.1f}%)")

# ========== GEOREFERENCIACIÓN ==========
print("\n[5/6] Georeferenciando casos...")
geometry = [Point(lon, lat) for lon, lat in zip(df['x'], df['y'])]
gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf_casos = gdf_casos.to_crs(grid.crs)

dentro = gdf_casos[gdf_casos.within(cali.geometry.iloc[0])]
fuera = len(gdf_casos) - len(dentro)
print(f"  • Casos dentro de Cali: {len(dentro):,}")
if fuera > 0:
    print(f"  • Casos fuera: {fuera:,}")

grid = grid.reset_index(drop=False).rename(columns={'index':'grid_id'})

casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
frecuencia_df = casos_con_celda.groupby("grid_id").size().rename("frecuencia").reset_index()
peso_promedio_df = casos_con_celda.groupby("grid_id")["peso_severidad"].mean().rename("peso_promedio").reset_index()
grid = grid.merge(frecuencia_df, on="grid_id", how="left")
grid = grid.merge(peso_promedio_df, on="grid_id", how="left")
grid["frecuencia"] = grid["frecuencia"].fillna(0).astype(int)
grid["peso_promedio"] = grid["peso_promedio"].fillna(0.0)

print(f"  ✓ Casos asignados a hexágonos")

# ========== CÁLCULO DE ÍNDICE ==========
def mapear_frecuencia(freq):
    if freq == 0: return 0
    elif 1 <= freq <= 3: return 1
    elif 4 <= freq <= 6: return 3.5
    elif 7 <= freq <= 9: return 5.5
    elif 10 <= freq <= 14: return 9
    else: return 15

frecuencia_por_celda = casos_con_celda.groupby("index_right").size()
peso_promedio_celda = casos_con_celda.groupby("index_right")["peso_severidad"].mean()

indice_final_celda = pd.Series(dtype=float)
for celda_id in frecuencia_por_celda.index:
    freq = frecuencia_por_celda[celda_id]
    freq_mapeada = mapear_frecuencia(freq)
    peso_prom = peso_promedio_celda[celda_id]
    indice_final_celda[celda_id] = freq_mapeada * peso_prom

indice_severidad_celda = (indice_final_celda / 90) * 100  # 90 = 15*6 (máximo posible)

grid["indice_final"] = grid["frecuencia"].apply(mapear_frecuencia) * grid["peso_promedio"]
grid["indice_severidad"] = (grid["indice_final"] / (15*6)) * 100
grid["indice_severidad"] = grid["indice_severidad"].fillna(0)

grid_cali = gpd.overlay(grid, cali, how="intersection")
grid_cali["casos"] = grid_cali["frecuencia"] 

# ========== QUINTILES ==========
indices = grid_cali["indice_severidad"].values
clasificacion = np.zeros_like(indices)

if np.any(indices > 0):
    q1 = np.percentile(indices[indices > 0], 20)
    q2 = np.percentile(indices[indices > 0], 40)
    q3 = np.percentile(indices[indices > 0], 60)
    q4 = np.percentile(indices[indices > 0], 80)

    clasificacion[indices == 0] = 0
    clasificacion[(indices > 0) & (indices <= q1)] = 1
    clasificacion[(indices > q1) & (indices <= q2)] = 2
    clasificacion[(indices > q2) & (indices <= q3)] = 3
    clasificacion[(indices > q3) & (indices <= q4)] = 4
    clasificacion[indices > q4] = 5

    print(f"\n  Quintiles:")
    print(f"    Q1: 0 - {q1:.2f}")
    print(f"    Q2: {q1:.2f} - {q2:.2f}")
    print(f"    Q3: {q2:.2f} - {q3:.2f}")
    print(f"    Q4: {q3:.2f} - {q4:.2f}")
    print(f"    Q5: {q4:.2f}+")

grid_cali["quintil"] = clasificacion

# ========== VISUALIZACIÓN ==========
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import osmnx as ox

print("\n[6/6] Generando mapa...")

# Diagnóstico después de: casos_con_celda = gpd.sjoin(...)
print("DEBUG: registros CSV combinados:", len(df))
print("DEBUG: puntos geocodificados:", len(gdf_casos))
print("DEBUG: dentro de Cali:", len(dentro))
print("DEBUG: sjoin total:", len(casos_con_celda))
print("DEBUG: sjoin index_right nulos:", casos_con_celda['index_right'].isna().sum())

# Ver ejemplos de filas sin index_right (no asignadas a ninguna celda)
print("\nEjemplos sin index_right (5):")
print(casos_con_celda[casos_con_celda['index_right'].isna()].head(5)[['archivo_fuente','x','y','nivel_severidad']].to_string(index=False))

# Mostrar resumen de frecuencia y suma vs casos asignados
frecuencia_por_celda = casos_con_celda.groupby("index_right").size()
print("\nFrecuencia por celda (head):")
print(frecuencia_por_celda.head(10))
print("Suma frecuencias (casos asignados por sjoin):", frecuencia_por_celda.sum())
print("Casos originales dentro de Cali:", len(dentro))

print(f"\n  Estadísticas de celdas:")
print(f"    • Total Celdas: {len(grid)}")
print(f"    • Con casos: {len(grid[grid['frecuencia'] > 0]):,}")
print(f"    • Sin casos: {len(grid[grid['frecuencia'] == 0]):,}")
print(f"    • Alta concentración (>15 casos): {len(grid[grid['frecuencia'] > 15]):,}")
print(f"     \n")

fig, ax = plt.subplots(figsize=(9, 8))

# Mapa base
cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.5, zorder=1)

# Limites administrativos
try:
    comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                    tags={'admin_level': ['8', '9', '10']})
    if len(comunas) > 0:
        comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        comunas = comunas.to_crs(cali.crs)
        comunas.plot(ax=ax, color="none", edgecolor="gray", linewidth=0.8, 
                     alpha=0.5, linestyle='--', zorder=2)
        print("  ✓ Límites administrativos")
except Exception as e:
    print("  ⚠️ No se pudieron cargar comunas:", e)

# 🎨 Definir el colormap tipo semáforo
# blanco (sin datos), verde, azul, naranja, rojo (máximo)
colors = ["white", "green", "blue", "orange", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("semaforo", colors, N=5)

# Graficar la cuadrícula con el nuevo colormap
grid_cali.plot(ax=ax, column="quintil", cmap=cmap,
               alpha=0.75, edgecolor=None, vmin=0, vmax=5, zorder=3)

# Barra de color personalizada
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Nivel de Riesgo", shrink=0.75)
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels([
    'Sin datos', 'Q1\nMuy Bajo', 'Q2\nBajo',
    'Q3\nMedio', 'Q4\nAlto', 'Q5\nMuy Alto'
])

# Título y etiquetas
plt.title(f"Mapa de Calor de Seguridad - Santiago de Cali", 
          fontsize=15, pad=20, weight='bold')
plt.xlabel("Coordenada X (metros)", fontsize=11)
plt.ylabel("Coordenada Y (metros)", fontsize=11)

# Estadísticas resumen
if grid_cali.empty or grid_cali['casos'].isna().all():
    print("No hay datos de casos.")
else:
    max_idx = grid_cali['casos'].idxmax()
    max_val = int(grid_cali.loc[max_idx, 'casos'] or 0)
    poly = grid_cali.loc[max_idx, 'geometry']
    centroid = poly.centroid
    pct_total = (max_val / grid_cali['casos'].sum() * 100) if grid_cali['casos'].sum() > 0 else 0
    print(f"Celda {max_idx}: {max_val} casos (centro: {centroid.x:.2f}, {centroid.y:.2f}) — {pct_total:.1f}% del total")

median = grid_cali['casos'].median()
median_activas = grid_cali.loc[grid_cali['casos'] > 0, 'casos'].median()
mean = grid_cali['casos'].mean()
maxv = grid_cali['casos'].max()
active_count = (grid_cali['casos']>0).sum()
noactivecount = (grid_cali['casos']==0).sum()
stats = (f"Índice Promedio: {grid_cali['casos'].mean():.1f}\n"
         f"Mediana: {median_activas:.1f}\n"
         f"Máximo: {maxv:.1f}\n"
         f"Promedio: {mean:.1f}\n"
         f"Celdas activas: {active_count:,}\n"
         f"Celdas no activas: {noactivecount:,}\n")
max_idx = grid_cali['casos'].idxmax()
centroid = grid_cali.loc[max_idx].geometry.centroid
max_val = int(grid_cali.loc[max_idx, 'casos'])
poly = grid_cali.loc[max_idx, 'geometry']
centroid = poly.centroid
print(f"Celda con más casos -> index: {max_idx}, casos: {max_val}")
print(f"Centroide -> x: {centroid.x:.2f}, y: {centroid.y:.2f}")
total_casos = int(grid_cali['casos'].sum())
celdas_con_casos = int((grid_cali['casos'] > 0).sum())
total_celdas = int(len(grid_cali))

print(f"Total casos asignados a celdas: {total_casos:,}")
print(f"Celdas con al menos 1 caso: {celdas_con_casos:,} de {total_celdas:,}")

# opcional: ver el polígono (coordenadas de los vértices)
print("Polígono (vértices):", list(poly.exterior.coords))
# opcional: ver todas las columnas de esa celda
print(grid_cali.loc[max_idx])
plt.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()


# 1) Intentar seleccionar por pertenencia espacial (más robusto)
casos_en_celda = gdf_casos[gdf_casos.within(poly)].copy()

# 2) Si no encuentra nada, intentar por columnas añadidas por el sjoin ('grid_id' o 'index_right')
if len(casos_en_celda) == 0:
    gid = grid_cali.loc[max_idx].get('grid_id', None)
    mask = pd.Series(False, index=casos_con_celda.index)
    if gid is not None:
        mask = mask | (casos_con_celda.get('grid_id') == gid)
    mask = mask | (casos_con_celda.get('index_right') == max_idx)
    casos_en_celda = casos_con_celda[mask].copy()

# Mostrar detalles y listar los casos pertenecientes a la celda con más casos
max_idx = grid_cali['casos'].idxmax()
poly = grid_cali.loc[max_idx, 'geometry']
print(f"\nCelda con más casos -> index: {max_idx}, casos: {int(grid_cali.loc[max_idx,'casos'])}")
# ...existing code...
from pathlib import Path
# ...existing code...
# Mostrar resumen y ejemplos
total = len(casos_en_celda)
print(f"Total de casos en la celda: {total:,}")
if total > 0:
    cols_show = [c for c in ['archivo_fuente','categoria','tipo_delito','nivel_severidad','nivel_agrupado','peso_severidad','x','y','geometry'] if c in casos_en_celda.columns]
    print("\nEjemplos (hasta 20) de casos en la celda:")
    print(casos_en_celda[cols_show].head(20).to_string(index=False))
    # Guardar detalle a CSV para revisión
    out = Path(__file__).resolve().parent / f"casos_celda_{max_idx}.csv"
    casos_en_celda.to_csv(out, index=False, encoding='utf-8')
    print(f"\nDetalle guardado en: {out}")
else:
    print("No se encontraron casos asociados a la celda (revisar CRS / sjoin).")