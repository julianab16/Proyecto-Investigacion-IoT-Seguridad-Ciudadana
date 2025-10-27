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
from sclim import mejorcelda
print("Mejor celda:", mejorcelda)
# ========== CONFIGURACIÓN ==========
LADO_HEX = mejorcelda  # lado del hexágono en metros

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

casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
print(f"  ✓ Casos asignados a hexágonos")

# ========== CÁLCULO DE ÍNDICE ==========
def mapear_frecuencia(freq):
    if freq == 0: return 0
    elif freq == 1: return 1
    elif 2 <= freq <= 5: return 3.5
    elif 6 <= freq <= 12: return 9
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

grid["frecuencia"] = grid.index.map(frecuencia_por_celda).fillna(0)
grid["peso_promedio"] = grid.index.map(peso_promedio_celda).fillna(0)
grid["indice_severidad"] = grid.index.map(indice_severidad_celda).fillna(0)
grid["casos"] = grid["frecuencia"]

print(f"\n  Estadísticas de celdas:")
print(f"    • Con casos: {len(grid[grid['frecuencia'] > 0]):,}")
print(f"    • Alta concentración (>12 casos): {len(grid[grid['frecuencia'] > 12]):,}")

grid_cali = gpd.overlay(grid, cali, how="intersection")

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
print("\n[6/6] Generando mapa...")
fig, ax = plt.subplots(figsize=(16, 14))

cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.5, zorder=1)

try:
    comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                    tags={'admin_level': ['8', '9', '10']})
    if len(comunas) > 0:
        comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        comunas = comunas.to_crs(cali.crs)
        comunas.plot(ax=ax, color="none", edgecolor="gray", linewidth=0.8, 
                    alpha=0.5, linestyle='--', zorder=2)
        print("  ✓ Límites administrativos")
except:
    pass

grid_cali.plot(ax=ax, column="quintil", cmap="inferno", 
               alpha=0.65, edgecolor=None, vmin=0, vmax=5, zorder=3)

sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=5))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Nivel de Riesgo", shrink=0.75)
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(['Sin datos', 'Q1\nMuy Bajo', 'Q2\nBajo', 
                     'Q3\nMedio', 'Q4\nAlto', 'Q5\nMuy Alto'])

plt.title(f"Mapa de Calor de Seguridad - Santiago de Cali\n({len(df):,} delitos georeferenciados - 2023)", 
          fontsize=15, pad=20, weight='bold')
plt.xlabel("Coordenada X (metros)", fontsize=11)
plt.ylabel("Coordenada Y (metros)", fontsize=11)

stats = f"Índice Promedio: {grid_cali['indice_severidad'].mean():.1f}\n"
stats += f"Zona Crítica: {grid_cali['indice_severidad'].max():.1f}"
plt.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nResumen:")
print(f"  • Archivos procesados: {archivos_encontrados}")
print(f"  • Total de delitos: {len(df):,}")
print(f"  • Hexágonos activos: {len(grid_cali[grid_cali['casos'] > 0]):,}")
print(f"  • Índice promedio: {grid_cali['indice_severidad'].mean():.2f}")
print(f"  • Zona más crítica: {grid_cali['indice_severidad'].max():.2f}\n")