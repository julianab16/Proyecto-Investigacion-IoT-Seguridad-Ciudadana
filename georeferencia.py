# ========== IMPORTS ==========
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# En georeferencia.py u otro archivo
import json
from pathlib import Path

resultados_dir = Path(__file__).resolve().parent / "resultados_optimizacion"

# Cargar mejorcelda
with open(resultados_dir / "mejorcelda.json") as f:
    config = json.load(f)
    mejorcelda = config['mejorcelda']
    mejor_metodo_nombre = config['mejor_metodo']

# Cargar métodos con sus tamaños
with open(resultados_dir / "metodos_optimizacion.json") as f:
    metodos_optimizados = json.load(f)

# Cargar nsga
#with open(resultados_dir / "nsga.json") as f:
#    nsga = json.load(f)

print("\n") 
print(f"Mejor tamaño de celda: {mejorcelda} m")
print(f"Método ganador: {mejor_metodo_nombre}")
#print(f"\nMétodo: {nsga[metodo]}, Tamaño de celda: {nsga[celda]}m")
for nombre in metodos_optimizados:
    h_optimo = metodos_optimizados[nombre]['h_optimo']
    print(f"Método: {nombre}, Tamaño de celda: {h_optimo:.1f}m")
print("\n") 

# ========== CONFIGURACIÓN ==========
LADO_HEX = mejorcelda   # lado del hexágono en metros

print("=" * 70)
print(" " * 10 + "MAPA DE CALOR DE SEGURIDAD CON ENFOQUE DE GÉNERO")
print(" " * 20 + "Santiago de Cali, Colombia")
print("=" * 70)

# ========== DEFINICIÓN DE PESOS POR TIPO DE DELITO ==========
# Peso base (S) × Factor de género (G) = Peso total (P)
PESOS_DELITOS = {
    # Nivel 1 - Bajo (S=1, G=1.0, P=1.0)
    'Hurto': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
    'Extorsion': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
    
    # Nivel 2 - Moderado (S=2, G=1.0, P=2.0)
    'Lesiones Personales': {'severidad': 2, 'factor_genero': 1.0, 'peso_total': 2.0, 'nivel': 'Moderado'},
    
    # Nivel 3 - Alto (S=3, G=1.5, P=4.5)
    'Delitos Sexuales': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
    'Violencia Intrafamiliar': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
    'Lesiones Personales Agravados': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},

    
    # Nivel 4 - Crítico (S=4, G=1.5, P=6.0)
    'Homicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'},
    'Feminicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'}
}

print("\n📊 SISTEMA DE PESOS CON ENFOQUE DE GÉNERO")
print("─" * 70)
print(f"{'Tipo de Delito':<25} {'Severidad':<12} {'F.Género':<10} {'Peso Total':<10}")
print("─" * 70)
for delito, config in PESOS_DELITOS.items():
    print(f"{delito:<25} {config['severidad']:<12} {config['factor_genero']:<10.1f} {config['peso_total']:<10.1f}")
print("─" * 70)
print("💡 Los delitos con enfoque de género (violencia sexual, intrafamiliar,")
print("   feminicidio) tienen un peso amplificado de 1.5x\n")

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
grid = grid.reset_index(drop=False).rename(columns={'index':'grid_id'})
print(f"✓ {len(polygons):,} hexágonos generados")

# ========== FUNCIONES DE PROCESAMIENTO ==========
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

def procesar_csv(archivo, nombre_categoria=None):
    """Procesa un archivo CSV y retorna DataFrame limpio"""
    try:
        print(f"\n  Cargando: {archivo}")
        df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')
        
        if len(df) == 0:
            print(f"    ⚠ Archivo vacío")
            return None
        
        print(f"    • Registros iniciales: {len(df):,}")
        
        df = limpiar_coordenadas(df, 'y', 'x')
        
        if df is None or len(df) == 0:
            print(f"    ⚠ Sin coordenadas válidas")
            return None
        
        # ✅ NORMALIZAR TODAS LAS COLUMNAS A MINÚSCULAS
        df.columns = df.columns.str.strip().str.lower()
        
        if nombre_categoria:
            df['categoria'] = nombre_categoria
        else:
            nombre_archivo = archivo.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
            df['categoria'] = nombre_archivo
        
        df['archivo_fuente'] = archivo.split('/')[-1]
        
        print(f"    ✓ {len(df):,} registros procesados - Categoría: {df['categoria'].iloc[0]}")
        return df
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return None

# ========== CARGA DE ARCHIVOS CSV ==========
print("\n[3/6] Cargando archivos de delitos...")

archivos_especificos = [
    ('Hurtos_fiscalia.csv', 'Hurto'),
    ('Homicidios_fiscalia.csv', 'Homicidio'),
    ('Delitos_Sexuales_fiscalia.csv', 'Delitos Sexuales'),
    ('Lesiones_fiscalia.csv', 'Lesiones Personales'),
    ('Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
    ('Extorsion_fiscalia.csv', 'Extorsion')
]

datasets = []
archivos_encontrados = 0

for archivo, categoria in archivos_especificos:
    try:
        df = procesar_csv(archivo, categoria)
        if df is not None:
            datasets.append(df)
            archivos_encontrados += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  ⚠ Error procesando {archivo}: {str(e)}")

if archivos_encontrados == 0:
    print("\n" + "="*70)
    print("✗ ERROR: No se encontró ningún archivo CSV válido")
    print("="*70)
    exit()

print(f"\n✓ Total de archivos procesados: {archivos_encontrados}")








# ...existing code...

# Después de concatenar todos los dataframes:
df = pd.concat(datasets, ignore_index=True)
print(f"✓ Total de registros combinados: {len(df):,}")

# NORMALIZAR COLUMNAS DEL DATAFRAME COMBINADO
df.columns = df.columns.str.strip().str.lower()

# ========== ASIGNACIÓN DE PESOS CON ENFOQUE DE GÉNERO ==========
print("\n[4/6] Calculando puntajes con enfoque de género...")

# INSPECCIONAR COLUMNA nivel_severidad EN LESIONES
print("\n🔍 INSPECCIÓN - Valores únicos de nivel_severidad en Lesiones:")
lesiones_df = df[df['categoria'].str.contains('Lesion', case=False, na=False)].copy()
if len(lesiones_df) > 0 and 'nivel_severidad' in lesiones_df.columns:
    valores_unicos = lesiones_df['nivel_severidad'].unique()
    print(f"  Valores encontrados ({len(valores_unicos)} únicos):")
    for i, val in enumerate(valores_unicos):  # Mostrar primeros 20
        count = (lesiones_df['nivel_severidad'] == val).sum()
        print(f"    {i+1}. '{val}' → {count} registros")

def asignar_peso_delito(row):
    """Asigna peso según categoría de delito"""
    try:
        # Acceder a las columnas de la fila
        categoria = str(row['categoria']).strip() if 'categoria' in row.index else ''
        categoria_lower = categoria.lower()

        # 1. Búsqueda exacta por categoría
        if categoria in PESOS_DELITOS:
            return PESOS_DELITOS[categoria]['peso_total']

        # 2. Si es Lesión Personal, usar NIVEL_SEVERIDAD
        if 'lesion' in categoria_lower:
            nivel_severidad = None
            if 'nivel_severidad' in row.index:
                valor = row['nivel_severidad']
                if pd.notna(valor):  # Verificar que no sea NaN
                    nivel_severidad = str(valor).strip().lower()
            
            if nivel_severidad is not None and nivel_severidad != '':
                # Nivel 3 = Lesión Agravada
                if 'nivel 3' in nivel_severidad or nivel_severidad == '3':
                    return PESOS_DELITOS['Lesiones Personales Agravados']['peso_total']  # 4.5
                elif 'nivel 2' in nivel_severidad or nivel_severidad == '2':
                    return PESOS_DELITOS['Lesiones Personales']['peso_total']  # 2.0
            return PESOS_DELITOS['Lesiones Personales']['peso_total']  # 2.0 por defecto
        
        # 3. Búsqueda flexible para otros delitos
        if 'hurto' in categoria_lower:
            return PESOS_DELITOS['Hurto']['peso_total']  # 1.0
        elif 'extorsion' in categoria_lower:
            return PESOS_DELITOS['Extorsion']['peso_total']  # 1.0
        elif 'delitos sexuales' in categoria_lower or 'sexo' in categoria_lower:
            return PESOS_DELITOS['Delitos Sexuales']['peso_total']  # 4.5
        elif 'intrafamiliar' in categoria_lower or 'domestica' in categoria_lower:
            return PESOS_DELITOS['Violencia Intrafamiliar']['peso_total']  # 4.5
        elif 'feminicidio' in categoria_lower:
            return PESOS_DELITOS['Feminicidio']['peso_total']  # 6.0
        elif 'homicidio' in categoria_lower:
            return PESOS_DELITOS['Homicidio']['peso_total']  # 6.0
        else:
            return 1.0  # Peso por defecto
    
    except Exception as e:
        print(f"⚠ Error en asignar_peso_delito: {e}")
        return 1.0

df['peso_delito'] = df.apply(asignar_peso_delito, axis=1)







try:
    gdf_temp = gpd.GeoDataFrame(df, geometry=[Point(lon, lat) for lon, lat in zip(df['x'], df['y'])], crs="EPSG:4326")
    gdf_temp = gdf_temp.to_crs(cali.crs)
    casos_dentro_cali = gdf_temp[gdf_temp.within(cali.geometry.iloc[0])]
except Exception:
    # Si falla la georreferenciación, caer al conjunto completo
    casos_dentro_cali = gpd.GeoDataFrame(df, geometry=[Point(lon, lat) for lon, lat in zip(df['x'], df['y'])], crs="EPSG:4326")
































# ========== ASIGNACIÓN DE PESOS CON ENFOQUE DE GÉNERO ==========
print("\n[4/6] Calculando puntajes con enfoque de género...")

print(f"\n  📊 Distribución por categoría de delito (dentro de Cali):")
print("  " + "─" * 66)
total_dentro = len(casos_dentro_cali)
if total_dentro == 0:
    print("    ⚠ No hay registros dentro de los límites de Cali.")
else:
    # Separar feminicidios de homicidios regulares
    casos_mostrar = casos_dentro_cali.copy()
    
    # Identificar feminicidios en la base de homicidios
    if 'archivo_fuente' in casos_mostrar.columns:
        homicidios_mask = casos_mostrar['archivo_fuente'] == 'Homicidios_fiscalia.csv'
        
        if homicidios_mask.any():
            # Normalizar columnas para buscar feminicidios
            casos_mostrar.columns = casos_mostrar.columns.str.strip().str.lower()
            
            # Buscar columna de feminicidios
            col_feminicidio = None
            for col in ['feminicidios', 'feminicidio']:
                if col in casos_mostrar.columns:
                    col_feminicidio = col
                    break
            
            if col_feminicidio is not None:
                # Separar feminicidios de homicidios
                feminicidios_mask = (homicidios_mask & 
                                   (casos_mostrar[col_feminicidio].astype(str).str.upper() == 'S'))
                homicidios_no_feminicidios_mask = (homicidios_mask & 
                                                 (casos_mostrar[col_feminicidio].astype(str).str.upper() == 'N'))
                
                # Actualizar categorías
                casos_mostrar.loc[feminicidios_mask, 'categoria'] = 'Feminicidio'
                casos_mostrar.loc[homicidios_no_feminicidios_mask, 'categoria'] = 'Homicidio (sin feminicidio)'
                
                # Actualizar peso para feminicidios
                casos_mostrar.loc[feminicidios_mask, 'peso_delito'] = PESOS_DELITOS['Feminicidio']['peso_total']
    
    # Mostrar distribución actualizada
    for cat in sorted(casos_mostrar['categoria'].unique()):
        count = len(casos_mostrar[casos_mostrar['categoria'] == cat])
        peso = casos_mostrar[casos_mostrar['categoria'] == cat]['peso_delito'].iloc[0]
        pct = (count / total_dentro) * 100
        print(f"    {cat:<30} {count:>6,} ({pct:>5.1f}%) | Peso: {peso:.1f}")
print("  " + "─" * 66)

# ========== GEOREFERENCIACIÓN Y CÁLCULO DE PUNTAJES ==========
print("\n[5/6] Georeferenciando y calculando puntajes por celda...")

geometry = [Point(lon, lat) for lon, lat in zip(df['x'], df['y'])]
gdf_casos = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf_casos = gdf_casos.to_crs(grid.crs)

dentro = gdf_casos[gdf_casos.within(cali.geometry.iloc[0])]
print(f"  • Casos dentro de Cali: {len(dentro):,}")

# Spatial join
casos_con_celda = gpd.sjoin(gdf_casos, grid, how="left", predicate="within")
casos_con_celda = casos_con_celda.dropna(subset=["grid_id"])
# Cálculo del Score por celda: Score_i = Σ(N_ij × P_j)
print("\n  🧮 Calculando Score_i = Σ(N_ij × P_j)")
puntaje_por_celda = casos_con_celda.groupby("grid_id")["peso_delito"].sum().rename("score_raw")
frecuencia_por_celda = casos_con_celda.groupby("grid_id").size().rename("num_eventos")

grid = grid.merge(puntaje_por_celda, on="grid_id", how="left")
grid = grid.merge(frecuencia_por_celda, on="grid_id", how="left")
grid["score_raw"] = grid["score_raw"].fillna(0.0)
grid["num_eventos"] = grid["num_eventos"].fillna(0).astype(int)

# Normalización: ScoreNorm = (Score - Score_min) / (Score_max - Score_min)
print("  📏 Normalizando puntajes...")
score_min = grid["score_raw"].min()
score_max = grid["score_raw"].max()

if score_max > score_min:
    grid["score_normalizado"] = (grid["score_raw"] - score_min) / (score_max - score_min)
else:
    grid["score_normalizado"] = 0.0

# Escalar a 0-100
grid["indice_inseguridad"] = grid["score_normalizado"] * 100

print(f"    • Score mínimo: {score_min:.2f}")
print(f"    • Score máximo: {score_max:.2f}")
print(f"    • Rango normalizado: 0.00 - 1.00")

# Intersección con Cali
grid_cali = gpd.overlay(grid, cali, how="intersection")

# ========== CLASIFICACIÓN POR PERCENTILES ==========
print("\n  📊 Clasificando en niveles de inseguridad (percentiles)...")

indices = grid_cali["indice_inseguridad"].values
clasificacion = np.zeros_like(indices, dtype=int)

if np.any(indices > 0):
    # Percentiles: 0-20, 20-40, 40-60, 60-80, 80-100
    p20 = np.percentile(indices[indices > 0], 20)
    p40 = np.percentile(indices[indices > 0], 40)
    p60 = np.percentile(indices[indices > 0], 60)
    p80 = np.percentile(indices[indices > 0], 80)
    
    clasificacion[indices == 0] = 0  # Sin datos
    clasificacion[(indices > 0) & (indices <= p20)] = 1  # Muy bajo
    clasificacion[(indices > p20) & (indices <= p40)] = 2  # Bajo
    clasificacion[(indices > p40) & (indices <= p60)] = 3  # Medio
    clasificacion[(indices > p60) & (indices <= p80)] = 4  # Alto
    clasificacion[indices > p80] = 5  # Muy alto
    
    print(f"\n    Percentiles de clasificación:")
    print(f"      • Muy Bajo:   0.00 - {p20:.2f}")
    print(f"      • Bajo:      {p20:.2f} - {p40:.2f}")
    print(f"      • Medio:     {p40:.2f} - {p60:.2f}")
    print(f"      • Alto:      {p60:.2f} - {p80:.2f}")
    print(f"      • Muy Alto:  {p80:.2f}+")

grid_cali["nivel_inseguridad"] = clasificacion

# Estadísticas finales
print(f"\n  📈 Distribución de celdas por nivel:")
for nivel in range(6):
    etiquetas = ['Sin datos', 'Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
    count = (grid_cali["nivel_inseguridad"] == nivel).sum()
    pct = (count / len(grid_cali)) * 100 if len(grid_cali) > 0 else 0
    print(f"    {etiquetas[nivel]:<12}: {count:>4} celdas ({pct:>5.1f}%)")

# ========== VISUALIZACIÓN ==========
print("\n[6/6] Generando mapa de calor...")

fig, ax = plt.subplots(figsize=(12, 10))

# Mapa base
cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.5, zorder=1)

# Colormap tipo semáforo
colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)

# Graficar hexágonos
grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
               alpha=0.75, edgecolor="black", linewidth=0.3, vmin=0, vmax=5, zorder=3)

# Límites administrativos
try:
    comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                    tags={'admin_level': ['8']})
    if len(comunas) > 0:
        comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        comunas = comunas.to_crs(cali.crs)
        comunas.plot(ax=ax, color="none", edgecolor="black", linewidth=2.0, 
                     alpha=0.6, linestyle='-', zorder=3)
        print("  ✓ Límites administrativos cargados")
except:
    print("  ⚠ No se pudieron cargar límites administrativos")


# Barra de color
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Nivel de Inseguridad", shrink=0.7, pad=0.02)
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(['Sin datos', 'Muy Bajo\n(0-20%)', 'Bajo\n(20-40%)',
                     'Medio\n(40-60%)', 'Alto\n(60-80%)', 'Muy Alto\n(80-100%)'])

# Título
plt.title("Mapa de Calor de Seguridad con Enfoque de Género\nSantiago de Cali", 
          fontsize=16, pad=20, weight='bold')
plt.xlabel("Coordenada X (metros)", fontsize=11)
plt.ylabel("Coordenada Y (metros)", fontsize=11)

# Panel de estadísticas
total_eventos = len(dentro) 
celdas_activas = (grid_cali['num_eventos'] > 0).sum()
total_celdas = len(grid_cali)
max_idx = grid_cali['indice_inseguridad'].idxmax()
max_score = grid_cali.loc[max_idx, 'indice_inseguridad']
max_eventos = grid_cali.loc[max_idx, 'num_eventos']

stats_text = (
    f"GENERAL STATISTICS\n"
    f"{'─'*21}\n"
    f"Total events: {total_eventos:,}\n"
    f"Total cells: {total_celdas:,}\n"
    f"Cells with data: {celdas_activas:,}\n"
    f"Cells without data: {total_celdas - celdas_activas:,}"
)

plt.text(0.56, 0.19, stats_text, transform=ax.transAxes, fontsize=9,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))

# left, bottom, right, top, wspace, hspace
plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.86)


plt.grid(True, alpha=0.5, linestyle='--')
plt.savefig('mapa_calor_genero_cali.png', dpi=300, bbox_inches='tight')
print("\n✓ Mapa guardado como 'mapa_calor_genero_cali.png'")
plt.show()

# ========== EXPORTAR DETALLE DE CELDA MÁS CRÍTICA ==========
print("\n📁 Exportando detalle de celda más crítica...")
poly = grid_cali.loc[max_idx, 'geometry']
casos_en_celda = gdf_casos[gdf_casos.within(poly)].copy()

if len(casos_en_celda) > 0:
    out_path = Path(__file__).resolve().parent / f"celda_critica_{max_idx}.csv"
    casos_en_celda.to_csv(out_path, index=False, encoding='utf-8')
    print(f"✓ Detalle guardado en: {out_path}")
    print(f"  • {len(casos_en_celda):,} eventos en esta celda")
else:
    print("⚠ No se encontraron casos en la celda crítica")

print("\n" + "="*70)
print("✓ PROCESO COMPLETADO")
print("="*70)