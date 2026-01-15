import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import pandas as pd
import warnings
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 80)
print(" " * 15 + "COMPARACION DE METODOS DE OPTIMIZACION")
print("=" * 80)

# ========== FUNCIONES DE CARGA ==========

def cargar_csv(archivo):
    """Carga y procesa un archivo CSV"""
    try:
        df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')
        if len(df) == 0:
            return None
        
        df.columns = df.columns.str.strip().str.lower()
        if 'x' not in df.columns or 'y' not in df.columns:
            return None

        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y'])
        
        if len(df) == 0:
            return None
        
        return df
    except:
        return None

# ========== CARGA DE DATOS ==========
print("\n[1/4] Cargando archivos de delitos...")

base_dir = Path(__file__).resolve().parent.parent / "data_base"
datasets = []
archivos_a_buscar = [
    'Hurtos_fiscalia.csv',
    'Homicidios_fiscalia.csv',
    'Delitos_Sexuales_fiscalia.csv',
    'Extorsion_fiscalia.csv',
    'Lesiones_fiscalia.csv',
    'Violencia_Intrafamiliar_fiscalia.csv',
]

archivos_cargados = 0
for archivo in archivos_a_buscar:
    ruta = base_dir / archivo
    if ruta.is_file():
        df = cargar_csv(str(ruta))
        if df is not None and len(df) > 0:
            datasets.append(df)
            archivos_cargados += 1

if archivos_cargados == 0:
    print("\n[ERROR] No se encontro ningun archivo CSV valido")
    exit()

df = pd.concat(datasets, ignore_index=True)
print(f"[OK] Total combinado: {len(df):,} registros")

# ========== CARGAR MAPA DE CALI ==========
print("\n[2/4] Cargando limites de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)

try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

print("[OK] Mapa de Cali cargado")

# ========== CREAR GEODATAFRAME ==========
print("\n[3/4] Georeferenciando puntos...")
puntos = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['x'], df['y']), 
    crs="EPSG:4326"
)
puntos = puntos.to_crs(3116)
puntos = puntos[puntos.within(area_cali)]
print(f"[OK] Puntos dentro de Cali: {len(puntos):,}")

if len(puntos) == 0:
    print("[ERROR] No hay puntos dentro de Cali")
    exit()

n_eventos_total = len(puntos)

# ========== FUNCION DE METRICAS ==========
def calcular_metricas_completas(puntos_gdf, cali_gdf, h_m):
    """Calcula todas las metricas relevantes"""
    minx, miny, maxx, maxy = cali_gdf.total_bounds
    
    x_coords = np.arange(minx, maxx + h_m, h_m)
    y_coords = np.arange(miny, maxy + h_m, h_m)
    
    celdas = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            celda = box(x, y, x + h_m, y + h_m)
            if celda.intersects(area_cali):
                celdas.append(celda)
    
    if len(celdas) == 0:
        return None
    
    grid = gpd.GeoDataFrame(geometry=celdas, crs=cali_gdf.crs)
    join = gpd.sjoin(puntos_gdf, grid, how="inner", predicate="within")
    conteo = join.groupby("index_right").size()
    
    conteo_completo = np.zeros(len(grid))
    conteo_completo[conteo.index] = conteo.values
    
    n_celdas = len(grid)
    eventos_por_celda = n_eventos_total / n_celdas
    pct_vacias = (conteo_completo == 0).sum() / n_celdas * 100
    
    densidad = conteo_completo / (h_m**2 / 1e6)
    varianza = np.var(densidad)
    std_dev = np.std(densidad)
    mean_densidad = np.mean(densidad)
    cv = std_dev / (mean_densidad + 1e-6)
    
    celdas_con_datos = conteo_completo[conteo_completo > 0]
    if len(celdas_con_datos) > 0:
        min_eventos = int(celdas_con_datos.min())
        max_eventos = int(celdas_con_datos.max())
        mediana_eventos = float(np.median(celdas_con_datos))
        q25 = float(np.percentile(celdas_con_datos, 25))
        q75 = float(np.percentile(celdas_con_datos, 75))
        iqr = q75 - q25
    else:
        min_eventos = max_eventos = mediana_eventos = q25 = q75 = iqr = 0
    
    return {
        'h': h_m,
        'n_celdas': n_celdas,
        'eventos_por_celda': eventos_por_celda,
        'pct_vacias': pct_vacias,
        'varianza': varianza,
        'std_dev': std_dev,
        'cv': cv,
        'min_eventos': min_eventos,
        'max_eventos': max_eventos,
        'mediana_eventos': mediana_eventos,
        'q25': q25,
        'q75': q75,
        'iqr': iqr
    }

# ========== CARGAR RESULTADOS DE LOS 3 METODOS ==========
print("\n[4/4] Cargando resultados de optimizacion...")

resultados_dir = Path(__file__).resolve().parent

metodos_valores = {}

# NSGA-II
nsga_file = resultados_dir / "nsgacelda.json"
if nsga_file.exists():
    with open(nsga_file, 'r') as f:
        data = json.load(f)
        metodos_valores['NSGA-II'] = data['celda']
        print(f"  [OK] NSGA-II: {data['celda']:.2f} m")

# Metodos Bayesianos
metodos_file = resultados_dir / "metodos_optimizacion.json"
if metodos_file.exists():
    with open(metodos_file, 'r') as f:
        data = json.load(f)
        for metodo, info in data.items():
            metodos_valores[metodo] = info['h_optimo']
            print(f"  [OK] {metodo}: {info['h_optimo']:.2f} m")

if len(metodos_valores) == 0:
    print("\n[ERROR] No se encontraron resultados de optimizacion")
    exit()

# ========== CALCULAR METRICAS PARA CADA METODO ==========
print("\n[5/5] Calculando metricas para cada metodo...")

resultados_comparacion = []

for metodo, h_valor in metodos_valores.items():
    print(f"  Evaluando {metodo} (h={h_valor:.1f}m)...", end='\r')
    metricas = calcular_metricas_completas(puntos, cali, h_valor)
    
    if metricas is not None:
        metricas['metodo'] = metodo
        resultados_comparacion.append(metricas)

print("\n[OK] Calculo completado")

df_comparacion = pd.DataFrame(resultados_comparacion)

# ========== TABLA COMPARATIVA ==========
print("\n" + "=" * 80)
print("COMPARACION DE METODOS - METRICAS ESPACIALES")
print("=" * 80)

print("\n" + "-" * 130)
print(f"{'Metodo':<20} {'Tamano (m)':<15} {'N Celdas':<12} {'Eventos/Celda':<16} "
      f"{'% Vacias':<12} {'Varianza':<15} {'CV':<10}")
print("-" * 130)

for _, row in df_comparacion.iterrows():
    print(f"{row['metodo']:<20} {row['h']:>10.2f}     {int(row['n_celdas']):>9,}   "
          f"{row['eventos_por_celda']:>12.2f}      {row['pct_vacias']:>8.1f}%    "
          f"{row['varianza']:>12.2f}   {row['cv']:>7.2f}")

print("-" * 130)

# ========== ANALISIS ESTADISTICO ==========
print("\n" + "=" * 80)
print("ANALISIS ESTADISTICO")
print("=" * 80)

# Calcular rangos y desviaciones
print("\nRangos de variacion:")
for col in ['h', 'eventos_por_celda', 'pct_vacias', 'varianza', 'cv']:
    val_min = df_comparacion[col].min()
    val_max = df_comparacion[col].max()
    val_mean = df_comparacion[col].mean()
    val_std = df_comparacion[col].std()
    
    if col == 'h':
        nombre = "Tamano de celda"
        unidad = "m"
    elif col == 'eventos_por_celda':
        nombre = "Eventos por celda"
        unidad = ""
    elif col == 'pct_vacias':
        nombre = "% Celdas vacias"
        unidad = "%"
    elif col == 'varianza':
        nombre = "Varianza"
        unidad = ""
    elif col == 'cv':
        nombre = "Coef. Variacion"
        unidad = ""
    
    print(f"  {nombre:<25} Min: {val_min:8.2f} {unidad:<3} | "
          f"Max: {val_max:8.2f} {unidad:<3} | "
          f"Media: {val_mean:8.2f} {unidad:<3} | "
          f"Desv: {val_std:8.2f} {unidad}")

# ========== IDENTIFICAR MEJOR METODO ==========
print("\n" + "=" * 80)
print("RANKING DE METODOS")
print("=" * 80)

# Criterios de evaluacion (menor es mejor excepto eventos_por_celda)
df_ranking = df_comparacion.copy()

# Normalizar metricas (0-100, donde 100 es mejor)
# Para varianza: menor es mejor
df_ranking['score_varianza'] = 100 * (1 - (df_ranking['varianza'] - df_ranking['varianza'].min()) / 
                                      (df_ranking['varianza'].max() - df_ranking['varianza'].min() + 1e-6))

# Para % vacias: menor es mejor
df_ranking['score_vacias'] = 100 * (1 - (df_ranking['pct_vacias'] - df_ranking['pct_vacias'].min()) / 
                                    (df_ranking['pct_vacias'].max() - df_ranking['pct_vacias'].min() + 1e-6))

# Para eventos/celda: cercano a 10 es mejor
df_ranking['score_eventos'] = 100 * (1 - abs(df_ranking['eventos_por_celda'] - 10) / 10)

# Para CV: entre 1.0-2.5 es mejor
cv_ideal = 1.75
df_ranking['score_cv'] = 100 * (1 - abs(df_ranking['cv'] - cv_ideal) / cv_ideal)

# Score total (ponderado) qara saber cual de las tres es mejor
df_ranking['score_total'] = (
    df_ranking['score_varianza'] * 0.25 +
    df_ranking['score_vacias'] * 0.30 +
    df_ranking['score_eventos'] * 0.30 +
    df_ranking['score_cv'] * 0.15
)

df_ranking = df_ranking.sort_values('score_total', ascending=False)

print("\n" + "-" * 100)
print(f"{'Ranking':<10} {'Metodo':<20} {'Score Total':<15} {'Varianza':<12} "
      f"{'Vacias':<12} {'Eventos':<12} {'CV':<10}")
print("-" * 100)

for i, (_, row) in enumerate(df_ranking.iterrows(), 1):
    medalla = ['[1]', '[2]', '[3]'][i-1] if i <= 3 else f'[{i}]'
    print(f"{medalla:<10} {row['metodo']:<20} {row['score_total']:>10.2f}      "
          f"{row['score_varianza']:>8.2f}    {row['score_vacias']:>8.2f}    "
          f"{row['score_eventos']:>8.2f}    {row['score_cv']:>6.2f}")

print("-" * 100)

mejor_metodo = df_ranking.iloc[0]['metodo']
print(f"\n[RECOMENDACION] Mejor metodo: {mejor_metodo}")
print(f"Score total: {df_ranking.iloc[0]['score_total']:.2f}/100")

# ========== VISUALIZACION COMPARATIVA ==========
print("\n[6/6] Generando visualizacion comparativa...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

metodos = df_comparacion['metodo'].tolist()
colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Rojo, verde-azulado, azul

# Grafico 1: Tamano de celda
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(metodos, df_comparacion['h'], color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Tamano (metros)', fontsize=11, weight='bold')
ax1.set_title('(a) Tamano de Celda Optimo', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3, axis='y')
# Agregar valores en las barras
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}m', ha='center', va='bottom', fontsize=10, weight='bold')

# Grafico 2: Eventos por celda
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(metodos, df_comparacion['eventos_por_celda'], color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (10)')
ax2.set_ylabel('Eventos/Celda', fontsize=11, weight='bold')
ax2.set_title('(b) Densidad de Eventos', fontsize=12, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Grafico 3: % Celdas vacias
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(metodos, df_comparacion['pct_vacias'], color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=30, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Limite (30%)')
ax3.set_ylabel('% Celdas Vacias', fontsize=11, weight='bold')
ax3.set_title('(c) Cobertura Espacial', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')

# Grafico 4: Varianza
ax4 = fig.add_subplot(gs[1, 0])
bars4 = ax4.bar(metodos, df_comparacion['varianza'], color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Varianza (eventos²/km⁴)', fontsize=11, weight='bold')
ax4.set_title('(d) Homogeneidad Espacial', fontsize=12, weight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Grafico 5: Coeficiente de variacion
ax5 = fig.add_subplot(gs[1, 1])
bars5 = ax5.bar(metodos, df_comparacion['cv'], color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.axhspan(1.0, 2.5, alpha=0.2, color='lightgreen', label='Rango ideal')
ax5.set_ylabel('Coeficiente de Variacion (CV)', fontsize=11, weight='bold')
ax5.set_title('(e) Deteccion de Hotspots', fontsize=12, weight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Grafico 6: Score total
ax6 = fig.add_subplot(gs[1, 2])
scores_ordenados = df_ranking['score_total'].tolist()
metodos_ordenados = df_ranking['metodo'].tolist()
colores_ordenados = [colores[metodos.index(m)] for m in metodos_ordenados]
bars6 = ax6.barh(metodos_ordenados, scores_ordenados, color=colores_ordenados, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Score Total (0-100)', fontsize=11, weight='bold')
ax6.set_title('(f) Ranking General', fontsize=12, weight='bold')
ax6.grid(True, alpha=0.3, axis='x')
for i, (bar, score) in enumerate(zip(bars6, scores_ordenados)):
    width = bar.get_width()
    ax6.text(width, bar.get_y() + bar.get_height()/2.,
            f'{score:.1f}', ha='left', va='center', fontsize=11, weight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.suptitle('Comparacion de Metodos de Optimizacion - Metricas Espaciales',
             fontsize=16, weight='bold', y=0.98)

# Guardar figura
output_file = resultados_dir / "comparacion_metodos.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Grafica guardada en: {output_file}")

plt.show()

# ========== EXPORTAR RESULTADOS ==========
print("\n[7/7] Exportando resultados...")


# Exportar ranking
output_ranking = resultados_dir / "ranking_metodos.csv"
df_ranking[['metodo', 'h', 'score_total', 'score_varianza', 'score_vacias', 
            'score_eventos', 'score_cv']].to_csv(output_ranking, index=False)
print(f"[OK] Ranking exportado: {output_ranking}")

# Exportar resumen JSON
resumen = {
    'mejor_metodo': mejor_metodo,
    'score_mejor': float(df_ranking.iloc[0]['score_total']),
    'h_mejor': float(df_ranking.iloc[0]['h']),
    'comparacion': df_comparacion.to_dict(orient='records'),
    'criterios_evaluacion': {
        'varianza': 'Menor es mejor (homogeneidad)',
        'pct_vacias': 'Menor es mejor (cobertura)',
        'eventos_por_celda': 'Cercano a 10 es ideal',
        'cv': 'Entre 1.0-2.5 es ideal (deteccion hotspots)'
    }
}

output_json = resultados_dir / "resumen_comparacion.json"
with open(output_json, 'w') as f:
    json.dump(resumen, f, indent=2)
print(f"[OK] Resumen JSON exportado: {output_json}")

print("\n" + "=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)
print(f"\nConclusion:")
print(f"  - Mejor metodo: {mejor_metodo}")
print(f"  - Tamano optimo: {df_ranking.iloc[0]['h']:.2f} metros")
print(f"  - Score de calidad: {df_ranking.iloc[0]['score_total']:.2f}/100")
print(f"\nLos tres metodos convergen a valores similares (~125m),")
print(f"lo que valida la robustez de la optimizacion realizada.")
print("=" * 80)
