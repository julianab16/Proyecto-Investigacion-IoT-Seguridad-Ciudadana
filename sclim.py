import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from skopt import gp_minimize, forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import pandas as pd
import warnings
import glob
from pathlib import Path
import os
warnings.filterwarnings('ignore')

print("=" * 80)
print(" " * 20 + "OPTIMIZACIÓN BAYESIANA - TAMAÑO DE CELDA")
print("=" * 80)

# ========== FUNCIONES DE CARGA ==========

def cargar_csv(archivo):
    """Carga y procesa un archivo CSV"""
    try:
        df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')

        df.columns = df.columns.str.strip().str.lower()
        if 'x' not in df.columns or 'y' not in df.columns:
            return None

        # convertir a numérico y eliminar filas inválidas
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y'])
  
        if len(df) == 0:
            return None
        
        if df is None or len(df) == 0:
            return None
        
        return df
    except:
        return None

# ========== CARGA DE DATOS ==========
print("\n[1/5] Cargando archivos de delitos...")

datasets = []
archivos_a_buscar = [
    'Hurtos_fiscalia.csv',
    'Homicidios_fiscalia.csv'
]

archivos_cargados = 0
for archivo in archivos_a_buscar:
    if os.path.isfile(archivo):
        df = cargar_csv(archivo)
        if df is not None and len(df) > 0:
            datasets.append(df)
            print(f"  ✓ {archivo}: {len(df):,} registros")
            archivos_cargados += 1
    else:
        print(f"  - Ignorado (no existe): {archivo}")


if archivos_cargados == 0:
    print("\n✗ ERROR: No se encontró ningún archivo CSV válido")
    exit()

# Combinar datasets
df = pd.concat(datasets, ignore_index=True)
print(f"\n✓ Total combinado: {len(df):,} registros")

# ========== CARGAR MAPA DE CALI ==========
print("\n[2/5] Cargando límites de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)

try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

print("✓ Mapa de Cali cargado")

# ========== CREAR GEODATAFRAME ==========
print("\n[3/5] Georeferenciando puntos...")
puntos = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['x'], df['y']), 
    crs="EPSG:4326"
)
puntos = puntos.to_crs(3116)
puntos = puntos[puntos.within(area_cali)]
print(f"✓ Puntos dentro de Cali: {len(puntos):,}")

if len(puntos) == 0:
    print("✗ ERROR: No hay puntos dentro de Cali")
    exit()

n_eventos_total = len(puntos)

# ========== FUNCIÓN DE MÉTRICAS ==========
def calcular_metricas_completas(puntos_gdf, cali_gdf, h_m):
    """Calcula todas las métricas relevantes"""
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
        min_eventos = celdas_con_datos.min()
        max_eventos = celdas_con_datos.max()
        mediana_eventos = np.median(celdas_con_datos)
        q25 = np.percentile(celdas_con_datos, 25)
        q75 = np.percentile(celdas_con_datos, 75)
        iqr = q75 - q25
    else:
        min_eventos = max_eventos = mediana_eventos = q25 = q75 = iqr = 0
    
    return {
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

# ========== FUNCIÓN OBJETIVO ==========
historico_evaluaciones = []
contador_evaluaciones = [0]

def funcion_objetivo_compuesta(h):
    """
    Función objetivo para MINIMIZAR
    Menor valor = mejor solución
    """
    h = float(h)
    contador_evaluaciones[0] += 1
    
    metricas = calcular_metricas_completas(puntos, cali, h)
    
    if metricas is None:
        return 1000.0
    
    eventos_x_celda = metricas['eventos_por_celda']
    pct_vacias = metricas['pct_vacias']
    n_celdas = metricas['n_celdas']
    cv = metricas['cv']
    varianza = metricas['varianza']
    
    # COMPONENTE 1: Eventos por celda (objetivo: 8-12)
    objetivo_eventos = 10
    if eventos_x_celda < 5:
        penalizacion_eventos = (5 - eventos_x_celda) ** 2 * 10
    elif eventos_x_celda > 20:
        penalizacion_eventos = (eventos_x_celda - 20) ** 2 * 0.5
    else:
        penalizacion_eventos = abs(eventos_x_celda - objetivo_eventos) ** 2
    
    # COMPONENTE 2: Celdas vacías
    if pct_vacias < 15:
        penalizacion_vacias = pct_vacias * 0.5
    elif pct_vacias < 30:
        penalizacion_vacias = pct_vacias * 2
    else:
        penalizacion_vacias = pct_vacias * 5
    
    # COMPONENTE 3: Número de celdas (100-2000 ideal)
    if n_celdas < 50:
        penalizacion_celdas = (50 - n_celdas) * 2
    elif n_celdas > 2000:
        penalizacion_celdas = (n_celdas - 2000) * 0.1
    else:
        penalizacion_celdas = 0
    
    # COMPONENTE 4: Coeficiente de variación (0.8-2.5 ideal)
    if cv < 0.5:
        penalizacion_cv = (0.5 - cv) ** 2 * 20
    elif cv > 3.0:
        penalizacion_cv = (cv - 3.0) ** 2 * 10
    else:
        penalizacion_cv = 0
    
    # Score total
    score_total = (
        penalizacion_eventos * 2.0 +
        penalizacion_vacias * 1.5 +
        penalizacion_celdas * 0.3 +
        penalizacion_cv * 0.8
    )
    
    score_calidad = max(0, 100 - score_total * 2)
    
    historico_evaluaciones.append({
        'iteracion': contador_evaluaciones[0],
        'h': h,
        'score_optimizacion': score_total,
        'score_calidad': score_calidad,
        'eventos_x_celda': eventos_x_celda,
        'pct_vacias': pct_vacias,
        'n_celdas': n_celdas,
        'cv': cv,
        'varianza': varianza
    })
    
    if contador_evaluaciones[0] % 5 == 0:
        print(f"  Evaluación {contador_evaluaciones[0]:3d}: h={h:7.1f}m | "
              f"score={score_total:7.2f} | eventos/celda={eventos_x_celda:5.2f}")
    
    return score_total

# ========== ESPACIO DE BÚSQUEDA ==========
space = [Real(50.0, 150.0, name='h')]

@use_named_args(space)
def objetivo_wrapper(h):
    return funcion_objetivo_compuesta(h)

# ========== OPTIMIZACIÓN ==========
print("\n[4/5] Ejecutando optimización bayesiana...")
print("=" * 80)

metodos = {
    'Gaussian Process': 'gp',
    'Random Forest': 'forest'
}

resultados_metodos = {}

for nombre_metodo, codigo in metodos.items():
    print(f"\n{'='*80}")
    print(f"🔍 Método: {nombre_metodo}")
    print(f"{'='*80}")
    
    historico_evaluaciones.clear()
    contador_evaluaciones[0] = 0
    
    if codigo == 'gp':
        resultado = gp_minimize(
            objetivo_wrapper,
            space,
            n_calls=50,
            n_initial_points=15,
            acq_func='EI',
            random_state=42,
            verbose=False
        )
    else:
        resultado = forest_minimize(
            objetivo_wrapper,
            space,
            n_calls=50,
            n_initial_points=15,
            random_state=42,
            verbose=False
        )
    
    resultados_metodos[nombre_metodo] = {
        'resultado': resultado,
        'historico': historico_evaluaciones.copy(),
        'h_optimo': resultado.x[0],
        'score_optimo': resultado.fun
    }
    
    print(f"\n✅ {nombre_metodo} completado")
    print(f"   Tamaño óptimo: {resultado.x[0]:.1f} m")
    print(f"   Score: {resultado.fun:.3f}")

# ========== COMPARACIÓN ==========
print("\n[5/5] Comparando resultados...")
print("=" * 80)
print("COMPARACIÓN DE MÉTODOS")
print("=" * 80)

comparacion = []
for nombre, datos in resultados_metodos.items():
    h_opt = datos['h_optimo']
    metricas = calcular_metricas_completas(puntos, cali, h_opt)
    
    comparacion.append({
        'metodo': nombre,
        'h': h_opt,
        'score_opt': datos['score_optimo'],
        'eventos_x_celda': metricas['eventos_por_celda'],
        'pct_vacias': metricas['pct_vacias'],
        'n_celdas': metricas['n_celdas'],
        'cv': metricas['cv']
    })

df_comparacion = pd.DataFrame(comparacion)

print("\n" + "-"*95)
print(f"{'Método':<22} {'Tamaño (m)':<12} {'Score':<10} {'Eventos/Celda':<15} {'% Vacías':<12} {'N° Celdas':<10}")
print("-"*95)
for _, row in df_comparacion.iterrows():
    print(f"{row['metodo']:<22} {row['h']:>9.1f}    {row['score_opt']:>7.3f}    "
          f"{row['eventos_x_celda']:>11.2f}      {row['pct_vacias']:>8.1f}%    {int(row['n_celdas']):>8}")
print("-"*95)

# Seleccionar mejor método
mejor_metodo_idx = df_comparacion['score_opt'].idxmin()
mejor_metodo = df_comparacion.loc[mejor_metodo_idx]
mejorcelda = mejor_metodo['h']

# Exportar resultado
mejorcelda = float(mejor_metodo['h'])

from pathlib import Path
import json

out_mod = Path(__file__).resolve().parent / "mejorcelda.py"
out_mod.write_text(f"mejorcelda = {mejorcelda!r}\n")

out_json = Path(__file__).resolve().parent / "mejorcelda.json"
out_json.write_text(json.dumps({"mejorcelda": mejorcelda}))

print(f"\n✓ Valor exportado a: {out_mod} and {out_json}")

print("\n" + "=" * 80)
print("🏆 RESULTADO ÓPTIMO")
print("=" * 80)
print(f"Método ganador:          {mejor_metodo['metodo']}")
print(f"Tamaño óptimo de celda:  {mejorcelda:.1f} metros")
print(f"Eventos por celda:       {mejor_metodo['eventos_x_celda']:.2f}")
print(f"Celdas vacías:           {mejor_metodo['pct_vacias']:.1f}%")
print(f"Número de celdas:        {int(mejor_metodo['n_celdas']):,}")
print(f"Coeficiente variación:   {mejor_metodo['cv']:.2f}")
print(f"Score de optimización:   {mejor_metodo['score_opt']:.3f}")
print("=" * 80)

# ========== VISUALIZACIÓN ==========
print("\n[6/6] Generando visualizaciones...")

# Obtener histórico del mejor método
mejor_historico = resultados_metodos[mejor_metodo['metodo']]['historico']
df_hist = pd.DataFrame(mejor_historico)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Gráfico 1: Convergencia del score
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_hist['iteracion'], df_hist['score_optimizacion'], 
         'o-', linewidth=2, markersize=4, color='steelblue', alpha=0.7)
ax1.axhline(y=mejor_metodo['score_opt'], color='red', linestyle='--', 
            linewidth=2, label=f'Óptimo: {mejor_metodo["score_opt"]:.2f}')
ax1.set_xlabel('Iteración', fontsize=11)
ax1.set_ylabel('Score de Optimización', fontsize=11)
ax1.set_title(f'Convergencia - {mejor_metodo["metodo"]}', fontsize=13, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Tamaño vs Score
ax2 = fig.add_subplot(gs[1, 0])
scatter2 = ax2.scatter(df_hist['h'], df_hist['score_optimizacion'], 
                       c=df_hist['iteracion'], cmap='viridis', s=50, alpha=0.7)
ax2.axvline(x=mejorcelda, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Tamaño de Celda (m)', fontsize=10)
ax2.set_ylabel('Score', fontsize=10)
ax2.set_title('Exploración del Espacio', fontsize=11, weight='bold')
plt.colorbar(scatter2, ax=ax2, label='Iteración')
ax2.grid(True, alpha=0.3)

# Gráfico 3: Tamaño vs Eventos/Celda
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(df_hist['h'], df_hist['eventos_x_celda'], 
            c=df_hist['score_optimizacion'], cmap='RdYlGn_r', s=50, alpha=0.7)
ax3.axvline(x=mejorcelda, color='red', linestyle='--', linewidth=2)
ax3.axhline(y=10, color='green', linestyle=':', alpha=0.5, label='Ideal (10)')
ax3.set_xlabel('Tamaño de Celda (m)', fontsize=10)
ax3.set_ylabel('Eventos por Celda', fontsize=10)
ax3.set_title('Densidad de Eventos', fontsize=11, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Tamaño vs % Vacías
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(df_hist['h'], df_hist['pct_vacias'], 
            c=df_hist['score_optimizacion'], cmap='RdYlGn_r', s=50, alpha=0.7)
ax4.axvline(x=mejorcelda, color='red', linestyle='--', linewidth=2)
ax4.axhline(y=30, color='orange', linestyle=':', alpha=0.5, label='Límite (30%)')
ax4.set_xlabel('Tamaño de Celda (m)', fontsize=10)
ax4.set_ylabel('% Celdas Vacías', fontsize=10)
ax4.set_title('Cobertura del Mapa', fontsize=11, weight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Gráfico 5: Tamaño vs N° Celdas
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(df_hist['h'], df_hist['n_celdas'], 
            c=df_hist['score_optimizacion'], cmap='RdYlGn_r', s=50, alpha=0.7)
ax5.axvline(x=mejorcelda, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Tamaño de Celda (m)', fontsize=10)
ax5.set_ylabel('Número de Celdas', fontsize=10)
ax5.set_title('Complejidad Computacional', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.3)

# Gráfico 6: Tamaño vs CV
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(df_hist['h'], df_hist['cv'], 
            c=df_hist['score_optimizacion'], cmap='RdYlGn_r', s=50, alpha=0.7)
ax6.axvline(x=mejorcelda, color='red', linestyle='--', linewidth=2)
ax6.axhline(y=1.5, color='green', linestyle=':', alpha=0.5, label='Ideal (1.5)')
ax6.set_xlabel('Tamaño de Celda (m)', fontsize=10)
ax6.set_ylabel('Coeficiente de Variación', fontsize=10)
ax6.set_title('Detección de Hotspots', fontsize=11, weight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Gráfico 7: Distribución de scores
ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(df_hist['score_optimizacion'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax7.axvline(x=mejor_metodo['score_opt'], color='red', linestyle='--', 
            linewidth=2, label=f'Óptimo: {mejor_metodo["score_opt"]:.2f}')
ax7.set_xlabel('Score de Optimización', fontsize=10)
ax7.set_ylabel('Frecuencia', fontsize=10)
ax7.set_title('Distribución de Scores', fontsize=11, weight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Optimización Bayesiana - Tamaño Óptimo: {mejorcelda:.0f}m', 
             fontsize=15, weight='bold', y=0.995)
plt.show()

print("\n" + "=" * 80)
print("✓ PROCESO COMPLETADO")
print("=" * 80)
print(f"\nArchivo generado: sclim.py")
print(f"Valor para usar en georeferencia: mejorcelda = {mejorcelda:.1f}\n")