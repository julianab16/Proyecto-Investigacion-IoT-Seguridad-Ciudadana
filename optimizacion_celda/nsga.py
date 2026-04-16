import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print(" " * 15 + "OPTIMIZACIÓN DE TAMAÑO DE CELDA - NSGA-II")
print("=" * 70)

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
print("\n[1/4] Cargando archivos de delitos...")

datasets = []
archivos_a_buscar = [
        ('data_base/Hurtos_fiscalia.csv', 'Hurto'),
        ('data_base/Homicidios_fiscalia.csv', 'Homicidio'),
        ('data_base/Delitos_Sexuales_fiscalia.csv', 'Delitos Sexuales'),
        ('data_base/Lesiones_fiscalia.csv', 'Lesiones Personales'),
        ('data_base/Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
        ('data_base/Extorsion_fiscalia.csv', 'Extorsion')
]

archivos_cargados = 0
for archivo, nombre in archivos_a_buscar:
    if os.path.isfile(archivo):
        df = cargar_csv(archivo)
        if df is not None and len(df) > 0:
            datasets.append(df)
            print(f"  ✓ {archivo}: {len(df):,} registros")
            archivos_cargados += 1
    else:
        print(f"  - Ignorado (no existe): {archivo}")

# También buscar cualquier otro CSV en la carpeta

if archivos_cargados == 0:
    print("\n✗ ERROR: No se encontró ningún archivo CSV válido")
    exit()

# Combinar todos los datasets
df = pd.concat(datasets, ignore_index=True)
print(f"\n✓ Total combinado: {len(df):,} registros")

# ========== CARGAR MAPA DE CALI ==========
print("\n[2/4] Cargando límites de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)  # CRS métrico para Colombia

try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

print("✓ Mapa de Cali cargado")

# ========== CREAR GEODATAFRAME ==========
print("\n[3/4] Georeferenciando puntos...")
puntos = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['x'], df['y']), 
    crs="EPSG:4326"
)
puntos = puntos.to_crs(3116)

# Filtrar puntos dentro de Cali
puntos = puntos[puntos.within(area_cali)]
print(f"✓ Puntos dentro de Cali: {len(puntos):,}")

if len(puntos) == 0:
    print("✗ ERROR: No hay puntos dentro de Cali")
    exit()

# ========== FUNCIONES DE OPTIMIZACIÓN ==========
def calcular_metricas(puntos_gdf, cali_gdf, h_m):
    """
    Calcula métricas para optimización:
    - f1: Varianza de densidad (homogeneidad espacial)
    - f2: Número de celdas (costo computacional)
    - f3: Porcentaje de celdas vacías (calidad del mapa)
    """
    minx, miny, maxx, maxy = cali_gdf.total_bounds
    
    # Crear grilla métrica
    x_coords = np.arange(minx, maxx + h_m, h_m)
    y_coords = np.arange(miny, maxy + h_m, h_m)
    
    celdas = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            celda = box(x, y, x + h_m, y + h_m)
            if celda.intersects(area_cali):
                celdas.append(celda)
    
    if len(celdas) == 0:
        return float('inf'), float('inf'), 100.0
    
    grid = gpd.GeoDataFrame(geometry=celdas, crs=cali_gdf.crs)
    
    # Contar puntos por celda
    join = gpd.sjoin(puntos_gdf, grid, how="inner", predicate="within")
    conteo = join.groupby("index_right").size()
    
    # Crear array completo con ceros para celdas vacías
    conteo_completo = np.zeros(len(grid))
    conteo_completo[conteo.index] = conteo.values
    
    # Calcular métricas
    # f1: Varianza de densidad
    densidad = conteo_completo / (h_m**2 / 1e6)  # eventos por km²
    varianza = float(np.var(densidad))
    
    # f2: Número de celdas
    num_celdas = float(len(grid))
    
    # f3: Porcentaje de celdas vacías
    pct_vacias = float((conteo_completo == 0).sum() / len(grid) * 100)
    
    return varianza, num_celdas, pct_vacias

# ========== PROBLEMA DE OPTIMIZACIÓN ==========
class OptimizacionMapaCalor(ElementwiseProblem):
    """
    Optimización multi-objetivo para tamaño de celda
    """
    def __init__(self):
        super().__init__(
            n_var=1,
            n_obj=3,
            xl=np.array([50.0]),   # Mínimo
            xu=np.array([250.0])   # Máximo 
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        h = float(x[0])
        varianza, num_celdas, pct_vacias = calcular_metricas(puntos, cali, h)
        
        # Normalizar objetivos
        f1 = varianza / 1000
        f2 = num_celdas / 100
        f3 = pct_vacias
        
        out["F"] = np.array([f1, f2, f3], dtype=float)


# ========== EJECUTAR NSGA-II ==========
print("\n[4/4] Ejecutando optimización NSGA-II...")
print("Esto puede tomar varios minutos...\n")

problem = OptimizacionMapaCalor()

algorithm = NSGA2(
    pop_size=50,
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 8)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True,
    save_history=False
)

# ========== ANALIZAR RESULTADOS ==========
print("\n" + "=" * 70)
print("RESULTADOS DE OPTIMIZACIÓN")
print("=" * 70)

X = res.X
F = res.F

# Calcular número total de evaluaciones
pop_size = 50
n_generaciones = 8
total_evaluaciones = pop_size * n_generaciones
soluciones_pareto = len(X)

print(f"\n📊 Información del algoritmo NSGA-II:")
print(f"   • Tamaño de población: {pop_size}")
print(f"   • Número de generaciones: {n_generaciones}")
print(f"   • Total de evaluaciones: {total_evaluaciones}")
print(f"   • Soluciones en frente de Pareto: {soluciones_pareto}")

# Calcular métricas detalladas para cada solución
resultados = []
for h in X[:, 0]:
    var, n_celdas, pct_vac = calcular_metricas(puntos, cali, h)
    eventos_por_celda = len(puntos) / n_celdas
    resultados.append({
        'h': h,
        'varianza': var,
        'n_celdas': int(n_celdas),
        'pct_vacias': pct_vac,
        'eventos_x_celda': eventos_por_celda
    })

resultados_df = pd.DataFrame(resultados)
resultados_df = resultados_df.sort_values('eventos_x_celda', ascending=False)

print("\nTOP 15 SOLUCIONES ÓPTIMAS:")
print("-" * 100)
print(f"{'Tamaño':<12} {'N° Celdas':<12} {'Eventos/Celda':<18} {'% Vacías':<12} {'Varianza':<15}")
print("-" * 100)
for _, row in resultados_df.head(15).iterrows():
    print(f"{row['h']:>7.1f} m    {row['n_celdas']:>8}    "
          f"{row['eventos_x_celda']:>13.2f}      {row['pct_vacias']:>8.1f}%    "
          f"{row['varianza']:>12.2f}")

# ========== SELECCIONAR MEJOR SOLUCIÓN DEL FRENTE DE PARETO ==========
print("\n" + "=" * 70)
print("SELECCIÓN DE SOLUCIÓN ÓPTIMA DEL FRENTE DE PARETO")
print("=" * 70)

print(f"\n📊 Configuración del algoritmo NSGA-II:")
print(f"   • Tamaño de población: {pop_size}")
print(f"   • Número de generaciones: {n_generaciones}")
print(f"   • Evaluaciones totales: {total_evaluaciones}")
print(f"   • Soluciones en frente de Pareto: {soluciones_pareto}")

print(f"\n🔍 Criterios de selección de la solución final:")
print(f"   1. Eventos por celda: entre 5 y 15 (ideal: 8)")
print(f"   2. Celdas vacías: < 70%")
print(f"   3. Número de celdas: razonable para visualización")

# Filtrar soluciones válidas
validas = resultados_df[
    (resultados_df['eventos_x_celda'] >= 5) & 
    (resultados_df['eventos_x_celda'] <= 15) &
    (resultados_df['pct_vacias'] < 70)
]

print(f"\n   ✓ Soluciones que cumplen criterios estrictos: {len(validas)}")

if len(validas) == 0:
    print(f"   ⚠️  Ninguna solución cumple criterios estrictos, relajando...")
    # Si no hay soluciones que cumplan criterios estrictos, relajar
    validas = resultados_df[
        (resultados_df['eventos_x_celda'] >= 3) & 
        (resultados_df['pct_vacias'] < 75)
    ]
    print(f"   ✓ Soluciones con criterios relajados: {len(validas)}")

if len(validas) > 0:
    # Elegir la que tenga eventos/celda más cercano a diez (valor ideal)
    diferencias = validas['eventos_x_celda'].sub(10).abs()
    mejor_idx = diferencias.idxmin()
    mejor = validas.loc[mejor_idx]
    criterio_usado = "más cercano a 8 eventos/celda (ideal)"
else:
    # Fallback: mejor por eventos/celda
    mejor_idx = resultados_df['eventos_x_celda'].idxmax()
    mejor = resultados_df.loc[mejor_idx]
    criterio_usado = "máximo eventos/celda (fallback)"

print(f"\n   🎯 Solución seleccionada por: {criterio_usado}")
print(f"   📍 Distancia al ideal (8 eventos/celda): {abs(mejor['eventos_x_celda'] - 8):.2f}")

print("\n" + "=" * 70)
print("★ RECOMENDACIÓN ÓPTIMA ★")
print("=" * 70)
print(f"Tamaño de celda:     {mejor['h']:.1f} metros")
print(f"Número de celdas:    {mejor['n_celdas']:,}")
print(f"Eventos por celda:   {mejor['eventos_x_celda']:.2f}")
print(f"Celdas vacías:       {mejor['pct_vacias']:.1f}%")
print(f"Varianza:            {mejor['varianza']:.2f}")
print("=" * 70)

# ========== ANÁLISIS ADICIONAL ==========
print("\nANÁLISIS COMPARATIVO:")
print("-" * 70)

# ========== VISUALIZACIÓN ==========
print("\n[5/5] Generando visualización comparativa...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Frente de Pareto (Varianza vs Número de celdas)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(F[:, 1], F[:, 0], 
                       c=F[:, 2], cmap='RdYlGn_r', s=100, alpha=0.7)
ax1.set_xlabel('N° Celdas (normalizado)', fontsize=11)
ax1.set_ylabel('Varianza (normalizada)', fontsize=11)
ax1.set_title('Frente de Pareto: Varianza vs N° Celdas', fontsize=12, weight='bold')
plt.colorbar(scatter1, ax=ax1, label='% Celdas Vacías')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Tamaño vs Eventos por celda
ax2 = axes[0, 1]
ax2.scatter(resultados_df['h'], resultados_df['eventos_x_celda'], 
            c=resultados_df['pct_vacias'], cmap='RdYlGn_r', s=100, alpha=0.7)
ax2.axhline(y=8, color='red', linestyle='--', label='Ideal (8 eventos/celda)')
ax2.axvline(x=mejor['h'], color='green', linestyle='--', linewidth=2, 
            label=f'Óptimo ({mejor["h"]:.0f}m)')
ax2.set_xlabel('Tamaño de Celda (metros)', fontsize=11)
ax2.set_ylabel('Eventos por Celda', fontsize=11)
ax2.set_title('Tamaño de Celda vs Densidad de Eventos', fontsize=12, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Tamaño vs % Celdas vacías
ax3 = axes[1, 0]
ax3.plot(resultados_df['h'], resultados_df['pct_vacias'], 'o-', 
         markersize=8, linewidth=2, color='steelblue')
ax3.axvline(x=mejor['h'], color='green', linestyle='--', linewidth=2,
            label=f'Óptimo ({mejor["h"]:.0f}m)')
ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Límite aceptable')
ax3.set_xlabel('Tamaño de Celda (metros)', fontsize=11)
ax3.set_ylabel('% Celdas Vacías', fontsize=11)
ax3.set_title('Tamaño de Celda vs Celdas Vacías', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Tamaño vs Número de celdas
ax4 = axes[1, 1]
ax4.plot(resultados_df['h'], resultados_df['n_celdas'], 's-', 
         markersize=8, linewidth=2, color='darkorange')
ax4.axvline(x=mejor['h'], color='green', linestyle='--', linewidth=2,
            label=f'Óptimo ({mejor["h"]:.0f}m)')
ax4.set_xlabel('Tamaño de Celda (metros)', fontsize=11)
ax4.set_ylabel('Número de Celdas', fontsize=11)
ax4.set_title('Tamaño de Celda vs Número Total de Celdas', fontsize=12, weight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Optimización Multi-Objetivo - Tamaño Óptimo: {mejor["h"]:.0f}m', 
             fontsize=14, weight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("✓ OPTIMIZACIÓN COMPLETADA")
print("=" * 70)
print(f"\nResumen:")
print(f"  • Total de eventos analizados: {len(puntos):,}")
print(f"  • Evaluaciones realizadas (NSGA-II): {total_evaluaciones}")
print(f"  • Soluciones en frente de Pareto: {soluciones_pareto}")
print(f"  • Tamaño óptimo de celda: {mejor['h']:.1f} metros")
print(f"  • Cobertura del mapa: {100 - mejor['pct_vacias']:.1f}%")
print("\nUsa este valor en tu código de georeferenciación:")
print(f"  LADO_HEX = {mejor['h']:.1f}  # metros\n")


from pathlib import Path
import json

resultados_dir = Path(__file__).resolve().parent.parent / "optimizacion_celda"
resultados_dir.mkdir(exist_ok=True)

# 1. Exportar mejorcelda
config = {
    'celda': float(mejor['h']),
    'metodo': 'NSGA II',
}

out_config = resultados_dir / "nsgacelda.json"
out_config.write_text(json.dumps(config, indent=2))
print(f"✓ Exportado: {out_config}")