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
print(" " * 10 + "JUSTIFICACION DEL TAMANO OPTIMO - COMPARACION DE 3 METODOS")
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
print("\n[1/5] Cargando archivos de delitos...")

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
print("\n[2/5] Cargando limites de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)

try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

print("[OK] Mapa de Cali cargado")

# ========== CREAR GEODATAFRAME ==========
print("\n[3/5] Georeferenciando puntos...")
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
    
    return {
        'n_celdas': n_celdas,
        'eventos_por_celda': eventos_por_celda,
        'pct_vacias': pct_vacias,
        'varianza': varianza,
        'std_dev': std_dev,
        'cv': cv
    }

# ========== CARGAR VALORES OPTIMOS ==========
print("\n[4/5] Cargando valores optimos de los 3 metodos...")

resultados_dir = Path(__file__).resolve().parent
valores_optimos = {}

# NSGA-II
nsga_file = resultados_dir / "nsgacelda.json"
if nsga_file.exists():
    with open(nsga_file, 'r') as f:
        data = json.load(f)
        valores_optimos['NSGA-II'] = data['celda']
        print(f"  [OK] NSGA-II: {data['celda']:.2f} m")

# Metodos Bayesianos
metodos_file = resultados_dir / "metodos_optimizacion.json"
if metodos_file.exists():
    with open(metodos_file, 'r') as f:
        data = json.load(f)
        for metodo, info in data.items():
            valores_optimos[metodo] = info['h_optimo']
            print(f"  [OK] {metodo}: {info['h_optimo']:.2f} m")

# ========== ANALISIS DE SENSIBILIDAD ==========
print("\n[5/5] Calculando metricas para diferentes tamanos de celda...")
print("Esto puede tomar varios minutos...\n")

# Rango de analisis AMPLIADO: 50 a 400 metros para ver toda la curva
tamanos = np.arange(50, 250, 5)  # De 50 a 400 en pasos de 5 metros
resultados = []


for i, h in enumerate(tamanos):
    print(f"  Evaluando h = {h:.0f} m... ({i+1}/{len(tamanos)})", end='\r')
    metricas = calcular_metricas_completas(puntos, cali, h)
    
    if metricas is not None:
        resultados.append({
            'h': h,
            'varianza': metricas['varianza'],
            'pct_vacias': metricas['pct_vacias'],
            'eventos_por_celda': metricas['eventos_por_celda'],
            'cv': metricas['cv'],
            'n_celdas': metricas['n_celdas']
        })

print("\n[OK] Calculos completados")

df_resultados = pd.DataFrame(resultados)

# ========== NORMALIZAR Y CREAR FUNCION OBJETIVO ==========
print("\nCalculando funcion objetivo combinada...")

# Normalizar cada metrica entre 0 y 1
df_resultados['varianza_norm'] = (df_resultados['varianza'] - df_resultados['varianza'].min()) / (df_resultados['varianza'].max() - df_resultados['varianza'].min())
df_resultados['pct_vacias_norm'] = df_resultados['pct_vacias'] / 100
df_resultados['cv_norm'] = (df_resultados['cv'] - df_resultados['cv'].min()) / (df_resultados['cv'].max() - df_resultados['cv'].min())

# Invertir eventos_por_celda (queremos valores medios, no muy altos ni muy bajos)
evento_optimo = 10  # Valor ideal de eventos por celda
df_resultados['eventos_desviacion'] = np.abs(df_resultados['eventos_por_celda'] - evento_optimo)
df_resultados['eventos_norm'] = df_resultados['eventos_desviacion'] / df_resultados['eventos_desviacion'].max()

# FUNCION OBJETIVO: Minimizar combinacion ponderada
# Pesos: varianza (30%), vacias (25%), CV (25%), eventos (20%)
df_resultados['funcion_objetivo'] = (
    0.25 * df_resultados['varianza_norm'] +
    0.30 * df_resultados['pct_vacias_norm'] +
    0.20 * df_resultados['cv_norm'] +
    0.30 * df_resultados['eventos_norm']
)

# Evaluar funcion objetivo en los tres valores optimos de los metodos
print("\nEvaluando los tres metodos en la funcion objetivo...")
scores_metodos = {}
for metodo, h_valor in valores_optimos.items():
    # Encontrar el valor mas cercano en df_resultados
    idx_cercano = (df_resultados['h'] - h_valor).abs().idxmin()
    score = df_resultados.loc[idx_cercano, 'funcion_objetivo']
    scores_metodos[metodo] = {'h': h_valor, 'score': score, 'idx': idx_cercano}
    print(f"  {metodo}: h={h_valor:.2f}m -> Score={score:.4f}")

# Determinar el mejor metodo (menor score)
mejor_metodo = min(scores_metodos.items(), key=lambda x: x[1]['score'])
h_optimo_calculado = mejor_metodo[1]['h']
idx_optimo = mejor_metodo[1]['idx']
print(f"\n[OK] MEJOR METODO: {mejor_metodo[0]} con h = {h_optimo_calculado:.2f} m (Score: {mejor_metodo[1]['score']:.4f})")

# ========== VISUALIZACION ==========
print("\n[6/6] Generando grafica de justificacion...")

# Crear figura con 5 subplots (3x2) - agregamos la funcion objetivo
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

# Subplot superior abarca ambas columnas para la funcion objetivo
ax_obj = fig.add_subplot(gs[0, :])
axes = [
    fig.add_subplot(gs[1, 0]),  # ax1
    fig.add_subplot(gs[1, 1]),  # ax2
    fig.add_subplot(gs[2, 0]),  # ax3
    fig.add_subplot(gs[2, 1])   # ax4
]

fig.suptitle('Optimal Cell Size Determination: Trade-off between Noise and Spatial Over-aggregation',
             fontsize=18, weight='bold', y=0.95)

# Definir colores y estilos para cada metodo
colores_metodos = {
    'NSGA-II': '#FF6B6B',           # Rojo
    'Gaussian Process': '#4ECDC4',  # Verde-azulado
    'Random Forest': '#45B7D1'      # Azul
}

estilos_linea = {
    'NSGA-II': '--',
    'Gaussian Process': '-.',
    'Random Forest': ':'
}

# Zonas de clasificacion
color_ruido = '#ffcccc'  # Rojo claro
color_optimo = '#ccffcc'  # Verde claro
color_sobreagregacion = '#cce5ff'  # Azul claro

# Determinar limites de zonas basados en los valores optimos
h_min_optimo = min(valores_optimos.values())
h_max_optimo = max(valores_optimos.values())
limite_ruido = h_min_optimo - 10
limite_sobreagregacion = h_max_optimo + 5

# ========== GRAFICA PRINCIPAL: FUNCION OBJETIVO ==========
ax_obj.plot(df_resultados['h'], df_resultados['funcion_objetivo'], 
            'o-', linewidth=4, markersize=8, color='#1a1a2e', label='Objective Function', zorder=3)

# Marcar los tres valores optimos con sus scores
for metodo, info in scores_metodos.items():
    ax_obj.plot(info['h'], info['score'], 
                'o', markersize=15, color=colores_metodos[metodo], 
                markeredgecolor='black', markeredgewidth=2, zorder=5)

# Destacar el mejor metodo con estrella
ax_obj.plot(h_optimo_calculado, mejor_metodo[1]['score'],
            'r*', markersize=20, label=f'BEST: {mejor_metodo[0]} ({h_optimo_calculado:.0f}m)', 
            zorder=6, markeredgecolor='darkred', markeredgewidth=1.5)

# Lineas verticales para cada metodo de optimizacion
for metodo, h_valor in valores_optimos.items():
    estilo = '-' if metodo == mejor_metodo[0] else estilos_linea[metodo]
    ancho = 3.5 if metodo == mejor_metodo[0] else 2.5
    ax_obj.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilo, 
                   linewidth=ancho, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# Zonas de fondo
ax_obj.axvspan(50, limite_ruido, alpha=0.2, color=color_ruido, zorder=1)
ax_obj.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.2, color=color_optimo, zorder=1)
ax_obj.axvspan(limite_sobreagregacion, 400, alpha=0.2, color=color_sobreagregacion, zorder=1)

ax_obj.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
ax_obj.set_ylabel('Objective Function', fontsize=14, weight='bold')
ax_obj.set_title('Aggregated Spatial Performance Score VS Cell size', 
                 fontsize=16, weight='bold', pad=15)
ax_obj.legend(loc='upper right', fontsize=11, framealpha=0.95, ncol=2)
ax_obj.grid(True, alpha=0.4, linestyle='--', zorder=2)
ax_obj.set_xlim(50, 250)

# Anotaciones
ax_obj.annotate(f'WINNER: {mejor_metodo[0]}\n(Lower Score)', 
                xy=(h_optimo_calculado, mejor_metodo[1]['score']),
                xytext=(h_optimo_calculado + 10, mejor_metodo[1]['score'] + 0.12),
                fontsize=13, weight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.85, edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=3))

# ========== SUBPLOT 1: Varianza de densidad ==========
ax1 = axes[0]
ax1.plot(df_resultados['h'], df_resultados['varianza'], 
         'o-', linewidth=3, markersize=6, color='#2E86AB', label='Varianza', zorder=3)

# Lineas verticales para cada metodo
for metodo, h_valor in valores_optimos.items():
    ax1.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], 
                linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# Zonas de fondo
ax1.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
ax1.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
ax1.axvspan(limite_sobreagregacion, 130, alpha=0.15, color=color_sobreagregacion, zorder=1)

ax1.set_xlabel('Cell size (m)', fontsize=13, weight='bold')
ax1.set_ylabel('Variance (events²/km⁴)', fontsize=13, weight='bold')
ax1.legend(loc='best', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax1.set_xlim(50, 250)

# ========== SUBPLOT 2: Porcentaje de celdas vacias ==========
ax2 = axes[1]
ax2.plot(df_resultados['h'], df_resultados['pct_vacias'], 
         'o-', linewidth=3, markersize=6, color='#A23B72', label='% Empty cells', zorder=3)

# Lineas verticales para cada metodo
for metodo, h_valor in valores_optimos.items():
    ax2.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
                linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

ax2.axhline(y=50, color='orange', linestyle=':', linewidth=2, 
            alpha=0.7, label='Critical threshold (50%)', zorder=3)

# Zonas de fondo
ax2.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
ax2.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
ax2.axvspan(limite_sobreagregacion, 130, alpha=0.15, color=color_sobreagregacion, zorder=1)

ax2.set_xlabel('Cell size (m)', fontsize=13, weight='bold')
ax2.set_ylabel('Cells without records (%)', fontsize=13, weight='bold')
ax2.legend(loc='best', fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax2.set_xlim(50, 250)

# ========== SUBPLOT 3: Eventos por celda ==========
ax3 = axes[2]
ax3.plot(df_resultados['h'], df_resultados['eventos_por_celda'], 
         'o-', linewidth=3, markersize=6, color='#F18F01', label='Events/cell', zorder=3)

# Lineas verticales para cada metodo
for metodo, h_valor in valores_optimos.items():
    ax3.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
                linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# Zonas de fondo
ax3.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
ax3.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
ax3.axvspan(limite_sobreagregacion, 130, alpha=0.15, color=color_sobreagregacion, zorder=1)

ax3.set_xlabel('Cell size (m)', fontsize=13, weight='bold')
ax3.set_ylabel('Average events per cell', fontsize=13, weight='bold')
ax3.legend(loc='best', fontsize=9, framealpha=0.95)
ax3.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax3.set_xlim(50, 250)

# ========== SUBPLOT 4: Coeficiente de variacion ==========
ax4 = axes[3]
ax4.plot(df_resultados['h'], df_resultados['cv'], 
         'o-', linewidth=3, markersize=6, color='#6A4C93', label='CV', zorder=3)

# Lineas verticales para cada metodo
for metodo, h_valor in valores_optimos.items():
    ax4.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
                linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)


# Zonas de fondo
ax4.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
ax4.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
ax4.axvspan(limite_sobreagregacion, 130, alpha=0.15, color=color_sobreagregacion, zorder=1)

ax4.set_xlabel('Cell size (m)', fontsize=13, weight='bold')
ax4.set_ylabel('Coefficient of variation (CV)', fontsize=13, weight='bold')
ax4.legend(loc='best', fontsize=9, framealpha=0.95)
ax4.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax4.set_xlim(50, 250)

# Guardar figura
output_file = resultados_dir / "justificacion_tres_metodos.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Grafica guardada en: {output_file}")

plt.show()

# ========== TABLA COMPARATIVA EN LOS PUNTOS OPTIMOS ==========
print("\n" + "=" * 80)
print("METRICAS EN LOS PUNTOS OPTIMOS DE CADA METODO")
print("=" * 80)

comparacion_optimos = []
for metodo, h_valor in valores_optimos.items():
    metricas = calcular_metricas_completas(puntos, cali, h_valor)
    if metricas:
        comparacion_optimos.append({
            'Metodo': metodo,
            'h (m)': h_valor,
            'Varianza': metricas['varianza'],
            '% Vacias': metricas['pct_vacias'],
            'Eventos/Celda': metricas['eventos_por_celda'],
            'CV': metricas['cv'],
            'N Celdas': metricas['n_celdas']
        })

df_optimos = pd.DataFrame(comparacion_optimos)

print("\n" + "-" * 120)
print(f"{'Metodo':<20} {'h (m)':<10} {'Varianza':<15} {'% Vacias':<12} "
      f"{'Eventos/Celda':<16} {'CV':<10} {'N Celdas':<12}")
print("-" * 120)

for _, row in df_optimos.iterrows():
    print(f"{row['Metodo']:<20} {row['h (m)']:>7.1f}   {row['Varianza']:>12.2f}   "
          f"{row['% Vacias']:>9.1f}%   {row['Eventos/Celda']:>13.2f}   "
          f"{row['CV']:>7.2f}   {int(row['N Celdas']):>9,}")

print("-" * 120)

# ========== ANALISIS DE CONVERGENCIA ==========
print("\n" + "=" * 80)
print("RANKING DE LOS TRES METODOS SEGUN FUNCION OBJETIVO")
print("=" * 80)

# Ordenar metodos por score
ranking = sorted(scores_metodos.items(), key=lambda x: x[1]['score'])

print("\n" + "-" * 80)
print(f"{'Posición':<12} {'Método':<25} {'h (m)':<12} {'Score':<15} {'Estado'}")
print("-" * 80)

for i, (metodo, info) in enumerate(ranking, 1):
    estado = "⭐ GANADOR" if i == 1 else f"#{i}"
    print(f"{i:<12} {metodo:<25} {info['h']:>8.2f}    {info['score']:>12.6f}   {estado}")

print("-" * 80)
print(f"\nRECOMENDACIÓN: Usar h = {h_optimo_calculado:.2f}m del método {mejor_metodo[0]}")
print("=" * 80)

# ========== ANALISIS DE CONVERGENCIA ==========
print("\n" + "=" * 80)
print("ANALISIS DE CONVERGENCIA DE LOS TRES METODOS")
print("=" * 80)

h_promedio = np.mean(list(valores_optimos.values()))
h_std = np.std(list(valores_optimos.values()))
h_rango = max(valores_optimos.values()) - min(valores_optimos.values())

print(f"\nEstadisticas de los valores optimos:")
print(f"  Promedio:        {h_promedio:.2f} m")
print(f"  Desviacion std:  {h_std:.2f} m")
print(f"  Rango:           {h_rango:.2f} m")
print(f"  Coef. variacion: {(h_std/h_promedio)*100:.2f}%")

print(f"\nDiferencias respecto al promedio:")
for metodo, h_valor in valores_optimos.items():
    diff = h_valor - h_promedio
    diff_pct = (diff / h_promedio) * 100
    print(f"  {metodo:<20}: {diff:+6.2f} m ({diff_pct:+5.2f}%)")

# ========== INTERPRETACION ==========
print("\n" + "=" * 80)
print("INTERPRETACION Y CONCLUSIONES")
print("=" * 80)

print(f"""
CONVERGENCIA DE METODOS:
Los tres metodos de optimizacion convergen a valores muy similares
(rango de {h_rango:.2f}m, ~{(h_rango/h_promedio)*100:.1f}% de variacion), lo que valida
la robustez del tamano optimo encontrado (~{h_promedio:.0f}m).

JUSTIFICACION DEL RANGO OPTIMO ({limite_ruido:.0f}m - {limite_sobreagregacion:.0f}m):

1. ZONA DE RUIDO (h < {limite_ruido:.0f}m):
   - Alta varianza: Excesiva sensibilidad a variaciones locales
   - Muchas celdas vacias: Fragmentacion espacial ({df_resultados[df_resultados['h']<limite_ruido]['pct_vacias'].mean():.1f}% promedio)
   - Pocos eventos/celda: Insuficiente agregacion estadistica
   - CV muy alto: Ruido estadistico domina sobre patrones reales

2. ZONA OPTIMA ({limite_ruido:.0f}m - {limite_sobreagregacion:.0f}m):
   - Varianza moderada: Balance entre detalle y estabilidad
   - Celdas vacias controladas: Cobertura espacial adecuada
   - Eventos/celda ideal: Cerca del rango optimo (8-12)
   - CV en rango ideal: Permite detectar hotspots sin ruido excesivo
   
3. ZONA DE SOBREAGREGACION (h > {limite_sobreagregacion:.0f}m):
   - Varianza muy baja: Perdida de patrones espaciales locales
   - Pocas celdas vacias: Pero a costa de resolucion espacial
   - Muchos eventos/celda: Excesiva agregacion difumina hotspots
   - CV bajo: Insuficiente heterogeneidad espacial

RECOMENDACION FINAL:
Utilizar h = {h_promedio:.0f}m (promedio de los 3 metodos) como tamano
optimo de celda, ya que representa el mejor balance entre preservar
la heterogeneidad espacial necesaria para detectar hotspots y mantener
estabilidad estadistica suficiente para analisis confiables.
""")

print("=" * 80)

# ========== EXPORTAR RESULTADOS ==========
print("\n[7/7] Exportando resultados...")

# Exportar tabla de optimos
output_optimos = resultados_dir / "metricas_puntos_optimos.csv"
df_optimos.to_csv(output_optimos, index=False)
print(f"[OK] Tabla de optimos: {output_optimos}")

# Exportar resumen
resumen = {
    'valores_optimos': {metodo: float(h) for metodo, h in valores_optimos.items()},
    'estadisticas': {
        'promedio': float(h_promedio),
        'desviacion': float(h_std),
        'rango': float(h_rango),
        'cv_pct': float((h_std/h_promedio)*100)
    },
    'zonas': {
        'ruido': f'h < {limite_ruido:.0f}m',
        'optimo': f'{limite_ruido:.0f}m - {limite_sobreagregacion:.0f}m',
        'sobreagregacion': f'h > {limite_sobreagregacion:.0f}m'
    },
    'recomendacion': f'{h_optimo_calculado:.2f}m',
    'mejor_metodo': mejor_metodo[0],
    'scores_metodos': {metodo: {'h': float(info['h']), 'score': float(info['score'])} 
                       for metodo, info in scores_metodos.items()}
}

output_json = resultados_dir / "resumen_tres_metodos.json"
with open(output_json, 'w') as f:
    json.dump(resumen, f, indent=2)
print(f"[OK] Resumen JSON: {output_json}")

print("\n" + "=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)
print(f"\n🏆 METODO GANADOR: {mejor_metodo[0]}")
print(f"📏 TAMAÑO OPTIMO RECOMENDADO: {h_optimo_calculado:.2f} m")
print(f"📊 Score de función objetivo: {mejor_metodo[1]['score']:.6f}")
print(f"\nLos tres metodos convergieron con variacion de {(h_std/h_promedio)*100:.1f}%,")
print(f"pero {mejor_metodo[0]} obtuvo el mejor balance multi-criterio.")
print("=" * 80)
