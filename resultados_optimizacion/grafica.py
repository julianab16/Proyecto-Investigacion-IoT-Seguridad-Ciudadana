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
print(" " * 20 + "ANÁLISIS DE SENSIBILIDAD - TAMAÑO DE CELDA")
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

# Buscar archivos en data_base/
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
            print(f"  [OK] {archivo}: {len(df):,} registros")
            archivos_cargados += 1
    else:
        print(f"  [-] No encontrado: {archivo}")

if archivos_cargados == 0:
    print("\n[ERROR] No se encontró ningún archivo CSV válido")
    exit()

# Combinar datasets
df = pd.concat(datasets, ignore_index=True)
print(f"\n[OK] Total combinado: {len(df):,} registros")

# ========== CARGAR MAPA DE CALI ==========
print("\n[2/4] Cargando límites de Cali...")
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
    
    return {
        'n_celdas': n_celdas,
        'eventos_por_celda': eventos_por_celda,
        'pct_vacias': pct_vacias,
        'varianza': varianza,
        'std_dev': std_dev,
        'cv': cv
    }

# ========== ANÁLISIS DE SENSIBILIDAD ==========
print("\n[4/4] Calculando métricas para diferentes tamaños de celda...")
print("Esto puede tomar varios minutos...\n")

# Rango de análisis: 50 a 130 metros
tamaños = np.arange(50, 131, 5)  # De 50 a 130 en pasos de 5 metros
resultados = []

for i, h in enumerate(tamaños):
    print(f"  Evaluando h = {h:.0f} m... ({i+1}/{len(tamaños)})", end='\r')
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

# ========== CARGAR VALORES ÓPTIMOS ==========
resultados_dir = Path(__file__).resolve().parent

# Cargar valores óptimos de los diferentes métodos
valores_optimos = {}

# NSGA-II
nsga_file = resultados_dir / "nsgacelda.json"
if nsga_file.exists():
    with open(nsga_file, 'r') as f:
        data = json.load(f)
        valores_optimos['NSGA-II'] = data['celda']

# Métodos bayesianos
metodos_file = resultados_dir / "metodos_optimizacion.json"
if metodos_file.exists():
    with open(metodos_file, 'r') as f:
        data = json.load(f)
        for metodo, info in data.items():
            valores_optimos[metodo] = info['h_optimo']

# Calcular promedio de los valores optimos
if valores_optimos:
    h_optimo = np.mean(list(valores_optimos.values()))
    print(f"\n[OK] Valor optimo promedio: {h_optimo:.1f} m")
else:
    h_optimo = 125.0  # Valor por defecto
    print(f"\n[WARN] Usando valor optimo por defecto: {h_optimo:.1f} m")

# ========== VISUALIZACIÓN ==========
print("\n[5/5] Generando gráfica de justificación...")

# Crear figura con 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Análisis de Sensibilidad: Trade-off entre Ruido y Sobreagregación Espacial',
             fontsize=16, weight='bold', y=0.995)

# Colores para las zonas
color_ruido = '#ffcccc'  # Rojo claro
color_optimo = '#ccffcc'  # Verde claro
color_sobreagregacion = '#cce5ff'  # Azul claro

# ========== SUBPLOT 1: Varianza de densidad ==========
ax1 = axes[0, 0]
ax1.plot(df_resultados['h'], df_resultados['varianza'], 
         'o-', linewidth=2.5, markersize=7, color='#2E86AB', label='Varianza')
ax1.axvline(x=h_optimo, color='darkgreen', linestyle='--', linewidth=2.5, 
            label=f'Óptimo ({h_optimo:.0f}m)', zorder=5)
ax1.axvspan(50, 80, alpha=0.2, color=color_ruido, label='Zona de ruido')
ax1.axvspan(80, h_optimo-5, alpha=0.15, color=color_optimo)
ax1.axvspan(h_optimo+5, 130, alpha=0.2, color=color_sobreagregacion, label='Sobreagregación')

ax1.set_xlabel('Tamaño de celda (m)', fontsize=12, weight='bold')
ax1.set_ylabel('Varianza de densidad (eventos²/km⁴)', fontsize=12, weight='bold')
ax1.set_title('(a) Varianza de la densidad de eventos', fontsize=13, weight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(50, 130)

# ========== SUBPLOT 2: Porcentaje de celdas vacías ==========
ax2 = axes[0, 1]
ax2.plot(df_resultados['h'], df_resultados['pct_vacias'], 
         'o-', linewidth=2.5, markersize=7, color='#A23B72', label='% Celdas vacías')
ax2.axvline(x=h_optimo, color='darkgreen', linestyle='--', linewidth=2.5,
            label=f'Óptimo ({h_optimo:.0f}m)', zorder=5)
ax2.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, 
            alpha=0.7, label='Umbral crítico (50%)')
ax2.axvspan(50, 80, alpha=0.2, color=color_ruido)
ax2.axvspan(80, h_optimo-5, alpha=0.15, color=color_optimo)
ax2.axvspan(h_optimo+5, 130, alpha=0.2, color=color_sobreagregacion)

ax2.set_xlabel('Tamaño de celda (m)', fontsize=12, weight='bold')
ax2.set_ylabel('Porcentaje de celdas sin registros (%)', fontsize=12, weight='bold')
ax2.set_title('(b) Celdas vacías (pérdida de cobertura)', fontsize=13, weight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(50, 130)

# ========== SUBPLOT 3: Eventos por celda ==========
ax3 = axes[1, 0]
ax3.plot(df_resultados['h'], df_resultados['eventos_por_celda'], 
         'o-', linewidth=2.5, markersize=7, color='#F18F01', label='Eventos/celda')
ax3.axvline(x=h_optimo, color='darkgreen', linestyle='--', linewidth=2.5,
            label=f'Óptimo ({h_optimo:.0f}m)', zorder=5)
ax3.axhspan(8, 12, alpha=0.2, color='lightgreen', label='Rango ideal (8-12)')
ax3.axvspan(50, 80, alpha=0.2, color=color_ruido)
ax3.axvspan(80, h_optimo-5, alpha=0.15, color=color_optimo)
ax3.axvspan(h_optimo+5, 130, alpha=0.2, color=color_sobreagregacion)

ax3.set_xlabel('Tamaño de celda (m)', fontsize=12, weight='bold')
ax3.set_ylabel('Eventos promedio por celda', fontsize=12, weight='bold')
ax3.set_title('(c) Densidad promedio de eventos', fontsize=13, weight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(50, 130)

# ========== SUBPLOT 4: Coeficiente de variación ==========
ax4 = axes[1, 1]
ax4.plot(df_resultados['h'], df_resultados['cv'], 
         'o-', linewidth=2.5, markersize=7, color='#6A4C93', label='CV')
ax4.axvline(x=h_optimo, color='darkgreen', linestyle='--', linewidth=2.5,
            label=f'Óptimo ({h_optimo:.0f}m)', zorder=5)
ax4.axhspan(1.0, 2.5, alpha=0.2, color='lightgreen', label='Rango ideal (1.0-2.5)')
ax4.axvspan(50, 80, alpha=0.2, color=color_ruido)
ax4.axvspan(80, h_optimo-5, alpha=0.15, color=color_optimo)
ax4.axvspan(h_optimo+5, 130, alpha=0.2, color=color_sobreagregacion)

ax4.set_xlabel('Tamaño de celda (m)', fontsize=12, weight='bold')
ax4.set_ylabel('Coeficiente de variación (CV)', fontsize=12, weight='bold')
ax4.set_title('(d) Heterogeneidad espacial (detección de hotspots)', fontsize=13, weight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(50, 130)

plt.tight_layout()

# Guardar figura
output_file = resultados_dir / "justificacion_tamano_celda.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Grafica guardada en: {output_file}")

plt.show()

# ========== ANÁLISIS CUANTITATIVO ==========
print("\n" + "=" * 80)
print("ANÁLISIS CUANTITATIVO DEL TRADE-OFF")
print("=" * 80)

# Encontrar valores en el rango óptimo
idx_optimo = (df_resultados['h'] - h_optimo).abs().idxmin()
metricas_optimo = df_resultados.loc[idx_optimo]

# Comparar con extremos
metricas_50 = df_resultados[df_resultados['h'] == 50].iloc[0]
metricas_130 = df_resultados[df_resultados['h'] == 130].iloc[0]

print("\nCOMPARACIÓN DE ESCENARIOS:")
print("-" * 80)
print(f"{'Métrica':<30} {'h=50m (ruido)':<20} {'h={:.0f}m (óptimo)':<20} {'h=130m (sobreagr.)':<20}".format(h_optimo))
print("-" * 80)

metricas_comparar = [
    ('Varianza', 'varianza', '{:.1f}'),
    ('% Celdas vacías', 'pct_vacias', '{:.1f}%'),
    ('Eventos/celda', 'eventos_por_celda', '{:.2f}'),
    ('Coef. variación', 'cv', '{:.2f}'),
    ('N° celdas', 'n_celdas', '{:.0f}')
]

for nombre, campo, formato in metricas_comparar:
    val_50 = formato.format(metricas_50[campo])
    val_opt = formato.format(metricas_optimo[campo])
    val_130 = formato.format(metricas_130[campo])
    print(f"{nombre:<30} {val_50:<20} {val_opt:<20} {val_130:<20}")

print("-" * 80)

# ========== INTERPRETACIÓN ==========
print("\nINTERPRETACIÓN DEL TRADE-OFF:")
print("-" * 80)
print(f"""
1. ZONA DE RUIDO (h < 80m):
   - Alta varianza: Excesiva sensibilidad a variaciones locales
   - Muchas celdas vacías: Fragmentación espacial, dificulta análisis
   - Bajo eventos/celda: Insuficiente agregación estadística
   - CV elevado: Ruido estadístico domina sobre patrones reales

2. ZONA ÓPTIMA (h ≈ {h_optimo:.0f}m):
   - Varianza moderada: Balance entre detalle y estabilidad
   - Celdas vacías controladas: Cobertura espacial adecuada
   - Eventos/celda adecuado: Suficiente agregación para análisis robusto
   - CV en rango ideal: Permite detectar hotspots sin ruido excesivo

3. ZONA DE SOBREAGREGACIÓN (h > {h_optimo+10:.0f}m):
   - Varianza muy baja: Pérdida de patrones espaciales locales
   - Pocas celdas vacías: Pero a costa de resolución espacial
   - Alto eventos/celda: Excesiva agregación difumina hotspots
   - CV bajo: Insuficiente heterogeneidad, homogeneización artificial
""")
print("-" * 80)

print("\n[OK] ANALISIS COMPLETADO")
print(f"\nConclusión: El tamaño optimo de {h_optimo:.0f}m representa el mejor balance")
print("entre preservar la heterogeneidad espacial necesaria para detectar hotspots")
print("y mantener estabilidad estadística suficiente para análisis confiables.")
print("=" * 80)

# ========== EXPORTAR RESULTADOS ==========
output_csv = resultados_dir / "analisis_sensibilidad.csv"
df_resultados.to_csv(output_csv, index=False)
print(f"\n[OK] Datos exportados a: {output_csv}")

# Exportar resumen
resumen = {
    'h_optimo': float(h_optimo),
    'rango_analizado': [50, 130],
    'n_evaluaciones': len(df_resultados),
    'metricas_optimo': {
        'varianza': float(metricas_optimo['varianza']),
        'pct_vacias': float(metricas_optimo['pct_vacias']),
        'eventos_por_celda': float(metricas_optimo['eventos_por_celda']),
        'cv': float(metricas_optimo['cv']),
        'n_celdas': int(metricas_optimo['n_celdas'])
    },
    'interpretacion': {
        'ruido': 'h < 80m: Alta varianza, muchas celdas vacías, CV elevado',
        'optimo': f'h ≈ {h_optimo:.0f}m: Balance entre detalle y estabilidad',
        'sobreagregacion': f'h > {h_optimo+10:.0f}m: Pérdida de heterogeneidad espacial'
    }
}

output_json = resultados_dir / "resumen_justificacion.json"
with open(output_json, 'w') as f:
    json.dump(resumen, f, indent=2)

print(f"[OK] Resumen exportado a: {output_json}\n")
