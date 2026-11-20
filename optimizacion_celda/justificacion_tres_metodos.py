import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import math
import pandas as pd
import warnings
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


df = pd.concat(datasets, ignore_index=True)
print(f"[OK] Total combinado: {len(df):,} registros")

# ========== CARGAR MAPA DE CALI ==========
print("\n[2/5] Cargando limites de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=2)
cali = cali.to_crs(3116)

# Mantener el mismo recorte y la misma geometría de celdas que georeferencia.py.
BBOX = (720414.01, 735029.27, 860572.64, 879817.30)  # minx, maxx, miny, maxy
bbox_gdf = gpd.GeoDataFrame(
    geometry=[box(BBOX[0], BBOX[2], BBOX[1], BBOX[3])], crs="EPSG:3116"
)
cali = gpd.overlay(cali, bbox_gdf, how="intersection")

area_cali = cali.geometry.union_all()


print("[OK] Mapa de Cali cargado")

# ========== CREAR GEODATAFRAME ==========
print("\n[3/5] Georeferenciando puntos...")
puntos = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['x'], df['y']), 
    crs="EPSG:4326"
)
puntos = puntos.to_crs(3116)
#puntos = puntos[puntos.within(area_cali)]
print(f"[OK] Puntos dentro de Cali: {len(puntos):,}")

n_eventos_total = len(puntos)

# ========== FUNCION DE METRICAS ==========
def calcular_metricas_completas(puntos_gdf, cali_gdf, h_m):
    """Calcula métricas usando la misma grilla hexagonal que el mapa."""
    lado_hex = float(h_m)
    ancho_hex = math.sqrt(3) * lado_hex
    paso_y = 1.5 * lado_hex
    minx, miny, maxx, maxy = cali_gdf.total_bounds
    minx -= ancho_hex
    miny -= lado_hex
    maxx += ancho_hex
    maxy += lado_hex

    centros = []
    for ix, x in enumerate(np.arange(minx, maxx + ancho_hex, ancho_hex)):
        offset_y = paso_y / 2.0 if ix % 2 == 1 else 0.0
        for y in np.arange(miny, maxy + paso_y, paso_y):
            centros.append((x, y + offset_y))

    def hexagono(cx, cy):
        coords = [(cx + lado_hex * math.cos(i * math.pi / 3.0),
                   cy + lado_hex * math.sin(i * math.pi / 3.0)) for i in range(6)]
        return Polygon(coords)

    celdas = [celda for cx, cy in centros
              for celda in [hexagono(cx, cy)] if celda.intersects(area_cali)]
    
    if len(celdas) == 0:
        return None
    
    grid = gpd.GeoDataFrame(geometry=celdas, crs=cali_gdf.crs)
    join = gpd.sjoin(puntos_gdf, grid, how="inner", predicate="within")
    conteo = join.groupby("index_right").size()
    
    conteo_completo = np.zeros(len(grid))
    conteo_completo[conteo.index] = conteo.values
    
    n_celdas = len(grid)
    eventos_por_celda = conteo_completo.sum() / n_celdas
    pct_vacias = (conteo_completo == 0).sum() / n_celdas * 100

    # Gini de eventos por celda: 0 indica reparto uniforme y valores mayores
    # indican una concentracion espacial mas desigual de los eventos.
    conteos_ordenados = np.sort(conteo_completo)
    suma_eventos = conteos_ordenados.sum()
    if suma_eventos > 0:
        posiciones = np.arange(1, n_celdas + 1)
        gini_eventos = (
            2 * np.sum(posiciones * conteos_ordenados) / (n_celdas * suma_eventos)
            - (n_celdas + 1) / n_celdas
        )
    else:
        gini_eventos = 0.0
    
    area_hex_km2 = (3 * math.sqrt(3) / 2 * lado_hex**2) / 1e6
    densidad = conteo_completo / area_hex_km2
    varianza = np.var(densidad)
    std_dev = np.std(densidad)
    mean_densidad = np.mean(densidad)
    cv = std_dev / (mean_densidad + 1e-6)
    
    return {
        'n_celdas': n_celdas,
        'eventos_por_celda': eventos_por_celda,
        'pct_vacias': pct_vacias,
        'gini_eventos': gini_eventos,
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
        valores_optimos['NSGA-II'] = round(float(data['celda']), 1)
        print(f"  [OK] NSGA-II: {valores_optimos['NSGA-II']:.1f} m")

# Metodos Bayesianos
metodos_file = resultados_dir / "metodos_optimizacion.json"
if metodos_file.exists():
    with open(metodos_file, 'r') as f:
        data = json.load(f)
        for metodo, info in data.items():
            valores_optimos[metodo] = round(float(info['h_optimo']), 1)
            print(f"  [OK] {metodo}: {valores_optimos[metodo]:.1f} m")

# ========== ANALISIS DE SENSIBILIDAD ==========
print("\n[5/5] Calculando metricas para diferentes tamanos de celda...")
print("Esto puede tomar varios minutos...\n")

# Rango de analisis: 50 a 250 metros con resolucion de 5m
tamanos = np.arange(50, 250, 5)
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
    0.30 * df_resultados['varianza_norm'] +
    0.25 * df_resultados['pct_vacias_norm'] +
    0.25 * df_resultados['cv_norm'] +
    0.20 * df_resultados['eventos_norm']
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

fig.suptitle('')

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


# # ========== GRAFICA INDIVIDUAL: FUNCION OBJETIVO ==========
# print("\n[Adicional] Generando grafica individual de Funcion Objetivo...")

# fig_obj, ax_obj_ind = plt.subplots(figsize=(18, 9))

# # Graficar funcion objetivo
# ax_obj_ind.plot(df_resultados['h'], df_resultados['funcion_objetivo'], 
#                 'o-', linewidth=4, markersize=8, color='#1a1a2e', label='Objective Function', zorder=3)

# # Marcar los tres valores optimos con sus scores
# for metodo, info in scores_metodos.items():
#     ax_obj_ind.plot(info['h'], info['score'], 
#                     'o', markersize=15, color=colores_metodos[metodo], 
#                     markeredgecolor='black', markeredgewidth=2, zorder=5)

# # Destacar el mejor metodo con estrella
# ax_obj_ind.plot(h_optimo_calculado, mejor_metodo[1]['score'],
#                 'r*', markersize=22, label=f'BEST: {mejor_metodo[0]} ({h_optimo_calculado:.0f}m)', 
#                 zorder=6, markeredgecolor='darkred', markeredgewidth=1.5)

# # Lineas verticales para cada metodo de optimizacion
# for metodo, h_valor in valores_optimos.items():
#     estilo = '-' if metodo == mejor_metodo[0] else estilos_linea[metodo]
#     ancho = 3.5 if metodo == mejor_metodo[0] else 2.5
#     ax_obj_ind.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilo, 
#                        linewidth=ancho, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# # Zonas de fondo
# ax_obj_ind.axvspan(50, limite_ruido, alpha=0.2, color=color_ruido, zorder=1)
# ax_obj_ind.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.2, color=color_optimo, zorder=1)
# ax_obj_ind.axvspan(limite_sobreagregacion, 250, alpha=0.2, color=color_sobreagregacion, zorder=1)

# ax_obj_ind.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
# ax_obj_ind.set_ylabel('Objective Function', fontsize=14, weight='bold')
# ax_obj_ind.set_title('')
# ax_obj_ind.legend(loc='upper right', fontsize=14, framealpha=0.95, ncol=2)
# ax_obj_ind.grid(True, alpha=0.4, linestyle='--', zorder=2)
# ax_obj_ind.set_xlim(50, 250)

# # Anotaciones
# ax_obj_ind.annotate(f'WINNER: {mejor_metodo[0]}\n(Lower Score)', 
#                     xy=(h_optimo_calculado, mejor_metodo[1]['score']),
#                     xytext=(h_optimo_calculado + 10, mejor_metodo[1]['score'] + 0.12),
#                     fontsize=14, weight='bold', color='darkred',
#                     bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.85, edgecolor='darkred', linewidth=2),
#                     arrowprops=dict(arrowstyle='->', color='darkred', lw=3))

# # Guardar figura individual
# output_funcion_obj = resultados_dir / "funcion_objetivo_individual.png"
# fig_obj.savefig(output_funcion_obj, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[OK] Grafica de funcion objetivo guardada en: {output_funcion_obj}")
# plt.close(fig_obj)

# # ========== GRAFICA INDIVIDUAL: VARIANZA ==========
# print("[Adicional] Generando grafica individual de Varianza...")

# fig_varianza, ax_var = plt.subplots(figsize=(12, 8))

# # Graficar varianza
# ax_var.plot(df_resultados['h'], df_resultados['varianza'], 
#             'o-', linewidth=3, markersize=6, color='#2E86AB', label='Varianza', zorder=3)

# # Lineas verticales para cada metodo
# for metodo, h_valor in valores_optimos.items():
#     ax_var.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], 
#                    linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# # Zonas de fondo
# ax_var.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
# ax_var.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
# ax_var.axvspan(limite_sobreagregacion, 250, alpha=0.15, color=color_sobreagregacion, zorder=1)

# ax_var.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
# ax_var.set_ylabel('Variance (events²/km⁴)', fontsize=14, weight='bold')
# ax_var.set_title('')
# ax_var.legend(loc='best', fontsize=14, framealpha=0.95)
# ax_var.grid(True, alpha=0.3, linestyle='--', zorder=2)
# ax_var.set_xlim(50, 250)

# # Guardar figura individual
# output_varianza = resultados_dir / "varianza_individual.png"
# fig_varianza.savefig(output_varianza, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[OK] Grafica de varianza guardada en: {output_varianza}")
# plt.close(fig_varianza)

# # ========== GRAFICA INDIVIDUAL: PORCENTAJE DE CELDAS VACIAS ==========
# print("[Adicional] Generando grafica individual de Porcentaje de celdas vacias...")

# fig_vacias, ax_vac = plt.subplots(figsize=(12, 8))

# # Graficar porcentaje de celdas vacias
# ax_vac.plot(df_resultados['h'], df_resultados['pct_vacias'], 
#             'o-', linewidth=3, markersize=6, color='#A23B72', label='% Empty cells', zorder=3)

# # Lineas verticales para cada metodo
# for metodo, h_valor in valores_optimos.items():
#     ax_vac.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
#                    linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# # Linea de umbral critico
# ax_vac.axhline(y=50, color='orange', linestyle=':', linewidth=2, 
#                alpha=0.7, label='Critical threshold (50%)', zorder=3)

# # Zonas de fondo
# ax_vac.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
# ax_vac.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
# ax_vac.axvspan(limite_sobreagregacion, 250, alpha=0.15, color=color_sobreagregacion, zorder=1)

# ax_vac.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
# ax_vac.set_ylabel('Cells without records (%)', fontsize=14, weight='bold')
# ax_vac.legend(loc='best', fontsize=14, framealpha=0.95)
# ax_vac.grid(True, alpha=0.3, linestyle='--', zorder=2)
# ax_vac.set_xlim(50, 250)

# # Guardar figura individual
# output_vacias = resultados_dir / "celdas_eventos_individual.png"
# fig_vacias.savefig(output_vacias, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[OK] Grafica de celdas vacias guardada en: {output_vacias}")
# plt.close(fig_vacias)



# # ========== SUBPLOT 3: Eventos por celda ==========
# fig_eventos, ax_even = plt.subplots(figsize=(12, 8))

# ax_even.plot(df_resultados['h'], df_resultados['eventos_por_celda'], 
#          'o-', linewidth=3, markersize=6, color='#F18F01', label='Events/cell', zorder=3)

# # Lineas verticales para cada metodo
# for metodo, h_valor in valores_optimos.items():
#     ax_even.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
#                 linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# # Zonas de fondo
# ax_even.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
# ax_even.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
# ax_even.axvspan(limite_sobreagregacion, 250, alpha=0.15, color=color_sobreagregacion, zorder=1)

# ax_even.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
# ax_even.set_ylabel('Average events per cell', fontsize=14, weight='bold')
# ax_even.set_title('')
# ax_even.legend(loc='best', fontsize=14, framealpha=0.95)
# ax_even.grid(True, alpha=0.3, linestyle='--', zorder=2)
# ax_even.set_xlim(50, 250)

# # Guardar figura individual
# output_eventos = resultados_dir / "eventos_celda_individual.png"
# fig_eventos.savefig(output_eventos, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[OK] Grafica de eventos por celda guardada en: {output_eventos}")
# plt.close(fig_eventos)

# # ========== GRAFICA INDIVIDUAL: COEFICIENTE DE VARIACION ==========
# print("[Adicional] Generando grafica individual de Coeficiente de variacion...")

# fig_cv, ax_cv = plt.subplots(figsize=(12, 8))

# # Graficar coeficiente de variacion
# ax_cv.plot(df_resultados['h'], df_resultados['cv'], 
#            'o-', linewidth=3, markersize=6, color='#6A4C93', label='CV', zorder=3)

# # Lineas verticales para cada metodo
# for metodo, h_valor in valores_optimos.items():
#     ax_cv.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo],
#                   linewidth=2.5, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.8, zorder=4)

# # Zonas de fondo
# ax_cv.axvspan(50, limite_ruido, alpha=0.15, color=color_ruido, zorder=1)
# ax_cv.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.15, color=color_optimo, zorder=1)
# ax_cv.axvspan(limite_sobreagregacion, 250, alpha=0.15, color=color_sobreagregacion, zorder=1)

# ax_cv.set_xlabel('Cell size (m)', fontsize=14, weight='bold')
# ax_cv.set_ylabel('Coefficient of variation (CV)', fontsize=14, weight='bold')
# ax_cv.set_title('')
# ax_cv.legend(loc='best', fontsize=14, framealpha=0.95)
# ax_cv.grid(True, alpha=0.3, linestyle='--', zorder=2)
# ax_cv.set_xlim(50, 250)

# # Guardar figura individual
# output_cv = resultados_dir / "coeficiente_variacion_individual.png"
# fig_cv.savefig(output_cv, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[OK] Grafica de coeficiente de variacion guardada en: {output_cv}")
# plt.close(fig_cv)



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
    estado = " GANADOR" if i == 1 else f"#{i}"
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

# ========== EXPORTAR RESULTADOS ==========
print("\n[7/7] Exportando resultados...")

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
    'recomendacion': round(float(h_optimo_calculado), 1),
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
print(f"\nMETODO GANADOR: {mejor_metodo[0]}")
print(f" TAMAÑO OPTIMO RECOMENDADO: {h_optimo_calculado:.2f} m")

# ========== GRAFICA COMBINADA 2x2 ==========

fig_width_in = 7.16   # dos columnas (o 3.5 para una)
fig_height_in = fig_width_in * 0.65   # altura ajustada

# Compartir eje X para alinear y eliminar etiquetas redundantes
fig_combinada, axes_4 = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in),
                                     sharex=True, sharey=False)

ax_cv_c, ax_even_c = axes_4[0, 0], axes_4[0, 1]
ax_vac_c, ax_var_c = axes_4[1, 0], axes_4[1, 1]

# ── Etiquetas de subfiguras centradas debajo ──
for ax, label in zip([ax_cv_c, ax_even_c], ['(a)', '(b)']):
    ax.text(0.5, -0.08, label, transform=ax.transAxes,
            fontsize=8, fontname='Times New Roman', ha='center', va='center')
    
for ax, label in zip([ax_vac_c, ax_var_c], ['(c)', '(d)']):
    ax.text(0.5, -0.24, label, transform=ax.transAxes,
            fontsize=8, fontname='Times New Roman', ha='center', va='center')

# ── Eliminar etiquetas del eje X en la fila superior ──
for ax in [ax_cv_c, ax_even_c]:
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelbottom=False)

# ── Reducir espaciado entre subplots ──
plt.subplots_adjust(hspace=0.05, wspace=0.06)

# (El resto del código de cada subplot se mantiene igual, pero sin repetir set_xlabel en los superiores)
# Ejemplo para el primero:
ax_cv_c.plot(df_resultados['h'], df_resultados['cv'], 'o-', linewidth=1, markersize=5, color='#6A4C93', label='CV', zorder=3)
for metodo, h_valor in valores_optimos.items():
    ax_cv_c.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], linewidth=1, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.6, zorder=4)
ax_cv_c.axvspan(50, limite_ruido, alpha=0.08, color=color_ruido, zorder=1)
ax_cv_c.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.08, color=color_optimo, zorder=1)
ax_cv_c.axvspan(limite_sobreagregacion, 250, alpha=0.08, color=color_sobreagregacion, zorder=1)
ax_cv_c.set_ylabel('Coefficient of variation (CV)', fontsize=8)
ax_cv_c.legend(loc='best', fontsize=8, framealpha=0.9)
ax_cv_c.tick_params(axis='both', labelsize=8)
ax_cv_c.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax_cv_c.set_xlim(50, 250)

# Subplot 2 (arriba derecha)
ax_even_c.plot(df_resultados['h'], df_resultados['eventos_por_celda'], 'o-', linewidth=1, markersize=5, color='#F18F01', label='Events/cell', zorder=3)
for metodo, h_valor in valores_optimos.items():
    ax_even_c.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], linewidth=1, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.6, zorder=4)
ax_even_c.axvspan(50, limite_ruido, alpha=0.08, color=color_ruido, zorder=1)
ax_even_c.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.08, color=color_optimo, zorder=1)
ax_even_c.axvspan(limite_sobreagregacion, 250, alpha=0.08, color=color_sobreagregacion, zorder=1)
ax_even_c.set_ylabel('Average events per cell', fontsize=8)
ax_even_c.legend(loc='best', fontsize=8, framealpha=0.9)
ax_even_c.tick_params(axis='both', labelsize=8)
ax_even_c.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax_even_c.set_xlim(50, 250)

# Subplot 3 (abajo izquierda)
ax_vac_c.plot(df_resultados['h'], df_resultados['pct_vacias'], 'o-', linewidth=1, markersize=5, color='#A23B72', label='% Empty cells', zorder=3)
for metodo, h_valor in valores_optimos.items():
    ax_vac_c.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], linewidth=1, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.6, zorder=4)
ax_vac_c.axvspan(50, limite_ruido, alpha=0.08, color=color_ruido, zorder=1)
ax_vac_c.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.08, color=color_optimo, zorder=1)
ax_vac_c.axvspan(limite_sobreagregacion, 250, alpha=0.08, color=color_sobreagregacion, zorder=1)
ax_vac_c.set_xlabel('Cell size (m)', fontsize=8)
ax_vac_c.set_ylabel('Cells without records (%)', fontsize=8)
ax_vac_c.legend(loc='best', fontsize=8, framealpha=0.9)
ax_vac_c.tick_params(axis='both', labelsize=8)
ax_vac_c.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax_vac_c.set_xlim(50, 250)

# Subplot 4 (abajo derecha)
ax_var_c.plot(df_resultados['h'], df_resultados['varianza'], 'o-', linewidth=1, markersize=5, color='#2E86AB', label='Variance', zorder=3)
for metodo, h_valor in valores_optimos.items():
    ax_var_c.axvline(x=h_valor, color=colores_metodos[metodo], linestyle=estilos_linea[metodo], linewidth=1, label=f'{metodo} ({h_valor:.0f}m)', alpha=0.6, zorder=4)
ax_var_c.axvspan(50, limite_ruido, alpha=0.08, color=color_ruido, zorder=1)
ax_var_c.axvspan(limite_ruido, limite_sobreagregacion, alpha=0.08, color=color_optimo, zorder=1)
ax_var_c.axvspan(limite_sobreagregacion, 250, alpha=0.08, color=color_sobreagregacion, zorder=1)
ax_var_c.set_xlabel('Cell size (m)', fontsize=8)
ax_var_c.set_ylabel('Variance (events²/km⁴)', fontsize=8)
ax_var_c.legend(loc='best', fontsize=8, framealpha=0.9)
ax_var_c.tick_params(axis='both', labelsize=8)
ax_var_c.grid(True, alpha=0.3, linestyle='--', zorder=2)
ax_var_c.set_xlim(50, 250)

# Ajuste final de espaciado y guardado
fig_combinada.tight_layout(pad=0.5)   # margen compacto
output_combinada = resultados_dir / "metricas_combinadas_2x2.pdf"
fig_combinada.savefig(output_combinada, format='pdf', bbox_inches='tight')
plt.close(fig_combinada)


def graficar_curvas_tendencia_metricas(puntos_gdf, cali_gdf, output_path=None):
    """Guarda las métricas de cobertura, intensidad y concentración por h.

    Calcula las métricas exactamente para h = 50, 100, 160, 200, 238 y 250 m.
    Usa ejes Y independientes para que las tres escalas sean legibles.

    Args:
        puntos_gdf: Eventos georreferenciados en el CRS de ``cali_gdf``.
        cali_gdf: Polígono de Cali utilizado para construir las grillas.
        output_path: Ruta opcional de salida. Por defecto se guarda junto a
            este script como ``curvas_tendencia_metricas.png``.

    Returns:
        Path: Ruta de la imagen guardada.
    """
    tamanos_h = np.array([50, 100, 160, 200, 238, 260], dtype=float)
    filas = []
    for h in tamanos_h:
        metricas = calcular_metricas_completas(puntos_gdf, cali_gdf, h)
        if metricas is None:
            raise ValueError(f"No se pudo construir la grilla para h={h:g} m.")
        filas.append(metricas)

    pct_vacias = np.array([fila['pct_vacias'] for fila in filas])
    eventos_celda = np.array([fila['eventos_por_celda'] for fila in filas])
    gini_eventos = np.array([fila['gini_eventos'] for fila in filas])

    fig, ax_vacias = plt.subplots(figsize=(7.16, 7.16 * 0.75))
    ax_eventos = ax_vacias.twinx()
    ax_gini = ax_vacias.twinx()
    ax_gini.spines['right'].set_position(('outward', 48))

    linea_vacias, = ax_vacias.plot(
        tamanos_h, pct_vacias, color='#A23B72', marker='o', linewidth=2,
        label='% de celdas vacías',
    )
    linea_eventos, = ax_eventos.plot(
        tamanos_h, eventos_celda, color='#F18F01', marker='s', linewidth=2,
        label='Promedio de eventos por celda',
    )
    linea_gini, = ax_gini.plot(
        tamanos_h, gini_eventos, color='#2E86AB', marker='^', linewidth=2,
        label='Gini de eventos por celda', 
    )

    ax_vacias.set_xlabel('Tamaño de celda h (m)', fontsize=11)
    ax_vacias.set_ylabel('% de celdas vacías', fontsize=11, color=linea_vacias.get_color())
    ax_eventos.set_ylabel('Eventos promedio por celda', fontsize=11, color=linea_eventos.get_color())
    ax_gini.set_ylabel('Índice Gini (eventos por celda)', fontsize=11, color=linea_gini.get_color())
    ax_vacias.tick_params(axis='y', colors=linea_vacias.get_color())
    ax_eventos.tick_params(axis='y', colors=linea_eventos.get_color())
    ax_gini.tick_params(axis='y', colors=linea_gini.get_color())

    ax_vacias.set_xticks(tamanos_h, [f'{h:g}' for h in tamanos_h])
    ax_vacias.set_xlim(40, 260)
    ax_vacias.grid(True, axis='both', alpha=0.3, linestyle='--')
    ax_vacias.axvline(160, color='#333333', linestyle=':', linewidth=1.5)

    lineas = [linea_vacias, linea_eventos, linea_gini]
    leyenda = ax_gini.legend(
        lineas,
        [linea.get_label() for linea in lineas],
        loc='upper center',
        framealpha=0.95,
        facecolor='white',
    )
    leyenda.set_zorder(200)
    fig.tight_layout()

    if output_path is None:
        output_path = Path(__file__).resolve().parent / 'curvas_tendencia_metricas.png'
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Gráfica de curvas de tendencia guardada en: {output_path}")
    return output_path


graficar_curvas_tendencia_metricas(puntos, cali)
