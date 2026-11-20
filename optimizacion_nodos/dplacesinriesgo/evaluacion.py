"""
Evaluación de Eficiencia en Distribución de Sensores
====================================================
Implementa la cadena metodológica completa (RQ1–RQ7):

  RQ1 - Risk-Weighted Coverage (RWC)          -> métrica principal
  RQ2 - Risk-Weighted Distance (DRW)          -> métrica secundaria
  RQ3 - Monte Carlo (baseline aleatorio)      -> validación aleatoria
  RQ4 - Wilcoxon signed-rank (pareado)        -> significancia estadística
  RQ5 - Boxplot + IC95% (t-Student)           -> robustez
  RQ6 - Correlación de Spearman               -> alineación espacial
  RQ7 - Global Coverage                       -> cobertura del área completa

CORRECCIONES respecto a la versión anterior:
  1. La grilla de riesgo se recorta y traslada EXACTAMENTE igual que en
     dplace_main.py (mismo CUADRO_X_MIN/Y_MIN/LADO), para que quede en el
     mismo sistema de coordenadas locales (0..AREA_SIZE) que los sensores
     guardados en los CSV. Antes se comparaban sensores en coordenadas
     locales contra centroides de celdas en coordenadas UTM crudas, lo que
     invalidaba todas las métricas.
  2. Las rutas de los CSV de sensores apuntan a OUTPUT_DIR (la misma carpeta
     que usa dplace_main.py), en vez de una ruta relativa frágil.
  3. Se agrega el test de Wilcoxon signed-rank pareado (RQ4), usando las
     N_SIMULATIONS réplicas de preprocesamiento (sim{i}_nodos_pre.csv) como
     muestras pareadas de "la propuesta" contra un baseline aleatorio
     generado con la misma semilla por réplica.
  4. El boxplot de la "Propuesta" ahora usa la dispersión real de esas
     réplicas, en vez de una sola línea horizontal.
  5. El intervalo de confianza al 95% usa el cuantil t de Student (no la
     aproximación normal z=1.96), tal como pide el documento de referencia.
  6. La asignación sensor -> celda usa polygon.contains real (spatial join
     de geopandas), no solo "centroide más cercano" (Voronoi), para ser
     consistente con el resto del pipeline (preprocessing_phase.py).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely.affinity import translate
from scipy.stats import spearmanr, wilcoxon, t as t_dist
from scipy.spatial import cKDTree

# ==========================
# PARÁMETROS GENERALES
# (deben coincidir EXACTAMENTE con dplace_main.py)
# ==========================
AREA_SIZE = 5900          # Lado del área en metros (5900 x 5900)
NUM_SENSORS = 1000        # Número de sensores a distribuir
NUM_MC_SIMULATIONS = 1000 # Número de replicaciones Monte Carlo
RANDOM_SEED = 42          # Semilla para reproducibilidad
N_SIMULATIONS = 15        # Número de corridas de preprocesamiento (para Wilcoxon)

CUADRO_X_MIN = 724900
CUADRO_Y_MIN = 868100
CUADRO_X_MAX = CUADRO_X_MIN + AREA_SIZE
CUADRO_Y_MAX = CUADRO_Y_MIN + AREA_SIZE

# --- RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputConRiesgo15clusters')  # misma carpeta de dplace_main.py

SENSORS_CSV = os.path.join(OUTPUT_DIR, "final_nodos.csv")
SIM_NODOS_PREFIX = os.path.join(OUTPUT_DIR, "sim{}_devices_poisson.csv")

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RISK_MAP_GEOJSON = os.path.join(PROJECT_ROOT, "images", "grid_cali.geojson")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "evaluacion_resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)



# ==========================
# 0. NORMALIZACIÓN DE LA GRILLA (idéntica a dplace_main.py)
# ==========================
def get_normalized_grid(grid_gdf, x_min, y_min, x_max, y_max):
    """
    Recorta la grilla al cuadro de estudio y traslada las geometrías para
    que queden en el sistema de coordenadas LOCAL (0..AREA_SIZE), el mismo
    que usan los CSV de sensores generados por dplace_main.py.
    """
    cuadro = box(x_min, y_min, x_max, y_max)
    gdf_cuadro = grid_gdf[grid_gdf.geometry.intersects(cuadro)].copy()

    dx, dy = x_min, y_min
    gdf_cuadro['geometry'] = gdf_cuadro.geometry.apply(
        lambda geom: translate(geom, xoff=-dx, yoff=-dy)
    )
    return gdf_cuadro


# ==========================
# 1. CARGA DE DATOS
# ==========================
def load_or_generate_sensors(csv_path=SENSORS_CSV, num_sensors=NUM_SENSORS, area_size=AREA_SIZE):
    """Carga los sensores finales desde final_nodos.csv (generado por dplace_main.py)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        sensors = df[['x', 'y']].values
        print(f"Sensores cargados desde {csv_path}: {sensors.shape[0]} puntos")
        return sensors


    rng = np.random.default_rng(RANDOM_SEED)
    return rng.uniform(0, area_size, size=(num_sensors, 2))


def load_all_sim_replicas(prefix=SIM_NODOS_PREFIX, n_sims=N_SIMULATIONS):
    """
    Carga las N_SIMULATIONS réplicas de preprocesamiento (antes de colapsar
    con K-Means en la Fase 2). Se usan como muestras PAREADAS de "la
    propuesta" para el test de Wilcoxon y para el boxplot con dispersión real.
    """
    replicas = []
    for i in range(1, n_sims + 1):
        path = prefix.format(i)
        if not os.path.exists(path):
            print(f"⚠ No se encontró {path}, se omite esta réplica.")
            continue
        df = pd.read_csv(path)
        replicas.append(df[['x', 'y']].values)

    if not replicas:
        print(
            "⚠ ADVERTENCIA: no se encontró ninguna réplica sim{i}_nodos_pre.csv. "
            "El test de Wilcoxon y el boxplot de 'Propuesta' no podrán calcularse "
            "con múltiples muestras; se omitirán esos análisis."
        )
    else:
        print(f"{len(replicas)} réplicas de preprocesamiento cargadas para Wilcoxon/boxplot.")
    return replicas


def load_or_generate_risk_map(geojson_path=RISK_MAP_GEOJSON, area_size=AREA_SIZE):
    """
    Carga el GeoJSON de riesgo, lo recorta/traslada al MISMO cuadro de
    estudio que usa dplace_main.py, calcula centroides y extrae la columna
    de riesgo continuo.
    """
    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        print(f"Mapa de riesgo cargado (crudo): {gdf.shape[0]} celdas")

        # ── Recorte + traslación: MISMO sistema de coordenadas que los sensores ──
        gdf = get_normalized_grid(gdf, CUADRO_X_MIN, CUADRO_Y_MIN, CUADRO_X_MAX, CUADRO_Y_MAX)
        print(f"Mapa de riesgo normalizado al cuadro de estudio: {gdf.shape[0]} celdas")

        gdf['x_center'] = gdf.geometry.centroid.x
        gdf['y_center'] = gdf.geometry.centroid.y
        gdf['risk'] = gdf['indice_inseguridad']

        if gdf['risk'].isna().any():
            print("Advertencia: hay valores NaN en 'indice_inseguridad'. Rellenando con 0.")
            gdf['risk'] = gdf['risk'].fillna(0)

        gdf = gdf.reset_index(drop=True)
        return gdf

    print(f"No se encontró {geojson_path}. Generando mapa de riesgo sintético...")
    rows, cols = 24, 25
    cell_w = area_size / cols
    cell_h = area_size / rows
    centers, geoms, risks = [], [], []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            x_center, y_center = (x0 + x1) / 2, (y0 + y1) / 2
            centers.append((x_center, y_center))
            geoms.append(box(x0, y0, x1, y1))
            dist_from_center = np.sqrt((x_center - area_size / 2) ** 2 + (y_center - area_size / 2) ** 2)
            risk = int(np.clip(np.round(4 * np.exp(-dist_from_center / 2000)), 0, 4))
            risks.append(risk)

    gdf = gpd.GeoDataFrame({
        'cell_id': np.arange(1, len(centers) + 1),
        'x_center': [c[0] for c in centers],
        'y_center': [c[1] for c in centers],
        'risk': risks,
        'geometry': geoms
    })
    print(f"Mapa de riesgo sintético generado con {gdf.shape[0]} celdas")
    return gdf


# ==========================
# 2. ASIGNACIÓN DE SENSORES A CELDAS (polygon.contains real)
# ==========================
def assign_sensors_to_cells(sensors, risk_df):
    """
    Asigna cada sensor a la celda cuyo polígono lo contiene (spatial join
    real, consistente con preprocessing_phase.py). Si un sensor no cae en
    ningún polígono (por ejemplo, quedó justo en un borde recortado), se
    asigna a la celda de centroide más cercano como respaldo.

    Devuelve un array con el conteo de sensores por celda, en el mismo
    orden que las filas de risk_df.
    """
    cells = len(risk_df)
    N_cell = np.zeros(cells, dtype=int)

    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in sensors],
        crs=risk_df.crs
    )

    joined = gpd.sjoin(points_gdf, risk_df[['geometry']], how='left', predicate='within')

    # Un sensor puede aparecer varias veces en el resultado si las geometrías
    # de la grilla se solapan. El índice izquierdo identifica al sensor; se
    # debe reducir el join a una sola celda por sensor antes de crear la
    # máscara de faltantes. De lo contrario, la máscara puede tener más
    # elementos que ``sensors`` (por ejemplo, 1088 frente a 1000).
    cell_idx = (
        joined['index_right']
        .groupby(level=0)
        .first()
        .reindex(np.arange(len(sensors)))
        .to_numpy(dtype=float)
        .copy()
    )  # NaN si el punto no cayó en ninguna celda

    # Respaldo: puntos sin celda asignada -> celda de centroide más cercano
    missing_mask = np.isnan(cell_idx)
    if missing_mask.any():
        cell_centers = risk_df[['x_center', 'y_center']].values
        tree = cKDTree(cell_centers)
        _, nearest_idx = tree.query(sensors[missing_mask], k=1)
        cell_idx[missing_mask] = nearest_idx

    cell_idx = cell_idx.astype(int)
    counts = np.bincount(cell_idx, minlength=cells)
    N_cell[:len(counts)] = counts[:cells]
    return N_cell


# ==========================
# 3. MÉTRICAS DE EFICIENCIA
# ==========================
def compute_metrics(sensors, risk_df):
    """
    Calcula RWC, DRW, GlobalCoverage y Spearman para una distribución de
    sensores dada. Retorna un diccionario con los valores.
    """
    N_cell = assign_sensors_to_cells(sensors, risk_df)
    risk_values = risk_df['risk'].values
    cell_centers = risk_df[['x_center', 'y_center']].values

    # RQ7 - Global Coverage: fracción de celdas con al menos un sensor
    coverage_global = np.mean(N_cell > 0)

    # RQ1 - Risk-Weighted Coverage
    total_risk = np.sum(risk_values)
    if total_risk > 0:
        rwc = np.sum(risk_values * (N_cell > 0)) / total_risk
    else:
        rwc = np.nan

    # RQ2 - Risk-Weighted Distance (solo celdas con riesgo > 0)
    risk_mask = risk_values > 0
    if np.any(risk_mask):
        sensor_tree = cKDTree(sensors)
        distances, _ = sensor_tree.query(cell_centers[risk_mask], k=1)
        risk_weights = risk_values[risk_mask]
        drw = np.sum(risk_weights * distances) / np.sum(risk_weights)
    else:
        drw = np.nan

    # RQ6 - Correlación de Spearman entre riesgo y densidad de sensores
    rho, p_rho = spearmanr(risk_values, N_cell)

    return {
        'RWC': rwc,
        'DRW': drw,
        'GlobalCoverage': coverage_global,
        'Spearman': rho,
        'p_Spearman': p_rho,
        'SensorCounts': N_cell
    }


# ==========================
# 4. SIMULACIÓN MONTE CARLO (RQ3)
# ==========================
def monte_carlo_simulation(risk_df, num_sim=NUM_MC_SIMULATIONS, num_sensors=NUM_SENSORS,
                            area_size=AREA_SIZE, seed=RANDOM_SEED):
    """
    Genera num_sim distribuciones aleatorias uniformes y calcula métricas
    para cada una. Retorna un DataFrame con los resultados.
    """
    rng = np.random.default_rng(seed)
    metrics_list = []
    for _ in range(num_sim):
        random_sensors = rng.uniform(0, area_size, size=(num_sensors, 2))
        m = compute_metrics(random_sensors, risk_df)
        metrics_list.append({
            'RWC': m['RWC'],
            'DRW': m['DRW'],
            'GlobalCoverage': m['GlobalCoverage'],
            'Spearman': m['Spearman']
        })
    return pd.DataFrame(metrics_list)


# ==========================
# 5. ANÁLISIS ESTADÍSTICO
# ==========================
def empirical_p_value(proposed_value, random_values, direction='greater'):
    """
    Valor p empírico: proporción de valores aleatorios al menos tan
    extremos como el valor propuesto (comparación contra la distribución
    nula generada por Monte Carlo).
    """
    if direction == 'greater':
        p = (np.sum(random_values >= proposed_value) + 1) / (len(random_values) + 1)
    elif direction == 'less':
        p = (np.sum(random_values <= proposed_value) + 1) / (len(random_values) + 1)
    else:
        raise ValueError("direction debe ser 'greater' o 'less'")
    return p


def confidence_interval(data, confidence=0.95):
    """
    Intervalo de confianza usando el cuantil t de Student (no la
    aproximación normal), tal como especifica el documento de referencia:
        IC95% = X̄ ± t_(0.975, n-1) * s / sqrt(n)
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    t_crit = t_dist.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    return mean - margin, mean + margin


def paired_wilcoxon_test(proposed_replicas, risk_df, area_size=AREA_SIZE, seed=RANDOM_SEED):
    """
    RQ4 - Wilcoxon signed-rank pareado.

    Para cada réplica k de la propuesta (una de las N_SIMULATIONS corridas
    de preprocesamiento), se genera un baseline aleatorio ÚNICO con el mismo
    número de sensores, y se calcula:
        Δ_RWC[k] = RWC_propuesto[k] - RWC_random[k]
        Δ_DRW[k] = DRW_random[k]    - DRW_propuesto[k]   (menor DRW = mejor)

    H0: mediana(Δ) = 0   vs   H1: mediana(Δ) > 0

    Retorna un dict con estadístico y p-valor para RWC y DRW, más las
    métricas por réplica (útiles para los boxplots).
    """
    rng = np.random.default_rng(seed)

    proposed_metrics_per_replica = []
    random_metrics_per_replica = []
    deltas_rwc, deltas_drw = [], []

    for sensors_k in proposed_replicas:
        n_k = sensors_k.shape[0]

        m_prop = compute_metrics(sensors_k, risk_df)
        random_sensors_k = rng.uniform(0, area_size, size=(n_k, 2))
        m_rand = compute_metrics(random_sensors_k, risk_df)

        proposed_metrics_per_replica.append(m_prop)
        random_metrics_per_replica.append(m_rand)

        deltas_rwc.append(m_prop['RWC'] - m_rand['RWC'])
        deltas_drw.append(m_rand['DRW'] - m_prop['DRW'])

    result = {
        'proposed_metrics_per_replica': proposed_metrics_per_replica,
        'random_metrics_per_replica': random_metrics_per_replica,
    }

    # Wilcoxon requiere al menos algunas diferencias no nulas y n>=1
    for name, deltas in (('RWC', deltas_rwc), ('DRW', deltas_drw)):
        deltas = np.asarray(deltas, dtype=float)
        if len(deltas) < 2 or np.allclose(deltas, 0):
            result[f'{name}_stat'] = np.nan
            result[f'{name}_p'] = np.nan
            print(f"⚠ No se pudo calcular Wilcoxon para {name} (muy pocas réplicas o sin variación).")
            continue
        try:
            stat, p = wilcoxon(deltas, alternative='greater')
        except ValueError as e:
            print(f"⚠ Wilcoxon falló para {name}: {e}")
            stat, p = np.nan, np.nan
        result[f'{name}_stat'] = stat
        result[f'{name}_p'] = p

    return result


# ==========================
# 6. EJECUCIÓN PRINCIPAL
# ==========================
def main():
    # ── Cargar datos ──
    sensors = load_or_generate_sensors()
    risk_df = load_or_generate_risk_map()
    replicas = load_all_sim_replicas()

    print("\n--- Métricas para la distribución propuesta (final_nodos.csv) ---")
    proposed_metrics = compute_metrics(sensors, risk_df)
    for key, val in proposed_metrics.items():
        if key != 'SensorCounts':
            print(f"{key}: {val:.4f}")

    # ── RQ3: Monte Carlo ──
    print(f"\nEjecutando Monte Carlo con {NUM_MC_SIMULATIONS} replicaciones...")
    mc_results = monte_carlo_simulation(risk_df)

    p_rwc = empirical_p_value(proposed_metrics['RWC'], mc_results['RWC'].values, 'greater')
    p_drw = empirical_p_value(proposed_metrics['DRW'], mc_results['DRW'].values, 'less')
    p_gc = empirical_p_value(proposed_metrics['GlobalCoverage'], mc_results['GlobalCoverage'].values, 'greater')
    p_spearman = empirical_p_value(proposed_metrics['Spearman'], mc_results['Spearman'].values, 'greater')

    print("\n--- RQ3: Comparación con distribución aleatoria (Monte Carlo, valor p empírico) ---")
    print(f"RWC propuesto: {proposed_metrics['RWC']:.4f} | p = {p_rwc:.4f}")
    print(f"DRW propuesto: {proposed_metrics['DRW']:.4f} | p = {p_drw:.4f}")
    print(f"Global Coverage propuesto: {proposed_metrics['GlobalCoverage']:.4f} | p = {p_gc:.4f}")
    print(f"Spearman propuesto: {proposed_metrics['Spearman']:.4f} | p = {p_spearman:.4f}")

    # ── RQ5: Intervalos de confianza (t-Student) para el baseline aleatorio ──
    print("\n--- RQ5: Intervalos de confianza 95% (t-Student) — baseline aleatorio ---")
    for col in ['RWC', 'DRW', 'GlobalCoverage', 'Spearman']:
        ci_low, ci_high = confidence_interval(mc_results[col].values)
        print(f"{col}: [{ci_low:.4f}, {ci_high:.4f}]")

    # ── RQ4: Wilcoxon signed-rank pareado (usa las réplicas de preprocesamiento) ──
    wilcoxon_result = None
    proposed_replica_df = None
    if replicas:
        print(f"\n--- RQ4: Wilcoxon signed-rank pareado ({len(replicas)} réplicas) ---")
        wilcoxon_result = paired_wilcoxon_test(replicas, risk_df)

        proposed_replica_df = pd.DataFrame([
            {'RWC': m['RWC'], 'DRW': m['DRW'], 'GlobalCoverage': m['GlobalCoverage'], 'Spearman': m['Spearman']}
            for m in wilcoxon_result['proposed_metrics_per_replica']
        ])

        for metric in ('RWC', 'DRW'):
            stat = wilcoxon_result.get(f'{metric}_stat')
            p = wilcoxon_result.get(f'{metric}_p')
            if p is not None and not np.isnan(p):
                print(f"{metric}: estadístico={stat:.4f} | p={p:.4f} "
                      f"({'significativo' if p < 0.05 else 'no significativo'} al 5%)")
            else:
                print(f"{metric}: no calculado (ver advertencia arriba)")

        print("\n--- RQ5: Intervalos de confianza 95% (t-Student) — réplicas de la propuesta ---")
        for col in ['RWC', 'DRW', 'GlobalCoverage', 'Spearman']:
            ci_low, ci_high = confidence_interval(proposed_replica_df[col].values)
            print(f"{col}: [{ci_low:.4f}, {ci_high:.4f}]")
    else:
        print(
            "\n--- RQ4: Wilcoxon signed-rank OMITIDO ---\n"
            "No se encontraron réplicas sim{i}_nodos_pre.csv en OUTPUT_DIR. "
            "Corre dplace_main.py completo (las N_SIMULATIONS simulaciones de "
            "preprocesamiento) antes de repetir esta evaluación."
        )

    # ── RQ5: Boxplots comparativos (Random vs Propuesta) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_names = ['RWC', 'DRW', 'GlobalCoverage', 'Spearman']

    for ax, metric in zip(axes.flatten(), metric_names):
        data_to_plot = [mc_results[metric].values]
        labels = ['Random']

        if proposed_replica_df is not None:
            data_to_plot.append(proposed_replica_df[metric].values)
            labels.append('Propuesta')

        ax.boxplot(data_to_plot, labels=labels, showfliers=False)
        # Valor puntual del resultado final (final_nodos.csv), como referencia adicional
        ax.axhline(y=proposed_metrics[metric], color='red', linestyle='--',
                    label='Propuesta (final)')
        ax.set_title(metric)
        ax.legend(fontsize=8)

    plt.tight_layout()
    boxplot_path = os.path.join(RESULTS_DIR, 'boxplots_comparacion.png')
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    print(f"\nBoxplots guardados en '{boxplot_path}'")

    # ── Guardar resumen numérico en CSV, para incluir en el reporte ──
    summary_rows = [{
        'metric': m,
        'proposed_final': proposed_metrics[m],
        'random_mc_mean': mc_results[m].mean(),
        'random_ci_low': confidence_interval(mc_results[m].values)[0],
        'random_ci_high': confidence_interval(mc_results[m].values)[1],
        'empirical_p_vs_random': {
            'RWC': p_rwc, 'DRW': p_drw, 'GlobalCoverage': p_gc, 'Spearman': p_spearman
        }[m],
        'wilcoxon_p': (wilcoxon_result.get(f'{m}_p') if wilcoxon_result and m in ('RWC', 'DRW') else np.nan),
    } for m in metric_names]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, 'summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Resumen numérico guardado en '{summary_path}'")


if __name__ == "__main__":
    main()