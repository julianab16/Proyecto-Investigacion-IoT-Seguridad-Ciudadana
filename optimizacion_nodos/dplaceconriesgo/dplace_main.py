"""
dplace_main.py : DPLACE Model - Main Orchestrator
=================================================
Gateway Placement for LoRaWAN Networks using (LEGACY MODE):
    - Phase 1: Pre-Processing  → Gap Statistics + Fuzzy C-Means
    - Phase 2: Processing      → Aggregate runs + K-Means Gap Statistics
    - Phase 3: Validation      → PDR, Coverage, CAPEX/OPEX evaluation

NEW MODE (Gateways Fijos → Optimizar Nodos):
    - Usa gateways fijos desde CSV
    - Reubica nodos (misma cantidad) dentro del área
    - Mantiene el algoritmo de validación (PDR/Cobertura/Costos)
    - NUEVO: usa índice de inseguridad para concentrar nodos en zonas críticas

Usage:
        python dplace_main.py

Outputs are saved to ./output/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import random

from preprocessing_phase import preprocessing  # LEGACY: usado solo en modo original
from processing_phase import processing        # LEGACY: usado solo en modo original
from validation_phase import validation

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# NUEVO: modo de ejecución
# - "fixed_gateways_original_flow": usa flujo original (Gap + FCM + K-Means) pero para optimizar nodos
# - "fixed_gateways_optimize_nodes": (LEGACY) reubicar nodos con movimiento mínimo hacia gateways
# - "legacy_gateway_search": pipeline original de búsqueda de gateways
RUN_MODE = "fixed_gateways_original_flow"  # CAMBIO: ahora el modo por defecto usa el flujo original

# NUEVO: CSVs de entrada (gateways fijos + nodos a optimizar)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NODES_CSV = os.path.join(PROJECT_ROOT, "data_base", "nodos_cuadrado.csv")
GATEWAYS_CSV = os.path.join(PROJECT_ROOT, "data_base", "gateways_cuadrado.csv")

# NUEVO: GeoJSON de índice de inseguridad (salida de georeferencia.py)
INSECURITY_GEOJSON = os.path.join(PROJECT_ROOT, "images", "grid_cali.geojson")
INSECURITY_INDEX_COL = "indice_inseguridad"
USE_INSECURITY_INDEX = True
INSECURITY_WEIGHT_SCALE = 4   # aumenta la concentración en zonas inseguras (1..1+scale)
INSECURITY_MAX_REP = 6        # límite de repetición de nodos para no inflar demasiado

# NUEVO (flujo original aplicado a nodos): parámetros para escenarios
NODES_JITTER_STD = 0.0     # 0 = sin ruido; >0 agrega variación por simulación
GATEWAY_ANCHOR_WEIGHT = 5  # cuánto "pesan" los gateways fijos en el clustering (anclaje)

# NUEVO: forzar K = número original de nodos (solo optimizar posiciones)
FORCE_K_TO_NODES = True

N_SIMULATIONS = 5          # LEGACY: Number of preprocessing runs (scenarios)
N_DEVICES = 200            # LEGACY: IoT devices per scenario
AXIS_RANGE = 10000         # LEGACY: Simulation area (units). En modo nuevo se recalcula con bounds.
COVERAGE_RADIUS = 2000     # Gateway coverage radius (same units)

# Gap Statistics parameters (LEGACY)
NREFS_PRE = 20             # References for preprocessing (lower for speed)
MAX_CLUSTERS_PRE = 12      # Max clusters in preprocessing
MAX_POINTS = 70            # Max devices per gateway cluster

NREFS_PROC = 50            # References for processing
MAX_CLUSTERS_PROC = 10     # Max clusters in processing

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def generate_devices(n=200, area=10000):
    """Generate random 2D IoT device positions. (LEGACY: modo búsqueda de gateways)"""
    x = [random.randint(1, area) for _ in range(n)]
    y = [random.randint(1, area) for _ in range(n)]
    return np.array(list(zip(x, y)))


# NUEVO: carga CSV de puntos (nodos o gateways fijos)
def load_points_csv(csv_path, id_col='id', x_col='x', y_col='y'):
    df = pd.read_csv(csv_path)
    required = {id_col, x_col, y_col}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV {csv_path} no tiene columnas requeridas: {missing}")
    points = df[[x_col, y_col]].to_numpy(dtype=float)
    return df, points


# NUEVO: bounds para plots en coordenadas reales
def compute_axis_bounds(*arrays, padding_ratio=0.05):
    all_points = np.vstack([a for a in arrays if a is not None and len(a) > 0])
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    pad_x = (max_x - min_x) * padding_ratio if max_x > min_x else 1.0
    pad_y = (max_y - min_y) * padding_ratio if max_y > min_y else 1.0
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


# NUEVO: cargar índice de inseguridad desde GeoJSON y asignarlo a nodos
def load_insecurity_index(nodes, geojson_path, index_col=INSECURITY_INDEX_COL, default_value=0.0):
    """
    Asigna a cada nodo el índice de inseguridad del polígono donde cae.
    Si algún nodo queda fuera, usa vecino más cercano (si es posible) o default.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        print(f"[WARN] geopandas/shapely no disponible: {e}. Usando índice 0.")
        return np.full(len(nodes), default_value, dtype=float)

    if not os.path.exists(geojson_path):
        print(f"[WARN] No se encontró {geojson_path}. Usando índice 0.")
        return np.full(len(nodes), default_value, dtype=float)

    grid = gpd.read_file(geojson_path)
    if grid.crs is None:
        grid = grid.set_crs("EPSG:3116")

    gdf_nodes = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in nodes],
        crs="EPSG:3116"
    )

    if gdf_nodes.crs != grid.crs:
        gdf_nodes = gdf_nodes.to_crs(grid.crs)

    joined = gpd.sjoin(gdf_nodes, grid[[index_col, "geometry"]], how="left", predicate="within")

    # Si hay nodos sin índice, intentar nearest
    if joined[index_col].isna().any():
        try:
            joined_near = gpd.sjoin_nearest(gdf_nodes, grid[[index_col, "geometry"]], how="left")
            joined[index_col] = joined[index_col].fillna(joined_near[index_col])
        except Exception:
            joined[index_col] = joined[index_col].fillna(default_value)

    return joined[index_col].fillna(default_value).to_numpy(dtype=float)


# NUEVO: construir escenarios desde nodos CSV (opcionalmente con ruido)
def build_nodes_scenario(base_nodes, jitter_std=0.0):
    if jitter_std <= 0:
        return base_nodes.copy()
    noise = np.random.normal(0, jitter_std, base_nodes.shape)
    return base_nodes + noise

def plot_devices_and_gateways(X, gateways, title, filepath, coverage_radius=None, u=None, axis_bounds=None):
    """Plot IoT devices with gateway positions and optional coverage circles."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Color devices by nearest gateway
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(gateways), 1)))
    if u is not None:
        assignments = np.argmax(u, axis=0)
        for j, gw in enumerate(gateways):
            mask = assignments == j
            ax.scatter(X[mask, 0], X[mask, 1], s=8, color=colors[j % len(colors)], alpha=0.5)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=8, color='#58a6ff', alpha=0.5)

    # Coverage circles
    if coverage_radius is not None:
        for j, gw in enumerate(gateways):
            circle = plt.Circle(
                (gw[0], gw[1]), coverage_radius,
                color=colors[j % len(colors)], fill=False,
                linestyle='--', linewidth=0.8, alpha=0.4
            )
            ax.add_patch(circle)

    # Gateway markers
    for j, gw in enumerate(gateways):
        ax.scatter(gw[0], gw[1], s=180, marker='^', color=colors[j % len(colors)],
                   edgecolors='white', linewidths=0.8, zorder=5)
        ax.annotate(f'GW{j+1}', (gw[0], gw[1]),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7, color='white', fontweight='bold')

    if axis_bounds is None:
        ax.set_xlim(0, AXIS_RANGE)
        ax.set_ylim(0, AXIS_RANGE)
    else:
        min_x, max_x, min_y, max_y = axis_bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    ax.set_title(title, color='white', fontsize=11, pad=10)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_gap_curve(gap_df, gap_sk_df, optimal_k, title_suffix, filepath):
    """Plot Gap Statistics curve and bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')

    # Gap curve
    ax1.plot(gap_df['clusterCount'], gap_df['gap'], color='#58a6ff', linewidth=2.5)
    ax1.scatter(
        gap_df[gap_df['clusterCount'] == optimal_k]['clusterCount'],
        gap_df[gap_df['clusterCount'] == optimal_k]['gap'],
        s=200, color='#f78166', zorder=5, label=f'Optimal k={optimal_k}'
    )
    ax1.set_title(f'Gap Statistic {title_suffix}', color='white', fontsize=10)
    ax1.set_xlabel('Number of Clusters (k)', color='#8b949e')
    ax1.set_ylabel('Gap Value', color='#8b949e')
    ax1.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    ax1.grid(True, color='#21262d', linewidth=0.5)

    # Gap_sk bar chart
    positive = gap_sk_df['Gap_sk'] >= 0
    bar_colors = ['#3fb950' if p else '#f78166' for p in positive]
    ax2.bar(gap_sk_df['clusterCount'], gap_sk_df['Gap_sk'], color=bar_colors, alpha=0.85)
    ax2.set_title(f'Gap[k] − Gap[k+1] − sk[k+1] {title_suffix}', color='white', fontsize=10)
    ax2.set_xlabel('Cluster Count', color='#8b949e')
    ax2.set_ylabel('Gap_sk', color='#8b949e')
    ax2.grid(True, color='#21262d', linewidth=0.5)
    ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_validation_map(val_result, title, filepath, axis_bounds=None):
    """Plot validation: covered vs uncovered devices + gateway positions."""
    device_df = val_result['device_df']
    gateways = val_result['gateways']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    covered = device_df[device_df['covered']]
    uncovered = device_df[~device_df['covered']]

    ax.scatter(covered['x'], covered['y'], s=8, color='#3fb950', alpha=0.6, label='Covered')
    ax.scatter(uncovered['x'], uncovered['y'], s=8, color='#f78166', alpha=0.6, label='Uncovered')

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(gateways), 1)))
    for j, gw in enumerate(gateways):
        circle = plt.Circle(
            (gw[0], gw[1]), COVERAGE_RADIUS,
            color=colors[j % len(colors)], fill=False,
            linestyle='--', linewidth=1.0, alpha=0.5
        )
        ax.add_patch(circle)
        ax.scatter(gw[0], gw[1], s=200, marker='^', color=colors[j % len(colors)],
                   edgecolors='white', linewidths=0.8, zorder=5)

    pdr = val_result['pdr']
    cov = val_result['coverage_ratio']
    info = (f"PDR: {pdr*100:.1f}%  |  Coverage: {cov*100:.1f}%  |  "
            f"Devices: {val_result['n_covered']}/{val_result['n_covered']+val_result['n_uncovered']}")
    ax.set_title(f"{title}\n{info}", color='white', fontsize=10, pad=10)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    if axis_bounds is None:
        ax.set_xlim(0, AXIS_RANGE)
        ax.set_ylim(0, AXIS_RANGE)
    else:
        min_x, max_x, min_y, max_y = axis_bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_summary(summary_records, filepath):
    """Plot PDR, coverage, CAPEX, and OPEX across all simulations."""
    df = pd.DataFrame(summary_records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    
    # Define 4 metrics to plot
    titles = ['PDR (%)', 'Coverage (%)', 'CAPEX ($)', 'OPEX Annual ($)']
    cols = ['pdr', 'coverage', 'capex_total', 'opex_annual']
    colors_bar = ['#58a6ff', '#3fb950', '#d2a8ff', '#ffa657']

    for ax, col, ttl, clr in zip(axes.flat, cols, titles, colors_bar):
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
        ax.bar(df['sim'], df[col], color=clr, alpha=0.85)
        ax.set_title(ttl, color='white', fontsize=10)
        ax.set_xlabel('Simulation', color='#8b949e')
        ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
        mean_val = df[col].mean()
        ax.axhline(mean_val, color='white', linewidth=1, linestyle='--', alpha=0.6)
        ax.text(df['sim'].iloc[-1] + 0.2, mean_val, f' μ={mean_val:.1f}',
                color='white', fontsize=8, va='center')

    plt.suptitle('DPLACE Model – Simulation Summary (Validation Phase)', color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DPLACE Model — LoRaWAN Gateway Placement")
    print("=" * 60)
    print(f"  Simulations  : {N_SIMULATIONS}")
    print(f"  Coverage R   : {COVERAGE_RADIUS}")
    print()

    # ─────────────────────────────────────────────
    # NUEVO MODO: flujo original aplicado a nodos (Gap + FCM + K-Means)
    # ─────────────────────────────────────────────
    if RUN_MODE == "fixed_gateways_original_flow":
        print("[MODE] Flujo original → Optimizar nodos con gateways fijos (CSV)")

        nodes_df, nodes = load_points_csv(NODES_CSV)
        gateways_df, gateways = load_points_csv(GATEWAYS_CSV)
        axis_bounds = compute_axis_bounds(nodes, gateways)

        print(f"  Nodos cargados     : {len(nodes)}")
        print(f"  Gateways fijos     : {len(gateways)}")
        print(f"  CSV nodos          : {NODES_CSV}")
        print(f"  CSV gateways       : {GATEWAYS_CSV}")
        print(f"  Jitter nodos (std) : {NODES_JITTER_STD}")
        print(f"  Peso gateways      : {GATEWAY_ANCHOR_WEIGHT}")
        print(f"  Forzar K = N nodos : {FORCE_K_TO_NODES}")
        print(f"  Usar índice inseg. : {USE_INSECURITY_INDEX}")
        print(f"  GeoJSON inseguridad: {INSECURITY_GEOJSON}")

        # MODIFICADO: Renombrar 'all_preprocessing_gateways' a 'all_preprocessing_nodes'
        # Razón: Ahora estamos optimizando NODOS, no gateways
        all_preprocessing_nodes = []  # MODIFICADO: 'gateways' → 'nodes'
        all_preprocessing_results = []

        # ─── PHASE 1 (adaptada): PRE-PROCESSING sobre nodos CSV ─────────
        print("─" * 50)
        print("PHASE 1 — PRE-PROCESSING (Nodos CSV)")
        print("─" * 50)

        # K forzado (si aplica) = total original de nodos CSV
        forced_k = len(nodes) if FORCE_K_TO_NODES else None

        # NUEVO: índice de inseguridad por nodo (para ponderar clustering)
        if USE_INSECURITY_INDEX:
            insecurity_values = load_insecurity_index(nodes, INSECURITY_GEOJSON)
            print(
                f"  Índice inseguridad: min={np.min(insecurity_values):.2f} "
                f"max={np.max(insecurity_values):.2f} mean={np.mean(insecurity_values):.2f}"
            )
        else:
            insecurity_values = np.zeros(len(nodes), dtype=float)

        for sim in range(N_SIMULATIONS):
            print(f"\n[Simulation {sim+1}/{N_SIMULATIONS}]")

            # CAMBIO: en lugar de generar dispositivos aleatorios, usar nodos CSV
            #X_base = build_nodes_scenario(nodes, jitter_std=NODES_JITTER_STD)
            X_base = nodes

            # NUEVO: Construir pesos a partir del índice de inseguridad (sin remuestrear)
            if USE_INSECURITY_INDEX:
                # Los insecurity_values ya se cargaron fuera del bucle (son los mismos para todas las simulaciones)
                vmin = np.min(insecurity_values)
                vmax = np.max(insecurity_values)
                if vmax > vmin:
                    norm = (insecurity_values - vmin) / (vmax - vmin)
                else:
                    norm = np.zeros_like(insecurity_values)
                sample_weights = 1.0 + norm * INSECURITY_WEIGHT_SCALE
            else:
                sample_weights = np.ones(len(nodes), dtype=float)

            # CAMBIO: Anclar gateways fijos duplicándolos (sigue igual)
            if GATEWAY_ANCHOR_WEIGHT > 0:
                gw_anchor = np.repeat(gateways, GATEWAY_ANCHOR_WEIGHT, axis=0)
            else:
                gw_anchor = np.empty((0, 2))

            # Plot nodos base + gateways fijos
            plot_devices_and_gateways(
                X_base, gateways,
                f"Nodos CSV – Sim {sim+1} (Gateways fijos)",
                os.path.join(OUTPUT_DIR, f'csv_sim{sim+1}_nodes.png'),
                coverage_radius=COVERAGE_RADIUS,
                axis_bounds=axis_bounds
            )

            # Llamar a preprocessing pasando los pesos
            pre = preprocessing(
                X_nodos=X_base,               # ← los nodos originales (sin remuestrear)
                gateways_fijos=gw_anchor,
                n_nodes_optimize=len(nodes),
                sample_weights=sample_weights,
                seed=SEED + sim  # ← pesos derivados del índice de inseguridad
            )


            # MODIFICADO: Usar nuevas claves del resultado de preprocessing
            # 'k_final' → 'n_nodes', 'gateways' → 'nodos_opt'
            # gap_df y gap_sk_df ya no se retornan (no hay búsqueda de k óptimo)
            
            # Plot curvas Gap - COMENTADO: Ya no se generan con k variable
            # plot_gap_curve(
            #     pre['gap_df'], pre['gap_sk_df'], pre['n_nodes'],
            #     f'(FCM – Nodos CSV – Sim {sim+1})',
            #     os.path.join(OUTPUT_DIR, f'csv_sim{sim+1}_gap_fcm.png')
            # )

            # Plot centros FCM (interpretados como "nodos optimizados")
            # MODIFICADO: Usar 'nodos_opt' en lugar de 'gateways'
            plot_devices_and_gateways(
                X_base, pre['nodos_opt'],  # MODIFICADO: 'pre['gateways']' → 'pre['nodos_opt']'
                f"FCM – Nodos CSV (k={pre['n_nodes']}) – Sim {sim+1}",  # MODIFICADO: 'k_final' → 'n_nodes'
                os.path.join(OUTPUT_DIR, f'csv_sim{sim+1}_fcm_nodes.png'),
                coverage_radius=COVERAGE_RADIUS,
                axis_bounds=axis_bounds
            )

            # MODIFICADO: Guardar nodos optimizados en lugar de gateways
            # all_preprocessing_gateways → all_preprocessing_nodes
            all_preprocessing_nodes.append(pre['nodos_opt'])  # MODIFICADO: Se usa 'nodos_opt' (no 'gateways')
            all_preprocessing_results.append(pre)

        # ─── PHASE 2 (adaptada): PROCESSING sobre nodos optimizados ───────────
        print("\n" + "─" * 50)
        print("PHASE 2 — PROCESSING (Nodos CSV)")
        print("─" * 50)

        # MODIFICADO: Llamar a processing con nuevos parámetros
        # all_preprocessing_gateways → all_preprocessing_nodes (contiene nodos optimizados)
        # NUEVO parámetro: n_nodos_fijos = len(nodes) (cantidad de nodos originales)
        # Se elimina nrefs y maxClusters (no se busca k óptimo)
        proc = processing(
            all_nodos_optimizados=all_preprocessing_nodes,  # MODIFICADO: Usar nodos en lugar de gateways
            n_nodos_fijos=len(nodes),  # NUEVO: Pasar cantidad FIJA de nodos
            verbose=True
        )



        # MODIFICADO: Plot curva Gap - COMENTADO: Ya no se genera (no hay búsqueda de k óptimo)
        # plot_gap_curve(
        #     proc['gap_df'], proc['gap_sk_df'], proc['n_nodos'],
        #     '(K-Means – Nodos CSV)',
        #     os.path.join(OUTPUT_DIR, 'csv_processing_gap_kmeans.png')
        # )

        # MODIFICADO: Usar nuevas claves del resultado de processing
        # 'gateways' → 'nodos_finales', 'k_optimal' → 'n_nodos'
        optimized_nodes = proc['nodos_finales']  # MODIFICADO: 'gateways' → 'nodos_finales'
        axis_bounds = compute_axis_bounds(nodes, gateways, optimized_nodes)

        plot_devices_and_gateways(
            optimized_nodes, gateways,
            f"Nodos óptimos (K={proc['n_nodos']}) + Gateways fijos",  # MODIFICADO: 'k_optimal' → 'n_nodos'
            os.path.join(OUTPUT_DIR, 'csv_processing_final_nodes.png'),
            coverage_radius=COVERAGE_RADIUS,
            axis_bounds=axis_bounds
        )

        # Guardar CSV de nodos óptimos
        optimized_df = pd.DataFrame(optimized_nodes, columns=['x', 'y'])
        optimized_df.insert(0, 'id', range(len(optimized_df)))
        out_nodes_path = os.path.join(OUTPUT_DIR, 'optimized_nodes_from_flow.csv')
        optimized_df.to_csv(out_nodes_path, index=False)

        # ─── PHASE 3 (misma validación) ─────────────────────────────────
        print("\n" + "─" * 50)
        print("PHASE 3 — VALIDATION (Gateways fijos)")
        print("─" * 50)

        val_before = validation(
            nodes,
            gateways,
            coverage_radius=COVERAGE_RADIUS,
            verbose=True
        )
        val_after = validation(
            optimized_nodes,
            gateways,
            coverage_radius=COVERAGE_RADIUS,
            verbose=True
        )

        plot_validation_map(
            val_before,
            "Validación — Nodos originales (CSV)",
            os.path.join(OUTPUT_DIR, "csv_validation_before.png"),
            axis_bounds=axis_bounds
        )
        plot_validation_map(
            val_after,
            "Validación — Nodos óptimos (flujo original)",
            os.path.join(OUTPUT_DIR, "csv_validation_after.png"),
            axis_bounds=axis_bounds
        )

        # Guardar resultados por dispositivo/gateway
        val_before['device_df'].to_csv(
            os.path.join(OUTPUT_DIR, "csv_device_results_before.csv"), index=False
        )
        val_after['device_df'].to_csv(
            os.path.join(OUTPUT_DIR, "csv_device_results_after.csv"), index=False
        )
        val_before['gateway_df'].to_csv(
            os.path.join(OUTPUT_DIR, "csv_gateway_results_before.csv"), index=False
        )
        val_after['gateway_df'].to_csv(
            os.path.join(OUTPUT_DIR, "csv_gateway_results_after.csv"), index=False
        )

        # Resumen comparativo (2 barras)
        summary_records = [
            {
                'sim': 1,
                'pdr': val_before['pdr'] * 100,
                'coverage': val_before['coverage_ratio'] * 100,
                'capex_total': val_before['capex_total'],
                'opex_annual': val_before['opex_annual']
            },
            {
                'sim': 2,
                'pdr': val_after['pdr'] * 100,
                'coverage': val_after['coverage_ratio'] * 100,
                'capex_total': val_after['capex_total'],
                'opex_annual': val_after['opex_annual']
            }
        ]

        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "csv_summary.csv"), index=False)
        plot_summary(summary_records, os.path.join(OUTPUT_DIR, "csv_summary_charts.png"))

        print(f"\n✔ Modo flujo original completado. Salidas en: {OUTPUT_DIR}")
        print(f"  CSV nodos óptimos: {out_nodes_path}")
        print(f"  Cobertura antes: {val_before['coverage_ratio']*100:.1f}% | después: {val_after['coverage_ratio']*100:.1f}%")
        return

    # ─────────────────────────────────────────────
    # LEGACY: flujo original (búsqueda de k óptimo y gateways)
    # (se conserva por referencia; no se usa en modo actual)
    # ─────────────────────────────────────────────

    all_preprocessing_gateways = []
    all_preprocessing_results = []
    summary_records = []

    # ─── PHASE 1: PRE-PROCESSING ───────────────
    print("─" * 50)
    print("PHASE 1 — PRE-PROCESSING")
    print("─" * 50)

    for sim in range(N_SIMULATIONS):
        print(f"\n[Simulation {sim+1}/{N_SIMULATIONS}]")
        X = generate_devices(N_DEVICES, AXIS_RANGE)

        # Save device scatter
        plot_devices_and_gateways(
            X, np.empty((0, 2)),
            f"IoT Devices – Sim {sim+1}",
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_devices.png')
        )

        pre = preprocessing(
            X,
            nrefs=NREFS_PRE,
            maxClusters=MAX_CLUSTERS_PRE,
            maxPoints=MAX_POINTS,
            verbose=True
        )

        # Save preprocessing plots
        plot_gap_curve(
            pre['gap_df'], pre['gap_sk_df'], pre['k_final'],
            f'(FCM – Sim {sim+1})',
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_gap_fcm.png')
        )
        plot_devices_and_gateways(
            X, pre['gateways'],
            f"FCM Clustering – Sim {sim+1} (k={pre['k_final']})",
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_fcm_result.png'),
            coverage_radius=COVERAGE_RADIUS,
            u=pre['u']
        )

        all_preprocessing_gateways.append(pre['gateways'])
        all_preprocessing_results.append(pre)

        # Save gateway CSV for this simulation
        gw_df = pd.DataFrame(pre['gateways'], columns=['x', 'y'])
        gw_df.to_csv(
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_gateways_pre.csv'), index=False
        )

    # ─── PHASE 2: PROCESSING ───────────────────
    print("\n" + "─" * 50)
    print("PHASE 2 — PROCESSING")
    print("─" * 50)

    proc = processing(
        all_preprocessing_gateways,
        nrefs=NREFS_PROC,
        maxClusters=MAX_CLUSTERS_PROC,
        verbose=True
    )

    # Save processing plots
    plot_gap_curve(
        proc['gap_df'], proc['gap_sk_df'], proc['k_optimal'],
        '(K-Means – Processing)',
        os.path.join(OUTPUT_DIR, 'processing_gap_kmeans.png')
    )

    # Plot all collected gateways + final centroids
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    ax.scatter(
        proc['all_gw_data'][:, 0], proc['all_gw_data'][:, 1],
        s=30, color='#58a6ff', alpha=0.4, label='Preprocessing GWs'
    )
    ax.scatter(
        proc['gateways'][:, 0], proc['gateways'][:, 1],
        s=250, marker='*', color='#ffa657', edgecolors='white',
        linewidths=0.8, zorder=5, label=f'Final GWs (K={proc["k_optimal"]})'
    )
    for j, gw in enumerate(proc['gateways']):
        circle = plt.Circle(
            (gw[0], gw[1]), COVERAGE_RADIUS,
            color='#ffa657', fill=False, linestyle='--', linewidth=0.8, alpha=0.35
        )
        ax.add_patch(circle)
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(f'Processing Phase – Final Gateway Positions (K={proc["k_optimal"]})',
                 color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'processing_final_gateways.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # Save final gateways CSV
    gw_final_df = pd.DataFrame(proc['gateways'], columns=['x', 'y'])
    gw_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_gateways.csv'), index=False)

    # ─── PHASE 3: VALIDATION ───────────────────
    print("\n" + "─" * 50)
    print("PHASE 3 — VALIDATION")
    print("─" * 50)

    for sim in range(N_SIMULATIONS):
        print(f"\n[Validation – Sim {sim+1}/{N_SIMULATIONS}]")
        X = all_preprocessing_results[sim]['X']
        pre_k = all_preprocessing_results[sim]['k_final']

        val = validation(
            X,
            proc['gateways'],
            coverage_radius=COVERAGE_RADIUS,
            verbose=True
        )

        plot_validation_map(
            val,
            f'Validation – Sim {sim+1}  (K={proc["k_optimal"]} gateways)',
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_validation.png')
        )

        # Save per-device results
        val['device_df'].to_csv(
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_device_results.csv'), index=False
        )

        # Save gateway statistics
        val['gateway_df'].to_csv(
            os.path.join(OUTPUT_DIR, f'sim{sim+1}_gateway_results.csv'), index=False
        )

        summary_records.append({
            'sim': sim + 1,
            'pdr': val['pdr'] * 100,
            'coverage': val['coverage_ratio'] * 100,
            'k_pre': pre_k,
            'k_proc': proc['k_optimal'],
            'capex_total': val['capex_total'],
            'opex_annual': val['opex_annual'],
            'packets_received': val['packets_received'],
            'packets_sent': val['packets_sent'],
            'coverage_constraint': val['coverage_constraint_met'],
            'shannon_constraint': val['shannon_constraint_met']
        })

    # ─── SUMMARY ───────────────────────────────
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)

    plot_summary(summary_records, os.path.join(OUTPUT_DIR, 'summary_charts.png'))

    print(f"\n✔ All outputs saved to: {OUTPUT_DIR}")
    print(f"  Final gateways (K={proc['k_optimal']}): final_gateways.csv")
    print(f"  Summary metrics: summary.csv")
    print(f"  Average PDR: {summary_df['pdr'].mean():.1f}%")
    print(f"  Average Coverage: {summary_df['coverage'].mean():.1f}%")


if __name__ == '__main__':
    main()
