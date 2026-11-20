"""
dplace_main.py : DPLACE Model - Main Orchestrator
=================================================
Gateway Placement for LoRaWAN Networks using:
  - Phase 1: Pre-Processing  → Gap Statistics + Fuzzy C-Means
  - Phase 2: Processing      → Aggregate runs + K-Means Gap Statistics

Usage:
    python dplace_main.py

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import geopandas as gpd
import matplotlib.colors as mcolors
from preprocessing_phase import preprocessing
from processing_phase import processing
from matplotlib.patches import Rectangle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

N = 1000           # IoT devices per scenario
AXIS_RANGE = 5900         # Simulation area (units)
COVERAGE_RADIUS = 2000     # Gateway coverage radius (same units)

# ─── POISSON PATTERN CONFIGURATION ─────────
LAMBDA_MEAN = 1000         # Expected devices per frame (tasa de llegada)
N_FRAMES = 15              # Temporal frames per simulation (dinamismo)
N_SIMULATIONS = 15         # Number of preprocessing runs (scenarios) diez

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputcon15clusters')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
GW_PRE_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_gateways_pre.csv')
STATS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_stats.json')  # o CSV
RESULTS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_pre_result.pkl')  # para objetos grandes
# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def plot_devices_and_gateways(X, gateways, title, filepath, coverage_radius=None, u=None):
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

    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(title, color='white', fontsize=11, pad=10)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def plot_k(all_k_results, filepath):
    x = np.arange(1, len(all_k_results) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    bars = ax.bar(
        x, all_k_results,
        color='steelblue',
        edgecolor='black'
    )

    for bar, k in zip(bars, all_k_results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(k),
            color='#8b949e',
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax.set_xlabel('Sample', color='white')
    ax.set_ylabel('Optimal K', color='white')
    ax.set_title('Optimal K Value Obtained for Each Sample', color='white')
    ax.set_xticks(x)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#8b949e')

    ax.tick_params(axis='both', colors='#8b949e')

    for spine in ax.spines.values():
        spine.set_color('#8b949e')

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    ax2.set_title(f'Gap[k] − Gap[k+1] + sk[k+1] {title_suffix}', color='white', fontsize=10)
    ax2.set_xlabel('Cluster Count', color='#8b949e')
    ax2.set_ylabel('Gap_sk', color='#8b949e')
    ax2.grid(True, color='#21262d', linewidth=0.5)
    ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_validation_map(val_result, title, filepath):
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
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_summary(summary_records, filepath):
    """Plot PDR and coverage across all simulations."""
    df = pd.DataFrame(summary_records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    titles = ['PDR (%)', 'Coverage (%)', 'Gateways (Preprocessing)', 'Gateways (Processing)']
    cols = ['pdr', 'coverage', 'k_pre', 'k_proc']
    colors_bar = ['#58a6ff', '#3fb950', '#d2a8ff', '#ffa657']

    from scipy import stats

    for ax, col, ttl, clr in zip(axes.flat, cols, titles, colors_bar):

        ax.set_facecolor('#161b22')

        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        ax.tick_params(colors='#8b949e')

        values = df[col].values

        mean_val = np.mean(values)

        # IC95
        sem = stats.sem(values)

        ci95 = stats.t.interval(
            0.95,
            len(values) - 1,
            loc=mean_val,
            scale=sem
        )

        ci_low, ci_high = ci95

        ax.bar(df['sim'], values, color=clr, alpha=0.85)

        ax.set_title(ttl, color='white', fontsize=10)

        ax.set_xlabel('Simulation', color='#8b949e')

        ax.grid(True, color='#21262d', linewidth=0.5, axis='y')

        # Media
        ax.axhline(mean_val, color='white',
                linewidth=1,
                linestyle='--',
                alpha=0.8)

        # Intervalo de confianza
        ax.axhspan(ci_low, ci_high,
                alpha=0.15,
                color='white')

        ax.text(
            df['sim'].iloc[-1] + 0.2,
            mean_val,
            f'μ={mean_val:.2f}\n95% CI [{ci_low:.2f}, {ci_high:.2f}]',
            color='white',
            fontsize=8,
            va='center'
        )

    plt.suptitle('DPLACE Model – Simulation Summary', color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def generate_devices(
    N,
    area,
    device_height=1.2,
    seed=None
):
    """Genera dispositivos IoT en 3D con altura fija."""

    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, area, N)
    y = np.random.uniform(0, area, N)

    # Altura fija típica de sensor
    z = np.full(N, device_height)

    return np.column_stack((x, y, z))



def generate_devices(N, area, device_height=1.2, seed=None):
    """
    Genera un escenario estático con exactamente n dispositivos
    distribuidos de forma uniforme en el área.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.uniform(0, area, N)
    y = np.random.uniform(0, area, N)
    z = np.full(N, device_height)
    
    return np.column_stack((x, y, z))


def generate_devices_poisson_frame(
    N,
    area,
    device_height=1.2,
    seed=None
):
    """
    Genera UN frame temporal de dispositivos con distribución espacial de Poisson.
    
    El número de dispositivos reales en este frame sigue: n_devices ~ Poisson(λ)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # El número de dispositivos para este snapshot sigue una distribución Poisson(λ)
    n_devices = np.random.poisson(N)
    
    # Manejo defensivo en caso de que Poisson resulte en 0
    if n_devices == 0:
        return np.empty((0, 3)), 0
    
    x = np.random.uniform(0, area, n_devices)
    y = np.random.uniform(0, area, n_devices)
    z = np.full(n_devices, device_height)
    
    devices = np.column_stack((x, y, z))
    return devices, n_devices


def generate_devices_poisson_aggregated(
    N,
    area,
    n_frames,
    device_height=1.2,
    base_seed=None
):
    """
    Genera múltiples frames temporales y los acumula verticalmente.
    
    Simula el dinamismo de una red IoT donde la densidad y las posiciones
    de los dispositivos activos varían en cada instante de tiempo.
    
    Args:
        lambda_mean: Tasa esperada de dispositivos activos por frame (λ)
        area: Lado del área cuadrada de simulación (por ejemplo, 10 para 100 km²)
        n_frames: Número de instantes de tiempo/snapshots a acumular
        device_height: Altura de instalación de los dispositivos (m)
        base_seed: Semilla para reproducibilidad
    
    Returns:
        aggregated_devices (array): Todas las coordenadas (x,y,z) acumuladas temporalmente.
        static_devices (array): Escenario estático base con exactamente lambda_mean dispositivos.
        df_stats (DataFrame): Estadísticas de dispositivos activos por cada frame.
    """
    if base_seed is not None:
        np.random.seed(base_seed)
    
    all_frames = []
    frame_data = []
    
    for frame_idx in range(n_frames):
        # Semilla única y reproducible por frame
        frame_seed = base_seed + frame_idx if base_seed is not None else None
        
        X_frame, n_frame = generate_devices_poisson_frame(
            N=N,
            area=area,
            device_height=device_height,
            seed=frame_seed
        )
        
        if n_frame > 0:
            all_frames.append(X_frame)
        
        frame_data.append({
            'frame': frame_idx,
            'n_devices': n_frame,
            'lambda': N
        })

    # Generar el escenario de comparación estático original (los 1,000 nodos fijos)
    static_devices = generate_devices(
        N=N, 
        area=area, 
        device_height=device_height, 
        seed=base_seed
    )
    
    # Agregación: Unir todos los escenarios temporales en una sola matriz
    if all_frames:
        aggregated_devices = np.vstack(all_frames)
    else:
        aggregated_devices = np.empty((0, 3))
    
    return static_devices

def plot_poisson_dynamism(frame_stats_df, sim_num, filepath):
    """
    Visualiza el dinamismo de Poisson:
    - Número de dispositivos por frame
    - Variabilidad alrededor de λ
    - Distribución observada vs teórica
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
    
    # ─── SUBPLOT 1: Número de dispositivos por frame ───
    ax = axes[0, 0]
    frames = frame_stats_df['frame'].values
    n_devices = frame_stats_df['n_devices'].values
    lambda_val = frame_stats_df['lambda'].iloc[0]
    
    bars = ax.bar(frames, n_devices, color='#58a6ff', alpha=0.7, edgecolor='#30363d')
    ax.axhline(lambda_val, color='#f78166', linestyle='--', linewidth=2, label=f'λ = {lambda_val}')
    ax.set_title('Dispositivos por Frame (Poisson)', color='white', fontsize=10)
    ax.set_xlabel('Frame', color='#8b949e')
    ax.set_ylabel('# Dispositivos', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 2: Desviación de λ ───
    ax = axes[0, 1]
    deviation = n_devices - lambda_val
    colors = ['#3fb950' if d >= 0 else '#f78166' for d in deviation]
    ax.bar(frames, deviation, color=colors, alpha=0.7, edgecolor='#30363d')
    ax.axhline(0, color='white', linestyle='-', linewidth=0.8)
    ax.set_title('Desviación de λ (Variabilidad)', color='white', fontsize=10)
    ax.set_xlabel('Frame', color='#8b949e')
    ax.set_ylabel('Δ Dispositivos', color='#8b949e')
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 3: Distribución observada ───
    ax = axes[1, 0]
    counts, bins, patches = ax.hist(n_devices, bins=15, color='#58a6ff', alpha=0.7, 
                                     edgecolor='#30363d', density=True)
    
    # Sobreponer Poisson teórica
    from scipy.stats import poisson
    x_poisson = np.arange(n_devices.min(), n_devices.max() + 1)
    y_poisson = poisson.pmf(x_poisson, lambda_val)
    ax.plot(x_poisson, y_poisson, 'o-', color='#f78166', linewidth=2, 
            markersize=6, label='Poisson(λ) teórica')
    
    ax.set_title('Distribución de Dispositivos', color='white', fontsize=10)
    ax.set_xlabel('# Dispositivos', color='#8b949e')
    ax.set_ylabel('Densidad', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 4: Estadísticas ───
    ax = axes[1, 1]
    ax.axis('off')
    
    n_total = n_devices.sum()
    n_mean = n_devices.mean()
    n_std = n_devices.std()
    n_min = n_devices.min()
    n_max = n_devices.max()
    cv = n_std / n_mean if n_mean > 0 else 0  # Coeficiente de variación
    
    stats_text = f"""
    ESTADÍSTICAS POISSON
    
    λ (esperada): {lambda_val}
    μ (observada): {n_mean:.1f}
    σ (desv est): {n_std:.1f}
    CV (coef var): {cv:.3f}
    
    Total dispositivos: {n_total}
    Mínimo por frame: {n_min}
    Máximo por frame: {n_max}
    Rango: {n_max - n_min}
    
    # Frames: {len(n_devices)}
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='white',
            family='monospace', bbox=dict(boxstyle='round', 
            facecolor='#21262d', edgecolor='#30363d', alpha=0.9))
    
    plt.suptitle(f'Dinamismo Poisson - Simulación {sim_num}', 
                 color='white', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_aggregated_devices_3d(N, sim_num, filepath):
    """
    Visualiza los dispositivos agregados de todos los frames.
    Muestra la cobertura compuesta que verá FCM.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Scatter 2D (ignorar z por ahora)
    ax.scatter(N[:, 0], N[:, 1], 
               s=4, color='#58a6ff', alpha=0.5)
    
    n_total = len(N)
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(f'Dispositivos Agregados - Sim {sim_num}\n' + 
                 f'Total: {n_total}',
                 color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_availability_set(
    all_gateways_list,
    all_devices_list,
    filepath,
    coverage_radius=None
):
    """
    Visualiza el AVAILABILITY SET: todas las ubicaciones de gateways
    de todas las simulaciones en un solo gráfico.
    
    Muestra:
    - Todos los dispositivos agregados (fondo)
    - Todas las ubicaciones de gateways (puntos de color)
    - Centroide final (estrella roja)
    - Heatmap de densidad de gateways
    
    Esto demuestra la RESILIENCIA: aunque los gateways varían,
    se concentran en zonas similares (robustez).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes:
        ax.set_facecolor('#0d1117')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
    
    # ─── SUBPLOT 1: Scatter de gateways ───
    ax = axes[0]
    
    # Todos los dispositivos (fondo)
    all_devices_combined = np.vstack(all_devices_list)
    ax.scatter(
        all_devices_combined[:, 0], all_devices_combined[:, 1],
        s=2, color='#6e7681', alpha=0.1, label='IoT Devices'
    )
    
    # Todas las ubicaciones de gateways
    all_gw_coords = np.vstack(all_gateways_list)
    scatter = ax.scatter(
        all_gw_coords[:, 0], all_gw_coords[:, 1],
        s=150, c=range(len(all_gw_coords)), cmap='viridis',
        alpha=0.6, edgecolors='white', linewidths=0.8,
        label='Gateway candidates'
    )
    
    # Centroide final (si existen gateways)
    if len(all_gw_coords) > 0:
        final_centroid = np.mean(all_gw_coords, axis=0)
        ax.scatter(
            final_centroid[0], final_centroid[1],
            s=500, marker='*', color='#f78166',
            edgecolors='white', linewidths=1.5, zorder=10,
            label='Final centroid'
        )
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title('Availability Set: Gateway Candidates\n(All simulations)', 
                 color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    
    # ─── SUBPLOT 2: Heatmap de densidad ───
    ax = axes[1]
    
    # Crear heatmap 2D de densidad
    hist, xedges, yedges = np.histogram2d(
        all_gw_coords[:, 0], all_gw_coords[:, 1],
        bins=20, range=[[0, AXIS_RANGE], [0, AXIS_RANGE]]
    )
    
    im = ax.imshow(
        hist.T, origin='lower',
        extent=[0, AXIS_RANGE, 0, AXIS_RANGE],
        cmap='hot', aspect='auto', alpha=0.8
    )
    
    # Overlay de dispositivos
    ax.scatter(
        all_devices_combined[:, 0], all_devices_combined[:, 1],
        s=2, color='cyan', alpha=0.05
    )
    
    # Overlay de gateways finales
    ax.scatter(
        all_gw_coords[:, 0], all_gw_coords[:, 1],
        s=80, marker='x', color='white', linewidths=1.5, alpha=0.7
    )
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title('Density Heatmap: Gateway Concentration\n(Robustness indicator)', 
                 color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gateway Density', color='#8b949e')
    cbar.ax.tick_params(colors='#8b949e')
    
    plt.suptitle('DPLACE Phase 1 — Availability Set (Poisson Resilience)', 
                 color='white', fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_gateway_resilience_stats(all_gateways_list, filepath):

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
    
    # Combinar todas las ubicaciones
    all_gw = np.vstack(all_gateways_list)
    
    # ─── SUBPLOT 1: Distribución X ───
    ax = axes[0, 0]
    ax.hist(all_gw[:, 0], bins=20, color='#58a6ff', alpha=0.7, edgecolor='#30363d')
    mean_x = np.mean(all_gw[:, 0])
    std_x = np.std(all_gw[:, 0])
    ax.axvline(mean_x, color='#f78166', linestyle='--', linewidth=2, label=f'μ={mean_x:.0f}')
    ax.axvline(mean_x - std_x, color='#3fb950', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'±σ={std_x:.0f}')
    ax.axvline(mean_x + std_x, color='#3fb950', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_title('Gateway X Distribution', color='white', fontsize=10)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Count', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 2: Distribución Y ───
    ax = axes[0, 1]
    ax.hist(all_gw[:, 1], bins=20, color='#3fb950', alpha=0.7, edgecolor='#30363d')
    mean_y = np.mean(all_gw[:, 1])
    std_y = np.std(all_gw[:, 1])
    ax.axvline(mean_y, color='#f78166', linestyle='--', linewidth=2, label=f'μ={mean_y:.0f}')
    ax.axvline(mean_y - std_y, color='#58a6ff', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'±σ={std_y:.0f}')
    ax.axvline(mean_y + std_y, color='#58a6ff', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_title('Gateway Y Distribution', color='white', fontsize=10)
    ax.set_xlabel('Y (m)', color='#8b949e')
    ax.set_ylabel('Count', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 3: Estadísticas de Resiliencia ───
    ax = axes[1, 0]
    ax.axis('off')
    
    # Calcular métricas
    distances_from_mean = np.sqrt(
        (all_gw[:, 0] - mean_x)**2 + 
        (all_gw[:, 1] - mean_y)**2
    )
    
    stats_text = f"""
    RESILIENCE METRICS
    (Poisson Aggregated)
    
    Gateway Candidates: {len(all_gw)}
    
    Position X:
      μ: {mean_x:.1f} m
      σ: {std_x:.1f} m
      CV: {std_x/mean_x if mean_x>0 else 0:.3f}
    
    Position Y:
      μ: {mean_y:.1f} m
      σ: {std_y:.1f} m
      CV: {std_y/mean_y if mean_y>0 else 0:.3f}
    
    Distance from Mean:
      μ: {np.mean(distances_from_mean):.1f} m
      σ: {np.std(distances_from_mean):.1f} m
      Max: {np.max(distances_from_mean):.1f} m
    
    Concentration:
      Within ±σ: {np.sum(distances_from_mean < np.mean(distances_from_mean)+np.std(distances_from_mean))/len(all_gw)*100:.1f}%
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='white',
            family='monospace', bbox=dict(boxstyle='round', 
            facecolor='#21262d', edgecolor='#30363d', alpha=0.9))
    

    axes[1, 1].axis('off')
    
    plt.suptitle('Gateway Resilience Analysis (Poisson Pattern)', 
                 color='white', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_global_poisson_variability(poisson_stats_list, filepath):
    """
    Gráfico GLOBAL: Muestra cómo varía la variabilidad Poisson
    entre TODAS las simulaciones.
    
    Visualiza:
    - Desviación estándar por simulación
    - Coeficiente de variación por simulación
    - Rango (min-max) por simulación
    - Resumen comparativo
    
    Esto responde: ¿Cuál simulación tuvo más variabilidad?
    ¿Cuál tuvo menos? ¿Hay patrón?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
    
    # Extraer datos
    sim_numbers = np.array([s['sim'] for s in poisson_stats_list])
    std_devs = np.array([s['std'] for s in poisson_stats_list])
    cvs = np.array([s['cv'] for s in poisson_stats_list])
    ranges = np.array([s['range'] for s in poisson_stats_list])
    means = np.array([s['mean'] for s in poisson_stats_list])
    
    # ─── SUBPLOT 1: Desviación Estándar por Simulación ───
    ax = axes[0, 0]
    bars = ax.bar(sim_numbers, std_devs, color='#58a6ff', alpha=0.7, edgecolor='#30363d')
    mean_std = np.mean(std_devs)
    ax.axhline(mean_std, color='#f78166', linestyle='--', linewidth=2, 
               label=f'μ(σ) = {mean_std:.1f}')
    ax.fill_between(sim_numbers, mean_std - np.std(std_devs), 
                     mean_std + np.std(std_devs), alpha=0.1, color='#f78166')
    ax.set_title('Desviación Estándar por Simulación', color='white', fontsize=10)
    ax.set_xlabel('Simulación', color='#8b949e')
    ax.set_ylabel('σ (dispositivos)', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 2: Coeficiente de Variación ───
    ax = axes[0, 1]
    bars = ax.bar(sim_numbers, cvs, color='#3fb950', alpha=0.7, edgecolor='#30363d')
    mean_cv = np.mean(cvs)
    ax.axhline(mean_cv, color='#f78166', linestyle='--', linewidth=2, 
               label=f'μ(CV) = {mean_cv:.3f}')
    ax.fill_between(sim_numbers, mean_cv - np.std(cvs), 
                     mean_cv + np.std(cvs), alpha=0.1, color='#f78166')
    ax.set_title('Coeficiente de Variación (CV = σ/μ)', color='white', fontsize=10)
    ax.set_xlabel('Simulación', color='#8b949e')
    ax.set_ylabel('CV', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 3: Rango (Min-Max) ───
    ax = axes[1, 0]
    bars = ax.bar(sim_numbers, ranges, color='#d2a8ff', alpha=0.7, edgecolor='#30363d')
    mean_range = np.mean(ranges)
    ax.axhline(mean_range, color='#f78166', linestyle='--', linewidth=2, 
               label=f'μ(Rango) = {mean_range:.1f}')
    ax.fill_between(sim_numbers, mean_range - np.std(ranges), 
                     mean_range + np.std(ranges), alpha=0.1, color='#f78166')
    ax.set_title('Rango (Max - Min) por Simulación', color='white', fontsize=10)
    ax.set_xlabel('Simulación', color='#8b949e')
    ax.set_ylabel('Rango (dispositivos)', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, axis='y')
    
    # ─── SUBPLOT 4: Resumen Comparativo ───
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    VARIABILIDAD POISSON — RESUMEN GLOBAL
    
    Número de Simulaciones: {len(poisson_stats_list)}
    
    DESVIACIÓN ESTÁNDAR (σ):
      μ: {mean_std:.2f} dispositivos
      σ: {np.std(std_devs):.2f}
      Mín: {np.min(std_devs):.2f}
      Máx: {np.max(std_devs):.2f}
      Rango: {np.max(std_devs) - np.min(std_devs):.2f}
    
    COEF. VARIACIÓN (CV):
      μ: {mean_cv:.4f}
      σ: {np.std(cvs):.4f}
      Mín: {np.min(cvs):.4f}
      Máx: {np.max(cvs):.4f}
    
    RANGO (Max-Min):
      μ: {mean_range:.1f} dispositivos
      σ: {np.std(ranges):.1f}
      Mín: {np.min(ranges):.1f}
      Máx: {np.max(ranges):.1f}
    
    INTERPRETACIÓN:
    → Mayor CV = Más variabilidad relativa
    → Menor CV = Más estabilidad Poisson
    → Rango grande = Más dispersión de frames
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', color='white',
            family='monospace', bbox=dict(boxstyle='round', 
            facecolor='#21262d', edgecolor='#30363d', alpha=0.9))
    
    plt.suptitle('DPLACE Phase 1 — Global Poisson Variability Analysis', 
                 color='white', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def save_checkpoint(sim_index, all_gateways, all_results, all_k):
    """Guarda el estado actual después de completar la simulación sim_index (1-based)."""
    state = {
        'last_completed_sim': sim_index,
        'total_simulations': N_SIMULATIONS, #sim
        'all_k_results': all_k,
        # Guardamos las rutas de los archivos de gateways y resultados (no los objetos en JSON)
        'gateways_files': [f'sim{i+1}_gateways_pre.csv' for i in range(sim_index)],
        'results_files': [f'sim{i+1}_pre_result.pkl' for i in range(sim_index)]
    }
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"{type(obj)} no es serializable")
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=convert_numpy)
    
    # Además, guardamos los arrays acumulados en disco (opcional, pero útil para reanudar)
    # Mejor guardarlos como objetos pickle
    with open(os.path.join(OUTPUT_DIR, 'all_gateways.pkl'), 'wb') as f:
        pickle.dump(all_gateways, f)
    with open(os.path.join(OUTPUT_DIR, 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    # all_k y poisson_stats ya están en el JSON, pero también podemos guardarlos en pickle si queremos

def load_checkpoint():
    """Carga el estado guardado, devuelve (start_sim, all_gateways, all_results, poisson_stats, all_k)."""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0, [], [], []  # empezar desde cero
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            state = json.load(f)

        last = state['last_completed_sim']
        # Cargar los arrays acumulados
        with open(os.path.join(OUTPUT_DIR, 'all_gateways.pkl'), 'rb') as f:
            all_gateways = pickle.load(f)
        with open(os.path.join(OUTPUT_DIR, 'all_results.pkl'), 'rb') as f:
            all_results = pickle.load(f)
        
        all_k = state['all_k_results']
        
        # Verificar consistencia (número de elementos coincide con last)
        assert len(all_gateways) == last
        assert len(all_results) == last
        assert len(all_k) == last
        
        return last, all_gateways, all_results, all_k
    except Exception:
        # Si algo falla, empezar desde cero (opcional: borrar checkpoint corrupto)
        return 0, [], [], []
    
# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  DPLACE Model — LoRaWAN Gateway Placement (Poisson Dynamics)")
    print("=" * 70)
    print(f"  Simulations             : {N_SIMULATIONS}")
    print(f"  λ (Expected devices)    : {LAMBDA_MEAN}")
    print(f"  Temporal frames/sim     : {N_FRAMES}")
    print(f"  Total scenarios         : {N_SIMULATIONS * N_FRAMES} (Poisson aggregated)")
    print(f"  Area                    : {AXIS_RANGE} m x {AXIS_RANGE} m")
    print(f"  Coverage Radius         : {COVERAGE_RADIUS} m")
    print("=" * 70)
    print()

    all_preprocessing_gateways = []
    all_preprocessing_results = []
    all_k_results = []

    # ─── PHASE 1: PRE-PROCESSING ───────────────
    print("─" * 50)
    print("PHASE 1 — PRE-PROCESSING")
    print("─" * 50)

    # Gap Statistics parameters
    NREFS_PRE = 80      # References for preprocessing
    NREFS_PROC = 90     # References for processing
    BASE_SEED = 42
    USE_POISSON_PATTERN = True

    # Intentar cargar checkpoint
    start_sim, all_preprocessing_gateways, all_preprocessing_results, all_k_results = load_checkpoint()
    
    if start_sim > 0:
        print(f"Reanudando desde la simulación {start_sim+1} (ya completadas {start_sim})")
    else:
        print("Iniciando desde cero")
    
    for sim in range(start_sim, N_SIMULATIONS):
        print(f"\n{'='*60}")
        print(f"SIMULACION NUMERO {sim+1}")
        print(f"{'='*60}")

        seed = BASE_SEED + sim
        # ════════════════════════════════════════════════════════════
        # GENERAR DISPOSITIVOS (según modo seleccionado)
        # ════════════════════════════════════════════════════════════
        if USE_POISSON_PATTERN:
            print(f"\n[Paso 1] Generando {N_FRAMES} frames con distribución Poisson(λ={LAMBDA_MEAN})...")
            n = generate_devices_poisson_aggregated(
                lambda_mean=LAMBDA_MEAN,
                area=AXIS_RANGE,
                n_frames=N_FRAMES,
                base_seed=seed
            )
            n_total = len(n)
            print(f"  ✓ Total dispositivos agregados: {n_total}")

            try:
                plot_aggregated_devices_3d(
                    N,
                    sim+1,
                    os.path.join(OUTPUT_DIR, f'sim{sim+1}_aggregated_devices.png')
                )
                print(f"  ✓ Gráfico aggregated_devices generado")
            except Exception as e:
                print(f"  ✗ ERROR en aggregated_devices: {e}")
            
        else:
            print(f"\n[Paso 1] Generando {N} dispositivos fijos uniformes...")
            n = generate_devices(N=N, area=AXIS_RANGE, device_height=1.2, seed=seed)
            n_total = len(n)
            print(f"  ✓ Total dispositivos: {n_total}")

        # ════════════════════════════════════════════════════════════
        # PASO 3: EJECUTAR PREPROCESSING CON DISPOSITIVOS AGREGADOS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 3] Ejecutando preprocessing (FCM) con dispositivos agregados...")
        
        try:
            pre = preprocessing(
                N,
                nrefs=NREFS_PRE,
                verbose=True,
                AREA_SIZE=AXIS_RANGE
            )
            print(f"  ✓ Preprocessing completado")

        except Exception as e:
            print(f"  ✗ ERROR en preprocessing: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save preprocessing plots
        try:
            plot_gap_curve(
                pre['gap_df'], pre['gap_sk_df'], pre['k_final'],
                f'(FCM – Sim {sim+1})',
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_gap_fcm.png')
            )
        except Exception as e:
            print(f"  ✗ ERROR en gap_fcm: {e}")
        
        try:
            plot_devices_and_gateways(
                N, pre['gateways'],
                f"FCM Clustering (Poisson) – Sim {sim+1} (k={pre['k_final']})",
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_fcm_result.png'),
                coverage_radius=COVERAGE_RADIUS,
                u=pre['u']
            )
        except Exception as e:
            print(f"  ✗ ERROR en fcm_result: {e}")

        all_preprocessing_gateways.append(pre['gateways'])
        all_preprocessing_results.append(pre)
        all_k_results.append(int(pre['k_gap']))

        # ════════════════════════════════════════════════════════════
        # PASO 4: GUARDAR RESULTADOS Y ESTADÍSTICAS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 4] Guardando resultados...")
        
        try:
            # Save aggregated devices CSV (con 3 dimensiones: x, y, z)
            dev_df = pd.DataFrame(N, columns=['x', 'y', 'z'])
            dev_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_devices_poisson.csv'), index=False
            )
            print(f"  ✓ Guardado: sim{sim+1}_devices_poisson.csv")
        except Exception as e:
            print(f"  ✗ ERROR guardando devices CSV: {e}")

        try:
            # Save gateway CSV for this simulation (x,y de centroide, z=30 para gateways)
            gw_array = np.column_stack((pre['gateways'][:, :2], np.full(len(pre['gateways']), 30)))
            gw_df = pd.DataFrame(gw_array, columns=['x', 'y', 'z'])
            gw_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_gateways_pre.csv'), index=False
            )
            print(f"  ✓ Guardado: sim{sim+1}_gateways_pre.csv")
        except Exception as e:
            print(f"  ✗ ERROR guardando gateways CSV: {e}")
        
                # Guardar resultado completo en pickle (para poder reanudar sin rehacer)
        with open(os.path.join(OUTPUT_DIR, f'sim{sim+1}_pre_result.pkl'), 'wb') as f:
            pickle.dump(pre, f)
        

        # Guardar checkpoint después de cada simulación
        save_checkpoint(sim+1, all_preprocessing_gateways, all_preprocessing_results, all_k_results)
        
        print(f"  ✓ Checkpoint guardado después de sim {sim+1}")

    # ═══════════════════════════════════════════════════════════════════
    # POST-PHASE 1: VISUALIZATION OF AVAILABILITY SET & RESILIENCE
    # ═══════════════════════════════════════════════════════════════════
    if len(all_preprocessing_gateways) < N_SIMULATIONS:
        print("No se completaron todas las simulaciones, no se ejecuta Phase 2.")
    else:
        plot_k(
            all_k_results,
            os.path.join(OUTPUT_DIR, f'k_final_analysis.png')
        )

        print("\n" + "=" * 70)

        # Visualización 2: Análisis de resiliencia
        # Visualización 3: VARIABILIDAD GLOBAL POISSON (NEW)

        # ─── PHASE 2: PROCESSING ───────────────────
        print("\n" + "─" * 50)
        print("PHASE 2 — PROCESSING")
        print("─" * 50)

        proc = processing(
            all_preprocessing_gateways,
            nrefs=NREFS_PROC,
            verbose=True
        )

        # Save processing plots
        plot_gap_curve(
            proc['gap_df'], proc['gap_sk_df'], proc['k_optimal'],
            '(K-Means – Processing)',
            os.path.join(OUTPUT_DIR, 'processing_gap_kmeans.png')
        )

        # Plot all collected gateways + final centroids + devices
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')
        
        # Graficar todos los dispositivos de todas las simulaciones
        n = generate_devices(N=N, area=AXIS_RANGE, device_height=1.2, seed=BASE_SEED)

        all_devices = np.vstack(n)
        ax.scatter(
            all_devices[:, 0], all_devices[:, 1],
            s=5, zorder=5, color='#58a6ff', alpha=0.7, label='IoT Devices'
        )
        
        # Gateways finales después de K-Means
        ax.scatter(
            proc['gateways'][:, 0], proc['gateways'][:, 1],
            s=200, marker='^', color='#ffa657', edgecolors='white',
            linewidths=1.5, zorder=4, label=f'Final GWs (K={proc["k_optimal"]})'
        )
        
        # Coverage circles para gateways finales
        for j, gw in enumerate(proc['gateways']):
            circle = plt.Circle(
                (gw[0], gw[1]), COVERAGE_RADIUS,
                color='#ffa657', fill=False, linestyle='--', linewidth=1, alpha=0.4
            )
            ax.add_patch(circle)
        
        ax.set_xlim(0, AXIS_RANGE)
        ax.set_ylim(0, AXIS_RANGE)
        ax.set_title(f'Processing Phase – Gateways & Devices (K={proc["k_optimal"]})',
                    color='white', fontsize=11)
        ax.set_xlabel('X (m)', color='#8b949e')
        ax.set_ylabel('Y (m)', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'processing_final_gateways2.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # Save final nodos CSV (x,y,z). N ya tiene forma (N, 3).
        CUADRO_X_MIN =  724900 
        CUADRO_Y_MIN =  868100

        # Save final gateways CSV (x,y de centroide, z=30 para gateways)
        gw_final = proc['gateways'].copy()
        gw_final[:, 0] += CUADRO_X_MIN
        gw_final[:, 1] += CUADRO_Y_MIN
        gw_final_array = np.column_stack((gw_final[:, :2], np.full(len(gw_final), 30)))
        gw_final_df = pd.DataFrame(gw_final_array, columns=['x', 'y', 'z'])
        gw_final_df.insert(0, 'id', np.arange(len(gw_final_df)))
        gw_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_gateways.csv'), index=False)

        nodo_final = n.copy()
        nodo_final[:, 0] += CUADRO_X_MIN
        nodo_final[:, 1] += CUADRO_Y_MIN
        nodo_final_df = pd.DataFrame(nodo_final, columns=['x', 'y', 'z'])
        nodo_final_df.insert(0, 'id', np.arange(len(nodo_final_df)))
        nodo_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_nodos.csv'), index=False)


        # Save all aggregated gateways in the same real coordinate system.
        gw_all = proc['all_gw_data'].copy()
        gw_all[:, 0] += CUADRO_X_MIN
        gw_all[:, 1] += CUADRO_Y_MIN
        gw_all_array = np.column_stack((gw_all[:, :2], np.full(len(gw_all), 1.2)))
        gw_all_df = pd.DataFrame(gw_all_array, columns=['x', 'y', 'z'])
        gw_all_df.to_csv(os.path.join(OUTPUT_DIR, 'all_preprocessing_gateways.csv'), index=False)




     # ─── Cargar mapa de inseguridad de Cali ───
        print("Cargando mapa de inseguridad de Cali...")
        grid_cali_path = os.path.join(PROJECT_ROOT, "images", "grid_cali.geojson")
        # (minx, maxx, miny, maxy), en EPSG:3116.

        BBOX_CALI = (720414.01, 735029.27, 860572.64, 879817.30)
        
        if not os.path.exists(grid_cali_path):
            print(f" No se encontró {grid_cali_path}. Genera el grid con georeferencia.py primero.")
            grid_cali = None
        else:
            grid_cali = gpd.read_file(grid_cali_path)
            # Las coordenadas del grid están en metros (EPSG:3116). Algunos
            # GeoJSON guardados previamente declaran EPSG:4326 por error.
            grid_cali = grid_cali.set_crs("EPSG:3116", allow_override=True)
            print(f"  ✓ Grid cargado: {len(grid_cali)} celdas")
        
        # El grid local también sirve como mapa base. Evita depender de
        # Nominatim/Internet, que puede fallar por certificados TLS o proxy.
        if grid_cali is not None and not grid_cali.empty:
            try:
                grid_union = grid_cali.geometry.union_all()
            except AttributeError:  # Compatibilidad con versiones antiguas de GeoPandas
                grid_union = grid_cali.geometry.unary_union
            cali = gpd.GeoDataFrame(
                geometry=[grid_union],
                crs=grid_cali.crs or "EPSG:3116"
            )
        else:
            cali = None

        # ─── GRÁFICO FINAL: Mapa de Calor + Gateways + Dispositivos ───
        print("\nGenerando gráfico final con mapa de calor...")
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(7.16, 5.37))

        # 1. Mapa base de Cali
        if cali is not None:
            cali.plot(ax=ax, color="white", edgecolor="black", linewidth=1.0, zorder=1)

        # 2. Límites = BBOX completo de Cali
        ax.set_xlim(BBOX_CALI[0], BBOX_CALI[1])
        ax.set_ylim(BBOX_CALI[2], BBOX_CALI[3])

        # 3. Mapa de calor de inseguridad
        if grid_cali is not None and not grid_cali.empty:
            colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
            cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
            grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                        alpha=0.75, edgecolor="black", linewidth=0.3,
                        vmin=0, vmax=5, zorder=3)
            
        # Función de transformación (usando el nuevo bbox)
        def transformar(coords, v):
            minx, maxx, miny, maxy = BBOX_DESTINO
            x = minx + (coords[:, 0] / AXIS_RANGE) * (maxx - minx)
            y = miny + (coords[:, 1] / AXIS_RANGE) * (maxy - miny)
            return np.column_stack((x, y))

        # ─── Definir el rectángulo de destino ───
        x_ini = 724900
        y_ini = 868100
        ancho = 5900
        alto = 5900
        x_fin = x_ini + ancho
        y_fin = y_ini + alto
        BBOX_DESTINO = (x_ini, x_fin, y_ini, y_fin)   # (minx, maxx, miny, maxy)

        # ─── Obtener los datos a transformar ───
        gw_centroids = proc['gateways']   # shape (K, 3)

        # ─── Calcular los límites reales de los datos ───
        x_min_data = min(all_devices[:, 0].min(), gw_centroids[:, 0].min())
        x_max_data = max(all_devices[:, 0].max(), gw_centroids[:, 0].max())
        y_min_data = min(all_devices[:, 1].min(), gw_centroids[:, 1].min())
        y_max_data = max(all_devices[:, 1].max(), gw_centroids[:, 1].max())

        # Añadir un pequeño margen (5%) para que no queden pegados a los bordes
        margin_x = (x_max_data - x_min_data) * 0.05
        margin_y = (y_max_data - y_min_data) * 0.05
        x_min_data -= margin_x
        x_max_data += margin_x
        y_min_data -= margin_y
        y_max_data += margin_y

        # ─── Función de transformación usando los límites reales ───
        def transformar_con_limites(coords):
            """Mapea las coordenadas desde [x_min_data, x_max_data] × [y_min_data, y_max_data]
            al rectángulo destino BBOX_DESTINO."""
            minx, maxx, miny, maxy = BBOX_DESTINO
            # Escalar x
            x = minx + (coords[:, 0] - x_min_data) / (x_max_data - x_min_data) * (maxx - minx)
            # Escalar y
            y = miny + (coords[:, 1] - y_min_data) / (y_max_data - y_min_data) * (maxy - miny)
            return np.column_stack((x, y))

        # ─── Aplicar transformación ───
        nodes_utm = transformar_con_limites(all_devices[:, :2])  # solo x, y
        gw_utm = transformar_con_limites(gw_centroids[:, :2])

        rect = Rectangle(
            (x_ini, y_ini), ancho, alto,
            linewidth=2.5, edgecolor='deepskyblue', facecolor='aqua',
            alpha=0.25, zorder=4
        )
        ax.add_patch(rect)
  
        # 5. Nodos DPLACE (preprocesamiento)
        nodes_utm = transformar(all_devices, BBOX_CALI)
        ax.scatter(nodes_utm[:, 0], nodes_utm[:, 1], s=3,
                marker='o', color='#00b4d8', alpha=0.75,
                edgecolors='white', linewidths=0.25,
                zorder=6)

        # 6. Gateways finales (centroides de K-Means)
        if proc is not None and 'gateways' in proc:
            gw_utm = transformar(proc['gateways'], BBOX_CALI)
            ax.scatter(gw_utm[:, 0], gw_utm[:, 1], s=20,
                    marker='^', color='black', edgecolors='black',
                    linewidths=1.0, zorder=7)

 
        # 8. Configuración final
        ax.set_title('DPLACE – Gateways Optimizados sobre Mapa de Inseguridad de Cali',
                    fontsize=10, color='black')
        ax.set_xlabel('X Coordinate (m)', fontsize=8)
        ax.set_ylabel('Y Coordinate (m)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'dplace_gateways_on_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()




        # ─── SUMMARY ───────────────────────────────
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)

        print(f"  Final gateways (K={proc['k_optimal']}): final_gateways.csv")
        print(f"  All preprocessing gateways: all_preprocessing_gateways.csv")
        print(f"  Summary metrics: summary.csv")

if __name__ == '__main__':
    main()
