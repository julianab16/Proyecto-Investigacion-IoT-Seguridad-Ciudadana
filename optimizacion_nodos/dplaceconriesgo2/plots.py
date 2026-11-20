import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

AXIS_RANGE = 5900

def plot_devices_and_gateways(X, gateways, title, filepath, AXIS_RANGE, coverage_radius=None, u=None):
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

    plt.figure(figsize=(12, 5))

    bars = plt.bar(
        x,
        all_k_results,
        color='steelblue',
        edgecolor='black'
    )

    # Mostrar el valor de K encima de cada barra
    for bar, k in zip(bars, all_k_results):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            str(k),
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.xlabel('Muestra')
    plt.ylabel('K óptimo')
    plt.title('Valor de K obtenido en cada muestra')
    plt.xticks(x)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    ax2.set_title(f'Gap[k] − Gap[k+1] + sk[k+1] {title_suffix}', color='white', fontsize=10)
    ax2.set_xlabel('Cluster Count', color='#8b949e')
    ax2.set_ylabel('Gap_sk', color='#8b949e')
    ax2.grid(True, color='#21262d', linewidth=0.5)
    ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def plot_poisson_final(X, sim_num, filepath, AXIS_RANGE):

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Scatter 2D (ignorar z por ahora)
    ax.scatter(X[:, 0], X[:, 1],   
               s=4, color='#58a6ff', alpha=0.5)
    
    n_total = len(X)
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(f'Dispositivos Agregados - Sim {sim_num}\n' + 
                 f'Total: {n_total}', color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def plot_aggregated_devices_3d(X_aggregated, frame_stats_df, sim_num, filepath, AXIS_RANGE):
    """
    Visualiza los dispositivos agregados de todos los frames.
    Muestra la cobertura compuesta que verá FCM.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Scatter 2D (ignorar z por ahora)
    ax.scatter(X_aggregated[:, 0], X_aggregated[:, 1], 
               s=4, color='#58a6ff', alpha=0.5)
    
    n_total = len(X_aggregated)
    n_frames = len(frame_stats_df)
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(f'Dispositivos Agregados - Sim {sim_num}\n' + 
                 f'Total: {n_total} | Frames: {n_frames} | λ: {frame_stats_df["lambda"].iloc[0]}',
                 color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

def plot_gap_inteval(clusters, gaps, filepath):

    intervalos = [(1, 4), (4, 12), (12, 15)]  # (inicio, fin) inclusive
    pendientes = []
    etiquetas = []

    for inicio, fin in intervalos:
        # Seleccionar puntos dentro del rango [inicio, fin]
        mask = (clusters >= inicio) & (clusters <= fin)
        if np.sum(mask) < 2:
            pendientes.append(np.nan)
            etiquetas.append(f'{inicio}-{fin}')
            continue
        x = clusters[mask]
        y = gaps[mask]
        # Ajuste lineal: pendiente = coeficiente de x
        coef = np.polyfit(x, y, 1)
        pendientes.append(coef[0])
        etiquetas.append(f'{inicio}-{fin}')

    # Crear la gráfica que realmente se guarda.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    bars = ax.bar(etiquetas, pendientes, color=['blue', 'orange', 'green'])

    # Mostrar el valor de la pendiente sobre cada barra de esta figura.
    for bar, pendiente in zip(bars, pendientes):
        if np.isnan(pendiente):
            continue
        ax.annotate(
            f'{pendiente:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if pendiente >= 0 else -4),
            textcoords='offset points',
            color='#8b949e',
            ha='center',
            va='bottom' if pendiente >= 0 else 'top',
            fontsize=8
        )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Rango de clusters', color='#8b949e')
    ax.set_ylabel('Pendiente de la curva Gap', color='#8b949e')
    ax.set_title('Pendiente de Gap en diferentes intervalos', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def generate_devices(
    n,
    area,
    device_height=1.2,
    seed=None
):
    """Genera dispositivos IoT en 3D con altura fija."""

    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, area, n)
    y = np.random.uniform(0, area, n)

    # Altura fija típica de sensor
    z = np.full(n, device_height)

    return np.column_stack((x, y, z))

def generate_devices_poisson_frame(
    cells,
    seed=None,
    device_height=1.2,
    N=1000,
    uniform_fraction=0.0,
    axis_range=AXIS_RANGE
):
    """
    Genera un número de dispositivos Poisson alrededor de expected_devices.
    - uniform_fraction: fracción repartida uniformemente entre TODAS las celdas.
    - El resto se reparte proporcionalmente al risk_index.
    - No se garantiza un nodo mínimo por celda; la asignación depende del riesgo.
    """
    if seed is not None:
        np.random.seed(seed)

    n_celdas = len(cells)
    if n_celdas == 0:
        return np.empty((0, 3))

    expected_devices = np.random.poisson(N)
    #expected_devices = expected_devices


    # ─── 1. Repartir todos los dispositivos sin mínimo por celda ───
    # La fracción uniforme es opcional; el resto se concentra según el riesgo.
    n_uniform = int(expected_devices * uniform_fraction)
    n_risk = expected_devices - n_uniform
    counts = np.zeros(n_celdas, dtype=int)

    if n_uniform > 0:
        counts += np.random.multinomial(
            n_uniform, np.ones(n_celdas) / n_celdas
        )

    if n_risk > 0:
        risk_values = np.array(
            [max(float(c["risk_index"]), 0.0) for c in cells],
            dtype=float
        )
        total_risk = risk_values.sum()
        if total_risk > 0:
            risk_probs = risk_values / total_risk
        else:
            # Si todas las celdas tienen riesgo cero, no hay criterio de riesgo.
            risk_probs = np.ones(n_celdas) / n_celdas
        counts += np.random.multinomial(n_risk, risk_probs)

    # ─── 2. Generar coordenadas ───
    nodes = []
    for idx, c in enumerate(cells):
        n = counts[idx]
        if n <= 0:
            continue
        # Limitar cada celda al cuadrado local del modelo.
        polygon = c["polygon"].intersection(box(0, 0, axis_range, axis_range))
        if polygon.is_empty:
            continue

        minx, miny, maxx, maxy = polygon.bounds
        for _ in range(n):
            for _ in range(100):
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                if polygon.contains(Point(x, y, device_height)):
                    nodes.append([x, y, device_height])
                    break
            else:
                centroid = polygon.centroid
                if polygon.contains(centroid):
                    nodes.append([centroid.x, centroid.y, device_height])

    return np.asarray(nodes, dtype=float).reshape(-1, 3)

def generate_devices_poisson_aggregated(
    cells,
    n_frames,
    device_height=1.2,
    base_seed=None,
    N=1000,
    uniform_fraction=0.0,
    axis_range=AXIS_RANGE
):
    """
    Genera múltiples frames y los AGREGA.
    
    Simula el dinamismo de una red IoT donde dispositivos entran/salen.
    
    Args:
        cells: Celdas con geometría e índice de riesgo.
        n_frames: Número de snapshots temporales
        base_seed: Seed para reproducibilidad
        expected_devices: Media de dispositivos por frame (λ).
        uniform_fraction: Fracción distribuida uniformemente entre celdas.
    
    Returns:
        aggregated_devices: Array con todos los dispositivos de todos los frames,
            con forma (N, 3).
    """
    if base_seed is not None:
        np.random.seed(base_seed)
    
    all_frames = []
    
    for frame_idx in range(n_frames):
        # Seed único para cada frame
        frame_seed = base_seed + frame_idx if base_seed is not None else None
        
        X_frame = generate_devices_poisson_frame(
            cells=cells,
            device_height=device_height,
            N=N,
            uniform_fraction=uniform_fraction,
            axis_range=axis_range,
            seed=frame_seed
        )

        n_frame = len(X_frame)
        if n_frame > 0:
            all_frames.append(X_frame)
    
    X = generate_devices_poisson_frame(
        cells=cells,
        device_height=device_height,
        N=N,
        uniform_fraction=uniform_fraction,
        axis_range=axis_range,
        seed=frame_seed
    )

    # Agregación: unir todos los frames
    if all_frames:
        aggregated = np.vstack(all_frames)
    else:
        aggregated = np.empty((0, 3))

    return X, aggregated