import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_devices_and_gateways_on_map(
    X,
    nodos,                 
    title,
    filepath,
    AXIS_RANGE,
    gateways,
    u=None,
    coverage_radius=None
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    if X is not None and len(X) > 0:
        ax.scatter(X[:, 0], X[:, 1], s=3, color='cyan', alpha=0.3, zorder=1, label='IoT Devices')

    # 2. Dibujar los NODOS FCM (clústeres encontrados) - TRIÁNGULOS
    if nodos is not None and len(nodos) > 0:
        colors_nodos = plt.cm.tab20(np.linspace(0, 1, max(len(nodos), 1)))
        
        # Si u está presente, colorear dispositivos por cluster
        if u is not None:
            assignments = np.argmax(u, axis=0)
            for j, gw in enumerate(nodos):
                mask = assignments == j
                ax.scatter(X[mask, 0] if X is not None else [], 
                           X[mask, 1] if X is not None else [], 
                           s=8, color=colors_nodos[j % len(colors_nodos)], alpha=0.5, zorder=2)
        
        # Dibujar los centroides (nodos) como triángulos
        for j, gw in enumerate(nodos):
            ax.scatter(gw[0], gw[1], s=200, marker='^', color=colors_nodos[j % len(colors_nodos)],
                       edgecolors='white', linewidths=1.0, zorder=5, label=f'Nodo {j+1}' if j == 0 else "")
            ax.annotate(f'FCM-{j+1}', (gw[0], gw[1]),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7, color='white', fontweight='bold')
            if coverage_radius is not None:
                circle = plt.Circle((gw[0], gw[1]), coverage_radius,
                                    color=colors_nodos[j % len(colors_nodos)], fill=False,
                                    linestyle='--', linewidth=0.8, alpha=0.4)
                ax.add_patch(circle)

    # 3. Dibujar GATEWAYS FIJOS (cuadrados rojos) - OPCIONAL, para referencia
    if gateways is not None and len(gateways) > 0:
        ax.scatter(gateways[:, 0], gateways[:, 1],
                   s=120, marker='s', color='red', edgecolors='white',
                   linewidths=1.0, zorder=6, label='Fixed Gateways (Referencia)')
        for j, gw in enumerate(gateways):
            ax.annotate(f'FG-{j+1}', (gw[0], gw[1]),
                        textcoords='offset points', xytext=(6, -10),
                        fontsize=6, color='red', fontweight='bold')

    # 3. Límites fijos (0 a AXIS_RANGE)
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)

    ax.set_title(title, color='white', fontsize=12)
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
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
    lambda_val = frame_stats_df['n_devices'].mean()   # Usar media observada
    
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

def plot_aggregated_devices(
    X_aggregated,
    frame_stats_df,
    sim_num,
    filepath,
    AXIS_RANGE,
    gateways,             
):
    """
    Visualiza los dispositivos agregados de todos los frames sobre el mapa de Cali.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    # Fondo blanco
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    
    # Después de dibujar los nodos candidatos, dibujar gateways fijos
    if gateways is not None and len(gateways) > 0:
        # Los marcamos con un cuadrado o rombo
        ax.scatter(gateways[:, 0], gateways[:, 1],
                   s=120, marker='s', color='red', edgecolors='white',
                   linewidths=1.0, zorder=6, label='Gateways')
        
    # Scatter 2D (ignorar z por ahora)
    ax.scatter(X_aggregated[:, 0], X_aggregated[:, 1], 
                s=4, color='#58a6ff', alpha=0.5)
    
    n_total = len(X_aggregated)
    n_frames = len(frame_stats_df)
    # El generador actual guarda ``n_devices``, pero no una columna ``lambda``.
    # Usar lambda cuando venga incluida y, en caso contrario, la media observada.
    if 'lambda' in frame_stats_df.columns and not frame_stats_df['lambda'].empty:
        lambda_value = frame_stats_df['lambda'].iloc[0]
    elif 'n_devices' in frame_stats_df.columns and not frame_stats_df['n_devices'].empty:
        lambda_value = frame_stats_df['n_devices'].mean()
    else:
        lambda_value = 0
    
    ax.set_xlim(0, AXIS_RANGE)
    ax.set_ylim(0, AXIS_RANGE)
    ax.set_title(f'Dispositivos Agregados - Sim {sim_num}\n' + 
                    f'Total: {n_total} | Frames: {n_frames} | λ: {lambda_value:.1f}',
                    color='white', fontsize=11)
    ax.set_xlabel('X (m)', color='#8b949e')
    ax.set_ylabel('Y (m)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_nodos_resilience_stats(all_gateways_list, filepath):

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