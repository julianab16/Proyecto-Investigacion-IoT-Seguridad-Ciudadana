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

from preprocessing_phase import preprocessing
from processing_phase import processing
from plots import plot_devices_and_gateways, plot_k, plot_gap_curve, generate_devices, generate_devices_poisson_aggregated, plot_poisson_dynamism, plot_aggregated_devices_3d, plot_gateway_resilience_stats, plot_global_poisson_variability

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

N_DEVICES = 1000           # IoT devices per scenario
AXIS_RANGE = 10000         # 100 km^2
COVERAGE_RADIUS = 2000     # Gateway coverage radius (same units)

# ─── POISSON PATTERN CONFIGURATION ─────────
LAMBDA_MEAN = 1000         # Expected devices per frame (tasa de llegada)
N_FRAMES = 25              # Temporal frames per simulation (dinamismo)
N_SIMULATIONS = 15         # Number of preprocessing runs (scenarios) diez
NREFS_PROC = 80            # References for processing

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outpu3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
GW_PRE_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_gateways_pre.csv')
STATS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_stats.json')  # o CSV
RESULTS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_pre_result.pkl')  # para objetos grandes

def save_checkpoint(sim_index, all_gateways, all_results, poisson_stats, all_k):
    """Guarda el estado actual después de completar la simulación sim_index (1-based)."""
    state = {
        'last_completed_sim': sim_index,
        'total_simulations': N_SIMULATIONS, #sim
        'all_k_results': all_k,
        'poisson_stats': poisson_stats,
        # Guardamos las rutas de los archivos de gateways y resultados (no los objetos en JSON)
        'gateways_files': [f'sim{i+1}_gateways_pre.csv' for i in range(sim_index)],
        'results_files': [f'sim{i+1}_pre_result.pkl' for i in range(sim_index)]
    }
    import numpy as np

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
        return 0, [], [], [], []  # empezar desde cero
    
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
        poisson_stats = state['poisson_stats']
        
        # Verificar consistencia (número de elementos coincide con last)
        assert len(all_gateways) == last
        assert len(all_results) == last
        assert len(all_k) == last
        assert len(poisson_stats) == last
        
        return last, all_gateways, all_results, poisson_stats, all_k
    except Exception:
        # Si algo falla, empezar desde cero (opcional: borrar checkpoint corrupto)
        return 0, [], [], [], []
    
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
    all_preprocessing = []
    all_preprocessing_results = []
    poisson_variability_stats = []  
    all_k_results = []

    # ─── PHASE 1: PRE-PROCESSING ───────────────
    print("─" * 50)
    print("PHASE 1 — PRE-PROCESSING")
    print("─" * 50)

    # Gap Statistics parameters
    NREFS_PRE = 80      # References for preprocessing
    BASE_SEED = 42
    USE_POISSON_PATTERN = True

    # Intentar cargar checkpoint
    start_sim, all_preprocessing_gateways, all_preprocessing_results, poisson_variability_stats, all_k_results = load_checkpoint()
    
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
        # PASO 1: GENERAR DISPOSITIVOS (según modo seleccionado)
        # ════════════════════════════════════════════════════════════
        if USE_POISSON_PATTERN:
            print(f"\n[Paso 1] Generando {N_FRAMES} frames con distribución Poisson(λ={LAMBDA_MEAN})...")
            X_aggregated, frame_stats_df = generate_devices_poisson_aggregated(
                lambda_mean=LAMBDA_MEAN,
                area=AXIS_RANGE,
                n_frames=N_FRAMES,
                base_seed=seed
            )
            n_total = len(X_aggregated)
            n_avg = frame_stats_df['n_devices'].mean()
            print(f" Total dispositivos agregados: {n_total}")
            print(f" Promedio por frame: {n_avg:.1f}")
            print(f" Mín/Máx por frame: {frame_stats_df['n_devices'].min()} / {frame_stats_df['n_devices'].max()}")
            
            # Capturar estadísticas (para gráficos posteriores)
            n_devices_frame = frame_stats_df['n_devices'].values
            std_frame = np.std(n_devices_frame)
            cv_frame = std_frame / n_avg if n_avg > 0 else 0
            range_frame = n_devices_frame.max() - n_devices_frame.min()

            poisson_variability_stats.append({
                'sim': int(sim + 1),
                'mean': float(n_avg),
                'std': float(std_frame),
                'cv': float(cv_frame),
                'range': int(range_frame),
                'min': int(n_devices_frame.min()),
                'max': int(n_devices_frame.max()),
                'total': int(n_total)
            })

            # ════════════════════════════════════════════════════════════
            # PASO 2: VISUALIZAR DINAMISMO (Variabilidad Poisson)
            # ════════════════════════════════════════════════════════════
            print(f"\n[Paso 2] Generando gráficos de dinamismo...")
            
            try:
                plot_poisson_dynamism(
                    frame_stats_df,
                    sim+1,
                    os.path.join(OUTPUT_DIR, f'sim{sim+1}_poisson_dynamism.png')
                )
                print(f" Gráfico poisson_dynamism generado")
            except Exception as e:
                print(f" ERROR en poisson_dynamism: {e}")
            
            try:
                plot_aggregated_devices_3d(
                    X_aggregated,
                    frame_stats_df,
                    sim+1,
                    os.path.join(OUTPUT_DIR, f'sim{sim+1}_aggregated_devices.png'),
                    AXIS_RANGE
                )
                print(f" Gráfico aggregated_devices generado")
            except Exception as e:
                print(f" ERROR en aggregated_devices: {e}")
            
        else:
            print(f"\n[Paso 1] Generando {N_DEVICES} dispositivos fijos uniformes...")
            X_aggregated = generate_devices(n=N_DEVICES, area=AXIS_RANGE, device_height=1.2, seed=seed)
            n_total = len(X_aggregated)
            print(f" Total dispositivos: {n_total}")
            # Crear un frame_stats_df vacío o dummy para no romper gráficos posteriores
            frame_stats_df = pd.DataFrame({'frame': [0], 'n_devices': [n_total], 'lambda': [0]})
            # No agregamos a poisson_variability_stats porque no aplica.
        
        # ════════════════════════════════════════════════════════════
        # PASO 3: EJECUTAR PREPROCESSING CON DISPOSITIVOS AGREGADOS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 3] Ejecutando preprocessing (FCM) con dispositivos agregados...")
        
        try:
            pre = preprocessing(
                X_aggregated,
                nrefs=NREFS_PRE,
                verbose=True,
                AREA_SIZE=AXIS_RANGE
            )
            print(f" Preprocessing completado")

        except Exception as e:
            print(f" ERROR en preprocessing: {e}")
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
            print(f" ERROR en gap_fcm: {e}")
        
        try:
            plot_devices_and_gateways(
                X_aggregated, pre['gateways'],
                f"FCM Clustering (Poisson) – Sim {sim+1} (k={pre['k_final']})",
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_fcm_result.png'),
                AXIS_RANGE,
                coverage_radius=COVERAGE_RADIUS,
                u=pre['u']
            )
        except Exception as e:
            print(f" ERROR en fcm_result: {e}")

        all_preprocessing_gateways.append(pre['gateways'])
        all_preprocessing_results.append(pre)
        all_k_results.append(int(pre['k_gap']))

        # ════════════════════════════════════════════════════════════
        # PASO 4: GUARDAR RESULTADOS Y ESTADÍSTICAS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 4] Guardando resultados...")
        
        try:
            # Save aggregated devices CSV (con 3 dimensiones: x, y, z)
            dev_df = pd.DataFrame(X_aggregated, columns=['x', 'y', 'z'])
            dev_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_devices_poisson.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_devices_poisson.csv")
        except Exception as e:
            print(f" ERROR guardando devices CSV: {e}")

        try:
            # Save frame statistics (para análisis de dinamismo)
            frame_stats_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_frame_statistics.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_frame_statistics.csv")
        except Exception as e:
            print(f" ERROR guardando frame statistics: {e}")

        try:
            # Save gateway CSV for this simulation (x,y de centroide, z=30 para gateways)
            gw_array = np.column_stack((pre['gateways'][:, :2], np.full(len(pre['gateways']), 30)))
            gw_df = pd.DataFrame(gw_array, columns=['x', 'y', 'z'])
            gw_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_gateways_pre.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_gateways_pre.csv")
        except Exception as e:
            print(f" ERROR guardando gateways CSV: {e}")
        
                # Guardar resultado completo en pickle (para poder reanudar sin rehacer)
        with open(os.path.join(OUTPUT_DIR, f'sim{sim+1}_pre_result.pkl'), 'wb') as f:
            pickle.dump(pre, f)
        

        # Guardar checkpoint después de cada simulación
        save_checkpoint(sim+1, all_preprocessing_gateways, all_preprocessing_results, 
                        poisson_variability_stats, all_k_results)
        
        print(f" Checkpoint guardado después de sim {sim+1}")

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
        plot_gateway_resilience_stats(
            all_preprocessing_gateways,
            os.path.join(OUTPUT_DIR, 'gateway_resilience_analysis.png')
        )
        print(" Generado: gateway_resilience_analysis.png")
        
        # Visualización 3: VARIABILIDAD GLOBAL POISSON (NEW)
        plot_global_poisson_variability(
            poisson_variability_stats,
            os.path.join(OUTPUT_DIR, 'poisson_global_variability.png')
        )

        for n in all_preprocessing_gateways:
            gw_array = np.column_stack(
                (n[:, :2],
                np.full(len(n), 30))
            )

            all_preprocessing.append(gw_array)

        # ─── PHASE 2: PROCESSING ───────────────────
        print("\n" + "─" * 50)
        print("PHASE 2 — PROCESSING")
        print("─" * 50)

        proc = processing(
            all_preprocessing,
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
        all_devices = np.vstack([pre['X'] for pre in all_preprocessing_results])
        ax.scatter(
            all_devices[:, 0], all_devices[:, 1],
            s=3, color='#6e7681', alpha=0.2, label='IoT Devices'
        )
        
        # Gateways finales después de K-Means
        ax.scatter(
            proc['gateways'][:, 0], proc['gateways'][:, 1],
            s=300, marker='*', color='#ffa657', edgecolors='white',
            linewidths=1.5, zorder=5, label=f'Final GWs (K={proc["k_optimal"]})'
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
        plt.savefig(os.path.join(OUTPUT_DIR, 'processing_final_gateways.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # Save final gateways CSV (x,y de centroide, z=30 para gateways)
        gw_final_array = np.column_stack((proc['gateways'][:, :2], np.full(len(proc['gateways']), 30)))
        gw_final_df = pd.DataFrame(gw_final_array, columns=['x', 'y', 'z'])
        gw_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_gateways.csv'), index=False)

        # Save ALL aggregated gateways (x,y de nodos preprocesamiento, z=30 para gateways)
        gw_all_array = np.column_stack((proc['all_gw_data'][:, :2], np.full(len(proc['all_gw_data']), 1.2)))
        gw_all_df = pd.DataFrame(gw_all_array, columns=['x', 'y', 'z'])
        gw_all_df.to_csv(os.path.join(OUTPUT_DIR, 'all_preprocessing_gateways.csv'), index=False)

        # ─── SUMMARY ───────────────────────────────
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)

        print(f"  Final gateways (K={proc['k_optimal']}): final_gateways.csv")
        print(f"  All preprocessing gateways: all_preprocessing_gateways.csv")
        print(f"  Summary metrics: summary.csv")

if __name__ == '__main__':
    main()