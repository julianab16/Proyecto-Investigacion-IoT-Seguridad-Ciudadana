"""
dplace_main.py : DPLACE Model - Main Orchestrator
=================================================
Nodo Placement for LoRaWAN Networks using:
  - Phase 1: Pre-Processing  → Gap Statistics + Fuzzy C-Means
  - Phase 2: Processing      → Aggregate runs + K-Means Gap Statistics

Usage:
    python dplace_main.py

"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import json
import pickle
from shapely.affinity import translate
from shapely.geometry import Point, box
from preprocessing_phase import preprocessing
from processing_phase import processing
from plots import plot_devices_and_gateways_on_map, plot_k, plot_gap_curve, plot_global_poisson_variability, plot_nodos_resilience_stats, plot_aggregated_devices, plot_poisson_dynamism
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N_DEVICES = 1000           # IoT devices per scenario
AXIS_RANGE = 10000 
COVERAGE_RADIUS = 2000     # Gateway coverage radius (same units)

# ─── POISSON PATTERN CONFIGURATION ─────────
N_FRAMES = 25              # Temporal frames per simulation (dinamismo)
N_SIMULATIONS = 15         # Number of preprocessing runs (scenarios) diez

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputConRiesgo1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
GW_PRE_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_gateways_pre.csv')
STATS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_stats.json')  # o CSV
RESULTS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_pre_result.pkl')  # para objetos grandes

# NUEVO: CSVs de entrada (gateways fijos + nodos a optimizar)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GATEWAYS_CSV = os.path.join(PROJECT_ROOT, "optimizacion_nodos", "dplacesinriesgo", "output2", "final_gateways.csv")

# NUEVO: GeoJSON de índice de inseguridad (salida de georeferencia.py)
INSECURITY_GEOJSON = os.path.join(PROJECT_ROOT, "images", "grid_cali.geojson")
# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_gateways(csv_file):
    """
    Lee el archivo CSV de gateways.

    Retorna:
        ndarray (N x 3)
    """

    df = pd.read_csv(csv_file)

    gateways = df[["x", "y", "z"]].to_numpy(dtype=float)

    print(f"[Gateways] {len(gateways)} gateways cargados")

    return gateways

import geopandas as gpd

def load_grid(geojson_file):
    """
    Carga el mapa de calor de delitos.

    Retorna una lista de celdas con:
        - id
        - nivel de inseguridad
        - índice de inseguridad
        - polígono
        - centroide
        - bbox
    """

    gdf = gpd.read_file(geojson_file)

    cells = []

    for _, row in gdf.iterrows():

        centroid = row.geometry.centroid

        cells.append({

            "id": int(row["grid_id"]),

            "risk": int(row["nivel_inseguridad"]),

            "risk_index": float(row["indice_inseguridad"]),

            "polygon": row.geometry,

            "centroid": (
                centroid.x,
                centroid.y
            ),

            "bbox": row.geometry.bounds

        })

    print(f"[GRID] {len(cells)} celdas cargadas")

    return cells

def load_grid_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        gdf = pickle.load(f)
    # Asegurar CRS (ya debería tenerlo)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3116", allow_override=True)
    # Verificar bounds
    bounds = gdf.total_bounds
    if not np.isfinite(bounds).all():
        # Calcular bounds manualmente
        b = gdf.geometry.bounds
        minx, miny = b['minx'].min(), b['miny'].min()
        maxx, maxy = b['maxx'].max(), b['maxy'].max()
        gdf.attrs['manual_bounds'] = (minx, miny, maxx, maxy)
    return gdf


def generate_devices_poisson_frame(
    cells,
    seed=None,
    device_height=1.2,
    expected_devices=1000      
):
    """
    Genera UN frame de dispositivos con distribución Poisson.
    
    El número de dispositivos sigue: n ~ Poisson(λ)
    
    Returns:
        devices (array): shape (n, 2) con coordenadas (x, y)
        n_devices (int): número real de dispositivos en este frame
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Número de dispositivos sigue Poisson(λ)
    #n_devices = np.random.poisson(lambda_mean)
    total = sum(c["risk_index"] for c in cells)
    devices = []

    if total == 0:
        return np.empty((0,2)), 0

    for c in cells:
        indice = c["risk_index"]
        lambda_cell = indice / total

        mean_devices = lambda_cell * expected_devices
        n_devices = np.random.poisson(mean_devices)

        polygon = c["polygon"]

        minx, miny, maxx, maxy = polygon.bounds

        creados = 0
        while creados < n_devices:

            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)

            if polygon.contains(Point(x, y, device_height)):
                devices.append([x, y, device_height])
                creados += 1
    
    devices = np.array(devices)
    return devices, len(devices)

def generate_devices_poisson_aggregated(
    cells,
    area,
    n_frames,
    base_seed=None,
    expected_devices=1000      # ← AÑADIR

):
    """
    Genera múltiples frames y los AGREGA.
    
    Simula el dinamismo de una red IoT donde dispositivos entran/salen.
    
    Args:
        area: Área de simulación
        n_frames: Número de snapshots temporales
        base_seed: Seed para reproducibilidad
    
    Returns:
        aggregated_devices: Todos los dispositivos de todos los frames
        frame_data: Estadísticas por frame (para gráficos)
    """
    if base_seed is not None:
        np.random.seed(base_seed)
    
    all_frames = []
    frame_data = []
    
    for frame_idx in range(n_frames):
        # Seed único para cada frame
        frame_seed = base_seed + frame_idx if base_seed is not None else None
        
        X_frame, n_frame = generate_devices_poisson_frame(
            cells=cells,
            seed=frame_seed,
            expected_devices=expected_devices
        )
        
        if n_frame > 0:
            all_frames.append(X_frame)
        
        frame_data.append({
            'frame': frame_idx,
            'n_devices': n_frame
        })
    
    # Agregación: unir todos los frames
    if all_frames:
        aggregated = np.vstack(all_frames)
    
    return aggregated, pd.DataFrame(frame_data)


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

def get_normalized_grid(grid_gdf, x_min, y_min, x_max, y_max):
    """Recorta y normaliza el grid al cuadrado [x_min, x_max] × [y_min, y_max]."""
    # Crear el polígono del cuadrado
    cuadro = box(x_min, y_min, x_max, y_max)
    
    # Filtrar celdas que intersectan el cuadrado
    gdf_cuadro = grid_gdf[grid_gdf.geometry.intersects(cuadro)].copy()
    
    # Trasladar todas las geometrías (Polygon, MultiPolygon, etc.)
    # usando la operación afín de Shapely, que conserva el tipo geométrico.
    dx = x_min
    dy = y_min
    gdf_cuadro['geometry'] = gdf_cuadro.geometry.apply(
        lambda geom: translate(geom, xoff=-dx, yoff=-dy)
    )
    
    # También normalizar centroides (si los usas)
    gdf_cuadro['centroid_x'] = gdf_cuadro.geometry.centroid.x
    gdf_cuadro['centroid_y'] = gdf_cuadro.geometry.centroid.y
    
    # Ajustar los límites totales
    gdf_cuadro.attrs['manual_bounds'] = (0, 0, x_max - x_min, y_max - y_min)
    return gdf_cuadro

def load_grid_from_gdf(gdf):
    cells = []
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        cells.append({
            "id": int(row["grid_id"]),
            "risk": int(row["nivel_inseguridad"]),
            "risk_index": float(row["indice_inseguridad"]),
            "polygon": row.geometry,
            "centroid": (centroid.x, centroid.y),
            "bbox": row.geometry.bounds
        })
    print(f"[GRID] {len(cells)} celdas cargadas desde GDF normalizado")
    return cells

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  DPLACE Model — LoRaWAN Gateway Placement (Poisson Dynamics)")
    print("=" * 70)
    print(f"  Simulations             : {N_SIMULATIONS}")
    print(f"  Temporal frames/sim     : {N_FRAMES}")
    print(f"  Total scenarios         : {N_SIMULATIONS * N_FRAMES} (Poisson aggregated)")
    print(f"  Area                    : {AXIS_RANGE} m x {AXIS_RANGE} m")
    print("=" * 70)
    print()

    all_preprocessing_nodos = []
    all_preprocessing_results = []
    poisson_variability_stats = []  
    all_k_results = []

    gateways = load_gateways(GATEWAYS_CSV)
    grid_gdf = load_grid_pickle(os.path.join(PROJECT_ROOT, "pipeline", "grid_cali.pkl"))

    CUADRO_X_MIN = 724000
    CUADRO_Y_MIN = 867500
    LADO = 10000  # metros
    CUADRO_X_MAX = CUADRO_X_MIN + LADO
    CUADRO_Y_MAX = CUADRO_Y_MIN + LADO

    # ─── Normalizar y recortar ───
    grid_norm = get_normalized_grid(grid_gdf, CUADRO_X_MIN, CUADRO_Y_MIN, CUADRO_X_MAX, CUADRO_Y_MAX)
    print(f"Celda en el cuadrado: {len(grid_norm)}")

    cells = load_grid_from_gdf(grid_norm)


    # ─── PHASE 1: PRE-PROCESSING ───────────────
    print("─" * 50)
    print("PHASE 1 — PRE-PROCESSING")
    print("─" * 50)

    # Gap Statistics parameters
    NREFS_PRE = 80      # References for preprocessing
    BASE_SEED = 42
    USE_POISSON_PATTERN = True

    # Intentar cargar checkpoint
    start_sim, all_preprocessing_nodos, all_preprocessing_results, poisson_variability_stats, all_k_results = load_checkpoint()
    
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
            print(f"\n[Paso 1] Generando {N_FRAMES} frames...")
            X_aggregated, frame_stats_df = generate_devices_poisson_aggregated(
                cells=cells,
                area=AXIS_RANGE,
                n_frames=N_FRAMES,
                base_seed=seed
            )
            n_total = len(X_aggregated)
            n_avg = frame_stats_df['n_devices'].mean()
            print(f"  Total dispositivos agregados: {n_total}")
            print(f"  Promedio por frame: {n_avg:.1f}")
            
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
                plot_aggregated_devices(
                    X_aggregated,
                    frame_stats_df,
                    sim+1,
                    os.path.join(OUTPUT_DIR, f'sim{sim+1}_aggregated_devices.png'),
                    AXIS_RANGE,
                    gateways
                )
                print(f" Gráfico aggregated_devices generado")
            except Exception as e:
                print(f" ERROR en aggregated_devices: {e}")
            
        # ════════════════════════════════════════════════════════════
        # PASO 3: EJECUTAR PREPROCESSING CON DISPOSITIVOS AGREGADOS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 3] Ejecutando preprocessing (FCM) con dispositivos agregados...")
        
        try:
            pre = preprocessing(
                X_aggregated,
                nrefs=NREFS_PRE,
                gateways=gateways,
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
            plot_devices_and_gateways_on_map(
                X_aggregated, 
                pre['nodos'],
                f"FCM Clustering (Poisson) – Sim {sim+1} (k={pre['k_final']})",
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_fcm_result.png'),
                AXIS_RANGE=AXIS_RANGE,
                gateways=gateways,   
                u=pre['u']
            )
        except Exception as e:
            print(f" ERROR en fcm_result: {e}")

        all_preprocessing_nodos.append(pre['nodos'])
        for n in all_preprocessing_nodos:
            gw_array = np.column_stack(
                (n[:, :2],
                np.full(len(n), 1.2))
            )

        all_preprocessing_results.append(gw_array)
        all_k_results.append(int(pre['k_gap']))

        # ════════════════════════════════════════════════════════════
        # PASO 4: GUARDAR RESULTADOS Y ESTADÍSTICAS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 4] Guardando resultados...")
        
        try:
            # Save aggregated devices CSV (con dimensiones: x, y, z)
            dev_df = pd.DataFrame(X_aggregated, columns=['x', 'y', 'z'])
            dev_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_devices_poisson.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_devices_poisson.csv")
        except Exception as e:
            print(f"  ERROR guardando devices CSV: {e}")

        try:
            # Save frame statistics (para análisis de dinamismo)
            frame_stats_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_frame_statistics.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_frame_statistics.csv")
        except Exception as e:
            print(f" ERROR guardando frame statistics: {e}")

        try:
            # Save gateway CSV for this simulation (x,y,z)
            # ─── Guardado de nodos de cada simulación ───
            nodo_array = pd.DataFrame(pre['nodos'][:, :2],  np.full(len(pre['gateways']), 30))
            nodo_df = pd.DataFrame(nodo_array, columns=['x', 'y', 'z'])
            nodo_df.to_csv(os.path.join(OUTPUT_DIR, f'sim{sim+1}_nodos_pre.csv'), index=False)
  
            print(f" Guardado: sim{sim+1}_nodos_pre.csv")
        except Exception as e:
            print(f" ERROR guardando nodos CSV: {e}")
        
                # Guardar resultado completo en pickle (para poder reanudar sin rehacer)
        with open(os.path.join(OUTPUT_DIR, f'sim{sim+1}_pre_result.pkl'), 'wb') as f:
            pickle.dump(pre, f)
        

        # Guardar checkpoint después de cada simulación
        save_checkpoint(sim+1, all_preprocessing_nodos, all_preprocessing_results, 
                        poisson_variability_stats, all_k_results)
        
        print(f" Checkpoint guardado después de sim {sim+1}")

    # ═══════════════════════════════════════════════════════════════════
    # POST-PHASE 1: VISUALIZATION OF AVAILABILITY SET & RESILIENCE
    # ═══════════════════════════════════════════════════════════════════
    if len(all_preprocessing_nodos) < N_SIMULATIONS:
        print("No se completaron todas las simulaciones, no se ejecuta Phase 2.")
    else:
        plot_k(
            all_k_results,
            os.path.join(OUTPUT_DIR, f'k_final_analysis.png')
        )

        print("\n" + "=" * 70)

        # Visualización 2: Análisis de resiliencia

        # Visualización 3: VARIABILIDAD GLOBAL POISSON (NEW)


        if USE_POISSON_PATTERN:
            # Estadísticas del availability set
            all_gw_combined = np.vstack(all_preprocessing_nodos)
            resilience_stats = {
                'n_nodos': len(all_gw_combined),
                'mean_x': np.mean(all_gw_combined[:, 0]),
                'mean_y': np.mean(all_gw_combined[:, 1]),
                'std_x': np.std(all_gw_combined[:, 0]),
                'std_y': np.std(all_gw_combined[:, 1]),
                'cv_x': np.std(all_gw_combined[:, 0]) / np.mean(all_gw_combined[:, 0]),
                'cv_y': np.std(all_gw_combined[:, 1]) / np.mean(all_gw_combined[:, 1])
            }
            
            print(f"\n[Estadísticas de Resiliencia]")
            print(f"  nodo candidates: {resilience_stats['n_nodos']}")
            print(f"  Media X: {resilience_stats['mean_x']:.1f}m (±{resilience_stats['std_x']:.1f}m)")
            print(f"  Media Y: {resilience_stats['mean_y']:.1f}m (±{resilience_stats['std_y']:.1f}m)")
            print(f"  Coef. Variación X: {resilience_stats['cv_x']:.3f}")
            print(f"  Coef. Variación Y: {resilience_stats['cv_y']:.3f}")

        # ─── PHASE 2: PROCESSING ───────────────────
        print("\n" + "─" * 50)
        print("PHASE 2 — PROCESSING")
        print("─" * 50)
        NREFS_PROC = 80            # References for processing

        proc = processing(
            all_preprocessing_nodos,
            nrefs=NREFS_PROC,
            AREA_SIZE=AXIS_RANGE,
            verbose=True
        )

        # Save processing plots
        plot_gap_curve(
            proc['gap_df'], proc['gap_sk_df'], proc['k_optimal'],
            '(K-Means – Processing)',
            os.path.join(OUTPUT_DIR, 'processing_gap_kmeans.png')
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xlim(0, AXIS_RANGE)
        ax.set_ylim(0, AXIS_RANGE)

        # Dibujar dispositivos agregados
        all_devices = np.vstack([pre['X'] for pre in all_preprocessing_results])
        ax.scatter(all_devices[:, 0], all_devices[:, 1], s=2, color='cyan', alpha=0.2)

        # Dibujar nodos finales
        ax.scatter(proc['nodos'][:, 0], proc['nodos'][:, 1], s=300, marker='*',
                color='#ffa657', edgecolors='white', linewidths=1.5, zorder=5,
                label=f'Final Nodos (K={proc["k_optimal"]})')
        # Círculos de cobertura
        for j, gw in enumerate(proc['nodos']):
            circle = plt.Circle((gw[0], gw[1]), COVERAGE_RADIUS,
                                color='#ffa657', fill=False, linestyle='--',
                                linewidth=1, alpha=0.4)
            ax.add_patch(circle)

        ax.legend()
        ax.set_title('Processing Phase – Nodos Finales sobre Cali', color='white')
        plt.savefig(os.path.join(OUTPUT_DIR, 'processing_final_nodos_on_map.png'), dpi=150)
        plt.close()

        # Save final nodos CSV (x,y.z)
        nodo_final = pd.DataFrame(proc['nodos'][:, :2], np.full(len(proc['nodos']), 1.2))
        nodo_final_df = pd.DataFrame(nodo_final, columns=['x', 'y', 'z'])
        nodo_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_nodos.csv'), index=False)

        # Save ALL aggregated nodos (x,y.z)
        nodo_all = pd.DataFrame(proc['all_gw_data'][:, :2], np.full(len(proc['all_gw_data']), 1.2))
        nodo_all_df = pd.DataFrame(nodo_all, columns=['x', 'y', 'z'])
        nodo_all_df.to_csv(os.path.join(OUTPUT_DIR, 'all_preprocessing_nodos.csv'), index=False)

        # ─── SUMMARY ───────────────────────────────
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)

        print(f"  Final nodos (K={proc['k_optimal']}): final_nodos.csv")
        print(f"  All preprocessing : all_preprocessing_nodos.csv")
        print(f"  Summary metrics: summary.csv")

if __name__ == '__main__':
    main()
