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
import geopandas as gpd
import json
import pickle
from shapely.affinity import translate
from shapely.geometry import box
from preprocessing_phase import preprocessing
from processing_phase import processing
from plots import plot_devices_and_gateways, plot_k, plot_gap_curve, plot_poisson_final, generate_devices, generate_devices_poisson_aggregated, plot_gap_inteval
import osmnx as ox
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

N = 1000           # IoT devices per scenario
AXIS_RANGE = 5900         # 100 km^2
COVERAGE_RADIUS = 2000     # Gateway coverage radius (same units)

# ─── POISSON PATTERN CONFIGURATION ─────────
LAMBDA_MEAN = 1000         # Expected devices per frame (tasa de llegada)
N_FRAMES = 2              # Temporal frames per simulation (dinamismo)
N_SIMULATIONS = 15         # Number of preprocessing runs (scenarios) diez
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
INSECURITY_GEOJSON = os.path.join(PROJECT_ROOT, "images", "grid_cali.geojson")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputConRiesgo1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
GW_PRE_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_gateways_pre.csv')
STATS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_stats.json')  # o CSV
RESULTS_PREFIX = os.path.join(OUTPUT_DIR, 'sim{}_pre_result.pkl')  # para objetos grandes

# None: continuar después de la última simulación guardada.
# Ejemplo: 5 vuelve a ejecutar desde la simulación 5.
RESUME_FROM_SIM = 11

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
    """Carga el estado guardado y devuelve los datos hasta la simulación anterior."""
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

        if RESUME_FROM_SIM is not None:
            if not 1 <= RESUME_FROM_SIM <= last:
                raise ValueError(
                    f"RESUME_FROM_SIM debe estar entre 1 y {last}; "
                    f"valor recibido: {RESUME_FROM_SIM}"
                )

            # La simulación se indexa desde cero en el bucle principal.
            # Para volver a ejecutar la simulación N, se conservan N-1.
            simulations_to_keep = RESUME_FROM_SIM - 1
            return (
                simulations_to_keep,
                all_gateways[:simulations_to_keep],
                all_results[:simulations_to_keep],
                all_k[:simulations_to_keep]
            )
        
        return last, all_gateways, all_results, all_k
    except Exception:
        # Si algo falla, empezar desde cero (opcional: borrar checkpoint corrupto)
        return 0, [], [], []
    
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
    return cells

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
    print(f"  Area                    : {AXIS_RANGE} m x {AXIS_RANGE} m")
    print(f"  Coverage Radius         : {COVERAGE_RADIUS} m")
    print("=" * 70)
    print()

    all_preprocessing_gateways = []
    all_preprocessing = []
    all_preprocessing_results = []
    all_k_results = []

    grid_gdf = gpd.read_file(INSECURITY_GEOJSON)

    CUADRO_X_MIN =  724900 
    CUADRO_Y_MIN =  868100
    LADO = AXIS_RANGE  # metros
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
    NREFS_PROC = 10            # References for processing
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
        # PASO 1: GENERAR DISPOSITIVOS (según modo seleccionado)
        # ════════════════════════════════════════════════════════════
        if USE_POISSON_PATTERN:
            print(f"\n[Paso 1] Generando {N_FRAMES} frames con distribución Poisson(λ={LAMBDA_MEAN})...")
            N, X  = generate_devices_poisson_aggregated(
                cells=cells,
                n_frames=N_FRAMES,
                base_seed=seed,
                axis_range=AXIS_RANGE
            )
            n_total = len(N)
            print(f" Total dispositivos agregados: {n_total}")

            # ════════════════════════════════════════════════════════════
            # PASO 2: VISUALIZAR DINAMISMO (Variabilidad Poisson)
            # ════════════════════════════════════════════════════════════
            
            try:
                plot_poisson_final(
                    N,
                    sim+1,
                    os.path.join(OUTPUT_DIR, f'sim{sim+1}_aggregated_devices.png'),
                    AXIS_RANGE
                )
                print(f" Gráfico aggregated_devices generado")
            except Exception as e:
                print(f" ERROR en aggregated_devices: {e}")
            
        else:
            print(f"\n[Paso 1] Generando {N} dispositivos fijos uniformes...")
            N = generate_devices(N=N, area=AXIS_RANGE, device_height=1.2, seed=seed)
            n_total = len(N)
            print(f" Total dispositivos: {n_total}")

        # ════════════════════════════════════════════════════════════
        # PASO 3: EJECUTAR PREPROCESSING CON DISPOSITIVOS AGREGADOS
        # ════════════════════════════════════════════════════════════
        print(f"\n[Paso 3] Ejecutando preprocessing (FCM) con dispositivos agregados...")
        
        try:
            pre = preprocessing(
                N,
                nrefs=NREFS_PRE,
                verbose=True,
                AREA_SIZE=AXIS_RANGE,
                cells=cells
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
                N, pre['gateways'],
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
            dev_df = pd.DataFrame(N, columns=['x', 'y', 'z'])
            dev_df.to_csv(
                os.path.join(OUTPUT_DIR, f'sim{sim+1}_devices_poisson.csv'), index=False
            )
            print(f" Guardado: sim{sim+1}_devices_poisson.csv")
        except Exception as e:
            print(f" ERROR guardando devices CSV: {e}")

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
        save_checkpoint(sim+1, all_preprocessing_gateways, all_preprocessing_results, all_k_results)
        
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

        gap_data = proc['gap_df']
        plot_gap_inteval(
            gap_data['clusterCount'].values,  gap_data['gap'].values,
            os.path.join(OUTPUT_DIR, 'processing_pendiente.png')
        )


        # Plot all collected gateways + final centroids + devices
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')

        N, X = generate_devices_poisson_aggregated(
            cells=cells,
            n_frames=N_FRAMES,
            base_seed=BASE_SEED,
            axis_range=AXIS_RANGE
        )

        # Graficar todos los dispositivos de todas las simulaciones
        all_devices = np.vstack(N)
        ax.scatter(
            all_devices[:, 0], all_devices[:, 1],
            s=6, zorder=5, color='#58a6ff', alpha=0.7, label='IoT Devices'
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
        plt.savefig(os.path.join(OUTPUT_DIR, 'processing_final_gateways.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # Save final nodos CSV in the real Cali map coordinates (EPSG:3116).
        # N is generated in the local normalized square used by the
        # model, so restore the original map origin before exporting it.
        nodo_final = N.copy()
        nodo_final[:, 0] += CUADRO_X_MIN
        nodo_final[:, 1] += CUADRO_Y_MIN
        nodo_final_df = pd.DataFrame(nodo_final, columns=['x', 'y', 'z'])
        nodo_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_nodos.csv'), index=False)


        # Save final gateways CSV in the real Cali map coordinates (EPSG:3116).
        gw_final = proc['gateways'].copy()
        gw_final[:, 0] += CUADRO_X_MIN
        gw_final[:, 1] += CUADRO_Y_MIN
        gw_final_array = np.column_stack((gw_final[:, :2], np.full(len(gw_final), 30)))
        gw_final_df = pd.DataFrame(gw_final_array, columns=['x', 'y', 'z'])
        gw_final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_gateways.csv'), index=False)

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
            colors = ["white", "#d9d9d9", "#b3b3b3", "#808080", "#4d4d4d", "#1a1a1a"]
            cmap = mcolors.LinearSegmentedColormap.from_list("seguridad_grises", colors, N=6)
            grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                        alpha=0.75, edgecolor="black", linewidth=0.3,
                        vmin=0, vmax=5, zorder=3)

        # ─── Convertir coordenadas locales a EPSG:3116 ───
        # Calcular un margen común antes de convertir para que nodos y
        # gateways queden dentro del cuadro de destino, sin alterar después
        # los límites de la gráfica.
        def convertir_a_epsg3116(coords_locales):
            coords_locales = np.asarray(coords_locales[:, :2], dtype=float)
            escala_x = x_max_local - x_min_local
            escala_y = y_max_local - y_min_local
            x = CUADRO_X_MIN + ((coords_locales[:, 0] - x_min_local) / escala_x) * AXIS_RANGE
            y = CUADRO_Y_MIN + ((coords_locales[:, 1] - y_min_local) / escala_y) * AXIS_RANGE
            return np.column_stack((x, y))

        gw_centroids = proc['gateways']   # shape (K, 3)
        datos_locales = np.vstack((all_devices[:, :2], gw_centroids[:, :2]))
        x_min_local, y_min_local = datos_locales.min(axis=0)
        x_max_local, y_max_local = datos_locales.max(axis=0)
        margen_x = max(x_max_local - x_min_local, 1.0) * 0.05
        margen_y = max(y_max_local - y_min_local, 1.0) * 0.05
        x_min_local -= margen_x
        x_max_local += margen_x
        y_min_local -= margen_y
        y_max_local += margen_y

        nodes_utm = convertir_a_epsg3116(all_devices)
        gw_utm = convertir_a_epsg3116(gw_centroids)

        # Rectángulo exacto usado por el modelo en coordenadas EPSG:3116.
        x_ini = CUADRO_X_MIN
        y_ini = CUADRO_Y_MIN
        ancho = AXIS_RANGE
        alto = AXIS_RANGE

        rect = Rectangle(
            (x_ini, y_ini), ancho, alto,
            linewidth=2.5, edgecolor='deepskyblue', facecolor='aqua',
            alpha=0.25, zorder=4
        )
        ax.add_patch(rect)
  
        # 5. Nodos DPLACE (preprocesamiento)
        ax.scatter(nodes_utm[:, 0], nodes_utm[:, 1], s=3,
                marker='o', color='#00b4d8', alpha=0.75,
                edgecolors='white', linewidths=0.25,
                zorder=6, label=f'Nodos DPLACE ({len(nodes_utm)})')

        # 6. Gateways finales (centroides de K-Means)
        if proc is not None and 'gateways' in proc:
            ax.scatter(gw_utm[:, 0], gw_utm[:, 1], s=20,
                    marker='^', color='black', edgecolors='black',
                    linewidths=1.0, zorder=7,
                    label=f'Gateways Finales (K={proc["k_optimal"]})')

 
        # 8. Final configuration
        ax.set_title('DPLACE – Optimized Gateways on Cali’s Insecurity Map',
                    fontsize=10, color='black')
        ax.set_xlabel('X Coordinate (m)', fontsize=8)
        ax.set_ylabel('Y Coordinate (m)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(facecolor='white', edgecolor='gray', fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'dplace_gateways_on_heatmap_riesgo.png'), dpi=300, bbox_inches='tight')
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