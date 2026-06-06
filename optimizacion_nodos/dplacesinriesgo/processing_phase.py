"""
processing_phase.py : Fase de Procesamiento del modelo DPLACE.

Referencia: DPLACE paper — "The processing phase is divided into two steps:
the computation of the number of gateways and the gateway placement for a
dynamic IoT scenario."

ESTADO: Verificado contra el paper. Sin bugs de lógica.
        Se agregan comentarios precisos con referencias a las ecuaciones.

Flujo según el paper:
─────────────────────────────────────────────────────────────────────────────
  1. ENTRADA: conjunto de posiciones GW_j = (x_j, y_j, z_j) de múltiples
     ejecuciones de preprocesamiento. Cada GW se trata como un NODO (3D).

  2. GAP STATISTICS (K-Means):
     - Itera k ∈ [1, 2, ..., K]
     - Calcula J_k (WCSS, Ec.10) sobre datos reales y B conjuntos aleatorios
     - Calcula Gap(k) (Ec.11), std(k) (Ec.3), s(k) (Ec.4)
     - Selecciona K óptimo por criterio de Ec.(5)

  3. K-MEANS FINAL (3D):
     - Agrupa las posiciones GW acumuladas con K clusters usando distancia euclidiana 3D
     - Los centroides C_k son las posiciones finales de los gateways (3D)
     - Distancia euclidiana ||GW_j − C_k|| incluye componente z

  4. SALIDA: K óptimo + posiciones C_k (3D) → pasan a la fase de validación.

Paper: "when we used K-means to generalize the clusters created during
pre-processing, we avoided the need to run for each event in the network."
"""

import numpy as np
from sklearn.cluster import KMeans
from cluster.optimalC_method.kmeans_gap_statistics import gap_statistics_kmeans


def processing(
    all_gateways,
    nrefs,
    verbose=True
):
    """
    Fase de Procesamiento: agrega posiciones de gateways de múltiples
    ejecuciones de preprocesamiento y determina la topología global óptima.

    El paper explica: ejecutar el algoritmo para cada cambio en la red es
    inviable. DPLACE ejecuta preprocesamiento con distintos despliegues IoT,
    acumula los GW_j resultantes, y generaliza con K-Means para obtener una
    solución robusta sin recalcular continuamente.

    Parámetros
    ----------
    all_gateways : list of ndarray
        Cada elemento es un ndarray (k_i, 3) con las posiciones de gateways
        GW_j = (x_j, y_j, z_j) producidas por una ejecución de preprocesamiento (3D).
        Paper: "stores the results of several executions of the pre-processing
        phase and turns the gateway positions into nodes."
    nrefs : int
        Número B de conjuntos de referencia para Gap Statistics.
    verbose : bool
        Imprimir información de progreso.

    Retorna
    -------
    dict con claves:
        'k_optimal'   : K óptimo de gateways (número de centroides C_k)
        'gateways'    : ndarray (K, 3) — posiciones finales C_k de gateways (3D: x, y, z)
        'labels'      : ndarray (N_total,) — asignación de cada GW a su cluster
        'gap_df'      : DataFrame ['clusterCount', 'gap'] — curva Gap(k)
        'gap_sk_df'   : DataFrame ['clusterCount', 'Gap_sk'] — barras Ec.(5)
        'all_gw_data' : ndarray (N_total, 3) — todos los GW acumulados (nodos 3D)
    """
    # ── ENTRADA: acumular todas las posiciones GW_j como nodos ────────────
    # Paper: "acquires a set of gateway placements by storing the results of
    # several executions of the pre-processing phase and turns the gateway
    # positions into nodes."
    # GW_j = (x_j, y_j, z_j) para j ∈ [1, M] de cada ejecución (3D)
    
    all_gw_data = np.vstack(all_gateways)   # shape (N_total, 3)

    maxClusters = 30
    #maxClusters = min(maxClusters, len(all_gw_data))
    
    if verbose:
        print(f"  [Processing] Nodos GW acumulados: {len(all_gw_data)}")
        print(f"  [Processing] Gap Statistics K-Means "
              f"(nrefs={nrefs}, maxClusters={maxClusters})...")

    # ── PASO 1: Gap Statistics → K óptimo ────────────────────────────────
    # Paper: "finds out the optimal number of gateways among the data set
    # by making use of the Gap statistics method adapted to K-Means."
    # Implementa Ec.(10) J_k, Ec.(11) Gap(k), Ec.(3) std(k),
    # Ec.(4) s(k), Ec.(5) criterio de selección.
    k_optimal, gap_df, gap_sk_df = gap_statistics_kmeans(
        all_gw_data, nrefs, maxClusters
    )

    if verbose:
        print(f"  [Processing] K óptimo = {k_optimal}")
        print(f"  [Processing] Ejecutando K-Means final con K={k_optimal}...")

    # ── PASO 2: K-Means final → posiciones óptimas C_k ───────────────────
    # Paper: "K-Means allows for the classification of each node in the data
    # set... to better centralize the gateway position within each quadrant."
    # "computes the Euclidean distance between a gateway GW_j (defined as a
    # node) and the central gateway C_k computed by the K-Means algorithm."
    # Los centroides C_k son las posiciones finales de los gateways (3D).

    # Ec.(10): K-Means minimiza J_k = Σ_k Σ_j ||GW_j^(k) − C_k||^2 (con z)
    km = KMeans(n_clusters=k_optimal, n_init=250, max_iter=500, random_state=42)
    km.fit(all_gw_data)
    final_gateways = km.cluster_centers_   # C_k: posiciones óptimas (K, 3), shape (K, 3)
    labels         = km.labels_

    if verbose:
        print(f"  [Processing] Posiciones finales C_k calculadas. "
              f"Shape: {final_gateways.shape}")

    # ── SALIDA → Fase de Validación ───────────────────────────────────────
    # Paper: "the result of the processing with the optimal gateway positions
    # and the IoT scenario generated in the pre-processing step is passed to
    # the validation phase."
    return {
        'k_optimal'   : k_optimal,
        'gateways'    : final_gateways,   # C_k: posiciones óptimas de gateways (K, 3)
        'labels'      : labels,           # asignación de cada nodo GW a su cluster
        'gap_df'      : gap_df,           # curva Gap(k) para graficar
        'gap_sk_df'   : gap_sk_df,        # valores Ec.(5) para gráfica de barras
        'all_gw_data' : all_gw_data       # todos los nodos GW acumulados (N_total, 3)
    }


