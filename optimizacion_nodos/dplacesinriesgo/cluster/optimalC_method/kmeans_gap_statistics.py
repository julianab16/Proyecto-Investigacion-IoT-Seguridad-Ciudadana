"""
kmeans_gap_statistics.py : Gap Statistics para K-Means (Fase de Procesamiento).
Encuentra el número óptimo de clusters K de gateways a partir del conjunto
acumulado de posiciones generadas en la fase de preprocesamiento.

Referencia: DPLACE paper — Processing Phase, Ecuaciones (10), (11), (3), (4), (5).

CRITERIO DE SELECCIÓN (Ecuación 5):
    Seleccionar el menor k tal que Gap(k) >= Gap(k+1) - s(k+1)
    Si ningún k cumple, elegir el que maximiza Gap(k).

VECTORIZACIÓN:
    - Preasignación de arrays con NumPy.
    - Generación de todas las referencias aleatorias de una sola vez.
    - Cálculo de inertias por lote usando listas por comprensión.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def gap_statistics_kmeans(data, nrefs, maxClusters):
    """
    Calcula el número óptimo de clusters K para K-Means usando Gap Statistic.
    Implementa el criterio original de la Ecuación (5).

    Parámetros
    ----------
    data : ndarray, shape (N_gateways, 3)
        Posiciones de gateways GW_j = (x_j, y_j, z_j) acumuladas (3D).
    nrefs : int
        Número B de conjuntos de referencia.
    maxClusters : int
        Número máximo de clusters a evaluar (k = 1 .. maxClusters).

    Retorna
    -------
    k_optimal : int
        Número óptimo de clusters.
    resultsdf : pd.DataFrame
        Columnas ['clusterCount', 'gap'].
    gp : pd.DataFrame
        Columnas ['clusterCount', 'Gap_sk'].
    """

    n_samples = data.shape[0]
    actual_max_k = min(maxClusters, n_samples)  # K-Means: n_clusters <= n_samples
    n_clusters_range = np.arange(1, actual_max_k+1)

    print(f"  [Processing] n_samples={n_samples}, maxClusters={maxClusters}, actual_max_k={actual_max_k}")
    
    # Preasignar arrays
    gaps = np.zeros(actual_max_k)
    s_errors = np.zeros(actual_max_k)
    std_j = np.zeros(actual_max_k)

    # Factor constante para s(k)
    gaps_sks = np.zeros(maxClusters)

    AREA_SIZE = 10000 

    # Para cada número de clusters k (asegurar k <= n_samples para K-Means)
    for idx, k in enumerate(range(1, actual_max_k + 1)):
        # --- 1. Generar B referencias aleatorias y calcular sus inertias ---
        inertias_ref = np.zeros(nrefs)
        for i in range(nrefs):
            # Referencia uniforme en el rango de los datos reales
            
            random_ref = np.random.uniform(low=0, high=AREA_SIZE, size=data.shape)
            if k == 1:
                inertias_ref[i] = np.sum((random_ref - random_ref.mean(axis=0))**2)
            else:
                km = KMeans(n_clusters=k, n_init=5, max_iter=300, random_state=i)
                km.fit(random_ref)
                inertias_ref[i] = km.inertia_
        
        log_refs = np.log(inertias_ref + 1e-10)
        # --- 2. Inertia sobre los datos reales ---
        if k == 1:
            inertia_real = np.sum((data - data.mean(axis=0))**2)
        else:
            km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            km.fit(data)
            inertia_real = km.inertia_
        
        log_real = np.log(inertia_real + 1e-10)
        # --- 3. Gap(k) = mean(log_refs) - log_real ---
        mean_log_ref = np.mean(log_refs)
        gaps[idx] = mean_log_ref - log_real

        # --- 4. Desviación estándar y s(k) ---
        std_j[idx] = np.std(log_refs, ddof=0)
        s_errors[idx] = std_j[idx] * np.sqrt(1 + 1 / nrefs)
    
    # Selección del k óptimo según Ecuación (5)
    optimal_k = None
    for idx in range(1, len(gaps) - 1):
        # Verificar criterio: Gap(j) >= Gap(j+1) - s(j+1)
        if gaps[idx] >= gaps[idx + 1] - s_errors[idx + 1]:  # ← CRITERIO REAL
            optimal_k = idx + 1
            print(f"  Optimal k={optimal_k} (primer k que cumple criterio)")
            break

    for idx in range(1, len(gaps) - 1):
        gaps_sks[idx] = gaps[idx] - gaps[idx + 1] + s_errors[idx + 1]

    # Fallback si no encuentra ninguno
    if optimal_k is None:
        optimal_k = np.argmax(gaps) + 1
        print(f"  No valid clusters encontrados. Usando k={optimal_k} (max Gap)")

    # Construcción de DataFrames para gráficos
    resultsdf = pd.DataFrame({
        'clusterCount': n_clusters_range,
        'gap': gaps
    })
    
    gp = pd.DataFrame({
        'clusterCount': n_clusters_range,
        'Gap_sk': gaps_sks
    })
            
    return optimal_k, resultsdf, gp
