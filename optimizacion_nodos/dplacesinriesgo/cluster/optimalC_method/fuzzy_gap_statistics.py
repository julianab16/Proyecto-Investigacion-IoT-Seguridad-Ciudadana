"""
fuzzy_gap_statistics.py : Gap Statistics para Fuzzy C-Means (Fase de Preprocesamiento).

Fórmulas implementadas (según documento DPLACE):
─────────────────────────────────────────────────────────────────────────────
  Gap(j)  = (1/B) * Σ log(J*_jb) − log(J_mj)               [ec. Gap Statistics]
  std(j)  = sqrt( (1/B) * Σ (log(J_jb) − Wkbs_j)^2 )       [desviación estándar]
  s(j)    = std(j) * sqrt(1 + 1/B)                           [error de simulación]
  Criterio: seleccionar menor j donde Gap(j) >= Gap(j+1) − s(j+1)
"""

import numpy as np
import pandas as pd
from cluster.fuzzycmeans.cmeans_algorithm import cmeans_with_sensitivity


def gap_statistics_fuzzy(data, nrefs, AREA_SIZE):
    """
    Calcula el número óptimo de clusters para Fuzzy C-Means usando Gap Statistics.
    Implementa el criterio original: menor j tal que Gap(j) >= Gap(j+1) - s(j+1).
    Versión vectorizada con NumPy para mayor eficiencia.
    """

    #n_samples = data.shape[0]
    #maxClusters = int(np.sqrt(n_samples))
    #maxClusters = max(2, maxClusters)
    maxClusters = 25

    print(f"  [Preprocessing] Gap Statistics (nrefs={nrefs} maxClusters={maxClusters})")

    n_clusters_range = np.arange(1, maxClusters+1)  # (1, 2, ..., maxClusters)

    # Preasignar arrays para almacenar resultados
    gaps = np.zeros(maxClusters)
    s_errors = np.zeros(maxClusters)
    std_j = np.zeros(maxClusters)
    gaps_sks = np.zeros(maxClusters)


    for idx, c in enumerate(n_clusters_range):
        # Array para almacenar los log(J*_jb) de las B referencias
        log_refs = np.zeros(nrefs)
        
        # Generar y procesar cada conjunto de referencia
        for i in range(nrefs):

            random_ref = np.random.uniform(low=0, high=AREA_SIZE, size=data.shape)

            # Calcular FCM sobre referencia (transponer para cmeans)
            _, _, _, _, jm, _, _ = cmeans_with_sensitivity(
                                        data=random_ref.T,
                                        c=c,
                                        m=2,
                                        error=0.005,
                                        maxiter=1000, 
                                        AREA_SIZE=AREA_SIZE
                                    )

            log_refs[i] = np.log(jm[-1])
     
        # Datos reales
        _, _, _, _, jm_real, _, _ = cmeans_with_sensitivity(
                                        data=data.T,
                                        c=c,
                                        m=2,
                                        error=0.005,
                                        maxiter=1000,
                                        AREA_SIZE=AREA_SIZE
                                    )

        log_real = np.log(jm_real[-1])

        if c in [1, 2]:
            print(f"  c={c}: log_real={log_real:.4f}, jm={jm_real[-1]:.2e}")

        # Gap(j) = media(log_refs) - log_real
        mean_log_ref = np.mean(log_refs)
        gaps[idx] = mean_log_ref - log_real
        
        # Desviación estándar (ddof=0 porque es población)
        std_j[idx] = np.std(log_refs, ddof=0)
        s_errors[idx] = std_j[idx] * np.sqrt(1 + 1 / nrefs)


    """ # Selección del k óptimo según Ecuación (5)
        valid_clusters = {}
        for idx in range(1, len(gaps) - 1):
            # Luego verificar criterio
            if gaps[idx] >= gaps[idx + 1] - s_errors[idx + 1]:  # ← CRITERIO REAL
                k = idx + 1
                valid_clusters[k] = gaps[idx]
                gaps_sks[idx] = gaps[idx] - gaps[idx + 1] + s_errors[idx + 1]

        # Construcción de DataFrames para gráficos
        resultsdf = pd.DataFrame({
            'clusterCount': n_clusters_range,
            'gap': gaps
        })
        
        gp = pd.DataFrame({
            'clusterCount': n_clusters_range,
            'Gap_sk': gaps_sks
        })

        # k óptimo: encontrar el k con MAYOR gap_value en valid_clusters
        if len(valid_clusters) > 0:
            # Top 2 de valid_clusters (ordenados por valor descendente)
            top_2_valid = sorted(valid_clusters.items(), key=lambda x: x[1], reverse=True)[:3]
            top_2_valid_ks = [k for k, v in top_2_valid]
            
            # Top 2 de gaps_sks (índices con mayores valores)
            top_2_gaps_sks_indices = np.argsort(-gaps_sks)[:3]
            top_2_gaps_sks_ks = top_2_gaps_sks_indices + 1  # Convertir a k (1-based)
            
            print(f"  Top 3 valid_clusters (k, gap): {top_2_valid}")
            print(f"  Top 3 gaps_sks (k): {top_2_gaps_sks_ks}")
            
            # Intersección
            intersection = np.intersect1d(top_2_valid_ks, top_2_gaps_sks_ks)
            
            if len(intersection) > 0:
                optimal_k = np.min(intersection)
                print(f"  Intersección: {intersection}")
                print(f"  Optimal k={optimal_k} (mínimo de intersección)")
            else:
                # Si no hay intersección, usar el mínimo de los top 2 valid_clusters
                optimal_k = min(top_2_valid_ks)
                print(f"  Sin intersección. Usando mínimo de top_2_valid_clusters: k={optimal_k}")
        else:
            # Fallback: usar k con máximo Gap directo
            optimal_k = np.argmax(gaps) + 1
            print(f"  No valid clusters encontrados. Usando k={optimal_k} (max Gap)")
                """
    # Selección del k óptimo según Ecuación (5)
    # Usar el primer k que cumpla el criterio
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