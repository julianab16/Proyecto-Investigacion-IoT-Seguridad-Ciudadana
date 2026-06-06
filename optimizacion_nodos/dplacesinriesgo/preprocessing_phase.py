"""
preprocessing_phase.py : Fase de Preprocesamiento del modelo DPLACE.

Aplica Gap Statistics + Fuzzy C-Means para determinar el número
y posiciones iniciales de los gateways en un despliegue IoT.

Fórmulas implementadas según el documento DPLACE:
─────────────────────────────────────────────────────────────────────────────
  FCM centroide:  c_j = Σ(m_ij^m · ED_i) / Σ(m_ij^m)
  FCM membresía:  m_ij = 1 / Σ_k (||ED_i - GW_j|| / ||ED_i - GW_k||)^(2/(m-1))
  Convergencia:   ||U^(r+1) - U^(r)|| <= ε
"""

import numpy as np
from cluster.fuzzycmeans.cmeans_algorithm import cmeans_with_sensitivity
from cluster.optimalC_method.fuzzy_gap_statistics import gap_statistics_fuzzy

def preprocessing(
    X,                         # shape (N, 2)
    nrefs,
    AREA_SIZE,
    m=2,
    error=0.005,
    maxiter=1000,
    verbose=True
):
    """
    Fase de Preprocesamiento según el texto original:
      - Gap Statistics con criterio de Ecuación (5)
      - FCM con verificación de sensibilidad RSSI (SF)
      - Sin points_limit
    """
    # Extraer las 3 coordenadas (x, y, z) para clustering
    X_2d = X[:, :3]
    
    # Preparar datos: shape (3, N) para cmeans
    alldata = np.vstack((X_2d[:, 0], X_2d[:, 1], X_2d[:, 2]))


    # Paso 1: Gap Statistics (versión corregida)
    k_optimal, gap_df, gap_sk_df = gap_statistics_fuzzy(X_2d, nrefs, AREA_SIZE)
    if verbose:
        print(f"  [Preprocessing] Número óptimo de gateways (Gap) = {k_optimal}")
    
    # Paso 2: FCM con verificación de sensibilidad
    if verbose:
        print(f"  [Preprocessing] Fuzzy C-Means con c={k_optimal} y verificación RSSI/SF...")
    
    # Llamada a FCM modificada
    cntr, u, u0, d, jm, p, fpc = cmeans_with_sensitivity(
        data=alldata,
        c=k_optimal,
        m=m,
        error=error,
        maxiter=maxiter,
        AREA_SIZE=AREA_SIZE)
    
    if verbose:
        print(f"  [Preprocessing] FCM convergió en {p} iteraciones, FPC={fpc:.4f}")
    
    return {
        'k_gap': k_optimal,
        'k_final': k_optimal,
        'gateways': cntr,
        'u': u,
        'd': d,
        'fpc': fpc,
        'jm': jm,
        'gap_df': gap_df,
        'gap_sk_df': gap_sk_df,
        'X': X,           # Guardar X completo (con z)
        'X_2d': X_2d     # Guardar X solo en 2D
    }