"""
kmeans_gap_statistics.py : Gap Statistics para K-Means (Fase de Procesamiento).
Encuentra el número óptimo de clusters de gateways a partir de múltiples
ejecuciones de preprocesamiento.

CORRECCIÓN APLICADA vs versión original:
─────────────────────────────────────────────────────────────────────────────
BUG [LÓGICO] — gaps_sks solo se asignaba cuando la condición se cumplía

  ORIGINAL (líneas 74-76):
      if gaps[k_idx] >= gaps[k_idx + 1] - sError[k_idx + 1]:
          gaps_sks[k_idx] = gaps[k_idx] - gaps[k_idx + 1] - sError[k_idx + 1]
      # Si la condición NO se cumple → gaps_sks[k_idx] queda en 0 (VALOR FALSO)

  PROBLEMA: El 0 falso hace que el argmax posterior pueda elegir k's incorrectos.
  Si ningún k cumple el criterio, todos los gaps_sks quedan en 0, y el top-3
  se selecciona por posición arbitraria (desempate no determinista).

  CORREGIDO:
      gaps_sks[k_idx] = gaps[k_idx] - gaps[k_idx + 1] - sError[k_idx + 1]  # SIEMPRE
      # Positivo → criterio Gap(k) >= Gap(k+1) - s(k+1) se CUMPLE  (buen k)
      # Negativo → criterio NO se cumple
      # El argmax selecciona correctamente el k con mayor margen positivo

Fórmulas implementadas (según documento DPLACE - Processing Phase):
─────────────────────────────────────────────────────────────────────────────
  J_k     = Σ_k Σ_j ||GW_j^(k) - C_k||^2     [función objetivo K-Means / WCSS]
  Gap(k)  = (1/B)*Σ_b log(J*_kb) − log(J_k)  [Gap Statistics]
  std(k)  = sqrt( (1/B)*Σ_b (log(J*_kb) − Wkbs_k)^2 )
  s(k)    = std(k) * sqrt(1 + 1/B)            [error de simulación]
  Criterio: seleccionar menor k donde Gap(k) >= Gap(k+1) − s(k+1)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def gap_statistics_kmeans(data, nrefs, maxClusters):
    """
    Calcula el número óptimo de clusters K para K-Means usando Gap Statistic.
    Usada en la Fase de Procesamiento de DPLACE.

    Parámetros
    ----------
    data : ndarray, shape (N_gateways, 2)
        Posiciones de gateways acumuladas de múltiples ejecuciones de
        preprocesamiento. GW_j = (x_j, y_j).
    nrefs : int
        Número B de conjuntos de datos aleatorios de referencia.
    maxClusters : int
        Número máximo de clusters K a evaluar.

    Retorna
    -------
    k         : int — número óptimo de clusters (gateways)
    resultsdf : DataFrame — columnas ['clusterCount', 'gap'] para curva Gap
    gp        : DataFrame — columnas ['clusterCount', 'Gap_sk'] para gráfica de barras
    """
    # ── Arreglos de acumulación ────────────────────────────────────────────
    sdk      = np.zeros(maxClusters)         # desviación estándar std(k)
    sError   = np.zeros(maxClusters)         # error de simulación s(k)
    BWkbs    = np.zeros(nrefs)               # log(J*_kb) para cada referencia b
    Wks      = np.zeros(maxClusters)         # log(J_k) datos reales
    Wkbs     = np.zeros(maxClusters)         # (1/B)*Σ log(J*_kb)
    gaps_sks = np.zeros(maxClusters - 1)     # Gap(k) − Gap(k+1) − s(k+1)

    gaps      = np.zeros(maxClusters - 1)
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    gp        = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    # ── Bucle principal: evaluar cada número de clusters k ────────────────
    for gap_index, c in enumerate(range(1, maxClusters+1)):

        # ── Paso 1: B referencias aleatorias → log(J*_kb) ────────────────
        # Genera B conjuntos desorganizados b ∈ {1,...,B} con la misma
        # extensión que los datos reales, calcula J*_kb (WCSS) para cada uno.
        for i in range(nrefs):
            # Conjunto aleatorio con la misma forma que los datos de gateways
            random_ref = np.random.random_sample(size=data.shape)  # (N_gw, 2)

            if c == 1:
                # Un solo cluster: WCSS = Σ ||x_i - media||^2
                inertia = np.sum((random_ref - random_ref.mean(axis=0)) ** 2)
            else:
                km = KMeans(n_clusters=c, n_init=5, max_iter=300, random_state=i)
                km.fit(random_ref)
                inertia = km.inertia_   # J*_kb = WCSS del conjunto aleatorio

            BWkbs[i] = np.log(inertia + 1e-10)   # log(J*_kb)

        # ── Paso 2: Ajustar K-Means a datos reales → log(J_k) ───────────
        # J_k = Σ_k Σ_j ||GW_j^(k) - C_k||^2  (WCSS real)
        if c == 1:
            inertia_real = np.sum((data - data.mean(axis=0)) ** 2)
        else:
            km = KMeans(n_clusters=c, n_init=10, max_iter=300, random_state=42)
            km.fit(data)
            inertia_real = km.inertia_

        Wks[gap_index] = np.log(inertia_real + 1e-10)

        # ── Paso 3: Gap(k) = (1/B)*Σ log(J*_kb) − log(J_k) ─────────────
        gap             = np.mean(BWkbs) - Wks[gap_index]
        gaps[gap_index] = gap

        resultsdf = pd.concat(
            [resultsdf, pd.DataFrame({'clusterCount': [c], 'gap': [gap]})],
            ignore_index=True
        )

        # ── Paso 4: std(k) y s(k) ────────────────────────────────────────
        # std(k) = sqrt( (1/B)*Σ(log(J*_kb) − Wkbs_k)^2 )
        # s(k)   = std(k) * sqrt(1 + 1/B)
        Wkbs[gap_index]   = np.mean(BWkbs)
        sdk[gap_index]    = np.sqrt(np.mean((BWkbs - Wkbs[gap_index]) ** 2))
        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1.0 / nrefs)

    # ── Calcular gaps_sks = Gap(k) − Gap(k+1) − s(k+1) ──────────────────
    # Criterio del documento: seleccionar menor k donde Gap(k) >= Gap(k+1) − s(k+1)
    for k_idx in range(len(gaps)):
        if k_idx < len(gaps) - 1:
            # FIX: asignar el valor SIEMPRE, no solo cuando la condición se cumple.
            # ORIGINAL: solo asignaba si gaps[k_idx] >= gaps[k_idx+1] - sError[k_idx+1]
            #           dejando gaps_sks[k_idx]=0 cuando NO se cumplía (valor falso).
            # CORRECTO: valor positivo → criterio cumple (buen k)
            #           valor negativo → criterio no cumple
            gaps_sks[k_idx] = gaps[k_idx] - gaps[k_idx + 1] - sError[k_idx + 1]
        else:
            gaps_sks[k_idx] = -20   # penalización para el último índice

        gp = pd.concat(
            [gp, pd.DataFrame({'clusterCount': [k_idx + 1], 'Gap_sk': [gaps_sks[k_idx]]})],
            ignore_index=True
        )

    # ── Selección del k óptimo ────────────────────────────────────────────
    # Top-3 por Gap más alto
    iter_points = [
        x[0] + 1 for x in sorted(enumerate(gaps), key=lambda x: x[1], reverse=True)[:3]
    ]
    # Top-3 por gaps_sks más alto (mayor margen sobre el criterio)
    iter_points_sk = [
        x[0] + 1 for x in sorted(enumerate(gaps_sks), key=lambda x: x[1], reverse=True)[:3]
    ]

    # Intersección: k que aparece en ambos top-3 → más robusto
    a = [g for g in iter_points_sk if g in iter_points]

    if len(a) != 0:
        # Evitar k=1 (trivial) si hay alternativas
        a_sin_uno = [x for x in a if x != 1]
        k = min(a_sin_uno) if a_sin_uno else min(iter_points_sk)
    else:
        k = min(iter_points_sk)

    return k, resultsdf, gp