"""
fuzzy_gap_statistics.py : Gap Statistics para Fuzzy C-Means (Fase de Preprocesamiento).

CORRECCIONES APLICADAS vs versión original:
─────────────────────────────────────────────────────────────────────────────
BUG 1 [CRÍTICO] — Orientación de datos (data shape)
  ORIGINAL : cmeans recibía data con shape (N, 2), pero cmeans espera (S=2, N).
             Esto causaba que n = data.shape[1] = 2 en vez de N,
             generando u0 de shape (C, 2) y fallos en las multiplicaciones matriciales.
  CORREGIDO: Se transpone data antes de pasarlo a cmeans:
             cmeans(data=data.T, ...) o cmeans(data=randomReference.T, ...)

BUG 2 [LÓGICO] — gaps_sks solo se asignaba cuando la condición se cumplía
  ORIGINAL : gaps_sks[k] = valor  SOLO SI gaps[k] >= gaps[k+1] - sError[k+1]
             Cuando la condición NO se cumplía, gaps_sks[k] quedaba en 0,
             introduciendo ceros falsos que distorsionaban el argmax.
  CORREGIDO: gaps_sks[k] = gaps[k] - gaps[k+1] - sError[k+1]  SIEMPRE.
             El valor positivo indica que el criterio se cumple (buen k),
             el valor negativo indica que NO se cumple. El argmax es correcto.

Fórmulas implementadas (según documento DPLACE):
─────────────────────────────────────────────────────────────────────────────
  Gap(j)  = (1/B) * Σ log(J*_jb) − log(J_mj)               [ec. Gap Statistics]
  std(j)  = sqrt( (1/B) * Σ (log(J_jb) − Wkbs_j)^2 )       [desviación estándar]
  s(j)    = std(j) * sqrt(1 + 1/B)                           [error de simulación]
  Criterio: seleccionar menor j donde Gap(j) >= Gap(j+1) − s(j+1)
"""

import numpy as np
import pandas as pd
from cluster.fuzzycmeans.cmeans_algorithm import cmeans


def gap_statistics_fuzzy(data, nrefs=3, maxClusters=15):
    """
    Calcula el número óptimo de clusters C para Fuzzy C-Means usando Gap Statistic.
    Basado en: Sentelle, Hong, Georgiopoulos, Anagnostopoulos.

    Parámetros
    ----------
    data : ndarray, shape (N, 2)
        Coordenadas 2D de los dispositivos IoT. shape = (N_dispositivos, 2).
        NOTA: internamente se transpone a (2, N) para cmeans.
    nrefs : int
        Número B de conjuntos de datos aleatorios de referencia.
    maxClusters : int
        Número máximo de clusters M a evaluar.

    Retorna
    -------
    k         : int — número óptimo de clusters (gateways)
    resultsdf : DataFrame — columnas ['clusterCount', 'gap'] para curva Gap
    gp        : DataFrame — columnas ['clusterCount', 'Gap_sk'] para gráfica de barras
    """
    # ── Arreglos de acumulación ────────────────────────────────────────────
    sdk    = np.zeros(maxClusters)          # desviación estándar por j
    sError = np.zeros(maxClusters)          # error de simulación s(j)
    BWkbs  = np.zeros(nrefs)               # log(J*_jb) para cada referencia b
    Wks    = np.zeros(maxClusters)          # log(J_mj) datos reales
    Wkbs   = np.zeros(maxClusters)          # (1/B)*Σ log(J*_jb) = media referencias
    gaps_sks = np.zeros(maxClusters - 1)   # Gap(k) − Gap(k+1) − s(k+1)

    gaps      = np.zeros(maxClusters - 1)  # Gap(j) para cada j
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    gp        = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    # ── Bucle principal: evaluar cada cantidad de clusters j ──────────────
    for gap_index, c in enumerate(range(1, maxClusters)):

        # ── Paso 1: B referencias aleatorias → calcular log(J*_jb) ──────
        # Fórmula: Gap(j) = (1/B)*Σ_b log(J*_jb) − log(J_mj)
        for i in range(nrefs):
            # Generar conjunto aleatorio con la misma forma que los datos originales
            randomReference = np.random.random_sample(size=data.shape)  # shape (N, 2)

            # FIX BUG 1: randomReference.T → shape (2, N) que es lo que espera cmeans
            cntr, u, u0, d, jm, p, fpc = cmeans(
                data=randomReference.T,   # ← CORREGIDO: .T para (2, N)
                c=c, m=2, error=0.005, maxiter=1000
            )
            # Usar último valor de la función objetivo (más estable que la media)
            BWkbs[i] = np.log(jm[-1])

        # ── Paso 2: Ajustar FCM a datos reales → calcular log(J_mj) ─────
        # FIX BUG 1: data.T → shape (2, N) que es lo que espera cmeans
        cntr, u, u0, d, jm, p, fpc = cmeans(
            data=data.T,   # ← CORREGIDO: .T para (2, N)
            c=c, m=2, error=0.005, maxiter=1000
        )

        # ── Paso 3: Calcular Gap(j) ──────────────────────────────────────
        # Gap(j) = (1/B)*Σ log(J*_jb) − log(J_mj)
        Wks[gap_index]  = np.log(jm[-1])
        gap             = np.mean(BWkbs) - Wks[gap_index]
        gaps[gap_index] = gap

        resultsdf = pd.concat(
            [resultsdf, pd.DataFrame({'clusterCount': [c], 'gap': [gap]})],
            ignore_index=True
        )

        # ── Paso 4: Desviación estándar y error de simulación ────────────
        # std(j)  = sqrt( (1/B) * Σ (log(J_jb) − Wkbs_j)^2 )
        # s(j)    = std(j) * sqrt(1 + 1/B)
        Wkbs[gap_index]   = np.mean(BWkbs)
        sdk[gap_index]    = np.sqrt(np.mean((BWkbs - Wkbs[gap_index]) ** 2))
        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1.0 / nrefs)

    # ── Calcular gaps_sks = Gap(k) − Gap(k+1) − s(k+1) ──────────────────
    # Criterio del documento: seleccionar menor j donde Gap(j) >= Gap(j+1) − s(j+1)
    # Un valor positivo de gaps_sks[k] indica que el criterio se CUMPLE para k.
    for k in range(len(gaps)):
        if k < len(gaps) - 1:
            # FIX BUG 2: asignar el valor SIEMPRE (no solo cuando condición se cumple)
            # ORIGINAL: solo asignaba si gaps[k] >= gaps[k+1] - sError[k+1]
            #           → dejaba gaps_sks[k]=0 cuando NO se cumplía (valor falso)
            # CORRECTO: asignar siempre el valor real; positivo=cumple, negativo=no cumple
            gaps_sks[k] = gaps[k] - gaps[k + 1] - sError[k + 1]
        else:
            gaps_sks[k] = -20  # penalización para el último índice

        gp = pd.concat(
            [gp, pd.DataFrame({'clusterCount': [k + 1], 'Gap_sk': [gaps_sks[k]]})],
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