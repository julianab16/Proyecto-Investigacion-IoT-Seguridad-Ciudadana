"""
preprocessing_phase.py : Fase de Preprocesamiento del modelo DPLACE - ADAPTADO PARA OPTIMIZAR NODOS.

Aplica Fuzzy C-Means para optimizar las POSICIONES de los nodos IoT
manteniendo su CANTIDAD FIJA, considerando los gateways fijos como puntos de influencia.
"""

import numpy as np
# import pandas as pd  # MODIFICADO: pandas no se usa en este módulo
from cluster.fuzzycmeans.cmeans_algorithm import cmeans  # MODIFICADO: Se removió points_limit (no necesario)
# from cluster.optimalC_method.fuzzy_gap_statistics import gap_statistics_fuzzy  # MODIFICADO: Gap Statistics no se usa


def preprocessing(
    X_nodos,  # MODIFICADO: Renombrado de X a X_nodos para mayor claridad
    gateways_fijos,  # NUEVO: Parámetro que recibe los gateways fijos del despliegue
    n_nodes_optimize,  # NUEVO: Número FIJO de nodos a optimizar (igual al CSV original)
    m=25,
    error=0.005,
    maxiter=1000,
    verbose=True,
    sample_weights=None,
    seed=None,
    **kwargs  # MODIFICADO: aceptar parámetros legacy (nrefs, maxClusters, maxPoints) sin romper llamadas
):
    """
    Fase de Preprocesamiento: optimiza las POSICIONES de los nodos IoT
    manteniendo su CANTIDAD FIJA usando Fuzzy C-Means.

    Parámetros
    ----------
    X_nodos : ndarray, shape (N_nodes, 2)  # MODIFICADO: Clarificar que son nodos, no dispositivos genéricos
        Coordenadas 2D (x, y) de los nodos IoT (con ruido/perturbación).
        N_i = (x_i, y_i) para i ∈ [1, N_nodes].
    gateways_fijos : ndarray, shape (K_gw, 2)  # NUEVO: Gateways fijos como anclajes
        Coordenadas 2D de los gateways fijos del despliegue.
    n_nodes_optimize : int  # NUEVO: Número FIJO de nodos a optimizar
        Cantidad de nodos a optimizar (NO se busca k óptimo, es FIJO).
    m : float
        Parámetro de ponderación (fuzziness) para FCM. Típicamente m=2.
    error : float
        Umbral de convergencia ε para FCM: ||U^(r+1) - U^(r)|| <= ε.
    maxiter : int
        Máximo de iteraciones para FCM.
    verbose : bool
        Imprimir información de progreso.

    Retorna
    -------
    dict con claves:
        'n_nodes'    : MODIFICADO: Número de nodos optimizados (FIJO = n_nodes_optimize)
        'nodos_opt'  : MODIFICADO: ndarray (n_nodes_optimize, 2) — posiciones optimizadas de nodos
        'u'          : matriz de membresía U_ij, shape (n_nodes_optimize, N_total)
        'd'          : matriz de distancias euclidianas, shape (n_nodes_optimize, N_total)
        'fpc'        : coeficiente de partición fuzzy (FPC)
        'jm'         : historial de la función objetivo J_mj
        'X_original' : posiciones originales de nodos (antes de optimizar)
        'gateways'   : NUEVO: referencia a los gateways fijos usados en el clustering
    """
    if kwargs and verbose:
        print(f"  [Preprocessing] Parámetros ignorados: {', '.join(kwargs.keys())}")  # MODIFICADO: aviso compatibilidad

    X_nodos = np.asarray(X_nodos)  # MODIFICADO: asegurar ndarray
    if X_nodos.ndim != 2 or X_nodos.shape[1] != 2:
        raise ValueError("X_nodos debe ser un ndarray de forma (N, 2)")  # MODIFICADO: validación de forma

    if gateways_fijos is None:
        gateways_fijos = np.empty((0, 2))  # MODIFICADO: evitar errores cuando no hay gateways
    gateways_fijos = np.asarray(gateways_fijos)  # MODIFICADO: asegurar ndarray
    if gateways_fijos.ndim == 1:
        gateways_fijos = gateways_fijos.reshape(1, -1)  # MODIFICADO: normalizar forma (1, 2)
    if gateways_fijos.size > 0 and gateways_fijos.shape[1] != 2:
        raise ValueError("gateways_fijos debe tener forma (K, 2)")  # MODIFICADO: validación de forma

    # ── Preparar datos para cmeans: shape (S=2, N) ────────────────────────
    # cmeans espera data con shape (S, N): S=features, N=samples
    # MODIFICADO: Extraer x, y de X_nodos (sin cambios en lógica)
    xpts = X_nodos[:, 0]
    ypts = X_nodos[:, 1]
    
    # MODIFICADO: Concatenar nodos + gateways fijos para influenciar el clustering
    # Los gateways fijos actúan como "puntos de anclaje" que guían la optimización
    # Idea: El FCM considerará tanto los nodos como los gateways en la definición de clusters
    X_combined = np.vstack([X_nodos, gateways_fijos])  # NUEVO: Combinar nodos + gateways
    xpts_combined = X_combined[:, 0]  # NUEVO: Extraer x de datos combinados
    ypts_combined = X_combined[:, 1]  # NUEVO: Extraer y de datos combinados
    alldata = np.vstack((xpts_combined, ypts_combined))  # NUEVO: shape (2, N_total)

    if verbose:
        print(f"  [Preprocessing] Optimizando {n_nodes_optimize} nodos con {len(gateways_fijos)} gateways fijos...")
        print(f"  [Preprocessing] Datos totales para FCM: {len(X_combined)} puntos (nodos + gateways)")

    # ── MODIFICADO: NO usar Gap Statistics ─────────────────────────────────
    # ELIMINADO: Paso 1 de búsqueda de k óptimo
    # NUEVO: k_final = n_nodes_optimize (FIJO, NO variable)
    # Razón: Ya sabemos cuántos nodos queremos optimizar, no necesitamos buscarlo
    k_clusters = max(1, int(np.sqrt(n_nodes_optimize)))   # ej: sqrt(200)=14
    if verbose:
        print(f"  [Preprocessing] Número de clusters de atracción: {k_clusters}")

    # ── MODIFICADO: NO usar points_limit ───────────────────────────────────
    # ELIMINADO: Paso 2 de ajuste por puntos máximos
    # Razón: No limitamos por puntos/cluster, optimizamos las posiciones de nodos directamente

    # ── Paso 3: FCM con k_final → posiciones optimizadas de nodos ──────────
    # MODIFICADO: Comentario para aclarar que estamos optimizando NODOS, no gateways
    # Fórmula centroide: n_j = Σ(m_ij^m · P_i) / Σ(m_ij^m) donde P_i ∈ {nodos + gateways}
    # Criterio convergencia: ||U^(r+1) − U^(r)|| <= ε
    # NUEVO: Los centroides FCM representan las posiciones optimizadas de los nodos
    if verbose:
        print(f"  [Preprocessing] Ejecutando Fuzzy C-Means (c={k_clusters}) para optimizar nodos...")

    # Construcción de ponderación combinada
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(X_nodos):
            raise ValueError("sample_weights debe tener la misma longitud que X_nodos")
        # Pesos para los gateways fijos (por defecto 1, pero podría ser 0 si no deben influir)
        gw_weights = np.ones(len(gateways_fijos), dtype=np.float64)
        all_weights = np.concatenate([sample_weights, gw_weights])
    else:
        all_weights = None

    cntr, u, u0, d, jm, p, fpc = cmeans(
        data=alldata,
        c=k_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=seed,
        sf=12,                     # o sacarlo como parámetro si se desea
        sample_weights=all_weights # ← pasar los pesos combinados
    )

    # cntr tiene shape (k_clusters, 2)
    # Asignar cada nodo original al centroide más cercano
    dists_to_cntr = np.linalg.norm(
        X_nodos[:, None, :] - cntr[None, :, :], axis=2
    )   # (N, k_clusters)
    assigned_cluster = np.argmin(dists_to_cntr, axis=1)   # (N,)

    # ── Calcular factor de atracción variable por peso de inseguridad ──
    if sample_weights is not None:
        # Normalizar pesos al rango [0,1]
        sw_min = np.min(sample_weights)
        sw_max = np.max(sample_weights)
        if sw_max > sw_min:
            sw_norm = (sample_weights - sw_min) / (sw_max - sw_min)
        else:
            sw_norm = np.zeros_like(sample_weights)
        # Alfa variable: nodos más inseguros se mueven más hacia el centroide
        alpha = 0.3 + 0.7 * sw_norm   # rango [0.3, 1.0]
    else:
        alpha = np.full(len(X_nodos), 0.7)

    # ── Mover cada nodo hacia su centroide asignado ────────────────────
    nodos_opt = X_nodos.copy()

    for i in range(len(X_nodos)):
        nodos_opt[i] = (1 - alpha[i]) * X_nodos[i] + alpha[i] * cntr[assigned_cluster[i]]

    if verbose:
        print(f"  [Preprocessing] FCM convergió en {p} iteraciones, FPC={fpc:.4f}")

    return {
        'n_nodes'    : n_nodes_optimize,  # MODIFICADO: Cambiar 'k_final' a 'n_nodes' para claridad
        'nodos_opt'  : nodos_opt,  # MODIFICADO: Renombrar 'gateways' a 'nodos_opt' (posiciones optimizadas de nodos)
        'u'          : u,  # MODIFICADO: Matriz U_ij con membresía de nodos vs puntos de datos
        'd'          : d,  # MODIFICADO: Distancias entre nodos optimizados y datos (nodos + gateways)
        'fpc'        : fpc,
        'jm'         : jm,
        'X_original' : X_nodos,  # MODIFICADO: Guardar nodos originales para comparación
        'gateways'   : gateways_fijos  # NUEVO: Guardar referencia a gateways fijos
    }