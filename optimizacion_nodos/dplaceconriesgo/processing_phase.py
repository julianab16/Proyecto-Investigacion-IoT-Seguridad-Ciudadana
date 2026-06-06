"""
processing_phase.py : Fase de Procesamiento del modelo DPLACE - ADAPTADO PARA CONSOLIDAR NODOS OPTIMIZADOS.

Agrega posiciones de nodos optimizados de múltiples ejecuciones de preprocesamiento
y aplica K-Means para encontrar la ubicación global óptima de los nodos manteniendo
su CANTIDAD FIJA.

ESTADO: Lógica adaptada para optimización de NODOS en lugar de gateways.
        Elimina búsqueda de k óptimo (k es FIJO = cantidad de nodos originales).

Fórmulas implementadas (adaptadas de DPLACE para NODOS):
─────────────────────────────────────────────────────────────────────────────
  Sec 3.1  J_k = Σ_k Σ_j ||N_j^(k) - C_k||^2   [WCSS — función objetivo K-Means]
  Sec 3.2  Centroides finales C_k = K-Means.cluster_centers_  (posiciones óptimas de nodos)
  Sec 3.3  Distancia euclidiana ||N_j − C_k|| (interna en sklearn)
"""

import numpy as np
from sklearn.cluster import KMeans
# MODIFICADO: Se removió la importación de gap_statistics_kmeans (no se necesita k óptimo)


def processing(
    all_nodos_optimizados,  # MODIFICADO: Renombrado de 'all_gateways' a 'all_nodos_optimizados'
    n_nodos_fijos,  # NUEVO: Número FIJO de nodos (igual al CSV original, NO se busca k óptimo)
    verbose=True
):
    """
    Fase de Procesamiento: consolida posiciones de nodos optimizados de múltiples
    ejecuciones de preprocesamiento y determina las posiciones globales óptimas.

    Idea principal (adaptada para NODOS):
      Ejecutar preprocesamiento para cada perturbación de los nodos es inviable.
      DPLACE-NODOS ejecuta preprocesamiento B veces (con diferentes jitters/ruidos),
      acumula las posiciones N_j resultantes, y aplica K-Means sobre ese
      conjunto agregado para generalizar la solución sin recalcular continuamente.

    Parámetros
    ----------
    all_nodos_optimizados : list of ndarray  # MODIFICADO: Descripción más clara
        Cada elemento es un ndarray (k_i, 2) con las posiciones optimizadas de nodos
        N_j = (x_j, y_j) de una ejecución de preprocesamiento.
    n_nodos_fijos : int  # NUEVO: Número FIJO de nodos a optimizar
        Cantidad de nodos originales (NO se busca k óptimo, es FIJO).
    verbose : bool
        Imprimir información de progreso.

    Retorna
    -------
    dict con claves:
        'n_nodos'     : MODIFICADO: Número de nodos finales (FIJO = n_nodos_fijos)
        'nodos_finales': MODIFICADO: ndarray (n_nodos_fijos, 2) — posiciones finales óptimas de nodos
        'labels'      : etiquetas de cluster para cada nodo acumulado en la consolidación
        'all_nodes_data' : MODIFICADO: ndarray (N_total, 2) — todos los nodos acumulados
    """
    # ── Entrada: acumular todas las posiciones N_j de preprocesamiento ──
    # N_j = (x_j, y_j) para j ∈ [1, K] de cada ejecución
    # MODIFICADO: Cambiar de 'gateways' a 'nodos' semánticamente
    all_nodes_data = np.vstack(all_nodos_optimizados)  # MODIFICADO: Renombrar a 'all_nodes_data'

    if verbose:
        print(f"  [Processing] Posiciones de nodos acumuladas: {len(all_nodes_data)}")
        print(f"  [Processing] k FIJO = {n_nodos_fijos} (cantidad de nodos a consolidar)")

    # ── MODIFICADO: NO usar Gap Statistics ───────────────────────────────
    # ELIMINADO: Paso 1 de búsqueda de k óptimo
    # NUEVO: k_final = n_nodos_fijos (FIJO, NO variable)
    # Razón: La cantidad de nodos es conocida (CSV original), no necesita búsqueda
    k_final = n_nodos_fijos  # NUEVO: Usar cantidad fija de nodos

    # ── Paso 2: K-Means → posiciones finales de nodos C_k ───────────────
    # MODIFICADO: Comentario para aclarar que estamos consolidando NODOS
    # Documento adaptado sec 3: "se agrupan los nodos usando K-Means, los centroides C_k
    # son las posiciones finales óptimas de los nodos"
    # Distancia utilizada: euclidiana ||N_j - C_k|| (interna en sklearn)
    # MODIFICADO: J_k = Σ_k Σ_j ||N_j^(k) - C_k||^2  minimizado por K-Means
    if verbose:
        print(f"  [Processing] Ejecutando K-Means con K={k_final} (sin búsqueda de k óptimo)...")

    if k_final == 1:
        # Caso trivial: un solo nodo → centroide = media de todos los nodos acumulados
        final_nodes = all_nodes_data.mean(axis=0, keepdims=True)  # MODIFICADO: 'final_gateways' → 'final_nodes'
        labels = np.zeros(len(all_nodes_data), dtype=int)  # MODIFICADO: Todos pertenecen al único cluster
    else:
        # MODIFICADO: K-Means con k_final FIJO (no óptimo)
        # n_init=20: múltiples inicializaciones para mayor robustez
        # max_iter=500: iteraciones suficientes para convergencia
        # random_state=42: reproducibilidad
        km = KMeans(n_clusters=k_final, n_init=20, max_iter=500, random_state=42)
        km.fit(all_nodes_data)  # MODIFICADO: Entrenar K-Means con todos los nodos acumulados
        final_nodes = km.cluster_centers_  # MODIFICADO: 'final_gateways' → 'final_nodes' (C_k: posiciones óptimas)
        labels = km.labels_  # MODIFICADO: Etiquetas de cluster para cada nodo

    if verbose:
        print(f"  [Processing] Posiciones finales de nodos calculadas.")
        print(f"  [Processing] WCSS (Within-Cluster Sum of Squares): {km.inertia_:.2f}" if k_final > 1 else "")

    return {
        'n_nodos'     : k_final,  # MODIFICADO: 'k_optimal' → 'n_nodos' (cantidad FIJA de nodos)
        'nodos_finales': final_nodes,  # MODIFICADO: 'gateways' → 'nodos_finales' (posiciones óptimas de nodos)
        'labels'      : labels,  # MODIFICADO: Etiquetas de consolidación
        'all_nodes_data' : all_nodes_data  # MODIFICADO: 'all_gw_data' → 'all_nodes_data'
    }