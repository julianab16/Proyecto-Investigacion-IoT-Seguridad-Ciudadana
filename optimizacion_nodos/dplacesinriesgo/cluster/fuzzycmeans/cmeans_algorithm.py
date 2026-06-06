"""
cmeans_algorithm.py : Fuzzy C-means clustering algorithm.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from cluster.fuzzycmeans.normalize_columns import normalize_columns, normalize_power_columns

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans_with_sensitivity(
    data,
    c,
    m,
    error,
    maxiter,
    metric='euclidean',
    init=None,
    seed=None,
    AREA_SIZE=10000):

    # Parámetros simples de propagación
    tx_power_dbm = 14

    # Inicialización FCM
    if init is None:
        if seed is not None:
            np.random.seed(seed)

        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()

    u = np.fmax(init, np.finfo(np.float64).eps)

    jm = np.zeros(0)
    p = 0

    while p < maxiter - 1:

        u_old = u.copy()

        u_old = normalize_columns(u_old)
        u_old = np.fmax(u_old, np.finfo(np.float64).eps)

        # =========================
        # Centros FCM (Eq. 7)
        # =========================
        um = u_old ** m

        cntr = um.dot(data.T) / np.atleast_2d(
            um.sum(axis=1)
        ).T

        # =========================
        # Distancia euclidiana
        # =========================
        dist_euclidean = cdist(
            data.T,
            cntr,
            metric=metric
        ).T

        # Evitar log10(0)
        d_safe = np.fmax(dist_euclidean, np.finfo(np.float64).eps)

        # Frecuencia LoRaWAN (MHz)
        reference_distance = 1.0      # d0 en metros
        pl0 = 40.0                    # PL(d0) en dB
        n_exp = 2.7                   # exponente de pérdidas

        # d_safe contiene la distancia euclidiana en metros (ya calculada)
        # Calcular path loss y RSSI
        path_loss = pl0 + 10 * n_exp * np.log10(d_safe / reference_distance)
                
        rssi = tx_power_dbm - path_loss
        
        # =========================
        # Verificación coberturap
        # =========================
        SF12_SENSITIVITY = -137  # dBm

        no_coverage = (
            rssi < SF12_SENSITIVITY
        )

        # =========================
        # Penalización distancia
        # =========================
        dist_adjusted = dist_euclidean.copy()

        max_distance = np.sqrt(2) * np.sqrt(AREA_SIZE)

        # Si no hay cobertura → asignar máxima distancia
        dist_adjusted[no_coverage] = max_distance

        dist_adjusted = np.fmax(
            dist_adjusted,
            np.finfo(np.float64).eps
        )
        
        # =========================
        # Función objetivo (Eq. 1)
        # =========================
        jm_val = np.sum(
            (u_old ** m) *
            (dist_adjusted ** 2) #ejecutar
        )

        # =========================
        # Membresía (Eq. 6)
        # =========================
        power = -2.0 / (m - 1)

        u = normalize_power_columns(
            dist_adjusted,
            power
        )

        #print(f"jm_val: {jm_val}")

        jm = np.hstack((jm, jm_val))

        p += 1

        # Eq. 9
        if np.linalg.norm(u - u_old) < error:
            break

    error = np.linalg.norm(u - u_old)
    fpc = _fp_coeff(u)

    return cntr, u, u0, dist_adjusted, jm, p, fpc