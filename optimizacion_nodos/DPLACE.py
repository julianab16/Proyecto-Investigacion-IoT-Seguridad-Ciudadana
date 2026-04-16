import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd

class DPLACE:
    """
    DPLACE: LoRaWAN Gateway Placement Model for Dynamic IoT Scenarios
    Versión optimizada (manteniendo los mismos parámetros del artículo)
    """

    def __init__(
        self,
        area_size: float = 10_000.0,
        n_devices: int = 1000,
        n_scenarios: int = 10,
        max_gateways: int = 30,
        n_ref_datasets: int = 10,
        fcm_m: float = 2.0,
        fcm_epsilon: float = 1e-5,
        fcm_max_iter: int = 100,
        tx_power_dbm: float = 20.0,
        antenna_gain_dbi: float = 0.0,
        gateway_height_m: float = 30.0,
        device_height_m: float = 1.2,
        frequency_mhz: float = 868.0,
        coverage_sensitivity_dbm: float = -137.0,
    ):
        self.area_size = area_size
        self.n_devices = n_devices
        self.n_scenarios = n_scenarios
        self.max_gateways = max_gateways
        self.n_ref_datasets = n_ref_datasets
        self.fcm_m = fcm_m
        self.fcm_epsilon = fcm_epsilon
        self.fcm_max_iter = fcm_max_iter
        self.tx_power_dbm = tx_power_dbm
        self.antenna_gain_dbi = antenna_gain_dbi
        self.gateway_height_m = gateway_height_m
        self.device_height_m = device_height_m
        self.frequency_mhz = frequency_mhz
        self.coverage_sensitivity_dbm = coverage_sensitivity_dbm

        # Precalcular la distancia máxima de cobertura (R_max)
        self.R_max = self._compute_max_coverage_distance()
        # Precalcular la distancia máxima del área (diagonal)
        self.max_dist = np.sqrt(2) * self.area_size

    # --------------------------------------------------------------------------
    # Modelo de pérdida de camino (Okumura-Hata) y cobertura
    # --------------------------------------------------------------------------
    def _okumura_hata_path_loss(self, distance_m: float) -> float:
        d_km = distance_m / 1000.0
        f = self.frequency_mhz
        h_b = self.gateway_height_m
        h_m = self.device_height_m
        a_hm = (1.1 * np.log10(f) - 0.7) * h_m - (1.56 * np.log10(f) - 0.8)
        L = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_b) - a_hm +
             (44.9 - 6.55 * np.log10(h_b)) * np.log10(d_km))
        # Para distancias muy cortas, usar modelo de espacio libre
        free_space = 20.0 * np.log10(distance_m) + 32.44 + 20.0 * np.log10(f)
        return max(L, free_space)

    def _rssi(self, distance_m: float) -> float:
        pl = self._okumura_hata_path_loss(distance_m)
        return self.tx_power_dbm + self.antenna_gain_dbi - pl

    def _compute_max_coverage_distance(self) -> float:
        """Calcula la distancia máxima a la que RSSI = sensibilidad (búsqueda binaria)."""
        low, high = 0.0, self.area_size * np.sqrt(2)
        for _ in range(50):
            mid = (low + high) / 2.0
            if self._rssi(mid) >= self.coverage_sensitivity_dbm:
                low = mid
            else:
                high = mid
        return low

    # --------------------------------------------------------------------------
    # Fuzzy C-Means completamente vectorizado
    # --------------------------------------------------------------------------
    def _fcm(self, devices: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray, float]:
        N = devices.shape[0]
        m = self.fcm_m
        epsilon = self.fcm_epsilon
        max_iter = self.fcm_max_iter
        beta = 2.0 / (m - 1)   # exponente usado en la actualización de membresía

        # Inicialización aleatoria de U (matriz de membresía)
        U = np.random.random((N, n_clusters))
        U = U / U.sum(axis=1, keepdims=True)

        prev_U = None
        for _ in range(max_iter):
            # 1. Calcular centros (Ecuación 7) de forma vectorizada
            um = U ** m          # (N, M)
            centers = (um.T @ devices) / um.sum(axis=0)[:, np.newaxis]  # (M, 2)
            # Si algún centro tiene suma cero (raro), asignar posición aleatoria
            zero_sum = np.isnan(centers[:, 0])
            if zero_sum.any():
                centers[zero_sum] = np.random.rand(zero_sum.sum(), 2) * self.area_size

            # 2. Matriz de distancias con restricción de cobertura
            # Diferencia entre cada dispositivo y cada centro: (N, M, 2)
            diff = devices[:, np.newaxis, :] - centers[np.newaxis, :, :]
            eucl_dist = np.linalg.norm(diff, axis=2)   # (N, M)
            # Aplicar cobertura: si distancia > R_max, asignar max_dist
            dist_matrix = np.where(eucl_dist <= self.R_max, eucl_dist, self.max_dist)

            # 3. Actualizar U (Ecuación 6) de forma vectorizada
            # Evitamos bucles explícitos usando broadcasting
            # Para cada fila i, calculamos denominador: sum_k (d_ij / d_ik)^beta
            # Para evitar división por cero: donde d_ik == 0, la fracción es infinita → U = 1 en ese j
            # Primero detectamos filas con algún cero
            zero_mask = (dist_matrix == 0)
            new_U = np.zeros_like(U)
            for i in range(N):   # Este bucle aún es necesario para manejar ceros correctamente
                row = dist_matrix[i]
                if np.any(row == 0):
                    # El dispositivo coincide exactamente con un centro
                    j_zero = np.where(row == 0)[0][0]  # tomar el primero
                    new_U[i, :] = 0.0
                    new_U[i, j_zero] = 1.0
                else:
                    # Cálculo vectorizado de la fila
                    ratios = row[:, np.newaxis] / row[np.newaxis, :]   # (M, M)
                    denom = np.sum(ratios ** beta, axis=1)
                    new_U[i, :] = 1.0 / denom

            # 4. Verificar convergencia
            if prev_U is not None:
                if np.linalg.norm(new_U - prev_U) < epsilon:
                    break
            prev_U = new_U
            U = new_U

        # Calcular función objetivo Jm (Ecuación 1) de forma vectorizada
        # Distancias finales (usando la misma regla de cobertura)
        diff_final = devices[:, np.newaxis, :] - centers[np.newaxis, :, :]
        eucl_dist_final = np.linalg.norm(diff_final, axis=2)
        dist_final = np.where(eucl_dist_final <= self.R_max, eucl_dist_final, self.max_dist)
        Jm = np.sum((U ** m) * (dist_final ** 2))

        return centers, U, Jm

    # --------------------------------------------------------------------------
    # Gap statistics para FCM (pre-procesamiento)
    # --------------------------------------------------------------------------
    def _gap_statistics_fcm(self, devices: np.ndarray) -> int:
        max_clusters = min(self.max_gateways, self.n_devices - 1)
        if max_clusters < 1:
            return 1

        # Datos originales
        logW_orig = []
        for k in range(1, max_clusters + 1):
            _, _, Jm = self._fcm(devices, k)
            logW_orig.append(np.log(Jm))

        # Datasets de referencia
        logW_ref = np.zeros((self.n_ref_datasets, max_clusters))
        for b in range(self.n_ref_datasets):
            ref_devices = np.random.rand(self.n_devices, 2) * self.area_size
            for k in range(1, max_clusters + 1):
                _, _, Jm = self._fcm(ref_devices, k)
                logW_ref[b, k-1] = np.log(Jm)

        logW_ref_mean = np.mean(logW_ref, axis=0)
        gap = logW_ref_mean - np.array(logW_orig)
        sd = np.std(logW_ref, axis=0, ddof=1)
        s_k = sd * np.sqrt(1 + 1.0 / self.n_ref_datasets)

        best_k = 1
        for k in range(1, max_clusters):
            if gap[k-1] >= gap[k] - s_k[k]:
                best_k = k
                break
        else:
            best_k = max_clusters
        return best_k

    # --------------------------------------------------------------------------
    # Gap statistics para K-Means (fase de procesamiento)
    # --------------------------------------------------------------------------
    def _gap_statistics_kmeans(self, points: np.ndarray) -> int:
        max_clusters = min(self.max_gateways, len(points) - 1)
        if max_clusters < 1:
            return 1

        logW_orig = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(points)
            logW_orig.append(np.log(kmeans.inertia_))

        logW_ref = np.zeros((self.n_ref_datasets, max_clusters))
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        for b in range(self.n_ref_datasets):
            ref_points = np.random.uniform(min_xy, max_xy, size=(len(points), 2))
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=b, n_init=10)
                kmeans.fit(ref_points)
                logW_ref[b, k-1] = np.log(kmeans.inertia_)

        logW_ref_mean = np.mean(logW_ref, axis=0)
        gap = logW_ref_mean - np.array(logW_orig)
        sd = np.std(logW_ref, axis=0, ddof=1)
        s_k = sd * np.sqrt(1 + 1.0 / self.n_ref_datasets)

        best_k = 1
        for k in range(1, max_clusters):
            if gap[k-1] >= gap[k] - s_k[k]:
                best_k = k
                break
        else:
            best_k = max_clusters
        return best_k

    # --------------------------------------------------------------------------
    # Algoritmo principal DPLACE
    # --------------------------------------------------------------------------
    def place_gateways(self) -> np.ndarray:
        print("DPLACE: Generating multiple IoT scenarios...")
        all_gateway_positions = []

        for scenario_idx in range(self.n_scenarios):
            devices = np.random.rand(self.n_devices, 2) * self.area_size
            optimal_m = self._gap_statistics_fcm(devices)
            print(f"  Scenario {scenario_idx+1}: optimal gateways = {optimal_m}")
            centers, _, _ = self._fcm(devices, optimal_m)
            all_gateway_positions.extend(centers)

        all_positions = np.array(all_gateway_positions)
        print(f"Total collected gateway positions: {len(all_positions)}")
        final_k = self._gap_statistics_kmeans(all_positions)
        print(f"Final number of gateways: {final_k}")

        kmeans = KMeans(n_clusters=final_k, random_state=0, n_init=10)
        kmeans.fit(all_positions)
        return kmeans.cluster_centers_

    # --------------------------------------------------------------------------
    # Guardar resultados en CSV
    # --------------------------------------------------------------------------
    def save_gateways_to_csv(self, gateways: np.ndarray, filepath: str = "gateways_optimizados.csv"):
        max_coverage_dist = self.R_max
        rssi_at_max = self._rssi(max_coverage_dist)
        path_loss_at_max = self._okumura_hata_path_loss(max_coverage_dist)

        data = {
            'id': np.arange(1, len(gateways)+1),
            'x': gateways[:, 0],
            'y': gateways[:, 1],
            'max_coverage_distance': max_coverage_dist,
            'rssi_at_max_distance': rssi_at_max,
            'path_loss_at_max_distance': path_loss_at_max
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Gateway data saved to: {filepath}")
        print(f"Total gateways: {len(gateways)}")
        print(f"Max coverage distance: {max_coverage_dist:.1f} meters")
        print("\nGateway summary:")
        print(df.to_string(index=False))


# ------------------------------------------------------------------------------
# Ejemplo de uso (con los parámetros del artículo)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    dplace = DPLACE(
        area_size=10_000.0,
        n_devices=1000,
        n_scenarios=10,
        max_gateways=30,
        n_ref_datasets=10,
        fcm_m=2.0,
        fcm_epsilon=1e-5,
        tx_power_dbm=20.0,
        antenna_gain_dbi=0.0,
        gateway_height_m=30.0,
        device_height_m=1.2,
        frequency_mhz=868.0,
        coverage_sensitivity_dbm=-137.0
    )

    gateways = dplace.place_gateways()
    dplace.save_gateways_to_csv(gateways, "optimizacion_nodos/gateways_dplace.csv")

    print("\nFinal gateway positions (x, y) in meters:")
    for i, gw in enumerate(gateways):
        print(f"  GW {i+1}: ({gw[0]:.1f}, {gw[1]:.1f})")