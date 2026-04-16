import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
import warnings
import pandas as pd

class DPLACE:
    """
    DPLACE: LoRaWAN Gateway Placement Model for Dynamic IoT Scenarios
    Implements the three-phase algorithm: pre-processing, processing, validation.
    (Validation phase is omitted as it involves NS-3 simulation)
    """

    def __init__(
        self,
        area_size: float = 10_000.0,  # area side in meters (10 km -> 100 km²)
        n_devices: int = 1000,
        n_scenarios: int = 10,       # T: number of different IoT deployments for dynamism
        max_gateways: int = 30,      # maximum number of gateways to test in Gap statistics
        n_ref_datasets: int = 10,    # B: number of reference datasets for Gap
        fcm_m: float = 2.0,          # weighting parameter for FCM
        fcm_epsilon: float = 1e-5,   # convergence tolerance for FCM
        fcm_max_iter: int = 100,
        tx_power_dbm: float = 20.0,  # gateway transmission power (dBm)
        antenna_gain_dbi: float = 0.0,
        gateway_height_m: float = 30.0,   # h_b
        device_height_m: float = 1.2,     # h_m
        frequency_mhz: float = 868.0,
        # Sensitivity for SF12 (most robust) as coverage threshold (dBm)
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

        # Sensitivity for different SF (Table 3) - not directly used in coverage check,
        # but kept for reference. Coverage uses the worst-case (SF12).
        self.sf_sensitivity = {
            7: -125, 8: -128, 9: -131, 10: -134, 11: -136, 12: -137
        }

    # --------------------------------------------------------------------------
    # Path loss model (Okumura-Hata urban)
    # --------------------------------------------------------------------------
    def _okumura_hata_path_loss(self, distance_m: float) -> float:
        """
        Compute path loss in dB using Okumura-Hata model for urban area.
        distance_m: distance between gateway and device in meters.
        Returns path loss in dB.
        """
        d_km = distance_m / 1000.0
        f = self.frequency_mhz
        h_b = self.gateway_height_m
        h_m = self.device_height_m

        # a(h_m) for medium/small city
        a_hm = (1.1 * np.log10(f) - 0.7) * h_m - (1.56 * np.log10(f) - 0.8)

        L = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_b) - a_hm +
             (44.9 - 6.55 * np.log10(h_b)) * np.log10(d_km))
        # Ensure non-negative path loss (for very short distances, clamp to free-space min)
        return max(L, 20.0 * np.log10(distance_m) + 32.44 + 20.0 * np.log10(f))

    def _rssi(self, distance_m: float) -> float:
        """Received Signal Strength Indicator in dBm."""
        pl = self._okumura_hata_path_loss(distance_m)
        rssi = self.tx_power_dbm + self.antenna_gain_dbi - pl
        return rssi

    def _is_covered(self, distance_m: float) -> bool:
        """Check if a device at given distance can be covered by a gateway."""
        rssi = self._rssi(distance_m)
        return rssi >= self.coverage_sensitivity_dbm

    def _max_coverage_distance(self) -> float:
        """
        Calculate the maximum coverage distance where RSSI equals the sensitivity threshold.
        Uses binary search to find the distance.
        """
        low, high = 0.0, self.area_size * np.sqrt(2)
        
        for _ in range(50):  # 50 iterations for precision
            mid = (low + high) / 2.0
            if self._is_covered(mid):
                low = mid
            else:
                high = mid
        
        return low

    # --------------------------------------------------------------------------
    # Fuzzy C-Means with coverage constraint
    # --------------------------------------------------------------------------
    def _fcm(self, devices: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray, float]:
        N = devices.shape[0]
        m = self.fcm_m
        epsilon = self.fcm_epsilon
        max_iter = self.fcm_max_iter
        max_dist = np.sqrt(2) * self.area_size

        # Inicializar matriz de membresía U (N x n_clusters)
        U = np.random.random((N, n_clusters))
        U = U / U.sum(axis=1, keepdims=True)

        prev_U = None
        for _ in range(max_iter):
            # Calcular centros (gateways) (Ecuación 7)
            centers = np.zeros((n_clusters, 2))
            for j in range(n_clusters):
                um = U[:, j] ** m
                if np.sum(um) > 0:
                    centers[j] = np.sum(um[:, np.newaxis] * devices, axis=0) / np.sum(um)
                else:
                    centers[j] = np.random.rand(2) * self.area_size

            # Calcular matriz de distancias con restricción de cobertura
            dist_matrix = np.zeros((N, n_clusters))
            for i in range(N):
                for j in range(n_clusters):
                    eucl_dist = np.linalg.norm(devices[i] - centers[j])
                    if self._is_covered(eucl_dist):
                        dist_matrix[i, j] = eucl_dist
                    else:
                        dist_matrix[i, j] = max_dist

            # Actualizar U (Ecuación 6)
            new_U = np.zeros((N, n_clusters))
            for i in range(N):
                row_zero = False
                for j in range(n_clusters):
                    if dist_matrix[i, j] == 0:
                        new_U[i, :] = 0.0
                        new_U[i, j] = 1.0
                        row_zero = True
                        break
                if row_zero:
                    continue
                for j in range(n_clusters):
                    denom = 0.0
                    for k in range(n_clusters):
                        denom += (dist_matrix[i, j] / dist_matrix[i, k]) ** (2.0 / (m - 1))
                    new_U[i, j] = 1.0 / denom
            
            # Verificar convergencia (Ecuación 9)
            if prev_U is not None:
                if np.linalg.norm(new_U - prev_U) < epsilon:
                    break
            prev_U = new_U
            U = new_U
        
        print(f" Calcular función objetivo Jm...")

        # Calcular función objetivo Jm (Ecuación 1) con cobertura
        Jm = 0.0
        for i in range(N):
            for j in range(n_clusters):
                eucl_dist = np.linalg.norm(devices[i] - centers[j])
                if self._is_covered(eucl_dist):
                    dist = eucl_dist
                else:
                    dist = max_dist
                Jm += (U[i, j] ** m) * (dist ** 2)

        return centers, U, Jm


    # --------------------------------------------------------------------------
    # Gap statistics for FCM (pre-processing)
    # --------------------------------------------------------------------------
    def _gap_statistics_fcm(self, devices: np.ndarray) -> int:
        print(f"  entro...")

        max_clusters = min(self.max_gateways, self.n_devices - 1)
        if max_clusters < 1:
            return 1
        
        print(f"  entro...")

        
        # Para los datos originales
        logW_orig = []
        for k in range(1, max_clusters + 1):
            _, _, Jm = self._fcm(devices, k)
            logW_orig.append(np.log(Jm))

        print(f"  entro...")


        # Para los datasets de referencia
        logW_ref = np.zeros((self.n_ref_datasets, max_clusters))
        for b in range(self.n_ref_datasets):
            ref_devices = np.random.rand(self.n_devices, 2) * self.area_size
            for k in range(1, max_clusters + 1):
                _, _, Jm = self._fcm(ref_devices, k)   # ← Usar Jm directamente
                logW_ref[b, k-1] = np.log(Jm)
        
        print(f"  entro...")

        # Cálculo de Gap, sd, s_k y selección del mejor k (igual que antes)
        logW_ref_mean = np.mean(logW_ref, axis=0)
        gap = logW_ref_mean - np.array(logW_orig)
        sd = np.std(logW_ref, axis=0, ddof=1)
        s_k = sd * np.sqrt(1 + 1.0 / self.n_ref_datasets)

        print(f"  entro...")


        best_k = 1
        for k in range(1, max_clusters):
            print(f"  Computing Gap for k={k}/{max_clusters}...")

            if gap[k-1] >= gap[k] - s_k[k]:
                best_k = k
                break
        else:
            best_k = max_clusters
        return best_k

    # --------------------------------------------------------------------------
    # Gap statistics for K-Means (processing phase)
    # --------------------------------------------------------------------------
    def _gap_statistics_kmeans(self, points: np.ndarray) -> int:
        """
        Determine optimal number of clusters for K-Means using Gap statistics.
        points: (N_points, 2) array (gateway positions from multiple scenarios).
        Returns optimal K.
        """
        max_clusters = min(self.max_gateways, len(points) - 1)
        if max_clusters < 1:
            return 1

        # Compute log(W_k) for original points
        logW_orig = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(points)
            inertia = kmeans.inertia_  # sum of squared distances (Equation 10)
            logW_orig.append(np.log(inertia))

        # Generate reference datasets
        logW_ref = np.zeros((self.n_ref_datasets, max_clusters))
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        for b in range(self.n_ref_datasets):
            ref_points = np.random.uniform(min_xy, max_xy, size=(len(points), 2))
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=b, n_init=10)
                kmeans.fit(ref_points)
                logW_ref[b, k-1] = np.log(kmeans.inertia_)

        # Gap statistic
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
    # Main DPLACE algorithm
    # --------------------------------------------------------------------------
    def place_gateways(self) -> np.ndarray:
        """
        Run DPLACE to compute final LoRaWAN gateway positions.
        Returns (K, 2) array of gateway coordinates.
        """
        print("DPLACE: Generating multiple IoT scenarios...")
        all_gateway_positions = []

        for scenario_idx in range(self.n_scenarios):
            # Generate random device positions uniformly in the area
            devices = np.random.rand(self.n_devices, 2) * self.area_size

            # Pre-processing: determine number of gateways for this scenario
            optimal_m = self._gap_statistics_fcm(devices)
            print(f"  Scenario {scenario_idx+1}: optimal gateways = {optimal_m}")

            # Compute gateway positions using FCM
            centers, U, Jm = self._fcm(devices, optimal_m)
            all_gateway_positions.extend(centers)

        all_positions = np.array(all_gateway_positions)
        print(f"Total collected gateway positions: {len(all_positions)}")

        # Processing phase: cluster all collected positions
        final_k = self._gap_statistics_kmeans(all_positions)
        print(f"Final number of gateways: {final_k}")

        kmeans = KMeans(n_clusters=final_k, random_state=0, n_init=10)
        kmeans.fit(all_positions)
        final_gateways = kmeans.cluster_centers_

        return final_gateways

    def save_gateways_to_csv(self, gateways: np.ndarray, filepath: str = "gateways_optimizados.csv"):
        """
        Save gateway positions with RSSI and path loss information to CSV.
        
        Parameters:
        -----------
        gateways : np.ndarray
            Array of gateway positions (K, 2)
        filepath : str
            Path where to save the CSV file
        """
        
        # Calculate maximum coverage distance (this applies to all gateways with same parameters)
        max_coverage_dist = self._max_coverage_distance()
        
        data = {
            'id': [],
            'x': [],
            'y': [],
            'max_coverage_distance': [],
            'rssi_at_max_distance': [],
            'path_loss_at_max_distance': []
        }
        
        for i, gateway in enumerate(gateways):
            # Calculate RSSI and path loss at max coverage distance
            rssi = self._rssi(max_coverage_dist)
            path_loss = self._okumura_hata_path_loss(max_coverage_dist)
            
            data['id'].append(i + 1)
            data['x'].append(gateway[0])
            data['y'].append(gateway[1])
            data['max_coverage_distance'].append(max_coverage_dist)
            data['rssi_at_max_distance'].append(rssi)
            data['path_loss_at_max_distance'].append(path_loss)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"Gateway data saved to: {filepath}")
        print(f"Total gateways: {len(gateways)}")
        print(f"Max coverage distance: {max_coverage_dist:.1f} meters")
        print("\nGateway summary:")
        print(df.to_string(index=False))


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create DPLACE instance with parameters from the paper (Section 4.1)
    dplace = DPLACE(
        area_size=10_000.0,          # 10 km side -> 100 km²
        n_devices=1000,
        n_scenarios=10,              # T = 10 different deployments
        max_gateways=30,
        n_ref_datasets=10,           # B = 10
        fcm_m=2.0,
        fcm_epsilon=1e-5,
        tx_power_dbm=20.0,
        antenna_gain_dbi=0.0,
        gateway_height_m=30.0,
        device_height_m=1.2,
        frequency_mhz=868.0,
        coverage_sensitivity_dbm=-137.0   # SF12 sensitivity
    )

    gateways = dplace.place_gateways()
    
    # Save gateways to CSV
    dplace.save_gateways_to_csv(gateways, "optimizacion_nodos/gateways_dplace.csv")
    
    print("\nFinal gateway positions (x, y) in meters:")
    for i, gw in enumerate(gateways):
        print(f"  GW {i+1}: ({gw[0]:.1f}, {gw[1]:.1f})")