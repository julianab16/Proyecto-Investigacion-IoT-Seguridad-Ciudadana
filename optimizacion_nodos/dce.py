import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class DPLACE:
    def __init__(self, area_size=100, n_devices=1000, gateway_height=30, device_height=1.2, 
                 coverage_radius=2, frequency=868, bandwidth=125, sf_range=(7,12),
                 message_size=20, app_period=600, sim_duration=1200, n_simulations=33):
        """
        Inicializa el modelo DPLACE con los parámetros del escenario LoRaWAN
        """
        self.area_size = area_size  # km²
        self.n_devices = n_devices
        self.gateway_height = gateway_height  # m
        self.device_height = device_height  # m
        self.coverage_radius = coverage_radius  # km
        self.frequency = frequency  # MHz
        self.bandwidth = bandwidth  # kHz
        self.sf_range = sf_range
        self.message_size = message_size  # bytes
        self.app_period = app_period  # segundos
        self.sim_duration = sim_duration  # segundos
        self.n_simulations = n_simulations
        
        # Sensibilidad del receptor por SF (dBm)
        self.sensitivity = {
            7: -125, 8: -127, 9: -129, 10: -131, 11: -133, 12: -137
        }
        
        # Parámetros del algoritmo
        self.m = 2  # ponderación fuzzy
        self.epsilon = 1e-4  # tolerancia de convergencia
        
    def generate_devices(self, seed=None):
        """Genera dispositivos aleatorios en el área de simulación"""
        if seed:
            np.random.seed(seed)
        # Coordenadas de dispositivos (km)
        devices = np.random.uniform(0, np.sqrt(self.area_size), (self.n_devices, 2))
        # Asignar SF aleatorio a cada dispositivo
        sf_values = np.random.choice(range(self.sf_range[0], self.sf_range[1]+1), self.n_devices)
        return devices, sf_values
    
    def calculate_distance(self, devices, gateways):
        """Calcula la matriz de distancias entre dispositivos y gateways"""
        return cdist(devices, gateways)
    
    def check_power_constraint(self, distances, sf_values, gateways):
        """Verifica restricción de potencia según sensibilidad del SF"""
        # Radio de cobertura del gateway = 2 km (del artículo)
        coverage_radius_km = self.coverage_radius  # 2 km
        
        max_distance_by_sf = {
            7: coverage_radius_km * 1.0,   # SF7: alcance normal
            8: coverage_radius_km * 1.05,
            9: coverage_radius_km * 1.10,
            10: coverage_radius_km * 1.15,
            11: coverage_radius_km * 1.18,
            12: coverage_radius_km * 1.20  # SF12: máximo alcance
        }
        
        distances_modified = distances.copy()
        for i, sf in enumerate(sf_values):
            max_dist = max_distance_by_sf.get(sf, coverage_radius_km)
            for j in range(distances.shape[1]):
                if distances[i, j] > max_dist:
                    distances_modified[i, j] = 1e6  # Penalización grande
        return distances_modified
    
    def fuzzy_cmeans_objective(self, devices, gateways, membership, sf_values):
        """Calcula J_mj: función objetivo de Fuzzy C-Means adaptada"""
        distances = self.calculate_distance(devices, gateways)
        distances = self.check_power_constraint(distances, sf_values, gateways)
        
        J = 0
        for j in range(len(gateways)):
            for i in range(len(devices)):
                J += (membership[i, j] ** self.m) * distances[i, j]
        return J
    
    def calculate_membership(self, devices, gateways, sf_values):
        """Calcula matriz de membresía μ_ij"""
        distances = self.calculate_distance(devices, gateways)
        distances = self.check_power_constraint(distances, sf_values, gateways)
        
        n_devices, n_gateways = distances.shape
        membership = np.zeros((n_devices, n_gateways))
        
        for i in range(n_devices):
            for j in range(n_gateways):
                if distances[i, j] == 0:
                    membership[i, j] = 1
                else:
                    sum_terms = 0
                    for k in range(n_gateways):
                        if distances[i, k] > 0:
                            sum_terms += (distances[i, j] / distances[i, k]) ** (2/(self.m-1))
                    membership[i, j] = 1 / sum_terms if sum_terms > 0 else 0
        return membership
    
    def update_centers(self, devices, membership):
        """Actualiza posiciones de gateways (c_j)"""
        membership_m = membership ** self.m
        sum_membership = membership_m.sum(axis=0, keepdims=True)
        sum_membership = np.where(sum_membership == 0, 1, sum_membership)
        new_centers = (membership_m.T @ devices) / sum_membership.T
        return new_centers
    
    def gap_statistics(self, devices, sf_values, max_gateways=30, B=10):
        """Calcula Gap Statistics para determinar número óptimo de gateways"""
        gaps = []
        std_errors = []
        
        # Calcular para cada número posible de gateways
        for j in range(1, min(max_gateways, len(devices)//5) + 1):
            # Log(J_mj) para datos reales
            gateways = self.initialize_gateways(devices, j)
            membership = self.calculate_membership(devices, gateways, sf_values)
            log_J_real = np.log(self.fuzzy_cmeans_objective(devices, gateways, membership, sf_values) + 1e-10)
            
            # Log(J*_mj,b) para B escenarios aleatorios
            log_J_random = []
            for b in range(B):
                random_devices = np.random.uniform(0, np.sqrt(self.area_size), devices.shape)
                random_sf = np.random.choice(range(self.sf_range[0], self.sf_range[1]+1), len(devices))
                gateways_random = self.initialize_gateways(random_devices, j)
                membership_random = self.calculate_membership(random_devices, gateways_random, random_sf)
                log_J_rand = np.log(self.fuzzy_cmeans_objective(random_devices, gateways_random, 
                                                                membership_random, random_sf) + 1e-10)
                log_J_random.append(log_J_rand)
            
            # Gap(j) = (1/B)*sum(log(J*)) - log(J)
            gap_j = np.mean(log_J_random) - log_J_real
            gaps.append(gap_j)
            
            # Error estándar con corrección s(j) = sd(j) * sqrt(1 + 1/B)
            std_j = np.std(log_J_random)
            s_j = std_j * np.sqrt(1 + 1/B)
            std_errors.append(s_j)
            
            print(f"    j={j}: Gap={gap_j:.4f}, s(j)={s_j:.4f}")
        
        # CRITERIO CORREGIDO (según artículo, ecuación 5)
        # Buscar el primer j donde Gap(j) >= Gap(j+1) - s(j+1)
        optimal_j = len(gaps)  # Valor por defecto: el máximo
        
        for j in range(len(gaps) - 1):
            if gaps[j] >= gaps[j + 1] - std_errors[j + 1]:
                optimal_j = j + 1
                break
        
        print(f"    → Número óptimo de gateways: {optimal_j}")

        return optimal_j
    
    def initialize_gateways(self, devices, n_gateways):
        """Inicializa gateways usando puntos aleatorios de dispositivos"""
        indices = np.random.choice(len(devices), n_gateways, replace=False)
        return devices[indices].copy()
    
    def fuzzy_cmeans(self, devices, sf_values, n_gateways, max_iter=100):
        """Algoritmo Fuzzy C-Means para posicionamiento inicial de gateways"""
        gateways = self.initialize_gateways(devices, n_gateways)
        
        for iteration in range(max_iter):
            membership = self.calculate_membership(devices, gateways, sf_values)
            new_gateways = self.update_centers(devices, membership)
            
            # Verificar convergencia
            if np.linalg.norm(new_gateways - gateways) < self.epsilon:
                break
            
            gateways = new_gateways
        
        return gateways, membership
    
    def phase1_preprocessing(self, devices, sf_values):
        """Fase 1: Pre-procesamiento - Determinar número y posiciones iniciales"""
        print("Fase 1: Pre-procesamiento")
        
        # Paso 1: Calcular número óptimo de gateways con Gap Statistics
        print("  - Calculando número óptimo de gateways (Gap Statistics)...")
        optimal_gateways = self.gap_statistics(devices, sf_values)
        print(f"    Número óptimo de gateways: {optimal_gateways}")
        
        # Paso 2: Posicionamiento inicial con Fuzzy C-Means
        print("  - Calculando posiciones iniciales con Fuzzy C-Means...")
        gateways, membership = self.fuzzy_cmeans(devices, sf_values, optimal_gateways)
        
        return gateways, membership, optimal_gateways
    
    def kmeans_clustering(self, gateway_positions, max_clusters=15, B=10):
        """
        Aplica K-Means para centralizar posiciones de gateways
        Retorna: centros optimizados y número de clusters K
        """
        from sklearn.cluster import KMeans
        
        # Calcular J_k para diferentes K
        J_k_values = []
        
        for k in range(1, min(max_clusters, len(gateway_positions))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(gateway_positions)
            J_k_values.append(kmeans.inertia_)  # Suma de cuadrados intra-cluster
        
        # Gap Statistics para K-Means
        gaps = []
        std_errors = []
        
        for k in range(1, min(max_clusters, len(gateway_positions))):
            # Log(J_k) para datos reales
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(gateway_positions)
            log_J_real = np.log(kmeans.inertia_ + 1e-10)
            
            # Datos aleatorios uniformes
            log_J_random = []
            for b in range(B):
                random_positions = np.random.uniform(0, np.sqrt(self.area_size), gateway_positions.shape)
                kmeans_rand = KMeans(n_clusters=k, random_state=b, n_init=10)
                kmeans_rand.fit(random_positions)
                log_J_rand = np.log(kmeans_rand.inertia_ + 1e-10)
                log_J_random.append(log_J_rand)
            
            gap_k = np.mean(log_J_random) - log_J_real
            gaps.append(gap_k)
            std_errors.append(np.std(log_J_random) * np.sqrt(1 + 1/B))
        
        # Seleccionar K óptimo
        optimal_K = 1
        for k in range(1, len(gaps)):
            if gaps[k] <= gaps[k-1] + std_errors[k-1]:
                optimal_K = k + 1
                break
        else:
            optimal_K = len(gaps)
        
        # Aplicar K-Means con K óptimo
        final_kmeans = KMeans(n_clusters=optimal_K, random_state=42, n_init=10)
        final_kmeans.fit(gateway_positions)
        
        return final_kmeans.cluster_centers_, optimal_K
    
    def phase2_processing(self, all_gateway_positions):
        """Fase 2: Procesamiento - Centralización con Gap Statistics + K-Means"""
        print("\nFase 2: Procesamiento")
        print("  - Centralizando posiciones de gateways con K-Means...")
        
        # Paso 3 y 4: Aplicar Gap Statistics + K-Means
        final_centers, optimal_K = self.kmeans_clustering(all_gateway_positions)
        
        print(f"    Número óptimo de centros finales (K): {optimal_K}")
        print(f"    Posiciones finales de gateways:\n{final_centers}")
        
        return final_centers, optimal_K
    
    def simulate_poisson_arrival(self, lambda_rate, sim_time):
        """Simula llegada de dispositivos siguiendo patrón de Poisson"""
        n_expected = int(lambda_rate * sim_time)
        # Generar tiempos de llegada Poisson
        inter_arrival = np.random.exponential(1/lambda_rate, n_expected)
        arrival_times = np.cumsum(inter_arrival)
        arrival_times = arrival_times[arrival_times <= sim_time]
        return arrival_times
    
    def run(self):
        """Ejecuta el algoritmo DPLACE completo con múltiples simulaciones"""
        print("="*60)
        print("DPLACE - LoRaWAN Gateway Placement Model")
        print("="*60)
        print(f"Parámetros:")
        print(f"  - Dispositivos: {self.n_devices}")
        print(f"  - Área: {self.area_size} km²")
        print(f"  - Radio cobertura: {self.coverage_radius} km")
        print(f"  - Simulaciones: {self.n_simulations}")
        print("="*60)
        
        all_gateway_positions = []
        
        # Múltiples ejecuciones para manejar dinamismo
        for sim in range(self.n_simulations):
            print(f"\n--- Simulación {sim+1}/{self.n_simulations} ---")
            
            # Generar escenario dinámico con patrón Poisson
            lambda_rate = self.n_devices / self.sim_duration
            arrival_times = self.simulate_poisson_arrival(lambda_rate, self.sim_duration)
            
            # Generar dispositivos activos en este instante
            n_active = min(len(arrival_times), self.n_devices)
            devices, sf_values = self.generate_devices(seed=sim)
            devices_active = devices[:n_active]
            sf_active = sf_values[:n_active]
            
            # Fase 1 para esta simulación
            gateways, _, _ = self.phase1_preprocessing(devices_active, sf_active)
            all_gateway_positions.append(gateways)
        
        # Consolidar todas las posiciones
        all_positions = np.vstack(all_gateway_positions)
        
        # Fase 2: Centralización
        final_gateways, optimal_K = self.phase2_processing(all_positions)
        
        # Métricas de calidad de servicio (QoS)
        coverage_estimate = len(final_gateways) * (np.pi * self.coverage_radius**2) / self.area_size
        coverage_estimate = min(1.0, coverage_estimate)
        
        print("\n" + "="*60)
        print("RESULTADOS FINALES")
        print("="*60)
        print(f"Gateways finales: {optimal_K}")
        print(f"Cobertura estimada: {coverage_estimate*100:.1f}%")
        print(f"Redundancia promedio: {len(all_gateway_positions)*self.n_devices/optimal_K:.1f} dispositivos por gateway")
        
        return {
            'gateways': final_gateways,
            'n_gateways': optimal_K,
            'coverage_estimate': coverage_estimate,
            'all_positions': all_positions
        }

# Ejecutar algoritmo
if __name__ == "__main__":
    # Inicializar DPLACE con parámetros especificados
    dplace = DPLACE(
        area_size=100,           # 100 km²
        n_devices=1000,          # 1000 dispositivos
        gateway_height=30,       # 30 metros
        device_height=1.2,       # 1.2 metros
        coverage_radius=2,       # 2 km
        frequency=868,           # 868 MHz
        bandwidth=125,           # 125 kHz
        sf_range=(7, 12),        # SF7 a SF12
        message_size=20,         # 20 bytes
        app_period=600,          # 600 segundos
        sim_duration=1200,       # 1200 segundos
        n_simulations=33         # 33 ejecuciones (95% confianza)
    )
    
    # Ejecutar DPLACE
    results = dplace.run()
    
    # Mostrar posiciones finales
    print("\nCoordenadas finales de gateways (km):")
    for i, gw in enumerate(results['gateways']):
        print(f"  Gateway {i+1}: ({gw[0]:.2f}, {gw[1]:.2f})")