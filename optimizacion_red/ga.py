import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict
import random
from sklearn.cluster import KMeans

class GeneticAlgorithmGatewayOptimizer:
    """
    Optimizador de posicionamiento de Gateways usando Algoritmo Genético
    """
    
    def __init__(self, 
                 nodes: np.ndarray,
                 n_gateways: int,
                 population_size: int = 150,
                 generations: int = 150,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.9,
                 elite_size: int = 10):
        """
        Args:
            nodes: Coordenadas [x, y] de los nodos IoT
            n_gateways: Número de gateways a posicionar
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Tasa de mutación
            crossover_rate: Tasa de cruce
            elite_size: Número de individuos elite a preservar
        """
        self.nodes = np.array(nodes)
        self.n_gateways = n_gateways
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # Límites del espacio de búsqueda
        self.x_min, self.x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        self.y_min, self.y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
        
        # Expandir límites un 10%
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        self.x_min -= x_range * 0.1
        self.x_max += x_range * 0.1
        self.y_min -= y_range * 0.1
        self.y_max += y_range * 0.1
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.population = None
        
    def initialize_population(self) -> List[np.ndarray]:
        """Crea población inicial de posiciones de gateways"""
        population = []
        n_seeds = max(1, self.population_size // 5)

        for _ in range(self.population_size - len(population)):
            # Generar posiciones aleatorias para las gateways
            gateways = np.zeros((self.n_gateways, 2))
            gateways[:, 0] = np.random.uniform(self.x_min, self.x_max, self.n_gateways)
            gateways[:, 1] = np.random.uniform(self.y_min, self.y_max, self.n_gateways)
            population.append(gateways)
        return population
    
    def fitness_function(self, gateways: np.ndarray) -> float:
        """
        Calcula el fitness (función de costo)
        Objetivo: Minimizar la suma de distancias cuadradas
        
        Args:
            gateways: Posiciones de las gateways [n_gateways, 2]
            
        Returns:
            fitness: Suma de distancias cuadradas (menor es mejor)
        """
        # Calcular distancias de cada nodo a cada gateway
        distances = cdist(self.nodes, gateways, metric='euclidean')
        
        # Para cada nodo, tomar la distancia mínima a cualquier gateway
        min_distances = np.min(distances, axis=1)
        
        # Suma de distancias cuadradas (WCSS)
        wcss = np.sum(min_distances ** 2)
        
        # Penalización si las gateways están muy cerca entre sí
        gw_distances = cdist(gateways, gateways, metric='euclidean')
        np.fill_diagonal(gw_distances, np.inf)
        min_gw_distance = np.min(gw_distances) if gw_distances.size > 0 else np.inf
        
        penalty = 0
        if min_gw_distance < 50:  # Penalizar si GWs están a menos de 50m
            penalty = wcss * 0.1 * (50 - min_gw_distance) / 50
        
        return wcss + penalty
    
    def evaluate_population(self, population: List[np.ndarray]) -> np.ndarray:
        """Evalúa el fitness de toda la población"""
        fitness_values = np.array([self.fitness_function(ind) for ind in population])
        return fitness_values
    
    def selection(self, population: List[np.ndarray], fitness_values: np.ndarray) -> np.ndarray:
        """
        Selección por torneo (robusta respecto al tamaño de la población)
        """
        tournament_size = min(3, len(population))
        candidates = random.sample(range(len(population)), tournament_size)
        best_idx = min(candidates, key=lambda i: fitness_values[i])
        return population[best_idx].copy()
    
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce aritmético: combina las posiciones de dos padres
        """
        # No hacer cruce si solo hay 1 gateway o si no toca cruce por probabilidad
        if self.n_gateways <= 1 or random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Selección segura de punto de cruce (solo válido si n_gateways > 1)
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        return child1, child2
    
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutación gaussiana: añade ruido a las posiciones
        """
        mutated = individual.copy()
        
        for i in range(self.n_gateways):
            if random.random() < self.mutation_rate:
                # Mutación en X
                mutated[i, 0] += np.random.normal(0, (self.x_max - self.x_min) * 0.05)
                mutated[i, 0] = np.clip(mutated[i, 0], self.x_min, self.x_max)
                
                # Mutación en Y
                mutated[i, 1] += np.random.normal(0, (self.y_max - self.y_min) * 0.05)
                mutated[i, 1] = np.clip(mutated[i, 1], self.y_min, self.y_max)
        
        return mutated
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el Algoritmo Genético
        
        Returns:
            best_gateways: Mejores posiciones de gateways encontradas
            best_fitness: Mejor fitness alcanzado
        """
        # Inicializar población
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluar población
            fitness_values = self.evaluate_population(population)
            
            # Guardar mejor solución
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.best_fitness:
                self.best_fitness = fitness_values[min_idx]
                self.best_solution = population[min_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if verbose and (generation % 20 == 0 or generation == self.generations - 1):
                print(f"Gen {generation:3d} | Best Fitness: {self.best_fitness:12.2f} | "
                      f"Avg: {np.mean(fitness_values):12.2f}")
            
            # Crear nueva población
            new_population = []
            
            # Elitismo: preservar mejores individuos
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generar resto de la población
            while len(new_population) < self.population_size:
                # Selección
                parent1 = self.selection(population, fitness_values)
                parent2 = self.selection(population, fitness_values)
                
                # Cruce
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutación
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        self.population = population
        return self.best_solution, self.best_fitness

class LoRaWISEPGAElbow:
    """
    Sistema LoRaWISEP con Método del Codo usando Algoritmo Genético
    Determina automáticamente el rango óptimo de k a evaluar
    """
    
    def __init__(self, nodes_coords: np.ndarray, auto_k_range: bool = True, 
                 max_k: int = None, min_k: int = 1, width: float = None, height: float = None):
        """
        Args:
            nodes_coords: Coordenadas [x, y] de nodos IoT
            auto_k_range: Si True, calcula automáticamente el rango de k
            max_k: Número máximo de gateways (ignorado si auto_k_range=True)
            min_k: Número mínimo de gateways a evaluar
        """
        self.nodes = np.array(nodes_coords)
        self.n_nodes = len(self.nodes)
        self.min_k = max(1, min_k)
        self.auto_k_range = auto_k_range
        self.width = 0
        self.height = 0
        self.area_km2 = 0

        # Calcular info del área primero (IMPORTANTE)
        self._calculate_area_info(width, height)
        # Calcular rango de k automáticamente o usar el especificado
        if auto_k_range:
            self.max_k = self._calculate_optimal_k_range()
            print(f"\n🔍 Rango de K determinado automáticamente: [{self.min_k}, {self.max_k}]")
        else:
            self.max_k = max_k
            print(f"\n📌 Usando rango de K manual: [{self.min_k}, {self.max_k}]")
        
        self.wcss_values = []
        self.optimal_k = None
        self.best_gateways = None
        self.ga_optimizers = {}
        
    
    def _calculate_area_info(self, width, height):
        """Calcula información del área y densidad"""
        self.x_min, self.x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        self.y_min, self.y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
        
        self.width = float(width)
        self.height = float(height)
        self.area_km2 = (self.width * self.height) / 1e6
        self.density = self.n_nodes / self.area_km2 if self.area_km2 > 0 else 0
        
        print(f"\n📊 Información del área:")
        print(f"   • Nodos totales: {self.n_nodes}")
        print(f"   • Área: {self.width:.1f}m × {self.height:.1f}m ({self.area_km2:.2f} km²)")
        print(f"   • Densidad: {self.density:.1f} nodos/km²")
    
    def _calculate_optimal_k_range(self) -> int:
        """
        Calcula automáticamente el rango óptimo de k basado en múltiples criterios
        
        Criterios considerados:
        1. Regla de Sturges (estadística)
        2. Densidad de nodos
        3. Análisis de distancias
        4. Cobertura teórica LoRaWAN
        5. Límites prácticos
        
        Returns:
            max_k: Número máximo de gateways a evaluar
        """
        print(f"\n🔬 Calculando rango óptimo de K...")
        
        # Criterio 1: Regla de Sturges (k ≈ 1 + log2(n))
        k_sturges = int(1 + 3.322 * np.log10(self.n_nodes))
        print(f"   • Regla de Sturges: k ≤ {k_sturges}")
        
        # Criterio 2: Densidad de nodos (1 GW por cada 50-150 nodos)
        k_density_min = max(1, self.n_nodes // 150)
        k_density_max = max(2, self.n_nodes // 50)
        k_density = (k_density_min + k_density_max) // 2
        print(f"   • Por densidad: k ∈ [{k_density_min}, {k_density_max}] → {k_density}")
        
        # Criterio 3: Análisis de distancias (calcular dispersión)
        centroid = np.mean(self.nodes, axis=0)
        distances_to_centroid = np.linalg.norm(self.nodes - centroid, axis=1)
        
        # Radio de cobertura típico LoRaWAN en urbano: ~2-5 km
        typical_coverage_radius = 2000  # metros
        k_coverage = max(1, int(np.ceil(self.area_km2 * 1e6 / (np.pi * typical_coverage_radius**2))))
        print(f"   • Por cobertura LoRaWAN (r≈2km): k ≥ {k_coverage}")
        
        # Criterio 4: Dispersión espacial
        # Usar percentil 90 de distancias para evitar outliers
        p90_distance = np.percentile(distances_to_centroid, 90)
        k_dispersion = max(2, int(np.ceil(p90_distance / 500)))  # 1 GW cada 500m de dispersión
        print(f"   • Por dispersión espacial (p90={p90_distance:.1f}m): k ≈ {k_dispersion}")
        
        # Criterio 5: Regla √n (heurística común en clustering)
        k_sqrt = int(np.ceil(np.sqrt(self.n_nodes)))
        print(f"   • Regla √n: k ≈ {k_sqrt}")
        
        # Criterio 6: Límites prácticos
        k_min_practical = 2  # Mínimo práctico
        k_max_practical = min(50, self.n_nodes // 5)  # Máximo práctico
        
        # Combinar criterios con pesos
        weights = {
            'sturges': 0.15,
            'density': 0.25,
            'coverage': 0.25,
            'dispersion': 0.20,
            'sqrt': 0.15
        }
        
        k_weighted = int(
            weights['sturges'] * k_sturges +
            weights['density'] * k_density +
            weights['coverage'] * k_coverage +
            weights['dispersion'] * k_dispersion +
            weights['sqrt'] * k_sqrt
        )
        
        print(f"\n   📊 K ponderado combinado: {k_weighted}")
        
        # Aplicar límites de seguridad
        # Rango: [max(criterios mínimos), min(criterios máximos)]
        k_min_suggested = max(k_min_practical, min(k_coverage, k_density_min))
        k_max_suggested = min(k_max_practical, max(k_sturges, k_density_max, k_sqrt))
        
        # Ajustar k_weighted a los límites
        k_final = np.clip(k_weighted, k_min_suggested, k_max_suggested)
        
        # Añadir margen de exploración (±30%)
        k_exploration_max = int(k_final * 1.3)
        k_exploration_max = min(k_exploration_max, k_max_practical)
        
        print(f"   ✅ Rango sugerido: [{k_min_suggested}, {k_exploration_max}]")
        print(f"   🎯 K central estimado: {k_final}")
        
        return k_exploration_max
        
    def calculate_wcss_with_ga(self, k: int, verbose: bool = False) -> float:
        """
        Calcula WCSS usando Algoritmo Genético para k gateways
        Args:
            k: Número de gateways
            verbose: Mostrar progreso del GA
            
        Returns:
            wcss: Mejor fitness (WCSS) encontrado
        """
        print(f"\n🧬 Optimizando para k={k} gateways...")
        
        # Crear optimizador GA
        ga = GeneticAlgorithmGatewayOptimizer(
            nodes=self.nodes,
            n_gateways=k,
            population_size=150,
            generations=150,
            mutation_rate=0.15,
            crossover_rate=0.9,
            elite_size=10
        )
        
        # Ejecutar optimización
        best_gateways, best_fitness = ga.optimize(verbose=verbose)
        
        # Guardar optimizador para análisis posterior
        self.ga_optimizers[k] = ga
        
        return best_fitness
    
    def calculate_wcss(self, k: int, verbose: bool = False) -> float:  
        """
        Calcula WCSS (Within-Cluster Sum of Squares) para un valor k dado
        Args:
            k: número de clusters
        Returns:
            wcss: suma de distancias cuadradas dentro de clusters
        """
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(self.nodes)
        return kmeans.inertia_
            
    def elbow_method(self, plot: bool = True, verbose_ga: bool = False) -> int:
        """
        Implementa Método del Codo con early stopping inteligente
        
        Args:
            plot: Si True, genera gráficas
            verbose_ga: Mostrar progreso detallado del GA
            early_stopping: Detener si no hay mejora significativa
            patience: Número de iteraciones sin mejora antes de detener
            
        Returns:
            optimal_k: Número óptimo de gateways
        """
        print("\n" + "="*70)
        print("     MÉTODO DEL CODO")
        print("     (Determinación automática de K)")
        print("="*70)
        
        k_range = range(self.min_k, min(self.max_k + 1, self.n_nodes))
        self.wcss_values = []
        
        # Para early stopping
        no_improvement_count = 0
        
        for i, k in enumerate(k_range):
            wcss = self.calculate_wcss_with_ga(k, verbose=verbose_ga)
            self.wcss_values.append(wcss)
            print(f"✓ k={k:2d} | WCSS = {wcss:12.2f}")
            
            # Early stopping basado en tasa de mejora
            if i > 0:
                improvement_rate = (self.wcss_values[i-1] - wcss) / self.wcss_values[i-1]
                print(f"   └─ Mejora: {improvement_rate*100:.2f}%")
                
                # Si la mejora es menor al 5%, considerar detención
                if improvement_rate < 0.05:
                    no_improvement_count += 1
                    print(f"   ⚠️  Mejora marginal detectada")
                else:
                    no_improvement_count = 0

        # Determinar k óptimo
        actual_k_range = range(self.min_k, self.min_k + len(self.wcss_values))
        self.optimal_k = self._find_elbow_point(actual_k_range, self.wcss_values)
        
        print(f"\n{'='*70}")
        print(f"🎯 K ÓPTIMO DETECTADO: {self.optimal_k} gateways")
        print(f"   └─ WCSS: {self.wcss_values[self.optimal_k - self.min_k]:.2f}")
        print(f"   └─ Evaluaciones realizadas: {len(self.wcss_values)}")
        print(f"{'='*70}\n")
        if plot:
            self._plot_elbow_analysis(actual_k_range)
        
        return self.optimal_k
    
    def _find_elbow_point(self, k_range, wcss_values, S: float = 1.0) -> int:
        """
        Algoritmo Kneedle mejorado para detectar el codo
        """
        k_list = np.array(list(k_range))
        wcss_array = np.array(wcss_values)
        
        if len(wcss_array) < 3:
            return k_list[0] if len(k_list) > 0 else 1
        
        # Normalizar a [0,1]
        k_norm = (k_list - k_list.min()) / (k_list.max() - k_list.min() + 1e-10)
        wcss_norm = (wcss_array - wcss_array.min()) / (wcss_array.max() - wcss_array.min() + 1e-10)
        
        # Calcular diferencias respecto a línea ideal
        differences = []
        for i in range(len(k_norm)):
            expected = 1 - k_norm[i]
            actual = 1 - wcss_norm[i]
            difference = actual - expected
            differences.append(difference)
        
        differences = np.array(differences)
        
        # Suavizar con media móvil
        if len(differences) > 3:
            window = min(3, len(differences))
            differences_smooth = np.convolve(differences, 
                                            np.ones(window)/window, 
                                            mode='same')
        else:
            differences_smooth = differences
        
        # Encontrar el codo
        threshold = S * np.std(differences_smooth)
        candidates = np.where(differences_smooth > threshold)[0]
        
        if len(candidates) > 0:
            elbow_idx = candidates[0]
        else:
            # Fallback: segunda derivada
            if len(differences_smooth) > 2:
                second_derivative = np.diff(differences_smooth, n=2)
                elbow_idx = np.argmax(np.abs(second_derivative)) + 1
            else:
                elbow_idx = np.argmax(differences_smooth)
        
        return k_list[elbow_idx]
    
    def _plot_elbow_analysis(self, k_range):
        """Genera visualización del Método del Codo"""        
        # Gráfica 1: Curva del Codo
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(list(k_range), self.wcss_values, 'o-', 
                linewidth=2.5, markersize=8, color='#2E86DE')
        ax1.axvline(x=self.optimal_k, color='red', linestyle='--', 
                   linewidth=2.5, label=f'K óptimo = {self.optimal_k}')
        
        # Encontrar índice del k óptimo en el rango actual
        k_list = list(k_range)
        if self.optimal_k in k_list:
            opt_idx = k_list.index(self.optimal_k)
            ax1.scatter([self.optimal_k], [self.wcss_values[opt_idx]], 
                       color='red', s=200, zorder=5, marker='*', 
                       edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('Número de Gateways (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('WCSS (Función de Costo)', fontsize=12, fontweight='bold')
        ax1.set_title('Método del Codo - Algoritmo Genético\n(K determinado automáticamente)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11, loc='best')
        
        # Gráfica 2: Tasa de reducción
        ax2 = plt.subplot(1, 2, 2)
        reduction_rates = []
        for i in range(1, len(self.wcss_values)):
            rate = (self.wcss_values[i-1] - self.wcss_values[i]) / self.wcss_values[i-1] * 100
            reduction_rates.append(rate)
        
        ax2.plot(list(k_range)[1:], reduction_rates, 'o-', 
                linewidth=2.5, markersize=8, color='#10AC84')
        ax2.axvline(x=self.optimal_k, color='red', linestyle='--', 
                   linewidth=2.5, label=f'K óptimo = {self.optimal_k}')
        ax2.axhline(y=5, color='orange', linestyle=':', 
                   linewidth=2, label='Umbral 5%', alpha=0.7)
        ax2.set_xlabel('Número de Gateways (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Reducción WCSS (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Tasa de Mejora por Gateway Adicional', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11, loc='best')
        

    def get_optimal_solution(self, plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene la solución óptima usando GA y asignaciones
        
        Returns:
            gateways: Posiciones óptimas de las gateways
            assignments: Asignación de cada nodo a su gateway más cercana
        """
        if self.optimal_k is None:
            raise ValueError("❌ Ejecuta elbow_method() primero")
        
        # Ejecutar GA para obtener la mejor solución con k óptimo
        print(f"\n🧬 Ejecutando Algoritmo Genético con k={self.optimal_k}...")
        ga = GeneticAlgorithmGatewayOptimizer(
            nodes=self.nodes,
            n_gateways=int(self.optimal_k),
            population_size=150,
            generations=150,
            mutation_rate=0.15,
            crossover_rate=0.9,
            elite_size=10
        )
        
        gateways, best_fitness = ga.optimize(verbose=True)
        
        # Guardar el optimizador GA para referencia
        self.ga_optimizers[self.optimal_k] = ga
        
        # Calcular asignaciones
        distances = cdist(self.nodes, gateways, metric='euclidean')
        assignments = np.argmin(distances, axis=1)
        
        print(f"✅ Optimización GA completada | Fitness final: {best_fitness:.2f}")
        
        # Calcular estadísticas
        stats = self._calculate_statistics(gateways, assignments)

        if plot:
            self._plot_solution(gateways, assignments)

            self._print_statistics(stats)
        
        self.best_gateways = gateways
        return gateways, assignments
    
    def _calculate_statistics(self, gateways: np.ndarray, assignments: np.ndarray) -> Dict:
        """Calcula estadísticas de la solución"""
        stats = {}
        
        unique, counts = np.unique(assignments, return_counts=True)
        stats['nodes_per_gw'] = dict(zip(unique, counts))
        
        distances = []
        for i, node in enumerate(self.nodes):
            gw = gateways[assignments[i]]
            dist = np.linalg.norm(node - gw)
            distances.append(dist)
        
        stats['distances'] = distances
        stats['avg_distance'] = np.mean(distances)
        stats['max_distance'] = np.max(distances)
        stats['min_distance'] = np.min(distances)
        stats['std_distance'] = np.std(distances)
        stats['load_balance_std'] = np.std(list(stats['nodes_per_gw'].values()))
        
        return stats
    
    def _print_statistics(self, stats: Dict):
        """Imprime estadísticas"""
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE LA SOLUCIÓN ÓPTIMA")
        print("="*70)
        print(f"   • Total de Nodos IoT: {len(self.nodes)}")
        print(f"   • Total de Gateways: {self.optimal_k}")
        print(f"\n📡 DISTRIBUCIÓN DE CARGA:")
        for gw_id, count in sorted(stats['nodes_per_gw'].items()):
            pct = count / len(self.nodes) * 100
            bar = '█' * int(pct / 2)
            print(f"   GW-{gw_id+1:2d}: {count:3d} nodos ({pct:5.1f}%) {bar}")
        
        print(f"\n📏 MÉTRICAS DE COBERTURA:")
        print(f"   • Distancia Promedio:  {stats['avg_distance']:8.2f} m")
        print(f"   • Distancia Mínima:    {stats['min_distance']:8.2f} m")
        print(f"   • Distancia Máxima:    {stats['max_distance']:8.2f} m")
        print(f"   • Desviación Estándar: {stats['std_distance']:8.2f} m")
        print(f"   • Balanceo de Carga:   {stats['load_balance_std']:8.2f} (std)")
        print("="*70 + "\n")
    
    def _plot_solution(self, gateways: np.ndarray, assignments: np.ndarray):
        """Visualiza la solución óptima"""
        plt.figure(figsize=(12, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, len(gateways)))
        
        for i in range(len(gateways)):
            cluster_nodes = self.nodes[assignments == i]
            if len(cluster_nodes) > 0:
                plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1],
                           c=[colors[i]], label=f'Gateway {i+1}',
                           alpha=0.6, s=80, edgecolors='black', linewidth=0.7)
        
        plt.scatter(gateways[:, 0], gateways[:, 1],
                   c='red', marker='*', s=800, edgecolors='black',
                   linewidth=3, label='Gateways LoRaWAN', zorder=10)
        
        for i, gw in enumerate(gateways):
            plt.annotate(f'GW-{i+1}', (gw[0], gw[1]),
                        xytext=(12, 12), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.6', 
                                 facecolor='yellow', 
                                 edgecolor='black',
                                 alpha=0.8, linewidth=2))
        
        for i, node in enumerate(self.nodes):
            gw = gateways[assignments[i]]
            plt.plot([node[0], gw[0]], [node[1], gw[1]],
                    'k-', alpha=0.08, linewidth=0.5)
        
        plt.xlabel('Coordenada X (metros)', fontsize=13, fontweight='bold')
        plt.ylabel('Coordenada Y (metros)', fontsize=13, fontweight='bold')
        plt.title(f'LoRaWISEP-GA: Posicionamiento Óptimo de {len(gateways)} Gateways\n'
                 f'(K determinado automáticamente con Algoritmo Genético)',
                 fontsize=15, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "="*80)
    print("  LoRaWISEP-GA: OPTIMIZACIÓN AUTOMÁTICA DE GATEWAYS")
    print("  (Determinación automática del número óptimo de gateways)")
    print("="*80 + "\n")
    
    # Cargar datos
    try:
        nodos = pd.read_csv("nodos_iot.csv")
        X = nodos[["X_m", "Y_m"]].values
        print(f"✅ CSV cargado: {len(X)} nodos")
    except:
        print("No se ha cargado el archivo")

    width = 1000
    height = 1000

    # ✅ MODO AUTOMÁTICO: Sin especificar max_k
    print("\n" + "="*80)
    print("MODO 1: DETERMINACIÓN AUTOMÁTICA DE K")
    print("="*80)
    print("Valor de X", len(X))

    optimizer_auto = LoRaWISEPGAElbow(
        nodes_coords=X,
        auto_k_range=True,  # ← Activa cálculo automático
        width=width,
        height=height
    )
    
    optimal_k = optimizer_auto.elbow_method(
        plot=False, 
        verbose_ga=False,
    )
    
    gateways, assignments = optimizer_auto.get_optimal_solution(plot=True)

    """
    optimizer_manual = LoRaWISEPGAElbow(
        nodes_coords=X,
        auto_k_range=False,
        max_k=30,
        min_k=1,
        width=1000,
        height=1000
    )
    optimal_k_manual = optimizer_manual.elbow_method(plot=True)
    gateways, assignments = optimizer_manual.get_optimal_solution(plot=False)
    """