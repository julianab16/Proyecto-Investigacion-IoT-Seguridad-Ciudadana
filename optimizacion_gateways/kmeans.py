import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class LoRaWISEPOptimization:
    """
    Sistema de optimización LoRaWISEP para posicionamiento de Gateways LoRaWAN
    
    Implementa:
    1. Método del Codo (Elbow Method) para determinar número óptimo de GWs
    2. K-Means para posicionamiento estratégico de las GWs
    """
    
    def __init__(self, nodes_coords, auto_k_range,  width, height, max_k = None, min_k  = None):
        """
        Inicializa el optimizador
        Args:
            nodes_coords: coordenadas [x, y] de nodos
            max_k: número máximo de clusters a evaluar
        """
        self.nodes = np.array(nodes_coords)
        self.n_nodes = len(self.nodes)
        self.wcss_values = []
        self.silhouette_scores = []
        self.auto_k_range = auto_k_range
        self.optimal_k = None
        self.gateways = None
        self.node_assignments = None
        self.kmeans_model = None
        self.width = width
        self.height = height

        self._calculate_area_info(self.width, self.height)
        if auto_k_range:
            self.max_k = self._calculate_optimal_k_range()
            self.min_k = 1
        else:
            self.max_k = max_k
            self.min_k = min_k
            print(f"\n Usando rango de K manual: [{self.min_k}, {self.max_k}]")

    def calculate_wcss(self, k):
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
    
    def _calculate_area_info(self, width, height):
        """Calcula información del área y densidad"""
        self.x_min, self.x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        self.y_min, self.y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
        
        self.width = float(width)
        self.height = float(height)
        self.area_km2 = (self.width * self.height) / 1e6
        self.density = self.n_nodes / self.area_km2 if self.area_km2 > 0 else 0
        
    
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
        print(f"\n Calculando rango óptimo de K...")
        
        # Criterio 1: Regla de Sturges (k ≈ 1 + log2(n))
        k_sturges = int(1 + 3.322 * np.log10(self.n_nodes))
        
        # Criterio 2: Densidad de nodos (1 GW por cada 50-150 nodos)
        k_density_min = max(1, self.n_nodes // 150)
        k_density_max = max(2, self.n_nodes // 50)
        k_density = (k_density_min + k_density_max) // 2
        
        # Criterio 3: Análisis de distancias (calcular dispersión)
        centroid = np.mean(self.nodes, axis=0)
        distances_to_centroid = np.linalg.norm(self.nodes - centroid, axis=1)
        
        # Radio de cobertura típico LoRaWAN en urbano: ~2-5 km
        typical_coverage_radius = 2000  # metros
        k_coverage = max(1, int(np.ceil(self.area_km2 * 1e6 / (np.pi * typical_coverage_radius**2))))
        
        # Criterio 4: Dispersión espacial
        # Usar percentil 90 de distancias para evitar outliers
        p90_distance = np.percentile(distances_to_centroid, 90)
        k_dispersion = max(2, int(np.ceil(p90_distance / 500)))  # 1 GW cada 500m de dispersión
        
        # Criterio 5: Regla √n (heurística común en clustering)
        k_sqrt = int(np.ceil(np.sqrt(self.n_nodes)))
        
        # Criterio 6: Límites prácticos
        k_min_practical = 1  # Mínimo práctico
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
        
        print(f"\n    K ponderado combinado: {k_weighted}")
        
        # Aplicar límites de seguridad
        # Rango: [max(criterios mínimos), min(criterios máximos)]
        k_min_suggested = max(k_min_practical, min(k_coverage, k_density_min))
        k_max_suggested = min(k_max_practical, max(k_sturges, k_density_max, k_sqrt))
        
        # Ajustar k_weighted a los límites
        k_final = np.clip(k_weighted, k_min_suggested, k_max_suggested)
        
        # Añadir margen de exploración (±30%)
        k_exploration_max = int(k_final * 1.3)
        k_exploration_max = min(k_exploration_max, k_max_practical)
        
        print(f"    Rango sugerido: [{k_min_suggested}, {k_exploration_max}]")
        print(f"    K central estimado: {k_final}")
        
        return k_exploration_max
    
    def elbow_method(self, plot):
        """
        Implementa el Método del Codo para determinar k óptimo
        
        Args:
            plot: si True, genera gráfica del método del codo
            
        Returns:
            optimal_k: número óptimo de gateways
        """
        max_k_int = self.max_k if isinstance(self.max_k, int) else max(self.max_k)
        k_range = range(1, min(max_k_int + 1, len(self.nodes)))
        self.wcss_values = []
        self.silhouette_scores = []
        
        for k in k_range:
            wcss = self.calculate_wcss(k)
            self.wcss_values.append(wcss)
            
            # Calcular Silhouette Score
            if k > 1:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(self.nodes)
                silhouette = silhouette_score(self.nodes, labels)
                self.silhouette_scores.append(silhouette)
            else:
                self.silhouette_scores.append(0)
            
            #print(f"k={k:2d} | WCSS: {wcss:10.2f} | Silhouette: {self.silhouette_scores[-1]:.4f}")
        
        # Determinar k óptimo usando segunda derivada
        self.optimal_k = self._find_elbow_point(k_range, self.wcss_values)
        
        print(f"\n✓ K ÓPTIMO DETECTADO: {self.optimal_k} gateways")
        print(f"  - WCSS: {self.wcss_values[self.optimal_k-1]:.2f}")
        print(f"  - Silhouette Score: {self.silhouette_scores[self.optimal_k-1]:.4f}")
        
        if plot:
            self._plot_elbow_curve(k_range)
        
        return self.optimal_k
    
    def _find_elbow_point(self, k_range, wcss_values, S=1.0):
        """
        Implementación del algoritmo Kneedle
        Paper: "Finding a 'Kneedle' in a Haystack" (Satopaa et al., 2011)
        
        Args:
            S: sensibilidad (default 1.0, valores más altos = más conservador)
        """
        k_list = np.array(list(k_range))
        wcss_array = np.array(wcss_values)
        
        # Normalizar a [0,1]
        k_norm = (k_list - k_list.min()) / (k_list.max() - k_list.min())
        wcss_norm = (wcss_array - wcss_array.min()) / (wcss_array.max() - wcss_array.min())
        
        # Calcular diferencias
        differences = []
        for i in range(len(k_norm)):
            # Distancia vertical entre punto y línea recta
            expected = 1 - k_norm[i]  # Para curva decreciente
            actual = 1 - wcss_norm[i]
            difference = actual - expected
            differences.append(difference)
        
        differences = np.array(differences)
        
        # Suavizar con media móvil
        if len(differences) > 3:
            window = 3
            differences_smooth = np.convolve(differences, 
                                            np.ones(window)/window, 
                                            mode='same')
        else:
            differences_smooth = differences
        
        # Encontrar máximo local que supera umbral
        threshold = S * np.std(differences_smooth)
        candidates = np.where(differences_smooth > threshold)[0]
        
        if len(candidates) > 0:
            # Tomar el primer candidato significativo
            elbow_idx = candidates[0]
        else:
            # Fallback: máxima diferencia
            elbow_idx = np.argmax(differences_smooth)
        
        return k_list[elbow_idx]
    
    def _plot_elbow_curve(self, k_range):
        """Genera gráfica del Método del Codo"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfica WCSS
        ax1.plot(list(k_range), self.wcss_values, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=self.optimal_k, color='r', linestyle='--', linewidth=2, 
                    label=f'K óptimo = {self.optimal_k}')
        ax1.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax1.set_title('Método del Codo - Determinación de K Óptimo', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Gráfica Silhouette Score
        ax2.plot(list(k_range)[1:], self.silhouette_scores[1:], 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=self.optimal_k, color='r', linestyle='--', linewidth=2,
                    label=f'K óptimo = {self.optimal_k}')
        ax2.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Análisis de Silhouette Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    def apply_kmeans(self, k=None, plot=True):
        """
        Aplica K-Means para posicionar las Gateways
        Args:
            k: número de gateways (usa optimal_k si no se especifica)
            plot: si True, genera visualización
            
        Returns:
            gateways: coordenadas de las gateways
            assignments: asignación de nodos a gateways
        """
        if k is None:
            if self.optimal_k is None:
                raise ValueError("Ejecuta elbow_method() primero o proporciona un valor k")
            k = self.optimal_k
        
        # Aplicar K-Means
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.node_assignments = self.kmeans_model.fit_predict(self.nodes)
        self.gateways = self.kmeans_model.cluster_centers_
        
        # Calcular estadísticas
        stats = self._calculate_statistics()
        #self._print_statistics(stats)
        
        if plot:
            self._plot_clusters()
        
        return self.gateways, self.node_assignments
    
    def _calculate_statistics(self):
        """Calcula estadísticas del clustering"""
        stats = {}
        
        # Nodos por cluster
        unique, counts = np.unique(self.node_assignments, return_counts=True)
        stats['nodes_per_cluster'] = dict(zip(unique, counts))
        
        # Distancias
        distances = []
        for i, node in enumerate(self.nodes):
            gw = self.gateways[self.node_assignments[i]]
            dist = np.linalg.norm(node - gw)
            distances.append(dist)
        
        stats['avg_distance'] = np.mean(distances)
        stats['max_distance'] = np.max(distances)
        stats['min_distance'] = np.min(distances)
        stats['std_distance'] = np.std(distances)
        
        # Cobertura
        stats['coverage_radius'] = stats['max_distance']
        
        return stats
    
    def _print_statistics(self, stats):
        """Imprime estadísticas del clustering"""
        print(f"\n📊 ESTADÍSTICAS DE OPTIMIZACIÓN:")
        print(f"   • Total de Nodos: {len(self.nodes)}")
        print(f"   • Total de Gateways: {len(self.gateways)}")
        print(f"\n📡 DISTRIBUCIÓN DE NODOS POR GATEWAY:")
        for cluster_id, count in stats['nodes_per_cluster'].items():
            print(f"   GW-{cluster_id + 1}: {count} nodos ({count/len(self.nodes)*100:.1f}%)")
        
        print(f"\n📏 MÉTRICAS DE DISTANCIA:")
        print(f"   • Distancia Promedio: {stats['avg_distance']:.2f} m")
        print(f"   • Distancia Mínima: {stats['min_distance']:.2f} m")
        print(f"   • Distancia Máxima: {stats['max_distance']:.2f} m")
        print(f"   • Desv. Estándar: {stats['std_distance']:.2f} m")
        print(f"   • Radio de Cobertura Requerido: {stats['coverage_radius']:.2f} m")
    
    def _plot_clusters(self):
        """Visualiza los clusters y gateways"""
        plt.figure(figsize=(12, 10))
        
        # Colores para clusters
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.gateways)))
        
        # Plotear nodos por cluster
        for i in range(len(self.gateways)):
            cluster_nodes = self.nodes[self.node_assignments == i]
            plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], 
                       c=[colors[i]], label=f'Cluster {i+1}', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plotear gateways
        plt.scatter(self.gateways[:, 0], self.gateways[:, 1], 
                   c='red', marker='*', s=500, edgecolors='black', 
                   linewidth=2, label='Gateways', zorder=5)
        
        # Añadir etiquetas a gateways
        for i, gw in enumerate(self.gateways):
            plt.annotate(f'GW-{i+1}', (gw[0], gw[1]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Dibujar líneas de conexión
        for i, node in enumerate(self.nodes):
            gw = self.gateways[self.node_assignments[i]]
            plt.plot([node[0], gw[0]], [node[1], gw[1]], 
                    'k-', alpha=0.1, linewidth=0.5)
        
        plt.xlabel('Coordenada X (m)', fontsize=12, fontweight='bold')
        plt.ylabel('Coordenada Y (m)', fontsize=12, fontweight='bold')
        plt.title(f'LoRaWISEP - Posicionamiento Óptimo de {len(self.gateways)} Gateways', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_gateway_positions(self):
        """Retorna las posiciones de las gateways"""
        if self.gateways is None:
            raise ValueError("Ejecuta apply_kmeans() primero")
        return self.gateways
    
    def get_node_assignments(self):
        """Retorna las asignaciones de nodos a gateways"""
        if self.node_assignments is None:
            raise ValueError("Ejecuta apply_kmeans() primero")
        return self.node_assignments


if __name__ == "__main__":

    # 1. CARGAR DATOS DESDE CSV
    import pandas as pd
    
    # Opción A: Cargar desde tu CSV
    nodos = pd.read_csv("nodos_iot.csv")
    X = nodos[["X_m", "Y_m"]].values
    print(f" Total de nodos encontrados: {len(X)}")

    nodes = X
    n_nodes = len(nodes)

    width = 10000
    height = 10000
    
    
    # Crear instancia del optimizador
    optimizer = LoRaWISEPOptimization(nodes, width, height, True)
    
    # Determinar número óptimo de gateways con Método del Codo
    optimizer.elbow_method(plot=True)
    
    # Posicionar gateways usando K-Means
    gateways, assignments = optimizer.apply_kmeans(plot=True)
    