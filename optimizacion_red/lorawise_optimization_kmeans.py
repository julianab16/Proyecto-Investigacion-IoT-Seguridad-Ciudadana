import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


class LoRaWISEPOptimization:
    """
    Sistema de optimización LoRaWISEP para posicionamiento de Gateways LoRaWAN
    
    Implementa:
    1. Método del Codo (Elbow Method) para determinar número óptimo de GWs
    2. K-Means para posicionamiento estratégico de las GWs
    """
    
    def __init__(self, nodes_coords, max_k=10, width=None, height=None):
        """
        Inicializa el optimizador
        Args:
            nodes_coords: coordenadas [x, y] de nodos
            max_k: número máximo de clusters a evaluar
            width: ancho del área en metros (opcional)
            height: alto del área en metros (opcional)
        """
        self.nodes = np.array(nodes_coords)
        self.max_k = max_k
        self.width = width
        self.height = height
        self.wcss_values = []
        self.silhouette_scores = []
        self.optimal_k = None
        self.gateways = None
        self.node_assignments = None
        self.kmeans_model = None
        
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
    
    def elbow_method(self, plot=True):
        """
        Implementa el Método del Codo para determinar k óptimo
        
        Args:
            plot: si True, genera gráfica del método del codo
            
        Returns:
            optimal_k: número óptimo de gateways
        """
        print("\n" + "="*70)
        print("     MÉTODO DEL CODO - DETERMINACIÓN DE K ÓPTIMO")
        print("="*70 + "\n")

        k_range = range(1, min(self.max_k + 1, len(self.nodes)))
        self.wcss_values = []
        self.silhouette_scores = []
        
        print("Evaluando diferentes valores de k...")
        print("-" * 70)
        
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
            
            print(f"k={k:2d} | WCSS: {wcss:12.2f} | Silhouette: {self.silhouette_scores[-1]:.4f}")
        
        # Determinar k óptimo usando algoritmo Kneedle
        self.optimal_k = self._find_elbow_point(k_range, self.wcss_values)
        
        print("-" * 70)
        print(f"\n🎯 K ÓPTIMO DETECTADO: {self.optimal_k} gateways")
        print(f"   • WCSS: {self.wcss_values[self.optimal_k-1]:.2f}")
        print(f"   • Silhouette Score: {self.silhouette_scores[self.optimal_k-1]:.4f}")
        print("="*70 + "\n")
        
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
        k_norm = (k_list - k_list.min()) / (k_list.max() - k_list.min() + 1e-10)
        wcss_norm = (wcss_array - wcss_array.min()) / (wcss_array.max() - wcss_array.min() + 1e-10)
        
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfica WCSS
        ax1.plot(list(k_range), self.wcss_values, 'o-', linewidth=2.5, 
                markersize=8, color='#2E86DE')
        ax1.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2.5, 
                    label=f'K óptimo = {self.optimal_k}')
        ax1.scatter([self.optimal_k], [self.wcss_values[self.optimal_k-1]], 
                   color='red', s=200, zorder=5, marker='*', 
                   edgecolors='black', linewidth=2)
        ax1.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax1.set_title('Método del Codo - Determinación de K Óptimo', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11)
        
        # Gráfica Silhouette Score
        ax2.plot(list(k_range)[1:], self.silhouette_scores[1:], 'o-', linewidth=2.5, 
                markersize=8, color='#10AC84')
        ax2.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2.5,
                    label=f'K óptimo = {self.optimal_k}')
        if self.optimal_k > 1:
            ax2.scatter([self.optimal_k], [self.silhouette_scores[self.optimal_k-1]], 
                       color='red', s=200, zorder=5, marker='*', 
                       edgecolors='black', linewidth=2)
        ax2.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Análisis de Silhouette Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
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
        
        print("\n" + "="*70)
        print(f"     APLICANDO K-MEANS CON {k} GATEWAYS")
        print("="*70 + "\n")
        
        # Aplicar K-Means
        self.kmeans_model = KMeans(n_clusters=k, init='k-means++', 
                                   random_state=42, n_init=10)
        self.node_assignments = self.kmeans_model.fit_predict(self.nodes)
        self.gateways = self.kmeans_model.cluster_centers_
        
        # Calcular estadísticas
        stats = self._calculate_statistics()
        self._print_statistics(stats)
        
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
        
        stats['distances'] = distances
        stats['avg_distance'] = np.mean(distances)
        stats['max_distance'] = np.max(distances)
        stats['min_distance'] = np.min(distances)
        stats['std_distance'] = np.std(distances)
        
        # Cobertura
        stats['coverage_radius'] = stats['max_distance']
        
        # Balanceo de carga
        stats['load_balance_std'] = np.std(list(stats['nodes_per_cluster'].values()))
        
        return stats
    
    def _print_statistics(self, stats):
        """Imprime estadísticas del clustering"""
        print("📊 ESTADÍSTICAS DE OPTIMIZACIÓN:")
        print(f"   • Total de Nodos: {len(self.nodes)}")
        print(f"   • Total de Gateways: {len(self.gateways)}")
        print(f"   • Densidad: {len(self.nodes)/len(self.gateways):.1f} nodos/GW")
        
        print(f"\n📡 DISTRIBUCIÓN DE NODOS POR GATEWAY:")
        for cluster_id, count in sorted(stats['nodes_per_cluster'].items()):
            pct = count / len(self.nodes) * 100
            bar = '█' * int(pct / 2)
            print(f"   GW-{cluster_id+1:2d}: {count:4d} nodos ({pct:5.1f}%) {bar}")
        
        print(f"\n📏 MÉTRICAS DE COBERTURA:")
        print(f"   • Distancia Promedio:  {stats['avg_distance']:8.2f} m")
        print(f"   • Distancia Mínima:    {stats['min_distance']:8.2f} m")
        print(f"   • Distancia Máxima:    {stats['max_distance']:8.2f} m")
        print(f"   • Desviación Estándar: {stats['std_distance']:8.2f} m")
        print(f"   • Radio de Cobertura:  {stats['coverage_radius']:8.2f} m")
        print(f"   • Balanceo de Carga:   {stats['load_balance_std']:8.2f} (std)")
        print("="*70 + "\n")
    
    def _plot_clusters(self):
        """Visualiza los clusters y gateways"""
        plt.figure(figsize=(14, 11))
        
        # Colores para clusters
        n_clusters = len(self.gateways)
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        # Plotear nodos por cluster
        for i in range(len(self.gateways)):
            cluster_nodes = self.nodes[self.node_assignments == i]
            plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], 
                       c=[colors[i]], label=f'Cluster {i+1}', 
                       alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # Plotear gateways
        plt.scatter(self.gateways[:, 0], self.gateways[:, 1], 
                   c='red', marker='*', s=800, edgecolors='black', 
                   linewidth=3, label='Gateways LoRaWAN', zorder=10)
        
        # Añadir etiquetas a gateways
        for i, gw in enumerate(self.gateways):
            plt.annotate(f'GW-{i+1}', (gw[0], gw[1]), 
                        xytext=(12, 12), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.6', 
                                 facecolor='yellow', 
                                 edgecolor='black',
                                 alpha=0.8, linewidth=2))
        
        # Dibujar líneas de conexión
        for i, node in enumerate(self.nodes):
            gw = self.gateways[self.node_assignments[i]]
            plt.plot([node[0], gw[0]], [node[1], gw[1]], 
                    'k-', alpha=0.08, linewidth=0.5)
        
        plt.xlabel('Coordenada X (metros)', fontsize=13, fontweight='bold')
        plt.ylabel('Coordenada Y (metros)', fontsize=13, fontweight='bold')
        plt.title(f'LoRaWISEP - Posicionamiento Óptimo de {len(self.gateways)} Gateways\n'
                 f'Optimizado con K-Means', 
                 fontsize=15, fontweight='bold', pad=20)
        
        # Ajustar leyenda si hay muchos clusters
        if n_clusters <= 12:
            plt.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
        else:
            # Mostrar solo leyenda de gateways si hay muchos clusters
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[-1]], [labels[-1]], loc='best', fontsize=10)
        
        plt.grid(True, alpha=0.3, linestyle='--')
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


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("     LoRaWISEP - SISTEMA DE OPTIMIZACIÓN DE GATEWAYS CON K-MEANS")
    print("="*80 + "\n")
    
    # PARÁMETROS DE ENTRADA
    N = 1000  # Número de nodos
    width = 10000  # Ancho del área (metros)
    height = 10000  # Alto del área (metros)
    
    # Cargar nodos desde CSV
    try:
        nodos = pd.read_csv("nodos_iot.csv")
        X = nodos[["X_m", "Y_m"]].values
        print(f"✓ Datos cargados desde CSV correctamente")
        
        # Validar datos
        if len(X) != N:
            print(f"⚠️  ADVERTENCIA: Se esperaban {N} nodos, pero se encontraron {len(X)}")
            N = len(X)
        
        # Verificar que los nodos estén dentro del área especificada
        out_of_bounds = np.sum((X[:, 0] < 0) | (X[:, 0] > width) | 
                               (X[:, 1] < 0) | (X[:, 1] > height))
        if out_of_bounds > 0:
            print(f"⚠️  {out_of_bounds} nodos están fuera del área {width}x{height}m")
        
    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo CSV")
        print("   Generando datos sintéticos para demostración...")
        
        # Generar datos sintéticos distribuidos uniformemente
        np.random.seed(42)
        X = np.random.uniform(0, 1, (N, 2))
        X[:, 0] *= width
        X[:, 1] *= height
    
    nodes = X
    
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"   • Total de nodos IoT: {len(nodes)}")
    print(f"   • Área de cobertura: {width}m × {height}m = {width*height/1e6:.2f} km²")
    print(f"   • Densidad: {len(nodes)/(width*height/1e6):.1f} nodos/km²")
    print(f"   • Rango X: [{nodes[:, 0].min():.1f}, {nodes[:, 0].max():.1f}] m")
    print(f"   • Rango Y: [{nodes[:, 1].min():.1f}, {nodes[:, 1].max():.1f}] m")
    
    # Determinar max_k basado en el número de nodos
    max_k = min(30, max(10, N // 75))
    print(f"   • Máximo k a evaluar: {max_k}")
    
    # Crear instancia del optimizador
    optimizer = LoRaWISEPOptimization(nodes, max_k=max_k)
    
    # PASO 1: Determinar número óptimo de gateways con Método del Codo
    optimal_k = optimizer.elbow_method(plot=False)
    
    # PASO 2: Posicionar gateways usando K-Means
    gateways, assignments = optimizer.apply_kmeans(plot=False)
    
    # PASO 3: Resumen final
    print("\n" + "="*80)
    print("✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80)
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   • Número óptimo de Gateways: {optimal_k}")
    print(f"   • Nodos por Gateway (promedio): {N/optimal_k:.1f}")
    print(f"   • WCSS final: {optimizer.wcss_values[optimal_k-1]:.2f}")
    print(f"\n💡 RECOMENDACIONES:")
    
    max_dist = optimizer._calculate_statistics()['max_distance']
    if max_dist < 1000:
        sf_rec = "SF7-SF9"
    elif max_dist < 3000:
        sf_rec = "SF9-SF11"
    else:
        sf_rec = "SF11-SF12"
    
    print(f"   • Radio de cobertura requerido: ~{max_dist:.0f}m")
    print(f"   • Spreading Factor sugerido: {sf_rec}")
    print(f"   • Frecuencia: 915 MHz (Región 2 - América)")
    print(f"\n💾 Las posiciones de las gateways están disponibles en:")
    print(f"   optimizer.gateways")
    print("="*80 + "\n")