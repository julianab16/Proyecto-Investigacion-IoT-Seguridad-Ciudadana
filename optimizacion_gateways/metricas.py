import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LoRaWANMetricsEvaluator:
    """
    Evaluador de métricas de rendimiento para redes LoRaWAN
    
    Métricas implementadas:
    - PDR (Packet Delivery Ratio): Tasa de entrega de paquetes
    - Rango de comunicación: Distancia entre nodos y gateways
    - RSSI (Received Signal Strength Indicator): Fuerza de la señal
    - SNR (Signal-to-Noise Ratio): Calidad de la señal
    """
    
    def __init__(self, nodes, gateways, assignments, frequency=915):
        """
        Inicializa el evaluador de métricas
        
        Args:
            nodes: array (n_nodes, 2) con coordenadas de nodos
            gateways: array (n_gw, 2) con coordenadas de gateways
            assignments: array (n_nodes,) con asignación de nodos a GWs
            frequency: frecuencia en MHz (915 para América, 868 para Europa)
        """
        self.nodes = np.array(nodes)
        self.gateways = np.array(gateways)
        self.assignments = np.array(assignments)
        self.frequency = frequency


        # Parámetros del modelo de propagación
        self.path_loss_exponent = 2.7  # Entorno urbano típico
        self.reference_distance = 1.0  # metros
        self.reference_loss = 20 * np.log10(4 * np.pi * self.reference_distance * 
                                            self.frequency * 1e6 / 3e8)
        
        # Parámetros LoRaWAN
        self.tx_power = 14  # dBm (potencia de transmisión típica)
        self.noise_floor = -120  # dBm (piso de ruido típico)
        self.sensitivity = -137  # dBm (sensibilidad del receptor para SF12)
        self.max_range = 15000  # metros (rango máximo teórico)
        
        # Resultados
        self.metrics = {}
        
    def calculate_distance(self, node_idx):
        """Calcula distancia entre nodo y su gateway asignado"""
        node = self.nodes[node_idx]
        gw = self.gateways[self.assignments[node_idx]]
        return np.linalg.norm(node - gw)
    
    def calculate_path_loss(self, distance):
        """
        Calcula pérdida por trayectoria usando modelo Log-Distance
        
        PL(d) = PL(d0) + 10*n*log10(d/d0) + Xσ
        
        Args:
            distance: distancia en metros
            
        Returns:
            path_loss: pérdida en dB
        """
        if distance < self.reference_distance:
            distance = self.reference_distance
            
        # Pérdida por trayectoria
        path_loss = (self.reference_loss + 
                    10 * self.path_loss_exponent * 
                    np.log10(distance / self.reference_distance))
        
        # Agregar shadowing (desvanecimiento por sombra) - distribución normal
        shadowing = np.random.normal(0, 6)  # σ = 6 dB típico en urbano
        
        return path_loss + shadowing
    
    def calculate_rssi(self, distance):
        """
        Calcula RSSI (Received Signal Strength Indicator)
        
        RSSI = Potencia_TX - Pérdida_Trayectoria
        
        Args:
            distance: distancia en metros
            
        Returns:
            rssi: RSSI en dBm
        """
        path_loss = self.calculate_path_loss(distance)
        rssi = self.tx_power - path_loss
        return rssi
    
    def calculate_snr(self, rssi):
        """
        Calcula SNR (Signal-to-Noise Ratio)
        
        SNR = RSSI - Piso_Ruido
        
        Args:
            rssi: RSSI en dBm
            
        Returns:
            snr: SNR en dB
        """
        # Agregar ruido adicional aleatorio
        noise = self.noise_floor + np.random.uniform(-3, 3)
        snr = rssi - noise
        return snr
    
    def calculate_pdr(self, rssi, snr):
        """
        Calcula PDR (Packet Delivery Ratio) basado en RSSI y SNR
        
        PDR depende de:
        - RSSI > Sensibilidad del receptor
        - SNR > Umbral mínimo
        - Modelo probabilístico de pérdida de paquetes
        
        Args:
            rssi: RSSI en dBm
            snr: SNR en dB
            
        Returns:
            pdr: PDR en porcentaje (0-100)
        """
        # Verificar si la señal es detectable
        if rssi < self.sensitivity:
            return 0.0
        
        # PDR basado en SNR usando función sigmoide
        snr_threshold = -7.5  # dB (umbral típico para SF12)
        snr_margin = snr - snr_threshold
        
        # Función logística para modelar PDR
        pdr = 100 / (1 + np.exp(-0.5 * snr_margin))
        
        # Penalizar por RSSI bajo (cercano a sensibilidad)
        rssi_margin = rssi - self.sensitivity
        if rssi_margin < 10:
            pdr *= (rssi_margin / 10)
        
        return np.clip(pdr, 0, 100)
    
    def evaluate_all_metrics(self):
        """Evalúa todas las métricas para todos los nodos"""
        print("\n" + "="*70)
        print("   EVALUACIÓN DE MÉTRICAS LoRaWAN - LoRaWISEP")
        print("="*70)
        
        n_nodes = len(self.nodes)
        
        # Arrays para almacenar resultados
        distances = np.zeros(n_nodes)
        rssi_values = np.zeros(n_nodes)
        snr_values = np.zeros(n_nodes)
        pdr_values = np.zeros(n_nodes)
        
        print(f"\n📡 Calculando métricas para {n_nodes} nodos...")
        
        for i in range(n_nodes):
            # Calcular distancia
            distances[i] = self.calculate_distance(i)
            
            # Calcular RSSI
            rssi_values[i] = self.calculate_rssi(distances[i])
            
            # Calcular SNR
            snr_values[i] = self.calculate_snr(rssi_values[i])
            
            # Calcular PDR
            pdr_values[i] = self.calculate_pdr(rssi_values[i], snr_values[i])
        
        # Almacenar resultados
        self.metrics = {
            'distance': distances,
            'rssi': rssi_values,
            'snr': snr_values,
            'pdr': pdr_values
        }
        
        # Calcular estadísticas
        #self._calculate_statistics()
        
        return self.metrics
    
    def _calculate_statistics(self):
        """Calcula estadísticas de las métricas"""
        print("\n" + "="*70)
        print(" ESTADÍSTICAS DE MÉTRICAS")
        print("="*70)
        
        metrics_stats = {}
        
        for metric_name, values in self.metrics.items():
            stats_dict = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
            metrics_stats[metric_name] = stats_dict
        
        # Imprimir estadísticas
        print("\n RANGO DE COMUNICACIÓN:")
        print(f"   • Distancia Promedio: {metrics_stats['distance']['mean']:.2f} m")
        print(f"   • Distancia Mínima: {metrics_stats['distance']['min']:.2f} m")
        print(f"   • Distancia Máxima: {metrics_stats['distance']['max']:.2f} m")
        print(f"   • Desviación Estándar: {metrics_stats['distance']['std']:.2f} m")
        
        print("\n RSSI (Fuerza de Señal):")
        print(f"   • RSSI Promedio: {metrics_stats['rssi']['mean']:.2f} dBm")
        print(f"   • RSSI Mínimo: {metrics_stats['rssi']['min']:.2f} dBm")
        print(f"   • RSSI Máximo: {metrics_stats['rssi']['max']:.2f} dBm")
        print(f"   • Sensibilidad Receptor: {self.sensitivity} dBm")
        
        # Clasificar calidad RSSI
        excellent = np.sum(self.metrics['rssi'] > -80)
        good = np.sum((self.metrics['rssi'] > -100) & (self.metrics['rssi'] <= -80))
        fair = np.sum((self.metrics['rssi'] > -120) & (self.metrics['rssi'] <= -100))
        poor = np.sum(self.metrics['rssi'] <= -120)
        
        print(f"\n   Clasificación de señal:")
        print(f"   • Excelente (>-80 dBm): {excellent} nodos ({excellent/len(self.nodes)*100:.1f}%)")
        print(f"   • Buena (-80 a -100 dBm): {good} nodos ({good/len(self.nodes)*100:.1f}%)")
        print(f"   • Regular (-100 a -120 dBm): {fair} nodos ({fair/len(self.nodes)*100:.1f}%)")
        print(f"   • Pobre (<-120 dBm): {poor} nodos ({poor/len(self.nodes)*100:.1f}%)")
        
        print("\n SNR (Calidad de Señal):")
        print(f"   • SNR Promedio: {metrics_stats['snr']['mean']:.2f} dB")
        print(f"   • SNR Mínimo: {metrics_stats['snr']['min']:.2f} dB")
        print(f"   • SNR Máximo: {metrics_stats['snr']['max']:.2f} dB")
        
        # Clasificar calidad SNR
        excellent_snr = np.sum(self.metrics['snr'] > 10)
        good_snr = np.sum((self.metrics['snr'] > 0) & (self.metrics['snr'] <= 10))
        fair_snr = np.sum((self.metrics['snr'] > -10) & (self.metrics['snr'] <= 0))
        poor_snr = np.sum(self.metrics['snr'] <= -10)
        
        print(f"\n   Clasificación de SNR:")
        print(f"   • Excelente (>10 dB): {excellent_snr} nodos ({excellent_snr/len(self.nodes)*100:.1f}%)")
        print(f"   • Buena (0 a 10 dB): {good_snr} nodos ({good_snr/len(self.nodes)*100:.1f}%)")
        print(f"   • Regular (-10 a 0 dB): {fair_snr} nodos ({fair_snr/len(self.nodes)*100:.1f}%)")
        print(f"   • Pobre (<-10 dB): {poor_snr} nodos ({poor_snr/len(self.nodes)*100:.1f}%)")
        
        print("\n PDR (Tasa de Entrega de Paquetes):")
        print(f"   • PDR Promedio: {metrics_stats['pdr']['mean']:.2f}%")
        print(f"   • PDR Mínimo: {metrics_stats['pdr']['min']:.2f}%")
        print(f"   • PDR Máximo: {metrics_stats['pdr']['max']:.2f}%")
        
        # Análisis de cobertura
        coverage_90 = np.sum(self.metrics['pdr'] >= 90)
        coverage_75 = np.sum(self.metrics['pdr'] >= 75)
        coverage_50 = np.sum(self.metrics['pdr'] >= 50)
        no_coverage = np.sum(self.metrics['pdr'] < 50)
        
        print(f"\n   Cobertura de red:")
        print(f"   • PDR ≥ 90%: {coverage_90} nodos ({coverage_90/len(self.nodes)*100:.1f}%)")
        print(f"   • PDR ≥ 75%: {coverage_75} nodos ({coverage_75/len(self.nodes)*100:.1f}%)")
        print(f"   • PDR ≥ 50%: {coverage_50} nodos ({coverage_50/len(self.nodes)*100:.1f}%)")
        print(f"   • PDR < 50%: {no_coverage} nodos ({no_coverage/len(self.nodes)*100:.1f}%)")
        
        # Evaluación general
        avg_pdr = metrics_stats['pdr']['mean']
        print(f"\n EVALUACIÓN GENERAL DE LA RED:")
        if avg_pdr >= 90:
            print(f"   EXCELENTE - PDR promedio: {avg_pdr:.2f}%")
        elif avg_pdr >= 75:
            print(f"   BUENA - PDR promedio: {avg_pdr:.2f}%")
        elif avg_pdr >= 50:
            print(f"   REGULAR - PDR promedio: {avg_pdr:.2f}%")
        else:
            print(f"   POBRE - PDR promedio: {avg_pdr:.2f}% - Se requiere optimización")
        
        self.metrics_stats = metrics_stats
    
    def plot_metrics(self):
        """Genera visualizaciones de las métricas"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Histograma de Distancias
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(self.metrics['distance'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(self.metrics['distance']), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {np.mean(self.metrics["distance"]):.1f}m')
        ax1.set_xlabel('Distancia (m)', fontweight='bold')
        ax1.set_ylabel('Frecuencia', fontweight='bold')
        ax1.set_title('Distribución de Distancias', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograma de RSSI
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(self.metrics['rssi'], bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(self.metrics['rssi']), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {np.mean(self.metrics["rssi"]):.1f} dBm')
        ax2.axvline(self.sensitivity, color='green', linestyle=':', 
                   linewidth=2, label=f'Sensibilidad: {self.sensitivity} dBm')
        ax2.set_xlabel('RSSI (dBm)', fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontweight='bold')
        ax2.set_title('Distribución de RSSI', fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma de SNR
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(self.metrics['snr'], bins=30, color='green', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(self.metrics['snr']), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {np.mean(self.metrics["snr"]):.1f} dB')
        ax3.axvline(-7.5, color='orange', linestyle=':', 
                   linewidth=2, label='Umbral: -7.5 dB')
        ax3.set_xlabel('SNR (dB)', fontweight='bold')
        ax3.set_ylabel('Frecuencia', fontweight='bold')
        ax3.set_title('Distribución de SNR', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. RSSI vs Distancia
        ax4 = plt.subplot(3, 3, 4)
        scatter = ax4.scatter(self.metrics['distance'], self.metrics['rssi'], 
                             c=self.metrics['pdr'], cmap='RdYlGn', s=50, alpha=0.6, 
                             edgecolors='black', linewidth=0.5)
        ax4.axhline(self.sensitivity, color='red', linestyle='--', 
                   linewidth=2, label='Sensibilidad')
        ax4.set_xlabel('Distancia (m)', fontweight='bold')
        ax4.set_ylabel('RSSI (dBm)', fontweight='bold')
        ax4.set_title('RSSI vs Distancia', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='PDR (%)')
        
        # 5. SNR vs Distancia
        ax5 = plt.subplot(3, 3, 5)
        scatter2 = ax5.scatter(self.metrics['distance'], self.metrics['snr'], 
                              c=self.metrics['pdr'], cmap='RdYlGn', s=50, alpha=0.6,
                              edgecolors='black', linewidth=0.5)
        ax5.axhline(-7.5, color='red', linestyle='--', linewidth=2, label='Umbral SNR')
        ax5.set_xlabel('Distancia (m)', fontweight='bold')
        ax5.set_ylabel('SNR (dB)', fontweight='bold')
        ax5.set_title('SNR vs Distancia', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax5, label='PDR (%)')
        
        # 6. PDR vs Distancia
        ax6 = plt.subplot(3, 3, 6)
        scatter3 = ax6.scatter(self.metrics['distance'], self.metrics['pdr'], 
                              c=self.metrics['snr'], cmap='viridis', s=50, alpha=0.6,
                              edgecolors='black', linewidth=0.5)
        ax6.axhline(90, color='green', linestyle='--', linewidth=2, label='PDR 90%')
        ax6.axhline(75, color='orange', linestyle='--', linewidth=2, label='PDR 75%')
        ax6.set_xlabel('Distancia (m)', fontweight='bold')
        ax6.set_ylabel('PDR (%)', fontweight='bold')
        ax6.set_title('PDR vs Distancia', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax6, label='SNR (dB)')
        
        # 7. Boxplot comparativo
        ax7 = plt.subplot(3, 3, 7)
        data_to_plot = [self.metrics['distance']]
        ax7.boxplot(data_to_plot, labels=['Distancia'])
        ax7.set_ylabel('Distancia (m)', fontweight='bold')
        ax7.set_title('Boxplot de Distancia', fontweight='bold', fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        # 8. Histograma de PDR
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(self.metrics['pdr'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        ax8.axvline(np.mean(self.metrics['pdr']), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {np.mean(self.metrics["pdr"]):.1f}%')
        ax8.set_xlabel('PDR (%)', fontweight='bold')
        ax8.set_ylabel('Frecuencia', fontweight='bold')
        ax8.set_title('Distribución de PDR', fontweight='bold', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Mapa de calor de métricas por Gateway
        ax9 = plt.subplot(3, 3, 9)
        gw_metrics = []
        for gw_id in range(len(self.gateways)):
            nodes_in_gw = self.assignments == gw_id
            if np.sum(nodes_in_gw) > 0:
                gw_metrics.append([
                    np.mean(self.metrics['distance'][nodes_in_gw]),
                    np.mean(self.metrics['rssi'][nodes_in_gw]),
                    np.mean(self.metrics['snr'][nodes_in_gw]),
                    np.mean(self.metrics['pdr'][nodes_in_gw])
                ])
            else:
                # Gateway sin nodos asignados
                gw_metrics.append([0, 0, 0, 0])
        
        if len(gw_metrics) > 0:
            gw_metrics_array = np.array(gw_metrics).T  # Shape: (4, n_gateways)
            
            # Solo plotear si hay datos válidos
            if gw_metrics_array.size > 0:
                im = ax9.imshow(gw_metrics_array, cmap='RdYlGn', aspect='auto')
                ax9.set_yticks(range(4))
                ax9.set_yticklabels(['Dist (m)', 'RSSI (dBm)', 'SNR (dB)', 'PDR (%)'])
                ax9.set_xticks(range(len(gw_metrics)))
                ax9.set_xticklabels([f'GW-{i+1}' for i in range(len(gw_metrics))], 
                                    rotation=45 if len(gw_metrics) > 10 else 0)
                ax9.set_title('Métricas Promedio por Gateway', fontweight='bold', fontsize=12)
                plt.colorbar(im, ax=ax9)
                
                # Agregar valores en las celdas (solo si no son demasiadas)
                if len(gw_metrics) <= 20:  # Evitar saturación visual
                    for i in range(4):
                        for j in range(len(gw_metrics)):
                            value = gw_metrics_array[i, j]
                            text = ax9.text(j, i, f'{value:.1f}',
                                        ha="center", va="center", 
                                        color="black" if value > 0 else "gray",
                                        fontsize=8)
            else:
                ax9.text(0.5, 0.5, 'No hay datos para mostrar', 
                        ha='center', va='center', transform=ax9.transAxes)
        else:
            ax9.text(0.5, 0.5, 'No hay gateways', 
                    ha='center', va='center', transform=ax9.transAxes)
        
        plt.tight_layout()
        plt.show()


from matplotlib.lines import Line2D

# ============================================================================
if __name__ == "__main__":
    # Importar el optimizador principal
    from kmeans import LoRaWISEPOptimization
    from ga import LoRaWISEPGAElbow
    #from lorawise_optimization_ga import LoRaWISEPGAElbow
    #from lorawise_optimizationkmeans import LoRaWISEPOptimization
    
    print("\n" + "="*70)
    print("   LoRaWISEP - OPTIMIZACIÓN Y EVALUACIÓN COMPLETA")
    print("="*70)
    
    # 1. Cargar datos
    try:
        nodos = pd.read_csv("nodos_iot.csv")
        X = nodos[["X_m", "Y_m"]].values
        print(f"\n CSV cargado: {len(X)} nodos")
    except:
        print("\n  Usando datos de ejemplo...")

    width = 10000
    height = 10000

    # ============================================================================
    # 2. EJECUTAR K-MEANS
    # ============================================================================
    print("\n" + "="*70)
    print("MÉTODO 1: K-MEANS")
    print("="*70)
    
    optimizer_kmeans = LoRaWISEPOptimization(X, 34)
    optimal_k_kmeans = optimizer_kmeans.elbow_method(plot=False)
    gateways_kmeans, assignments_kmeans = optimizer_kmeans.apply_kmeans(plot=False)
    """
    optimizer_kmeans = LoRaWISEPOptimization(X, max_k=13)
    optimal_k_kmeans = optimizer_kmeans.elbow_method(plot=False)
    gateways_kmeans, assignments_kmeans = optimizer_kmeans.apply_kmeans(plot=False)
    """

    # ============================================================================
    # 3. EJECUTAR ALGORITMO GENÉTICO
    # ============================================================================
    print("\n" + "="*70)
    print("MÉTODO 2: ALGORITMO GENÉTICO")
    print("="*70)
    
    optimizer = LoRaWISEPGAElbow(
        nodes_coords=X,
        auto_k_range=False,
        max_k=34,
        min_k=1,
        width=width,
        height=height
    )
    optimal_k_manual = optimizer.elbow_method(plot=False)
    gateways_ga, assignments_ga = optimizer.get_optimal_solution(plot=False)

    # ============================================================================
    # 5. EVALUACIÓN DE MÉTRICAS PARA AMBOS MÉTODOS
    # ============================================================================
    print("\n" + "="*70)
    print("EVALUACIÓN DE MÉTRICAS - K-MEANS")
    print("="*70)
    
    evaluator_kmeans = LoRaWANMetricsEvaluator(
        nodes=X,
        gateways=gateways_kmeans,
        assignments=assignments_kmeans,
        frequency=915
    )
    metrics_kmeans = evaluator_kmeans.evaluate_all_metrics()
    #evaluator_kmeans.plot_metrics()
    evaluator_kmeans.export_results('lorawise_kmeans_metrics.csv')
    
    print("\n" + "="*70)
    print("EVALUACIÓN DE MÉTRICAS - ALGORITMO GENÉTICO")
    print("="*70)
    
    evaluator_ga = LoRaWANMetricsEvaluator(
        nodes=X,
        gateways=gateways_ga,
        assignments=assignments_ga,
        frequency=915
    )
    metrics_ga = evaluator_ga.evaluate_all_metrics()
    #evaluator_ga.plot_metrics()
    evaluator_ga.export_results('lorawise_ga_metrics.csv')
    
    # ============================================================================
    # 6. COMPARACIÓN DE MÉTRICAS
    # ============================================================================
    print("\n" + "="*70)
    print(" COMPARACIÓN DE RENDIMIENTO")
    print("="*70)
    
    print(f"\n{'Métrica':<25} {'K-Means':<15} {'GA':<15} {'Mejor':<10}")
    print("-" * 70)
    
    metrics_comparison = {
        'Distancia Promedio (m)': (
            np.mean(metrics_kmeans['distance']),
            np.mean(metrics_ga['distance'])
        ),
        'RSSI Promedio (dBm)': (
            np.mean(metrics_kmeans['rssi']),
            np.mean(metrics_ga['rssi'])
        ),
        'SNR Promedio (dB)': (
            np.mean(metrics_kmeans['snr']),
            np.mean(metrics_ga['snr'])
        ),
        'PDR Promedio (%)': (
            np.mean(metrics_kmeans['pdr']),
            np.mean(metrics_ga['pdr'])
        )
    }
    
    for metric_name, (kmeans_val, ga_val) in metrics_comparison.items():
        # Determinar cuál es mejor (para distancia menor es mejor, para otros mayor es mejor)
        if 'Distancia' in metric_name:
            better = 'K-Means' if kmeans_val < ga_val else 'GA'
        else:
            better = 'K-Means' if kmeans_val > ga_val else 'GA'
        
        print(f"{metric_name:<25} {kmeans_val:<15.2f} {ga_val:<15.2f} {better:<10}")
    
    print("\n" + "="*70)
    print(" EVALUACIÓN COMPLETA FINALIZADA")
    print("="*70)

