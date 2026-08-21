# ========== IMPORTS ==========
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import random
np.random.seed(42)
random.seed(42)

# Importar clase base de georeferenciación
sys.path.append(str(Path(__file__).parent))
from georeferencia import GeoreferenciaMapa

# Importar optimizador LoRaWISEP
opt_dir = Path(__file__).parent / 'optimizacion_gateways'

sys.path.insert(0, str(opt_dir))
from optimizacion_gateways.kmeans import LoRaWISEPOptimization
from optimizacion_gateways.ga import LoRaWISEPGAElbow
from optimizacion_gateways.metricas import LoRaWANMetricsEvaluator
from metricas import gtw_distance, gtw_distance_uniforme, gtw_distance_aleatorio
class GeoreferenciaRedLoRaWAN(GeoreferenciaMapa):
    """
    Extensión de GeoreferenciaMapa que integra optimización de red LoRaWAN
    
    Hereda toda la funcionalidad de georeferenciación y añade:
    - Generación de nodos IoT desde datos de delitos
    - Optimización de posicionamiento de gateways LoRaWAN
    - Visualización de red completa con cobertura
    """
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda, bbox):
        """
        Args:
            archivos_especificos: lista de tuplas (archivo.csv, categoría)
            PESOS_DELITOS: diccionario con pesos por tipo de delito
            mejorcelda: tamaño óptimo de celda hexagonal (metros)
        """
        # Inicializar clase base
        super().__init__(archivos_especificos, PESOS_DELITOS, mejorcelda, bbox)
        
        # Atributos adicionales para red LoRaWAN
        self.nodos_iot = None
        self.nodos_iot_riesgo = None
        self.nodos_iot_uniforme = None
        self.nodos_iot_aleatorio = None
        self.gateways = None
        self.gateways_ga = None
        self.assignments = None
        self.optimizer = None

    def generar_nodos_iot_desde_delitos(self, num_nodos_exacto=None):
        
        """
        Genera posiciones de nodos IoT basándose en datos de delitos
        
        Args:
            estrategia: método de distribución de nodos
            min_nivel: nivel mínimo de inseguridad para colocar nodos (1-6)
            num_nodos_exacto: número exacto de nodos a generar (None = automático)
            
        Returns:
            nodos_coords: array (N, 2) con coordenadas de nodos IoT
        """
        print("\n Generando nodos IoT desde datos de delitos...")
        
        nodos_coords = []
        
        if num_nodos_exacto is not None:
            # Filtrar celdas con datos
            celdas_activas = self.grid_cali[self.grid_cali['nivel_inseguridad'] > 0].copy()
            
            # Calcular peso de cada celda (proporcional al índice de inseguridad)
            celdas_activas['peso'] = celdas_activas['indice_inseguridad']
            suma_pesos = celdas_activas['peso'].sum()
            
            if suma_pesos == 0:
                # Si todos tienen peso 0, distribuir uniformemente
                celdas_activas['nodos_asignados'] = num_nodos_exacto // len(celdas_activas)
            else:
                # Asignar nodos proporcionalmente
                celdas_activas['nodos_asignados'] = (
                    celdas_activas['peso'] / suma_pesos * num_nodos_exacto
                ).round().astype(int)
            
            # Ajustar para llegar exactamente a num_nodos_exacto
            total_asignado = celdas_activas['nodos_asignados'].sum()
            diferencia = num_nodos_exacto - total_asignado
            
            if diferencia > 0:
                # Agregar nodos faltantes a las celdas con mayor peso
                celdas_ordenadas = celdas_activas.nlargest(diferencia, 'peso')
                for idx in celdas_ordenadas.index:
                    celdas_activas.loc[idx, 'nodos_asignados'] += 1
            elif diferencia < 0:
                # Quitar nodos sobrantes de las celdas con menor peso (que tengan al menos 1)
                celdas_con_nodos = celdas_activas[celdas_activas['nodos_asignados'] > 0]
                celdas_ordenadas = celdas_con_nodos.nsmallest(abs(diferencia), 'peso')
                for idx in celdas_ordenadas.index:
                    celdas_activas.loc[idx, 'nodos_asignados'] -= 1
            
            # Generar nodos dentro de cada celda
            for idx, row in celdas_activas.iterrows():
                n_nodos = int(row['nodos_asignados'])
                if n_nodos == 0:
                    continue
                
                poly = row['geometry']
                minx, miny, maxx, maxy = poly.bounds
                
                for _ in range(n_nodos):
                    intentos = 0
                    while intentos < 100:  # Límite de intentos para evitar bucles infinitos
                        x = np.random.uniform(minx, maxx)
                        y = np.random.uniform(miny, maxy)
                        punto = Point(x, y)
                        if poly.contains(punto):
                            nodos_coords.append([x, y])
                            break
                        intentos += 1
                    
                    if intentos >= 100:
                        # Si no se puede colocar dentro, usar centroide
                        centroid = poly.centroid
                        nodos_coords.append([centroid.x, centroid.y])
        else:
            print(f"cero nodos IoT generados")
     
        self.nodos_iot = np.array(nodos_coords)
        print(f"{len(self.nodos_iot)} nodos IoT generados")
        return self.nodos_iot
    
    def optimizar_gateways_kmeans(self):
        """
        Optimiza posicionamiento de gateways LoRaWAN usando LoRaWISEP
        
        Args:
            auto_k: si True, determina K automáticamente con método del codo
            max_k: número máximo de gateways (solo si auto_k=False)
            
        Returns:
            gateways: array (K, 2) con posiciones de gateways
            assignments: array (N,) con asignación nodo->gateway
        """
        print("\nOptimizando posicionamiento de gateways LoRaWAN...")
        
        # Calcular dimensiones del área de cobertura
        x_coords = self.nodos_iot[:, 0]
        y_coords = self.nodos_iot[:, 1]
        
        width = float(x_coords.max() - x_coords.min())
        height = float(y_coords.max() - y_coords.min())
        
        # Rango para búsqueda de K óptimo
        k_range = range(2, min(21, len(self.nodos_iot) // 10 + 1))
        """
        print(f"  • Nodos IoT: {len(self.nodos_iot)}")
        print(f"  • Área de cobertura: {width:.1f} x {height:.1f} metros")
        print(f"  • Rango de búsqueda K: {k_range.start} - {k_range.stop - 1}")
        """
        # Crear optimizador con parámetros requeridos
        self.optimizer = LoRaWISEPOptimization(
            self.nodos_iot,
            width=width,
            height=height,
            #max_k=38,
            #min_k=1,
            auto_k_range=True
        )
        
        # Determinar K óptimo con método del codo
        self.optimizer.elbow_method(plot=False)
        
        # Posicionar gateways con K-Means
        gateways_km, assignments_km = self.optimizer.apply_kmeans(plot=False)
        
        # Guardar resultados específicos de KMeans
        self.gateways_kmeans = np.array(gateways_km)
        self.assignments_kmeans = np.array(assignments_km)
        # También actualizar referencia genérica (mantener último método si se desea)
        self.gateways = self.gateways_kmeans
        self.assignments = self.assignments_kmeans
        
        print(f"\n{len(self.gateways)} gateways optimizados")
        
        return self.gateways, self.assignments
    
    def optimizar_gateways_ga(self):

        # Calcular dimensiones del área de cobertura
        x_coords = self.nodos_iot[:, 0]
        y_coords = self.nodos_iot[:, 1]
        
        width = float(x_coords.max() - x_coords.min())
        height = float(y_coords.max() - y_coords.min())
        
        # Rango para búsqueda de K óptimo
        """
        k_range = range(2, min(21, len(self.nodos_iot) // 10 + 1))
        
        print(f"  • Nodos IoT: {len(self.nodos_iot)}")
        print(f"  • Área de cobertura: {self.width:.1f} x {self.height:.1f} metros")
        print(f"  • Rango de búsqueda K: {k_range.start} - {k_range.stop - 1}")
        """

        optimizer = LoRaWISEPGAElbow(
            nodes_coords=self.nodos_iot,
            auto_k_range=True,
            #max_k=38,
            #min_k=1,
            width=width,
            height=height
        )
        optimal_k = optimizer.elbow_method(
            plot=False,                  # Mostrar gráficas
            verbose_ga=False             # No mostrar progreso detallado del GA
        )
        
        # Obtener solución óptima
        gateways_ga, assignments_ga = optimizer.get_optimal_solution(plot=False)
        
        # Guardar resultados específicos de GA
        self.gateways_ga = np.array(gateways_ga)
        self.assignments_ga = np.array(assignments_ga)
        # Actualizar referencia genérica
        self.gateways = self.gateways_ga
        self.assignments = self.assignments_ga
                
        # Guardar referencia al optimizador
        self.optimizer = optimizer
        
        print(f"K óptimo determinado: {optimal_k}")

        # Crear DataFrame con las posiciones en metros
        df = pd.DataFrame(
            self.gateways_ga,
            columns=['x', 'y']
        )
        
        # Agregar ID de nodo
        df.insert(0, 'id', range(1, len(self.gateways_ga) + 1))

        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_simple = output_path / "gateways.csv"

        df.to_csv(out_path_simple, index=False)
        
        print(f"\nPosiciones de nodos IoT guardadas en: {output_path}")

        return self.gateways, self.assignments
    
 
    def comparar_metodos(self, export_csv=False):
            """Usa metricas.LoRaWANMetricsEvaluator para evaluar KMeans vs GA y decidir cuál es mejor."""
            if getattr(self, 'gateways_kmeans', None) is None or getattr(self, 'gateways_ga', None) is None:
                raise ValueError("Ejecuta optimizar_gateways_kmeans() y optimizar_gateways_ga() antes de comparar.")
            
            print("\nEvaluando métricas: K-Means vs GA")
            
            eval_km = LoRaWANMetricsEvaluator(nodes=self.nodos_iot,
                                              gateways=self.gateways_kmeans,
                                              assignments=self.assignments_kmeans,
                                              frequency=915)
            metrics_km = eval_km.evaluate_all_metrics()
            if export_csv:
                eval_km.export_results('metrics_kmeans.csv')
            
            eval_ga = LoRaWANMetricsEvaluator(nodes=self.nodos_iot,
                                              gateways=self.gateways_ga,
                                              assignments=self.assignments_ga,
                                              frequency=915)
            metrics_ga = eval_ga.evaluate_all_metrics()
            if export_csv:
                eval_ga.export_results('metrics_ga.csv')
            
            # Métricas de comparación (medias)
            cmp = {
                'Distancia_mean': (np.mean(metrics_km['distance']), np.mean(metrics_ga['distance'])),
                'RSSI_mean': (np.mean(metrics_km['rssi']), np.mean(metrics_ga['rssi'])),
                'SNR_mean': (np.mean(metrics_km['snr']), np.mean(metrics_ga['snr'])),
                'PDR_mean': (np.mean(metrics_km['pdr']), np.mean(metrics_ga['pdr']))
            }
            
            print("\nComparación (K-Means  |  GA) [mejor marcado con *]")
            for name, (v_km, v_ga) in cmp.items():
                if 'Distancia' in name:
                    better = 'K-Means' if v_km < v_ga else 'GA'
                else:
                    better = 'K-Means' if v_km > v_ga else 'GA'
                mark_km = '*' if better == 'K-Means' else ' '
                mark_ga = '*' if better == 'GA' else ' '
                print(f"{name:<15}: {v_km:8.2f} {mark_km} | {v_ga:8.2f} {mark_ga}")
            
            # Decisión simple: elegir método con mayor PDR promedio
            pdr_km = cmp['PDR_mean'][0]
            pdr_ga = cmp['PDR_mean'][1]
            if pdr_ga > pdr_km:
                mejor = 'GA'
                self.gateways = self.gateways_ga
                self.assignments = self.assignments_ga
            else:
                mejor = 'K-Means'
                self.gateways = self.gateways_kmeans
                self.assignments = self.assignments_kmeans
            
            print(f"\nMétodo seleccionado para visualización final: {mejor} (PDR: KMeans={pdr_km:.2f}%, GA={pdr_ga:.2f}%)")
    
            print("\n" + "="*70)
            print(" COMPARACIÓN DE POSICIONES DE GATEWAYS")
            print("="*70)
            
            # Mostrar comparación gráfica lado a lado
            try:
                self.visualizar_red_completa_comparativa()
            except Exception as e:
                print(f"No se pudo mostrar la comparación gráfica: {e}")
            
            return {'kmeans': metrics_km, 'ga': metrics_ga, 'mejor': mejor}
    
    def visualizar_red_completa_comparativa(self, mostrar_conexiones=False, guardar=True):
        """
        Muestra en un mismo mapa de Cali:
         - el mapa de calor (grid_cali)
         - los nodos IoT
         - los gateways obtenidos por K-Means (triángulos magenta)
         - los gateways obtenidos por GA (círculos negros)
        """
        print("\nVisualización comparativa de gateways (K-Means vs GA)...")

        fig, ax = plt.subplots(figsize=(14, 12))

        # Mapa base y grilla
        self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.0, zorder=1)
        try:
            self.grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=mcolors.ListedColormap(
                ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]), alpha=0.6,
                edgecolor="black", linewidth=0.25, vmin=0, vmax=5, zorder=2)
        except Exception:
            pass

        # IoT Nodes
        if getattr(self, 'nodos_iot', None) is not None and len(self.nodos_iot) > 0:
            ax.scatter(self.nodos_iot[:, 0], self.nodos_iot[:, 1],
                       c='blue', s=25, alpha=0.7, label=f'IoT Nodes ({len(self.nodos_iot)})',
                       zorder=4, edgecolors='none')

        # Gateways K-Means
        if getattr(self, 'gateways_kmeans', None) is not None and len(self.gateways_kmeans) > 0:
            ax.scatter(self.gateways_kmeans[:, 0], self.gateways_kmeans[:, 1],
                       c='#FF1493', marker='^', s=200, edgecolors='white', linewidth=1.2,
                       label=f'K-Means ({len(self.gateways_kmeans)})', zorder=6, alpha=0.95)

        # Gateways GA
        if getattr(self, 'gateways_ga', None) is not None and len(self.gateways_ga) > 0:
            ax.scatter(self.gateways_ga[:, 0], self.gateways_ga[:, 1],
                       c='black', marker='o', s=200, edgecolors='white', linewidth=1.2,
                       label=f'GA ({len(self.gateways_ga)})', zorder=5, alpha=0.95)

        # Opcional: mostrar líneas nodo->gateway para cada método (ligeras y diferentes colores)
        if mostrar_conexiones:
            # K-Means conexiones (rosa, más transparente)
            try:
                if getattr(self, 'assignments_kmeans', None) is not None:
                    for i, nodo in enumerate(self.nodos_iot):
                        gw = self.gateways_kmeans[self.assignments_kmeans[i]]
                        ax.plot([nodo[0], gw[0]], [nodo[1], gw[1]], color='#FF1493', alpha=0.06, linewidth=0.4, zorder=2)
            except Exception:
                pass

            # GA conexiones (negro/gris, muy transparente)
            try:
                if getattr(self, 'assignments_ga', None) is not None:
                    for i, nodo in enumerate(self.nodos_iot):
                        gw = self.gateways_ga[self.assignments_ga[i]]
                        ax.plot([nodo[0], gw[0]], [nodo[1], gw[1]], color='black', alpha=0.05, linewidth=0.4, zorder=2)
            except Exception:
                pass

        # Etiquetas y estética
        ax.set_xlabel("X Coordinate (meters)", fontsize=12)
        ax.set_ylabel("Y Coordinate (meters)", fontsize=12)
        ax.set_title("Comparison of Gateways in Santiago de Cali: K-Means vs GA", fontsize=14, weight='bold', pad=12)

        # Leyenda coherente
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=8, label=f'IoT Nodes ({len(self.nodos_iot) if getattr(self, "nodos_iot", None) is not None else 0})'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF1493', markersize=10, label=f'K-Means ({len(self.gateways_kmeans) if getattr(self, "gateways_kmeans", None) is not None else 0})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=9, label=f'GA ({len(self.gateways_ga) if getattr(self, "gateways_ga", None) is not None else 0})')
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=True, edgecolor='black', bbox_to_anchor=(0.3, 0.98))

        ax.grid(True, alpha=0.25, linestyle='--')
        plt.tight_layout()

        if guardar:
            images_dir = Path(__file__).resolve().parent / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            outname = images_dir / 'comparativa_gateways_en_mapa_cali.png'
            plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparativa guardada: {outname}")

        #plt.show()

    def visualizar_red_completa(self, mostrar_conexiones=True, guardar=True, name=None):
        """
        Visualiza mapa de calor + nodos IoT + gateways LoRaWAN
        
        Args:
            mostrar_conexiones: si True, dibuja líneas nodo->gateway
            guardar: si True, guarda imagen PNG
        """
        print("\n Generando visualización completa de red LoRaWAN...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Mapa base
        self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.5, zorder=1)
        
        # Colormap para niveles de inseguridad
        colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
        
        # Hexágonos con niveles de inseguridad
        self.grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                           alpha=0.65, edgecolor="black", linewidth=0.3, vmin=0, vmax=5, zorder=2)
        
        # Conexiones nodo->gateway
        if mostrar_conexiones and self.nodos_iot is not None and self.gateways is not None:
            for i, nodo in enumerate(self.nodos_iot):
                gw = self.gateways[self.assignments[i]]
                ax.plot([nodo[0], gw[0]], [nodo[1], gw[1]], 
                       'gray', alpha=0.15, linewidth=0.5, zorder=3)
        
        # IoT Nodes
        if self.nodos_iot is not None:
            ax.scatter(self.nodos_iot[:, 0], self.nodos_iot[:, 1],
                      c='blue', s=25, alpha=0.7, label=f'IoT Nodes ({len(self.nodos_iot)})', 
                      zorder=4, edgecolors='darkblue', linewidth=0.5)
        
        # Gateways LoRaWAN
        if self.gateways is not None:
            ax.scatter(self.gateways[:, 0], self.gateways[:, 1],
                      c='red', marker='*', s=200, edgecolors='darkred',
                      linewidth=1.5, label=f'Gateways LoRaWAN ({len(self.gateways)})', zorder=5)
            
            # Etiquetas de gateways
            for i, gw in enumerate(self.gateways):
                ax.annotate(f'GW-{i+1}', (gw[0], gw[1]),
                           xytext=(12, 12), textcoords='offset points',
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', 
                                    edgecolor='black', alpha=0.9, linewidth=2))
        
        # Barra de color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Level of Insecurity", shrink=0.6, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['No data', 'Very Low\n(0-20%)', 'Low\n(20-40%)',
                            'Medium\n(40-60%)', 'High\n(60-80%)', 'Very High\n(80-100%)'])
        
        # Título y etiquetas
        ax.set_xlabel("X Coordinate (meters)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Y Coordinate (meters)", fontsize=12, fontweight='bold')
        ax.set_title("Optimized LoRaWAN Network for Public Safety",
                    fontsize=14, fontweight='bold', pad=10)
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=8, label=f'IoT Nodes ({len(self.nodos_iot) if getattr(self, "nodos_iot", None) is not None else 0})'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',  markersize=16, label=f'Gateways LoRaWAN ({len(self.gateways) if getattr(self, "gateways_kmeans", None) is not None else 0})')
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=True, edgecolor='black', bbox_to_anchor=(0.4, 0.98))

        ax.grid(True, alpha=0.25, linestyle='--')
        
        plt.tight_layout()
        
        if guardar:
            images_dir = Path(__file__).resolve().parent / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            outname = (images_dir / name) if name else (images_dir / 'red_lorawan_cali_optimizada.png')
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            print(f"Mapa guardado: {outname}")
        
        #plt.show()
        

    def visualizar_gtw_distance_comparativo(self, res_uniforme, res_aleatorio, res_riesgo,
                                            guardar=True, nombre_mapa='qos_logdistance_comparativo'):
        """
        Genera una sola figura con 3 subgráficas:
        - gtw_distance (riesgo)
        - gtw_distance_uniforme (uniforme)
        - gtw_distance_aleatorio (aleatorio)
        """

        escenarios = [
            ("Uniform Base Model", self.nodos_iot_uniforme, res_uniforme['assignments_modelo'], res_uniforme['dataframe']['calidad']),
            ("Random Base Model", self.nodos_iot_aleatorio, res_aleatorio['assignments_modelo'], res_aleatorio['dataframe']['calidad']),
            ("Risk Model", self.nodos_iot_riesgo, res_riesgo['assignments_modelo'], res_riesgo['dataframe']['calidad'])

        ]

        color_map = {
            'Excellent': "green",
            'Good': "yellow",
            'Regular': "#f05209",
        }

        calidad_map = {
            'Excelente': 'Excellent',
            'Buena': 'Good',
            'Excelent': 'Excellent'
        }

        # Configuración global de fuente
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8

        textwidth_pt = 472.03123
        fig_width_in = textwidth_pt / 72.27   # ≈ 6.531 in
        fig_height_in = fig_width_in * 0.9            # ≈ 5.88 in (más que suficiente)
        fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))

        # Asigna los tres mapas a las tres primeras celdas
        ax1, ax2 = axes[0, 0], axes[0, 1]   # fila superior
        ax3, ax_leg = axes[1, 0], axes[1, 1] # fila inferior: mapa 3 + leyenda

        
        # Lista de los ejes donde se dibujan los mapas
        map_axes = [ax1, ax2, ax3]

        # Dibujar cada uno de los tres escenarios
        for ax, (titulo, nodos, assignments_modelo, calidad) in zip(map_axes, escenarios):
            # Mapa base de Cali
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=0.8, zorder=1)
            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                except Exception:
                    pass

            gateways = np.asarray(self.gateways_ga)
            nodos = np.asarray(nodos)

            # Normalizar las etiquetas de calidad al inglés
            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            # Nodos IoT
            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=28, alpha=0.75,
                    edgecolors='none', zorder=4)

            # Gateways GA
            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=250,
                    edgecolors='white', linewidth=0.8, zorder=3)
            # Etiquetas y formato
            ax.set_xlabel('X Coordinate (m)', fontsize=8)
            ax.set_ylabel('Y Coordinate (m)', fontsize=8)
            ax.set_title(titulo, fontsize=9, weight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        # Panel de leyenda unificada (cuarta celda)
        ax_leg.axis('off')  # ocultar ejes
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'],
                markersize=8, label="Excellent"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'],
                markersize=8, label="Good"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'],
                markersize=8, label="Regular"),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
                markersize=10, label="GA Gateways")
        ]
        ax_leg.legend(handles=legend_handles, loc='center', fontsize=9,
                    frameon=True, edgecolor='gray')

        # Ajuste fino de espaciados
        plt.tight_layout(pad=1.5, w_pad=2.5, h_pad=2.5)

        if guardar:
            images_dir = Path(__file__).resolve().parent / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            outname = images_dir / (nombre_mapa)
            plt.savefig(outname, format='pdf', bbox_inches='tight')
            print(f" Figura comparativa guardada: {outname}")

        #plt.show()

    def visualizar_gtw_distance(self, res_uniforme, res_aleatorio, res_riesgo, nombre_mapa='qos_logdistance_comparativo.pdf'):

        escenarios = [
            ("Uniform Base Model", self.nodos_iot_uniforme, res_uniforme['assignments_modelo'], res_uniforme['dataframe']['calidad']),
            ("Random Base Model", self.nodos_iot_aleatorio, res_aleatorio['assignments_modelo'], res_aleatorio['dataframe']['calidad']),
            ("Risk Model", self.nodos_iot_riesgo, res_riesgo['assignments_modelo'], res_riesgo['dataframe']['calidad'])
        ]

        color_map = {
            'Excellent': "green",
            'Good': "yellow",
            'Regular': "#f05209",
        }

        calidad_map = {
            'Excelente': 'Excellent',
            'Buena': 'Good',
            'Excelent': 'Excellent'
        }

        # ── Dimensiones para doble columna (tres mapas en fila) ──
        fig_width_in = 7.16
        fig_height_in = fig_width_in * 0.45   # ajusta según necesites

        # Crear una sola figura con 3 ejes en una fila
        fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in))

        # Ajustar espacio entre subplots (se aplica a la figura)
        fig.subplots_adjust(wspace=0.6)

        # Etiquetas de subfiguras
        etiquetas = ['(a)', '(b)', '(c)']

        for ax, (titulo, nodos, assignments_modelo, calidad), etiq in zip(axes, escenarios, etiquetas):
            # Mapa base de Cali
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=0.8, zorder=1)

            if self.bbox is not None:
                minx, maxx, miny, maxy = self.bbox
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

            gateways = np.asarray(self.gateways_ga)
            nodos = np.asarray(nodos)

            # Verificar que nodos no esté vacío
            if len(nodos) == 0:
                print(f"Advertencia: No hay nodos para {titulo}")
                continue

            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            # Aumenta el tamaño de los puntos para que sean visibles
            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=1, alpha=0.8,
                    edgecolors='none', zorder=4)
            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=15,
                    edgecolors='white', linewidth=0.6, zorder=3)

            ax.set_title(titulo, fontsize=8, weight='bold')
            ax.set_xlabel('X Coordinate (m)', fontsize=8)
            ax.set_ylabel('Y Coordinate (m)', fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

            # Añadir etiqueta (a), (b), (c) centrada debajo
            ax.text(0.5, -0.20, etiq, transform=ax.transAxes,
                    fontsize=8, fontname='Times New Roman', ha='center', va='center')

        # ── Leyenda unificada (fuera de los ejes, a la derecha) ──
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'], markersize=8, label="Excellent"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'], markersize=8, label="Good"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label="Regular"),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label="Gateways")
        ]

        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.5),
                fontsize=8, frameon=True, edgecolor='gray')

        # Ajustar márgenes para que la leyenda no se superponga
        fig.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.12)

        # Guardar
        images_dir = Path(__file__).resolve().parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        outname = images_dir / nombre_mapa
        fig.savefig(outname, format='pdf', bbox_inches='tight')
        print(f"Figura comparativa guardada: {outname}")

        return {
            'riesgo': res_riesgo,
            'uniforme': res_uniforme,
            'aleatorio': res_aleatorio
        }

    def guardar_cuadro(self):

        x_ini = 724000
        y_ini = 867500
        ancho = 10000
        alto = 10000
        x_fin = x_ini + ancho
        y_fin = y_ini + alto
        cuadro = box(x_ini, y_ini, x_fin, y_fin)

        # Seleccionar las celdas del mapa de calor que pertenecen al cuadro.
        celdas_cuadro = self.grid_cali[self.grid_cali.geometry.intersects(cuadro)]
        informacion_cuadro = []
        for indice_celda, (_, celda) in enumerate(celdas_cuadro.iterrows(), start=1):
            geometria = celda.geometry
            centroide = geometria.centroid
            interseccion = geometria.intersection(cuadro)
            informacion_cuadro.append({
                'id_celda': indice_celda,
                'indice_inseguridad': float(celda['indice_inseguridad']),
                'nivel_inseguridad': int(celda['nivel_inseguridad']),
                'centroide': {
                    'x': float(centroide.x),
                    'y': float(centroide.y)
                },
                'area_interseccion_m2': float(interseccion.area)
            })

        # Guardar únicamente datos serializables por JSON.
        output_path = Path(__file__).resolve().parent / 'data_base'
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / 'indice_inseguridad_cuadro_10000x10000.json'
        datos_json = {
            'cuadro': {
                'ancho_m': ancho,
                'alto_m': alto
            },
            'numero_celdas': len(informacion_cuadro),
            'indice_inseguridad': informacion_cuadro
        }
        with json_path.open('w', encoding='utf-8') as archivo:
            json.dump(datos_json, archivo, ensure_ascii=False, indent=2)

        plt.rcParams['font.family'] = 'Arial'
        fig_width_in = 7.16  #  dos columnas
        fig_height_in = fig_width_in * 0.75    
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
                
        # Mapa base
        self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=1.0, zorder=1)
        
        # Si hay bounding box, establecer límites exactos
        if self.bbox is not None:
            minx, maxx, miny, maxy = self.bbox
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        
        # Colormap
        colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
        
        # Hexágonos
        self.grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                        alpha=0.75, edgecolor="black", linewidth=0.3, vmin=0, vmax=5, zorder=3)

        # Resaltar el cuadro de 1.000 m x 1.000 m sobre el mapa de calor.
        rect = Rectangle(
            (x_ini, y_ini), ancho, alto,
            linewidth=2.5, edgecolor='deepskyblue', facecolor='aqua',
            alpha=0.25, zorder=4
        )
        ax.add_patch(rect)
        
        plt.xlabel("X Coordinate (m)", fontsize=8)
        plt.ylabel("Y Coordinate (m)", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        plt.grid(True, alpha=0.5, linestyle='--')
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.86)

        images_dir = Path(__file__).resolve().parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        outname = images_dir / 'mil_cuadro.png'
        plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Cuadro guardado: {outname}")
        print(f"Información de inseguridad guardada: {json_path}")

        return datos_json
    
    def ejecutar_pipeline_georeferenciacion(self):
        """
        Ejecuta el pipeline completo de georeferenciación
        (Reutiliza el método de la clase base)
        """
        print("\n" + "="*70)
        print("FASE 1: GEOREFERENCIACIÓN Y ANÁLISIS DE DELITOS")
        print("="*70)
        
        # Ejecutar pipeline de la clase base
        self.mostrar_encabezado()
        self.mostrar_sistema_pesos()
        self.cargar_mapa_base()
        self.crear_grilla_hexagonal()
        self.cargar_archivos_delitos()
        self.calcular_pesos_delitos()
        self.mostrar_distribucion_delitos()
        self.georreferenciar_y_calcular_scores()
        self.clasificar_niveles_inseguridad()
        
        print("\n Georeferenciación completada")

    def ejecutar_pipeline_red_lorawan(self, num_nodos=1000):
        """
        Ejecuta el pipeline completo de optimización de red LoRaWAN
        """
        print("\n" + "="*70)
        print("FASE 2: OPTIMIZACIÓN DE RED LoRaWAN")
        
        # Generar nodos IoT
        if num_nodos is not None:
            self.generar_nodos_iot_desde_delitos(num_nodos_exacto=num_nodos)
        else:
            self.generar_nodos_iot_desde_delitos()
        
        # Optimizar gateways
        self.optimizar_gateways_ga()
        self.optimizar_gateways_kmeans()
        
        self.comparar_metodos()
        
        # Ejecutar K-Means y guardar su mapa
        # for visualization use kmeans results
        self.gateways = red_cali.gateways_kmeans
        self.assignments = red_cali.assignments_kmeans
        self.visualizar_red_completa(mostrar_conexiones=True, guardar=True, name='red_lorawan_kmeans.png')
        
        # Ejecutar GA y guardar su mapa
        # for visualization use ga results
        self.gateways = red_cali.gateways_ga
        self.assignments = red_cali.assignments_ga
        self.visualizar_red_completa(mostrar_conexiones=True, guardar=True, name='red_lorawan_ga.png')
        print("\n" + "="*70)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import json
    import pickle
    import os
    
    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "optimizacion_celda"
    
    with open(resultados_dir / "resumen_tres_metodos.json") as f:
        config = json.load(f)
        mejorcelda = config['recomendacion']
        mejor_metodo_nombre = config['mejor_metodo']
        
    # Definir pesos de delitos
    PESOS_DELITOS = {
        'Hurto': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
        'Extorsion': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
        'Lesiones Personales': {'severidad': 2, 'factor_genero': 1.0, 'peso_total': 2.0, 'nivel': 'Moderado'},
        'Delitos Sexuales': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Violencia Intrafamiliar': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Lesiones Personales Agravados': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Homicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'},
        'Feminicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'}
    }
    
    archivos_especificos = [
        ('data_base/Hurtos_fiscalia.csv', 'Hurto'),
        ('data_base/Homicidios_fiscalia.csv', 'Homicidio'),
        ('data_base/Delitos_Sexuales_fiscalia.csv', 'Delitos Sexuales'),
        ('data_base/Lesiones_fiscalia.csv', 'Lesiones Personales'),
        ('data_base/Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
        ('data_base/Extorsion_fiscalia.csv', 'Extorsion')
    ]
    
    # ========== CREAR INSTANCIA ==========
    BBOX = (720414.01, 735029.27, 860572.64, 879817.30)  # (minx, maxx, miny, maxy)

    red_cali = GeoreferenciaRedLoRaWAN(archivos_especificos, PESOS_DELITOS, mejorcelda, bbox=BBOX)
    
    pipeline_flag = "pipeline/pipeline_red_lorawan_done.pkl"

    pipeline_geo = "pipeline/pipeline_georeferenciacion_done.pkl"

    pipeline_nodos = "pipeline/pipeline_nodos_done.pkl"

    # Pipeline de georeferenciación
    if not os.path.exists(pipeline_geo):
        # Ejecuta el pipeline
        red_cali.ejecutar_pipeline_georeferenciacion()
        # Guarda el resultado (si la función retorna algo)
        with open(pipeline_geo, "wb") as f:
            pickle.dump(red_cali, f)
    else:
        print("El pipeline ya fue ejecutado, no se recalcula.")
        with open(pipeline_geo, "rb") as f:
            red_cali = pickle.load(f)

    # Pipeline de red LoRaWan
    if not os.path.exists(pipeline_flag):
        # Ejecuta el pipeline
        red_cali.ejecutar_pipeline_red_lorawan(num_nodos=1000)
        # Guarda el resultado (si la función retorna algo)
        with open(pipeline_flag, "wb") as f:
            pickle.dump(red_cali, f)
    else:
        print("El pipeline ya fue ejecutado, no se recalcula.")
        with open(pipeline_flag, "rb") as f:
            red_cali = pickle.load(f)

    cache_comparativo = "pipeline/comparativo_resultados.pkl"

    if os.path.exists(cache_comparativo):
        with open(cache_comparativo, "rb") as f:
            datos = pickle.load(f)
            res_uniforme = datos['res_uniforme']
            res_aleatorio = datos['res_aleatorio']
            res_riesgo = datos['res_riesgo']
            red_cali.nodos_iot_uniforme = datos.get('nodos_uniforme')
            red_cali.nodos_iot_aleatorio = datos.get('nodos_aleatorio')
            red_cali.nodos_iot_riesgo = datos.get('nodos_riesgo')
        print("Resultados comparativos y nodos cargados desde caché.")

    else:
        # Parámetros (ajusta según tu lógica)
        factor_densidad = 1.0
        max_intentos = 100

        # Cálculos reales (justo lo que mostraste)
        res_riesgo = red_cali.gtw_distance(
            factor_densidad=factor_densidad,
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )
        res_uniforme = red_cali.gtw_distance_uniforme(
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )
        res_aleatorio = red_cali.gtw_distance_aleatorio(
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )

        # Asignamos a variables locales para el guardado y la llamada posterior
        res_uniforme = res_uniforme
        res_aleatorio = res_aleatorio
        res_riesgo = res_riesgo

        # Guardar en caché para la próxima ejecución
        with open(cache_comparativo, "wb") as f:
            pickle.dump({
                'res_uniforme': res_uniforme,
                'res_aleatorio': res_aleatorio,
                'res_riesgo': res_riesgo,
                'nodos_uniforme': red_cali.nodos_iot_uniforme,
                'nodos_aleatorio': red_cali.nodos_iot_aleatorio,
                'nodos_riesgo': red_cali.nodos_iot_riesgo
            }, f)

    # Llamada final (fuera del if/else)
    red_cali.visualizar_gtw_distance(
        res_uniforme=res_uniforme,
        res_aleatorio=res_aleatorio,
        res_riesgo=res_riesgo,
        nombre_mapa='qos_logdistance_comparativo.pdf'  # PDF vectorial
    )
    red_cali.guardar_cuadro()