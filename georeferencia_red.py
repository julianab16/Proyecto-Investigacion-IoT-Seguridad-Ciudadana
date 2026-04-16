# ========== IMPORTS ==========
import sys
from pathlib import Path
from math import log
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Importar clase base de georeferenciación
sys.path.append(str(Path(__file__).parent))
from georeferencia import GeoreferenciaMapa

# Importar optimizador LoRaWISEP
opt_dir = Path(__file__).parent / 'optimizacion_red'
# Priorizar el directorio de optimización en sys.path
sys.path.insert(0, str(opt_dir))
from kmeans import LoRaWISEPOptimization
from ga import LoRaWISEPGAElbow
from metricas import LoRaWANMetricsEvaluator

class GeoreferenciaRedLoRaWAN(GeoreferenciaMapa):

    """
    Extensión de GeoreferenciaMapa que integra optimización de red LoRaWAN
    
    Hereda toda la funcionalidad de georeferenciación y añade:
    - Generación de nodos IoT desde datos de delitos
    - Optimización de posicionamiento de gateways LoRaWAN
    - Visualización de red completa con cobertura
    """
    
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda):
        """
        Args:
            archivos_especificos: lista de tuplas (archivo.csv, categoría)
            PESOS_DELITOS: diccionario con pesos por tipo de delito
            mejorcelda: tamaño óptimo de celda hexagonal (metros)
        """
        # Inicializar clase base
        super().__init__(archivos_especificos, PESOS_DELITOS, mejorcelda)
        
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
            print(f"  cero nodos IoT generados")
        
     
        self.nodos_iot = np.array(nodos_coords)
        print(f"  ✓ {len(self.nodos_iot)} nodos IoT generados")
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
        print("\n Optimizando posicionamiento de gateways LoRaWAN...")
        
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
        
        print(f"\n  ✓ {len(self.gateways)} gateways optimizados")
        
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
        
        print(f"  • K óptimo determinado: {optimal_k}")

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
        
        print(f"\n  ✓ Posiciones de nodos IoT guardadas en: {output_path}")

        return self.gateways, self.assignments
    
 
    def comparar_metodos(self, export_csv=False):
            """Usa metricas.LoRaWANMetricsEvaluator para evaluar KMeans vs GA y decidir cuál es mejor."""
            if getattr(self, 'gateways_kmeans', None) is None or getattr(self, 'gateways_ga', None) is None:
                raise ValueError("Ejecuta optimizar_gateways_kmeans() y optimizar_gateways_ga() antes de comparar.")
            
            print("\n Evaluando métricas: K-Means vs GA")
            
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
                print(f"  {name:<15}: {v_km:8.2f} {mark_km} | {v_ga:8.2f} {mark_ga}")
            
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
            
            print(f"\n Método seleccionado para visualización final: {mejor} (PDR: KMeans={pdr_km:.2f}%, GA={pdr_ga:.2f}%)")
    
            print("\n" + "="*70)
            print(" COMPARACIÓN DE POSICIONES DE GATEWAYS")
            print("="*70)
            
            # Mostrar comparación gráfica lado a lado
            try:
                self.visualizar_red_completa_comparativa()
            except Exception as e:
                print(f"⚠ No se pudo mostrar la comparación gráfica: {e}")
            
            return {'kmeans': metrics_km, 'ga': metrics_ga, 'mejor': mejor}
    
    def visualizar_red_completa_comparativa(self, mostrar_conexiones=False, guardar=True):
        """
        Muestra en un mismo mapa de Cali:
         - el mapa de calor (grid_cali)
         - los nodos IoT
         - los gateways obtenidos por K-Means (triángulos magenta)
         - los gateways obtenidos por GA (círculos negros)
        """
        print("\n Visualización comparativa de gateways (K-Means vs GA)...")

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
            print(f"✓ Comparativa guardada: {outname}")

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
            print(f"✓ Mapa guardado: {outname}")
        
        #plt.show()
        

    def gtw_distance(self, factor_densidad=1.0, max_intentos=100,
                     graficar=True, guardar=True, nombre_mapa='qos_logdistance_riesgo.png'):

        print("\n Recalculando nodos IoT desde grid_cali con riesgo...")
        nodos_coords = []

        celdas_activas = self.grid_cali
        for _, row in celdas_activas.iterrows():
            indice = float(row.get('indice_inseguridad', 0.0))

            # Sin límite fijo de nodos por celda
            n_nodos = max(1, int(np.ceil(indice * factor_densidad)))

            poly = row['geometry']
            minx, miny, maxx, maxy = poly.bounds

            for _ in range(n_nodos):
                intentos = 0
                while intentos < max_intentos:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    punto = Point(x, y)
                    if poly.contains(punto):
                        nodos_coords.append([x, y])
                        break
                    intentos += 1

                if intentos >= max_intentos:
                    centroide = poly.centroid
                    nodos_coords.append([centroide.x, centroide.y])

        self.nodos_iot_riesgo = np.array(nodos_coords)
        print(f"  • Nodos recalculados: {len(self.nodos_iot_riesgo)}")

        # Crear DataFrame con las posiciones en metros
        df_nodos = pd.DataFrame(
            self.nodos_iot_riesgo,
            columns=['x', 'y']
        )
        
        # Agregar ID de nodo
        df_nodos.insert(0, 'id', range(1, len(self.nodos_iot_riesgo) + 1))

        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_simple = output_path / "nodos.csv"

        df_nodos.to_csv(out_path_simple, index=False)
        
        print(f"\n  ✓ Posiciones de nodos IoT guardadas en: {output_path}")
        
        nodos = np.asarray(self.nodos_iot_riesgo)
        
        gateways = np.asarray(self.gateways_ga)

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelent',
            np.where(rssi_dbm > -100, 'Good',
                     'Regular')
        )


        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
    
        clasificacion = []

        celdas_con_sensores = self.grid_cali
        for i, nodo in enumerate(nodos):
            punto = Point(nodo[0], nodo[1])
            clasi = "Sin datos"
            for _, celda in celdas_con_sensores.iterrows():
                if celda['geometry'].contains(punto):
                    nivel = int(celda['nivel_inseguridad'])
                    clasi = colores_nivel[nivel]
                    break
            clasificacion.append(clasi)

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'clasificacion': clasificacion,
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })
        
        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_simple = output_path / "informacion_nodos_riesgo.csv"

        df.to_csv(out_path_simple, index=False)
        

        # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            }
        }

        print(f"  • Nodos dentro del rango: {resumen['calidad_nodos']['Buena']+resumen['calidad_nodos']['Excelente']}")

        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                                        
                    # Contar sensores por nivel usando las asignaciones y celdas
                    # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                    sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    
                    celdas_con_sensores = self.grid_cali
                    for i, nodo in enumerate(nodos):
                        punto = Point(nodo[0], nodo[1])
                        for _, celda in celdas_con_sensores.iterrows():
                            if celda['geometry'].contains(punto):
                                nivel = int(celda['nivel_inseguridad'])
                                sensores_por_nivel[nivel] += 1
                                break
                except Exception:
                    pass
            
            print("\n   Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Clasificacion':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)
            total_celdas = 0
            total_sensores = 0
            
            # Lista para guardar los datos de cada nivel
            datos_niveles = []
            
            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                total_celdas += count_celdas
                total_sensores += count_sensores
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
                datos_niveles.append({
                    'Nivel': nivel,
                    'Clasificacion': colores_nivel[nivel],
                    'Celdas': count_celdas,
                    'Sensores': count_sensores
                })
            
            print("  " + "-"*65)
            
            df_nodos_riesgo = pd.DataFrame(datos_niveles)
            
            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "celdas_nodos_riesgo.csv"
            
            df_nodos_riesgo.to_csv(out_path_simple, index=False)
            
            sensores_por_celda = []
            for _, celda in self.grid_cali.iterrows():
                poly = celda['geometry']
                count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_riesgo)
                sensores_por_celda.append(count)
            
            if len(sensores_por_celda) > 0:
                y = np.array(sensores_por_celda, dtype=float)
                n = len(y)
                y_sorted = np.sort(y)
                suma_total = float(np.sum(y_sorted))
                if suma_total == 0:
                    gini = 0.0
                else:
                    i = np.arange(1, n + 1, dtype=float)
                    gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                    gini = max(0.0, min(1.0, gini))
            else:
                gini = 0.0
            
            print(f"Gini total de sensores en el mapa: {gini:.4f}")
            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")

            color_map = {
                'Excellent': "green",
                'Good': "yellow",
                'Regular': "#f05209",
                'Excelente': "green",
                'Buena': "yellow",
                'Excelent': "green"
            }
  
            df = pd.DataFrame(
            self.nodos_iot_riesgo,
            columns=['x', 'y']
            )
        
            # Agregar ID de nodo
            df.insert(0, 'id', range(1, len(self.nodos_iot_riesgo) + 1))

            # Filtrar nodos dentro del cuadrado
            x_ini = 727000
            y_ini = 867300
            ancho = 4000
            alto = 4000
            rect = Rectangle((x_ini, y_ini), ancho, alto, linewidth=5, edgecolor='deepskyblue', facecolor='aqua', alpha=0.8, zorder=4)
            ax.add_patch(rect)
            x_fin = x_ini + ancho
            y_fin = y_ini + alto
            nodos_cuadrado = df[(df['x'] >= x_ini) & (df['x'] <= x_fin) & (df['y'] >= y_ini) & (df['y'] <= y_fin)]
            if len(nodos_cuadrado) == 1000:
                            print(f"✓ Mil nodos dentro")
            else:
                print(f"No hay mil nodos: {len(nodos_cuadrado)}")

            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_cuadrado = output_path / "nodos_cuadrado.csv"
            nodos_cuadrado.to_csv(out_path_cuadrado, index=False)


            calidad_map = {
                'Excelente': 'Excellent',
                'Buena': 'Good',
                'Excelent': 'Excellent'
            }
            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excelente'], markersize=8, label=f"Excelente ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Buena'], markersize=8, label=f"Buena ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"Gateways GA ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title('Number of Nodes by Log-Distance using Risk Model', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }
    

    def gtw_distance_uniforme(self, max_intentos=100,
                     graficar=True, guardar=True, nombre_mapa='qos_logdistance_uniforme.png'):


        print("\n Recalculando nodos IoT desde grid_cali uniforme...")
        nodos_coords = []
        
        celdas = self.grid_cali
        #de forma uniforme
        M = len(celdas)
        n_base = 7825 // M
        resto = 7825 - (n_base * M)
        sensores = np.full(M, n_base)
        if resto > 0:
            indices = np.random.choice(M, size=resto, replace=False)
            sensores[indices] += 1

        #de forma aleatoria
        M = len(celdas)
        # Para cada nodo, elige una celda al azar
        #indices = np.random.choice(M, size=7828, replace=True)
        for idx, (_, row) in enumerate(celdas.iterrows()):
            n_nodos = sensores[idx]
            poly = row['geometry']
            minx, miny, maxx, maxy = poly.bounds

            for _ in range(n_nodos):
                intentos = 0
                while intentos < max_intentos:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    punto = Point(x, y)
                    if poly.contains(punto):
                        nodos_coords.append([x, y])
                        break
                    intentos += 1
                if intentos >= max_intentos:
                    centroide = poly.centroid
                    nodos_coords.append([centroide.x, centroide.y])
        
        self.nodos_iot_uniforme = np.array(nodos_coords)
        print(f"  • Nodos recalculados: {len(self.nodos_iot_uniforme)}")
        
        nodos = np.asarray(self.nodos_iot_uniforme)
        
        gateways = np.asarray(self.gateways_ga)

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelente',
            np.where(rssi_dbm > -100, 'Buena',
                     'Regular')
        )

        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })

        # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            }
        }

        print(f"  • Nodos dentro del rango: {resumen['calidad_nodos']['Buena']+resumen['calidad_nodos']['Excelente']}")
        print(f"  • Gateways GA: {resumen['n_gateways_ga']}")
        """
        print("  • Distancia (m): "
              f"min={resumen['distancia_m']['min']:.2f}, "
              f"mean={resumen['distancia_m']['mean']:.2f}, "
              f"max={resumen['distancia_m']['max']:.2f}")
        print("  • Path Loss (dB): "
              f"min={resumen['path_loss_db']['min']:.2f}, "
              f"mean={resumen['path_loss_db']['mean']:.2f}, "
              f"max={resumen['path_loss_db']['max']:.2f}")
        print("  • RSSI (dBm): "
              f"min={resumen['rssi_dbm']['min']:.2f}, "
              f"mean={resumen['rssi_dbm']['mean']:.2f}, "
              f"max={resumen['rssi_dbm']['max']:.2f}")
        print("  • Calidad nodos: "
              f"Excelente={resumen['calidad_nodos']['Excelente']}, "
              f"Buena={resumen['calidad_nodos']['Buena']}, "
              f"Regular={resumen['calidad_nodos']['Regular']}"
              )
        """

        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                    
                    # Mostrar conteo de celdas por nivel de inseguridad
                    colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
                    
                    # Contar sensores por nivel usando las asignaciones y celdas
                    # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                    sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    
                    celdas_con_sensores = self.grid_cali
                    for i, nodo in enumerate(nodos):
                        punto = Point(nodo[0], nodo[1])
                        for _, celda in celdas_con_sensores.iterrows():
                            if celda['geometry'].contains(punto):
                                nivel = int(celda['nivel_inseguridad'])
                                sensores_por_nivel[nivel] += 1
                                break
                except Exception:
                    pass
            
            print("\n   Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Color':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)
            total_celdas = 0
            total_sensores = 0

            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                total_celdas += count_celdas
                total_sensores += count_sensores
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
            print("  " + "-"*65)
            
            sensores_por_celda = []
            for _, celda in self.grid_cali.iterrows():
                poly = celda['geometry']
                count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_uniforme)
                sensores_por_celda.append(count)
            
            if len(sensores_por_celda) > 0:
                y = np.array(sensores_por_celda, dtype=float)
                n = len(y)
                y_sorted = np.sort(y)
                suma_total = float(np.sum(y_sorted))
                if suma_total == 0:
                    gini = 0.0
                else:
                    i = np.arange(1, n + 1, dtype=float)
                    gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                    gini = max(0.0, min(1.0, gini))
            else:
                gini = 0.0
            
            print(f"Gini total de sensores en el mapa: {gini:.4f}")

            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")

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
    
            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'], markersize=8, label=f"Excellent ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'], markersize=8, label=f"Good ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"GA Gateways ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            
            ax.set_title('Number of Nodes by Log-Distance using Uniform Base Mode', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }
    
    def gtw_distance_aleatorio(self, max_intentos=100,
                     graficar=True, guardar=True, nombre_mapa='qos_logdistance_aleatorio.png'):


        print("\n Recalculando nodos IoT desde grid_cali aleatorio...")
        nodos_coords = []
        
        celdas = self.grid_cali


        #de forma aleatoria
        M = len(celdas)
        # Para cada nodo, elige una celda al azar
        indices = np.random.choice(M, size=7825, replace=True)
        for idx in indices:
            row = celdas.iloc[idx]
            poly = row['geometry']
            minx, miny, maxx, maxy = poly.bounds

            intentos = 0
            while intentos < max_intentos:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    punto = Point(x, y)
                    if poly.contains(punto):
                        nodos_coords.append([x, y])
                        break
                    intentos += 1
            if intentos >= max_intentos:
                    centroide = poly.centroid
                    nodos_coords.append([centroide.x, centroide.y])
        
        self.nodos_iot_aleatorio = np.array(nodos_coords)
        print(f"  • Nodos recalculados: {len(self.nodos_iot_aleatorio)}")
        
        nodos = np.asarray(self.nodos_iot_aleatorio)
        
        gateways = np.asarray(self.gateways_ga)


        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelente',
            np.where(rssi_dbm > -100, 'Buena',
                     'Regular')
        )

        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })

        # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            }
        }

        print(f"  • Nodos dentro del rango: {resumen['calidad_nodos']['Buena']+resumen['calidad_nodos']['Excelente']}")
        """
        print("  • Distancia (m): "
              f"min={resumen['distancia_m']['min']:.2f}, "
              f"mean={resumen['distancia_m']['mean']:.2f}, "
              f"max={resumen['distancia_m']['max']:.2f}")
        print("  • Path Loss (dB): "
              f"min={resumen['path_loss_db']['min']:.2f}, "
              f"mean={resumen['path_loss_db']['mean']:.2f}, "
              f"max={resumen['path_loss_db']['max']:.2f}")
        print("  • RSSI (dBm): "
              f"min={resumen['rssi_dbm']['min']:.2f}, "
              f"mean={resumen['rssi_dbm']['mean']:.2f}, "
              f"max={resumen['rssi_dbm']['max']:.2f}")
        print("  • Calidad nodos: "
              f"Excelente={resumen['calidad_nodos']['Excelente']}, "
              f"Buena={resumen['calidad_nodos']['Buena']}, "
              f"Regular={resumen['calidad_nodos']['Regular']}"
              )
        """
        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)

                    # Mostrar conteo de celdas por nivel de inseguridad
                    colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
                    
                    # Contar sensores por nivel usando las asignaciones y celdas
                    # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                    sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    
                    celdas_con_sensores = self.grid_cali
                    for i, nodo in enumerate(nodos):
                        punto = Point(nodo[0], nodo[1])
                        for _, celda in celdas_con_sensores.iterrows():
                            if celda['geometry'].contains(punto):
                                nivel = int(celda['nivel_inseguridad'])
                                sensores_por_nivel[nivel] += 1
                                break
                except Exception:
                    pass
            
            print("\n  Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Color':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)
            total_celdas = 0
            total_sensores = 0
            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                total_celdas += count_celdas
                total_sensores += count_sensores
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
            print("  " + "-"*65)

            # Calcular el Gini total de sensores en todas las celdas
            sensores_por_celda = []
            for _, celda in self.grid_cali.iterrows():
                poly = celda['geometry']
                count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_aleatorio)
                sensores_por_celda.append(count)
            
            if len(sensores_por_celda) > 0:
                y = np.array(sensores_por_celda, dtype=float)
                n = len(y)
                y_sorted = np.sort(y)
                suma_total = float(np.sum(y_sorted))
                if suma_total == 0:
                    gini = 0.0
                else:
                    i = np.arange(1, n + 1, dtype=float)
                    gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                    gini = max(0.0, min(1.0, gini))
            else:
                gini = 0.0
            
            print(f"Gini total de sensores en el mapa: {gini:.4f}")
            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")
            print(f"  Celdas sin sensores (nivel 0): {sensores_por_nivel.get(0, 0)} sensores en {len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == 0])} celdas")

            color_map = {
                'Excelente': "green",
                'Buena': "yellow",
                'Regular': "#f05209",
            }
            colores_nodos = [color_map[c] for c in calidad]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excelente'], markersize=8, label=f"Excelente ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Buena'], markersize=8, label=f"Buena ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"Gateways GA ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title('Number of Nodes by Log-Distance using Random Base Model', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }

    def visualizar_gtw_distance_comparativo(self, factor_densidad=1.0,
                                            max_intentos=100, mostrar_conexiones=False,
                                            guardar=True, nombre_mapa='qos_logdistance_comparativo.png'):
        """
        Genera una sola figura con 3 subgráficas:
        - gtw_distance (riesgo)
        - gtw_distance_uniforme (uniforme)
        - gtw_distance_aleatorio (aleatorio)
        """
        print("\n Generating comparative figure (risk / uniform / random)...")

        # Ejecutar cálculos sin graficar ni guardar imágenes individuales
        res_riesgo = self.gtw_distance(
            factor_densidad=factor_densidad,
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )
        res_uniforme = self.gtw_distance_uniforme(
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )
        res_aleatorio = self.gtw_distance_aleatorio(
            max_intentos=max_intentos,
            graficar=False,
            guardar=False
        )

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

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, (titulo, nodos, assignments_modelo, calidad) in zip(axes, escenarios):
            # Mapa base
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)
            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                except Exception:
                    pass

            gateways = np.asarray(self.gateways_ga)
            nodos = np.asarray(nodos)

            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            if mostrar_conexiones:
                gw_sel = gateways[assignments_modelo]
                for i in range(len(nodos)):
                    ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                            color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'], markersize=8, label="Excellent"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'], markersize=8, label="Good"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label="Regular"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label="GA Gateways")
            ]
            ax.legend(handles=handles, loc='lower right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title(titulo, fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')

        plt.tight_layout()

        if guardar:
            images_dir = Path(__file__).resolve().parent / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            outname = images_dir / nombre_mapa
            plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Figura comparativa guardada: {outname}")

        #plt.show()

        return {
            'riesgo': res_riesgo,
            'uniforme': res_uniforme,
            'aleatorio': res_aleatorio
        }
    

    def visualizar_gtw_distance(self, res_uniforme, res_aleatorio, res_riesgo, nombre_mapa='qos_logdistance_comparativo.png'):

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

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, (titulo, nodos, assignments_modelo, calidad) in zip(axes, escenarios):
            # Mapa base
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)
            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                except Exception:
                    pass

            gateways = np.asarray(self.gateways_ga)
            nodos = np.asarray(nodos)

            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'], markersize=8, label="Excellent"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'], markersize=8, label="Good"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label="Regular"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label="GA Gateways")
            ]
            ax.legend(handles=handles, loc='lower right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title(titulo, fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')

        plt.tight_layout()

        images_dir = Path(__file__).resolve().parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        outname = images_dir / nombre_mapa
        plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figura comparativa guardada: {outname}")

        #plt.show()

        return {
            'riesgo': res_riesgo,
            'uniforme': res_uniforme,
            'aleatorio': res_aleatorio
        }

        
    def guardarmetricas(self):

        fig, ax = plt.subplots(figsize=(16, 12))

        # Mapa base si está disponible
        self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

        nodos_riesgo = np.asarray(self.nodos_iot_riesgo) if self.nodos_iot_riesgo is not None else np.empty((0, 2))
        gateways_riesgo = np.asarray(self.gateways_ga) if self.gateways_ga is not None else np.empty((0, 2))
        assignments_modelo_riesgo = getattr(self, 'assignments_logdistance_ga', None)

        if getattr(self, 'grid_cali', None) is not None:
            try:
                self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                                    
                # Contar sensores por nivel usando las asignaciones y celdas
                # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                
                celdas_con_sensores = self.grid_cali
                for i, nodo in enumerate(nodos_riesgo):
                    punto = Point(nodo[0], nodo[1])
                    for _, celda in celdas_con_sensores.iterrows():
                        if celda['geometry'].contains(punto):
                            nivel = int(celda['nivel_inseguridad'])
                            sensores_por_nivel[nivel] += 1
                            break
            except Exception:
                pass

        df = pd.DataFrame(
        self.nodos_iot_riesgo,
        columns=['x', 'y']
        )
    
        # Filtrar nodos dentro del cuadrado
        x_ini = 726000
        y_ini = 867300
        ancho = 4000
        alto = 4000
        rect = Rectangle((x_ini, y_ini), ancho, alto, linewidth=5, edgecolor='deepskyblue', facecolor='aqua', alpha=0.8, zorder=3)
        ax.add_patch(rect)
        x_fin = x_ini + ancho
        y_fin = y_ini + alto
        nodos_cuadrado = df[(df['x'] >= x_ini) & (df['x'] <= x_fin) & (df['y'] >= y_ini) & (df['y'] <= y_fin)]
        if len(nodos_cuadrado) == 1000:
                        print(f"✓ Mil nodos dentro")
        else:
            print(f"No hay mil nodos: {len(nodos_cuadrado)}")
        
        nodos_cuadrado.insert(0, 'id', range(0, len(nodos_cuadrado)))

        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_cuadrado = output_path / "nodos_cuadrado.csv"
        nodos_cuadrado.to_csv(out_path_cuadrado, index=False)

        # Guardar gateways en CSV adicional (solo dentro del cuadrado)
        df_gateways = pd.DataFrame(self.gateways_ga, columns=['x', 'y']) if self.gateways_ga is not None else pd.DataFrame(columns=['x', 'y'])
        if not df_gateways.empty:
            df_gateways = df_gateways[(df_gateways['x'] >= x_ini) & (df_gateways['x'] <= x_fin) &
                                      (df_gateways['y'] >= y_ini) & (df_gateways['y'] <= y_fin)]
        df_gateways.insert(0, 'id', range(0, len(df_gateways)))
        out_path_gateways = output_path / "gateways_cuadrado.csv"
        df_gateways.to_csv(out_path_gateways, index=False)


        if len(nodos_riesgo) > 0:
            ax.scatter(nodos_riesgo[:, 0], nodos_riesgo[:, 1], c='#f05209', s=22, alpha=0.75,
                        edgecolors='none', zorder=4, label=f'Nodos ({len(nodos_riesgo)})')

        if len(gateways_riesgo) > 0:
            ax.scatter(gateways_riesgo[:, 0], gateways_riesgo[:, 1], c='black', marker='^', s=220,
                        edgecolors='white', linewidth=1.2, zorder=3,
                        label=f'Gateways GA ({len(gateways_riesgo)})')

        # Conexiones suaves según asignación del modelo
        if assignments_modelo_riesgo is not None and len(gateways_riesgo) > 0:
            gw_sel = gateways_riesgo[assignments_modelo_riesgo]
            for i in range(len(nodos_riesgo)):
                ax.plot([nodos_riesgo[i, 0], gw_sel[i, 0]], [nodos_riesgo[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)


        ax.set_xlabel('X Coordinate (meters)', fontsize=14)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
        ax.set_title('Number of Nodes by Log-Distance using Risk Model', fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.25, linestyle='--')
        plt.tight_layout()

        images_dir = Path(__file__).resolve().parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        outname = images_dir / 'qos_logdistance_riesgo.png'
        plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Mapa QoS guardado: {outname}")

        #plt.show() 

        df = pd.read_csv("data_base/nodos_cuadrado.csv")

        nodos = df[['x','y']].to_numpy()
        
        df = pd.read_csv("data_base/gateways_cuadrado.csv")

        print(len(nodos))
        
        gateways = df[['x','y']].to_numpy()

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        # Calcular distancia usando fórmula euclidiana explícita
        gw_assigned = gateways[assignments_modelo]
        dist = np.sqrt((nodos[:, 0] - gw_assigned[:, 0])**2 + (nodos[:, 1] - gw_assigned[:, 1])**2)
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Estimar SNR y asignar Spreading Factor (SF)
        noise_floor_dbm = -120.0 + np.random.uniform(-3.0, 3.0, size=len(nodos))
        snr_db = rssi_dbm - noise_floor_dbm
        sf = np.select(
            [snr_db > 7.5, snr_db > 5.0, snr_db > 2.5, snr_db > 0.0, snr_db > -2.5],
            [7, 8, 9, 10, 11],
            default=12
        )

        # Potencia de transmisión (tp) por nodo
        tp_dbm = np.full(len(nodos), tx_power_dbm)

        # Estimar packets_sent a partir de una PDR logística
        pdr = 100.0 / (1.0 + np.exp(-(snr_db - 2.0)))
        total_packets = 100.0
        packets_sent = np.maximum(1, np.round(total_packets * (pdr / 100.0))).astype(int)

        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(0, len(nodos)),
            'id_gateway': assignments_modelo,
            'pos_x': nodos[:, 0],
            'pos_y': nodos[:, 1],
            'distancia_m': dist,
            'sf': sf.astype(int),
            'tp': tp_dbm,
            'packets_sent': packets_sent
        })
        
        
        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_simple = output_path / "info_milnodos_riesgo.csv"

        df.to_csv(out_path_simple, index=False)
        
    
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
        
        print("\n✓ Georeferenciación completada")

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
    
    print(f"\n Tamaño de celda óptimo: {mejorcelda} m")
    
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
    red_cali = GeoreferenciaRedLoRaWAN(archivos_especificos, PESOS_DELITOS, mejorcelda)
    
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

    if not os.path.exists(pipeline_nodos):
        # Ejecuta el pipeline
        red_cali.visualizar_gtw_distance_comparativo()
        # Guarda el resultado (si la función retorna algo)
        with open(pipeline_nodos, "wb") as f:
            pickle.dump(red_cali, f)
    else:
        print("El pipeline ya fue ejecutado, no se recalcula.")
        with open(pipeline_nodos, "rb") as f:
            red_cali = pickle.load(f)

    red_cali.guardarmetricas()