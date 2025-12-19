# ========== IMPORTS ==========
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from shapely.geometry import Point
from matplotlib.lines import Line2D

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
        self.gateways = None
        self.assignments = None
        self.optimizer = None
        
    def generar_nodos_iot_desde_delitos(self, estrategia='ponderada', min_nivel=3, num_nodos_exacto=None):
        """
        Genera posiciones de nodos IoT basándose en datos de delitos
        
        Estrategias:
        - 'ponderada': coloca más nodos en zonas de alta inseguridad
        - 'uniforme': distribución uniforme por celda
        - 'critica': solo en zonas críticas (nivel >= min_nivel)
        - 'exacto': número exacto de nodos distribuidos proporcionalmente
        
        Args:
            estrategia: método de distribución de nodos
            min_nivel: nivel mínimo de inseguridad para colocar nodos (1-5)
            num_nodos_exacto: número exacto de nodos a generar (None = automático)
            
        Returns:
            nodos_coords: array (N, 2) con coordenadas de nodos IoT
        """
        print("\n[🔧] Generando nodos IoT desde datos de delitos...")
        
        if self.grid_cali is None:
            raise ValueError("Ejecuta ejecutar_pipeline_georeferenciacion() primero")
        
        nodos_coords = []
        
        if num_nodos_exacto is not None:
            # Modo EXACTO: distribuir N nodos proporcionalmente al índice de inseguridad
            print(f"  • Modo: Número exacto de nodos = {num_nodos_exacto}")
            
            # Filtrar celdas con datos
            celdas_activas = self.grid_cali[self.grid_cali['nivel_inseguridad'] > 0].copy()
            
            if len(celdas_activas) == 0:
                raise ValueError("No hay celdas con datos de inseguridad")
            
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
        
        elif estrategia == 'ponderada':
            # Colocar nodos proporcionales al índice de inseguridad
            for idx, row in self.grid_cali.iterrows():
                nivel = row['nivel_inseguridad']
                if nivel == 0:
                    continue
                
                # Número de nodos: más en zonas críticas
                if nivel == 5:  # Muy alto
                    n_nodos = 5
                elif nivel == 4:  # Alto
                    n_nodos = 3
                elif nivel == 3:  # Medio
                    n_nodos = 2
                else:  # Bajo, Muy bajo
                    n_nodos = 1
                
                poly = row['geometry']
                minx, miny, maxx, maxy = poly.bounds
                
                for _ in range(n_nodos):
                    while True:
                        x = np.random.uniform(minx, maxx)
                        y = np.random.uniform(miny, maxy)
                        punto = Point(x, y)
                        if poly.contains(punto):
                            nodos_coords.append([x, y])
                            break
        
        elif estrategia == 'uniforme':
            for idx, row in self.grid_cali.iterrows():
                if row['nivel_inseguridad'] > 0:
                    centroid = row['geometry'].centroid
                    nodos_coords.append([centroid.x, centroid.y])
        
        elif estrategia == 'critica':
            for idx, row in self.grid_cali.iterrows():
                if row['nivel_inseguridad'] >= min_nivel:
                    poly = row['geometry']
                    minx, miny, maxx, maxy = poly.bounds
                    
                    n_nodos = 3 if row['nivel_inseguridad'] == 5 else 2
                    
                    for _ in range(n_nodos):
                        while True:
                            x = np.random.uniform(minx, maxx)
                            y = np.random.uniform(miny, maxy)
                            punto = Point(x, y)
                            if poly.contains(punto):
                                nodos_coords.append([x, y])
                                break
        
        self.nodos_iot = np.array(nodos_coords)
        
        print(f"  ✓ {len(self.nodos_iot)} nodos IoT generados")
        print(f"  • Estrategia: {estrategia if num_nodos_exacto is None else 'exacto'}")
        if estrategia == 'critica':
            print(f"  • Nivel mínimo: {min_nivel}")
        
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
        print("\n[⚙️] Optimizando posicionamiento de gateways LoRaWAN...")
        
        if self.nodos_iot is None:
            raise ValueError("Ejecuta generar_nodos_iot_desde_delitos() primero")
        
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
        if self.nodos_iot is None:
            raise ValueError("Ejecuta generar_nodos_iot_desde_delitos() primero")
        
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
        
        print(f"\n  ✓ {len(self.gateways)} gateways optimizados con Algoritmo Genético")
        print(f"  • K óptimo determinado: {optimal_k}")

        return self.gateways, self.assignments
 
    def comparar_metodos(self, export_csv=True):
            """Usa metricas.LoRaWANMetricsEvaluator para evaluar KMeans vs GA y decidir cuál es mejor."""
            if getattr(self, 'gateways_kmeans', None) is None or getattr(self, 'gateways_ga', None) is None:
                raise ValueError("Ejecuta optimizar_gateways_kmeans() y optimizar_gateways_ga() antes de comparar.")
            
            print("\n[🔎] Evaluando métricas: K-Means vs GA")
            
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
            
            print(f"\n➡️ Método seleccionado para visualización final: {mejor} (PDR: KMeans={pdr_km:.2f}%, GA={pdr_ga:.2f}%)")
    
            print("\n" + "="*70)
            print("📍 COMPARACIÓN DE POSICIONES DE GATEWAYS")
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
        print("\n[📊] Visualización comparativa de gateways (K-Means vs GA)...")

        if getattr(self, 'cali', None) is None or getattr(self, 'grid_cali', None) is None:
            raise ValueError("Ejecuta la georreferenciación antes (ejecutar_pipeline_georeferenciacion).")

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

        plt.show()

    def visualizar_red_completa(self, mostrar_conexiones=True, guardar=True, name=None):
        """
        Visualiza mapa de calor + nodos IoT + gateways LoRaWAN
        
        Args:
            mostrar_conexiones: si True, dibuja líneas nodo->gateway
            guardar: si True, guarda imagen PNG
        """
        print("\n[📊] Generando visualización completa de red LoRaWAN...")
        
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
        
        plt.show()
        
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
        
    def ejecutar_pipeline_red_lorawan(self, estrategia_nodos='ponderada', num_nodos=1000, metodo_optimizacion='kmeans', mostrar_conexiones=True):
        """
        Ejecuta el pipeline completo de optimización de red LoRaWAN
        
        Args:
            estrategia_nodos: 'ponderada', 'uniforme', 'critica' o None (si num_nodos especificado)
            num_nodos: número exacto de nodos IoT (None = automático según estrategia)
            auto_k_gateways: si True, determina K de gateways automáticamente
            mostrar_conexiones: si True, muestra conexiones nodo->gateway en mapa
        """
        print("\n" + "="*70)
        print("FASE 2: OPTIMIZACIÓN DE RED LoRaWAN")
        print("="*70)
        
        # Generar nodos IoT
        if num_nodos is not None:
            self.generar_nodos_iot_desde_delitos(num_nodos_exacto=num_nodos)
        else:
            self.generar_nodos_iot_desde_delitos(estrategia=estrategia_nodos)
        
        # Optimizar gateways
        if metodo_optimizacion == 'ga':
            self.optimizar_gateways_ga()
        else:
            self.optimizar_gateways_kmeans()
                    
        # Visualizar usando el método seleccionado (comparar_metodos ya ajustó self.gateways/self.assignments)
        self.visualizar_red_completa(mostrar_conexiones=mostrar_conexiones, guardar=True)
                    
        print("\n" + "="*70)
        print("✅ PROCESO COMPLETO FINALIZADO")
        print("="*70)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "resultados_optimizacion"
    
    with open(resultados_dir / "mejorcelda.json") as f:
        config = json.load(f)
        mejorcelda = config['mejorcelda']
    
    print(f"\n🔧 Tamaño de celda óptimo: {mejorcelda} m\n")
    
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
    
    # ========== FASE 1: GEOREFERENCIACIÓN ==========
    red_cali.ejecutar_pipeline_georeferenciacion()
    
    # ========== FASE 2: RED LoRaWAN ==========
    red_cali.generar_nodos_iot_desde_delitos(num_nodos_exacto=1000)
    
    # Ejecutar K-Means y guardar su mapa
    red_cali.optimizar_gateways_kmeans()
    # for visualization use kmeans results
    red_cali.gateways = red_cali.gateways_kmeans
    red_cali.assignments = red_cali.assignments_kmeans
    red_cali.visualizar_red_completa(mostrar_conexiones=True, guardar=True, name='red_lorawan_kmeans.png')
    
    # Ejecutar GA y guardar su mapa
    red_cali.optimizar_gateways_ga()
    # for visualization use ga results
    red_cali.gateways = red_cali.gateways_ga
    red_cali.assignments = red_cali.assignments_ga
    red_cali.visualizar_red_completa(mostrar_conexiones=True, guardar=True, name='red_lorawan_ga.png')
    
    # Comparativa: ambos sets sobre el mapa de Cali
    red_cali.visualizar_red_completa_comparativa(mostrar_conexiones=True, guardar=True)

