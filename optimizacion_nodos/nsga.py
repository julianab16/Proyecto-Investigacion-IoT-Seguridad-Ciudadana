"""
NSGA-II Algorithm for IoT Node Optimization
Multiobjetivo:
- MAX PDR (Packet Delivery Rate)
- MIN Energy Consumption
- MIN Latency
Variables: SF, TP, λ (packets_sent)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
import random
import os
import json
from typing import Tuple, List
import sys
from pathlib import Path
import pickle

# Agregar path para importar desde raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))
from georeferencia_red import GeoreferenciaRedLoRaWAN

# Configuración de rutas
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # optimizacion_nodos/
PROJECT_PATH = os.path.dirname(BASE_PATH)  # Raíz del proyecto
DATA_PATH = os.path.join(PROJECT_PATH, 'data_base')
CSV_FILE = os.path.join(DATA_PATH, 'info_nodos_lorawan.csv')

# =====================================================================
# CARGAR DATOS
# =====================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos del CSV de nodos LoRaWAN"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

# =====================================================================
# FUNCIONES DE CÁLCULO DEL FITNESS
# =====================================================================

class FitnessCalculator:
    """Calcula las métricas de fitness para los nodos"""
    
    # Path Loss Model Parameters (LoRaWAN)
    PL_PARAMS = {
        7: {"a": 120.76, "b": 1.88},   # SF7
        8: {"a": 123.26, "b": 1.99},
        9: {"a": 125.60, "b": 2.01},
        10: {"a": 128.18, "b": 2.05},
        11: {"a": 130.54, "b": 2.08},
        12: {"a": 132.63, "b": 2.10},
    }
    
    # Energy consumption per transmission (mJ)
    ENERGY_PER_BYTE = 0.001  # mJ
    SF_ENERGY_FACTOR = {7: 1.0, 8: 1.2, 9: 1.4, 10: 1.6, 11: 1.8, 12: 2.0}
    TP_ENERGY_FACTOR = {14: 1.0, 20: 1.5, 27: 2.0}  # dBm
    
    # Latency parameters (ms) - Time on Air REAL para 51 bytes, BW=125kHz, CR=4/5
    # Valores corregidos basados en cálculos reales de LoRa (no aproximados)
    SF_LATENCY_FACTOR = {7: 102, 8: 194, 9: 349, 10: 617, 11: 1151, 12: 2302}
    # Sensibilidades reales del chip SX1276 (dBm)
    SF_SENSITIVITY = {
        7:  -123.0,
        8:  -126.0,
        9:  -129.0,
        10: -132.0,
        11: -134.5,
        12: -137.0
    }

    @staticmethod
    def calculate_pdr(sf: int, tp: float, distance: float, lambda_packets: int) -> float:
        if lambda_packets == 0:
            return 0.0
        if sf not in FitnessCalculator.PL_PARAMS:
            return 0.0

        # Calcular potencia recibida
        params = FitnessCalculator.PL_PARAMS[sf]
        path_loss = params["a"] + params["b"] * np.log10(distance + 1)
        rx_power = tp - path_loss

        # Comparar contra sensibilidad real del SF
        sensitivity = FitnessCalculator.SF_SENSITIVITY[sf]
        margin = rx_power - sensitivity  # dB sobre/bajo el umbral

        # PDR basado en margen de enlace
        if margin >= 0:
            # Señal por encima del umbral — PDR cercano a 1, degradado suavemente
            pdr = 1.0 / (1.0 + np.exp(-margin))
        else:
            # Señal bajo el umbral — PDR cae rápidamente
            pdr = 1.0 / (1.0 + np.exp(-margin * 0.5))

        return float(np.clip(pdr, 0, 1))    
    
    
    @staticmethod
    def calculate_energy(sf: int, tp: int, lambda_packets: int, payload_bytes: int = 51) -> float:
        """
        Calcula consumo de energía (mJ) basado en:
        - SF
        - TP
        - Distance
        - λ (packets_sent)
        - Payload size (default 51 bytes para header)
        
        ECUACIÓN:
        Energy(SF,TP,λ) = Payload × K_SF × K_TP × λ
        
        Donde:
        K_SF = factor energético del SF (1.0 a 2.0)
        K_TP = factor energético del TP (1.0 a 2.0)
        Payload = 51 bytes (default LoRaWAN)
        
        Retorna: Energía en mJ
        """
        if lambda_packets == 0:
            return 0.0
        
        # Energía por transmisión = (payload + overhead) * energy_per_byte
        sf_factor = FitnessCalculator.SF_ENERGY_FACTOR.get(sf, 1.0)
        tp_factor = FitnessCalculator.TP_ENERGY_FACTOR.get(tp, 1.0)
        
        energy_per_tx = payload_bytes * FitnessCalculator.ENERGY_PER_BYTE * sf_factor * tp_factor
        
        # Energía total
        total_energy = energy_per_tx * lambda_packets
        
        return float(total_energy)
    
    @staticmethod
    def calculate_latency(sf: int, lambda_packets: int, distance: float = 0) -> float:
        """
        Calcula latencia (ms) basada en:
        - SF
        - λ (packets_sent)
        
        ECUACIÓN (CORREGIDA):
        Latency(SF,λ) = L_ToA + T_wait
        
        Donde:
        L_ToA = Time on Air real del paquete (SF 7-12): 102 a 2302 ms
        T_wait = (λ-1) × 1000 ms (tiempo entre retransmisiones con duty cycle)
        
        NOTA: propagation_latency (distance / c) es insignificante (~12 ns para 3.7 km)
        y NO se incluye en el cálculo. La latencia real viene del ToA del paquete
        más el tiempo de espera de las ventanas RX (modelado como duty cycle).
        
        Retorna: Latencia en ms
        """
        # Latencia de transmisión basada en Time on Air real del SF (51 bytes, 125kHz, CR=4/5)
        sf_latency = FitnessCalculator.SF_LATENCY_FACTOR.get(sf, 2302)
        
        # Tiempo de espera entre retransmisiones (duty cycle de LoRaWAN)
        if lambda_packets > 1:
            wait_time = (lambda_packets - 1) * 1000  # 1 segundo entre paquetes
        else:
            wait_time = 0
        
        # Latencia TOTAL (sin propagation_latency que es negligible)
        total_latency = sf_latency + wait_time
        
        return float(total_latency)

# =====================================================================
# CONFIGURACIÓN DE DEAP PARA NSGA-II
# =====================================================================

# Eliminar creadores previos si existen
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

# Crear clases de fitness y individuo
# Pesos: (1, -1, -1) porque queremos MAXIMIZAR PDR y MINIMIZAR Energy y Latency
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# =====================================================================
# ATRIBUTOS Y RESTRICCIONES
# =====================================================================

# Rango de valores permitidos
SF_RANGE = list(range(7, 13))  # 7 a 12
TP_RANGE = [14, 20, 27]  # dBm (valores comunes en LoRaWAN)

# Genes: [x, y, SF, TP, lambda]
# Variables globales para límites (se establecen en NSGAOptimizer.__init__)
GLOBAL_XMIN = None
GLOBAL_XMAX = None
GLOBAL_YMIN = None
GLOBAL_YMAX = None
GLOBAL_NODE_X = None
GLOBAL_NODE_Y = None

def init_individual():
    """Inicializa un individuo: [x, y, SF, TP, lambda_packets]
    
    NOTA: Desviación inicial de 500m (σ=500) permite exploración robusta.
    Con σ=500, ~95% de los puntos están dentro de ±1000m del nodo original,
    dando cobertura significativa para ciudades grandes como Cali.
    """
    base_x = GLOBAL_NODE_X if GLOBAL_NODE_X is not None else 750000
    base_y = GLOBAL_NODE_Y if GLOBAL_NODE_Y is not None else 870000
    
    # Desviación de 500m: permite exploración de ±1km alrededor del nodo
    x = base_x + random.gauss(0, 500)
    y = base_y + random.gauss(0, 500)
    
    if GLOBAL_XMIN is not None:
        x = np.clip(x, GLOBAL_XMIN, GLOBAL_XMAX)
        y = np.clip(y, GLOBAL_YMIN, GLOBAL_YMAX)
    
    individual = creator.Individual([
        float(x),
        float(y),
        random.choice(SF_RANGE),
        random.choice(TP_RANGE),
        random.randint(1, 5)
    ])
    return individual

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =====================================================================
# FUNCIÓN DE EVALUACIÓN MULTIOBJETIVO
# =====================================================================

def evaluate_node(individual: list, node_data: dict) -> Tuple[float, float, float]:
    """
    Evalúa un individuo (configuración SF, TP, λ) para un nodo específico
    
    Retorna tupla: (PDR, -Energy, -Latency)
    Negativa en Energy y Latency porque queremos MINIMIZAR
    """
    x, y, sf, tp, lambda_packets = individual  # ← 5 genes ahora
        
    distance = np.sqrt((x - node_data['gw_x'])**2 + (y - node_data['gw_y'])**2)
    
    # lambda_packets siempre está entre 1-5 (de random.randint(1, 5) en init_individual)
    actual_lambda = int(lambda_packets)
    
    # Cálculos
    pdr = FitnessCalculator.calculate_pdr(sf, tp, distance, actual_lambda)
    energy = FitnessCalculator.calculate_energy(sf, tp, actual_lambda)
    latency = FitnessCalculator.calculate_latency(sf, actual_lambda, distance)
    
    # Retornar: (PDR a maximizar, -Energy a minimizar, -Latency a minimizar)
    return (float(pdr), float(-energy), float(-latency))

# =====================================================================
# OPERADORES GENÉTICOS
# =====================================================================

def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Crossover uniforme respetando restricciones"""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def custom_mutation(individual: list) -> tuple:
    """Mutación: x,y con desplazamiento gaussiano; SF,TP,lambda aleatorio
    
    NOTA: Desplazamiento de 300m (σ=300) en lugar de 60m permite:
    - Exploración local efectiva durante la evolución
    - ~95% de cambios dentro de ±600m (exploración significativa)
    - Convergencia gradual hacia soluciones mejores
    """
    if random.random() < 0.2:
        # Desplazamiento X: 300m (±600m en 95% de casos)
        dx = random.gauss(0, 300)
        individual[0] += dx
        if GLOBAL_XMIN is not None:
            individual[0] = np.clip(individual[0], GLOBAL_XMIN, GLOBAL_XMAX)
    
    if random.random() < 0.2:
        # Desplazamiento Y: 300m (±600m en 95% de casos)
        dy = random.gauss(0, 300)
        individual[1] += dy
        if GLOBAL_YMIN is not None:
            individual[1] = np.clip(individual[1], GLOBAL_YMIN, GLOBAL_YMAX)
    
    if random.random() < 0.2:
        individual[2] = random.choice(SF_RANGE)
    
    if random.random() < 0.2:
        individual[3] = random.choice(TP_RANGE)
    
    if random.random() < 0.2:
        individual[4] = random.randint(1, 5)
    
    return (individual,)

toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selNSGA2)

# =====================================================================
# ALGORITMO NSGA-II
# =====================================================================

class NSGAOptimizer:
    """Optimizador NSGA-II para configuración de nodos LoRaWAN"""
    
    def __init__(self, data: pd.DataFrame, pop_size: int = 50, generations: int = 100, cali=None):
        """
        Inicializa el optimizador
        
        Args:
            data: DataFrame con info de nodos
            pop_size: Tamaño de población
            generations: Número de generaciones
            cali: GeoDataFrame con límites espaciales
        """
        global GLOBAL_XMIN, GLOBAL_XMAX, GLOBAL_YMIN, GLOBAL_YMAX
        
        self.cali = cali
        if cali is not None:
            self.xmin, self.ymin, self.xmax, self.ymax = cali.total_bounds
            GLOBAL_XMIN, GLOBAL_YMIN, GLOBAL_XMAX, GLOBAL_YMAX = self.xmin, self.ymin, self.xmax, self.ymax
        else:
            self.xmin, self.ymin = 720414.01, 860572.64
            self.xmax, self.ymax = 735029.27, 879817.30
            GLOBAL_XMIN, GLOBAL_YMIN, GLOBAL_XMAX, GLOBAL_YMAX = self.xmin, self.ymin, self.xmax, self.ymax
        
        self.data = data
        self.pop_size = pop_size
        self.generations = generations
        self.history = tools.History()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
    
    def run_optimization_for_node(self, node_idx: int) -> list:
        """
        Ejecuta NSGA-II para un nodo específico usando eaMuPlusLambda
        
        Retorna: Lista de individuos en la frontera de Pareto
        """
        global GLOBAL_NODE_X, GLOBAL_NODE_Y
        
        node_data = self.data.iloc[node_idx].to_dict()
        GLOBAL_NODE_X = node_data.get('pos_x', 750000)
        GLOBAL_NODE_Y = node_data.get('pos_y', 870000)
        
        def eval_func(individual):
            return evaluate_node(individual, node_data)
        
        toolbox.register("evaluate", eval_func)
        
        pop = toolbox.population(n=self.pop_size)
        
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # NSGA-II con eaMuPlusLambda (estrategia mu+lambda)
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                                  mu=self.pop_size,
                                                  lambda_=self.pop_size,
                                                  cxpb=0.7, 
                                                  mutpb=0.3, 
                                                  ngen=self.generations, 
                                                  verbose=False)
        
        return pop
    
    def run_optimization_global(self) -> dict:
        """
        Ejecuta optimización multiobjetivo para todos los nodos
        
        Retorna: Diccionario con resultados por nodo
        """
        results = {}
        
        print("\n" + "="*70)
        print("NSGA-II OPTIMIZATION - IoT NODES")
        print("="*70)
        print(f"Nodos: {len(self.data)}")
        print(f"Población: {self.pop_size} | Generaciones: {self.generations}")
        print(f"Objetivos: MAX(PDR), MIN(Energy), MIN(Latency)")
        print("="*70 + "\n")
        
        # Optimizar cada nodo
        for idx in range(len(self.data)):
            node = self.data.iloc[idx]
            print(f"[{idx+1}/{len(self.data)}] NODO {int(node['node_id'])} (Gateway: {int(node['gateway_id'])})")
            print(f"     Distancia: {node['distance_m']:.1f}m | Paquetes: {int(node['packets_sent'])}")
            
            # Optimizar
            final_pop = self.run_optimization_for_node(idx)
            
            # Extraer frontera de Pareto (individuos no-dominados)
            pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
            
            results[int(node['node_id'])] = {
                'node_id': idx,
                'gateway_id': int(node['gateway_id']),
                'original_distance_m': float(node['distance_m']),  # ← Distancia ORIGINAL del nodo
                'original_packets': int(node['packets_sent']),
                'original_sf': int(node['sf']),
                'original_tp': int(node['tp']),
                'pareto_solutions': []
            }
            
            # Guardar soluciones de Pareto
            for ind in pareto_front:
                x, y, sf, tp, lambda_val = ind
                pdr, neg_energy, neg_latency = ind.fitness.values
                
                # Calcular distancia optimizada desde la nueva posición al gateway
                # Usar 'node' que está disponible en este scope (objeto del DataFrame)
                optimized_distance = np.sqrt((x - node['gw_x'])**2 + (y - node['gw_y'])**2)
                
                results[int(node['node_id'])]['pareto_solutions'].append({
                    'x': float(x),
                    'y': float(y),
                    'optimized_distance_m': float(optimized_distance),
                    'SF': int(sf),
                    'TP': int(tp),
                    'lambda': int(lambda_val),
                    'PDR': float(pdr),
                    'Energy_mJ': float(-neg_energy),
                    'Latency_ms': float(-neg_latency)
                })
            
            print(f"     ✓ Soluciones Pareto: {len(pareto_front)}")
        
        return results


def save_results(results: dict, output_dir: str = None):
    """Guarda resultados en CSV"""
    if output_dir is None:
        output_dir = os.path.join(BASE_PATH, 'resultados')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar CSV con resumen
    summary_data = []
    for node_id, node_results in results.items():
        for sol_idx, sol in enumerate(node_results['pareto_solutions']):
            summary_data.append({
                'node_id': node_id,
                'gateway_id': node_results['gateway_id'],
                'original_distance_m': node_results['original_distance_m'],
                'optimized_distance_m': round(sol['optimized_distance_m'], 2),
                'original_packets': node_results['original_packets'],
                'solution_idx': sol_idx + 1,
                'x': round(sol['x'], 2),
                'y': round(sol['y'], 2),
                'SF': sol['SF'],
                'TP': sol['TP'],
                'lambda': sol['lambda'],
                'PDR': round(sol['PDR'], 4),
                'Energy_mJ': round(sol['Energy_mJ'], 4),
                'Latency_ms': round(sol['Latency_ms'], 2)
            })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'nsga_pareto_solutions.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ CSV guardado: {csv_path}")
    
    return df_summary

def final_csv():
    """Extrae las mejores soluciones por objetivo (PDR, Energy, Latency) para cada nodo"""
    
    # Ruta relativa corregida
    csv_path = os.path.join(BASE_PATH, 'resultados', 'nsga_pareto_solutions.csv')
    
    df = pd.read_csv(csv_path)

    # Almacenar las mejores soluciones
    best_solutions = []
    
    # Agrupar por nodo
    for node_id, group in df.groupby('node_id'):
        
        # Mejor PDR
        idx = group['PDR'].idxmax() and group['Energy_mJ'].idxmin() and group['Latency_ms'].idxmin()
        best_row = df.loc[idx].to_dict()

        if (idx is not None):
            best_solutions.append(best_row)
    
    # Crear DataFrame con las mejores soluciones
    df_best = pd.DataFrame(best_solutions)
    
    # Guardar en CSV
    output_path = os.path.join(BASE_PATH, 'resultados', 'nsga_best_solutions.csv')
    df_best.to_csv(output_path, index=False)
    print(f"✓ Mejores soluciones guardadas: {output_path}")
    
    return df_best

def ejecutar_pipeline():
    df_nodes = load_data(CSV_FILE)

    # En nsga.py, antes de crear NSGAOptimizer
    pickle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline_georeferenciacion_done.pkl')
    with open(pickle_path, 'rb') as f:
        red_cali = pickle.load(f)

    # Ahora usa los datos del pickle
    cali = red_cali.cali
    grid_cali = red_cali.grid_cali

    # Y creas el optimizador
    optimizer = NSGAOptimizer(df_nodes, pop_size=80, generations=150, cali=cali)
        
    # Crear y ejecutar optimizador
    results = optimizer.run_optimization_global()
    
    # Guardar y visualizar resultados
    df_summary = save_results(results)
    
    # Mostrar estadísticas generales
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    print(f"Nodos optimizados: {len(results)}")
    
    total_solutions = sum(len(r['pareto_solutions']) for r in results.values())
    print(f"Total soluciones Pareto: {total_solutions}")
    print(f"Promedio soluciones/nodo: {total_solutions/len(results):.1f}")

    print("\n✓ Optimización completada exitosamente")

    return df_summary

if __name__ == "__main__":

    
    pipeline_nsga = "pipeline/pipeline_nodos_nsga.pkl"

    # Pipeline de georeferenciación
    if not os.path.exists(pipeline_nsga):
        # Ejecuta el pipeline
        ejecutar = ejecutar_pipeline()
        # Guarda el resultado (si la función retorna algo)
        with open(pipeline_nsga, "wb") as f:
            pickle.dump(ejecutar, f)
    else:
        print("El pipeline ya fue ejecutado, no se recalcula.")
        with open(pipeline_nsga, "rb") as f:
            red_cali = pickle.load(f) 

    final_csv()
