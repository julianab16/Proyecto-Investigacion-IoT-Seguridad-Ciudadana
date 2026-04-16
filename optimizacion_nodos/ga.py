"""
GA Algorithm for IoT Node Optimization (MONO-OBJETIVO)
Objetivo único: Maximizar f = w_1 \cdot \hat{PDR} - w_2 \cdot \hat{Energy} - w_3 \cdot \hat{Latency}
Variables: x, y (posición), SF, TP, λ (packets_sent)

Salida: Una única solución óptima con las mejores posiciones de los nodos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import os
from typing import Tuple, List, Dict
import sys
from pathlib import Path
import pickle

# Agregar path para importar desde raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuración de rutas
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # optimizacion_nodos/
PROJECT_PATH = os.path.dirname(BASE_PATH)  # Raíz del proyecto
DATA_PATH = os.path.join(PROJECT_PATH, 'data_base')
CSV_FILE = os.path.join(DATA_PATH, 'info_nodos_lorawan.csv')

# =====================================================================
# PESOS PARA LA FUNCIÓN OBJETIVO MONO-OBJETIVO
# =====================================================================
W_PDR = 1.0      # Peso para PDR (maximizar)
W_ENERGY = 1.0   # Peso para Energy (minimizar)
W_LATENCY = 1.0  # Peso para Latency (minimizar)

# Normalizar (escalas típicas)
MAX_ENERGY  = 51 * 0.001 * 2.0 * 2.0 * 5   # SF12, TP27, λ=5 → 1.02 mJ
MAX_LATENCY = 300 + (5-1)*1000              # SF12, λ=5 → 4300 ms

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
        """Calcula consumo de energía (mJ)"""
        if lambda_packets == 0:
            return 0.0
        
        sf_factor = FitnessCalculator.SF_ENERGY_FACTOR.get(sf, 1.0)
        tp_factor = FitnessCalculator.TP_ENERGY_FACTOR.get(tp, 1.0)
        
        energy_per_tx = payload_bytes * FitnessCalculator.ENERGY_PER_BYTE * sf_factor * tp_factor
        total_energy = energy_per_tx * lambda_packets
        
        return float(total_energy)
    
    @staticmethod
    def calculate_latency(sf: int, lambda_packets: int, 
                          ) -> float:
        """Calcula latencia (ms) - Time on Air REAL + duty cycle """
        sf_latency = FitnessCalculator.SF_LATENCY_FACTOR.get(sf, 2302)
        
        if lambda_packets > 1:
            wait_time = (lambda_packets - 1) * 1000
        else:
            wait_time = 0
        
        total_latency = sf_latency + wait_time
        
        return float(total_latency)

# =====================================================================
# CONFIGURACIÓN DE DEAP PARA GA MONO-OBJETIVO
# =====================================================================

# Eliminar creadores previos si existen
if hasattr(creator, "FitnessSingle"):
    del creator.FitnessSingle
if hasattr(creator, "Individual"):
    del creator.Individual

# Crear clases de fitness y individuo (mono-objetivo, weights=(1.0,) para maximizar)
creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessSingle)

toolbox = base.Toolbox()

# =====================================================================
# ATRIBUTOS Y RESTRICCIONES
# =====================================================================

SF_RANGE = list(range(7, 13))  # 7 a 12
TP_RANGE = [14, 20, 27]  # dBm

# Variables globales para límites
GLOBAL_XMIN = None
GLOBAL_XMAX = None
GLOBAL_YMIN = None
GLOBAL_YMAX = None
GLOBAL_NODE_X = None
GLOBAL_NODE_Y = None

def init_individual():
    """Inicializa un individuo: [x, y, SF, TP, lambda_packets]"""
    base_x = GLOBAL_NODE_X if GLOBAL_NODE_X is not None else 750000
    base_y = GLOBAL_NODE_Y if GLOBAL_NODE_Y is not None else 870000
    
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
# FUNCIÓN DE EVALUACIÓN MONO-OBJETIVO
# =====================================================================

def evaluate_node(individual: list, node_data: dict) -> Tuple[float,]:
    """
    Evalúa un individuo con función mono-objetivo
    
    f = w_1 * \hat{PDR} - w_2 * \hat{Energy} - w_3 * \hat{Latency}
    
    Retorna tupla de un elemento: (fitness,)
    """
    x, y, sf, tp, lambda_packets = individual
    
    # Normalizar valores
    distance = np.sqrt((x - node_data['gw_x'])**2 + (y - node_data['gw_y'])**2)
    actual_lambda = int(lambda_packets)
    
    # Calcular métricas
    pdr = FitnessCalculator.calculate_pdr(sf, tp, distance, actual_lambda)
    energy = FitnessCalculator.calculate_energy(sf, tp, actual_lambda)
    latency = FitnessCalculator.calculate_latency(sf, actual_lambda, distance)
    
    energy_norm  = energy  / MAX_ENERGY   # rango [0, 1]
    latency_norm = latency / MAX_LATENCY  # rango [0, 1]
    pdr_norm = pdr  # [0, 1]

    # Función objetivo mono-objetivo
    fitness = W_PDR * pdr_norm - W_ENERGY * energy_norm - W_LATENCY * latency_norm
    
    return (float(fitness),)

# =====================================================================
# OPERADORES GENÉTICOS
# =====================================================================

def custom_mutation(individual: list) -> Tuple[list]:
    """Mutación gaussiana para x,y; aleatoria para SF, TP, lambda"""
    if random.random() < 0.5:  # Mutar posición
        individual[0] += random.gauss(0, 300)
        individual[1] += random.gauss(0, 300)
        
        if GLOBAL_XMIN is not None:
            individual[0] = np.clip(individual[0], GLOBAL_XMIN, GLOBAL_XMAX)
            individual[1] = np.clip(individual[1], GLOBAL_YMIN, GLOBAL_YMAX)
    
    if random.random() < 0.3:  # Mutar SF
        individual[2] = random.choice(SF_RANGE)
    
    if random.random() < 0.3:  # Mutar TP
        individual[3] = random.choice(TP_RANGE)
    
    if random.random() < 0.2:  # Mutar lambda
        individual[4] = random.randint(1, 5)
    
    return (individual,)

def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Crossover uniforme"""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)


# =====================================================================
# OPTIMIZADOR GA MONO-OBJETIVO
# =====================================================================

class GAOptimizerMonoObjective:
    """Optimizador GA mono-objetivo para nodos LoRaWAN"""
    
    def __init__(self, data: pd.DataFrame, population_size: int = 100,
                 generations: int = 50, mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8, cali=None):
        """Inicializa el optimizador"""
        global GLOBAL_XMIN, GLOBAL_XMAX, GLOBAL_YMIN, GLOBAL_YMAX
        
        self.cali = cali
        if cali is not None:
            self.xmin, self.ymin, self.xmax, self.ymax = cali.total_bounds
            GLOBAL_XMIN, GLOBAL_YMIN, GLOBAL_XMAX, GLOBAL_YMAX = self.xmin, self.ymin, self.xmax, self.ymax
 
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        toolbox.register("evaluate", lambda ind, nd: evaluate_node(ind, nd))
    
    def run_optimization_for_node(self, node_idx: int) -> List:
        """Optimiza un nodo específico y retorna el mejor individuo"""
        global GLOBAL_NODE_X, GLOBAL_NODE_Y
        
        node_data = self.data.iloc[node_idx].to_dict()
        GLOBAL_NODE_X = node_data.get('pos_x', 750000)
        GLOBAL_NODE_Y = node_data.get('pos_y', 870000)
        
        def eval_func(individual):
            return evaluate_node(individual, node_data)
        
        toolbox.register("evaluate", eval_func)
        
        # Crear población inicial
        pop = toolbox.population(n=self.population_size)
        
        # Evaluar población inicial
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Ejecutar algoritmo genético
        for gen in range(self.generations):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Aplicar crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Aplicar mutación
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluar individuos modificados
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Reemplazar población
            pop[:] = offspring
        
        # Retornar el mejor individuo
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind
    
    def run_optimization_global(self) -> Dict:
        """Optimiza todos los nodos y retorna resultados"""
        results = {}
        
        print("\n" + "="*70)
        print("GA MONO-OBJETIVO - OPTIMIZACIÓN DE NODOS IoT LoRaWAN")
        print("="*70)
        print(f"Función: f = {W_PDR}*PDR - {W_ENERGY}*Energy - {W_LATENCY}*Latency")
        print(f"Nodos: {len(self.data)}")
        print(f"Población: {self.population_size} | Generaciones: {self.generations}")
        print("="*70 + "\n")
        
        # Optimizar cada nodo
        for idx in range(len(self.data)):
            node = self.data.iloc[idx]
            print(f"[{idx+1}/{len(self.data)}] Nodo {int(node['node_id'])} (Gateway: {int(node['gateway_id'])})")
            
            # Ejecutar optimización
            best_individual = self.run_optimization_for_node(idx)
            
            x, y, sf, tp, lambda_val = best_individual
            pdr, energy, latency = (
                FitnessCalculator.calculate_pdr(sf, tp, np.sqrt((x - node['gw_x'])**2 + (y - node['gw_y'])**2), int(lambda_val)),
                FitnessCalculator.calculate_energy(sf, tp, int(lambda_val)),
                FitnessCalculator.calculate_latency(sf, int(lambda_val), np.sqrt((x - node['gw_x'])**2 + (y - node['gw_y'])**2))
            )
            optimized_distance = np.sqrt((x - node['gw_x'])**2 + (y - node['gw_y'])**2)
            
            results[int(node['node_id'])] = {
                'node_id': idx,
                'gateway_id': int(node['gateway_id']),
                'x': float(x),
                'y': float(y),
                'optimized_distance_m': float(optimized_distance),
                'SF': int(sf),
                'TP': int(tp),
                'lambda': int(lambda_val),
                'PDR': float(pdr),
                'Energy_mJ': float(energy),
                'Latency_ms': float(latency)
            }
            
            print(f"     ✓ PDR={pdr:.3f} | Energy={energy:.2f}mJ | Latency={latency:.1f}ms")
        
        return results

# =====================================================================
# GUARDAR RESULTADOS
# =====================================================================

def save_results(results: Dict, output_dir: str = None) -> pd.DataFrame:
    """Guarda resultados en CSV"""
    output_dir =  os.path.join(BASE_PATH, 'resultados')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear DataFrame con resultados
    data_list = []
    for node_id, node_results in results.items():
        data_list.append({
            'node_id': node_id,
            'gateway_id': node_results['gateway_id'],
            'x': node_results['x'],
            'y': node_results['y'],
            'optimized_distance_m': node_results['optimized_distance_m'],
            'SF': node_results['SF'],
            'TP': node_results['TP'],
            'lambda': node_results['lambda'],
            'PDR': node_results['PDR'],
            'Energy_mJ': node_results['Energy_mJ'],
            'Latency_ms': node_results['Latency_ms']
        })
    
    df_results = pd.DataFrame(data_list)
    
    # Guardar CSV
    output_path = os.path.join(output_dir, 'ga_best_solutions.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Resultados guardados en: {output_path}")
    
    return df_results

# =====================================================================
# PIPELINE DE EJECUCIÓN
# =====================================================================

def ejecutar_pipeline():
    """Pipeline principal de ejecución"""
    df_nodes = load_data(CSV_FILE)
    
    # Cargar georeferenciación (si existe)
    cali = None
    pickle_path = os.path.join(PROJECT_PATH, 'pipeline_georeferenciacion_done.pkl')
    try:
        with open(pickle_path, 'rb') as f:
            red_cali = pickle.load(f)
        cali = red_cali.cali
    except (FileNotFoundError, AttributeError, Exception) as e:
        print(f"⚠ No se pudo cargar georeferenciación: {e}")
        print("  Continuando sin restricciones geoespaciales...")
    
    # Crear y ejecutar optimizador
    optimizer = GAOptimizerMonoObjective(
        df_nodes,
        population_size=100,
        generations=50,
        mutation_rate=0.2,
        crossover_rate=0.8,
        cali=cali
    )
    
    # Optimizar
    results = optimizer.run_optimization_global()
    
    # Guardar resultados
    df_summary = save_results(results)
    
    # Estadísticas finales
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    print(f"Nodos optimizados: {len(results)}")
    print(f"PDR promedio: {df_summary['PDR'].mean():.3f}")
    print(f"Energy promedio: {df_summary['Energy_mJ'].mean():.2f} mJ")
    print(f"Latency promedio: {df_summary['Latency_ms'].mean():.1f} ms")
    print("="*70)
    print("\n✓ Optimización completada exitosamente")
    
    return df_summary

if __name__ == "__main__":

    resultado = ejecutar_pipeline()