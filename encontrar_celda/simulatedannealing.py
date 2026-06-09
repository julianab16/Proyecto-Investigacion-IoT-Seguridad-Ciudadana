import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================
# CONFIGURACIÓN ESPECÍFICA PARA TU PROYECTO
# ============================================
df = pd.read_csv("SEMANA.csv")

CITY_AREA_KM2 = 619  # Área de tu ciudad en km²
TOTAL_RECORDS = len(df)  # Cantidad de datos en tu CSV
CITY_SIDE_KM = np.sqrt(CITY_AREA_KM2)  # ~24.88 km

# Cargar tus datos reales (ajusta la ruta y columnas)
# coords = pd.read_csv('tu_archivo.csv')[['latitud', 'longitud']].values

# SIMULACIÓN DE DATOS (reemplaza con tus datos reales)
np.random.seed(42)
coords = np.random.rand(TOTAL_RECORDS, 2) * [CITY_SIDE_KM, CITY_SIDE_KM]

def limpiar_coordenadas_miles(df, lat_col='lat', lon_col='lon'):
    def limpiar_numero(numero):
        numero_str = str(numero).strip()
        # Si tiene coma, es separador decimal
        numero_str = numero_str.replace(',', '')
        if numero_str.startswith("-7"):
            numero_str= numero_str[0:3] + '.' + numero_str[3:]
        else:
            numero_str = numero_str[0:1] + '.' + numero_str[1:]
        try:
            return numero_str
        except ValueError:
            return np.nan
    # Aplicar limpieza
    df[lat_col] = df[lat_col].apply(limpiar_numero)
    df[lon_col] = df[lon_col].apply(limpiar_numero)
    
    return df

if 'x' in df.columns and 'y' in df.columns:
    lat_col, lon_col = 'y', 'x'  # y=latitud, x=longitud
    print(f"Usando columnas: {lat_col} (latitud), {lon_col} (longitud)")
else:
    print("Error: No se encontraron columnas 'x' e 'y'")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    exit()

df = limpiar_coordenadas_miles(df, lat_col, lon_col)

if {'x', 'y'}.issubset(df.columns):
    coords = df[['x', 'y']].to_numpy()
else:
    np.random.seed(42)
    coords = np.random.rand(TOTAL_RECORDS, 2) * [CITY_SIDE_KM, CITY_SIDE_KM]
try:
    coords = coords.astype(float)
except Exception:
    df_coords = pd.DataFrame(coords, columns=['x', 'y'])
    # Eliminar caracteres no numéricos (excepto '-' y '.') y convertir a numérico
    df_coords = df_coords.replace({r'[^\d\.-]': ''}, regex=True)
    df_coords = df_coords.apply(pd.to_numeric, errors='coerce')
    df_coords = df_coords.dropna()
    if df_coords.empty:
        raise SystemExit("Error: No hay coordenadas numéricas válidas tras limpieza.")
    coords = df_coords.to_numpy()

print(f"📍 Área de la ciudad: {CITY_AREA_KM2} km²")
print(f"📊 Total de registros: {TOTAL_RECORDS}")
print(f"📏 Dimensión estimada: {CITY_SIDE_KM:.2f} km x {CITY_SIDE_KM:.2f} km")
print(f"🎯 Densidad promedio: {TOTAL_RECORDS/CITY_AREA_KM2:.2f} datos/km²\n")

# ============================================
# FUNCIÓN DE FITNESS (ENERGÍA)
# ============================================
def calculate_fitness(cell_size_m):
    """
    Calcula la función de energía/fitness para un tamaño de celda dado.
    Menor energía = mejor solución.
    """
    if cell_size_m <= 50 or cell_size_m > 10000:  # Límites físicos
        return 1e8
    
    city_side_m = CITY_SIDE_KM * 1000
    
    # Calcular grilla
    grid_x = int(np.ceil(city_side_m / cell_size_m))
    grid_y = int(np.ceil(city_side_m / cell_size_m))
    total_cells = grid_x * grid_y
    
    if total_cells == 0 or total_cells > 15000:
        return 1e8
    
    # Asignar puntos a celdas
    coords_m = coords * 1000
    idx_x = np.floor(coords_m[:, 0] / cell_size_m).astype(int)
    idx_y = np.floor(coords_m[:, 1] / cell_size_m).astype(int)
    
    # Validar índices
    valid_mask = (idx_x >= 0) & (idx_x < grid_x) & (idx_y >= 0) & (idx_y < grid_y)
    idx_x = idx_x[valid_mask]
    idx_y = idx_y[valid_mask]
    
    if len(idx_x) == 0:
        return 1e8
    
    # Contar puntos por celda
    cell_indices = list(zip(idx_x, idx_y))
    unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
    
    # Métricas
    occupied_cells = len(counts)
    occupied_ratio = occupied_cells / total_cells
    empty_cells = total_cells - occupied_cells
    
    # Estadísticas de distribución
    mean_density = np.mean(counts)
    variance = np.var(counts)
    std_dev = np.std(counts)
    
    # FUNCIÓN DE ENERGÍA (minimizar)
    # Componente 1: Penalizar celdas vacías
    energy_empty = empty_cells * 5
    
    # Componente 2: Penalizar alta varianza (queremos homogeneidad)
    energy_variance = variance * 2
    
    # Componente 3: Buscar densidad óptima (3-8 datos por celda)
    target_density = 5
    energy_density = abs(mean_density - target_density) * 50
    
    # Componente 4: Penalizar ratios de ocupación extremos
    if occupied_ratio < 0.3 or occupied_ratio > 0.9:
        energy_ratio = abs(occupied_ratio - 0.6) * 3000
    else:
        energy_ratio = 0
    
    # Componente 5: Penalizar desviación estándar alta
    energy_std = std_dev * 3
    
    # Energía total
    total_energy = (energy_empty + energy_variance + 
                   energy_density + energy_ratio + energy_std)
    
    return total_energy

# ============================================
# ALGORITMO SIMULATED ANNEALING
# ============================================
def simulated_annealing(initial_temp=5000, min_temp=0.1, cooling_rate=0.95, 
                       max_iterations=1000, initial_solution=None):
    """
    Implementación de Simulated Annealing para optimizar tamaño de celda.
    """
    # Temperatura inicial y final
    T = initial_temp
    T_min = min_temp
    alpha = cooling_rate
    
    # Solución inicial
    if initial_solution is None:
        current = random.uniform(100, 3000)  # Entre 100m y 3km
    else:
        current = initial_solution
    
    best = current
    current_energy = calculate_fitness(current)
    best_energy = current_energy
    
    # Historial para visualización
    history = {
        'iteration': [],
        'temperature': [],
        'current_solution': [],
        'current_energy': [],
        'best_solution': [],
        'best_energy': [],
        'accepted': []
    }
    
    iteration = 0
    accepted_moves = 0
    rejected_moves = 0
    
    print("🌡️  Iniciando Simulated Annealing...")
    print(f"Temperatura inicial: {T:.0f}")
    print(f"Temperatura final: {T_min:.2f}")
    print(f"Factor de enfriamiento: {alpha}\n")
    
    # Ciclo principal
    while T > T_min and iteration < max_iterations:
        # Generar solución vecina (perturbación adaptativa)
        perturbation = np.random.normal(0, T/10)  # Perturbación proporcional a T
        new = current + perturbation
        
        # Mantener dentro de límites
        new = np.clip(new, 100, 5000)
        
        # Calcular energías
        new_energy = calculate_fitness(new)
        delta_E = new_energy - current_energy
        
        # Criterio de aceptación (Metropolis)
        if delta_E < 0:  # Mejora
            accept = True
        else:  # Empeoramiento
            probability = math.exp(-delta_E / T)
            accept = random.random() < probability
        
        # Actualizar solución
        if accept:
            current = new
            current_energy = new_energy
            accepted_moves += 1
            
            # Actualizar mejor solución
            if current_energy < best_energy:
                best = current
                best_energy = current_energy
        else:
            rejected_moves += 1
        
        # Guardar historial
        history['iteration'].append(iteration)
        history['temperature'].append(T)
        history['current_solution'].append(current)
        history['current_energy'].append(current_energy)
        history['best_solution'].append(best)
        history['best_energy'].append(best_energy)
        history['accepted'].append(accept)
        
        # Enfriamiento
        T *= alpha
        iteration += 1
        
        # Progreso cada 100 iteraciones
        if iteration % 100 == 0:
            acceptance_rate = accepted_moves / (accepted_moves + rejected_moves) * 100
            print(f"Iter {iteration:4d} | T={T:7.2f} | "
                  f"Actual={current:.0f}m | Mejor={best:.0f}m | "
                  f"E={best_energy:.2f} | Aceptación={acceptance_rate:.1f}%")
            accepted_moves = 0
            rejected_moves = 0
    
    return best, best_energy, history

# ============================================
# EJECUTAR OPTIMIZACIÓN
# ============================================
print("="*70)
optimal_cell_size, optimal_energy, history = simulated_annealing(
    initial_temp=5000,
    min_temp=0.1,
    cooling_rate=0.95,
    max_iterations=1000
)

# ============================================
# CALCULAR ESTADÍSTICAS FINALES
# ============================================
city_side_m = CITY_SIDE_KM * 1000
grid_x = int(np.ceil(city_side_m / optimal_cell_size))
grid_y = int(np.ceil(city_side_m / optimal_cell_size))
total_cells = grid_x * grid_y

coords_m = coords * 1000
idx_x = np.floor(coords_m[:, 0] / optimal_cell_size).astype(int)
idx_y = np.floor(coords_m[:, 1] / optimal_cell_size).astype(int)

valid_mask = (idx_x >= 0) & (idx_x < grid_x) & (idx_y >= 0) & (idx_y < grid_y)
idx_x = idx_x[valid_mask]
idx_y = idx_y[valid_mask]

unique_cells, counts = np.unique(list(zip(idx_x, idx_y)), axis=0, return_counts=True)
counts = np.array(counts) if counts is not None else np.array([])

occupied_cells = len(counts)
occupied_percent = (occupied_cells / total_cells * 100) if total_cells > 0 else 0.0

mean_counts = float(np.mean(counts)) if counts.size > 0 else float('nan')
median_counts = float(np.median(counts)) if counts.size > 0 else float('nan')
std_counts = float(np.std(counts)) if counts.size > 0 else float('nan')
min_counts = int(counts.min()) if counts.size > 0 else None
max_counts = int(counts.max()) if counts.size > 0 else None

# ============================================
# MOSTRAR RESULTADOS
print("\n" + "="*70)
print("🏆 RESULTADOS FINALES - SIMULATED ANNEALING")
print("="*70)
print(f"✅ Tamaño óptimo de celda: {optimal_cell_size:.0f} metros ({optimal_cell_size/1000:.3f} km)")
print(f"⚡ Energía final: {optimal_energy:.2f}")
print(f"📐 Grilla resultante: {grid_x} x {grid_y} = {total_cells} celdas")
print(f"📊 Celdas ocupadas: {occupied_cells} ({occupied_percent:.1f}%)")

print(f"🔄 Iteraciones totales: {len(history['iteration'])}")
print("="*70)
