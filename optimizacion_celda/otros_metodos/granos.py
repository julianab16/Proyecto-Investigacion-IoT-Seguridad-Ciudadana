import numpy as np
import random, math
import matplotlib.pyplot as plt

import numpy as np 
import random, math 

# ----------------------------------------
# Parámetros del problema
# ----------------------------------------
area_cali = 619  # km²
n_puntos = 2000  # puntos (por ejemplo, lugares o eventos)
densidad = n_puntos / area_cali  # puntos por km²

# ----------------------------------------
# Función de aptitud (fitness)
# ----------------------------------------
def fitness(cell_size_km):
    """
    Evalúa qué tan buena es una celda de lado 'cell_size_km'.
    Penaliza si la densidad de puntos por celda está muy fuera del rango deseado.
    """
    area_celda = cell_size_km**2
    puntos_por_celda = area_celda * densidad
    
    # Queremos entre 30 y 80 puntos por celda
    objetivo = 55
    penalizacion = abs(puntos_por_celda - objetivo)
    
    # Añadimos ruido aleatorio pequeño para simular incertidumbre
    return penalizacion + np.random.rand() * 5

# ----------------------------------------
# Parámetros del Simulated Annealing
# ----------------------------------------
T = 1000      # temperatura inicial
T_min = 1     # temperatura mínima
alpha = 0.90  # tasa de enfriamiento

current = random.uniform(0.1, 10)  # tamaño inicial de celda (km)
best = current
historia = []

# ----------------------------------------
# Bucle principal del SA
# ----------------------------------------
while T > T_min:
    new = current + np.random.uniform(-0.3, 0.3)
    if new <= 0: 
        continue

    E_current = fitness(current)
    E_new = fitness(new)

    # Criterio de aceptación
    if E_new < E_current or random.random() < math.exp((E_current - E_new) / T):
        current = new
        if fitness(current) < fitness(best):
            best = current

    historia.append((T, current, fitness(current)))
    T *= alpha

# ----------------------------------------
# Resultados
# ----------------------------------------
best_m = best * 1000.0

print(f"Con Simulated Annealing")
print(f"Tamaño de celda óptimo: {best:.2f} km ({best_m:.0f} m)")
area_optima = best**2
n_celdas = area_cali / area_optima
print(f"Área por celda: {area_optima:.2f} km²")
print(f"Número de celdas en Cali: {n_celdas:.0f}")


import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# reproducibilidad
np.random.seed(42)
random.seed(42)

# -------------------------------------------------
# 1️⃣ Simular puntos dentro del área de Cali
# -------------------------------------------------
area_cali = 619  # km²
lado = np.sqrt(area_cali)  # aprox. 24.87 km

# simulamos 2000 coordenadas (x, y) dentro de un cuadrado de lado km
coords = np.random.rand(2000, 2) * lado

# -------------------------------------------------
# 2️⃣ Probar distintos números de clusters (K)
# -------------------------------------------------
inertias = []
silhouettes = []
ks = range(10, 150, 10)
ks_list = list(ks)

for k in ks_list:
    # usar n_init explícito para compatibilidad
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords)
    labels = kmeans.labels_
    inertias.append(kmeans.inertia_)
    # calcular silhouette (asegurarnos de tener al menos 2 clusters y menos clusters que puntos)
    if 1 < k < len(coords):
        silhouettes.append(silhouette_score(coords, labels))
    else:
        silhouettes.append(np.nan)
    print(f"K={k:3d} | Inercia={kmeans.inertia_:10.2f} | Silhouette={silhouettes[-1]:.3f}")

# -------------------------------------------------
# 4️⃣ Seleccionar automáticamente el mejor K
# -------------------------------------------------
# usar nanargmax por si hay NaNs (pocas posibilidades aquí)
best_idx = int(np.nanargmax(silhouettes))
k_opt = ks_list[best_idx]

# -------------------------------------------------
# 5️⃣ Calcular tamaño promedio de celda (km y m)
# -------------------------------------------------
area_celda_prom = area_cali / k_opt                # km²
lado_promedio_km = np.sqrt(area_celda_prom)        # km
lado_promedio_m = lado_promedio_km * 1000.0        # m
n_celdas = area_cali / area_celda_prom
print(f"\nNúmero óptimo de clusters según Silhouette: {best_idx}")
print(f"Área promedio de cada zona: {area_celda_prom:.2f} km²")
print(f"Lado promedio de cada celda: {lado_promedio_km:.2f} km ({lado_promedio_m:.0f} m)")



import numpy as np
from geneticalgorithm import geneticalgorithm as ga

# -------------------------------------------------
# 1️⃣ Simular datos espaciales dentro del área de Cali
# -------------------------------------------------
area_cali = 619  # km²
lado = np.sqrt(area_cali)  # ≈ 24.87 km (convertimos a metros para el GA)
lado_m = lado * 1000  # 24.870 m

coords = np.random.rand(2000, 2) * [lado_m, lado_m]  # puntos en Cali (m)

# -------------------------------------------------
# 2️⃣ Función de aptitud (fitness)
# -------------------------------------------------
# ...existing code...
import math
# ...existing code...

# -------------------------------------------------
# 2️⃣ Función de aptitud (fitness) - versión robusta
# -------------------------------------------------
def fitness(cell_size):
    try:
        # geneticalgorithm pasa un array, tomar el primer valor
        s = float(cell_size[0]) if hasattr(cell_size, "__len__") else float(cell_size)
        if s <= 0:
            return 1e6

        grid_x = int(math.floor(lado_m / s))
        grid_y = int(math.floor(lado_m / s))
        if grid_x <= 0 or grid_y <= 0:
            return 1e6

        # índices enteros de celda para cada punto
        idx_x = (coords[:, 0] // s).astype(int)
        idx_y = (coords[:, 1] // s).astype(int)

        # codificar par (ix,iy) a entero para usar np.unique eficientemente
        key = idx_x * (grid_y + 1) + idx_y
        unique, counts = np.unique(key, return_counts=True)

        occupied_ratio = len(unique) / (grid_x * grid_y)
        variance = np.var(counts) if len(counts) > 0 else 0.0

        score = abs(occupied_ratio - 0.5) * 1000 + variance
        return float(score)

    except Exception:
        # devolver un valor alto si algo falla para que el GA lo evite
        return 1e6

# -------------------------------------------------
# 3️⃣ Configurar el algoritmo genético (añadir funtimeout)
# -------------------------------------------------
varbound = np.array([[50, 2000]])  # entre 50 m y 2000 m

algorithm_param = {
    'max_num_iteration': 300,
    'population_size': 50,
    'mutation_probability': 0.1,
    'elit_ratio': 0.1,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 50,
    'funtimeout': 30  # segundos por evaluación total (ajusta según necesidad)
}

model = ga(
    function=fitness,
    dimension=1,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param
)

# ...existing code...

# -------------------------------------------------
# 4️⃣ Ejecutar el modelo
# -------------------------------------------------
model.run()

# -------------------------------------------------
# 5️⃣ Obtener resultados
# -------------------------------------------------
output = model.output_dict
best_size = output['variable'][0]
print(f"Con geneticalgorithm")

print(f"Tamaño de celda óptimo: {best_size:.2f} metros")

area_celda = (best_size / 1000)**2  # km²
n_celdas = area_cali / area_celda

print(f"Área por celda: {area_celda:.2f} km²")
print(f"Número aproximado de celdas: {n_celdas:.0f}")
