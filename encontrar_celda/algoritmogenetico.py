import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# ============================================
# CONFIGURACIÓN ESPECÍFICA PARA TU PROYECTO
# ============================================
df = pd.read_csv("violencia-db/SEMANA.csv")
CITY_AREA_KM2 = 619  # Área de tu ciudad en km²
TOTAL_RECORDS = len(df)  # número de filas/registros
CITY_SIDE_KM = np.sqrt(CITY_AREA_KM2)  # ~24.88 km (asumiendo forma cuadrada)

# Cargar tus datos reales (ajusta la ruta y columnas). Si no existen columnas x,y, simular.
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

print(f"Puntos cargados (coords.shape): {coords.shape}")
print(f"📍 Área de la ciudad: {CITY_AREA_KM2} km²")
print(f"📊 Total de registros: {TOTAL_RECORDS}")
print(f"📏 Dimensión estimada: {CITY_SIDE_KM:.2f} km x {CITY_SIDE_KM:.2f} km")
print(f"🎯 Densidad promedio: {TOTAL_RECORDS/CITY_AREA_KM2:.2f} datos/km²\n")

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
    None

df = limpiar_coordenadas_miles(df, lat_col, lon_col)

# ============================================
# FUNCIÓN DE FITNESS MEJORADA
# ============================================
def fitness(cell_size):
    cell_size = cell_size[0]
    
    # Validar que el tamaño de celda sea razonable
    if cell_size <= 0:
        return 1e8
    
    # Convertir a metros para mayor precisión
    cell_size_m = cell_size
    city_side_m = CITY_SIDE_KM * 1000
    
    # Calcular número de celdas en cada eje
    grid_x = int(np.ceil(city_side_m / cell_size_m))
    grid_y = int(np.ceil(city_side_m / cell_size_m))
    total_cells = grid_x * grid_y
    
    if total_cells == 0 or total_cells > 10000:  # Evitar grillas extremas
        return 1e8
    
    # Asignar cada punto a una celda
    coords_m = coords * 1000  # Convertir coordenadas a metros
    idx_x = np.floor(coords_m[:, 0] / cell_size_m).astype(int)
    idx_y = np.floor(coords_m[:, 1] / cell_size_m).astype(int)
    
    # Contar puntos por celda
    cell_indices = list(zip(idx_x, idx_y))
    unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
    
    # Métricas de calidad
    occupied_cells = len(counts)
    occupied_ratio = occupied_cells / total_cells
    empty_cells = total_cells - occupied_cells
    
    # Varianza de densidad (queremos homogeneidad)
    variance = np.var(counts)
    
    # Datos mínimos recomendados por celda
    min_data_per_cell = TOTAL_RECORDS / total_cells
    
    # FUNCIÓN DE FITNESS COMPUESTA
    # Penalizar si hay muchas celdas vacías O si hay muy pocas celdas
    penalty_empty = empty_cells * 10
    penalty_variance = variance * 0.5
    penalty_sparse = abs(min_data_per_cell - 5) * 100  # Objetivo: ~5 datos/celda
    
    # Bonificar si el ratio de ocupación está entre 40-80%
    if 0.4 <= occupied_ratio <= 0.8:
        bonus = -1000
    else:
        bonus = abs(occupied_ratio - 0.6) * 2000
    
    fitness_value = penalty_empty + penalty_variance + penalty_sparse + bonus
    
    return fitness_value

# ============================================
# EJECUTAR ALGORITMO GENÉTICO
# ============================================
print("🧬 Iniciando optimización con Algoritmo Genético...\n")

# Rango de búsqueda: desde 100m hasta 5000m (ajustable)
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 50,
    'mutation_probability': 0.2,
    'elit_ratio': 0.01,
    'crossover_probability': 0.8,
    'crossover_type': 'uniform',   # <-- agregar para evitar KeyError; otras opciones: 'one_point','two_points'
    'max_iteration_without_improv': None,
    'funtimeout': None,
    'parents_portion': 0.3
}

model = ga(
    function=fitness,
    dimension=1,
    variable_type='real',
    variable_boundaries=np.array([[100, 5000]]),
    algorithm_parameters=algorithm_param
)

model.run()

# ============================================
# RESULTADOS
# ============================================
optimal_cell_size_m = model.output_dict['variable'][0]
optimal_cell_size_km = optimal_cell_size_m / 1000

city_side_m = CITY_SIDE_KM * 1000
grid_x = int(np.ceil(city_side_m / optimal_cell_size_m))
grid_y = int(np.ceil(city_side_m / optimal_cell_size_m))
total_cells = grid_x * grid_y

# Calcular estadísticas con el tamaño óptimo
coords_m = coords * 1000
idx_x = np.floor(coords_m[:, 0] / optimal_cell_size_m).astype(int)
idx_y = np.floor(coords_m[:, 1] / optimal_cell_size_m).astype(int)
unique_cells, counts = np.unique(list(zip(idx_x, idx_y)), axis=0, return_counts=True)

print("\n" + "="*50)
print("🎯 RESULTADOS DE LA OPTIMIZACIÓN")
print("="*50)
print(f"✅ Tamaño óptimo de celda: {optimal_cell_size_m:.0f} metros ({optimal_cell_size_km:.2f} km)")
print(f"📐 Grilla resultante: {grid_x} x {grid_y} = {total_cells} celdas")
print(f"📊 Celdas ocupadas: {len(counts)} ({len(counts)/total_cells*100:.1f}%)")
print(f"📍 Promedio de datos por celda ocupada: {np.mean(counts):.1f}")
print(f"📈 Mediana de datos por celda: {np.median(counts):.1f}")
print(f"📉 Varianza: {np.var(counts):.2f}")
print(f"🔢 Min-Max datos por celda: {counts.min()}-{counts.max()}")
print("="*50)

# ============================================
# RECOMENDACIONES ADICIONALES
# ============================================
print("\n📋 RECOMENDACIONES:")
print(f"1. Para mapas de calor: usa celdas de {optimal_cell_size_m:.0f}m")
print(f"2. Si necesitas más detalle: prueba {optimal_cell_size_m/2:.0f}m (más celdas)")
print(f"3. Si necesitas menos ruido: prueba {optimal_cell_size_m*2:.0f}m (menos celdas)")