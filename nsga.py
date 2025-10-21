import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# 1Cargar y limpiar datos
# -------------------------------------------------
df = pd.read_csv("..\Proyecto Investigacion IoT Seguridad Ciudadana\SEMANA.csv")

def limpiar_coordenadas_miles(df, lat_col='lat', lon_col='lon'):
    """Limpia coordenadas con formato incorrecto"""
    def limpiar_numero(numero):
        numero_str = str(numero).strip()
        numero_str = numero_str.replace(',', '')
        
        if numero_str.startswith("-7"):
            numero_str = numero_str[0:3] + '.' + numero_str[3:]
        elif numero_str.startswith("3"):
            numero_str = numero_str[0:1] + '.' + numero_str[1:]
        
        try:
            return float(numero_str)
        except ValueError:
            return np.nan
    
    df[lat_col] = df[lat_col].apply(limpiar_numero)
    df[lon_col] = df[lon_col].apply(limpiar_numero)
    return df

# Detectar columnas de coordenadas
if 'x' in df.columns and 'y' in df.columns:
    lon_col, lat_col = 'x', 'y'
elif 'lon' in df.columns and 'lat' in df.columns:
    lon_col, lat_col = 'lon', 'lat'
else:
    lon_candidates = [c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower() or 'x' == c.lower()]
    lat_candidates = [c for c in df.columns if 'lat' in c.lower() or 'y' == c.lower()]
    if lon_candidates and lat_candidates:
        lon_col = lon_candidates[0]
        lat_col = lat_candidates[0]
    else:
        raise SystemExit("No se encontraron columnas de coordenadas.")

df = limpiar_coordenadas_miles(df, lat_col=lat_col, lon_col=lon_col)
df = df.dropna(subset=[lat_col, lon_col])
print(f"Puntos válidos: {len(df)}")

# Obtener límites de Cali y reproyectar
# -------------------------------------------------
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia")
cali = cali.to_crs(3116)  # CRS métrico para Colombia

# Crear GeoDataFrame de puntos
puntos = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), 
    crs="EPSG:4326"
)
puntos = puntos.to_crs(3116)

# Filtrar puntos dentro de Cali
try:
    area_cali = cali.geometry.union_all()
except AttributeError:
    area_cali = cali.geometry.unary_union

puntos = puntos[puntos.within(area_cali)]
print(f"Puntos dentro de Cali: {len(puntos)}")

if len(puntos) == 0:
    raise SystemExit("No hay puntos dentro de Cali después del filtrado.")

# Función de cálculo de métricas
# -------------------------------------------------
def calcular_metricas(puntos_gdf, cali_gdf, h_m):
    """
    Calcula métricas para optimización:
    - f1: Varianza de densidad (homogeneidad espacial)
    - f2: Número de celdas (costo computacional)
    - f3: Porcentaje de celdas vacías (calidad del mapa)
    """
    minx, miny, maxx, maxy = cali_gdf.total_bounds
    
    # Crear grilla métrica
    x_coords = np.arange(minx, maxx + h_m, h_m)
    y_coords = np.arange(miny, maxy + h_m, h_m)
    
    celdas = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            celda = box(x, y, x + h_m, y + h_m)
            # Solo incluir celdas que intersectan con Cali
            if celda.intersects(area_cali):
                celdas.append(celda)
    
    if len(celdas) == 0:
        return float('inf'), float('inf'), 100.0
    
    grid = gpd.GeoDataFrame(geometry=celdas, crs=cali_gdf.crs)
    
    # Contar puntos por celda
    join = gpd.sjoin(puntos_gdf, grid, how="inner", predicate="within")
    conteo = join.groupby("index_right").size()
    
    # Crear array completo con ceros para celdas vacías
    conteo_completo = np.zeros(len(grid))
    conteo_completo[conteo.index] = conteo.values
    
    # Calcular métricas
    # f1: Varianza de densidad (normalizada por área)
    densidad = conteo_completo / (h_m**2 / 1e6)  # eventos por km²
    varianza = float(np.var(densidad))
    
    # f2: Número de celdas (costo computacional)
    num_celdas = float(len(grid))
    
    # f3: Porcentaje de celdas vacías
    pct_vacias = float((conteo_completo == 0).sum() / len(grid) * 100)
    
    # Penalizar si hay muy pocas celdas con datos o demasiadas vacías
    if pct_vacias > 80:
        varianza *= 2  # Penalización
    
    # Calcular eventos promedio por celda
    eventos_por_celda = len(puntos_gdf) / num_celdas
    
    # Penalizar si hay menos de 3 eventos por celda en promedio
    if eventos_por_celda < 3:
        varianza *= 1.5
    
    return varianza, num_celdas, pct_vacias

# Definir problema de optimización
# -------------------------------------------------
class OptimizacionMapaCalor(ElementwiseProblem):
    """
    Optimización multi-objetivo para tamaño de celda:
    - Minimizar varianza de densidad (homogeneidad)
    - Minimizar número de celdas (eficiencia)
    - Minimizar celdas vacías (calidad)
    """
    def __init__(self):
        # Rango de tamaño de celda: 50m a 2000m
        super().__init__(
            n_var=1,
            n_obj=3,  # 3 objetivos
            xl=np.array([50.0]),
            xu=np.array([500.0])
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        h = float(x[0])
        varianza, num_celdas, pct_vacias = calcular_metricas(puntos, cali, h)
        
        # Normalizar objetivos para mejor convergencia
        f1 = varianza / 1000  # Varianza normalizada
        f2 = num_celdas / 100  # Número de celdas normalizado
        f3 = pct_vacias  # Porcentaje de vacías
        
        out["F"] = np.array([f1, f2, f3], dtype=float)

# Ejecutar NSGA-II
# -------------------------------------------------
print("\nEjecutando optimización NSGA-II...")
print("Esto puede tomar varios minutos...\n")

problem = OptimizacionMapaCalor()

algorithm = NSGA2(
    pop_size=50,  # Población más grande para mejor exploración
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 80)  # 80 generaciones

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True,
    save_history=False
)

# Analizar y visualizar resultados
# -------------------------------------------------
print("\n" + "="*60)
print("RESULTADOS DE OPTIMIZACIÓN")
print("="*60)

F = res.F
X = res.X

# Desnormalizar para mostrar
F_real = F.copy()
F_real[:, 0] *= 1000  # Varianza real
F_real[:, 1] *= 100   # Número de celdas real

# Calcular métricas adicionales para cada solución
resultados = []
for i, h in enumerate(X[:, 0]):
    var, n_celdas, pct_vac = calcular_metricas(puntos, cali, h)
    eventos_por_celda = len(puntos) / n_celdas
    resultados.append({
        'h': h,
        'varianza': var,
        'n_celdas': int(n_celdas),
        'pct_vacias': pct_vac,
        'eventos_x_celda': eventos_por_celda
    })

# Ordenar por eventos por celda (métrica de calidad)
resultados_df = pd.DataFrame(resultados)
resultados_df = resultados_df.sort_values('eventos_x_celda', ascending=False)

print("\nTOP 10 SOLUCIONES ÓPTIMAS:")
print("-" * 90)
print(f"{'Tamaño':<12} {'N° Celdas':<12} {'Eventos/Celda':<15} {'% Vacías':<12} {'Varianza':<12}")
print("-" * 90)
for _, row in resultados_df.head(10).iterrows():
    print(f"{row['h']:>7.1f} m    {row['n_celdas']:>8}    "
          f"{row['eventos_x_celda']:>11.2f}      {row['pct_vacias']:>8.1f}%    "
          f"{row['varianza']:>10.2f}")

# Encontrar la solución más balanceada
mejor_idx = resultados_df['eventos_x_celda'].sub(7).abs().idxmin()
mejor = resultados_df.loc[mejor_idx]

print("\n" + "="*60)
print("RECOMENDACIÓN ÓPTIMA:")
print("="*60)
print(f"Tamaño de celda:     {mejor['h']:.1f} metros")
print(f"Número de celdas:    {mejor['n_celdas']}")
print(f"Eventos por celda:   {mejor['eventos_x_celda']:.2f}")
print(f"Celdas vacías:       {mejor['pct_vacias']:.1f}%")
print(f"Varianza:            {mejor['varianza']:.2f}")
print("="*60)
