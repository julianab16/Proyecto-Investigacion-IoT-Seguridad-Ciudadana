import json 
from pathlib import Path
import numpy as np 
from scipy.interpolate import griddata 
import scipy.io as sio
from shapely.geometry import shape


# Parámetros del recorte
ancho = 5900
alto  = 5900
x_ini = 724900
y_ini = 868100

x_fin, y_fin = x_ini + ancho, y_ini + alto

# Leer GeoJSON (el pipeline lo exporta dentro de images/)
base_dir = Path(__file__).resolve().parent
geojson_path = base_dir / 'images' / 'grid_cali.geojson'
if not geojson_path.exists():
    geojson_path = base_dir / 'grid_cali.geojson'

with geojson_path.open('r', encoding='utf-8') as f: 
    geojson_data = json.load(f)

points_x, points_y, risk_values = [], [], []

# Extraer coordenadas y valor de riesgo
for feature in geojson_data['features']: 
    geom = feature['geometry'] 
    props = feature['properties'] 
    # Acepta ambos nombres de atributo; el valor predeterminado es el que
    # exporta georeferencia.py.
    risk_val = props.get('indice_inseguridad', props.get('risk'))
    try:
        risk_val = float(risk_val)
    except (TypeError, ValueError):
        continue
    if not np.isfinite(risk_val):
        continue

    # La grilla exportada contiene polígonos; usar su centroide. También se
    # admiten geometrías Point si el GeoJSON de entrada las contiene.
    try:
        geometry = shape(geom)
        if geometry.is_empty:
            continue
        point = geometry if geometry.geom_type == 'Point' else geometry.centroid
        gx, gy = point.x, point.y
    except (TypeError, ValueError, KeyError):
        continue

    if x_ini <= gx <= x_fin and y_ini <= gy <= y_fin:
        points_x.append(gx - x_ini) # Mapear a [0, 1000]
        points_y.append(gy - y_ini) # Mapear a [0, 1000]
        risk_values.append(risk_val)

# Generar malla regular 101x101 (resolución de 10 m)
if not risk_values:
    raise ValueError(
        f'No se encontraron valores de riesgo dentro del recorte definido en {geojson_path}'
    )

grid_x, grid_y = np.meshgrid(np.linspace(0, 1000, 101), np.linspace(0, 1000, 101))

# Interpolación lineal; completar fuera del casco convexo con el vecino más
# cercano para que la matriz no contenga NaN (incluye el caso de un solo punto).
points = np.column_stack((points_x, points_y))
values = np.asarray(risk_values, dtype=float)
if len(values) >= 3:
    R_grid = griddata(points, values, (grid_x, grid_y), method='linear')
else:
    R_grid = np.full(grid_x.shape, np.nan, dtype=float)

missing = ~np.isfinite(R_grid)
if missing.any():
    R_grid[missing] = griddata(points, values, (grid_x[missing], grid_y[missing]), method='nearest')

if R_grid.shape != (101, 101):
    raise RuntimeError(f'Se esperaba una matriz 101x101, se obtuvo {R_grid.shape}')

# Guardar matriz R_grid para MATLAB 
sio.savemat(base_dir / 'R_grid.mat', {'R_grid': R_grid}) 
print("¡Matriz R_grid.mat creada exitosamente!")
