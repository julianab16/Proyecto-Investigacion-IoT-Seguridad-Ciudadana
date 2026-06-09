import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Point

# ========== CONFIGURACIÓN ==========
N = 1000  # Número de nodos
print("=" * 70)
print(" " * 15 + "GENERACIÓN DE NODOS IoT EN CALI")
print("=" * 70)

# ========== CARGA DEL MAPA DE CALI ==========
print(f"\n[1/4] Cargando mapa de Santiago de Cali...")
cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=None)
cali_proj = cali.to_crs(3116)  # EPSG:3116 - MAGNA-SIRGAS Colombia Bogotá

# Obtener límites
xmin, ymin, xmax, ymax = cali_proj.total_bounds
print(f"✓ Mapa cargado")
print(f"  📍 Límites: X=[{xmin:,.0f}, {xmax:,.0f}] m, Y=[{ymin:,.0f}, {ymax:,.0f}] m")

# ========== GENERACIÓN DE NODOS ==========
print(f"\n[2/4] Generando {N} nodos dentro de Cali...")
nodos_coords = []
intentos = 0
max_intentos = N * 100

while len(nodos_coords) < N and intentos < max_intentos:
    x = np.random.uniform(xmin, xmax)
    y = np.random.uniform(ymin, ymax)
    punto = Point(x, y)
    
    if cali_proj.geometry.iloc[0].contains(punto):
        nodos_coords.append([x, y])
    
    intentos += 1
    if len(nodos_coords) % 100 == 0:
        print(f"  • Progreso: {len(nodos_coords)}/{N} nodos")

print(f"✓ {len(nodos_coords)} nodos generados")

# ========== CREAR DATAFRAME ==========
print(f"\n[3/4] Creando DataFrame...")
nodos = pd.DataFrame({
    "ID": np.arange(1, len(nodos_coords) + 1),
    "X_m": [coord[0] for coord in nodos_coords],
    "Y_m": [coord[1] for coord in nodos_coords]
})

# ========== GUARDAR ARCHIVO ==========
nombre_archivo = "nodos_iot_cali.csv"
nodos.to_csv(nombre_archivo, index=False)
print(f"✓ Archivo '{nombre_archivo}' guardado")

# Estadísticas
print(f"\n📊 ESTADÍSTICAS:")
print(f"   • Total de nodos: {len(nodos)}")
print(f"   • X: [{nodos['X_m'].min():,.0f}, {nodos['X_m'].max():,.0f}] m")
print(f"   • Y: [{nodos['Y_m'].min():,.0f}, {nodos['Y_m'].max():,.0f}] m")

# ========== VISUALIZACIÓN ==========
print(f"\n[4/4] Generando visualización...")
fig, ax = plt.subplots(figsize=(12, 10))

# Dibujar límites de Cali
cali_proj.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.5, linewidth=1.5)

# Dibujar nodos
ax.scatter(nodos["X_m"], nodos["Y_m"], color='red', s=10, alpha=0.6, label=f'Nodos IoT (n={N})')

ax.set_title(f'Distribución de {N} Nodos IoT en Santiago de Cali', fontsize=14, fontweight='bold')
ax.set_xlabel('X (metros - EPSG:3116)', fontsize=11)
ax.set_ylabel('Y (metros - EPSG:3116)', fontsize=11)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nodos_iot_cali.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfica guardada: 'nodos_iot_cali.png'")
plt.show()

print("\n" + "=" * 70)
print("✅ PROCESO COMPLETADO")
print("=" * 70)
