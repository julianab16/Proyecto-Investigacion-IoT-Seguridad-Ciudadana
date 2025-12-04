import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Número de nodos
N = 1000

# Área 1000m x 1000m
width = 10000
height = 10000

# Generación aleatoria uniforme
X = np.random.uniform(0, width, N)
Y = np.random.uniform(0, height, N)

# DataFrame con los nodos
nodos = pd.DataFrame({
    "ID": np.arange(1, N + 1),
    "X_m": X,
    "Y_m": Y
})

print(nodos)

nombre_archivo = "nodos_iot.csv"
nodos.to_csv(nombre_archivo, index=False)
print(f"✅ Archivo '{nombre_archivo}' creado con éxito ({N} nodos).")

plt.figure(figsize=(6,6))
plt.scatter(nodos["X_m"], nodos["Y_m"], color='royalblue', s=40)
plt.title("Distribución de 100 nodos IoT en área 1000m x 1000m")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.show()
