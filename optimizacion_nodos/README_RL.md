# RL: Aprendizaje por Refuerzo para Optimización de Nodos

## ¿Qué es RL (Reinforcement Learning)?

**RL** es como **entrenar a un agente a través de prueba y error**, dándole recompensas por buen comportamiento.

Imagina que entrenas a un robot:
- 🤖 El robot prueba acciones (moverse, cambiar parámetros)
- ✅ Si hace algo bien → **+reward** (aprecia!)
- ❌ Si hace algo malo → **-reward** (penaliza)
- 🧠 El robot **aprende qué hacer** para maximizar recompensas

---

## El Algoritmo: Q-Learning

**Q-Learning** es un algoritmo de RL que aprende una tabla **Q** que dice:

```
Q[estado, acción] = "qué tan bueno es hacer esta acción en este estado"
```

### Ejemplo Simple (Modo Hexagonal)

```
Estado: "Celda 47, SF=9, TP=20, λ=2"
Acciones posibles:
  - Mover a celda vecina 48 → Q = 0.45
  - Mover a celda vecina 46 → Q = 0.52 ← ¡MEJOR!
  - Mover a celda vecina 52 → Q = 0.38
  - Cambiar SF a 10 → Q = 0.30

El agente elige la acción con Q más alto (mover a celda 46)
```

---

## Estados, Acciones y Recompensas

### 📍 Estado (el lugar donde estamos)

**Modo Hexagonal (mapa real de Cali):**
```python
Estado = (cell_id, SF, TP, λ)

cell_id              # ID de la celda hexagonal (0-176) de Cali
SF     ∈ {7, 8, 9, 10, 11, 12}      # Spreading Factor LoRaWAN
TP     ∈ {14, 20, 27} dBm           # Transmit Power
λ      ∈ {1, 2, 3, 4, 5}            # Paquetes por minuto
```

### 🎯 Acciones (qué puede hacer el agente)

**En el mapa hexagonal de Cali:**
- Mover a celdas adyacentes (max 6 celdas vecinas)
- Cambiar parámetros LoRaWAN:
  - Incrementar/Decrementar SF (7-12)
  - Incrementar/Decrementar TP (14, 20, 27 dBm)
  - Incrementar/Decrementar λ (1-5 paquetes/min)

### 🏆 Recompensa (cómo calificamos cada acción)

```python
Reward = α·PDR - β·Energy_norm - γ·Latency_norm

Donde:
α = 1.0  (importancia del PDR)
β = 1.0  (importancia de energía)
γ = 1.0  (importancia de latencia)

PDR ∈ [0, 1]                      Probabilidad de entrega
Energy_norm = min(Energy / 1.02, 1)  Consumo normalizado
Latency_norm = min(Latency / 4300, 1) Latencia normalizada
```

**Interpretación:**
- Si PDR sube → +más recompensa
- Si Energy sube → -menos recompensa
- Si Latency sube → -menos recompensa

---

## Cómo se Entrena el Agente

### 📚 Paso a Paso: Una Iteración de Entrenamiento

```
Episodio 1:
├─ Iniciar en estado: (cell_id=47, SF=9, TP=20, λ=2)
│
├─ Paso 1: ε-greedy selecciona acción
│  └─ 90% de veces: acción con mayor Q (explotación)
│  └─ 10% de veces: acción aleatoria (exploración)
│
├─ Paso 2: Observar recompensa
│  └─ Reward = 0.45 (PDR buena, pero energía alta)
│
├─ Paso 3: Actualizar Q-table
│  └─ Q[estado_anterior, acción] ← Q + α(reward + γ·max_Q_siguiente - Q)
│
└─ Paso 4: Moverse al siguiente estado y repetir

Después de 1000 episodios:
┌─────────────────────┐
│  Q-table bien      │
│  entrenada:        │
│  sabe qué hacer    │
└─────────────────────┘
```

### 📊 Hiperparámetros

```python
LEARNING_RATE = 0.1       # α: cuánto aprender de cada iteración
DISCOUNT_FACTOR = 0.95    # γ: importancia de recompensas futuras
EPSILON_START = 1.0       # Exploración inicial (100%)
EPSILON_DECAY = 0.995     # Reduce exploración cada episodio
EPSILON_MIN = 0.01        # Mínima exploración (1%)
```

**Interpretación:**
- `α = 0.1`: cambio gradual, no bruscos
- `γ = 0.95`: valora futuro pero más el presente
- Epsilon decae: al principio explora mucho, luego menos

---

## Modo de Operación: Hexagonal (Mapa Real de Cali)

```
    ◇ ◇ ◇ ◇
   ◇ ◇ ◇ ◇ ◇
  ◇ ◇ ◆ ◇ ◇ ◇  ← Agente explorando celda hexagonal
   ◇ ◇ ◇ ◇ ◇
    ◇ ◇ ◇ ◇

- 177 celdas hexagonales reales del mapa de Cali
- CRS geográfico: EPSG:3116 (UTM para Cali)
- Cada celda georreferenciada con coordenadas reales
- Cada celda contiene datos de índice de inseguridad (0-100)
- Agente optimiza posición en geografía real
```

**Integración con GeoJSON:**
```python
# El archivo grid_cali.geojson contiene:
# - 177 celdas hexagonales
# - Centros de celda (lat/lon)
# - Vecinos (Queen adjacency)
# - Índices de inseguridad por celda

# Uso automático en rl.py:
geojson_path = Path(...) / "images" / "grid_cali.geojson"
optimizer = RLNodeOptimizer(...)
optimizer.optimize_all_nodes(geojson_path=geojson_path)
```

---

## Fórmula Clave: Actualización Q-Learning

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]

Donde:
s     = estado actual
a     = acción tomada
r     = recompensa observada
s'    = estado siguiente
α     = learning rate (0.1)
γ     = discount factor (0.95)

En humano:
"Nueva Q = Q vieja + (aprendizaje) × (error)"
"error = (recompensa actual + valor del futuro) - valor estimado"
```

**Ejemplo numérico (Modo Hexagonal):**
```
Q_viejo[(85, 9, 20, 2), "mover a celda vecina"] = 0.40
                ↓
        cell_id=85 (en Cali)

Después de acción:
- Recompensa recibida: r = 0.45
- Max Q en estado siguiente: 0.52

Q_nuevo = 0.40 + 0.1 × (0.45 + 0.95×0.52 - 0.40)
        = 0.40 + 0.1 × (0.45 + 0.494 - 0.40)
        = 0.40 + 0.1 × 0.544
        = 0.40 + 0.0544
        = 0.4544  ← Mejor aprendida que antes
```

---

## Obtener la Solución Final

Después de entrenar, el agente sabe **cuál es la mejor acción en cada estado**:

```python
# Obtener mejor configuración (usando Q-table entrenada)
best_config = optimizer.get_best_configuration(num_steps=50)

print(best_config)
# {
#    'node_id': 0,
#    'opt_x': 753500.45,
#    'opt_y': 872100.23,
#    'SF': 9,
#    'TP': 20,
#    'lambda': 3,
#    'PDR': 0.92,
#    'Energy_mJ': 0.45,
#    'Latency_ms': 350,
#    'Reward': 0.65
# }
```

**Cómo se obtiene:**
1. Comienza en estado inicial
2. Selecciona acción con MAYOR Q (sin exploración)
3. Se mueve al siguiente estado
4. Repite durante 50 pasos (hasta converger)
5. Devuelve la mejor configuración encontrada

---

## Comparación: RL vs GA vs NSGA-II

| Aspecto | RL | GA (Mono) | NSGA-II |
|--------|----|----|---------|
| **Objetivo** | Maximizar 1 función | Una función ponderada | Tres independientes |
| **Resultado** | UNA solución optimizada | UNA solución | Múltiples (Pareto) |
| **Velocidad** | 🟢 Rápido (~2-3 min) | 🟡 Medio (~5-8 min) | 🔴 Lento (~10-15 min) |
| **Modo Espacial** | 🟢 Mapa Hexagonal Real (177 celdas Cali) | 🟡 Grilla Artificial (10×10) | 🟡 Grilla Artificial (10×10) |
| **Georreferencia** | 🟢 EPSG:3116 (UTM Cali) | 🔴 No | 🔴 No |
| **Realismo** | 🟢 Máximo (geografía real) | 🟡 Medio | 🟡 Medio |
| **Flexibilidad** | 🟡 Pesos fijos | 🟡 Pesos fijos | 🟢 Múltiples opciones |
| **Exploración** | 🟢 ε-greedy inteligente | 🟡 Mutación aleatoria | 🟡 Mutación aleatoria |

---

## Parámetros de Ejecución

```python
# Uso actual (ÚNICAMENTE MODO HEXAGONAL)
from optimizacion_nodos.rl import RLNodeOptimizer

optimizer = RLNodeOptimizer(
    node_data={'id': 0, 'x': 750000, 'y': 870000},  # UTM EPSG:3116
    gateway_data={'id': 4, 'x': 751000, 'y': 871000},  # UTM EPSG:3116
    alpha=0.1,                 # Learning rate (qué tan rápido aprende)
    gamma=0.95,                # Discount factor (valor del futuro vs presente)
    epsilon_start=1.0,         # Exploración inicial (100%)
    epsilon_decay=0.995,       # Reducción de exploración por episodio
    epsilon_min=0.01,          # Exploración mínima (1%)
    reward_weights={           # Pesos de objetivos (normalizados)
        'pdr': 1.0,           # Maximizar probabilidad entrega
        'energy': 1.0,        # Minimizar consumo
        'latency': 1.0        # Minimizar latencia
    }
)

results = optimizer.optimize_all_nodes(
    nodes_csv='data_base/info_nodos_lorawan.csv',
    episodes=500,              # Episodios de entrenamiento por nodo
    output_csv='optimizacion_nodos/resultados/rl_optimized_nodes.csv',
    geojson_path='images/grid_cali.geojson',  # CARGADO AUTOMÁTICAMENTE
    max_nodes=None             # None = todos, o número específico para prueba
)
```

**Tiempo estimado:**
- Prueba (max_nodes=50): ~2-3 minutos
- Completo (969 nodos): ~12-15 minutos
- Variable según CPU y episodios configurados

---

## Salida: CSV con Soluciones (Modo Hexagonal)

**Ubicación:** `optimizacion_nodos/resultados/rl_optimized_nodes.csv`

```
node_id, gateway_id, cell_id, lat, lon, SF, TP, lambda, PDR, Energy_mJ, Latency_ms, Reward
0,       4,          85,      3.45, -76.50, 9, 20, 3, 0.92, 0.45, 350, 0.65
1,       2,          92,      3.44, -76.48, 8, 20, 2, 0.88, 0.30, 200, 0.58
2,       1,          78,      3.46, -76.52, 10, 27, 4, 0.95, 0.60, 450, 0.72
...
```

**Columnas explicadas:**
- `node_id`: ID único del nodo LoRaWAN
- `gateway_id`: ID del gateway más cercano
- `cell_id`: **ID de la celda hexagonal** (0-176 en mapa de Cali)
- `lat, lon`: Coordenadas geográficas del centro de la celda
- `SF, TP, lambda`: Parámetros LoRaWAN optimizados por RL
- `PDR, Energy_mJ, Latency_ms`: Métricas de rendimiento
- `Reward`: Recompensa acumulada (medida de optimización)

**Una solución por nodo:** El agente explora y devuelve la mejor configuración que encontró (no múltiples opciones como en NSGA-II).

---

## Ventajas de RL (Hexagonal)

✅ **Rápido** — 2-3 minutos para 969 nodos  
✅ **Realista** — Usa mapa hexagonal real de Cali (177 celdas)  
✅ **Georreferenciado** — EPSG:3116 (coordenadas UTM auténticas)  
✅ **Adaptativo** — ε-greedy balancea exploración y explotación  
✅ **Inteligente** — Aprende relaciones estado-acción-recompensa  
✅ **Sencillo** — Algoritmo clásico Q-Learning bien estudiado  

---

## Resumen Rápido

1. **RL = Aprendizaje por refuerzo** después de probar acciones
2. **Q-Learning** = Tabla que mapea (estado, acción) → calidad
3. **Estado** = celda_hexagonal + Parámetros LoRaWAN (SF, TP, λ)
4. **Acción** = Mover a celda vecina O cambiar parámetro
5. **Recompensa** = PDR_normalizado - Energy_normalizado - Latency_normalizado
6. **Entrenamiento** = 500 episodios de exploración y aprendizaje
7. **Solución** = LA mejor configuración después del aprendizaje
8. **Modo único**: Hexagonal (mapa real de Cali, 177 celdas)

**Listo para usar:** `python optimizacion_nodos/rl.py` 🚀  
**Salida:** `optimizacion_nodos/resultados/rl_optimized_nodes.csv`
