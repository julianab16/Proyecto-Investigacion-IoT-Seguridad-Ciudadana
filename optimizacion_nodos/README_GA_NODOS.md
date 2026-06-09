# GA - Optimización de Nodos LoRaWAN (Algoritmo Genético Mono-Objetivo)

## 🎯 ¿Qué hace?

Optimiza la **posición y parámetros de cada nodo IoT** para mejorar su conexión con gateways.

Para cada nodo, determina:
- **x, y** → Mejor posición geográfica
- **SF** (Spreading Factor) → 7-12
- **TP** (Transmit Power) → 14, 20, 27 dBm
- **λ** (Lambda/Packets) → 1-5 paquetes por hora

**Objetivo:** Maximizar PDR, minimizar energía y latencia.

---

## 📐 Función Objetivo (Lo que optimiza)

```
f = w₁ × PDR̂ - w₂ × Ê - w₃ × L̂

Donde:
  PDR̂    = Packet Delivery Rate normalizado (0-1)
  Ê      = Consumo de energía normalizado (0-1)
  L̂      = Latencia normalizada (0-1)
  w₁, w₂, w₃ = Pesos (todos = 1.0 por defecto)
```

**Ejemplo:**
```
PDR = 0.85  (85% paquetes entregados)
E = 0.3 mJ  (normalizado a MAX_ENERGY = 1.02 mJ)
L = 200 ms  (normalizado a MAX_LATENCY = 4300 ms)

f = 1.0 × 0.85 - 1.0 × 0.29 - 1.0 × 0.047 = 0.513

Mayor f = Mejor solución
```

---

## 🧬 ¿Cómo funciona el Algoritmo Genético?

### **Paso 1: Inicialización**
```
Crear 100 individuos (soluciones candidatas)

Cada individuo = [x, y, SF, TP, λ]
                 ↓  ↓  ↓   ↓  ↓
                 Posición | Parámetros
```

### **Paso 2: Evaluar**
```
Para cada individuo:
  1. Calcular distancia a gateway
  2. Estimar PDR (basado en SF, potencia, distancia)
  3. Calcular energía consumida
  4. Estimar latencia
  5. Combinar en función f
```

### **Paso 3: Cruce (Crossover)**
```
Parent 1: [x₁, y₁, SF₁=7, TP₁=20, λ₁=2]
Parent 2: [x₂, y₂, SF₂=12, TP₂=27, λ₂=5]
           ↓
Hijo:     [x₂, y₁, SF₂=12, TP₁=20, λ₂=5]
          (mezcla características)
```

### **Paso 4: Mutación**
```
Individuo: [x, y, SF, TP, λ]
           ↓         ↓
Mutado:    [x+ruido, SF=aleatorio, ...]
           (introduce variación)
```

### **Paso 5: Seleccionar mejores y repetir**
```
Generación 1 → Generación 2 → ... → Generación 50
(Evoluciona hacia mejor solución)
```

---

## 📊 Clases Principales

### **1. FitnessCalculator**
Calcula las 3 métricas de rendimiento:

```python
# PDR (Packet Delivery Rate)
pdr = FitnessCalculator.calculate_pdr(sf, tp, distancia, lambda)
# Basado en: Path Loss, SNR requerido, modelo LoRaWAN

# Energía (consumo en mJ)
energy = FitnessCalculator.calculate_energy(sf, tp, lambda)
# Basado en: SF (spreading factor), TP (potencia), paquetes

# Latencia (en ms)
latency = FitnessCalculator.calculate_latency(sf, lambda, distancia)
# Basado en: SF, número de paquetes, propagación
```

### **2. GAOptimizerMonoObjective**
Ejecuta el algoritmo genético:

```python
optimizer = GAOptimizerMonoObjective(
    data=df_nodos,           # CSV con nodos
    population_size=100,     # Individuos
    generations=50,          # Iteraciones
    mutation_rate=0.2,       # 20% de mutación
    crossover_rate=0.8,      # 80% cruce
    cali=cali_geodata        # Límites geográficos
)

results = optimizer.run_optimization_global()  # Optimiza todos
```

---

## 🔧 Cómo se usa

### **Opción 1: Ejecutar todo**
```python
python ga.py
```

**Produce:**
```
GA MONO-OBJETIVO - OPTIMIZACIÓN DE NODOS
================================================
Nodos: 969
Población: 100 | Generaciones: 50
================================================

[1/969] Nodo 0 (Gateway: 0)
    ✓ PDR=0.897 | Energy=0.23mJ | Latency=145ms

[2/969] Nodo 1 (Gateway: 0)
    ✓ PDR=0.885 | Energy=0.25mJ | Latency=152ms
    
... (969 nodos)

✓ Resultados guardados en: optimizacion_nodos/resultados/ga_best_solutions.csv
```

---

## 📋 Parámetros Configurables

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `population_size` | 100 | Más = mejor exploración, más lento |
| `generations` | 50 | Más = mejor convergencia, más lento |
| `mutation_rate` | 0.2 (20%) | Más = más variación, menos convergencia |
| `crossover_rate` | 0.8 (80%) | Más = más herencia, menos novedad |

---

## 📊 Entrada y Salida

### **Entrada: CSV de nodos**
```
node_id, pos_x, pos_y, gateway_id, gw_x, gw_y, sf, tp, packets_sent
0,       750000, 870000, 0,        750100, 870100, 7, 20, 2
1,       750500, 870500, 0,        750100, 870100, 7, 20, 2
...
```

### **Salida: CSV optimizado**
```
node_id, gateway_id, opt_x, opt_y, SF, TP, lambda, PDR, Energy_mJ, Latency_ms
0,       0,          750245, 870123, 8, 20, 2,    0.897, 0.23,    145
1,       0,          750623, 870512, 7, 14, 2,    0.885, 0.21,    140
...
```

---

## ⚙️ Cálculos Internos (Simplificado)

### **PDR (Probabilidad de entrega)**
```
1. Path Loss = 120.76 + 1.88 × log₁₀(distancia)
2. Potencia recibida = Potencia_TX - Path_Loss
3. SNR = Potencia_RX - Ruido
4. Si SNR > 5dB:  PDR = 1 / (1 + e^(-SNR/10))
   Si SNR ≤ 5dB:  PDR = 0
```

### **Energía (consumo)**
```
Energía = Bytes × Factor_SF × Factor_TP × λ
        = 51 × factor_SF[SF] × factor_TP[TP] × λ

Ejemplo: SF=12, TP=27, λ=5
Factor_SF = 2.0, Factor_TP = 2.0
Energy = 51 × 2.0 × 2.0 × 5 = 1020 mJ
```

### **Latencia**
```
Latencia = Latencia_SF + Espera + Propagación
         = SF_latency + (λ-1)×1000ms + distancia/3e8*1000ms
```

---


---

## 💡 Ventajas del GA para esto

✅ Explora múltiples combinaciones (SF × TP × λ)  
✅ Encuentra balance entre PDR, energía y latencia  
✅ No se queda en soluciones locales (gracias a mutación)  
✅ Escalable a 969+ nodos  
✅ Independiente de distribución de nodos  

---

## ⏱️ Tiempo de Ejecución

Con 969 nodos:
- **Population=100, Generations=50:** ~10-15 min
- **Population=150, Generations=100:** ~30-45 min

Cada nodo se optimiza independientemente.

---

## 📁 Archivos Generados

```
optimizacion_nodos/
├── resultados/
│   └── ga_best_solutions.csv          ← Resultados finales
└── pipeline_nodos_ga.pkl              ← Cache de ejecución
```

---

## 🚀 Resumen

| Aspecto | Descripción |
|--------|-------------|
| **Algoritmo** | Genético (evolución) |
| **Objetivo** | Mono-objetivo (1 función f) |
| **Variables** | Posición (x,y) + Parámetros (SF,TP,λ) |
| **Entrada** | 969 nodos LoRaWAN |
| **Salida** | Posición y parámetros optimizados |
| **Métrica** | f = PDR - Energy - Latency |
| **Tiempo** | 10-15 min para todos |

**→ Resultado: Red LoRaWAN optimizada con máxima cobertura y eficiencia**
