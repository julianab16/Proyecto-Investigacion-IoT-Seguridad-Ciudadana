# NSGA-II: Optimización Multiobjetivo de Nodos LoRaWAN

## ¿Qué es NSGA-II?

**NSGA-II** (Non-dominated Sorting Genetic Algorithm II) es un algoritmo genético que **busca los mejores compromisos** entre múltiples objetivos que están en conflicto.

En nuestro caso optimiza **3 objetivos a la vez**:
- ✅ **Maximizar PDR** → Más paquetes llegan al gateway
- 🔋 **Minimizar Energía** → Los nodos gastan menos batería
- ⏱️ **Minimizar Latencia** → Los datos llegan más rápido

---

## El Problema: 3 Objetivos en Conflicto

Imagina que quieres mejorar los nodos del IoT:

| Objetivo | Solución | Problema |
|----------|----------|----------|
| **Más PDR** | Usa SF alto (12) + Potencia alta (27 dBm) | 📴 Cuesta más energía y tarda más |
| **Menos energía** | Usa SF bajo (7) + Potencia baja (14 dBm) | 📉 Menos nodos pueden comunicarse |
| **Menos latencia** | Usa solo 1 paquete | 📉 Si falla, no hay reintento |

**No puedes optimizar los 3 al mismo tiempo.** NSGA-II busca un **equilibrio inteligente**.

---

## Cómo Funciona el Algoritmo

### 1️⃣ **Inicialización: Crear Soluciones Aleatorias**

```python
# Cada "individuo" es una configuración del nodo
Individuo = [SF, TP, λ]

SF    ∈ {7, 8, 9, 10, 11, 12}     # Spreading Factor
TP    ∈ {14, 20, 27} dBm          # Potencia de transmisión
λ     ∈ {1, 2, 3, 4, 5}           # Número de paquetes a enviar
```



### 2️⃣ **Evaluar: Calcular los 3 Objetivos**

Para cada individuo, se calcula:

```python
# 1. PDR (Probabilidad de entrega)
PDR = f(SF, TP, distancia_al_gateway)
PDR ∈ [0, 1]

# 2. Energía consumida (mJ = milij julios)
Energy = SF_factor × TP_factor × λ
Energy ∈ [0, muchos]

# 3. Latencia (ms = milisegundos)
Latency = tiempo_SF + tiempo_espera + tiempo_propagación
Latency ∈ [50, 5000+]
```


### 3️⃣ **Reproducción: Crear Hijos**

**Crossover (Cruce):**
```
Padre1:   [SF=10, TP=27, λ=2]
Padre2:   [SF=7,  TP=14, λ=4]
                     ↓ Intercambiar genes
Hijo1:    [SF=7,  TP=27, λ=2]  ← Combina lo mejor de cada padre
Hijo2:    [SF=10, TP=14, λ=4]
```

**Mutación (Cambio aleatorio):**
```
Hijo antes:  [SF=7, TP=27, λ=2]
                   ↓ Cambio aleatorio (20% de probabilidad)
Hijo después: [SF=9, TP=14, λ=2]  ← El SF cambió de 7 a 9
```

### 4️⃣ **Selección: Elegir los Mejores**

El algoritmo compara todos los individuos y mantiene los **no-dominados**:

```
Solución A: PDR=0.95, Energy=0.8mJ, Latency=400ms   ← Buena en PDR
Solución B: PDR=0.90, Energy=0.4mJ, Latency=200ms   ← Buena en Energy/Latency
Solución C: PDR=0.85, Energy=0.9mJ, Latency=500ms   ← ❌ PEOR en TODO
                                                       (se descarta)
```

**Clave:** Una solución es "dominada" si otra es mejor en al menos un objetivo **sin ser peor en los otros**.

### 5️⃣ **Repetir**

Regresa al paso 2 (evaluar), genera nuevos hijos, selecciona los mejores... **150 generaciones**.

---

## Las Funciones Clave

### `FitnessCalculator.calculate_pdr()`
```python
PDR = 1 - (1 - P_recepción)^λ

P_recepción = Q(-(SNR)/√2)  # Función Q (estadística)
SNR = TP - PL - Ruido        # Relación Señal/Ruido
PL = a + b·log10(d)          # Path Loss (pérdida por distancia)
```

**En palabras simples:** Si un paquete tiene 90% de chance de llegar, y envías 3 paquetes, casi seguro uno llega.

### `FitnessCalculator.calculate_energy()`
```python
Energy = Bytes × K_SF × K_TP × λ

K_SF = factor energético del SF
K_TP = factor energético del TP
λ    = número de paquetes
```

**Ejemplo:** SF=12 + TP=27dBm + λ=5 = **máximo consumo energético**.

### `FitnessCalculator.calculate_latency()`
```python
Latency = L_SF + T_espera + L_propagación

L_SF        = 50ms a 300ms (según SF)
T_espera    = (λ-1) × 1000ms  (1 seg entre reintentos)
L_propagación ≈ distancia / velocidad_luz
```

**Ejemplo:** SF=12, λ=3 → 300 + 2000 + 0 = 2300ms

---

## Salida: Frontera de Pareto

El algoritmo **NO devuelve UNA solución**, sino **múltiples soluciones válidas**:

| SF | TP | λ | PDR | Energy | Latency | Perfil |
|----|----|----|-----|--------|---------|--------|
| 7 | 27 | 5 | 0.99 | 1.02 | 4050 | **Máxima confiabilidad** |
| 9 | 20 | 3 | 0.92 | 0.45 | 350 | **Equilibrio** ⭐ |
| 12 | 14 | 1 | 0.75 | 0.10 | 50 | **Mínima latencia** |

**Tú eliges cuál usar según necesites:**
- 🏥 Hospital → Máximo PDR (primer fila)
- 🏭 Industria → Equilibrio (segunda fila)
- 🚗 Taxi → Latencia mínima (tercera fila)

---

## Parámetros de Ejecución

```python
NSGAOptimizer(
    data=nodos,           # 969 nodos a optimizar
    pop_size=80,          # 80 individuos por generación
    generations=150,      # 150 generaciones
    cxpb=0.7,            # 70% probabilidad de crossover
    mutpb=0.3            # 30% probabilidad de mutación
)
```

**Tiempo estimado:** 10-15 minutos para todos los nodos.

---

## Archivos de Salida

### `nsga_pareto_solutions.csv`
```
node_id, gateway_id, SF, TP, lambda, PDR, Energy_mJ, Latency_ms
0,       4,          7,  27, 5,      0.99, 1.02,      4050
0,       4,          9,  20, 3,      0.92, 0.45,      350
0,       4,          12, 14, 1,      0.75, 0.10,      50
1,       2,          8,  20, 2,      0.88, 0.30,      200
...
```

Cada nodo tiene **múltiples soluciones** (su frontera de Pareto).

---

## Ventajas vs Algoritmo GA

| Aspecto | GA (Mono-objetivo) | NSGA-II (Multi-objetivo) |
|--------|------------------|------------------------|
| **Objetivo** | Una función f = PDR - Energy - Latency | Tres objetivos independientes |
| **Resultado** | UNA solución por nodo | Frontera de Pareto (varias) |
| **Flexibilidad** | Fija los pesos w₁, w₂, w₃ | Tú eliges después |
| **Decisión** | Automática | **Manual según contexto** |

**NSGA-II es mejor si:**
- ✅ No sabes qué priorizar (PDR vs Energy vs Latency)
- ✅ Diferentes nodos necesitan diferentes configuraciones
- ✅ Quieres ver **todas las opciones válidas**

---


---

## Resumen Rápido

1. **NSGA-II busca equilibrios** entre 3 objetivos en conflicto
2. **Usa genética:** crossover + mutación + selección
3. **Genera múltiples soluciones** (frontera de Pareto)
5. **Mejor que GA mono-objetivo** para problemas multiobjetivo
