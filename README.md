# Proyecto de investigación: IoT y seguridad ciudadana

Este repositorio reúne código y resultados para analizar delitos georreferenciados en Cali, construir mapas de riesgo, optimizar la ubicación de celdas y gateways, y analizar simulaciones de redes LoRaWAN/OMNeT++.

## Contenido del repositorio

### Archivos de la raíz

| Archivo | Contenido |
|---|---|
| `.gitignore` | Reglas que excluyen de Git datos, entornos, cachés y otros archivos generados o locales. |
| `requirements.txt` | Dependencias Python para los análisis geoespaciales, gráficos, optimización y estadística. |
| `georeferencia.py` | Pipeline principal de georreferenciación de delitos, generación de grilla hexagonal, cálculo de puntajes y mapas de riesgo. |
| `georeferencia_red.py` | Análisis y visualización de redes de gateways, con métricas de distancia y riesgo. |
| `grafica_estaciones.py` | Generación de una grilla sobre Cali y gráficos relacionados con estaciones de policía. |
| `metricas.py` | Funciones de métricas espaciales y de concentración, incluyendo Gini, Moran, Getis-Ord Gi*, IDR, entropía y gradientes. |

### `analisis_metricas_omnet/`

Contiene scripts y resultados del procesamiento de simulaciones OMNeT++ para distintos niveles de riesgo.

| Archivo o carpeta | Contenido |
|---|---|
| `analisis.py` | Lee resultados de simulación y calcula tablas de métricas por escenario. Su configuración actual guarda resultados en `final/`. |
| `comparar.py` | Compara resultados entre escenarios y genera resúmenes y gráficas. |

### `optimizacion_celda/`

Código y resultados para evaluar tamaños de celda y métodos de optimización.

| Archivo o carpeta | Contenido |
|---|---|
| `justificacion_tres_metodos.py` | Compara métodos y calcula métricas para justificar el tamaño de celda; incluye gráficas de tendencia. |
| `nsga.py` | Implementación de optimización multiobjetivo con NSGA. |
| `sclim.py` | Optimización bayesiana del tamaño de celda mediante Gaussian processes y optimizadores de `scikit-optimize`. |

### `optimizacion_gateways/`

Herramientas para generar y ubicar gateways en una red.

| Archivo o carpeta | Contenido |
|---|---|
| `ga.py` | Optimizador de ubicación mediante algoritmo genético. |
| `kmeans.py` | Optimiza ubicaciones mediante K-Means. |
| `metricas.py` | Métricas para evaluar soluciones de ubicación de gateways. |

### `images/`

Mapas, diagramas y resultados gráficos exportados por los scripts.
