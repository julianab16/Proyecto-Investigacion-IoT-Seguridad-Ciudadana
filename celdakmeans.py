"""
Script para calcular el tamaño óptimo de celda hexagonal
usando el método del codo sobre métricas de cobertura y uniformidad
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from kneed import KneeLocator
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Importar georeferenciación
sys.path.insert(0, str(Path(__file__).parent))
from georeferencia import GeoreferenciaMapa

def calcular_metrica_calidad(grid_gdf, eventos_gdf):
    """
    Calcula métrica de calidad del tamaño de celda basándose en:
    - Cobertura: % de celdas con al menos 1 evento
    - Uniformidad: inverso de desviación estándar de eventos/celda
    - Balance: penaliza celdas muy llenas o muy vacías
    
    Returns:
        score: métrica combinada (mayor es mejor)
    """
    # Contar eventos por celda
    eventos_por_celda = []
    for idx, celda in grid_gdf.iterrows():
        eventos_en_celda = eventos_gdf[eventos_gdf.within(celda.geometry)]
        eventos_por_celda.append(len(eventos_en_celda))
    
    eventos_por_celda = np.array(eventos_por_celda)
    
    # Métricas
    cobertura = np.sum(eventos_por_celda > 0) / len(eventos_por_celda)  # 0-1
    
    if eventos_por_celda.max() > 0:
        uniformidad = 1 / (1 + np.std(eventos_por_celda))  # 0-1 (mayor es más uniforme)
    else:
        uniformidad = 0
    
    # Penalizar extremos (celdas demasiado llenas o vacías)
    eventos_promedio = np.mean(eventos_por_celda[eventos_por_celda > 0]) if np.any(eventos_por_celda > 0) else 0
    balance = 1 / (1 + abs(eventos_promedio - 10))  # Ideal ~10 eventos/celda
    
    # Score combinado (ajustar pesos según prioridad)
    score = 0.4 * cobertura + 0.4 * uniformidad + 0.2 * balance
    
    return score, cobertura, uniformidad, balance, eventos_por_celda

def optimizar_tamaño_celda(archivos_especificos, PESOS_DELITOS):
    """
    Encuentra el tamaño óptimo de celda hexagonal usando método del codo
    
    Returns:
        tamano_optimo: tamaño en metros
        resultados: diccionario con todas las métricas
    """
    print("\n" + "="*70)
    print("🔍 OPTIMIZACIÓN DE TAMAÑO DE CELDA HEXAGONAL")
    print("="*70)
    
    # Rango de tamaños a probar (metros)
    tamanos = np.arange(50, 1001, 50)  # 50m hasta 1000m en pasos de 50m
    scores = []
    coberturas = []
    uniformidades = []
    balances = []
    num_celdas = []
    
    print(f"\n📏 Probando tamaños desde {tamanos[0]}m hasta {tamanos[-1]}m...")
    print(f"   (Total: {len(tamanos)} iteraciones)\n")
    
    for tamano in tamanos:
        # Crear instancia temporal de georeferenciación
        gm = GeoreferenciaMapa(archivos_especificos, PESOS_DELITOS, tamano)
        
        try:
            # Ejecutar solo lo necesario
            gm.cargar_mapa_base()
            gm.crear_grilla_hexagonal()
            gm.cargar_archivos_delitos()
            gm.calcular_pesos_delitos()
            gm.grid_cali = gpd.overlay(gm.grid, gm.cali, how="intersection")

            # Obtener eventos georreferenciados
            eventos_gdf = gm.casos_dentro_cali
            
            if eventos_gdf is None or len(eventos_gdf) == 0:
                print(f"⚠ Tamaño {tamano}m: sin eventos válidos, omitiendo...")
                scores.append(0)
                coberturas.append(0)
                uniformidades.append(0)
                balances.append(0)
                num_celdas.append(0)
                continue
            
            if gm.grid_cali is None or len(gm.grid_cali) == 0:
                print(f"⚠ Tamaño {tamano}m: grid_cali vacío, omitiendo...")
                scores.append(0)
                coberturas.append(0)
                uniformidades.append(0)
                balances.append(0)
                num_celdas.append(0)
                continue

            # Calcular métricas
            score, cob, unif, bal, eventos_arr = calcular_metrica_calidad(gm.grid_cali, eventos_gdf)
            
            scores.append(score)
            coberturas.append(cob)
            uniformidades.append(unif)
            balances.append(bal)
            num_celdas.append(len(gm.grid_cali))
            
            if tamano % 200 == 0:  # Mostrar progreso cada 200m
                print(f"  ✓ {tamano:4d}m | Score: {score:.3f} | Celdas: {len(gm.grid_cali):5d} | Cobertura: {cob*100:.1f}%")
        
        except Exception as e:
            print(f"  ✗ {tamano}m: Error ({e})")
            scores.append(0)
            coberturas.append(0)
            uniformidades.append(0)
            balances.append(0)
            num_celdas.append(0)
    
    # Convertir a arrays
    scores = np.array(scores)
    tamanos_validos = tamanos[scores > 0]
    scores_validos = scores[scores > 0]
    
    if len(scores_validos) == 0:
        raise RuntimeError("No se pudo calcular ninguna métrica válida")
    
    # Detectar codo (buscamos el punto donde empieza a disminuir la mejora)
    # Invertir scores porque queremos el punto donde deja de mejorar mucho
    try:
        kn = KneeLocator(tamanos_validos.tolist(), scores_validos.tolist(), 
                        curve='concave', direction='increasing', S=1.0)
        tamano_optimo = kn.knee
        if tamano_optimo is None:
            # Fallback: usar máximo score
            tamano_optimo = tamanos_validos[np.argmax(scores_validos)]
            print(f"\n⚠ KneeLocator no encontró codo, usando máximo score")
    except:
        tamano_optimo = tamanos_validos[np.argmax(scores_validos)]
        print(f"\n⚠ Error en KneeLocator, usando máximo score")
    
    print(f"\n🎯 Tamaño óptimo detectado: {tamano_optimo} metros")
    
    # Graficar curva
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score combinado
    ax1.plot(tamanos, scores, 'bo-', linewidth=2, markersize=5)
    ax1.axvline(tamano_optimo, color='red', linestyle='--', linewidth=2, 
                label=f'Óptimo = {tamano_optimo}m')
    ax1.set_xlabel('Tamaño de celda (m)')
    ax1.set_ylabel('Score de calidad')
    ax1.set_title('Método del Codo - Score Combinado')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cobertura
    ax2.plot(tamanos, np.array(coberturas)*100, 'go-', linewidth=2, markersize=5)
    ax2.axvline(tamano_optimo, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tamaño de celda (m)')
    ax2.set_ylabel('Cobertura (%)')
    ax2.set_title('Cobertura de Eventos')
    ax2.grid(True, alpha=0.3)
    
    # Uniformidad
    ax3.plot(tamanos, uniformidades, 'mo-', linewidth=2, markersize=5)
    ax3.axvline(tamano_optimo, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Tamaño de celda (m)')
    ax3.set_ylabel('Uniformidad')
    ax3.set_title('Uniformidad de Distribución')
    ax3.grid(True, alpha=0.3)
    
    # Número de celdas
    ax4.plot(tamanos, num_celdas, 'co-', linewidth=2, markersize=5)
    ax4.axvline(tamano_optimo, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Tamaño de celda (m)')
    ax4.set_ylabel('Número de celdas')
    ax4.set_title('Cantidad de Celdas Generadas')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Optimización de Tamaño de Celda Hexagonal', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar
    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "optimizacion_tamano_celda.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {out_dir / 'optimizacion_tamano_celda.png'}")
    plt.show()
    
    # Resultados
    idx_optimo = np.where(tamanos == tamano_optimo)[0][0]
    resultados = {
        'tamano_optimo': int(tamano_optimo),
        'score': float(scores[idx_optimo]),
        'cobertura': float(coberturas[idx_optimo]),
        'uniformidad': float(uniformidades[idx_optimo]),
        'balance': float(balances[idx_optimo]),
        'num_celdas': int(num_celdas[idx_optimo]),
        'todos_tamanos': tamanos.tolist(),
        'todos_scores': scores.tolist()
    }
    
    # Guardar JSON
    resultados_dir = Path(__file__).parent / "resultados_optimizacion"
    resultados_dir.mkdir(exist_ok=True)
    with open(resultados_dir / "optimizacion_celda.json", 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"✓ Resultados guardados: {resultados_dir / 'optimizacion_celda.json'}")
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("🏆 RESULTADOS DE OPTIMIZACIÓN")
    print("=" * 70)
    print(f"✅ Tamaño óptimo: {tamano_optimo} metros")
    print(f"📊 Score de calidad: {scores[idx_optimo]:.4f}")
    print(f"📐 Cobertura: {coberturas[idx_optimo]*100:.1f}%")
    print(f"📏 Uniformidad: {uniformidades[idx_optimo]:.4f}")
    print(f"⚖️  Balance: {balances[idx_optimo]:.4f}")
    print(f"🔢 Número de celdas: {num_celdas[idx_optimo]}")
    print("=" * 70)
    
    return tamano_optimo, resultados


if __name__ == "__main__":
    # Definir pesos de delitos
    PESOS_DELITOS = {
        'Hurto': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0},
        'Extorsion': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0},
        'Lesiones Personales': {'severidad': 2, 'factor_genero': 1.0, 'peso_total': 2.0},
        'Delitos Sexuales': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5},
        'Violencia Intrafamiliar': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5},
        'Homicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0},
        'Feminicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0}
    }
    
    archivos_especificos = [
        ('data_base/Hurtos_fiscalia.csv', 'Hurto'),
        ('data_base/Homicidios_fiscalia.csv', 'Homicidio'),
        ('data_base/Delitos_Sexuales_fiscalia.csv', 'Delitos Sexuales'),
        ('data_base/Lesiones_fiscalia.csv', 'Lesiones Personales'),
        ('data_base/Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
        ('data_base/Extorsion_fiscalia.csv', 'Extorsion')
    ]
    
    # Ejecutar optimización
    tamano_optimo, resultados = optimizar_tamaño_celda(archivos_especificos, PESOS_DELITOS)
    
    # Guardar para usar en otros scripts
    resultado_simple = {'mejorcelda': tamano_optimo}
    resultados_dir = Path(__file__).parent / "resultados_optimizacion"
    with open(resultados_dir / "mejorcelda.json", 'w') as f:
        json.dump(resultado_simple, f, indent=2)
    
    print(f"\n✅ Valor guardado en mejorcelda.json para uso en otros scripts")