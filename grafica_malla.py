
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import random

Homicidio = namedtuple("Homicidio", ["codigo_caso", "ubicacion"])

# homicidios ficticios
homicidios = []
for i in range(500):
    x = random.uniform(0, 25000)  # coordenada X en metros (0 a 25 km)
    y = random.uniform(0, 25000)  # coordenada Y en metros (0 a 25 km)
    homicidios.append(Homicidio(codigo_caso=f"H{i+1}", ubicacion=(x, y)))


def grafica_malla_real(homicidios):
    # Dimensiones aproximadas de la grilla real
    nx, ny = 500, 495   # ≈ 247.500 celdas
    matriz = np.zeros((ny, nx))  # todas las celdas empiezan vacías

    # Normalizar coordenadas de homicidios a la malla
    for h in homicidios:
        # Suponiendo que las coordenadas de h.ubicacion son (x, y) en metros y el área total es 619000000 m² ~ 25 km x 25 km
        i = int((h.ubicacion[0] / 25000) * nx)  # normalizamos X
        j = int((h.ubicacion[1] / 25000) * ny)  # normalizamos Y
        if 0 <= i < nx and 0 <= j < ny:
            matriz[j, i] += 1   # sumamos homicidio en esa celda

    plt.figure(figsize=(10, 8))
    plt.imshow(matriz, cmap='Reds', interpolation='nearest')
    plt.colorbar(label="Número de homicidios")
    plt.title("Malla real (~247.600 celdas de 50×50m)")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    plt.show()

def algoritmo_malla():
    Caso = namedtuple("Caso", ["tipo", "ubicacion"])

    # datos ficticios

    homicidios = [Caso("H", (random.uniform(0, 25000), random.uniform(0, 25000))) for _ in range(900)]
    hurtos = [Caso("R", (random.uniform(0, 25000), random.uniform(0, 25000))) for _ in range(3000)]
    vgenero = [Caso("VG", (random.uniform(0, 25000), random.uniform(0, 25000))) for _ in range(400)]


    # configuración de la malla
    nx, ny = 500, 495   # ~247.500 celdas de 50×50 m
    matriz = np.zeros((ny, nx))

    # esos de cada tipo de violencia
    pesos = {"H": 5, "VG": 3, "R": 1}  # homicidio > violencia de género > hurto

    # ubicar casos en la malla
    for lista in [homicidios, hurtos, vgenero]:
        for c in lista:
            i = int((c.ubicacion[0] / 25000) * nx)
            j = int((c.ubicacion[1] / 25000) * ny)
            if 0 <= i < nx and 0 <= j < ny:
                matriz[j, i] += pesos[c.tipo]

    # calcular QUINTILES (5 divisiones)
    valores = matriz.flatten()
    valores_no_cero = valores[valores > 0]  # ignoramos celdas vacías

    # calcular los 5 quintiles (20%, 40%, 60%, 80%, 100%)
    q1 = np.percentile(valores_no_cero, 20)  # Quintil 1
    q2 = np.percentile(valores_no_cero, 40)  # Quintil 2  
    q3 = np.percentile(valores_no_cero, 60)  # Quintil 3
    q4 = np.percentile(valores_no_cero, 80)  # Quintil 4
    q5 = np.percentile(valores_no_cero, 100) # Quintil 5 (máximo)

    print("CLASIFICACIÓN POR QUINTILES")
    print(f"Q1 (0-20%):   0.00 - {q1:.2f}  [Riesgo Muy Bajo]")
    print(f"Q2 (20-40%): {q1:.2f} - {q2:.2f}  [Riesgo Bajo]")
    print(f"Q3 (40-60%): {q2:.2f} - {q3:.2f}  [Riesgo Medio]")
    print(f"Q4 (60-80%): {q3:.2f} - {q4:.2f}  [Riesgo Alto]")
    print(f"Q5 (80-100%): {q4:.2f} - {q5:.2f}  [Riesgo Crítico]")

    print("\nESTADÍSTICAS DEL MAPA")
    print(f"Total de celdas: {nx * ny:,}")
    print(f"Celdas con incidentes: {len(valores_no_cero):,}")
    print(f"Porcentaje ocupado: {(len(valores_no_cero)/(nx*ny)*100):.1f}%")
    

    # 7. Crear matriz clasificada por quintiles
    matriz_clasificada = np.zeros_like(matriz)
    
    # Asignar valores de quintil a cada celda
    matriz_clasificada[matriz == 0] = 0  # Sin datos
    matriz_clasificada[(matriz > 0) & (matriz <= q1)] = 1  # Q1
    matriz_clasificada[(matriz > q1) & (matriz <= q2)] = 2  # Q2
    matriz_clasificada[(matriz > q2) & (matriz <= q3)] = 3  # Q3
    matriz_clasificada[(matriz > q3) & (matriz <= q4)] = 4  # Q4
    matriz_clasificada[matriz > q4] = 5  # Q5

    # Contar celdas por quintil
    for i in range(1, 6):
        count = np.sum(matriz_clasificada == i)
        pct = (count / len(valores_no_cero)) * 100 if len(valores_no_cero) > 0 else 0
        print(f"Quintil {i}: {count:,} celdas ({pct:.1f}%)")

    # 8. Graficar con colores ascendentes
    plt.figure(figsize=(12, 10))
    
    # Usar colormap que va de colores fríos a calientes (ascendente)
    # 'viridis' va de púrpura oscuro → azul → verde → amarillo → amarillo brillante
    # 'plasma' va de púrpura oscuro → magenta → naranja → amarillo
    # 'inferno' va de negro → púrpura → rojo → naranja → amarillo
    
    im = plt.imshow(matriz_clasificada, cmap="inferno", interpolation="nearest", 
                    vmin=0, vmax=5)
    
    # Crear colorbar personalizada
    cbar = plt.colorbar(im, label="Nivel de Riesgo", shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['Sin datos', 'Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Crítico'])
    
    plt.title("Mapa de Calor de Seguridad - Clasificación por Quintiles\n(Colores ascendentes: Oscuro=Seguro, Claro=Peligroso)", 
              fontsize=14, pad=20)
    plt.xlabel("Coordenada X (metros × 50)", fontsize=12)
    plt.ylabel("Coordenada Y (metros × 50)", fontsize=12)
    
    # Añadir texto explicativo
    plt.figtext(0.02, 0.02, 
                "Pesos: Homicidio=5, Violencia Género=3, Hurto=1\n" +
                f"Casos: {len(homicidios)} H, {len(vgenero)} VG, {len(hurtos)} R",
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()

    return matriz, matriz_clasificada, (q1, q2, q3, q4, q5)



def ejemplo_casos_multiples():
    """
    Ejemplo específico: qué pasa cuando varios casos caen en la misma celda
    """
    print("=" * 70)
    print("🎯 EJEMPLO: MÚLTIPLES CASOS EN LA MISMA CELDA")
    print("=" * 70)
    
    # Configuración simple
    nx, ny = 5, 5  # Malla 5x5 para ver mejor el efecto
    area_ciudad = 500
    matriz = np.zeros((ny, nx))
    pesos = {"H": 5, "VG": 3, "R": 1}
    
    Caso = namedtuple("Caso", ["tipo", "ubicacion", "descripcion"])
    
    # Casos que deliberadamente caen en las mismas celdas
    casos_ejemplo = [
        # CELDA [2,1] - Múltiples VG
        Caso("VG", (110, 210), "Violencia doméstica #1"),
        Caso("VG", (120, 220), "Violencia doméstica #2"),  
        Caso("VG", (130, 230), "Violencia doméstica #3"),
        
        # CELDA [1,3] - Mix de casos
        Caso("R", (310, 110), "Hurto en tienda"),
        Caso("VG", (320, 120), "Violencia en bar"),
        Caso("R", (330, 130), "Hurto de celular"),
        
        # CELDA [3,4] - Casos muy graves
        Caso("H", (410, 310), "Homicidio #1"),
        Caso("H", (420, 320), "Homicidio #2"),
        
        # CELDA [0,0] - Solo hurtos
        Caso("R", (50, 50), "Hurto bicicleta"),
        
        # CELDA [4,4] - Un solo VG
        Caso("VG", (450, 450), "Violencia pareja"),
    ]
    
    print("📍 CASOS Y SUS UBICACIONES:")
    print()
    
    # Procesar cada caso y mostrar el cálculo
    celdas_ocupadas = {}
    
    for caso in casos_ejemplo:
        x, y = caso.ubicacion
        i = int((x / area_ciudad) * nx)
        j = int((y / area_ciudad) * ny)
        
        # Acumular en la matriz
        if 0 <= i < nx and 0 <= j < ny:
            matriz[j, i] += pesos[caso.tipo]
            
            # Registrar para análisis
            celda_key = (j, i)
            if celda_key not in celdas_ocupadas:
                celdas_ocupadas[celda_key] = {
                    'casos': [],
                    'valor_total': 0,
                    'coordenadas': []
                }
            
            celdas_ocupadas[celda_key]['casos'].append(caso)
            celdas_ocupadas[celda_key]['valor_total'] = matriz[j, i]
            celdas_ocupadas[celda_key]['coordenadas'].append((x, y))
        
        print(f"  {caso.descripcion:20} → ({x:3.0f},{y:3.0f}) → Celda[{j},{i}] → +{pesos[caso.tipo]} puntos")
    
    print("\n" + "="*50)
    print("📊 ANÁLISIS POR CELDA:")
    print("="*50)
    
    # Definir colores según valor total
    def obtener_color_descripcion(valor_total):
        if valor_total <= 1:
            return "🟦 Azul muy claro", "Riesgo mínimo"
        elif valor_total <= 3:
            return "🟨 Amarillo", "Riesgo bajo"
        elif valor_total <= 6:
            return "🟧 Naranja", "Riesgo medio"
        elif valor_total <= 9:
            return "🟥 Rojo", "Riesgo alto"
        else:
            return "🟫 Rojo oscuro/Marrón", "Riesgo crítico"
    
    for celda_key in sorted(celdas_ocupadas.keys()):
        j, i = celda_key
        info = celdas_ocupadas[celda_key]
        valor_total = info['valor_total']
        casos = info['casos']
        
        print(f"\n🏠 CELDA [{j},{i}] - VALOR TOTAL: {valor_total}")
        
        # Mostrar composición de casos
        tipos_count = {"H": 0, "VG": 0, "R": 0}
        for caso in casos:
            tipos_count[caso.tipo] += 1
        
        composicion = []
        if tipos_count["H"] > 0:
            composicion.append(f"{tipos_count['H']} Homicidio(s)")
        if tipos_count["VG"] > 0:
            composicion.append(f"{tipos_count['VG']} Violencia(s) Género")
        if tipos_count["R"] > 0:
            composicion.append(f"{tipos_count['R']} Hurto(s)")
        
        print(f"   Composición: {', '.join(composicion)}")
        
        # Mostrar cálculo
        calculo_partes = []
        if tipos_count["H"] > 0:
            calculo_partes.append(f"{tipos_count['H']}×{pesos['H']}")
        if tipos_count["VG"] > 0:
            calculo_partes.append(f"{tipos_count['VG']}×{pesos['VG']}")
        if tipos_count["R"] > 0:
            calculo_partes.append(f"{tipos_count['R']}×{pesos['R']}")
        
        print(f"   Cálculo: {' + '.join(calculo_partes)} = {valor_total}")
        
        # Mostrar color resultante
        color_desc, riesgo_desc = obtener_color_descripcion(valor_total)
        print(f"   Color final: {color_desc} - {riesgo_desc}")
    
    # Casos especiales destacados
    print("\n" + "🎯" * 20)
    print("CASOS ESPECIALES RESPONDIDOS:")
    print("🎯" * 20)
    
    print("\n❓ '¿Qué pasa con 3 casos de VG en la misma celda?'")
    print("   ✅ Celda [2,1]: 3 VG = 3×3 = 9 puntos")
    print("   ✅ Color: 🟥 Rojo (riesgo alto)")
    print("   ✅ NO se ve como VG individual, sino como zona de alto riesgo")
    
    print("\n❓ '¿Se diferencia de 1 homicidio + 1 VG + 1 hurto?'")  
    print("   ✅ Celda [1,3]: 2R + 1VG = 2×1 + 1×3 = 5 puntos")
    print("   ✅ Color: 🟧 Naranja (riesgo medio)")
    print("   ✅ Diferente a las 3 VG (que dan 9 puntos)")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Puntos originales con celdas
    ax1.set_xlim(0, area_ciudad)
    ax1.set_ylim(0, area_ciudad)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Casos Originales\n(Múltiples en misma celda)', fontsize=12)
    
    # Dibujar malla
    for i in range(nx + 1):
        x = i * (area_ciudad / nx)
        ax1.axvline(x, color='black', alpha=0.7, linewidth=1)
    for j in range(ny + 1):
        y = j * (area_ciudad / ny)
        ax1.axhline(y, color='black', alpha=0.7, linewidth=1)
    
    # Plotear casos con diferentes colores por tipo
    colors = {"H": "red", "VG": "orange", "R": "blue"}
    sizes = {"H": 100, "VG": 60, "R": 30}
    
    for caso in casos_ejemplo:
        x, y = caso.ubicacion
        ax1.scatter(x, y, c=colors[caso.tipo], s=sizes[caso.tipo], 
                   alpha=0.8, edgecolors='black', linewidth=1)
    
    # Subplot 2: Mapa de calor final
    im = ax2.imshow(matriz, cmap='hot', interpolation='nearest', 
                    extent=[0, area_ciudad, 0, area_ciudad], origin='lower')
    ax2.set_title('Mapa de Calor Final\n(Valores acumulados)', fontsize=12)
    
    # Mostrar valores en cada celda ocupada
    for fila in range(ny):
        for col in range(nx):
            if matriz[fila, col] > 0:
                x_centro = (col + 0.5) * (area_ciudad / nx)
                y_centro = (fila + 0.5) * (area_ciudad / ny)
                ax2.text(x_centro, y_centro, f'{int(matriz[fila, col])}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white' if matriz[fila, col] > 5 else 'black',
                        fontsize=14)
    
    plt.colorbar(im, ax=ax2, label='Valor de Riesgo Acumulado')
    
    # Leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Homicidio (peso=5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=8, label='Violencia Género (peso=3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=6, label='Hurto (peso=1)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return matriz, celdas_ocupadas


# Ejecutar el ejemplo
if __name__ == "__main__":
    grafica_malla_real(homicidios)
    matriz_ejemplo = ejemplo_casos_multiples()
    
