import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import geopandas as gpd
from kneed import KneeLocator  # Para detectar el codo automáticamente

from georeferencia import GeoreferenciaMapa

class GeoreferenciaRedLoRaWAN(GeoreferenciaMapa):
    """
    Extensión de GeoreferenciaMapa que integra optimización de red LoRaWAN
    
    Hereda toda la funcionalidad de georeferenciación y añade:
    - Generación de nodos IoT desde datos de delitos
    - Optimización de posicionamiento de gateways LoRaWAN
    - Visualización de red completa con cobertura
    """
    
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda):
        """
        Args:
            archivos_especificos: lista de tuplas (archivo.csv, categoría)
            PESOS_DELITOS: diccionario con pesos por tipo de delito
            mejorcelda: tamaño óptimo de celda hexagonal (metros)
        """
        # Inicializar clase base
        super().__init__(archivos_especificos, PESOS_DELITOS, mejorcelda)

    def ejecutar_pipeline_georeferenciacion(self):
        """
        Ejecuta el pipeline completo de georeferenciación
        (Reutiliza el método de la clase base)
        """
        print("\n" + "="*70)
        print("FASE 1: GEOREFERENCIACIÓN Y ANÁLISIS DE DELITOS")
        print("="*70)
        
        # Ejecutar pipeline de la clase base
        self.mostrar_encabezado()
        self.mostrar_sistema_pesos()
        self.cargar_mapa_base()
        self.crear_grilla_hexagonal()
        self.cargar_archivos_delitos()
        self.calcular_pesos_delitos()
        self.mostrar_distribucion_delitos()
        self.georreferenciar_y_calcular_scores()
        self.clasificar_niveles_inseguridad()
        
        print("\n✓ Georeferenciación completada")





if __name__ == "__main__":
    import json
    
    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "resultados_optimizacion"
    
    with open(resultados_dir / "mejorcelda.json") as f:
        config = json.load(f)
        mejorcelda = config['mejorcelda']
    
    print(f"\n🔧 Tamaño de celda óptimo: {mejorcelda} m\n")
    
    # Definir pesos de delitos
    PESOS_DELITOS = {
        'Hurto': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
        'Extorsion': {'severidad': 1, 'factor_genero': 1.0, 'peso_total': 1.0, 'nivel': 'Bajo'},
        'Lesiones Personales': {'severidad': 2, 'factor_genero': 1.0, 'peso_total': 2.0, 'nivel': 'Moderado'},
        'Delitos Sexuales': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Violencia Intrafamiliar': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Lesiones Personales Agravados': {'severidad': 3, 'factor_genero': 1.5, 'peso_total': 4.5, 'nivel': 'Alto'},
        'Homicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'},
        'Feminicidio': {'severidad': 4, 'factor_genero': 1.5, 'peso_total': 6.0, 'nivel': 'Crítico'}
    }
    
    archivos_especificos = [
        ('data_base/Hurtos_fiscalia.csv', 'Hurto'),
        ('data_base/Homicidios_fiscalia.csv', 'Homicidio'),
        ('data_base/Delitos_Sexuales_fiscalia.csv', 'Delitos Sexuales'),
        ('data_base/Lesiones_fiscalia.csv', 'Lesiones Personales'),
        ('data_base/Violencia_Intrafamiliar_fiscalia.csv', 'Violencia Intrafamiliar'),
        ('data_base/Extorsion_fiscalia.csv', 'Extorsion')
    ]

    red_cali = GeoreferenciaRedLoRaWAN(archivos_especificos, PESOS_DELITOS, mejorcelda)
    red_cali.ejecutar_pipeline_georeferenciacion()
    
    # Buscar GeoDataFrame con los eventos georreferenciados (varias posibilidades)
    events_gdf = None
    for attr in ('casos_georreferenciados', 'df_todos_delitos', 'gdf_delitos', 'casos_dentro_cali'):
        if hasattr(red_cali, attr):
            events_gdf = getattr(red_cali, attr)
            break
    
    if events_gdf is None:
        # intentar buscar dataframes en atributos comunes
        candidates = [v for k, v in red_cali.__dict__.items() if isinstance(v, gpd.GeoDataFrame)]
        if candidates:
            events_gdf = candidates[0]
    
    if events_gdf is None or len(events_gdf) == 0:
        raise RuntimeError("No se encontró GeoDataFrame de eventos georreferenciados en la instancia. Revisa atributos de GeoreferenciaMapa.")
    
    # Asegurar geometría y sistema de referencia (projectar a metric si está en lat/lon)
    if events_gdf.crs is None:
        print("⚠ GeoDataFrame sin CRS; asumiendo EPSG:4326 y proyectando a 3857.")
        events_gdf = events_gdf.set_crs(epsg=4326, allow_override=True)
    if events_gdf.crs.to_epsg() in (4326, None):
        events_proj = events_gdf.to_crs(epsg=3857)
    else:
        events_proj = events_gdf.to_crs(events_gdf.crs)  # queda igual si ya está proyectado
    
    # Asegurar que todas las geometrías tengan un punto para extraer coordenadas
    geom_types = events_proj.geometry.geom_type.unique()
    print(f"Geometry types found: {geom_types}")
    
    # Reemplazar geometrías no-Point por su centroid (seguro para clustering)
    events_proj = events_proj.copy()
    events_proj['geom_point'] = events_proj.geometry.apply(
        lambda g: g if g is None else (g if g.geom_type == 'Point' else g.centroid)
    )

    # Filtrar filas sin geometría
    events_proj = events_proj[events_proj['geom_point'].notna()].copy()
    # Extraer coordenadas de forma segura
    coords = np.vstack([events_proj['geom_point'].x.values, events_proj['geom_point'].y.values]).T
    
    # ========== CLUSTERING GLOBAL DE TODOS LOS DELITOS ==========
    
    # Detectar columna de tipo de delito
    tipo_delito_col = None
    for col in ['categoria', 'TIPO_DELITO', 'tipo_delito', 'DELITO', 'delito', 'CONDUCTA']:
        if col in events_proj.columns:
            tipo_delito_col = col
            print(f"\n✓ Usando columna: '{tipo_delito_col}' para clasificar severidad")
            break
    
    if tipo_delito_col is None:
        print(f"\n⚠ Columnas disponibles: {events_proj.columns.tolist()}")
        raise KeyError(f"No se encontró columna de tipo de delito.")
    
    def get_severidad(tipo_delito):
        """Retorna severidad (1-4) del tipo de delito"""
        for key, val in PESOS_DELITOS.items():
            if key.lower() in str(tipo_delito).lower():
                return val['severidad']
        return 1  # Default: Bajo
    
    events_proj['severidad'] = events_proj[tipo_delito_col].apply(get_severidad)
    
    print("\n" + "="*70)
    print("📊 CLUSTERING GLOBAL DE TODOS LOS DELITOS")
    print("="*70)
    
    # --- MÉTODO DEL CODO PARA K ÓPTIMO (sobre TODOS los delitos) ---
    print("\n📈 Calculando K óptimo con método del codo...")
    
    n_samples = coords.shape[0]
    max_k = min(50, n_samples // 100)  # Máximo K razonable
    k_range = range(2, max_k + 1)
    inercias = []
    
    print(f"N delitos totales: {n_samples}")
    print(f"Probando K desde {min(k_range)} hasta {max(k_range)}...")
    
    # Calcular inercia para cada K
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(coords)
        inercias.append(kmeans.inertia_)
        if k % 5 == 0:
            print(f"  K={k:2d} | Inercia: {kmeans.inertia_:,.0f}")
    
    # Detectar codo
    try:
        kn = KneeLocator(list(k_range), inercias, curve='convex', direction='decreasing', S=1.0)
        K_opt = kn.knee
        if K_opt is None:
            raise ValueError("KneeLocator no encontró codo claro")
        print(f"\n🎯 K óptimo detectado: {K_opt}")
    except Exception as e:
        print(f"⚠ KneeLocator falló ({e}). Usando método alternativo...")
        if len(inercias) >= 3:
            second_diff = np.diff(inercias, 2)
            K_opt = list(k_range)[np.argmax(second_diff) + 1]
            print(f"🎯 K óptimo (2da derivada): {K_opt}")
        else:
            K_opt = 10
            print(f"⚠ Usando K por defecto = {K_opt}")
    
    # Aplicar K-Means final sobre TODOS los delitos
    print(f"\n🔄 Aplicando K-Means con K={K_opt} sobre {n_samples} delitos...")
    kmeans_final = KMeans(n_clusters=K_opt, random_state=42, n_init='auto')
    labels = kmeans_final.fit_predict(coords)
    centroids = kmeans_final.cluster_centers_
    
    # Añadir cluster a cada evento
    events_proj['cluster'] = labels
    
    # Estadísticas por cluster
    print("\n" + "=" * 70)
    print("🏆 RESULTADOS CLUSTERING GLOBAL")
    print("=" * 70)
    print(f"✅ K óptimo (zonas criminales): {K_opt}")
    print(f"📊 Delitos por cluster promedio: {len(coords) / K_opt:.1f}")
    print(f"📉 Inercia final: {kmeans_final.inertia_:,.0f}")
    print(f"📏 Tamaño de celda hexagonal: {mejorcelda} m")
    print("=" * 70)
    
    # Mostrar distribución de severidad por cluster
    print("\n📋 Distribución de severidad por cluster:")
    for cluster_id in range(K_opt):
        cluster_data = events_proj[events_proj['cluster'] == cluster_id]
        sev_counts = cluster_data['severidad'].value_counts().sort_index()
        print(f"\n  Cluster {cluster_id}: {len(cluster_data)} delitos")
        for sev, count in sev_counts.items():
            print(f"    Severidad {sev}: {count} casos ({count/len(cluster_data)*100:.1f}%)")
    
    # ========== VISUALIZACIÓN ==========
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Mapa base de Cali
    if hasattr(red_cali, 'cali') and red_cali.cali is not None:
        red_cali.cali.plot(ax=ax, color='lightgray', edgecolor='black', 
                          linewidth=1.2, alpha=0.3, zorder=0)
        print("\n✓ Mapa base de Cali dibujado")
    else:
        ax.set_facecolor('#f0f0f0')
        print("\n⚠ No se encontró mapa de Cali, usando fondo gris")
    
    # Colores por severidad
    colores_severidad = {
        1: 'green',      # Bajo
        2: 'yellow',     # Moderado
        3: 'orange',     # Alto
        4: 'red'         # Crítico
    }
    
    nombres_severidad = {
        1: 'Bajo (Hurto, Extorsión)',
        2: 'Moderado (Lesiones)',
        3: 'Alto (Delitos Sexuales, V. Intrafamiliar)',
        4: 'Crítico (Homicidio, Feminicidio)'
    }
    
    # Graficar delitos por severidad (mismo cluster, diferente color por severidad)
    for sev in sorted(events_proj['severidad'].unique()):
        sev_data = events_proj[events_proj['severidad'] == sev]
        sev_coords = np.vstack([sev_data['geom_point'].x.values, 
                               sev_data['geom_point'].y.values]).T
        
        ax.scatter(sev_coords[:, 0], sev_coords[:, 1], 
                  c=colores_severidad[sev], s=20, alpha=0.5, zorder=2,
                  label=f'{nombres_severidad[sev]} ({len(sev_data)} casos)')
    
    # Centroides (todos juntos, en negro)
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              c='black', marker='*', s=500, 
              edgecolors='white', linewidths=2, zorder=3,
              label=f'Centroides (n={K_opt})')
    
    # Configuración de gráfica
    ax.set_title(f'Clustering Global de Delitos (K-Means, K={K_opt})\n' + 
                f'Agrupación de {n_samples:,} delitos en {K_opt} zonas criminales',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Guardar
    proj_dir = Path(__file__).resolve().parent
    out_dir = proj_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "clustering_global.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\n✓ Imagen guardada: {out_img}")
    plt.show()
