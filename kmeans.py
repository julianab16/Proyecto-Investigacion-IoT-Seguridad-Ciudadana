from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.colors as mcolors
from georeferencia import GeoreferenciaMapa

class GeoreferenciaRedLoRaWAN(GeoreferenciaMapa):
    """
    Extensión de GeoreferenciaMapa que integra clustering de coordenadas
    """
    
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda):
        super().__init__(archivos_especificos, PESOS_DELITOS, mejorcelda)

    def ejecutar_pipeline_georeferenciacion(self):
        """Ejecuta el pipeline completo de georeferenciación"""
        print("\n" + "="*70)
        print("FASE 1: GEOREFERENCIACIÓN Y ANÁLISIS DE DELITOS")
        print("="*70)
        
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

class Kmeans():
    def __init__(self, red_cali):
        self.red_cali = red_cali
    
    def ejecutar_kmeans(self):
        """Ejecuta K-Means simple sin ponderación"""
        # ========== OBTENER COORDENADAS DENTRO DE CALI ==========
        print("\n" + "="*70)
        print("📍 EXTRAYENDO COORDENADAS DE DELITOS DENTRO DE CALI")
        print("="*70)
        
        # Verificar que exista gdf_casos y cali
        if not hasattr(self.red_cali, 'gdf_casos') or self.red_cali.gdf_casos is None:
            raise RuntimeError("❌ No se encontró gdf_casos (eventos georreferenciados)")
        
        if not hasattr(self.red_cali, 'cali') or self.red_cali.cali is None:
            raise RuntimeError("❌ No se encontró el mapa base de Cali")
        
        # FILTRAR SOLO CASOS DENTRO DE CALI
        print(f"  • Total de casos georreferenciados: {len(self.red_cali.gdf_casos):,}")
        
        events_gdf = self.red_cali.gdf_casos[
            self.red_cali.gdf_casos.within(self.red_cali.cali.geometry.iloc[0])
        ].copy()
        
        print(f"  • Casos dentro de los límites de Cali: {len(events_gdf):,}")
        print(f"  • Casos filtrados (fuera de Cali): {len(self.red_cali.gdf_casos) - len(events_gdf):,}")
        
        # Proyectar a sistema métrico si es necesario
        if events_gdf.crs is None:
            print("⚠ Sin CRS, asumiendo EPSG:4326")
            events_gdf = events_gdf.set_crs(epsg=4326, allow_override=True)
        
        if events_gdf.crs.to_epsg() == 4326:
            print("🔄 Proyectando de EPSG:4326 a EPSG:3116 (Colombia Oeste)")
            events_proj = events_gdf.to_crs(epsg=3116)
        else:
            events_proj = events_gdf.copy()
            print(f"✓ Usando CRS existente: {events_proj.crs}")
        
        # Asegurar geometrías Point (convertir centroides si es necesario)
        events_proj = events_proj.copy()
        events_proj['geom_point'] = events_proj.geometry.apply(
            lambda g: g if (g is not None and g.geom_type == 'Point') else (g.centroid if g is not None else None)
        )
        events_proj = events_proj[events_proj['geom_point'].notna()].copy()
        
        # Extraer coordenadas como array NumPy
        coords = np.vstack([
            events_proj['geom_point'].x.values,
            events_proj['geom_point'].y.values
        ]).T
        
        print(f"✓ Coordenadas extraídas: {coords.shape[0]:,} puntos")
        print(f"  • X min/max: {coords[:, 0].min():.1f} / {coords[:, 0].max():.1f}")
        print(f"  • Y min/max: {coords[:, 1].min():.1f} / {coords[:, 1].max():.1f}")
        
        # ========== MÉTODO DEL CODO (K ÓPTIMO) ==========
        print("\n" + "="*70)
        print("📈 CALCULANDO K ÓPTIMO CON MÉTODO DEL CODO")
        print("="*70)
        
        n_samples = coords.shape[0]
        max_k = min(50, n_samples // 100)
        k_range = range(2, max_k + 1)
        inercias = []
        
        print(f"  • Muestras totales: {n_samples:,}")
        print(f"  • Rango de K: {min(k_range)} - {max(k_range)}")
        print(f"\n  Calculando inercias...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(coords)
            inercias.append(kmeans.inertia_)
            
            if k % 5 == 0:
                print(f"    K={k:2d} | Inercia: {kmeans.inertia_:,.0f}")
        
        # Detectar codo
        try:
            kn = KneeLocator(list(k_range), inercias, curve='convex', direction='decreasing', S=1.0)
            K_opt = kn.knee
            
            if K_opt is None:
                raise ValueError("KneeLocator no detectó codo claro")
            
            print(f"\n🎯 K óptimo detectado (Elbow Method): {K_opt}")
            
        except Exception as e:
            print(f"\n⚠ KneeLocator falló: {e}")
            print("  Usando método de 2da derivada...")
            
            if len(inercias) >= 3:
                second_diff = np.diff(inercias, 2)
                K_opt = list(k_range)[np.argmax(second_diff) + 1]
                print(f"🎯 K óptimo (2da derivada): {K_opt}")
            else:
                K_opt = 10
                print(f"⚠ Usando K por defecto: {K_opt}")
        
        # ========== K-MEANS FINAL ==========
        print("\n" + "="*70)
        print(f"🔄 APLICANDO K-MEANS CON K={K_opt}")
        print("="*70)
        
        kmeans_final = KMeans(n_clusters=K_opt, random_state=42, n_init='auto')
        labels = kmeans_final.fit_predict(coords)
        centroids = kmeans_final.cluster_centers_
        
        # Añadir cluster a cada evento
        events_proj['cluster'] = labels
        
        # ========== ESTADÍSTICAS ==========
        print("\n🏆 RESULTADOS:")
        print("─"*70)
        print(f"  ✅ Número de clusters: {K_opt}")
        print(f"  📊 Delitos por cluster (promedio): {n_samples / K_opt:.1f}")
        print(f"  📉 Inercia final: {kmeans_final.inertia_:,.0f}")
        print("─"*70)
        
        # Distribución por cluster
        print("\n📋 Distribución de delitos por cluster:")
        for cluster_id in range(K_opt):
            cluster_data = events_proj[events_proj['cluster'] == cluster_id]
            print(f"\n  Cluster {cluster_id}: {len(cluster_data):,} delitos")
            
            # Top 3 tipos de delito
            if 'categoria' in cluster_data.columns:
                top_tipos = cluster_data['categoria'].value_counts().head(3)
                for tipo, count in top_tipos.items():
                    pct = (count / len(cluster_data)) * 100
                    print(f"    • {tipo}: {count} ({pct:.1f}%)")
        
        # ========== VISUALIZACIÓN ==========
        print("\n" + "="*70)
        print("🎨 GENERANDO VISUALIZACIÓN")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Mapa base de Cali
        if hasattr(self.red_cali, 'cali') and self.red_cali.cali is not None:
            self.red_cali.cali.plot(ax=ax, color='lightgray', edgecolor='black', 
                            linewidth=1.2, alpha=0.3, zorder=0)
            print("  ✓ Mapa base de Cali cargado")
        else:
            ax.set_facecolor('#f0f0f0')
            print("  ⚠ Sin mapa base (fondo gris)")
        
        # Scatter plot de todos los delitos (coloreados por cluster)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                            c=labels, cmap='tab20', s=15, alpha=0.6, zorder=2,
                            edgecolors='black', linewidths=0.3)
        
        # Centroids (red stars)
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                c='red', marker='*', s=300, 
                edgecolors='black', linewidths=1.5, zorder=3,
                label=f'Centroids (n={K_opt})')
        
        # Configuración
        ax.set_title(f'Agrupación de Coordenadas de Delitos en Cali (K-Means, K={K_opt})\n' + 
                    f'Total: {n_samples:,} delitos dentro de Cali agrupados en {K_opt} clusters',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID', shrink=0.7)
        
        plt.tight_layout()
        
        # Guardar
        proj_dir = Path(__file__).resolve().parent
        out_dir = proj_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_img = out_dir / "clustering_coordenadas.png"
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"\n✓ Imagen guardada: {out_img}")
        
        plt.show()
    
    def ejecutar_kmeans_con_peso(self):
        """
        Ejecuta K-Means ponderado por severidad de delitos
        Crea mapa de calor con colores según peligrosidad
        """
        # ========== OBTENER COORDENADAS DENTRO DE CALI ==========
        print("\n" + "="*70)
        print("📍 EXTRAYENDO COORDENADAS DE DELITOS DENTRO DE CALI (CON PESOS)")
        print("="*70)
        
        # Verificar que exista gdf_casos y cali
        if not hasattr(self.red_cali, 'gdf_casos') or self.red_cali.gdf_casos is None:
            raise RuntimeError("❌ No se encontró gdf_casos (eventos georreferenciados)")
        
        if not hasattr(self.red_cali, 'cali') or self.red_cali.cali is None:
            raise RuntimeError("❌ No se encontró el mapa base de Cali")
        
        # FILTRAR SOLO CASOS DENTRO DE CALI
        print(f"  • Total de casos georreferenciados: {len(self.red_cali.gdf_casos):,}")
        
        events_gdf = self.red_cali.gdf_casos[
            self.red_cali.gdf_casos.within(self.red_cali.cali.geometry.iloc[0])
        ].copy()
        
        print(f"  • Casos dentro de los límites de Cali: {len(events_gdf):,}")
        print(f"  • Casos filtrados (fuera de Cali): {len(self.red_cali.gdf_casos) - len(events_gdf):,}")
        
        if len(events_gdf) == 0:
            raise RuntimeError("❌ No hay casos dentro de los límites de Cali")
        
        # Proyectar a sistema métrico
        if events_gdf.crs is None:
            print("⚠ Sin CRS, asumiendo EPSG:4326")
            events_gdf = events_gdf.set_crs(epsg=4326, allow_override=True)
        
        if events_gdf.crs.to_epsg() == 4326:
            print("🔄 Proyectando de EPSG:4326 a EPSG:3116 (Colombia Oeste)")
            events_proj = events_gdf.to_crs(epsg=3116)
        else:
            events_proj = events_gdf.copy()
            print(f"✓ Usando CRS existente: {events_proj.crs}")
        
        # Asegurar geometrías Point
        events_proj = events_proj.copy()
        events_proj['geom_point'] = events_proj.geometry.apply(
            lambda g: g if (g is not None and g.geom_type == 'Point') else (g.centroid if g is not None else None)
        )
        events_proj = events_proj[events_proj['geom_point'].notna()].copy()
        
        # Extraer coordenadas y pesos
        coords = np.vstack([
            events_proj['geom_point'].x.values,
            events_proj['geom_point'].y.values
        ]).T
        
        # Obtener pesos de delitos (ya calculados en georeferencia.py)
        if 'peso_delito' not in events_proj.columns:
            print("⚠ No se encontró 'peso_delito', asignando peso=1.0 a todos")
            events_proj['peso_delito'] = 1.0
        
        pesos = events_proj['peso_delito'].values
        
        print(f"\n✓ Coordenadas extraídas: {coords.shape[0]:,} puntos")
        print(f"  • X min/max: {coords[:, 0].min():.1f} / {coords[:, 0].max():.1f}")
        print(f"  • Y min/max: {coords[:, 1].min():.1f} / {coords[:, 1].max():.1f}")
        print(f"  • Peso min/max: {pesos.min():.1f} / {pesos.max():.1f}")
        
        # ========== NO EXPANDIR DATOS, USAR DIRECTAMENTE ==========
        print("\n⚖️  Usando pesos como sample_weight en K-Means...")
        print(f"  • Total de puntos únicos: {len(coords):,}")
        
        # ========== MÉTODO DEL CODO (K ÓPTIMO) ==========
        print("\n" + "="*70)
        print("📈 CALCULANDO K ÓPTIMO CON MÉTODO DEL CODO")
        print("="*70)
        
        n_samples = coords.shape[0]
        max_k = 3222
        k_range = range(2, max_k + 1)
        inercias = []
        
        print(f"  • Muestras totales: {n_samples:,}")
        print(f"  • Rango de K: {min(k_range)} - {max(k_range)}")
        print(f"\n  Calculando inercias (con ponderación)...")
        """
        for k in k_range:
            # ✅ USAR sample_weight EN LUGAR DE EXPANDIR DATOS
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(coords, sample_weight=pesos)
            inercias.append(kmeans.inertia_)
            
            if k % 5 == 0:
                print(f"    K={k:2d} | Inercia: {kmeans.inertia_:,.0f}")
        
        # Detectar codo
        try:
            kn = KneeLocator(list(k_range), inercias, curve='convex', direction='decreasing', S=1.0)
            K_opt = kn.knee
            
            if K_opt is None:
                raise ValueError("KneeLocator no detectó codo claro")
            
            print(f"\n🎯 K óptimo detectado (Elbow Method): {K_opt}")
            
        except Exception as e:
            print(f"\n⚠ KneeLocator falló: {e}")
            print("  Usando método de 2da derivada...")
            
            if len(inercias) >= 3:
                second_diff = np.diff(inercias, 2)
                K_opt = list(k_range)[np.argmax(second_diff) + 1]
                print(f"🎯 K óptimo (2da derivada): {K_opt}")
            else:
                K_opt = 10
                print(f"⚠ Usando K por defecto: {K_opt}")
                """
        K_opt = 1000 #3326
        # ========== K-MEANS FINAL CON PESOS ==========
        print("\n" + "="*70)
        print(f"🔄 APLICANDO K-MEANS PONDERADO CON K={K_opt}")
        print("="*70)
        
        # ✅ USAR sample_weight DIRECTAMENTE
        kmeans_final = KMeans(n_clusters=K_opt, random_state=42, n_init='auto')
        kmeans_final.fit(coords, sample_weight=pesos)
        centroids = kmeans_final.cluster_centers_
        
        # Asignar cada delito a su cluster más cercano
        labels = kmeans_final.predict(coords)
        events_proj['cluster'] = labels
        
        # ========== CALCULAR PELIGROSIDAD POR CLUSTER ==========
        print("\n🎯 Calculando nivel de peligrosidad por cluster...")
        
        peligrosidad_cluster = {}
        for cluster_id in range(K_opt):
            cluster_data = events_proj[events_proj['cluster'] == cluster_id]
            
            # Score total del cluster = suma de pesos
            score_total = cluster_data['peso_delito'].sum()
            num_delitos = len(cluster_data)
            score_promedio = score_total / num_delitos if num_delitos > 0 else 0
            
            peligrosidad_cluster[cluster_id] = {
                'score_total': score_total,
                'num_delitos': num_delitos,
                'score_promedio': score_promedio
            }
        
        # Normalizar scores a rango [0, 5] (niveles de peligrosidad)
        scores_totales = np.array([p['score_total'] for p in peligrosidad_cluster.values()])
        # Calcular percentiles solo para clusters con datos
        scores_con_datos = scores_totales[scores_totales > 0]

        if len(scores_con_datos) > 0:
            p20 = np.percentile(scores_con_datos, 20)
            p40 = np.percentile(scores_con_datos, 40)
            p60 = np.percentile(scores_con_datos, 60)
            p80 = np.percentile(scores_con_datos, 80)
            
            print(f"\n  📊 Percentiles de peligrosidad:")
            print(f"    • P20 (Muy Bajo): {p20:.2f}")
            print(f"    • P40 (Bajo):     {p40:.2f}")
            print(f"    • P60 (Medio):    {p60:.2f}")
            print(f"    • P80 (Alto):     {p80:.2f}")
            print(f"    • Max (Muy Alto): {scores_con_datos.max():.2f}")
        else:
            p20 = p40 = p60 = p80 = 0

        # Asignar niveles basados en percentiles
        for cluster_id in peligrosidad_cluster:
            score = peligrosidad_cluster[cluster_id]['score_total']
            
            # Clasificar en niveles 0-5 usando percentiles
            if score == 0:
                nivel = 0  # Sin datos
            elif score <= p20:
                nivel = 1  # Muy Bajo (0-20%)
            elif score <= p40:
                nivel = 2  # Bajo (20-40%)
            elif score <= p60:
                nivel = 3  # Medio (40-60%)
            elif score <= p80:
                nivel = 4  # Alto (60-80%)
            else:
                nivel = 5  # Muy Alto (80-100%)
            
            # Calcular score normalizado para referencia
            if np.max(scores_totales) > 0:
                score_norm = score / np.max(scores_totales)
            else:
                score_norm = 0.0
            
            peligrosidad_cluster[cluster_id]['nivel'] = nivel
            peligrosidad_cluster[cluster_id]['score_normalizado'] = score_norm
            
        # ========== ESTADÍSTICAS ==========
        print("\n🏆 RESULTADOS:")
        print("─"*70)
        print(f"  ✅ Número de clusters: {K_opt}")
        print(f"  📊 Delitos por cluster (promedio): {len(events_proj) / K_opt:.1f}")
        print(f"  📉 Inercia final: {kmeans_final.inertia_:,.0f}")
        print("─"*70)
                
        print("\n📋 Peligrosidad por cluster (ordenado de más a menos peligroso):")
        clusters_ordenados = sorted(
            peligrosidad_cluster.items(),
            key=lambda x: x[1]['score_total'],
            reverse=True
        )
        
        for cluster_id, stats in clusters_ordenados:
            """
            nivel = stats['nivel']
            print(f"\n  🔴 Cluster {cluster_id}: {etiquetas[nivel]} (Nivel {nivel})")
            print(f"      • Delitos: {stats['num_delitos']:,}")
            print(f"      • Score total: {stats['score_total']:.1f}")
            print(f"      • Score promedio: {stats['score_promedio']:.2f}")
            print(f"      • Score normalizado: {stats['score_normalizado']:.3f}")
            
            # Top 3 tipos de delito
            cluster_data = events_proj[events_proj['cluster'] == cluster_id]
            if 'categoria' in cluster_data.columns:
                top_tipos = cluster_data['categoria'].value_counts().head(3)
                for tipo, count in top_tipos.items():
                    pct = (count / len(cluster_data)) * 100
                    print(f"        - {tipo}: {count} ({pct:.1f}%)")
            """
        
        # ========== VISUALIZACIÓN MAPA DE CALOR ==========
        print("\n" + "="*70)
        print("🎨 GENERANDO MAPA DE CALOR DE PELIGROSIDAD")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Mapa base de Cali
        if hasattr(self.red_cali, 'cali') and self.red_cali.cali is not None:
            self.red_cali.cali.plot(ax=ax, color='lightgray', edgecolor='black', 
                                linewidth=1.2, alpha=0.3, zorder=0)
            print("  ✓ Mapa base de Cali cargado")
        
        # Colormap personalizado (igual que georeferencia.py)
        colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
        
        # Asignar colores a cada punto según nivel de su cluster
        colores_puntos = [peligrosidad_cluster[label]['nivel'] for label in labels]
        
        # Scatter de delitos (coloreados por nivel de peligrosidad)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                            c=colores_puntos, cmap=cmap, s=20, alpha=0.75,
                            edgecolors='black', linewidths=0.3, vmin=0, vmax=5, zorder=3)
        # Centroides con color según peligrosidad
        centroid_colors = [peligrosidad_cluster[i]['nivel'] for i in range(K_opt)]
        centroid_sizes = [peligrosidad_cluster[i]['num_delitos'] for i in range(K_opt)]
        
        # Normalizar tamaños (min=200, max=800)
        size_min, size_max = 100, 200
        if max(centroid_sizes) > min(centroid_sizes):
            sizes_norm = [
                size_min + (s - min(centroid_sizes)) / (max(centroid_sizes) - min(centroid_sizes)) * (size_max - size_min)
                for s in centroid_sizes
            ]
        else:
            sizes_norm = [size_min] * len(centroid_sizes)
        
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                c="black", cmap=cmap, marker='*', s=sizes_norm, 
                edgecolors='white', linewidths=1.2, zorder=4, vmin=0, vmax=5,
                label=f'Centroides (n={K_opt})')
        # Configuración
        ax.set_title(
            f'Danger Heatmap by Clustering\n' + 
            f'Santiago de Cali - {len(events_proj):,} crimes grouped in {K_opt} zones',
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label="Level of Insecurity", 
                        shrink=0.7, ticks=[0, 1, 2, 3, 4, 5])
        cbar.ax.set_yticklabels(['No data', 'Very Low\n(0-20%)', 'Low\n(20-40%)',
                            'Medium\n(40-60%)', 'High\n(60-80%)', 'Very High\n(80-100%)'])
        
        plt.tight_layout()
        
        # Guardar
        proj_dir = Path(__file__).resolve().parent
        out_dir = proj_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_img = out_dir / "mapa_calor_peligrosidad_kmeans.png"
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"\n✓ Imagen guardada: {out_img}")
        
        plt.show()
            

if __name__ == "__main__":
    # ========== CONFIGURACIÓN ==========
    resultados_dir = Path(__file__).resolve().parent / "resultados_optimizacion"
    
    with open(resultados_dir / "mejorcelda.json") as f:
        config = json.load(f)
        mejorcelda = config['mejorcelda']
    
    print(f"\n🔧 Tamaño de celda óptimo: {mejorcelda} m\n")
    
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

    # ========== EJECUTAR GEOREFERENCIACIÓN ==========
    red_cali = GeoreferenciaRedLoRaWAN(archivos_especificos, PESOS_DELITOS, mejorcelda)
    red_cali.ejecutar_pipeline_georeferenciacion()

    print("\n" + "="*70)
    print("FASE 2: CLUSTERING DE COORDENADAS CON PONDERACIÓN POR PELIGROSIDAD")
    print("="*70)
    
    kmeans_analyzer = Kmeans(red_cali)
    resultados = kmeans_analyzer.ejecutar_kmeans_con_peso()
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)