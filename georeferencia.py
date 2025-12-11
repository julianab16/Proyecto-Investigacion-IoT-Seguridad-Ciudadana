# ========== IMPORTS ==========
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

class GeoreferenciaMapa:
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda):
        """
        Inicializa la clase de georeferenciación
        
        Args:
            archivos_especificos: lista de tuplas (archivo.csv, categoría)
            PESOS_DELITOS: diccionario con pesos por tipo de delito
            mejorcelda: tamaño óptimo de celda hexagonal (metros)
        """
        self.archivos_especificos = archivos_especificos
        self.PESOS_DELITOS = PESOS_DELITOS
        self.mejorcelda = mejorcelda
        
        # Atributos que se inicializarán durante el procesamiento
        self.cali = None
        self.grid = None
        self.grid_cali = None
        self.df = None
        self.gdf_casos = None
        self.casos_dentro_cali = None
        
    def mostrar_encabezado(self):
        """Muestra el encabezado del sistema"""
        print("=" * 70)
        print(" " * 10 + "MAPA DE CALOR DE SEGURIDAD CON ENFOQUE DE GÉNERO")
        print(" " * 20 + "Santiago de Cali, Colombia")
        print("=" * 70)
        
    def mostrar_sistema_pesos(self):
        """Muestra el sistema de pesos por tipo de delito"""
        print("\n📊 SISTEMA DE PESOS CON ENFOQUE DE GÉNERO")
        print("─" * 70)
        print(f"{'Tipo de Delito':<25} {'Severidad':<12} {'F.Género':<10} {'Peso Total':<10}")
        print("─" * 70)
        for delito, config in self.PESOS_DELITOS.items():
            print(f"{delito:<25} {config['severidad']:<12} {config['factor_genero']:<10.1f} {config['peso_total']:<10.1f}")
        print("─" * 70)
        print("💡 Los delitos con enfoque de género (violencia sexual, intrafamiliar,")
        print("   feminicidio) tienen un peso amplificado de 1.5x\n")
        
    def cargar_mapa_base(self):
        """Carga el mapa base de Cali"""
        print("\n[1/6] Cargando mapa base de Cali...")
        self.cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=None)
        self.cali = self.cali.to_crs(3116)
        print("✓ Mapa base cargado correctamente")
        
    def crear_grilla_hexagonal(self):
        """Crea la grilla hexagonal sobre Cali"""
        print("\n[2/6] Generando grilla hexagonal...")
        
        LADO_HEX = self.mejorcelda
        ancho_hex = math.sqrt(3) * LADO_HEX
        paso_x = ancho_hex
        paso_y = 1.5 * LADO_HEX
        
        xmin, ymin, xmax, ymax = self.cali.total_bounds
        xmin -= ancho_hex
        ymin -= LADO_HEX
        xmax += ancho_hex
        ymax += LADO_HEX
        
        # Generar centros de hexágonos
        centers = []
        x_vals = np.arange(xmin, xmax + paso_x, paso_x)
        y_vals = np.arange(ymin, ymax + paso_y, paso_y)
        
        for ix, x in enumerate(x_vals):
            y_offset = (paso_y / 2.0) if (ix % 2 == 1) else 0.0
            for y in y_vals:
                centers.append((x, y + y_offset))
        
        # Función para crear hexágonos
        def hexagon(center_x, center_y, s):
            angles = [i * math.pi / 3.0 for i in range(6)]
            coords = [(center_x + s * math.cos(a), center_y + s * math.sin(a)) for a in angles]
            return Polygon(coords)
        
        # Unión del área de Cali
        try:
            area_cali = self.cali.geometry.union_all()
        except AttributeError:
            area_cali = self.cali.geometry.unary_union
        
        # Crear hexágonos que intersectan con Cali
        polygons = [hexagon(cx, cy, LADO_HEX) for cx, cy in centers 
                   if hexagon(cx, cy, LADO_HEX).intersects(area_cali)]
        
        self.grid = gpd.GeoDataFrame(geometry=polygons, crs=self.cali.crs)
        self.grid = self.grid.reset_index(drop=False).rename(columns={'index':'grid_id'})
        
        print(f"✓ {len(polygons):,} hexágonos generados")
        
    @staticmethod
    def limpiar_coordenadas(df, lat_col='y', lon_col='x'):
        """Limpia y valida coordenadas"""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        
        lat_col = lat_col.lower()
        lon_col = lon_col.lower()
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return None
        
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df = df.dropna(subset=[lat_col, lon_col])
        df = df[(df[lat_col] >= 2.5) & (df[lat_col] <= 4.5)]
        df = df[(df[lon_col] >= -77.5) & (df[lon_col] <= -75.5)]
        
        return df
    
    def procesar_csv(self, archivo, nombre_categoria=None):
        """Procesa un archivo CSV y retorna DataFrame limpio"""
        try:
            print(f"\n  Cargando: {archivo}")
            df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')
            
            if len(df) == 0:
                print(f"    ⚠ Archivo vacío")
                return None
            
            print(f"    • Registros iniciales: {len(df):,}")
            
            df = self.limpiar_coordenadas(df, 'y', 'x')
            
            if df is None or len(df) == 0:
                print(f"    ⚠ Sin coordenadas válidas")
                return None
            
            df.columns = df.columns.str.strip().str.lower()
            
            if nombre_categoria:
                df['categoria'] = nombre_categoria
            else:
                nombre_archivo = archivo.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
                df['categoria'] = nombre_archivo
            
            df['archivo_fuente'] = archivo.split('/')[-1]
            
            print(f"    ✓ {len(df):,} registros procesados - Categoría: {df['categoria'].iloc[0]}")
            return df
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return None
    
    def cargar_archivos_delitos(self):
        """Carga todos los archivos CSV de delitos"""
        print("\n[3/6] Cargando archivos de delitos...")
        
        datasets = []
        archivos_encontrados = 0
        
        for archivo, categoria in self.archivos_especificos:
            try:
                df = self.procesar_csv(archivo, categoria)
                if df is not None:
                    datasets.append(df)
                    archivos_encontrados += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"  ⚠ Error procesando {archivo}: {str(e)}")
        
        if archivos_encontrados == 0:
            print("\n" + "="*70)
            print("✗ ERROR: No se encontró ningún archivo CSV válido")
            print("="*70)
            raise FileNotFoundError("No se encontraron archivos CSV válidos")
        
        print(f"\n✓ Total de archivos procesados: {archivos_encontrados}")
        
        # Concatenar datasets
        self.df = pd.concat(datasets, ignore_index=True)
        print(f"✓ Total de registros combinados: {len(self.df):,}")
        
        # Normalizar columnas
        self.df.columns = self.df.columns.str.strip().str.lower()
        
    def inspeccionar_nivel_severidad(self):
        """Inspecciona valores únicos de nivel_severidad en Lesiones"""
        print("\n🔍 INSPECCIÓN - Valores únicos de nivel_severidad en Lesiones:")
        lesiones_df = self.df[self.df['categoria'].str.contains('Lesion', case=False, na=False)].copy()
        
        if len(lesiones_df) > 0 and 'nivel_severidad' in lesiones_df.columns:
            valores_unicos = lesiones_df['nivel_severidad'].unique()
            print(f"  Valores encontrados ({len(valores_unicos)} únicos):")
            for i, val in enumerate(valores_unicos):
                count = (lesiones_df['nivel_severidad'] == val).sum()
                print(f"    {i+1}. '{val}' → {count} registros")
                
    def asignar_peso_delito(self, row):
        """Asigna peso según categoría de delito"""
        try:
            categoria = str(row['categoria']).strip() if 'categoria' in row.index else ''
            categoria_lower = categoria.lower()
            
            # 1. Búsqueda exacta
            if categoria in self.PESOS_DELITOS:
                return self.PESOS_DELITOS[categoria]['peso_total']
            
            # 2. Lesiones con nivel de severidad
            if 'lesion' in categoria_lower:
                nivel_severidad = None
                if 'nivel_severidad' in row.index:
                    valor = row['nivel_severidad']
                    if pd.notna(valor):
                        nivel_severidad = str(valor).strip().lower()
                
                if nivel_severidad is not None and nivel_severidad != '':
                    if 'nivel 3' in nivel_severidad or nivel_severidad == '3':
                        return self.PESOS_DELITOS['Lesiones Personales Agravados']['peso_total']
                    elif 'nivel 2' in nivel_severidad or nivel_severidad == '2':
                        return self.PESOS_DELITOS['Lesiones Personales']['peso_total']
                return self.PESOS_DELITOS['Lesiones Personales']['peso_total']
            
            # 3. Búsqueda flexible
            if 'hurto' in categoria_lower:
                return self.PESOS_DELITOS['Hurto']['peso_total']
            elif 'extorsion' in categoria_lower:
                return self.PESOS_DELITOS['Extorsion']['peso_total']
            elif 'delitos sexuales' in categoria_lower or 'sexo' in categoria_lower:
                return self.PESOS_DELITOS['Delitos Sexuales']['peso_total']
            elif 'intrafamiliar' in categoria_lower or 'domestica' in categoria_lower:
                return self.PESOS_DELITOS['Violencia Intrafamiliar']['peso_total']
            elif 'feminicidio' in categoria_lower:
                return self.PESOS_DELITOS['Feminicidio']['peso_total']
            elif 'homicidio' in categoria_lower:
                return self.PESOS_DELITOS['Homicidio']['peso_total']
            else:
                return 1.0
        
        except Exception as e:
            print(f"⚠ Error en asignar_peso_delito: {e}")
            return 1.0
            
    def calcular_pesos_delitos(self):
        """Calcula pesos de delitos y casos dentro de Cali"""
        print("\n[4/6] Calculando puntajes con enfoque de género...")
        
        # Inspeccionar nivel_severidad
        self.inspeccionar_nivel_severidad()
        
        # Asignar pesos
        self.df['peso_delito'] = self.df.apply(self.asignar_peso_delito, axis=1)
        
        # Georreferenciar temporalmente para filtrar casos dentro de Cali
        try:
            gdf_temp = gpd.GeoDataFrame(
                self.df, 
                geometry=[Point(lon, lat) for lon, lat in zip(self.df['x'], self.df['y'])], 
                crs="EPSG:4326"
            )
            gdf_temp = gdf_temp.to_crs(self.cali.crs)
            self.casos_dentro_cali = gdf_temp[gdf_temp.within(self.cali.geometry.iloc[0])]
        except Exception:
            self.casos_dentro_cali = gpd.GeoDataFrame(
                self.df, 
                geometry=[Point(lon, lat) for lon, lat in zip(self.df['x'], self.df['y'])], 
                crs="EPSG:4326"
            )
            
    def mostrar_distribucion_delitos(self):
        """Muestra distribución por categoría de delito"""
        print(f"\n  📊 Distribución por categoría de delito (dentro de Cali):")
        print("  " + "─" * 66)
        
        total_dentro = len(self.casos_dentro_cali)
        
        if total_dentro == 0:
            print("    ⚠ No hay registros dentro de los límites de Cali.")
            return
        
        # Separar feminicidios de homicidios
        casos_mostrar = self.casos_dentro_cali.copy()
        
        if 'archivo_fuente' in casos_mostrar.columns:
            homicidios_mask = casos_mostrar['archivo_fuente'] == 'Homicidios_fiscalia.csv'
            
            if homicidios_mask.any():
                casos_mostrar.columns = casos_mostrar.columns.str.strip().str.lower()
                
                col_feminicidio = None
                for col in ['feminicidios', 'feminicidio']:
                    if col in casos_mostrar.columns:
                        col_feminicidio = col
                        break
                
                if col_feminicidio is not None:
                    feminicidios_mask = (homicidios_mask & 
                                       (casos_mostrar[col_feminicidio].astype(str).str.upper() == 'S'))
                    homicidios_no_feminicidios_mask = (homicidios_mask & 
                                                     (casos_mostrar[col_feminicidio].astype(str).str.upper() == 'N'))
                    
                    casos_mostrar.loc[feminicidios_mask, 'categoria'] = 'Feminicidio'
                    casos_mostrar.loc[homicidios_no_feminicidios_mask, 'categoria'] = 'Homicidio (sin feminicidio)'
                    casos_mostrar.loc[feminicidios_mask, 'peso_delito'] = self.PESOS_DELITOS['Feminicidio']['peso_total']
        
        # Mostrar distribución
        for cat in sorted(casos_mostrar['categoria'].unique()):
            count = len(casos_mostrar[casos_mostrar['categoria'] == cat])
            peso = casos_mostrar[casos_mostrar['categoria'] == cat]['peso_delito'].iloc[0]
            pct = (count / total_dentro) * 100
            print(f"    {cat:<30} {count:>6,} ({pct:>5.1f}%) | Peso: {peso:.1f}")
        
        print("  " + "─" * 66)
        
    def georreferenciar_y_calcular_scores(self):
        """Georeferencia casos y calcula scores por celda"""
        print("\n[5/6] Georeferenciando y calculando puntajes por celda...")
        
        # Crear GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(self.df['x'], self.df['y'])]
        self.gdf_casos = gpd.GeoDataFrame(self.df, geometry=geometry, crs="EPSG:4326")
        self.gdf_casos = self.gdf_casos.to_crs(self.grid.crs)
        
        dentro = self.gdf_casos[self.gdf_casos.within(self.cali.geometry.iloc[0])]
        print(f"  • Casos dentro de Cali: {len(dentro):,}")
        
        # Spatial join
        casos_con_celda = gpd.sjoin(self.gdf_casos, self.grid, how="left", predicate="within")
        casos_con_celda = casos_con_celda.dropna(subset=["grid_id"])
        
        # Calcular scores
        print("\n  🧮 Calculando Score_i = Σ(N_ij × P_j)")
        puntaje_por_celda = casos_con_celda.groupby("grid_id")["peso_delito"].sum().rename("score_raw")
        frecuencia_por_celda = casos_con_celda.groupby("grid_id").size().rename("num_eventos")
        
        self.grid = self.grid.merge(puntaje_por_celda, on="grid_id", how="left")
        self.grid = self.grid.merge(frecuencia_por_celda, on="grid_id", how="left")
        self.grid["score_raw"] = self.grid["score_raw"].fillna(0.0)
        self.grid["num_eventos"] = self.grid["num_eventos"].fillna(0).astype(int)
        
        # Normalización
        print("  📏 Normalizando puntajes...")
        score_min = self.grid["score_raw"].min()
        score_max = self.grid["score_raw"].max()
        
        if score_max > score_min:
            self.grid["score_normalizado"] = (self.grid["score_raw"] - score_min) / (score_max - score_min)
        else:
            self.grid["score_normalizado"] = 0.0
        
        self.grid["indice_inseguridad"] = self.grid["score_normalizado"] * 100
        
        print(f"    • Score mínimo: {score_min:.2f}")
        print(f"    • Score máximo: {score_max:.2f}")
        print(f"    • Rango normalizado: 0.00 - 1.00")
        
        # Intersección con Cali
        self.grid_cali = gpd.overlay(self.grid, self.cali, how="intersection")
        
    def clasificar_niveles_inseguridad(self):
        """Clasifica celdas por niveles de inseguridad usando percentiles"""
        print("\n  📊 Clasificando en niveles de inseguridad (percentiles)...")
        
        indices = self.grid_cali["indice_inseguridad"].values
        clasificacion = np.zeros_like(indices, dtype=int)
        
        if np.any(indices > 0):
            p20 = np.percentile(indices[indices > 0], 20)
            p40 = np.percentile(indices[indices > 0], 40)
            p60 = np.percentile(indices[indices > 0], 60)
            p80 = np.percentile(indices[indices > 0], 80)
            
            clasificacion[indices == 0] = 0
            clasificacion[(indices > 0) & (indices <= p20)] = 1
            clasificacion[(indices > p20) & (indices <= p40)] = 2
            clasificacion[(indices > p40) & (indices <= p60)] = 3
            clasificacion[(indices > p60) & (indices <= p80)] = 4
            clasificacion[indices > p80] = 5
            
            print(f"\n    Percentiles de clasificación:")
            print(f"      • Muy Bajo:   0.00 - {p20:.2f}")
            print(f"      • Bajo:      {p20:.2f} - {p40:.2f}")
            print(f"      • Medio:     {p40:.2f} - {p60:.2f}")
            print(f"      • Alto:      {p60:.2f} - {p80:.2f}")
            print(f"      • Muy Alto:  {p80:.2f}+")
        
        self.grid_cali["nivel_inseguridad"] = clasificacion
        
        # Estadísticas
        print(f"\n  📈 Distribución de celdas por nivel:")
        for nivel in range(6):
            etiquetas = ['Sin datos', 'Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
            count = (self.grid_cali["nivel_inseguridad"] == nivel).sum()
            pct = (count / len(self.grid_cali)) * 100 if len(self.grid_cali) > 0 else 0
            print(f"    {etiquetas[nivel]:<12}: {count:>4} celdas ({pct:>5.1f}%)")
            
    def visualizar_mapa(self):
        """Genera y guarda el mapa de calor"""
        print("\n[6/6] Generando mapa de calor...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Mapa base
        self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=2.5, zorder=1)
        
        # Colormap
        colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
        
        # Hexágonos
        self.grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                           alpha=0.75, edgecolor="black", linewidth=0.3, vmin=0, vmax=5, zorder=3)
        
        # Límites administrativos
        try:
            comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                            tags={'admin_level': ['8']})
            if len(comunas) > 0:
                comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                comunas = comunas.to_crs(self.cali.crs)
                comunas.plot(ax=ax, color="none", edgecolor="black", linewidth=2.0, 
                            alpha=0.6, linestyle='-', zorder=3)
                print("  ✓ Límites administrativos cargados")
        except:
            print("  ⚠ No se pudieron cargar límites administrativos")
        
        # Barra de color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Nivel de Inseguridad", shrink=0.7, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Sin datos', 'Muy Bajo\n(0-20%)', 'Bajo\n(20-40%)',
                            'Medio\n(40-60%)', 'Alto\n(60-80%)', 'Muy Alto\n(80-100%)'])
        
        # Título
        plt.title("Mapa de Calor de Seguridad con Enfoque de Género\nSantiago de Cali", 
                 fontsize=16, pad=20, weight='bold')
        plt.xlabel("Coordenada X (metros)", fontsize=11)
        plt.ylabel("Coordenada Y (metros)", fontsize=11)
        
        # Panel de estadísticas
        dentro = self.gdf_casos[self.gdf_casos.within(self.cali.geometry.iloc[0])]
        total_eventos = len(dentro)
        celdas_activas = (self.grid_cali['num_eventos'] > 0).sum()
        total_celdas = len(self.grid_cali)
        
        stats_text = (
            f"GENERAL STATISTICS\n"
            f"{'─'*21}\n"
            f"Total events: {total_eventos:,}\n"
            f"Total cells: {total_celdas:,}\n"
            f"Cells with data: {celdas_activas:,}\n"
            f"Cells without data: {total_celdas - celdas_activas:,}"
        )
        
        plt.text(0.64, 0.16, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
        
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.86)
        plt.grid(True, alpha=0.5, linestyle='--')
        images_dir = Path(__file__).resolve().parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        out_path = images_dir / "mapa_calor_genero_cali.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print("\n✓ Mapa guardado como 'mapa_calor_genero_cali.png'")
        plt.show()
        
    def ejecutar_pipeline_completo(self):
        """Ejecuta todo el pipeline de procesamiento"""
        self.mostrar_encabezado()
        self.mostrar_sistema_pesos()
        self.cargar_mapa_base()
        self.crear_grilla_hexagonal()
        self.cargar_archivos_delitos()
        self.calcular_pesos_delitos()
        self.mostrar_distribucion_delitos()
        self.georreferenciar_y_calcular_scores()
        self.clasificar_niveles_inseguridad()
        self.visualizar_mapa()
        
        print("\n" + "="*70)
        print("✓ PROCESO COMPLETADO")
        print("="*70)


if __name__ == "__main__":
    import json
    
    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "resultados_optimizacion"
    
    with open(resultados_dir / "mejorcelda.json") as f:
        config = json.load(f)
        mejorcelda = config['mejorcelda']
        mejor_metodo_nombre = config['mejor_metodo']
    
    with open(resultados_dir / "metodos_optimizacion.json") as f:
        metodos_optimizados = json.load(f)
        print(f"\nMejor tamaño de celda: {mejorcelda} m")

    print(f"\nMejor tamaño de celda: {mejorcelda} m")
    print(f"Método ganador: {mejor_metodo_nombre}")
    for nombre in metodos_optimizados:
        h_optimo = metodos_optimizados[nombre]['h_optimo']
        print(f"Método: {nombre}, Tamaño de celda: {h_optimo:.1f}m")
    print()
    
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
    
    # Crear instancia y ejecutar
    mapa = GeoreferenciaMapa(archivos_especificos, PESOS_DELITOS, mejorcelda)
    mapa.ejecutar_pipeline_completo()