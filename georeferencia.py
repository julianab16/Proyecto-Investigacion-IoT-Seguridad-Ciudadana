# ========== IMPORTS ==========
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import libpysal, warnings, contextlib, builtins, sys, os, logging
from esda.moran import Moran
from libpysal.weights import KNN
from matplotlib.patches import Rectangle

class GeoreferenciaMapa:
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda, bbox):
        """
        Inicializa la clase de georeferenciación
        
        Args:
            archivos_especificos: lista de tuplas (archivo.csv, categoría)
            PESOS_DELITOS: diccionario con pesos por tipo de delito
            mejorcelda: tamaño óptimo de celda hexagonal (metros)
            bbox: tupla (minx, maxx, miny, maxy) para recortar el mapa (EPSG:3116)
        """
        self.archivos_especificos = archivos_especificos
        self.PESOS_DELITOS = PESOS_DELITOS
        self.mejorcelda = mejorcelda
        self.bbox = bbox  # Bounding box EXACTO (EPSG:3116)
        
        # Atributos que se inicializarán durante el procesamiento
        self.cali = None
        self.grid = None
        self.grid_cali = None
        self.df = None
        self.gdf_casos = None
        self.casos_dentro_cali = None

    def _crear_pesos(self):
        """Crea la matriz de pesos sin imprimir warnings de islands."""

        @contextlib.contextmanager
        def _silence_all():
            old_print = builtins.print
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            fnull = open(os.devnull, 'w')
            prev_disable = logging.root.manager.disable
            try:
                builtins.print = lambda *args, **kwargs: None
                sys.stdout = fnull
                sys.stderr = fnull
                logging.disable(logging.CRITICAL)
                yield
            finally:
                builtins.print = old_print
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                logging.disable(prev_disable)
                fnull.close()

        with _silence_all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    w = libpysal.weights.Queen.from_dataframe(self.grid_cali, use_index=True)
                except Exception:
                    w = libpysal.weights.Rook.from_dataframe(self.grid_cali, use_index=True)
        return w
        
    def mostrar_encabezado(self):
        """Muestra el encabezado del sistema"""
        print("=" * 70)
        print(" " * 10 + "MAPA DE CALOR DE SEGURIDAD CON ENFOQUE DE GÉNERO")
        print(" " * 20 + "Santiago de Cali, Colombia")
        print("=" * 70)
        
    def mostrar_sistema_pesos(self):
        """Muestra el sistema de pesos por tipo de delito"""
        print("\n SISTEMA DE PESOS CON ENFOQUE DE GÉNERO")
        print("─" * 70)
        print(f"{'Tipo de Delito':<25} {'Severidad':<12} {'F.Género':<10} {'Peso Total':<10}")
        print("─" * 70)
        for delito, config in self.PESOS_DELITOS.items():
            print(f"{delito:<25} {config['severidad']:<12} {config['factor_genero']:<10.1f} {config['peso_total']:<10.1f}")
        print("─" * 70)

    def cargar_mapa_base(self):
        """Carga el mapa base de Cali"""
        print("\n[1/6] Cargando mapa base de Cali...")
        self.cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=None)
        self.cali = self.cali.to_crs(3116)
        
        # Si hay bounding box definido, recortar el mapa
        if self.bbox is not None:
            minx, maxx, miny, maxy = self.bbox
            bbox_polygon = box(minx, miny, maxx, maxy)
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:3116")
            self.cali = gpd.overlay(self.cali, bbox_gdf, how='intersection')
        
        # Calcular e imprimir el área de Cali
        area_m2 = self.cali.geometry.area.sum()
        area_km2 = area_m2 / 1e6
        print(f"\n   Área de Cali: {area_m2:,.2f} m² ({area_km2:,.2f} km²)")
        
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
            
            print(f"    • Registros iniciales: {len(df):,}")
            
            df = self.limpiar_coordenadas(df, 'y', 'x')
            
            df.columns = df.columns.str.strip().str.lower()
            
            if nombre_categoria:
                df['categoria'] = nombre_categoria
            else:
                nombre_archivo = archivo.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
                df['categoria'] = nombre_archivo
            
            df['archivo_fuente'] = archivo.split('/')[-1]
            
            #print(f"    {len(df):,} registros procesados - Archivo: {df['categoria'].iloc[0]}")
            return df
            
        except Exception as e:
            print(f"    Error: {str(e)}")
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
                print(f"  Error procesando {archivo}: {str(e)}")
        
        print(f"\n✓ Total de archivos procesados: {archivos_encontrados}")
        
        # Concatenar datasets
        self.df = pd.concat(datasets, ignore_index=True)
        print(f" Total de registros combinados: {len(self.df):,}")
        
        # Normalizar columnas
        self.df.columns = self.df.columns.str.strip().str.lower()
        
    def inspeccionar_nivel_severidad(self):
        """Inspecciona valores únicos de nivel_severidad en Lesiones"""
        print("\n INSPECCIÓN - Valores únicos de nivel_severidad en Lesiones:")
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
        print(f"\n   Distribución por categoría de delito (dentro de Cali):")
        print("  " + "─" * 66)
        
        total_dentro = len(self.casos_dentro_cali)
        
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
        print("\n   Calculando Score_i = Σ(N_ij × P_j)")
        puntaje_por_celda = casos_con_celda.groupby("grid_id")["peso_delito"].sum().rename("score_raw")
        frecuencia_por_celda = casos_con_celda.groupby("grid_id").size().rename("num_eventos")
        
        self.grid = self.grid.merge(puntaje_por_celda, on="grid_id", how="left")
        self.grid = self.grid.merge(frecuencia_por_celda, on="grid_id", how="left")
        self.grid["score_raw"] = self.grid["score_raw"].fillna(0.0)
        self.grid["num_eventos"] = self.grid["num_eventos"].fillna(0).astype(int)
        
        # Normalización
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
        print("\n   Clasificando en niveles de inseguridad (percentiles)...")
        
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
        print(f"\n   Distribución de celdas por nivel:")
        for nivel in range(6):
            etiquetas = ['Sin datos', 'Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
            count = (self.grid_cali["nivel_inseguridad"] == nivel).sum()
            pct = (count / len(self.grid_cali)) * 100 if len(self.grid_cali) > 0 else 0
            print(f"    {etiquetas[nivel]:<12}: {count:>4} celdas ({pct:>5.1f}%)")
            
    def visualizar_mapa(self):
        """Genera y guarda el mapa de calor"""
        print("\n Generando mapa de calor...")
        plt.rcParams['font.family'] = 'Arial'
        textwidth_pt = 472.03123
        fig_width_in = textwidth_pt / 72.27         # ≈ 6.531 in
        fig_height_in = fig_width_in * 0.75         # proporción deseada, ajústala a tu gusto
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
                
        # Mapa base
        self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=1.0, zorder=1)
        
        # Si hay bounding box, establecer límites exactos
        if self.bbox is not None:
            minx, maxx, miny, maxy = self.bbox
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        
        # Colormap
        colors = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.LinearSegmentedColormap.from_list("seguridad", colors, N=6)
        
        # Hexágonos
        self.grid_cali.plot(ax=ax, column="nivel_inseguridad", cmap=cmap,
                           alpha=0.75, edgecolor="black", linewidth=0.3, vmin=0, vmax=5, zorder=3)
        
        # Guardar mapa simple (solo mapa base + hexágonos, sin ejes)
        images_dir = Path(__file__).resolve().parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        ax.axis('off')  # Ocultar ejes
        out_path_simple = images_dir / "mapa_calor_solo_cali.png"
        plt.savefig(out_path_simple, dpi=600, bbox_inches='tight', pad_inches=0)
        print(f"  Mapa simple guardado: {out_path_simple.name}")
        ax.axis('on')  # Volver a mostrar ejes para el mapa completo

        # Límites administrativos
        try:
            comunas = ox.features_from_place("Santiago de Cali, Colombia", 
                                            tags={'admin_level': ['8']})
            if len(comunas) > 0:
                comunas = comunas[comunas.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                comunas = comunas.to_crs(self.cali.crs)
                comunas.plot(ax=ax, color="none", edgecolor="black", linewidth=1.0, 
                            alpha=0.6, linestyle='-', zorder=3)
                print("  Límites administrativos cargados")
        except:
            print("  No se pudieron cargar límites administrativos")
        

        plt.xlabel("X Coordinate (m)", fontsize=9)
        plt.ylabel("Y Coordinate (m)", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

        # Barra de color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Level of Insecurity", shrink=0.7, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['No data', 'Very Low\n(0-20%)', 'Low\n(20-40%)',
                            'Medium\n(40-60%)', 'High\n(60-80%)', 'Very High\n(80-100%)'])
        cbar.set_label("Level of Insecurity", fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        
        # Panel de estadísticas
        dentro = self.gdf_casos[self.gdf_casos.within(self.cali.geometry.iloc[0])]
        total_eventos = len(dentro)
        celdas_activas = (self.grid_cali['num_eventos'] > 0).sum()
        total_celdas = len(self.grid_cali)
        
        stats_text = (
            f"GENERAL STATISTICS\n"
            f"{'─'*15}\n"
            f"Total events: {total_eventos:,}\n"
            f"Total cells: {total_celdas:,}\n"
            f"Cells with data: {celdas_activas:,}\n"
            f"Cells without data: {total_celdas - celdas_activas:,}"
        )
        
        plt.text(0.615, 0.02, stats_text, transform=ax.transAxes, fontsize=7.1,
                verticalalignment='bottom', family='Arial',linespacing=1,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.6))
        
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.86)
        plt.grid(True, alpha=0.5, linestyle='--')
        
        # Guardar mapa completo (con todos los elementos)
        out_path_completo = images_dir / "mapa_calor_genero_cali.pdf"
        plt.savefig(out_path_completo, format='pdf', bbox_inches='tight')
        print(f" Mapa completo guardado: {out_path_completo.name}")
        
        #plt.show()
    
    def calcular_gini_espacial(self):
        """
        Calcula el Índice de Gini espacial sobre num_eventos (delitos por celda).
        Incluye todas las celdas (incluidas las de 0 eventos) para medir concentración real.
        """
        print("\n[Análisis] Calculando Índice de Gini espacial (num_eventos)...")
    
        # Serie de eventos por celda (incluye ceros)
        y = self.grid_cali.get('num_eventos', pd.Series(dtype=int)).fillna(0).astype(float).values
        n = len(y)
    
        # Fórmula estándar del Gini (orden ascendente). Si suma_total=0 => Gini=0 (uniforme)
        y_sorted = np.sort(y)
        suma_total = float(np.sum(y_sorted))
        if suma_total == 0:
            gini = 0.0
        else:
            i = np.arange(1, n + 1, dtype=float)
            gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
            gini = max(0.0, min(1.0, gini))
    
        # Concentración: porcentaje de eventos en el top 20% de celdas (por eventos)
        k_top = max(1, int(math.ceil(0.2 * n)))
        top_vals = np.sort(y)[-k_top:]
        concentracion_top20 = 0.0 if suma_total == 0 else (np.sum(top_vals) / suma_total) * 100.0
    
        # Estadísticas
        media = float(np.mean(y))
        mediana = float(np.median(y))
        desviacion_std = float(np.std(y))
        cv = 0.0 if media == 0 else (desviacion_std / media) * 100.0
        total_eventos = int(suma_total)
        total_celdas = n
        celdas_con_eventos = int(np.sum(y > 0))
    
        resultados = {
            'gini_espacial': gini,
            'interpretacion': (
                "Baja desigualdad - Distribución relativamente uniforme" if gini < 0.3 else
                "Desigualdad moderada - Cierta concentración espacial" if gini < 0.5 else
                "Alta desigualdad - Concentración significativa" if gini < 0.7 else
                "Desigualdad muy alta - Eventos fuertemente concentrados"
            ),
            'total_eventos': total_eventos,
            'total_celdas': total_celdas,
            'celdas_con_eventos': celdas_con_eventos,
            'porcentaje_celdas_activas': round((celdas_con_eventos / total_celdas) * 100, 2),
            'concentracion_top20': round(concentracion_top20, 2),
            'media_eventos_por_celda': round(media, 2),
            'mediana_eventos_por_celda': round(mediana, 2),
            'desviacion_std': round(desviacion_std, 2),
            'coeficiente_variacion': round(cv, 2),
            'max_eventos_celda': int(np.max(y)) if n > 0 else 0,
            'min_eventos_celda': int(np.min(y)) if n > 0 else 0
        }
    
        # Output
        print("\n" + "="*70)
        print(" ÍNDICE DE GINI ESPACIAL - DISTRIBUCIÓN DE DELITOS")
        print(f"  Índice de Gini: {gini:.6f}")
        print(f"  Interpretación: {resultados['interpretacion']}")
        print(f"\n   Distribución:")
        print(f"     • Celdas con eventos: {celdas_con_eventos:,} ({resultados['porcentaje_celdas_activas']:.1f}%)")
        print(f"     • Celdas sin eventos: {total_celdas - celdas_con_eventos:,}")
        print(f"\n  Concentración:")
        print(f"     • Top 20% de celdas contiene: {resultados['concentracion_top20']:.1f}% de los eventos")
        print(f"\n   Estadísticas:")
        print(f"     • Media: {media:.2f} | Mediana: {mediana:.2f} | STD: {desviacion_std:.2f} | CV: {cv:.1f}%")
        print("="*70)
    
        return resultados
    
    def calcular_moran_i(self):
        """
        Moran's I sobre las celdas del mapa de calor (num_eventos).
        Indica si zonas cercanas tienden a tener valores similares (agrupación) o diferentes (dispersión).
        """
        print("\n[Análisis] Calculando Índice de Moran I (autocorrelación espacial) con num_eventos...")
    
        from esda.moran import Moran
        from libpysal.weights import KNN
    
        # Pesos espaciales (Queen/Rook) y fallback a KNN si hay islas
        w = self._crear_pesos()
        if len(w.islands) > 0 or w.mean_neighbors < 1.0:
            for k in (8, 10, 12, 15, 20):
                wk = KNN.from_dataframe(self.grid_cali, k=k, use_index=True)
                if len(wk.islands) == 0 and wk.mean_neighbors >= 1.0:
                    w = wk
                    print(f"    • Pesos KNN usados (k={k}) para evitar islas")
                    break
        w.transform = 'r'
        y = self.grid_cali['num_eventos'].values.astype(float)
        # Moran's I (z-normal y permutaciones para p-valor robusto)
        moran = Moran(y, w, permutations=999)
        z_score = (moran.I - moran.EI) / np.sqrt(moran.VI_norm)
    
        if abs(z_score) > 2.58:
            significancia = "Muy significativo (p_norm < 0.01)"
            sig_nivel = "***"
        elif abs(z_score) > 1.96:
            significancia = "Significativo (p_norm < 0.05)"
            sig_nivel = "**"
        elif abs(z_score) > 1.65:
            significancia = "Marginal (p_norm < 0.10)"
            sig_nivel = "*"
        else:
            significancia = "No significativo"
            sig_nivel = ""
    
        if moran.I > 0.3:
            interpretacion = "Fuerte agrupación espacial (valores similares juntos)"
            patron = "Agrupación alta"
        elif moran.I > 0:
            interpretacion = "Agrupación espacial moderada"
            patron = "Agrupación moderada"
        elif moran.I > -0.3:
            interpretacion = "Distribución aleatoria"
            patron = "Aleatorio"
        else:
            interpretacion = "Dispersión espacial (vecinos con valores distintos)"
            patron = "Dispersión"
    
        resultados = {
            'moran_i': round(moran.I, 4),
            'valor_esperado': round(moran.EI, 4),
            'varianza_norm': round(moran.VI_norm, 6),
            'z_score': round(z_score, 4),
            'p_norm': round(moran.p_norm, 6),
            'p_perm': round(moran.p_sim, 6),
            'significancia': significancia,
            'significancia_nivel': sig_nivel,
            'interpretacion': interpretacion,
            'patron_espacial': patron,
            'n_vecindarios': w.n,
            'vecinos_promedio': round(w.mean_neighbors, 2),
            'variable_analizada': 'num_eventos'
        }
    
        print("\n" + "="*70)
        print(" ÍNDICE DE MORAN I (num_eventos)")
        print(f"  Moran I: {moran.I:.4f} {sig_nivel} | E[I]: {moran.EI:.4f}")
        print(f"  Z-score: {z_score:.4f} | p_norm: {moran.p_norm:.6f} | p_perm: {moran.p_sim:.6f}")
        print(f"  Interpretación: {interpretacion} ({patron})")
        print(f"  Vecinos promedio: {w.mean_neighbors:.2f}")
        print("="*70)
    
        return resultados
    
    def calcular_getis_ord_gi_star(self):
        """
        Calcula Getis-Ord Gi* (hotspots/coldspots) usando el número de eventos por celda (num_eventos).
        Incluye fallback a pesos KNN para evitar islas y NaN.
        """
        print("\n[Análisis] Calculando Getis-Ord Gi* (Hotspot Analysis) con num_eventos...")
    
        from esda.getisord import G_Local
        from libpysal.weights import KNN
    
        # Usar matriz de pesos silenciosa (Queen/Rook)
        w = self._crear_pesos()
    
        # Fallback a KNN si hay islas o pocos vecinos promedio
        if len(w.islands) > 0 or w.mean_neighbors < 1.0:
            # Aumentar k hasta eliminar islas (límite 20)
            for k in (8, 10, 12, 15, 20):
                wk = KNN.from_dataframe(self.grid_cali, k=k, use_index=True)
                if len(wk.islands) == 0 and wk.mean_neighbors >= 1.0:
                    w = wk
                    print(f"    • Pesos KNN usados (k={k}) para evitar islas")
                    break
            else:
                print("    ⚠ Persisten islas con KNN; se continúa con la mejor opción disponible")
    
        # Asegurar num_eventos en grid_cali
        if 'num_eventos' not in self.grid_cali.columns:
            if 'grid_id' in self.grid_cali.columns and 'num_eventos' in self.grid.columns:
                self.grid_cali = self.grid_cali.merge(
                    self.grid[['grid_id', 'num_eventos']],
                    on='grid_id', how='left'
                )
            self.grid_cali['num_eventos'] = self.grid_cali.get('num_eventos', 0).fillna(0).astype(int)
    
        # Preparar datos
        y = self.grid_cali['num_eventos'].values.astype(float)
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=0.0)
    
        # Chequeo de variación
        if float(np.std(y)) == 0.0:
            print("  ✗ Sin variación en num_eventos; no se pueden detectar hotspots/coldspots")
            return {
                'total_celdas': len(self.grid_cali),
                'variable_analizada': 'num_eventos',
                'hotspots': {'total': 0, 'porcentaje': 0.0},
                'coldspots': {'total': 0, 'porcentaje': 0.0},
                'no_significativo': {'total': len(self.grid_cali), 'porcentaje': 100.0}
            }
    
        # Transformar pesos y calcular Gi*
        w.transform = 'r'
        gi_star = G_Local(y, w, star=True, permutations=999)
    
        gi_values = gi_star.Zs
        p_values = gi_star.p_sim
    
        # Clasificación por significancia
        clasificacion = np.zeros(len(gi_values), dtype=int)
        clasificacion[(gi_values > 2.58) & (p_values < 0.01)] = 3
        clasificacion[(gi_values > 1.96) & (gi_values <= 2.58) & (p_values < 0.05)] = 2
        clasificacion[(gi_values > 1.65) & (gi_values <= 1.96) & (p_values < 0.10)] = 1
        clasificacion[(gi_values < -2.58) & (p_values < 0.01)] = -3
        clasificacion[(gi_values < -1.96) & (gi_values >= -2.58) & (p_values < 0.05)] = -2
        clasificacion[(gi_values < -1.65) & (gi_values >= -1.96) & (p_values < 0.10)] = -1
    
        # Guardar resultados en grid_cali
        self.grid_cali['gi_star_z'] = gi_values
        self.grid_cali['gi_star_p'] = p_values
        self.grid_cali['gi_star_class'] = clasificacion
    
        # Conteos y porcentajes
        total_celdas = len(self.grid_cali)
        hotspot_99 = int((clasificacion == 3).sum())
        hotspot_95 = int((clasificacion == 2).sum())
        hotspot_90 = int((clasificacion == 1).sum())
        coldspot_99 = int((clasificacion == -3).sum())
        coldspot_95 = int((clasificacion == -2).sum())
        coldspot_90 = int((clasificacion == -1).sum())
        no_significativo = int((clasificacion == 0).sum())
    
        total_hotspots = hotspot_99 + hotspot_95 + hotspot_90
        total_coldspots = coldspot_99 + coldspot_95 + coldspot_90
    
        pct_hotspots = (total_hotspots / total_celdas) * 100 if total_celdas else 0.0
        pct_coldspots = (total_coldspots / total_celdas) * 100 if total_celdas else 0.0
        pct_no_sig = (no_significativo / total_celdas) * 100 if total_celdas else 0.0
    
        gi_mean = float(np.nanmean(gi_values))
        gi_std = float(np.nanstd(gi_values))
        gi_min = float(np.nanmin(gi_values))
        gi_max = float(np.nanmax(gi_values))
    
        resultados = {
            'total_celdas': total_celdas,
            'variable_analizada': 'num_eventos',
            'hotspots': {
                'total': total_hotspots,
                'porcentaje': round(pct_hotspots, 2),
                'confianza_99': hotspot_99,
                'confianza_95': hotspot_95,
                'confianza_90': hotspot_90
            },
            'coldspots': {
                'total': total_coldspots,
                'porcentaje': round(pct_coldspots, 2),
                'confianza_99': coldspot_99,
                'confianza_95': coldspot_95,
                'confianza_90': coldspot_90
            },
            'no_significativo': {
                'total': no_significativo,
                'porcentaje': round(pct_no_sig, 2)
            },
            'estadisticas_gi': {
                'media': round(gi_mean, 4),
                'desv_std': round(gi_std, 4),
                'minimo': round(gi_min, 4),
                'maximo': round(gi_max, 4)
            }
        }
    
        print("\n" + "="*70)
        print(" ANÁLISIS GETIS-ORD Gi* (num_eventos)")
        print(f"  • Total de celdas: {total_celdas:,}")
        print(f"  • Hotspots: {total_hotspots} ({pct_hotspots:.1f}%) | Coldspots: {total_coldspots} ({pct_coldspots:.1f}%)")
        print(f"  • No significativo: {no_significativo} ({pct_no_sig:.1f}%)")
        print(f"  • Z Gi* rango: [{gi_min:.4f}, {gi_max:.4f}] | Media: {gi_mean:.4f} | STD: {gi_std:.4f}")
        print("="*70)
    
        return resultados
    
    def calcular_hci_bci(self):
        """
        Calcula HCI/BCI usando el número de delitos por celda (num_eventos):
        - HCI_i = (N_i - N_mean) / N_std  si N_i > N_mean
        - BCI_i = (N_mean - N_i) / N_std  si N_i < N_mean
        """
        print("\n[Análisis] Calculando HCI y BCI (Índices de Clusters) con num_eventos...")

        w = self._crear_pesos()
        w.transform = 'r'

        eventos = self.grid_cali['num_eventos'].values.astype(float)

        # Estadísticas
        n_mean = np.mean(eventos)
        n_std = np.std(eventos)
        n_median = np.median(eventos)

        print(f"    • Eventos medios por celda: {n_mean:.2f}")
        print(f"    • Desviación estándar: {n_std:.2f}")
        print(f"    • Mediana: {n_median:.2f}")

        # HCI/BCI como z-scores unilaterales
        hci = np.zeros(len(eventos))
        bci = np.zeros(len(eventos))

        if n_std > 0:
            mask_hot = eventos > n_mean
            hci[mask_hot] = (eventos[mask_hot] - n_mean) / n_std

            mask_cold = eventos < n_mean
            bci[mask_cold] = (n_mean - eventos[mask_cold]) / n_std

        # Guardar resultados
        self.grid_cali['hci'] = hci
        self.grid_cali['bci'] = bci

        # Clasificación por intensidad (en std)
        hci_class = np.zeros(len(hci), dtype=int)
        hci_class[(hci > 0.5) & (hci <= 1.5)] = 1
        hci_class[(hci > 1.5) & (hci <= 2.5)] = 2
        hci_class[hci > 2.5] = 3

        bci_class = np.zeros(len(bci), dtype=int)
        bci_class[(bci > 0.5) & (bci <= 1.5)] = 1
        bci_class[(bci > 1.5) & (bci <= 2.5)] = 2
        bci_class[bci > 2.5] = 3

        self.grid_cali['hci_class'] = hci_class
        self.grid_cali['bci_class'] = bci_class

        total_celdas = len(self.grid_cali)
        hci_muy_alto = (hci_class == 3).sum()
        hci_alto = (hci_class == 2).sum()
        hci_moderado = (hci_class == 1).sum()
        hci_total = hci_muy_alto + hci_alto + hci_moderado

        bci_muy_alto = (bci_class == 3).sum()
        bci_alto = (bci_class == 2).sum()
        bci_moderado = (bci_class == 1).sum()
        bci_total = bci_muy_alto + bci_alto + bci_moderado

        normal = (hci_class == 0) & (bci_class == 0)
        normal_count = int(normal.sum())

        pct_hci = (hci_total / total_celdas) * 100
        pct_bci = (bci_total / total_celdas) * 100
        pct_normal = (normal_count / total_celdas) * 100

        # Top 10 hot clusters y promedio de vecinos
        top_n = 10
        top_hci_idx = np.argsort(hci)[-top_n:][::-1]

        # Mapeo entre índice de DataFrame y posición
        index_list = list(self.grid_cali.index)
        index_pos = {idx: pos for pos, idx in enumerate(index_list)}
        pos_index = {pos: idx for pos, idx in enumerate(index_list)}

        lag_eventos = []
        for pos in top_hci_idx:
            if hci[pos] > 0.5:
                idx_label = pos_index[pos]
                neigh_labels = w.neighbors.get(idx_label, [])
                neigh_pos = [index_pos[n] for n in neigh_labels if n in index_pos]
                if len(neigh_pos) > 0:
                    lag_eventos.append(float(np.mean(self.grid_cali.iloc[neigh_pos]['num_eventos'].values)))
                else:
                    lag_eventos.append(0.0)

        resultados = {
            'total_celdas': total_celdas,
            'eventos_estadisticas': {
                'media': round(n_mean, 4),
                'std': round(n_std, 4),
                'mediana': round(n_median, 4),
                'min': int(np.min(eventos)),
                'max': int(np.max(eventos))
            },
            'hot_clusters': {
                'total': int(hci_total),
                'porcentaje': round(pct_hci, 2),
                'muy_alto': int(hci_muy_alto),
                'alto': int(hci_alto),
                'moderado': int(hci_moderado),
                'hci_max': round(float(np.max(hci)), 4),
                'hci_mean': round(float(np.mean(hci[hci > 0])) if np.any(hci > 0) else 0.0, 4)
            },
            'background_clusters': {
                'total': int(bci_total),
                'porcentaje': round(pct_bci, 2),
                'muy_alto': int(bci_muy_alto),
                'alto': int(bci_alto),
                'moderado': int(bci_moderado),
                'bci_max': round(float(np.max(bci)), 4),
                'bci_mean': round(float(np.mean(bci[bci > 0])) if np.any(bci > 0) else 0.0, 4)
            },
            'normal': {
                'total': normal_count,
                'porcentaje': round(pct_normal, 2)
            },
            'top_hot_clusters': []
        }

        for i, pos in enumerate(top_hci_idx):
            if hci[pos] > 0.5:
                resultados['top_hot_clusters'].append({
                    'celda_id': int(self.grid_cali.iloc[pos]['grid_id']),
                    'hci': round(float(hci[pos]), 4),
                    'num_eventos': int(self.grid_cali.iloc[pos]['num_eventos']),
                    'eventos_vecinos_promedio': round(lag_eventos[i], 2) if i < len(lag_eventos) else 0.0
                })

        # Output
        print("\n" + "="*70)
        print(" HCI/BCI con num_eventos (conteo de delitos)")
        print(f"  • Media eventos/celda: {n_mean:.2f} | STD: {n_std:.2f} | Mediana: {n_median:.2f}")
        print(f"  • Hot clusters: {hci_total} ({pct_hci:.1f}%) | HCI max: {np.max(hci):.4f}")
        print(f"  • Background clusters: {bci_total} ({pct_bci:.1f}%) | BCI max: {np.max(bci):.4f}")
        print(f"  • Normal: {normal_count} ({pct_normal:.1f}%)")
        print("="*70)

        return resultados

    def calcular_indice_densidad_relativa(self):
        """
        Índice de Densidad Relativa (IDR) basado en delitos:
        - Construye la distribución de conteos por categoría de delito en cada celda.
        - Para cada celda, compara sus conteos (ponderados por PESOS_DELITOS) contra
          el promedio de sus vecinos.
        - IDR_total = (S_i - promedio_vecinos(S_i)) / std_global(S), donde
          S_i = Σ_k (conteo_i,k * peso_k).
        - También calcula IDR por categoría: (c_i,k - promedio_vecinos(c_i,k)) / std_global_k.
        """
        print("\n[Análisis] Calculando Índice de Densidad Relativa (IDR) basado en delitos...")

        # Matriz de pesos sin mensajes
        w = self._crear_pesos()
        w.transform = 'r'

        # Asignar cada caso a su celda
        casos_con_celda = gpd.sjoin(self.gdf_casos, self.grid, how="left", predicate="within").dropna(subset=["grid_id"])

        # Categorías y pesos
        categorias = sorted(casos_con_celda['categoria'].dropna().astype(str).unique())
        pesos = np.array([
            self.PESOS_DELITOS.get(cat, {'peso_total': 1.0})['peso_total']
            for cat in categorias
        ], dtype=float)

        # Tabla de conteos por celda y categoría
        tabla = casos_con_celda.groupby(['grid_id', 'categoria']).size().unstack(fill_value=0)
        for cat in categorias:
            if cat not in tabla.columns:
                tabla[cat] = 0
        tabla = tabla[categorias]
        tabla = tabla.reindex(self.grid_cali['grid_id'].values, fill_value=0)

        counts = tabla.to_numpy(dtype=float)              # [n_celdas, K]
        n_celdas, K = counts.shape

        # Índices para recorrer vecinos
        index_list = list(self.grid_cali.index)
        index_pos = {idx: pos for pos, idx in enumerate(index_list)}

        # Promedio de vecinos por categoría
        vecinos_mean_cat = np.zeros((n_celdas, K), dtype=float)

        for idx in index_list:
            pos = index_pos[idx]
            vecinos = w.neighbors.get(idx, [])
            pos_vecinos = [index_pos[v] for v in vecinos if v in index_pos]
            if len(pos_vecinos) > 0:
                vecinos_mean_cat[pos, :] = counts[pos_vecinos, :].mean(axis=0)
            else:
                # Sin vecinos → usar el propio conteo para no sesgar
                vecinos_mean_cat[pos, :] = counts[pos, :]

        # IDR total ponderado por PESOS_DELITOS
        suma_ponderada = (counts * pesos).sum(axis=1)                  # S_i
        vecinos_mean_ponderado = (vecinos_mean_cat * pesos).sum(axis=1)
        std_global = float(np.std(suma_ponderada))

        if std_global > 0:
            idr_total = (suma_ponderada - vecinos_mean_ponderado) / std_global
        else:
            idr_total = np.zeros(n_celdas, dtype=float)

        # Clasificación del IDR total
        idr_abs = np.abs(idr_total)
        idr_class = np.zeros(n_celdas, dtype=int)
        idr_class[(idr_abs > 0.5) & (idr_abs <= 1.5)] = 1
        idr_class[(idr_abs > 1.5) & (idr_abs <= 2.5)] = 2
        idr_class[idr_abs > 2.5] = 3

        # IDR por categoría (no ponderado, por claridad)
        idr_por_categoria = {}
        for j, cat in enumerate(categorias):
            std_k = float(np.std(counts[:, j]))
            if std_k > 0:
                idr_k = (counts[:, j] - vecinos_mean_cat[:, j]) / std_k
            else:
                idr_k = np.zeros(n_celdas, dtype=float)
            idr_por_categoria[cat] = idr_k
            # Guardar columnas por categoría (nombre seguro)
            col_safe = "idr_cat_" + (
                cat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            )
            self.grid_cali[col_safe] = idr_k

        # Guardar columnas principales
        self.grid_cali['idr_total_delitos'] = idr_total
        self.grid_cali['idr_abs'] = idr_abs
        self.grid_cali['idr_class'] = idr_class
        self.grid_cali['vecinos_mean_total_delitos'] = vecinos_mean_ponderado

        # Resumen
        total = n_celdas
        pos_local = int((idr_total > 0).sum())
        neg_local = int((idr_total < 0).sum())
        normal = int((idr_class == 0).sum())
        moderado = int((idr_class == 1).sum())
        alto = int((idr_class == 2).sum())
        muy_alto = int((idr_class == 3).sum())

        # Top categorías que más contribuyen (media |IDR_cat|)
        contrib_cat = sorted(
            [(cat, float(np.mean(np.abs(idr_por_categoria[cat])))) for cat in categorias],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        resultados = {
            'total_celdas': total,
            'categorias': categorias,
            'idr_total': {
                'media': round(float(np.mean(idr_total)), 4),
                'std': round(float(np.std(idr_total)), 4),
                'min': round(float(np.min(idr_total)), 4),
                'max': round(float(np.max(idr_total)), 4),
            },
            'clasificacion_total': {
                'normal': normal,
                'moderado': moderado,
                'alto': alto,
                'muy_alto': muy_alto
            },
            'signo_local_total': {
                'mas_denso_que_vecinos': pos_local,
                'menos_denso_que_vecinos': neg_local
            },
            'top_categorias_por_contribucion_idr': contrib_cat
        }

        # Mostrar
        print("\n" + "="*70)
        print(" IDR DELITOS (Ponderado por categoría) - CONTRASTE LOCAL")
        print(f"  Categorías: {', '.join(categorias)}")
        print(f"  STD global (ponderado): {std_global:.4f}")
        print(f"\n  Clasificación (|IDR_total| en std):")
        print(f"     • Normal (<0.5): {normal}")
        print(f"     • Moderado (0.5-1.5): {moderado}")
        print(f"     • Alto (1.5-2.5): {alto}")
        print(f"     • Muy Alto (>2.5): {muy_alto}")
        print(f"\n  Signo del contraste (total):")
        print(f"     • Más denso que vecinos (IDR > 0): {pos_local}")
        print(f"     • Menos denso que vecinos (IDR < 0): {neg_local}")
        if len(contrib_cat) > 0:
            print("\n  Top categorías por |IDR_cat| medio:")
            for cat, val in contrib_cat:
                print(f"     • {cat}: {val:.4f}")
        print("="*70)

        return resultados

    def calcular_entropia_shannon_vecindario(self, categorias=None):
        """
        Calcula la Homogeneidad del vecindario (Entropía de Shannon) usando la
        distribución por tipos de delito dentro del vecindario (celda + vecinos).
    
        - Se toma la celda y sus vecinos según la matriz de pesos.
        - Se suman los conteos por categoría en ese vecindario.
        - Se calcula H = -Σ p_k ln(p_k), donde p_k es la proporción por categoría.
        - Homogeneidad = 1 - H / ln(K), con K = número de categorías de delito.
        """
        print("\n[Análisis] Entropía de Shannon por vecindario basada en delitos...")
    
        w = self._crear_pesos()
        w.transform = 'r'

    
        # Asignar cada caso a su celda
        casos_con_celda = gpd.sjoin(self.gdf_casos, self.grid, how="left", predicate="within").dropna(subset=["grid_id"])

        # Categorías de delito
        if categorias is None:
            categorias = sorted(casos_con_celda['categoria'].dropna().astype(str).unique())
        K = len(categorias)
    
        # Tabla de cuentas por celda y categoría
        tabla = casos_con_celda.groupby(['grid_id', 'categoria']).size().unstack(fill_value=0)
        for cat in categorias:
            if cat not in tabla.columns:
                tabla[cat] = 0
        tabla = tabla[categorias]  # orden fijo de columnas
        tabla = tabla.reindex(self.grid_cali['grid_id'].values, fill_value=0)  # alinear con grid_cali
    
        counts = tabla.to_numpy(dtype=float)  # [n_celdas, K]
    
        # Índices para recorrer vecinos
        index_list = list(self.grid_cali.index)
        index_pos = {idx: pos for pos, idx in enumerate(index_list)}
    
        H = np.zeros(len(self.grid_cali))
        homogeneidad = np.zeros(len(self.grid_cali))
    
        for idx in index_list:
            pos = index_pos[idx]
            vecinos = w.neighbors.get(idx, [])
            pos_vecinos = [index_pos[v] for v in vecinos if v in index_pos]
            grupo_pos = [pos] + pos_vecinos
    
            grupo_counts = counts[grupo_pos, :].sum(axis=0)
            total = float(grupo_counts.sum())
    
            if total <= 0 or K == 0:
                H[pos] = 0.0
                homogeneidad[pos] = 1.0
                continue
    
            p = grupo_counts / total
            p_pos = p[p > 0]
            H_local = -np.sum(p_pos * np.log(p_pos))
            H[pos] = H_local
    
            Hmax = np.log(K) if K > 1 else 1.0
            homogeneidad[pos] = 1.0 - (H_local / Hmax if Hmax > 0 else 0.0)
    
        # Guardar columnas
        self.grid_cali['shannon_entropy_delitos'] = H
        self.grid_cali['homogeneidad_vecindario_delitos'] = homogeneidad
    
        resultados = {
            'bins': K,
            'categorias': categorias,
            'entropia': {
                'media': round(float(np.mean(H)), 4),
                'min': round(float(np.min(H)), 4),
                'max': round(float(np.max(H)), 4),
            },
            'homogeneidad': {
                'media': round(float(np.mean(homogeneidad)), 4),
                'min': round(float(np.min(homogeneidad)), 4),
                'max': round(float(np.max(homogeneidad)), 4),
            }
        }
    
        print("\n" + "="*70)
        print(" Entropía de Shannon por Delitos (vecindario)")
        print(f"  Categorías (K): {K} → {', '.join(categorias)}")
        print(f"  Entropía media: {resultados['entropia']['media']:.4f}")
        print(f"  Homogeneidad media: {resultados['homogeneidad']['media']:.4f}")
        print("="*70)
    
        return resultados

    def calcular_gradiente_espacial(self, variable='num_eventos', usar_knn=True):
        """
        Calcula la Gradiente espacial del fenómeno en las celdas del mapa.
        Identifica transiciones suaves/moderadas/abruptas entre vecindarios.

        - variable: columna de la intensidad (por defecto 'num_eventos').
        - usar_knn: si hay islas, usa KNN para asegurar vecinos.
        """
        print("\n[Análisis] Calculando Gradiente espacial (transiciones locales)...")

        # Matriz de pesos silenciosa y fallback KNN si hay islas
        w = self._crear_pesos()
        if usar_knn:
            try:
                from libpysal.weights import KNN
                if len(w.islands) > 0 or w.mean_neighbors < 1.0:
                    for k in (8, 10, 12, 15, 20):
                        wk = KNN.from_dataframe(self.grid_cali, k=k, use_index=True)
                        if len(wk.islands) == 0 and wk.mean_neighbors >= 1.0:
                            w = wk
                            print(f"    • Pesos KNN usados (k={k}) para evitar islas")
                            break
            except Exception:
                pass
        w.transform = 'r'

        # Coordenadas de centroides y valores
        centroids = self.grid_cali.geometry.centroid
        X = centroids.x.values
        Y = centroids.y.values

        vals = self.grid_cali[variable].fillna(0).astype(float).values
        if float(np.std(vals)) == 0.0:
            print("  ⚠ Sin variación global; gradiente será 0.")
        
        n = len(self.grid_cali)
        grad_rms = np.zeros(n, dtype=float)          # √(mean((Δv)^2))
        grad_maxdiff = np.zeros(n, dtype=float)      # max |Δv|
        grad_gx = np.zeros(n, dtype=float)           # ∂v/∂x (ajuste plano)
        grad_gy = np.zeros(n, dtype=float)           # ∂v/∂y
        grad_mag = np.zeros(n, dtype=float)          # √(gx^2 + gy^2)
        grad_dir_deg = np.zeros(n, dtype=float)      # dirección en grados

        # Mapeos de índices
        index_list = list(self.grid_cali.index)
        pos_by_idx = {idx: pos for pos, idx in enumerate(index_list)}

        # Cálculo local por vecindario
        for idx in index_list:
            i = pos_by_idx[idx]
            neigh_labels = w.neighbors.get(idx, [])
            neigh_pos = [pos_by_idx[n] for n in neigh_labels if n in pos_by_idx]

            if len(neigh_pos) == 0:
                # Sin vecinos: gradiente neutro
                grad_rms[i] = 0.0
                grad_maxdiff[i] = 0.0
                grad_gx[i] = 0.0
                grad_gy[i] = 0.0
                grad_mag[i] = 0.0
                grad_dir_deg[i] = 0.0
                continue

            dv = vals[neigh_pos] - vals[i]
            dx = X[neigh_pos] - X[i]
            dy = Y[neigh_pos] - Y[i]

            # RMS y máximo de diferencias
            grad_rms[i] = float(np.sqrt(np.mean(dv**2))) if len(dv) > 0 else 0.0
            grad_maxdiff[i] = float(np.max(np.abs(dv))) if len(dv) > 0 else 0.0

            # Ajuste de plano local: dv ≈ gx*dx + gy*dy
            A = np.column_stack([dx, dy])
            try:
                beta, _, _, _ = np.linalg.lstsq(A, dv, rcond=None)
                gx, gy = float(beta[0]), float(beta[1])
            except Exception:
                gx, gy = 0.0, 0.0

            grad_gx[i] = gx
            grad_gy[i] = gy
            grad_mag[i] = float(np.sqrt(gx*gx + gy*gy))
            grad_dir_deg[i] = float((np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0)

        # Clasificación por intensidad de transición usando grad_rms
        rms_mean = float(np.mean(grad_rms))
        rms_std = float(np.std(grad_rms))
        # Umbrales: suave ≤ μ+0.5σ, moderada ≤ μ+1.5σ, abrupta > μ+1.5σ
        class_trans = np.zeros(n, dtype=int)
        class_trans[(grad_rms > (rms_mean + 0.5 * rms_std)) & (grad_rms <= (rms_mean + 1.5 * rms_std))] = 1
        class_trans[grad_rms > (rms_mean + 1.5 * rms_std)] = 2

        # Guardar columnas
        suf = variable.lower()
        self.grid_cali[f'grad_rms_{suf}'] = grad_rms
        self.grid_cali[f'grad_maxdiff_{suf}'] = grad_maxdiff
        self.grid_cali[f'grad_gx_{suf}'] = grad_gx
        self.grid_cali[f'grad_gy_{suf}'] = grad_gy
        self.grid_cali[f'grad_mag_{suf}'] = grad_mag
        self.grid_cali[f'grad_dir_deg_{suf}'] = grad_dir_deg
        self.grid_cali[f'grad_class_{suf}'] = class_trans  # 0: suave, 1: moderada, 2: abrupta

        # Resumen
        suaves = int((class_trans == 0).sum())
        moderadas = int((class_trans == 1).sum())
        abruptas = int((class_trans == 2).sum())

        resultados = {
            'variable': variable,
            'resumen_clases': {
                'suaves': suaves,
                'moderadas': moderadas,
                'abruptas': abruptas
            },
            'estadisticas_rms': {
                'media': round(rms_mean, 4),
                'std': round(rms_std, 4),
                'min': round(float(np.min(grad_rms)), 4),
                'max': round(float(np.max(grad_rms)), 4)
            }
        }

        print("\n" + "="*70)
        print(f" GRADIENTE ESPACIAL ({variable})")
        print(f"  • Transiciones: Suaves={suaves}, Moderadas={moderadas}, Abruptas={abruptas}")
        print(f"  • RMS diferencias → Media: {rms_mean:.4f} | STD: {rms_std:.4f}")
        print("="*70)

        return resultados
    

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
        
        # Calcular métricas espaciales
        #gini_results = self.calcular_gini_espacial()
        #moran_results = self.calcular_moran_i()
        #gi_star_results = self.calcular_getis_ord_gi_star()
        #hci_bci_results = self.calcular_hci_bci()
        #idr_results = self.calcular_indice_densidad_relativa()
        #shannon_results = self.calcular_entropia_shannon_vecindario()
        #gradiente = self.calcular_gradiente_espacial()

        # Visualizar hotspots
        print("\n" + "="*70)
        print("✓ PROCESO COMPLETADO")
        print("="*70)


if __name__ == "__main__":
    import json

    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "optimizacion_celda"

    with open(resultados_dir / "resumen_tres_metodos.json") as f:
        config = json.load(f)
        mejorcelda = config['recomendacion']
        mejor_metodo_nombre = config['mejor_metodo']

    print(f"\nMejor tamaño de celda: {mejorcelda} m")
    print(f"Método ganador: {mejor_metodo_nombre}")
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
    
    # Bounding box EXACTO (EPSG:3116)
    BBOX = (720414.01, 735029.27, 860572.64, 879817.30)  # (minx, maxx, miny, maxy)
    
    # Crear instancia y ejecutar
    mapa = GeoreferenciaMapa(archivos_especificos, PESOS_DELITOS, mejorcelda, bbox=BBOX)
    mapa.ejecutar_pipeline_completo()

    # Exportar grid_cali a GeoJSON
    output_path = Path("images") / "grid_cali.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPORTING HEXAGONAL GRID TO GEOJSON")
    print(f"{'='*80}")
    print(f"Saving grid_cali ({len(mapa.grid_cali)} cells) to: {output_path}")

    # Exportar a GeoJSON usando método manual (evitar problemas con pyogrio)
    geojson_data = mapa.grid_cali.__geo_interface__
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"\nGrid properties:")
    print(f"  - Total cells: {len(mapa.grid_cali)}")
    print(f"  - Insecurity index range: 0-100")
    print(f"  - CRS: {mapa.grid_cali.crs}")
    print(f"{'='*80}\n")
