# ========== IMPORTS ==========
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import libpysal, warnings, contextlib, builtins, sys, os, logging
from metricas import calcular_entropia_shannon_vecindario, calcular_getis_ord_gi_star, calcular_gini_espacial, calcular_gradiente_espacial, calcular_hci_bci, calcular_indice_densidad_relativa, calcular_moran_i, guardar_metricas_escenarios_csv

class GeoreferenciaMapa:
    def __init__(self, archivos_especificos, PESOS_DELITOS, mejorcelda, bbox=None):
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
        self.cali = ox.geocode_to_gdf("Santiago de Cali, Colombia", which_result=2)
        self.cali = self.cali.to_crs(3116)

        for i in [1, 2]:
            gdf = ox.geocode_to_gdf(
                "Santiago de Cali, Colombia",
                which_result=i
            )

            gdf = gdf.to_crs("EPSG:3116")

            print(f"\nResultado {i}")
            print("Área km²:", gdf.geometry.area.sum()/1e6)
            print("Bounds:", gdf.total_bounds)
        
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
        print(f"\n   Box: {self.cali.total_bounds}")

        
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
        
        print(f" {len(polygons):,} hexágonos generados")
        
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
        """Carga un CSV y conserva solo hechos de julio 2023 a julio 2024."""
        try:
            print(f"\n  Cargando: {archivo}")
            df = pd.read_csv(archivo, encoding='utf-8', on_bad_lines='skip')

            # Normalizar encabezados para encontrar Año_Hecho y Mes_Hecho
            # sin depender de mayúsculas ni espacios accidentales.
            df.columns = df.columns.str.strip().str.lower()
            print(f" Registros iniciales: {len(df):,}")

            columnas_fecha = {'año_hecho', 'mes_hecho'}
            faltantes = columnas_fecha.difference(df.columns)
            if faltantes:
                print(
                    f"    No se procesa {archivo}: faltan las columnas "
                    f"{', '.join(sorted(faltantes))}."
                )
                return None

            anio = pd.to_numeric(df['año_hecho'], errors='coerce')
            mes_texto = df['mes_hecho'].astype('string').str.strip().str.lower()
            meses = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
                'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
                'septiembre': 9, 'setiembre': 9, 'octubre': 10,
                'noviembre': 11, 'diciembre': 12,
            }
            mes = pd.to_numeric(mes_texto, errors='coerce')
            mes_nombres = mes_texto.map(meses)
            mes = mes.fillna(mes_nombres)

            periodo = (anio * 100 + mes).between(202307, 202407)
            df = df.loc[periodo].copy()
            print(f" Registros entre julio de 2023 y julio de 2024: {len(df):,}")

            df = self.limpiar_coordenadas(df, 'y', 'x')
            if df is None:
                print(f"    No se procesa {archivo}: faltan columnas de coordenadas x/y.")
                return None

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
        
        print(f"\n Total de archivos procesados: {archivos_encontrados}")
        
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

        casos_dentro_cali = self.gdf_casos[
            self.gdf_casos.within(self.cali.geometry.union_all())
        ]
        
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
        fig_width_in = 7.16  #  dos columnas
        fig_height_in = fig_width_in * 0.75         
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
        print(f" Mapa simple guardado: {out_path_simple.name}")
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
        

        plt.xlabel("X Coordinate (m)", fontsize=8)
        plt.ylabel("Y Coordinate (m)", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        # Barra de color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Level of Insecurity", shrink=0.7, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['No data', 'Very Low\n(0-20%)', 'Low\n(20-40%)',
                            'Medium\n(40-60%)', 'High\n(60-80%)', 'Very High\n(80-100%)'])
        cbar.set_label("Level of Insecurity", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
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
        
        plt.text(0.615, 0.02, stats_text, transform=ax.transAxes, fontsize=7.8,
                verticalalignment='bottom', family='Arial',linespacing=1,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.6))
        
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.86)
        plt.grid(True, alpha=0.5, linestyle='--')
        
        # Guardar mapa completo (con todos los elementos)
        out_path_completo = images_dir /  "Experimento" / "mapa_calor_genero_cali_20232024.pdf"
        plt.savefig(out_path_completo, format='pdf', bbox_inches='tight')
        print(f" Mapa completo guardado: {out_path_completo.name}")

    def guardar_panel_comparativo_resoluciones(self, output_path=None):
        """Genera y guarda un panel 1x3 para comparar h=100, 160 y 238 m.

        Cada mapa se calcula con los casos ya georreferenciados en el pipeline.
        Los niveles de riesgo se asignan por percentiles (como en
        ``clasificar_niveles_inseguridad``) y se muestran con una paleta y una
        escala categórica compartidas.

        Args:
            output_path: Ruta opcional del PNG. Por defecto se guarda en
                ``images/panel_comparativo_resoluciones.png`` junto al script.

        Returns:
            Path: Ruta de la imagen guardada.
        """
        if self.cali is None or self.cali.empty:
            raise RuntimeError("Primero debe cargarse el mapa base de Cali.")
        if self.gdf_casos is None:
            raise RuntimeError("Primero deben georreferenciarse los casos.")
        if "peso_delito" not in self.gdf_casos.columns:
            raise ValueError("Los casos georreferenciados no contienen 'peso_delito'.")

        resoluciones = (100, 160, 238)
        colores = ["white", "green", "blue", "yellow", "#f05209", "#1a0f0a"]
        cmap = mcolors.ListedColormap(colores, name="riesgo_compartido")
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

        try:
            area_cali = self.cali.geometry.union_all()
        except AttributeError:
            area_cali = self.cali.geometry.unary_union
        xmin, ymin, xmax, ymax = self.cali.total_bounds

        grids = []
        for lado_hex in resoluciones:
            ancho_hex = math.sqrt(3) * lado_hex
            paso_y = 1.5 * lado_hex
            x_vals = np.arange(xmin - ancho_hex, xmax + ancho_hex, ancho_hex)
            y_vals = np.arange(ymin - lado_hex, ymax + lado_hex, paso_y)

            poligonos = []
            for ix, x in enumerate(x_vals):
                y_offset = paso_y / 2.0 if ix % 2 else 0.0
                for y in y_vals:
                    centro_x, centro_y = x, y + y_offset
                    coords = [
                        (centro_x + lado_hex * math.cos(i * math.pi / 3.0),
                         centro_y + lado_hex * math.sin(i * math.pi / 3.0))
                        for i in range(6)
                    ]
                    hexagono = Polygon(coords)
                    if hexagono.intersects(area_cali):
                        poligonos.append(hexagono)

            grid = gpd.GeoDataFrame(
                {"grid_id": np.arange(len(poligonos))},
                geometry=poligonos,
                crs=self.cali.crs,
            )
            grid_cali = gpd.overlay(grid, self.cali, how="intersection")

            casos = self.gdf_casos[["peso_delito", "geometry"]].copy()
            casos_con_celda = gpd.sjoin(
                casos, grid[["grid_id", "geometry"]], how="inner", predicate="within"
            )
            puntajes = casos_con_celda.groupby("grid_id")["peso_delito"].sum()
            grid_cali["score_raw"] = grid_cali["grid_id"].map(puntajes).fillna(0.0)

            minimo = grid_cali["score_raw"].min()
            maximo = grid_cali["score_raw"].max()
            if maximo > minimo:
                indices = (grid_cali["score_raw"] - minimo) / (maximo - minimo) * 100
            else:
                indices = pd.Series(0.0, index=grid_cali.index)

            niveles = np.zeros(len(grid_cali), dtype=int)
            positivos = indices.to_numpy() > 0
            if positivos.any():
                p20, p40, p60, p80 = np.percentile(indices[positivos], [20, 40, 60, 80])
                valores = indices.to_numpy()
                niveles[positivos & (valores <= p20)] = 1
                niveles[positivos & (valores > p20) & (valores <= p40)] = 2
                niveles[positivos & (valores > p40) & (valores <= p60)] = 3
                niveles[positivos & (valores > p60) & (valores <= p80)] = 4
                niveles[positivos & (valores > p80)] = 5
            grid_cali["nivel_inseguridad"] = niveles
            grids.append(grid_cali)

        plt.rcParams["font.family"] = "Arial"
        fig_width_in = 7.16  # dos columnas
        fig_height_in = fig_width_in * 0.59
        fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in))
        bounds = self.cali.total_bounds
        for ax, lado_hex, grid in zip(axes, resoluciones, grids):
            self.cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.55, zorder=1)
            grid.plot(
                ax=ax, column="nivel_inseguridad", cmap=cmap, norm=norm,
                edgecolor="black", linewidth=0.15, zorder=2,
            )
            if self.bbox is not None:
                minx, maxx, miny, maxy = self.bbox
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)
            else:
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            ax.set_aspect("equal")
            ax.set_title(f"h = {lado_hex} m", fontsize=11, pad=5)
            ax.set_axis_off()

        etiquetas = ["Sin datos", "Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
        fig.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, facecolor=colores[i], edgecolor="black", label=etiquetas[i])
                for i in range(len(etiquetas))
            ],
            loc="lower center", ncol=6, frameon=False, fontsize=10,
            bbox_to_anchor=(0.5, 0.035),
        )
        #fig.suptitle("Comparación de mapas de riesgo por resolución", fontsize=12, y=0.96)
        fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.15, wspace=0.04)

        if output_path is None:
            output_path = Path(__file__).resolve().parent / "images" / "panel_comparativo_resoluciones.png"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600)
        plt.close(fig)
        print(f"Panel comparativo guardado: {output_path}")
        return output_path

    def calcular_metricas(self, output_path=None):
        """Calcula las metricas espaciales con la configuracion ya cargada."""
        if self.grid is None or self.grid.empty or self.grid_cali is None or self.grid_cali.empty:
            raise RuntimeError("Primero debe generarse y procesarse la grilla.")
        if self.gdf_casos is None or self.gdf_casos.empty:
            raise RuntimeError("Primero deben georreferenciarse los casos.")

        metricas = {
            "gini_espacial": calcular_gini_espacial(self, variable="score_raw"),
            "moran_i": calcular_moran_i(self, variable="score_raw"),
            "getis_ord_gi_star": calcular_getis_ord_gi_star(self, variable="score_raw"),
            "hci_bci": calcular_hci_bci(self, variable="score_raw"),
            "indice_densidad_relativa": calcular_indice_densidad_relativa(self, gpd),
            "entropia_shannon_vecindario": calcular_entropia_shannon_vecindario(self, gpd),
            "gradiente_espacial": calcular_gradiente_espacial(
                self, variable="score_raw"
            ),
        }
        resultados = {"metricas": metricas}
        self.resultados_metricas = resultados

        if output_path is None:
            output_path = (
                Path(__file__).resolve().parent
                / "images" / "Experimento" / "resultados_metricas_2023_2024.csv"
            )
        # El exportador actual escribe una fila por entrada; esta es la unica configuracion.
        return guardar_metricas_escenarios_csv(
            {"Configuracion actual": resultados}, output_path
        )

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
        self.guardar_panel_comparativo_resoluciones()

        self.calcular_metricas()

        
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

if __name__ == "__main__":
    import json

    # Cargar configuración
    resultados_dir = Path(__file__).resolve().parent / "optimizacion_celda"

    with open(resultados_dir / "resumen_tres_metodos.json") as f:
        config = json.load(f)
        mejorcelda = config['recomendacion']
        mejor_metodo_nombre = config['mejor_metodo']

    print(f"\nMejor tamaño de celda: {mejorcelda:.1f} m")
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

    # Seleccionar solo las columnas que realmente necesitas
    columns_to_keep = [
        'grid_id',
        'nivel_inseguridad',
        'indice_inseguridad',
        'score_raw',
        'num_eventos',
        'geometry',
        'lon',
        'lat',
        "bbox_west",
        "bbox_south",
        "bbox_east",
        "bbox_north"
    ]

    # Filtrar y hacer copia (evita problemas de referencia)
    grid_export = mapa.grid_cali[columns_to_keep].copy()

    # Asegurar geometrías válidas
    grid_export = grid_export[grid_export.geometry.is_valid]

    # Guardar con Fiona (engine='fiona' evita problemas de tipos con pyogrio)
    try:
        grid_export.to_file(output_path, driver='GeoJSON', engine='fiona')
    except Exception as e:
        print(f"Error con engine='fiona': {e}")
        # Fallback: usar el método manual con __geo_interface__ (más lento pero seguro)
        geojson_data = grid_export.__geo_interface__
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"\nGrid properties:")
    print(f"Total cells: {len(grid_export)}")
    print(f"Insecurity index range: {grid_export['indice_inseguridad'].min():.2f} - {grid_export['indice_inseguridad'].max():.2f}")
    print(f"{'='*80}\n")

    # Guardar grid_cali en pickle (preserva geometrías y CRS)
    import pickle
    pickle_path = Path("pipeline") / "grid_cali.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(mapa.grid_cali, f)
    print(f"Grid guardado en pickle: {pickle_path}")
