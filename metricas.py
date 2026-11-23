import matplotlib
matplotlib.use('Agg') 
import math
import numpy as np
import pandas as pd
import random
import importlib
import importlib.util
import sys
import types
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point
from matplotlib.patches import Rectangle
np.random.seed(42)
random.seed(42)


def _cargar_modulo_esda(nombre):
    """Carga una subrutina de esda sin ejecutar su __init__ agregado.

    Algunas instalaciones de esda importan scikit-learn completo al cargar el
    paquete, aunque solo se necesite Moran o Getis-Ord. El modulo puede estar
    dañado en componentes no usados (por ejemplo sklearn.svm._classes).
    """
    nombre_modulo = f"esda.{nombre}"
    if nombre_modulo in sys.modules:
        return sys.modules[nombre_modulo]

    spec = importlib.util.find_spec("esda")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("No se encontro el paquete esda.")

    paquete_previo = sys.modules.get("esda")
    if paquete_previo is None:
        paquete = types.ModuleType("esda")
        paquete.__path__ = list(spec.submodule_search_locations)
        paquete.__package__ = "esda"
        paquete.__file__ = spec.origin
        sys.modules["esda"] = paquete
    try:
        return importlib.import_module(nombre_modulo)
    finally:
        if paquete_previo is None:
            sys.modules.pop("esda", None)


def guardar_metricas_escenarios_csv(resultados_por_escenario, output_path):
    """Guarda solo las 25 metricas solicitadas, con una fila por escenario."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for escenario, resumen in resultados_por_escenario.items():
        metricas = resumen.get("metricas", {})
        gini = metricas.get("gini_espacial", {})
        moran = metricas.get("moran_i", {})
        getis = metricas.get("getis_ord_gi_star", {})
        hci_bci = metricas.get("hci_bci", {})
        hotspots = getis.get("hotspots", {})
        coldspots = getis.get("coldspots", {})
        estadisticas_gi = getis.get("estadisticas_gi", {})
        hot_clusters = hci_bci.get("hot_clusters", {})
        background_clusters = hci_bci.get("background_clusters", {})
        normal = hci_bci.get("normal", {})
        idr = metricas.get("indice_densidad_relativa", {})
        clasificacion_idr = idr.get("clasificacion_total", {})
        signo_idr = idr.get("signo_local_total", {})
        gradiente = metricas.get("gradiente_espacial", {})
        clases_gradiente = gradiente.get("resumen_clases", {})
        estadisticas_rms = gradiente.get("estadisticas_rms", {})
        shannon = metricas.get("entropia_shannon_vecindario", {})

        filas.append({
            "Escenario": escenario,
            "Gini espacial": gini.get("gini_espacial"),
            "Celdas con al menos un evento (%)": gini.get("porcentaje_celdas_activas"),
            "Moran's I": moran.get("moran_i"),
            "Moran's I - Valor esperado": moran.get("valor_esperado"),
            "Moran's I - Z-score": moran.get("z_score"),
            "Moran's I - p-valor (perm)": moran.get("p_perm"),
            "Getis-Ord Gi* - Hotspots total": hotspots.get("total"),
            "Getis-Ord Gi* - Hotspots (%)": hotspots.get("porcentaje"),
            "Getis-Ord Gi* - Coldspots total": coldspots.get("total"),
            "Getis-Ord Gi* - Z mínimo": estadisticas_gi.get("minimo"),
            "Getis-Ord Gi* - Z máximo": estadisticas_gi.get("maximo"),
            "HCI - Hot clusters (%)": hot_clusters.get("porcentaje"),
            "BCI - Background clusters (%)": background_clusters.get("porcentaje"),
            "Normal (%)": normal.get("porcentaje"),
            "HCI máximo": hot_clusters.get("hci_max"),
            "Gradiente espacial - Suaves": clases_gradiente.get("suaves"),
            "Gradiente espacial - Moderadas": clases_gradiente.get("moderadas"),
            "Gradiente espacial - Abruptas": clases_gradiente.get("abruptas"),
            "Gradiente RMS - Media": estadisticas_rms.get("media"),
            "Gradiente RMS - Desv. estándar": estadisticas_rms.get("std"),
            "IDR - Moderado + Alto + Muy alto": sum(
                clasificacion_idr.get(nivel, 0) for nivel in ("moderado", "alto", "muy_alto")
            ),
            "IDR - Más denso que vecinos": signo_idr.get("mas_denso_que_vecinos"),
            "IDR - Menos denso que vecinos": signo_idr.get("menos_denso_que_vecinos"),
            "Entropía Shannon - Media": shannon.get("entropia", {}).get("media"),
            "Homogeneidad - Media": shannon.get("homogeneidad", {}).get("media"),
        })

    columnas = list(filas[0].keys()) if filas else ["Escenario"]
    pd.DataFrame(filas, columns=columnas).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    print(f"CSV de metricas guardado: {output_path}")
    return output_path

def calcular_gini_espacial(self, variable='num_eventos'):
        """
        Calcula el Índice de Gini espacial sobre num_eventos (delitos por celda).
        Incluye todas las celdas (incluidas las de 0 eventos) para medir concentración real.
        """
        print("\n[Análisis] Calculando Índice de Gini espacial (num_eventos)...")
    
        # Serie de eventos por celda (incluye ceros)
        y = self.grid_cali.get(variable, pd.Series(dtype=int)).fillna(0).astype(float).values
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
        total_eventos = round(suma_total, 4)
        total_celdas = n
        celdas_con_eventos = int(np.sum(y > 0))
    
        resultados = {
            'gini_espacial': gini,
            'variable_analizada': variable,
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
            'max_eventos_celda': round(float(np.max(y)), 4) if n > 0 else 0,
            'min_eventos_celda': round(float(np.min(y)), 4) if n > 0 else 0
        }
    
        # Output
        print("\n" + "="*70)
        print(" ÍNDICE DE GINI ESPACIAL - DISTRIBUCIÓN DE DELITOS")
        print(f" Índice de Gini: {gini:.6f}")
        print(f" Interpretación: {resultados['interpretacion']}")
        print(f"\n   Distribución:")
        print(f"  Celdas con eventos: {celdas_con_eventos:,} ({resultados['porcentaje_celdas_activas']:.1f}%)")
        print(f"  Celdas sin eventos: {total_celdas - celdas_con_eventos:,}")
        print(f"\n  Concentración:")
        print(f"  Top 20% de celdas contiene: {resultados['concentracion_top20']:.1f}% de los eventos")
        print(f"\n  Estadísticas:")
        print(f"  Media: {media:.2f} | Mediana: {mediana:.2f} | STD: {desviacion_std:.2f} | CV: {cv:.1f}%")
        print("="*70)
    
        return resultados
    
def calcular_moran_i(self, variable='num_eventos'):
    """
    Moran's I sobre las celdas del mapa de calor (num_eventos).
    Indica si zonas cercanas tienden a tener valores similares (agrupación) o diferentes (dispersión).
    """
    print("\n[Análisis] Calculando Índice de Moran I (autocorrelación espacial) con num_eventos...")

    Moran = _cargar_modulo_esda("moran").Moran
    from libpysal.weights import KNN

    # Pesos espaciales (Queen/Rook) y fallback a KNN si hay islas
    w = self._crear_pesos()
    if len(w.islands) > 0 or w.mean_neighbors < 1.0:
        for k in (8, 10, 12, 15, 20):
            wk = KNN.from_dataframe(self.grid_cali, k=k, use_index=True)
            if len(wk.islands) == 0 and wk.mean_neighbors >= 1.0:
                w = wk
                print(f"Pesos KNN usados (k={k}) para evitar islas")
                break
    w.transform = 'r'
    y = self.grid_cali[variable].values.astype(float)
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
            'variable_analizada': variable
    }

    print("\n" + "="*70)
    print(" ÍNDICE DE MORAN I (num_eventos)")
    print(f"  Moran I: {moran.I:.4f} {sig_nivel} | E[I]: {moran.EI:.4f}")
    print(f"  Z-score: {z_score:.4f} | p_norm: {moran.p_norm:.6f} | p_perm: {moran.p_sim:.6f}")
    print(f"  Interpretación: {interpretacion} ({patron})")
    print(f"  Vecinos promedio: {w.mean_neighbors:.2f}")
    print("="*70)

    return resultados

def calcular_getis_ord_gi_star(self, variable='num_eventos'):
    """
    Calcula Getis-Ord Gi* (hotspots/coldspots) usando el número de eventos por celda (num_eventos).
    Incluye fallback a pesos KNN para evitar islas y NaN.
    """

    G_Local = _cargar_modulo_esda("getisord").G_Local
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
                print(f"Pesos KNN usados (k={k}) para evitar islas")
                break
        else:
            print("Persisten islas con KNN; se continúa con la mejor opción disponible")

    # Asegurar num_eventos en grid_cali
    if variable not in self.grid_cali.columns:
        if 'grid_id' in self.grid_cali.columns and variable in self.grid.columns:
            self.grid_cali = self.grid_cali.merge(
                self.grid[['grid_id', variable]],
                on='grid_id', how='left'
            )
        self.grid_cali[variable] = self.grid_cali.get(variable, 0).fillna(0)

    # Preparar datos
    y = self.grid_cali[variable].values.astype(float)
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=0.0)

    # Chequeo de variación
    if float(np.std(y)) == 0.0:
        return {
            'total_celdas': len(self.grid_cali),
            'variable_analizada': variable,
            'hotspots': {'total': 0, 'porcentaje': 0.0},
            'coldspots': {'total': 0, 'porcentaje': 0.0},
            'no_significativo': {'total': len(self.grid_cali), 'porcentaje': 100.0}
        }

    # Transformar pesos y calcular Gi*
    w.transform = 'r'
    # esda 2.9 aun usa np.row_stack internamente; NumPy reciente lo retiro.
    row_stack_original = getattr(np, 'row_stack', None)
    if row_stack_original is None:
        np.row_stack = np.vstack
    try:
        gi_star = G_Local(y, w, star=True, permutations=999)
    finally:
        if row_stack_original is None:
            del np.row_stack

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
        'variable_analizada': variable,
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
    print(f" Total de celdas: {total_celdas:,}")
    print(f" Hotspots: {total_hotspots} ({pct_hotspots:.1f}%) | Coldspots: {total_coldspots} ({pct_coldspots:.1f}%)")
    print(f" No significativo: {no_significativo} ({pct_no_sig:.1f}%)")
    print(f" Z Gi* rango: [{gi_min:.4f}, {gi_max:.4f}] | Media: {gi_mean:.4f} | STD: {gi_std:.4f}")
    print("="*70)

    return resultados

def calcular_hci_bci(self, variable='num_eventos'):
    """
    Calcula HCI/BCI usando el número de delitos por celda (num_eventos):
    - HCI_i = (N_i - N_mean) / N_std  si N_i > N_mean
    - BCI_i = (N_mean - N_i) / N_std  si N_i < N_mean
    """
    print("\n[Análisis] Calculando HCI y BCI (Índices de Clusters) con num_eventos...")

    w = self._crear_pesos()
    w.transform = 'r'

    eventos = self.grid_cali[variable].values.astype(float)

    # Estadísticas
    n_mean = np.mean(eventos)
    n_std = np.std(eventos)
    n_median = np.median(eventos)

    print(f" Eventos medios por celda: {n_mean:.2f}")
    print(f" Desviación estándar: {n_std:.2f}")
    print(f" Mediana: {n_median:.2f}")

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
                lag_eventos.append(float(np.mean(self.grid_cali.iloc[neigh_pos][variable].values)))
            else:
                lag_eventos.append(0.0)

    resultados = {
        'total_celdas': total_celdas,
        'variable_analizada': variable,
        'eventos_estadisticas': {
            'media': round(n_mean, 4),
            'std': round(n_std, 4),
            'mediana': round(n_median, 4),
            'min': round(float(np.min(eventos)), 4),
            'max': round(float(np.max(eventos)), 4)
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
                'valor_analizado': round(float(self.grid_cali.iloc[pos][variable]), 4),
                'eventos_vecinos_promedio': round(lag_eventos[i], 2) if i < len(lag_eventos) else 0.0
            })

    # Output
    print("\n" + "="*70)
    print(" HCI/BCI con num_eventos (conteo de delitos)")
    print(f" Media eventos/celda: {n_mean:.2f} | STD: {n_std:.2f} | Mediana: {n_median:.2f}")
    print(f" Hot clusters: {hci_total} ({pct_hci:.1f}%) | HCI max: {np.max(hci):.4f}")
    print(f" Background clusters: {bci_total} ({pct_bci:.1f}%) | BCI max: {np.max(bci):.4f}")
    print(f" Normal: {normal_count} ({pct_normal:.1f}%)")
    print("="*70)

    return resultados

def calcular_indice_densidad_relativa(self, gpd):
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
    print(f" Categorías: {', '.join(categorias)}")
    print(f" STD global (ponderado): {std_global:.4f}")
    print(f"\n Clasificación (|IDR_total| en std):")
    print(f"  Normal (<0.5): {normal}")
    print(f"  Moderado (0.5-1.5): {moderado}")
    print(f"  Alto (1.5-2.5): {alto}")
    print(f"  Muy Alto (>2.5): {muy_alto}")
    print(f"\n Signo del contraste (total):")
    print(f"  Más denso que vecinos (IDR > 0): {pos_local}")
    print(f"  Menos denso que vecinos (IDR < 0): {neg_local}")
    if len(contrib_cat) > 0:
        print("\n  Top categorías por |IDR_cat| medio:")
        for cat, val in contrib_cat:
            print(f" {cat}: {val:.4f}")
    print("="*70)

    return resultados

def calcular_entropia_shannon_vecindario(self, gpd, categorias=None):
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
    if 'peso_delito_escenario' in casos_con_celda.columns:
        tabla = casos_con_celda.pivot_table(
            index='grid_id', columns='categoria', values='peso_delito_escenario',
            aggfunc='sum', fill_value=0,
        )
    else:
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
                        print(f"Pesos KNN usados (k={k}) para evitar islas")
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
        print("  Sin variación global; gradiente será 0.")
    
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
    print(f"GRADIENTE ESPACIAL ({variable})")
    print(f"Transiciones: Suaves={suaves}, Moderadas={moderadas}, Abruptas={abruptas}")
    print(f"RMS diferencias → Media: {rms_mean:.4f} | STD: {rms_std:.4f}")
    print("="*70)

    return resultados

def gtw_distance(self, factor_densidad=1.0, max_intentos=100,
        graficar=True, guardar=True, nombre_mapa='qos_logdistance_riesgo.png'):

        # Verificar si ya existen nodos IoT riesgo
        if getattr(self, 'nodos_iot_riesgo', None) is not None:
            print("\nUsando nodos IoT riesgo ya generados")
            print(f"Nodos reutilizados: {len(self.nodos_iot_riesgo)}")
        else:
            print("\n Generando nodos IoT desde grid_cali con riesgo...")
            nodos_coords = []

            celdas_activas = self.grid_cali
            for _, row in celdas_activas.iterrows():
                indice = float(row.get('indice_inseguridad', 0.0))

                # Sin límite fijo de nodos por celda
                n_nodos = max(1, int(np.ceil(indice * factor_densidad)))

                poly = row['geometry']
                minx, miny, maxx, maxy = poly.bounds

                for _ in range(n_nodos):
                    intentos = 0
                    while intentos < max_intentos:
                        x = np.random.uniform(minx, maxx)
                        y = np.random.uniform(miny, maxy)
                        punto = Point(x, y)
                        if poly.contains(punto):
                            nodos_coords.append([x, y])
                            break
                        intentos += 1

                    if intentos >= max_intentos:
                        centroide = poly.centroid
                        nodos_coords.append([centroide.x, centroide.y])

            self.nodos_iot_riesgo = np.array(nodos_coords)
            print(f"  • Nodos generados: {len(self.nodos_iot_riesgo)}")

        nodos = np.asarray(self.nodos_iot_riesgo)
        gateways = np.asarray(self.gateways_ga)
            
        # Guardar nodos en CSV
        x_ini = 726000
        y_ini = 867300
        ancho = 4000
        alto = 4000
        x_fin = x_ini + ancho
        y_fin = y_ini + alto

        if guardar:
            df_nodos = pd.DataFrame(
                self.nodos_iot_riesgo,
                columns=['x', 'y']
            )
            
            # Agregar ID de nodo
            df_nodos.insert(0, 'id', range(1, len(self.nodos_iot_riesgo) + 1))

            # Filtrar nodos dentro del cuadrado
            nodos_cuadrado = df_nodos[(df_nodos['x'] >= x_ini) & (df_nodos['x'] <= x_fin) & (df_nodos['y'] >= y_ini) & (df_nodos['y'] <= y_fin)]
            if len(nodos_cuadrado) == 1000:
                            print(f"Mil nodos dentro")
            else:
                print(f"No hay mil nodos: {len(nodos_cuadrado)}")

            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_cuadrado = output_path / "nodos_cuadrado.csv"
            nodos_cuadrado.to_csv(out_path_cuadrado, index=False)

            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "nodos_riesgo.csv"

            df_nodos.to_csv(out_path_simple, index=False)
            print(f"Nodos riesgo guardados en: {out_path_simple}")
            
            # Calcular Gini
            
            y = np.array(sensores_por_celda, dtype=float)
            n = len(y)
            y_sorted = np.sort(y)
            suma_total = float(np.sum(y_sorted))
            if suma_total == 0:
                gini = 0.0
            else:
                i = np.arange(1, n + 1, dtype=float)
                gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                gini = max(0.0, min(1.0, gini))

        sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
        celdas_con_sensores = self.grid_cali
        for i, nodo in enumerate(nodos):
            punto = Point(nodo[0], nodo[1])
            for _, celda in celdas_con_sensores.iterrows():
                if celda['geometry'].contains(punto):
                    nivel = int(celda['nivel_inseguridad'])
                    sensores_por_nivel[nivel] += 1
                    break
        
        sensores_por_celda = []
        for _, celda in self.grid_cali.iterrows():
            poly = celda['geometry']
            count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_riesgo)
            sensores_por_celda.append(count)
            
        total_celdas = 0
        total_sensores = 0
        for nivel in range(6):
            total_celdas += len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
            total_sensores += sensores_por_nivel.get(nivel, 0)

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelent',
            np.where(rssi_dbm > -100, 'Good',
                    'Regular')
        )


        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
    
        clasificacion = []

        celdas_con_sensores = self.grid_cali
        for i, nodo in enumerate(nodos):
            punto = Point(nodo[0], nodo[1])
            clasi = "Sin datos"
            for _, celda in celdas_con_sensores.iterrows():
                if celda['geometry'].contains(punto):
                    nivel = int(celda['nivel_inseguridad'])
                    clasi = colores_nivel[nivel]
                    break
            clasificacion.append(clasi)

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'clasificacion': clasificacion,
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })
        
        output_path = Path(__file__).resolve().parent / "data_base"
        output_path.mkdir(parents=True, exist_ok=True)
        out_path_simple = output_path / "informacion_nodos_riesgo.csv"

        df.to_csv(out_path_simple, index=False)
            
            
        # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            },
            'total_celdas': int(total_celdas),
            'total_sensores': int(total_sensores)
        }


        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                except Exception:
                    pass

            rect = Rectangle((x_ini, y_ini), ancho, alto, linewidth=5,
                             edgecolor='deepskyblue', facecolor='aqua', alpha=0.8, zorder=4)
            ax.add_patch(rect)
            
            print("\n   Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Clasificacion':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)
            
            # Lista para guardar los datos de cada nivel
            datos_niveles = []
            
            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
                datos_niveles.append({
                    'Nivel': nivel,
                    'Clasificacion': colores_nivel[nivel],
                    'Celdas': count_celdas,
                    'Sensores': count_sensores,
                    'Promedio': round(promedio, 2)
                })
            
            print("  " + "-"*65)
            print(f"Gini total de sensores en el mapa: {gini:.4f}")
            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")
            
            # Agregar fila de TOTAL al DataFrame
            datos_niveles.append({
                'Nivel': 'TOTAL',
                'Clasificacion': '',
                'Celdas': total_celdas,
                'Sensores': total_sensores,
                'Promedio': gini
            })
            
            df_nodos_riesgo = pd.DataFrame(datos_niveles)
            
            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "resumen_niveles_riesgo.csv"
            
            df_nodos_riesgo.to_csv(out_path_simple, index=False)
            print(f"\nResumen de niveles guardado en: {out_path_simple}")

            color_map = {
                'Excellent': "green",
                'Good': "yellow",
                'Regular': "#f05209",
                'Excelente': "green",
                'Buena': "yellow",
                'Excelent': "green"
            }
  

            calidad_map = {
                'Excelente': 'Excellent',
                'Buena': 'Good',
                'Excelent': 'Excellent'
            }
            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excelente'], markersize=8, label=f"Excelente ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Buena'], markersize=8, label=f"Buena ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"Gateways GA ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title('Number of Nodes by Log-Distance using Risk Model', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }
    

def gtw_distance_uniforme(self, max_intentos=100,
        graficar=True, guardar=True, nombre_mapa='qos_logdistance_uniforme.png'):

        # Verificar si ya existen nodos IoT uniforme
        if getattr(self, 'nodos_iot_uniforme', None) is not None:
            print("\n Usando nodos IoT uniforme ya generados")
            print(f"Nodos reutilizados: {len(self.nodos_iot_uniforme)}")
        else:
            print("\n Generando nodos IoT desde grid_cali uniforme...")
            nodos_coords = []
            
            celdas = self.grid_cali
            #de forma uniforme
            M = len(celdas)
            n_base = 7825 // M
            resto = 7825 - (n_base * M)
            sensores = np.full(M, n_base)
            if resto > 0:
                indices = np.random.choice(M, size=resto, replace=False)
                sensores[indices] += 1

            #total_uniforme = n_base * M
            #if total_uniforme != total_sensores:
                #print(f"  ⚠ Ajuste uniforme: {total_sensores} -> {total_uniforme} sensores (M={M})")
            #sensores = np.full(M, n_base)
            

            #de forma aleatoria
            M = len(celdas)
            # Para cada nodo, elige una celda al azar
            #indices = np.random.choice(M, size=7828, replace=True)
            for idx, (_, row) in enumerate(celdas.iterrows()):
                n_nodos = sensores[idx]
                poly = row['geometry']
                minx, miny, maxx, maxy = poly.bounds

                for _ in range(n_nodos):
                    intentos = 0
                    while intentos < max_intentos:
                        x = np.random.uniform(minx, maxx)
                        y = np.random.uniform(miny, maxy)
                        punto = Point(x, y)
                        if poly.contains(punto):
                            nodos_coords.append([x, y])
                            break
                        intentos += 1
                    if intentos >= max_intentos:
                        centroide = poly.centroid
                        nodos_coords.append([centroide.x, centroide.y])
            
            self.nodos_iot_uniforme = np.array(nodos_coords)
            print(f"  • Nodos generados: {len(self.nodos_iot_uniforme)}")

        nodos = np.asarray(self.nodos_iot_uniforme)
        gateways = np.asarray(self.gateways_ga)

        # Guardar nodos en CSV
        if guardar:
            df_nodos = pd.DataFrame(
                self.nodos_iot_uniforme,
                columns=['x', 'y']
            )
            
            # Agregar ID de nodo
            df_nodos.insert(0, 'id', range(1, len(self.nodos_iot_uniforme) + 1))

            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "nodos_uniforme.csv"

            df_nodos.to_csv(out_path_simple, index=False)
            print(f"Nodos uniforme guardados en: {out_path_simple}")

            # Calcular Gini
            y = np.array(sensores_por_celda, dtype=float)
            n = len(y)
            y_sorted = np.sort(y)
            suma_total = float(np.sum(y_sorted))
            if suma_total == 0:
                gini = 0.0
            else:
                i = np.arange(1, n + 1, dtype=float)
                gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                gini = max(0.0, min(1.0, gini))

        sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
        celdas_con_sensores = self.grid_cali
        for i, nodo in enumerate(nodos):
            punto = Point(nodo[0], nodo[1])
            for _, celda in celdas_con_sensores.iterrows():
                if celda['geometry'].contains(punto):
                    nivel = int(celda['nivel_inseguridad'])
                    sensores_por_nivel[nivel] += 1
                    break
        
        sensores_por_celda = []
        for _, celda in self.grid_cali.iterrows():
            poly = celda['geometry']
            count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_uniforme)
            sensores_por_celda.append(count)

        total_celdas = 0
        total_sensores = 0
        for nivel in range(6):
            total_celdas += len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
            total_sensores += sensores_por_nivel.get(nivel, 0)

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelente',
            np.where(rssi_dbm > -100, 'Buena',
                    'Regular')
        )

        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })

                    # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            },
            'total_celdas': int(total_celdas),
            'total_sensores': int(total_sensores)
        }

        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)
                    
                    # Mostrar conteo de celdas por nivel de inseguridad
                    colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
                    
                    # Contar sensores por nivel usando las asignaciones y celdas
                    # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                    sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    
                    celdas_con_sensores = self.grid_cali
                    for i, nodo in enumerate(nodos):
                        punto = Point(nodo[0], nodo[1])
                        for _, celda in celdas_con_sensores.iterrows():
                            if celda['geometry'].contains(punto):
                                nivel = int(celda['nivel_inseguridad'])
                                sensores_por_nivel[nivel] += 1
                                break
                except Exception:
                    pass
            
            print("\n   Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Color':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)

            # Lista para guardar datos de cada nivel
            datos_niveles = []

            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
                datos_niveles.append({
                    'Nivel': nivel,
                    'Clasificacion': colores_nivel[nivel],
                    'Celdas': count_celdas,
                    'Sensores': count_sensores,
                    'Promedio': round(promedio, 2)
                })
            
            print("  " + "-"*65)
            print(f"Gini total de sensores en el mapa: {gini:.4f}")
            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")

            # Agregar fila de TOTAL al DataFrame
            datos_niveles.append({
                'Nivel': 'TOTAL',
                'Clasificacion': '',
                'Celdas': total_celdas,
                'Sensores': total_sensores,
                'Promedio': gini
            })
            
            df_nodos_uniforme = pd.DataFrame(datos_niveles)
            
            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "resumen_niveles_uniforme.csv"
            
            df_nodos_uniforme.to_csv(out_path_simple, index=False)
            print(f"Resumen de niveles guardado en: {out_path_simple}")

            color_map = {
                'Excellent': "green",
                'Good': "yellow",
                'Regular': "#f05209",
            }

            calidad_map = {
                'Excelente': 'Excellent',
                'Buena': 'Good',
                'Excelent': 'Excellent'
            }
    
            calidad_norm = [calidad_map.get(c, c) for c in calidad]
            colores_nodos = [color_map[c] for c in calidad_norm]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excellent'], markersize=8, label=f"Excellent ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Good'], markersize=8, label=f"Good ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"GA Gateways ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            
            ax.set_title('Number of Nodes by Log-Distance using Uniform Base Mode', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }
    
def gtw_distance_aleatorio(self, max_intentos=100,
        graficar=True, guardar=True, nombre_mapa='qos_logdistance_aleatorio.png'):

        # Verificar si ya existen nodos IoT aleatorio
        if getattr(self, 'nodos_iot_aleatorio', None) is not None:
            print("\nUsando nodos IoT aleatorio ya generados")
            print(f"Nodos reutilizados: {len(self.nodos_iot_aleatorio)}")
        else:
            print("\n Generando nodos IoT desde grid_cali aleatorio...")
            nodos_coords = []
            
            celdas = self.grid_cali

            #de forma aleatoria
            M = len(celdas)
            # Para cada nodo, elige una celda al azar
            indices = np.random.choice(M, size=7825, replace=True)
            for idx in indices:
                row = celdas.iloc[idx]
                poly = row['geometry']
                minx, miny, maxx, maxy = poly.bounds

                intentos = 0
                while intentos < max_intentos:
                        x = np.random.uniform(minx, maxx)
                        y = np.random.uniform(miny, maxy)
                        punto = Point(x, y)
                        if poly.contains(punto):
                            nodos_coords.append([x, y])
                            break
                        intentos += 1
                if intentos >= max_intentos:
                        centroide = poly.centroid
                        nodos_coords.append([centroide.x, centroide.y])
            
            self.nodos_iot_aleatorio = np.array(nodos_coords)
            print(f"  • Nodos generados: {len(self.nodos_iot_aleatorio)}")
        
        nodos = np.asarray(self.nodos_iot_aleatorio)
            
        gateways = np.asarray(self.gateways_ga)

        # Guardar nodos en CSV
        if guardar:
            df_nodos = pd.DataFrame(
                self.nodos_iot_aleatorio,
                columns=['x', 'y']
            )
            
            # Agregar ID de nodo
            df_nodos.insert(0, 'id', range(1, len(self.nodos_iot_aleatorio) + 1))

            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "nodos_aleatorio.csv"

            df_nodos.to_csv(out_path_simple, index=False)
            print(f"Nodos aleatorio guardados en: {out_path_simple}")
            
            # Calcular Gini, total_celdas y total_sensores (siempre, no solo si graficar)
            
            y = np.array(sensores_por_celda, dtype=float)
            n = len(y)
            y_sorted = np.sort(y)
            suma_total = float(np.sum(y_sorted))
            if suma_total == 0:
                gini = 0.0
            else:
                i = np.arange(1, n + 1, dtype=float)
                gini = (2.0 * np.sum(i * y_sorted)) / (n * suma_total) - (n + 1.0) / n
                gini = max(0.0, min(1.0, gini))


        sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
        celdas_con_sensores = self.grid_cali
        for i, nodo in enumerate(nodos):
            punto = Point(nodo[0], nodo[1])
            for _, celda in celdas_con_sensores.iterrows():
                if celda['geometry'].contains(punto):
                    nivel = int(celda['nivel_inseguridad'])
                    sensores_por_nivel[nivel] += 1
                    break
        
        sensores_por_celda = []
        for _, celda in self.grid_cali.iterrows():
            poly = celda['geometry']
            count = sum(poly.contains(Point(x, y)) for x, y in self.nodos_iot_aleatorio)
            sensores_por_celda.append(count)

        total_celdas = 0
        total_sensores = 0
        for nivel in range(6):
            total_celdas += len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
            total_sensores += sensores_por_nivel.get(nivel, 0)

        # Parámetros de propagación (urbano)
        frequency_mhz = 915.0
        path_loss_exponent = 2.7
        d0 = 1.0  # m
        tx_power_dbm = 14.0

        # Pérdida en distancia de referencia (espacio libre)
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency_mhz * 1e6 / 3e8)

        # Distancias de cada nodo a cada gateway -> matriz (N, K)
        delta = nodos[:, np.newaxis, :] - gateways[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=2)
        dist_clip = np.maximum(dist_matrix, d0)

        # Path loss y RSSI para cada pareja nodo-gateway
        path_loss_matrix = pl_d0 + 10 * path_loss_exponent * np.log10(dist_clip / d0)
        rssi_matrix = tx_power_dbm - path_loss_matrix

        # Elegir mejor gateway por mayor RSSI (equivale a menor path loss)
        assignments_modelo = np.argmax(rssi_matrix, axis=1)

        # Extraer métricas del mejor enlace por nodo
        idx = np.arange(len(nodos))
        dist = dist_matrix[idx, assignments_modelo]
        path_loss_db = path_loss_matrix[idx, assignments_modelo]
        rssi_dbm = rssi_matrix[idx, assignments_modelo]

        # Clasificación de calidad por RSSI
        calidad = np.where(
            rssi_dbm > -80, 'Excelente',
            np.where(rssi_dbm > -100, 'Buena',
                    'Regular')
        )

        # Guardar asignación basada en modelo
        self.assignments_logdistance_ga = assignments_modelo

        # DataFrame por nodo
        df = pd.DataFrame({
            'id_nodo': np.arange(1, len(nodos) + 1),
            'id_gateway_logdistance': assignments_modelo + 1,
            'distancia_m': dist,
            'path_loss_db': path_loss_db,
            'rssi_dbm': rssi_dbm,
            'calidad': calidad
        })

        # Resumen
        conteo_calidad = df['calidad'].value_counts().to_dict()
        resumen = {
            'n_nodos': int(len(df)),
            'n_gateways_ga': int(len(gateways)),
            'distancia_m': {
                'min': float(df['distancia_m'].min()),
                'mean': float(df['distancia_m'].mean()),
                'max': float(df['distancia_m'].max())
            },
            'path_loss_db': {
                'min': float(df['path_loss_db'].min()),
                'mean': float(df['path_loss_db'].mean()),
                'max': float(df['path_loss_db'].max())
            },
            'rssi_dbm': {
                'min': float(df['rssi_dbm'].min()),
                'mean': float(df['rssi_dbm'].mean()),
                'max': float(df['rssi_dbm'].max())
            },
            'calidad_nodos': {
                'Excelente': int(conteo_calidad.get('Excelente', 0)),
                'Buena': int(conteo_calidad.get('Buena', 0)),
                'Regular': int(conteo_calidad.get('Regular', 0))
            },
            'total_celdas': int(total_celdas),
            'total_sensores': int(total_sensores)
        }

        if graficar:
            fig, ax = plt.subplots(figsize=(16, 12))

            # Mapa base si está disponible
            self.cali.plot(ax=ax, color='white', edgecolor='black', linewidth=2.0, zorder=1)

            if getattr(self, 'grid_cali', None) is not None:
                try:
                    self.grid_cali.plot(ax=ax, color="white", edgecolor="black", linewidth=0.25, zorder=2)

                    # Mostrar conteo de celdas por nivel de inseguridad
                    colores_nivel = {0: "Sin datos", 1:  "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"}
                    
                    # Contar sensores por nivel usando las asignaciones y celdas
                    # Solo buscar en celdas con nivel > 0 (donde se generan nodos)
                    sensores_por_nivel = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    
                    celdas_con_sensores = self.grid_cali
                    for i, nodo in enumerate(nodos):
                        punto = Point(nodo[0], nodo[1])
                        for _, celda in celdas_con_sensores.iterrows():
                            if celda['geometry'].contains(punto):
                                nivel = int(celda['nivel_inseguridad'])
                                sensores_por_nivel[nivel] += 1
                                break
                except Exception:
                    pass
            
            print("\n  Celdas y sensores por nivel de inseguridad:")
            print("  " + "-"*65)
            print(f"  {'Nivel':<8} {'Color':<12} {'Celdas':<10} {'Sensores':<10} {'Promedio':<10}")
            print("  " + "-"*65)
            
            # Lista para guardar datos de cada nivel
            datos_niveles = []
            
            for nivel in range(6):
                count_celdas = len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == nivel])
                count_sensores = sensores_por_nivel.get(nivel, 0)
                promedio = (count_sensores / count_celdas) if count_celdas > 0 else 0
                
                print(f"  {nivel:<8} {colores_nivel[nivel]:<12} {count_celdas:<10} {count_sensores:<10} {promedio:<10.2f}")
                datos_niveles.append({
                    'Nivel': nivel,
                    'Clasificacion': colores_nivel[nivel],
                    'Celdas': count_celdas,
                    'Sensores': count_sensores,
                    'Promedio': round(promedio, 2)
                })
            
            print("  " + "-"*65)
            print(f"Gini total de sensores en el mapa: {gini:.4f}")
            print(f"  {'TOTAL':<8} {'':<12} {total_celdas:<10} {total_sensores:<10} Gini={gini:<10}")
            print(f"  Celdas sin sensores (nivel 0): {sensores_por_nivel.get(0, 0)} sensores en {len(self.grid_cali[self.grid_cali['nivel_inseguridad'] == 0])} celdas")

            # Agregar fila de TOTAL al DataFrame
            datos_niveles.append({
                'Nivel': 'TOTAL',
                'Clasificacion': '',
                'Celdas': total_celdas,
                'Sensores': total_sensores,
                'Promedio': gini
            })
            
            df_nodos_aleatorio = pd.DataFrame(datos_niveles)
            
            output_path = Path(__file__).resolve().parent / "data_base"
            output_path.mkdir(parents=True, exist_ok=True)
            out_path_simple = output_path / "resumen_niveles_aleatorio.csv"
            
            df_nodos_aleatorio.to_csv(out_path_simple, index=False)
            print(f"Resumen de niveles guardado en: {out_path_simple}")

            color_map = {
                'Excelente': "green",
                'Buena': "yellow",
                'Regular': "#f05209",
            }
            colores_nodos = [color_map[c] for c in calidad]

            ax.scatter(nodos[:, 0], nodos[:, 1], c=colores_nodos, s=22, alpha=0.75,
                       edgecolors='none', zorder=4, label=f'Nodos ({len(nodos)})')

            ax.scatter(gateways[:, 0], gateways[:, 1], c='black', marker='^', s=220,
                       edgecolors='white', linewidth=1.2, zorder=3,
                       label=f'Gateways GA ({len(gateways)})')

            # Conexiones suaves según asignación del modelo
            gw_sel = gateways[assignments_modelo]
            for i in range(len(nodos)):
                ax.plot([nodos[i, 0], gw_sel[i, 0]], [nodos[i, 1], gw_sel[i, 1]],
                        color='gray', alpha=0.05, linewidth=0.4, zorder=3)

            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Excelente'], markersize=8, label=f"Excelente ({resumen['calidad_nodos']['Excelente']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Buena'], markersize=8, label=f"Buena ({resumen['calidad_nodos']['Buena']})"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['Regular'], markersize=8, label=f"Regular ({resumen['calidad_nodos']['Regular']})"),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label=f"Gateways GA ({len(gateways)})")
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=True, edgecolor='black')

            ax.set_xlabel('X Coordinate (meters)', fontsize=14)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
            ax.set_title('Number of Nodes by Log-Distance using Random Base Model', fontsize=14, weight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.25, linestyle='--')
            plt.tight_layout()

            if guardar:
                images_dir = Path(__file__).resolve().parent / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                outname = images_dir / nombre_mapa
                plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Mapa QoS guardado: {outname}")

            #plt.show()

        return {
            'dataframe': df,
            'resumen': resumen,
            'assignments_modelo': assignments_modelo
        }
