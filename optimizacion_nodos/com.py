import pandas as pd
import os
import numpy as np

class Com:

    @staticmethod
    def _metricas(filepath: str) -> dict:
        """Lee un CSV y devuelve el promedio de PDR, Energy_mJ, Latency_ms y lambda."""
        df = pd.read_csv(filepath)
        return {
            'PDR':        pd.to_numeric(df['PDR'],        errors='coerce').mean(),
            'Energy_mJ':  pd.to_numeric(df['Energy_mJ'],  errors='coerce').mean(),
            'Latency_ms': pd.to_numeric(df['Latency_ms'], errors='coerce').mean(),
            'lambda':     pd.to_numeric(df['lambda'],     errors='coerce').mean(),
        }

    @staticmethod
    def _calcular_rangos(csvs: list) -> dict:
        """
        Detecta min/max de cada métrica en todos los CSVs.
        
        Args:
            csvs: Lista de rutas a archivos CSV
            
        Returns:
            Dict con ranges: {'PDR': {'min': x, 'max': y}, ...}
        """
        rangos = {
            'PDR': {'min': float('inf'), 'max': float('-inf')},
            'Energy_mJ': {'min': float('inf'), 'max': float('-inf')},
            'Latency_ms': {'min': float('inf'), 'max': float('-inf')},
            'lambda': {'min': float('inf'), 'max': float('-inf')},
        }
        
        for csv_path in csvs:
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                for metrica in rangos.keys():
                    if metrica in df.columns:
                        valores = pd.to_numeric(df[metrica], errors='coerce').dropna()
                        if len(valores) > 0:
                            rangos[metrica]['min'] = min(rangos[metrica]['min'], valores.min())
                            rangos[metrica]['max'] = max(rangos[metrica]['max'], valores.max())
            except Exception as e:
                print(f"  ⚠ Error leyendo {csv_path}: {e}")
        
        print("\n📊 RANGOS DETECTADOS:")
        for metrica, rango in rangos.items():
            if rango['min'] != float('inf'):
                print(f"  {metrica:<12}: [{rango['min']:.4f}, {rango['max']:.4f}]")
        
        return rangos

    @staticmethod
    def _normalizar_metricas(metricas: dict, rangos: dict) -> dict:
        """
        Normaliza todas las métricas con fórmulas específicas:
        - PDR (beneficio): (value - min) / (max - min)           → 0=malo, 1=bueno
        - Energy (costo): (max - value) / (max - min)            → 0=malo, 1=bueno
        - Latency (costo): (max - value) / (max - min)           → 0=malo, 1=bueno
        - lambda (costo): (max - value) / (max - min)            → 0=malo, 1=bueno
        
        Args:
            metricas: Dict con PDR, Energy_mJ, Latency_ms, lambda
            rangos: Dict con min/max de cada métrica
            
        Returns:
            Dict normalizado + score combinado
        """
        norm = {}
        
        # PDR: Beneficio (directa)
        rango_pdr = rangos['PDR']['max'] - rangos['PDR']['min']
        if rango_pdr > 0:
            norm['PDR'] = (metricas['PDR'] - rangos['PDR']['min']) / rango_pdr
        else:
            norm['PDR'] = 0.0
        norm['PDR'] = np.clip(norm['PDR'], 0, 1)
        
        # Energy: Costo (invertida)
        rango_energy = rangos['Energy_mJ']['max'] - rangos['Energy_mJ']['min']
        if rango_energy > 0:
            norm['Energy_mJ'] = (rangos['Energy_mJ']['max'] - metricas['Energy_mJ']) / rango_energy
        else:
            norm['Energy_mJ'] = 0.0
        norm['Energy_mJ'] = np.clip(norm['Energy_mJ'], 0, 1)
        
        # Latency: Costo (invertida)
        rango_latency = rangos['Latency_ms']['max'] - rangos['Latency_ms']['min']
        if rango_latency > 0:
            norm['Latency_ms'] = (rangos['Latency_ms']['max'] - metricas['Latency_ms']) / rango_latency
        else:
            norm['Latency_ms'] = 0.0
        norm['Latency_ms'] = np.clip(norm['Latency_ms'], 0, 1)
        
        # Lambda: Costo (invertida)
        rango_lambda = rangos['lambda']['max'] - rangos['lambda']['min']
        if rango_lambda > 0:
            norm['lambda'] = (rangos['lambda']['max'] - metricas['lambda']) / rango_lambda
        else:
            norm['lambda'] = 0.0
        norm['lambda'] = np.clip(norm['lambda'], 0, 1)
        
        # Score total: Promedio de todas las métricas normalizadas
        norm['score'] = (norm['PDR'] + norm['Energy_mJ'] + norm['Latency_ms'] + norm['lambda']) / 4.0
        
        return norm


    @staticmethod
    def evaluar_mejor_algoritmo(nsga_res: str, nsga_comp: str,
                                ga_res:   str, ga_comp:   str,
                                rl_res:   str, rl_comp:   str) -> dict:
        """
        Normaliza y compara algoritmos con métricas normalizadas [0,1].
        Calcula scores con:
        - PDR (beneficio): (PDR - min) / (max - min)
        - Energy (costo): (max - Energy) / (max - min)
        - Latency (costo): (max - Latency) / (max - min)
        - Lambda (costo): (max - lambda) / (max - min)
        
        Score = (PDR_norm + Energy_norm + Latency_norm + Lambda_norm) / 4
        """
        algoritmos = {
            'NSGA-II': (nsga_res, nsga_comp),
            'GA':      (ga_res,   ga_comp),
            'RL':      (rl_res,   rl_comp),
        }

        # ✓ Calcular rangos de TODAS las métricas en todos los CSVs
        todos_csvs = [nsga_res, nsga_comp, ga_res, ga_comp, rl_res, rl_comp]
        rangos = Com._calcular_rangos(todos_csvs)

        resultados = {}

        for nombre, (res, comp) in algoritmos.items():
            try:
                m_algo  = Com._metricas(res)
                m_omnet = Com._metricas(comp)

                # ✓ Normalizar métricas del algoritmo
                norm_algo = Com._normalizar_metricas(m_algo, rangos)
                
                # ✓ Normalizar métricas de OMNeT++
                norm_omnet = Com._normalizar_metricas(m_omnet, rangos)

                # ✓ Calcular diferencias normalizadas
                diff_pdr = abs(norm_algo['PDR'] - norm_omnet['PDR'])
                diff_energy = abs(norm_algo['Energy_mJ'] - norm_omnet['Energy_mJ'])
                diff_latency = abs(norm_algo['Latency_ms'] - norm_omnet['Latency_ms'])
                diff_lambda = abs(norm_algo['lambda'] - norm_omnet['lambda'])
                
                # ✓ Total = suma de diferencias (método de Manhattan)
                diff_total = diff_pdr + diff_energy + diff_latency + diff_lambda

                resultados[nombre] = {
                    'algoritmo': m_algo,
                    'omnet':     m_omnet,
                    'norm_algo': norm_algo,
                    'norm_omnet': norm_omnet,
                    'diff': {
                        'PDR_norm':        diff_pdr,
                        'Energy_mJ_norm':  diff_energy,
                        'Latency_ms_norm': diff_latency,
                        'lambda_norm':     diff_lambda,
                        'total':           diff_total,
                    }
                }

            except Exception as e:
                print(f" Error en {nombre}: {e}")

        if not resultados:
            return None

        # Elegir el mejor por menor diferencia total
        mejor = min(resultados.items(), key=lambda x: x[1]['diff']['total'])

        return {
            'mejor_algoritmo': mejor[0],
            'diff_total':      mejor[1]['diff']['total'],
            'resultados':      resultados,
        }


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    resultado = Com.evaluar_mejor_algoritmo(
        nsga_res  = os.path.join(BASE_PATH, 'resultados',  'nsga_pareto_solutions.csv'),
        nsga_comp = os.path.join(BASE_PATH, 'comparacion', 'nodos_lorawan_nsga.csv'),
        ga_res    = os.path.join(BASE_PATH, 'resultados',  'ga_best_solutions.csv'),
        ga_comp   = os.path.join(BASE_PATH, 'comparacion', 'nodos_lorawan_ga.csv'),
        rl_res    = os.path.join(BASE_PATH, 'resultados',  'rl_optimized_nodes.csv'),
        rl_comp   = os.path.join(BASE_PATH, 'comparacion', 'nodos_lorawan_rl.csv'),
    )

    if not resultado:
        print("Error: no se pudo evaluar ningún algoritmo.")
    else:
        print("\n" + "=" * 100)
        print("COMPARACIÓN DE ALGORITMOS DE OPTIMIZACIÓN LoRaWAN (MÉTRICAS NORMALIZADAS [0,1])")
        print("Método: Total = |ΔPDR| + |ΔEnergy| + |ΔLatency| + |Δlambda|")
        print("=" * 100)

        for nombre, datos in resultado['resultados'].items():
            print(f"\n{'─' * 100}")
            print(f" {nombre}")
            print(f"{'─' * 100}")
            
            # Tabla de valores brutos
            print(f"\n  {'VALORES BRUTOS':<40} {'Algoritmo':>15} {'OMNeT++':>15}")
            print(f"  {'-'*70}")
            for m in ['PDR', 'Energy_mJ', 'Latency_ms', 'lambda']:
                v_algo = datos['algoritmo'][m]
                v_omn  = datos['omnet'][m]
                print(f"  {m:<40} {v_algo:>15.4f} {v_omn:>15.4f}")
            
            # Tabla de valores normalizados
            print(f"\n  {'VALORES NORMALIZADOS [0,1]':<40} {'Algoritmo':>15} {'OMNeT++':>15} {'Diferencia':>15}")
            print(f"  {'-'*85}")
            for m, key_diff in [('PDR', 'PDR_norm'), ('Energy_mJ', 'Energy_mJ_norm'), 
                                ('Latency_ms', 'Latency_ms_norm'), ('lambda', 'lambda_norm')]:
                v_algo = datos['norm_algo'][m]
                v_omn  = datos['norm_omnet'][m]
                v_diff = datos['diff'][key_diff]
                print(f"  {m:<40} {v_algo:>15.4f} {v_omn:>15.4f} {v_diff:>15.4f}")
            
            # Total
            diff_total = datos['diff']['total']
            print(f"  {'-'*85}")
            print(f"  {'TOTAL Δ (suma diferencias)':<40} {'':>15} {'':>15} {diff_total:>15.4f}")

        print(f"\n{'=' * 100}")
        print(f"   MEJOR ALGORITMO: {resultado['mejor_algoritmo']}")
        print(f"   Diferencia Total: {resultado['diff_total']:.4f}")
        print(f"   Fórmula: |ΔPDR| + |ΔEnergy| + |ΔLatency| + |Δlambda|")
        print("=" * 100 + "\n")