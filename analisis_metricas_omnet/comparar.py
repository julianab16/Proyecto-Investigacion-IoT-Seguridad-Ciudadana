"""
comparar.py — adaptado al output real de analisis2.py
======================================================

Lee:
    analisis_metricas_omnet/por_nodo/metricas_nodo_<cfg><sufijo>.csv
    analisis_metricas_omnet/por_nodo/metricas_gw_<cfg><sufijo>.csv
    analisis_metricas_omnet/por_nodo/metricas_agregadas_<cfg><sufijo>.csv
    analisis_metricas_omnet/por_nodo/totales_<cfg><sufijo>.csv
    analisis_metricas_omnet/por_nodo/der_plr_por_run_<cfg><sufijo>.csv
    analisis_metricas_omnet/histogramas/rssi_<cfg><sufijo>.csv

Genera en analisis_metricas_omnet/resultados_comparacion/:
    pdr_global_escenarios.png
    mean_delay_escenarios.png
    mean_toa_escenarios.png
    rssi_histograma_todos.png
    sf_histograma_todos.png
    resumen_escenarios.csv
"""
import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ESCENARIOS = ['Baja','Media', 'Alta']
COLOR_SIN  = '#4C72B0'
COLOR_CON  = '#DD8452'
HATCHES = ['/','-']
PAGE_WIDTH_IN  = 7.16 #7.12
PAGE_HEIGHT_IN = 4.5 #4.45
import matplotlib as mpl
mpl.rcParams.update({
    'font.family':     'serif',
    'font.serif':      'Arial',
    'font.size':        11,
    'axes.labelsize':   15,
    'axes.titlesize':   15,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'legend.fontsize':  15,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.4,
    'lines.linewidth':  1.0,
    'axes.linewidth':   0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# nombre interno → (etiqueta corta, etiqueta larga)
SCENARIO_META = {
    'Baja':  ('600s', 'Low'),
    'Media': ('200s', 'Medium'),
    'Alta':  ('100s',  'High'),
}

def _label_corto(cfg):
    meta = SCENARIO_META.get(cfg)
    return meta[1] if meta else cfg

def _aplicar_textura_fija(barras, hatch):
    for barra in barras:
        barra.set_hatch(hatch)
# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
def load_nodo(path):
    """
    metricas_nodo_*.csv:
      node_id, packets_sent_mean/std, finalSF_mean/std,
      finalTP_mean/std, mac_numSent_mean/std, mac_numReceived_mean/std,
      mac_numCollisions_mean/std, mac_numRetry_mean/std, toa_ms_from_finalSF
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    for c in df.columns:
        if c == 'node_id':
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['packets_sent_mean', 'finalsf_mean',
              'mac_numCollisions_mean', 'toa_ms_from_finalsf']:
        if c not in df.columns:
            df[c] = np.nan
    return df

def load_agregadas(path):
    """metricas_agregadas_*.csv: metric, mean, std"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    df['std']  = pd.to_numeric(df['std'],  errors='coerce')
    return {row['metric']: (row['mean'], row['std'])
            for _, row in df.iterrows()}

def load_totales(path):
    """
    totales_*.csv: metric, mean, std → dict[metric] = (mean, std)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    df['std']  = pd.to_numeric(df['std'],  errors='coerce')
    return {row['metric']: (row['mean'], row['std'])
            for _, row in df.iterrows()}

def load_der_plr_por_run(path):
    """Carga DER_r y PLR_r calculados para cada ejecución."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {'run', 'der', 'plr'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} no contiene las columnas requeridas: {sorted(missing)}"
        )
    for column in ['run', 'der', 'plr']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df.dropna(subset=['run', 'der', 'plr']).copy()

def calcular_estadisticas_plr(df):
    """Calcula media y desviación estándar poblacional de los PLR_r."""
    plr = df['plr'].to_numpy(dtype=float)
    if plr.size == 0:
        raise ValueError('No hay valores PLR válidos para calcular estadísticas')
    return float(np.mean(plr)), float(np.std(plr, ddof=0))


def load_hist(path):
    """
    rssi_*.csv: bin_low, bin_high, densidad_mean, densidad_std,
                count_mean, n_runs
    """
    return pd.read_csv(path)

def cargar_posiciones_gw(path_ini):
    """
    Lee gateways_posiciones*.ini → dict[gw_id] = (x, y) en metros.
    Formato esperado:
        *.loRaGW[0].**.initialX = 970.72m
        *.loRaGW[0].**.initialY = 1069.94m
    También acepta .mobility.initialX por compatibilidad.
    """
    gw_pos = {}
    pattern = re.compile(
        r'^\*\.loRaGW\[(\d+)\]\.(?:\*\*|mobility)\.initial([XY])\s*=\s*([-\d.]+)\s*m\s*$'
    )
    with open(path_ini) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                gid = int(m.group(1))
                axis = m.group(2)
                val = float(m.group(3))
                gw_pos.setdefault(gid, {})[axis] = val
    return {gid: (d.get('X', np.nan), d.get('Y', np.nan))
            for gid, d in gw_pos.items()}


def asignar_zona(x, y, gw_pos, umbrales = (50, 100, 150)):
    if pd.isna(x) or pd.isna(y) or not gw_pos:
        return 'unknown'

    d = min(np.hypot(x-gx, y-gy)
            for gx, gy in gw_pos.values()
            if pd.notna(gx) and pd.notna(gy))

    if d < umbrales[0]:
        return 'near'      # <500 m
    elif d < umbrales[1]:
        return 'mid'       # 500–1000 m
    elif d < umbrales[2]:
        return 'far'       # 1–2 km
    else:
        return 'edge'      # >2 km


def plot_pdr_rssi_por_zona(nodo_con, nodo_sin,
                           gw_ini_con, gw_ini_sin, out_path):
    """
    Grafica únicamente el RSSI medio por zona y escenario.
    """
    gw_pos_con = cargar_posiciones_gw(gw_ini_con)
    gw_pos_sin = cargar_posiciones_gw(gw_ini_sin)

    ZONAS = ['near', 'mid', 'far', 'edge']
    ZLABELS = [
        '<50',
        '50-100',
        '100-150',
        '≥150'
    ]
    
    def agregar_por_zona(df, gw_pos):
        if df is None or df.empty:
            return {}

        required = ['positionx','positiony',
                    'rssi_dbm_mean']

        if not all(c in df.columns for c in required):
            return {}

        sub = df.copy()

        sub['zona'] = sub.apply(
            lambda r: asignar_zona(r.positionx, r.positiony, gw_pos),
            axis=1
        )

        sub = sub[sub['zona'] != 'unknown']

        out = {}
        for z, g in sub.groupby('zona'):
            out[z] = (
                g['rssi_dbm_mean'].mean(),
                len(g)
            )

        return out

    fig, axes = plt.subplots(
        1, 3, figsize=(8.5, PAGE_HEIGHT_IN),
        gridspec_kw={'width_ratios': [1.7, 1.7, 1.7]},
    )

    for col, cfg in enumerate(ESCENARIOS):
        z_sin = agregar_por_zona(nodo_sin.get(cfg), gw_pos_sin)
        z_con = agregar_por_zona(nodo_con.get(cfg), gw_pos_con)
        if not z_sin and not z_con:
            axes[col].set_visible(False)
            continue

        x = np.arange(len(ZONAS)); w = 0.38

        ax = axes[col]
        ax.set_ylim(-120, -70)
        rssi_sin = [z_sin.get(z, (np.nan,)*2)[0] for z in ZONAS]
        rssi_con = [z_con.get(z, (np.nan,)*2)[0] for z in ZONAS]
        barras_sin = ax.bar(x - w/2, rssi_sin, w, label='Baseline',
                    color=COLOR_SIN, edgecolor='black', linewidth=0.5)
        barras_con = ax.bar(x + w/2, rssi_con, w, label='RiSeG', 
                    color=COLOR_CON, edgecolor='black', linewidth=0.5)
        _aplicar_textura_fija(barras_sin, '/')   # Baseline siempre /
        _aplicar_textura_fija(barras_con, '-')   # RiSeG   siempre -

        ax.set_xticks(x)
        ax.set_xticklabels(ZLABELS, rotation=0, ha='center', fontsize=10,
        )
        # if col == len(ESCENARIOS) // 2:
        #     ax.set_xticklabels(
        #         ZLABELS, rotation=0, ha='center', fontsize=8,
        #     )
        # else:
        #     ax.set_xticklabels([])
        ax.set_ylabel('RSSI mean (dBm)')
        ax.set_title(_label_corto(cfg))
        ax.grid(True, alpha=0.2, axis='y')
        ax.legend()

    fig.supxlabel('Distance to nearest GW (m)', y=0.055, fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, format='png', dpi=300)
    plt.close()
    print(f"  Guardado: {out_path}")

# ─────────────────────────────────────────────
# FIGURA 1: PDR global (LoRa_NS_DER)
# ─────────────────────────────────────────────
def plot_plr_por_escenario(plr_con, plr_sin, out_path):
    """
    Packet Loss Ratio promedio entre ejecuciones, en %.
    La barra es la media de PLR_r y la barra de error es su desviación
    estándar poblacional.
    Zoom natural en la zona 0–1%, donde sí hay señal para comparar.
    """
    labels, vals_sin, vals_con, err_sin, err_con = [], [], [], [], []
    for cfg in ESCENARIOS:
        if cfg not in plr_con or cfg not in plr_sin:
            continue
        p_sin = plr_sin[cfg]
        p_con = plr_con[cfg]
        labels.append(_label_corto(cfg))
        vals_sin.append(p_sin[0] * 100.0)
        vals_con.append(p_con[0] * 100.0)
        err_sin.append(p_sin[1] * 100.0)
        err_con.append(p_con[1] * 100.0)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))

    barras_sin = ax.bar(x - w/2, vals_sin, w, yerr=err_sin, capsize=4,
                        label='Baseline', color=COLOR_SIN,
                        edgecolor='black', linewidth=0.6)
    barras_con = ax.bar(x + w/2, vals_con, w, yerr=err_con, capsize=4,
                        label='RiSeG', color=COLOR_CON,
                        edgecolor='black', linewidth=0.6)
    _aplicar_textura_fija(barras_sin, '/')   # Baseline siempre /
    _aplicar_textura_fija(barras_con, '-')   # RiSeG   siempre -
    ymax = 1
    for xi, val in zip(x - w/2, vals_sin):
        ax.text(xi, val + ymax * 0.03, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=COLOR_SIN)
    for xi, val in zip(x + w/2, vals_con):
        ax.text(xi, val + ymax * 0.03, f'{val:.2f}%',
                ha='center', va='bottom',fontsize=12,
                fontweight='bold', color=COLOR_CON)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Interval by scenario')
    ax.set_ylabel('Packet Loss Ratio (%)')

    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, format='png', dpi=300)
    plt.close()
    print(f"  Guardado: {out_path}")

# ─────────────────────────────────────────────
# FIGURA 3: Densidad de RSSI (histograma agregado)
# ─────────────────────────────────────────────
def plot_rssi_hist(hist_con, hist_sin, out_path):
    cfgs = [ESCENARIOS[0], ESCENARIOS[1], ESCENARIOS[2]]
    fig, axes = plt.subplots(
        1, len(cfgs),
        figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN),
        squeeze=False,
    )
    axes = axes.ravel()

    # Valores exactos de sensibilidad (SX1276, BW=125kHz)
    SF_SENSITIVITY = {
        'SF9': -129.0,
        'SF10': -132.0
    }

    for ax, cfg in zip(axes, cfgs):
        h_sin = hist_sin.get(cfg)
        h_con = hist_con.get(cfg)

        if (h_sin is None or h_sin.empty) and (h_con is None or h_con.empty):
            ax.set_visible(False)
            continue

        ancho = None
        if h_sin is not None and not h_sin.empty:
            ancho = float((h_sin["bin_high"] - h_sin["bin_low"]).mean())
        elif h_con is not None and not h_con.empty:
            ancho = float((h_con["bin_high"] - h_con["bin_low"]).mean())

        if h_sin is not None and not h_sin.empty:
            centros = (h_sin["bin_low"] + h_sin["bin_high"]) / 2
            barras_sin = ax.bar(
                centros, h_sin["densidad_mean"],
                width=ancho * 0.9 if ancho else 1.0,
                alpha=0.55, label="Baseline", color=COLOR_SIN,
                edgecolor="black", linewidth=0.4,
            )
            _aplicar_textura_fija(barras_sin, '/')   # Baseline siempre /

        if h_con is not None and not h_con.empty:
            centros = (h_con["bin_low"] + h_con["bin_high"]) / 2
            barras_con = ax.bar(
                centros, h_con["densidad_mean"],
                width=ancho * 0.9 if ancho else 1.0,
                alpha=0.55, label="RiSeG", color=COLOR_CON,
                edgecolor="black", linewidth=0.4,
            )
            _aplicar_textura_fija(barras_con, '-')

        # --- AGREGADO: Líneas verticales para SF9 y SF10 ---
        y_max = ax.get_ylim()[1]
        
        for sf_name, dbm_val in SF_SENSITIVITY.items():
            estilo_linea = ':' if sf_name == 'SF9' else '--'
            ax.axvline(x=dbm_val, color='#800000', linestyle=estilo_linea, linewidth=1.5, 
                       alpha=0.8, label=f'{sf_name}')
            
            # Posicionamos el texto un poco a la derecha de la línea para que no se encime
            ax.text(dbm_val + 0.5, y_max * 0.85, sf_name, color='red', 
                    rotation=90, va='top', ha='left', fontsize=9, 
                    fontweight='bold', alpha=0.8)

        ax.set_xlabel("RSSI (dBm)")
        ax.set_title(_label_corto(cfg))
        ax.grid(True, alpha=0.3)
        
        # Aseguramos que la leyenda no tape las líneas (la ponemos arriba a la izquierda o derecha)
        ax.legend(loc='upper left', fontsize=9)

    axes[0].set_ylabel("Density")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Guardado: {out_path}")

def plot_rssi_hist2(hist_con, hist_sin, out_path):

    cfgs = [ESCENARIOS[2]]
    fig, axes = plt.subplots(
        1, len(cfgs),
        figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN),
        squeeze=False,
    )
    axes = axes.ravel()

    # Valores exactos de sensibilidad (SX1276, BW=125kHz)
    SF_SENSITIVITY = {
        'SF9': -129.0,
        'SF10': -132.0
    }
    COLOR_VINO = '#800000'  # Tono vino tinto

    for ax, cfg in zip(axes, cfgs):
        h_sin = hist_sin.get(cfg)
        h_con = hist_con.get(cfg)

        if (h_sin is None or h_sin.empty) and (h_con is None or h_con.empty):
            ax.set_visible(False)
            continue

        ancho = None
        if h_sin is not None and not h_sin.empty:
            ancho = float((h_sin["bin_high"] - h_sin["bin_low"]).mean())
        elif h_con is not None and not h_con.empty:
            ancho = float((h_con["bin_high"] - h_con["bin_low"]).mean())

        n_barras_sin = 0
        if h_sin is not None and not h_sin.empty:
            centros = (h_sin["bin_low"] + h_sin["bin_high"]) / 2
            barras_sin = ax.bar(
                centros, h_sin["densidad_mean"],
                width=ancho * 0.9 if ancho else 1.0,
                alpha=0.55, label="Baseline", color=COLOR_SIN,
                edgecolor="black", linewidth=0.4,
            )
            _aplicar_textura_fija(barras_sin, '/')   # Baseline siempre /

        if h_con is not None and not h_con.empty:
            centros = (h_con["bin_low"] + h_con["bin_high"]) / 2
            barras_con = ax.bar(
                centros, h_con["densidad_mean"],
                width=ancho * 0.9 if ancho else 1.0,
                alpha=0.55, label="RiSeG", color=COLOR_CON,
                edgecolor="black", linewidth=0.4,
            )
            _aplicar_textura_fija(barras_con, '-')

        # --- AGREGADO: Líneas verticales para SF9 y SF10 en vino tinto ---
        y_max = ax.get_ylim()[1]
        
        for sf_name, dbm_val in SF_SENSITIVITY.items():
            estilo_linea = ':' if sf_name == 'SF9' else '--'
            ax.axvline(x=dbm_val, color=COLOR_VINO, linestyle=estilo_linea, linewidth=1.8, 
                       alpha=0.9, label=f'{sf_name} ({dbm_val} dBm)')
            

        ax.set_xlabel("RSSI (dBm)")
        ax.set_title(_label_corto(cfg))
        ax.grid(True, alpha=0.3)
        
        # Ajustamos la leyenda para que no tape las líneas
        ax.legend(loc='upper left', fontsize=15)

    axes[0].set_ylabel("Density")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Guardado: {out_path}")

# ─────────────────────────────────────────────
# FIGURA 4: Distribución de SF final por nodo
# ─────────────────────────────────────────────

def plot_sf_hist(nodo_con, nodo_sin, out_path):
    SF_VALUES = [7, 8, 9, 10, 11, 12]
    
    # Definimos los bordes de los bins: [6.5, 7.5), [7.5, 8.5), ..., [11.5, 12.5)
    bins = np.arange(6.5, 12.5 + 1.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), sharey=True)

    x = np.arange(len(SF_VALUES))

    for ax, cfg in zip(axes, ESCENARIOS):
        df_sin = nodo_sin.get(cfg)
        df_con = nodo_con.get(cfg)
        if df_sin is None and df_con is None:
            ax.set_visible(False)
            continue

        def contar(df):
            if df is None or df.empty:
                return [0] * len(SF_VALUES)
            # Extraemos los valores reales (sin redondear)
            v = df['finalsf_mean'].dropna().to_numpy()
            if len(v) == 0:
                return [0] * len(SF_VALUES)
            # np.histogram cuenta cuántos caen en cada bin sin redondear el valor
            counts, _ = np.histogram(v, bins=bins)
            return counts.tolist()

        c_sin = contar(df_sin)
        c_con = contar(df_con)

        ax.plot(x, c_sin, '-o', label='Baseline', color=COLOR_SIN,
            linewidth=2, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)
        ax.plot(x, c_con, '-o', label='RiSeG', color=COLOR_CON,
            linewidth=2, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'SF{s}' for s in SF_VALUES])
        ax.set_xlabel('Final node SF')
        ax.set_title(_label_corto(cfg))
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    axes[0].set_ylabel('Number of nodes', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, format='png', dpi=300)
    plt.close()
    print(f"  Guardado: {out_path}")

# ─────────────────────────────────────────────
# RESUMEN CSV
# ─────────────────────────────────────────────
def escribir_resumen(tot_con, tot_sin, nodo_con, nodo_sin, out_path):
    rows = []
    for cfg in ESCENARIOS:
        if cfg not in tot_con and cfg not in tot_sin:
            continue
        tc = tot_con.get(cfg, {})
        ts = tot_sin.get(cfg, {})

        def g(d, k):
            v = d.get(k); return v[0] if v else np.nan

        def gs(d, k):
            v = d.get(k); return v[1] if v else np.nan

        def frac_sf_bajos(d_nodo):
            df = d_nodo.get(cfg)
            if df is None or df.empty:
                return np.nan
            sf = df['finalsf_mean'].round()
            return float(((sf >= 7) & (sf <= 9)).sum()) / len(sf)

        rows.append({
            'escenario':            cfg,
            'PDR_global_sin':       g(ts, 'PDR_global'),
            'PDR_global_con':       g(tc, 'PDR_global'),
            'PDR_global_sin_std':   gs(ts, 'PDR_global'),
            'PDR_global_con_std':   gs(tc, 'PDR_global'),
            'PLR_global_sin':       g(ts, 'PLR_global'),
            'PLR_global_con':       g(tc, 'PLR_global'),
            'PLR_global_sin_std':   gs(ts, 'PLR_global'),
            'PLR_global_con_std':   gs(tc, 'PLR_global'),
            'delay_ms_sin':         g(ts, 'mean_delay_ms'),
            'delay_ms_con':         g(tc, 'mean_delay_ms'),
            'delay_ms_sin_std':     gs(ts, 'mean_delay_ms'),
            'delay_ms_con_std':     gs(tc, 'mean_delay_ms'),
            'total_sent_sin':       g(ts, 'total_sent'),
            'total_sent_con':       g(tc, 'total_sent'),
            'total_dropped_DC_sin': g(ts, 'total_dropped_DC'),
            'total_dropped_DC_con': g(tc, 'total_dropped_DC'),
            'frac_SF7_9_sin':       frac_sf_bajos(nodo_sin),
            'frac_SF7_9_con':       frac_sf_bajos(nodo_con),
        })

    if not rows:
        print("  [SKIP] Resumen: sin datos")
        return
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Guardado: {out_path}")



def plot_mean_delay(tot_con, tot_sin, nodo_con, nodo_sin, out_path):
    """
    Mean Delay por escenario (panel izq) y SF promedio por nodo (panel der).

    La estabilidad del delay es CONSECUENCIA DIRECTA de la estabilidad
    del SF: ToA(SF) domina el delay, y con SF constante, delay constante.

    Requiere nodo_con/nodo_sin para calcular el SF promedio.
    """
    # ─── Panel izquierdo: Mean Delay ───
    labels, d_sin, d_con, e_sin, e_con = [], [], [], [], []
    sf_sin, sf_con = [], []
    for cfg in ESCENARIOS:
        if cfg not in tot_con or cfg not in tot_sin:
            continue
        a = tot_sin[cfg].get('mean_delay_ms')
        b = tot_con[cfg].get('mean_delay_ms')
        df_sin = nodo_sin.get(cfg)
        df_con = nodo_con.get(cfg)

        if a is None or b is None or df_sin is None or df_con is None:
            continue
        labels.append(_label_corto(cfg))
        d_sin.append(a[0]); e_sin.append(a[1])
        d_con.append(b[0]); e_con.append(b[1])
        v_sin = df_sin['finalsf_mean'].mean() if 'finalsf_mean' in df_sin else np.nan
        v_con = df_con['finalsf_mean'].mean() if 'finalsf_mean' in df_con else np.nan
        sf_sin.append(v_sin)
        sf_con.append(v_con)

    # ─── Figura ───
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))

    x = np.arange(len(labels))
    w = 0.38

    # ── Panel izquierdo ──
    ax1.errorbar(x - w/2, d_sin, yerr=e_sin, fmt='o', capsize=5,
                 linewidth=0, markersize=10,
                 color=COLOR_SIN, markeredgecolor='black',
                 markeredgewidth=0.6, label='Baseline')
    ax1.errorbar(x + w/2, d_con, yerr=e_con, fmt='o', capsize=5,
                 linewidth=0, markersize=10,
                 color=COLOR_CON, markeredgecolor='black',
                 markeredgewidth=0.6, label='RiSeG')

    # Línea guía horizontal (promedio global)
    all_d = [v for v in d_sin + d_con if pd.notna(v)]
    if all_d:
        y_mean = np.mean(all_d)
        ax1.axhline(y_mean, color='dimgray', linestyle='--',
                    linewidth=1, alpha=0.6,
                    label=f'Global mean = {y_mean:.1f} ms')

    for xi, v in zip(x - w/2, d_sin):
        if pd.notna(v):
            ax1.annotate(f'{v:.2f}', (xi, v), xytext=(0, 12),
                         textcoords='offset points', ha='center',
                         fontsize=9, fontweight='bold', color=COLOR_SIN)
    for xi, v in zip(x + w/2, d_con):
        if pd.notna(v):
            ax1.annotate(f'{v:.2f}', (xi, v), xytext=(0, -18),
                         textcoords='offset points', ha='center',
                         fontsize=9, fontweight='bold', color=COLOR_CON)

    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('Interval by scenario')
    ax1.set_ylabel('Mean Delay (ms)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend( loc='best')

    # ── Panel derecho ──
    barras_sin = ax2.bar(x - w/2, sf_sin, w, label='Baseline',
                         color=COLOR_SIN, edgecolor='black', linewidth=0.6)
    barras_con = ax2.bar(x + w/2, sf_con, w, label='RiSeG',
                         color=COLOR_CON, edgecolor='black', linewidth=0.6)
    _aplicar_textura_fija(barras_sin, '/')   # Baseline siempre /
    _aplicar_textura_fija(barras_con, '-')   # RiSeG   siempre -

    for xi, v in zip(x - w/2, sf_sin):
        if pd.notna(v):
            ax2.annotate(f'{v:.2f}', (xi, v), xytext=(0, 4),
                         textcoords='offset points', ha='center',
                         fontsize=9, fontweight='bold', color=COLOR_SIN)
    for xi, v in zip(x + w/2, sf_con):
        if pd.notna(v):
            ax2.annotate(f'{v:.2f}', (xi, v), xytext=(0, 4),
                         textcoords='offset points', ha='center',
                         fontsize=9, fontweight='bold', color=COLOR_CON)

    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel('Interval by scenario')
    ax2.set_ylabel('Mean final SF (average over nodes)')
    ax2.set_ylim(0, 14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, format='png', dpi=300)
    plt.close()
    print(f"  Guardado: {out_path}")


def plot_mean_toa(tot_con, tot_sin, out_path):
    """Compara mean ToA por escenario, con desviación estándar."""
    labels, toa_sin, toa_con, err_sin, err_con = [], [], [], [], []
    for cfg in ESCENARIOS:
        a = tot_sin.get(cfg, {}).get('mean_toa')
        b = tot_con.get(cfg, {}).get('mean_toa')
        if a is None and b is None:
            continue
        labels.append(_label_corto(cfg))
        toa_sin.append(a[0] if a else np.nan)
        err_sin.append(a[1] if a else 0.0)
        toa_con.append(b[0] if b else np.nan)
        err_con.append(b[1] if b else 0.0)

    if not labels:
        print('  [SKIP] Gráfica mean ToA: no hay métrica mean_toa en los CSV totales')
        return

    fig, ax1 = plt.subplots(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))
    x = np.arange(len(labels))
    w = 0.38

    ax1.errorbar(
        x - w/2, toa_sin, yerr=err_sin, fmt='o', capsize=5,
        linewidth=0, markersize=10, color=COLOR_SIN,
        markeredgecolor='black', markeredgewidth=0.6, label='Baseline',
    )
    ax1.errorbar(
        x + w/2, toa_con, yerr=err_con, fmt='o', capsize=5,
        linewidth=0, markersize=10, color=COLOR_CON,
        markeredgecolor='black', markeredgewidth=0.6, label='RiSeG',
    )
    for xi, value in zip(x - w/2, toa_sin):
        if pd.notna(value):
            ax1.annotate(
                f'{value:.2f}', (xi, value), xytext=(0, 12),
                textcoords='offset points', ha='center', fontsize=9,
                fontweight='bold', color=COLOR_SIN,
            )
    for xi, value in zip(x + w/2, toa_con):
        if pd.notna(value):
            ax1.annotate(
                f'{value:.2f}', (xi, value), xytext=(0, -18),
                textcoords='offset points', ha='center', fontsize=9,
                fontweight='bold', color=COLOR_CON,
            )
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('Interval by scenario')
    ax1.set_ylabel('Mean ToA (ms)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='best')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, format='png', dpi=300)
    plt.close(fig)
    print(f"  Guardado: {out_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',   default='analisis_metricas_omnet/final')
    parser.add_argument('--sufijo_con', default='conRiesgo')
    parser.add_argument('--sufijo_sin', default='sinRiesgo')
    args = parser.parse_args()

    base     = args.base_dir
    nodo_dir = os.path.join(base, 'por_nodo')
    hist_dir = os.path.join(base, 'histogramas')
    out_root = os.path.join(base, 'resultados_comparacion')
    os.makedirs(out_root, exist_ok=True)

    nodo_con, nodo_sin = {}, {}
    tot_con,  tot_sin  = {}, {}
    plr_con, plr_sin = {}, {}
    hist_con, hist_sin = {}, {}
    agg_con, agg_sin = {}, {}

    for cfg in ESCENARIOS:
        for sufijo, d_nodo, d_tot, d_plr, d_hist, d_agg in [
            (args.sufijo_con, nodo_con, tot_con, plr_con, hist_con, agg_con),
            (args.sufijo_sin, nodo_sin, tot_sin, plr_sin, hist_sin, agg_sin),
        ]:
            nombre = f'{cfg}{sufijo}'
            p_nodo = os.path.join(nodo_dir, f'metricas_nodo_{nombre}.csv')
            p_tot  = os.path.join(nodo_dir, f'totales_{nombre}.csv')
            p_der_plr = os.path.join(
                nodo_dir, f'der_plr_por_run_{nombre}.csv'
            )
            p_hist = os.path.join(hist_dir, f'rssi_{nombre}.csv')
            p_agg  = os.path.join(nodo_dir, f'metricas_agregadas_{nombre}.csv')

            if os.path.exists(p_nodo):
                d_nodo[cfg] = load_nodo(p_nodo)
            else:
                print(f"  [MISS] {p_nodo}")
            if os.path.exists(p_tot):
                d_tot[cfg] = load_totales(p_tot)
            else:
                print(f"  [MISS] {p_tot}")
            if os.path.exists(p_der_plr):
                plr_runs = load_der_plr_por_run(p_der_plr)
                d_plr[cfg] = calcular_estadisticas_plr(plr_runs)
            else:
                print(f"  [MISS] {p_der_plr}")
            if os.path.exists(p_hist):
                d_hist[cfg] = load_hist(p_hist)
            else:
                print(f"  [MISS] {p_hist}")
            if os.path.exists(p_agg):
                d_agg[cfg] = load_agregadas(p_agg)
            else:
                print(f"  [MISS] {p_agg}")

    print("\nGenerando figuras...")
    plot_plr_por_escenario(plr_con, plr_sin,
                                        os.path.join(out_root, 'plr_global_escenarios.png'))

    plot_rssi_hist(hist_con, hist_sin,
                                     os.path.join(out_root, 'rssi_histogramatodos.png'))
    plot_rssi_hist2(hist_con, hist_sin,
                                     os.path.join(out_root, 'rssi_histograma.png'))
    plot_sf_hist(nodo_con, nodo_sin,
                                 os.path.join(out_root, 'sf_histograma_todos.png'))

    plot_mean_delay(tot_con, tot_sin, nodo_con, nodo_sin,
                          os.path.join(out_root, 'mean_delay_escenarios.png'))
    plot_mean_toa(tot_con, tot_sin,
                  os.path.join(out_root, 'mean_toa_escenarios.png'))
    escribir_resumen(tot_con, tot_sin, nodo_con, nodo_sin,
                     os.path.join(out_root, 'resumen_escenarios.csv'))
    gw_ini_con = 'analisis_metricas_omnet/final/gateways_posicionesRiesgo.ini'   # ← .ini, no .csv
    gw_ini_sin = 'analisis_metricas_omnet/final/gateways_posiciones.ini'         # ← .ini, no .csv
    plot_pdr_rssi_por_zona(nodo_con, nodo_sin,
                           gw_ini_con, gw_ini_sin,
                              os.path.join(out_root, 'pdr_rssi_por_zona.png'))

    print("\nListo.")


    # ─────────────────────────────────────────────
    # COMPROBACIÓN: PDR y PLR por escenario y sufijo
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PDR / PLR GLOBAL — TODOS LOS ESCENARIOS")
    print("=" * 70)
    print(f"{'Escenario':<18}{'Sufijo':<12}{'PDR (%)':>12}{'PLR (%)':>12}{'std (%)':>12}")
    print("-" * 70)

    for cfg in ESCENARIOS:
        for sufijo in ['sinRiesgo', 'conRiesgo']:
            p_tot = os.path.join(nodo_dir, f'totales_{cfg}{sufijo}.csv')
            if not os.path.exists(p_tot):
                print(f"{cfg:<18}{sufijo:<12}{'NO EXISTE':>36}")
                continue

            totales = load_totales(p_tot)
            pdr_tuple = totales.get('PDR_global')
            plr_tuple = plr_sin.get(cfg) if sufijo == 'sinRiesgo' else plr_con.get(cfg)
            if pdr_tuple is None or plr_tuple is None:
                print(f"{cfg:<18}{sufijo:<12}{'SIN PDR_global/PLR por ejecución':>36}")
                continue

            pdr_mean = pdr_tuple[0]
            plr_mean = plr_tuple[0] * 100
            plr_std  = plr_tuple[1] * 100

            print(f"{cfg:<18}{sufijo:<12}"
                  f"{pdr_mean * 100:>12.4f}"
                  f"{plr_mean:>12.4f}"
                  f"{plr_std:>12.4f}")

    print("=" * 70)
if __name__ == '__main__':
    main()
