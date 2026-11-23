import pandas as pd
import numpy as np
import csv
import re
import os

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
ESCENARIOS = [ 'Alta','Baja','Media']
OUT_DIR = 'analisis_metricas_omnet/final'

NODE_SCALARS = {
    'sentPackets',   
    'finalSF',      
    'finalTP',
    'numSent',
    'numReceived',
    'numCollisions',
    'numRetry',
    'positionX',      
    'positionY',
}

GW_SCALARS = {
    'GW_forwardedDown',           # LoRaGWMac
    'GW_droppedDC',               # LoRaGWMac  
    'DER - Data Extraction Rate', # LoRaGWRadio
}

AGGREGATE_SCALARS = {
    'LoRa_NS_DER',                # PDR agregado 
    'totalReceivedPackets',
    'DER SF7', 'DER SF8', 'DER SF9', 'DER SF10', 'DER SF11', 'DER SF12',
    'counterUniqueReceivedPacketsPerSF SF7',
    'counterUniqueReceivedPacketsPerSF SF8',
    'counterUniqueReceivedPacketsPerSF SF9',
    'counterUniqueReceivedPacketsPerSF SF10',
    'counterUniqueReceivedPacketsPerSF SF11',
    'counterUniqueReceivedPacketsPerSF SF12',
}

# Histogramas a extraer
HISTOGRAMAS = {
    'receivedRSSI': 'receivedRSSI',   # nombre real en el .sca
}

# =====================================================================
# UTILIDADES
# =====================================================================

# =====================================================================
# TOPOLOGÍA (distancias + SF) — NUEVO
# =====================================================================
# Extrae del .sca (exportado a CSV):
#   meanDistToPickedGW_nodeN     (N = 0..9)
#   meanDist_nodeN_gwG           (N = 0..9, G = 0..9)
#   SFcount_nodeN_SFS            (N = 0..9, S = 7..12)
#   meanSF_nodeN                 (N = 0..9)
#   distanceGWtoServer_gwG       (G = 0..9)
# =====================================================================
MAX_NODES_TOP = 10
MAX_GWS_TOP   = 10
SF_RANGE      = range(7, 13)


def _stats_sobre_runs(df, grupo_cols, col='value'):
    """
    Promedia por (run, grupo_cols) y luego agrega mean sobre runs.
    Devuelve DataFrame con columnas grupo_cols + ['mean'].
    """
    if df.empty:
        return pd.DataFrame(columns=list(grupo_cols) + ['mean'])
    por_run = df.groupby(['run'] + list(grupo_cols))[col].mean().reset_index()
    stats = por_run.groupby(list(grupo_cols))[col].agg(['mean']).reset_index()
    return stats


def extraer_topologia(df, max_nodes=MAX_NODES_TOP, max_gws=MAX_GWS_TOP):
    """
    Devuelve dict con 5 listas de tuplas:
      dist_node_gw   : [(node_id, gw_id, mean), ...]
      sf_count       : [(node_id, sf, mean), ...]
      sf_mean        : [(node_id, mean), ...]
      dist_gw_server : [(gw_id, mean), ...]
    """
    out = {
        'dist_node_gw'   : [],
        'sf_count'       : [],
        'sf_mean'        : [],
        'dist_gw_server' : [],
    }

    # --- meanDist_nodeN_gwG ---
    sub = df[df['name'].str.startswith('meanDist_node', na=False)][['run','name','value']].dropna()
    if not sub.empty:
        ext = sub['name'].str.extract(r'meanDist_node(\d+)_gw(\d+)')
        sub['node_id'] = pd.to_numeric(ext[0], errors='coerce')
        sub['gw_id']   = pd.to_numeric(ext[1], errors='coerce')
        sub = sub.dropna(subset=['node_id','gw_id'])
        sub['node_id'] = sub['node_id'].astype(int)
        sub['gw_id']   = sub['gw_id'].astype(int)
        sub = sub[(sub['node_id'] < max_nodes) & (sub['gw_id'] < max_gws)]
        stats = _stats_sobre_runs(sub, ['node_id','gw_id'])
        for _, r in stats.iterrows():
            out['dist_node_gw'].append((int(r['node_id']), int(r['gw_id']),
                                        float(r['mean'])))


    # --- meanSF_nodeN ---
    sub = df[df['name'].str.startswith('meanSF_node', na=False)][['run','name','value']].dropna()
    if not sub.empty:
        sub['node_id'] = pd.to_numeric(
            sub['name'].str.extract(r'meanSF_node(\d+)')[0], errors='coerce')
        sub = sub.dropna(subset=['node_id'])
        sub['node_id'] = sub['node_id'].astype(int)
        sub = sub[sub['node_id'] < max_nodes]
        stats = _stats_sobre_runs(sub, ['node_id'])
        for _, r in stats.iterrows():
            out['sf_mean'].append((int(r['node_id']), float(r['mean'])))

    # --- distanceGWtoServer_gwG ---
    sub = df[df['name'].str.startswith('distanceGWtoServer_gw', na=False)][['run','name','value']].dropna()
    if not sub.empty:
        sub['gw_id'] = pd.to_numeric(
            sub['name'].str.extract(r'distanceGWtoServer_gw(\d+)')[0], errors='coerce')
        sub = sub.dropna(subset=['gw_id'])
        sub['gw_id'] = sub['gw_id'].astype(int)
        sub = sub[sub['gw_id'] < max_gws]
        stats = _stats_sobre_runs(sub, ['gw_id'])
        for _, r in stats.iterrows():
            out['dist_gw_server'].append((int(r['gw_id']), float(r['mean'])))

    return out


def guardar_topologia(nombre_cfg, df, carpeta_salida):
    """
    Guarda 'topologia_<cfg>.csv' en formato long:
      tipo,node_id,gw_id,valor

    Guarda las distancias nodo-gateway en un CSV independiente:
      distancia_nodo_gw_<cfg>.csv
    """
    topo = extraer_topologia(df)
    out_path = os.path.join(carpeta_salida, f'topologia_{nombre_cfg}.csv')
    out_dist_path = os.path.join(
        carpeta_salida, f'distancia_nodo_gw_{nombre_cfg}.csv'
    )

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tipo', 'node_id', 'gw_id', 'valor'])

        for nid, mean in topo['sf_mean']:
            w.writerow(['sf_asignado', nid, '', mean])
        for gid, mean in topo['dist_gw_server']:
            w.writerow(['dist_gw_server', '', gid, round(mean, 3)])

    with open(out_dist_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'gw_id', 'distancia nodo gw'])
        for nid, gid, mean in topo['dist_node_gw']:
            w.writerow([nid, gid, round(mean, 3)])

    total = len(topo['sf_mean']) + len(topo['dist_gw_server'])
    print(f"  → {out_path}  ({total} filas)")
    print(f"  → {out_dist_path}  ({len(topo['dist_node_gw'])} filas)")
    return topo


def toa(nombre_cfg, carpeta_salida, csv_in=None):
    """
    Extrae 'meanToA_nodeN' del .sca exportado y guarda:
      toa_nodo_<cfg>.csv con columnas:
        node_id, toa_ms_mean, toa_ms_std
    """
    if csv_in is None:
        csv_in = f'data_base/results_{nombre_cfg}.csv'

    df = cargar_csv(csv_in)

    sub = df[df['name'].str.startswith('meanToA_node', na=False)][['run','name','value']].dropna()

    sub['node_id'] = pd.to_numeric(
        sub['name'].str.extract(r'meanToA_node(\d+)')[0], errors='coerce')
    sub = sub.dropna(subset=['node_id'])
    sub['node_id'] = sub['node_id'].astype(int)
    sub = sub[sub['node_id'] < MAX_NODES_TOP]

    # meanToA ya viene en segundos → a ms
    sub['value_ms'] = sub['value'] * 1000.0

    stats = _stats_sobre_runs(sub, ['node_id'], col='value_ms')

    out_path = os.path.join(carpeta_salida, f'toa_nodo_{nombre_cfg}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'toa_ms_mean'])
        for _, r in stats.sort_values('node_id').iterrows():
            nid = int(r['node_id'])
            w.writerow([
                nid,
                round(float(r['mean']), 4),
            ])

    print(f"  → {out_path}  ({len(stats)} nodos)")

def delay(nombre_cfg, carpeta_salida, csv_in=None):
    """
    Extrae 'endToEndDelay_nodeN' del .sca exportado y guarda:
      delay_nodo_<cfg>.csv con columnas:
        node_id, delay_ms_mean
    """
    if csv_in is None:
        csv_in = f'data_base/results_{nombre_cfg}.csv'
    if not os.path.exists(csv_in):
        print(f"  [delay] No existe {csv_in}")
        return

    df = cargar_csv(csv_in)

    sub = df[df['name'].str.startswith('endToEndDelay_node', na=False)][['run','name','value']].dropna()

    sub['node_id'] = pd.to_numeric(
        sub['name'].str.extract(r'endToEndDelay_node(\d+)')[0], errors='coerce')
    sub = sub.dropna(subset=['node_id'])
    sub['node_id'] = sub['node_id'].astype(int)
    sub = sub[sub['node_id'] < MAX_NODES_TOP]

    # endToEndDelay ya viene en segundos → a ms
    sub['value_ms'] = sub['value'] * 1000.0

    stats = _stats_sobre_runs(sub, ['node_id'], col='value_ms')

    out_path = os.path.join(carpeta_salida, f'delay_nodo_{nombre_cfg}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'delay_ms_mean'])
        for _, r in stats.sort_values('node_id').iterrows():
            nid = int(r['node_id'])
            w.writerow([
                nid,
                round(float(r['mean']), 4),
            ])

    print(f"  → {out_path}  ({len(stats)} nodos)")

def _normalizar_run(df):
    """
    Extrae el número de run desde la columna 'run'.
    OMNeT++ 6.0 con opp_scavetool exporta run como entero (0,1,2...)
    o como string tipo 'config-run-0-...'. Manejo ambos.
    """
    if 'run' not in df.columns:
        df['run'] = 0
        return df
    s = df['run'].astype(str)
    # Caso entero puro
    m = s.str.fullmatch(r'\d+')
    if m.all():
        df['run'] = s.astype(int)
        return df
    # Caso string con guiones
    extracted = s.str.extract(r'-(\d+)-')[0]
    df['run'] = pd.to_numeric(extracted, errors='coerce').fillna(0).astype(int)
    return df


def _to_float(x):
    """Extrae el primer float de un string tipo '1.23 dBm' o '1.23e-4'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', str(x))
    return float(m.group(0)) if m else np.nan


def _promedio_sobre_runs(df, grupo, col='value'):
    """
    1) Promedia por (run, grupo) → 1 valor por run
    Devuelve dict[grupo] -> (mean)
    """
    if df.empty:
        return {}
    por_run = df.groupby(['run', grupo])[col].mean().reset_index()
    stats = por_run.groupby(grupo)[col].agg(['mean', 'std'])
    return {idx: (row['mean'], row['std'] if pd.notna(row['std']) else 0.0)
            for idx, row in stats.iterrows()}


# =====================================================================
# CARGA
# =====================================================================
def cargar_csv(path):
    df = pd.read_csv(path, sep=',', engine='python',
                     quotechar='"', skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    if 'value' in df.columns:
        df['value'] = df['value'].apply(_to_float)
    df = _normalizar_run(df)
    return df


# =====================================================================
# MÉTRICAS POR NODO (índice NED real)
# =====================================================================
def extraer_metricas_server_por_nodo(df):
    """
    Extrae scalars del NetworkServer con nombre dinámico:
      rssi_node<N>, snir_node<N>, pkts_node<N>_gw<G>
    ASUMEN que el patch macToNedIndex está aplicado (N = ID NED).
    Devuelve dict[node_id] -> dict[metric] = (mean, std)
    """
    out = {}

    # --- rssi_node<N> y snir_node<N> ---
    for pref, key in [('rssi_node', 'rssi_dBm'),
                      ('snir_node', 'snir_dB')]:
        sub = df[df['name'].str.startswith(pref, na=False)][['run', 'name', 'value']].copy()
        if sub.empty:
            continue
        sub['node_id'] = pd.to_numeric(
            sub['name'].str.extract(rf'{pref}(\d+)')[0], errors='coerce')
        sub = sub.dropna(subset=['node_id', 'value'])
        if sub.empty:
            continue
        sub['node_id'] = sub['node_id'].astype(int)
        stats = _promedio_sobre_runs(sub, 'node_id')
        for nid, (mean, std) in stats.items():
            out.setdefault(nid, {})[key] = (mean, std)

    # --- pkts_node<N>_gw<G>: sumar sobre GWs por (run, nodo) ---
    sub = df[df['name'].str.startswith('pkts_node', na=False)][['run', 'name', 'value']].copy()
    if not sub.empty:
        sub['node_id'] = pd.to_numeric(
            sub['name'].str.extract(r'pkts_node(\d+)_gw')[0], errors='coerce')
        sub = sub.dropna(subset=['node_id', 'value'])
        if not sub.empty:
            sub['node_id'] = sub['node_id'].astype(int)
            # sumar sobre GWs por (run, nodo), luego promediar sobre runs
            por_run = sub.groupby(['run', 'node_id'])['value'].sum().reset_index()
            stats = por_run.groupby('node_id')['value'].agg(['mean', 'std'])
            for nid, row in stats.iterrows():
                out.setdefault(nid, {})['packets_received'] = (
                    row['mean'], row['std'] if pd.notna(row['std']) else 0.0)

    return out

def extraer_metricas_por_nodo(df):
    """
    Solo scalars que viven en módulos loRaNodes[N] con N = índice NED.
    Devuelve dict[node_id] -> dict[metric] = (mean, std)
    """
    nodos = df[df['module'].str.contains('loRaNodes', na=False, regex=False)].copy()
    if nodos.empty:
        return {}
    nodos['node_id'] = pd.to_numeric(
        nodos['module'].str.extract(r'loRaNodes\[(\d+)\]')[0],
        errors='coerce'
    )
    nodos = nodos.dropna(subset=['node_id'])
    nodos['node_id'] = nodos['node_id'].astype(int)

    resultado = {}
    for metric in NODE_SCALARS:
        sub = nodos[nodos['name'].str.strip() == metric][['run', 'node_id', 'value']].copy()
        sub = sub.dropna(subset=['value'])
        if sub.empty:
            continue
        stats = _promedio_sobre_runs(sub, 'node_id')
        for nid, (mean, std) in stats.items():
            resultado.setdefault(nid, {})[metric] = (mean, std)
    return resultado

# =====================================================================
# MÉTRICAS POR GATEWAY
# =====================================================================
def extraer_metricas_por_gw(df):
    """
    Extrae scalars de módulos loRaGW[G].
    Devuelve dict[gw_id] -> dict[metric] = (mean, std)
    """
    gws = df[df['module'].str.contains('loRaGW', na=False, regex=False)].copy()
    if gws.empty:
        return {}
    gws['gw_id'] = pd.to_numeric(
        gws['module'].str.extract(r'loRaGW\[(\d+)\]')[0],
        errors='coerce'
    )
    gws = gws.dropna(subset=['gw_id'])
    gws['gw_id'] = gws['gw_id'].astype(int)

    resultado = {}
    for metric in GW_SCALARS:
        sub = gws[gws['name'].str.strip() == metric][['run', 'gw_id', 'value']].copy()
        sub = sub.dropna(subset=['value'])
        if sub.empty:
            continue
        stats = _promedio_sobre_runs(sub, 'gw_id')
        for gid, (mean, std) in stats.items():
            resultado.setdefault(gid, {})[metric] = (mean, std)
    return resultado


# =====================================================================
# MÉTRICAS AGREGADAS (server)
# =====================================================================
def extraer_metricas_agregadas(df):
    """
    Scalars únicos por run. Devuelve dict[metric] = (mean, std) sobre runs.
    """
    resultado = {}
    for metric in AGGREGATE_SCALARS:
        sub = df[df['name'].str.strip() == metric][['run', 'value']].copy()
        sub = sub.dropna(subset=['value'])
        if sub.empty:
            continue
        # Si hay múltiples filas por run (no debería), promediamos
        por_run = sub.groupby('run')['value'].mean()
        resultado[metric] = (por_run.mean(),
                             por_run.std() if len(por_run) > 1 else 0.0)
    return resultado


# =====================================================================
# ToA (Semtech SX1272)
# =====================================================================
def calcular_toa(sf, bw=125e3, cr=1, payload_bytes=20,
                 preamble_symbols=8, header_enabled=True,
                 crc=True, low_data_rate_optimize=None):
    """
    bw en Hz, cr = 1..4 (representa 4/5, 4/6, 4/7, 4/8).
    Fórmula oficial Semtech.
    """
    if low_data_rate_optimize is None:
        low_data_rate_optimize = 1 if (bw == 125e3 and sf >= 11) else 0
    t_sym = (2 ** sf) / bw
    t_preamble = (preamble_symbols + 4.25) * t_sym
    # payloadSymbNb
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) \
                - 20 * (0 if header_enabled else 1)
    denominator = 4 * (sf - 2 * low_data_rate_optimize)
    n_payload = 8 + max(np.ceil(numerator / denominator) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym


# =====================================================================
# HISTOGRAMAS
# =====================================================================
def extraer_histograma(archivo_csv, archivo_salida, nombre_hist='receivedRSSI'):
    """
    Extrae histograma desde CSV de opp_scavetool.
    - Convierte binvalues a float (no int).
    - Maneja N+1 edges vs N counts.
    - Normaliza a DENSIDAD por run antes de agregar.
    """
    if not os.path.exists(archivo_csv):
        print(f"  [hist] No existe: {archivo_csv}")
        return None

    filas = []
    n_hist = 0

    with open(archivo_csv, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None
        header = [h.strip().lower() for h in header]

        try:
            idx_run    = header.index('run')
            idx_type   = header.index('type')
            idx_name   = header.index('name')
            idx_edges  = header.index('binedges')
            idx_values = header.index('binvalues')
        except ValueError as e:
            print(f"  [hist] Faltan columnas: {e}")
            return None

        for row in reader:
            if len(row) <= max(idx_run, idx_type, idx_name, idx_edges, idx_values):
                continue
            if row[idx_type].strip() != 'histogram':
                continue
            if row[idx_name].strip() != nombre_hist:
                continue

            run_id = row[idx_run]
            m = re.search(r'-(\d+)-', run_id)
            run_num = int(m.group(1)) if m else (int(run_id) if run_id.isdigit() else n_hist)

            edges_str  = row[idx_edges].strip().strip('"')
            values_str = row[idx_values].strip().strip('"')

            edges  = [float(x) for x in edges_str.split()  if x]
            counts = [float(x) for x in values_str.split() if x]

            # Validación: N bins requieren N+1 edges
            if len(edges) == len(counts) + 1:
                pass
            elif len(edges) == len(counts):
                print(f"  [hist] Warning: {len(edges)} edges y {len(counts)} counts. "
                      f"Falta último edge en run {run_num}.")
                edges = edges + [edges[-1]]
            else:
                print(f"  [hist] SKIP mal formado: {len(edges)} edges / {len(counts)} counts")
                continue

            total = sum(counts) if sum(counts) > 0 else 1.0
            for i, c in enumerate(counts):
                ancho = edges[i + 1] - edges[i]
                if ancho <= 0:
                    continue
                densidad = c / (total * ancho)
                filas.append((run_num, edges[i], edges[i + 1], densidad, c))

            n_hist += 1

    if not filas:
        print(f"  [hist] No se encontró '{nombre_hist}' en {archivo_csv}")
        return None

    df = pd.DataFrame(filas, columns=['run', 'bin_low', 'bin_high', 'densidad', 'count'])
    agg = (df.groupby(['bin_low', 'bin_high'])
             .agg(densidad_mean=('densidad', 'mean'),
                  densidad_std=('densidad', 'std'),
                  count_mean=('count', 'mean'),
                  n_runs=('run', 'nunique'))
             .reset_index())
    agg['densidad_std'] = agg['densidad_std'].fillna(0.0)

    agg.to_csv(archivo_salida, index=False)
    print(f"  [hist] {nombre_hist}: {n_hist} runs, {len(agg)} bins → {archivo_salida}")
    return df


# =====================================================================
# GUARDADO
# =====================================================================
def guardar_metricas_escenario(nombre_cfg, archivo_data, carpeta_salida):
    print(f"\n=== Escenario: {nombre_cfg} ===")
    df = cargar_csv(archivo_data)
    print(f"  Filas: {len(df)} | Runs: {df['run'].nunique()}")

    # --- Métricas por nodo (NED index real) ---
    por_nodo = extraer_metricas_por_nodo(df)
    por_nodo_server = extraer_metricas_server_por_nodo(df)
    for nid, metrics in por_nodo_server.items():
        por_nodo.setdefault(nid, {}).update(metrics)

    out_nodo = os.path.join(carpeta_salida, f'metricas_nodo_{nombre_cfg}.csv')
    with open(out_nodo, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'node_id',
            'positionX', 'positionY',
            'packets_sent_mean', 'packets_sent_std',
            'packets_received_mean', 'packets_received_std',
            'rssi_dBm_mean', 'rssi_dBm_std',
            'snir_dB_mean', 'snir_dB_std',
            'finalSF_mean', 'finalSF_std',
            'finalTP_mean', 'finalTP_std',
            'mac_numSent_mean', 'mac_numSent_std',
            'mac_numReceived_mean', 'mac_numReceived_std',
            'mac_numCollisions_mean', 'mac_numCollisions_std',
            'mac_numRetry_mean', 'mac_numRetry_std',
            'toa_ms_from_finalSF',
        ])
        for nid in sorted(por_nodo.keys()):
            m = por_nodo[nid]
            def g(k):
                return m.get(k, (np.nan, np.nan))

            sf_mean, sf_std = g('finalSF')
            toa_ms = round(calcular_toa(int(round(sf_mean))) * 1000, 4) \
                    if pd.notna(sf_mean) else ''

            pos_x = g('positionX')[0]
            pos_y = g('positionY')[0]

            w.writerow([
                nid,
                round(pos_x, 3) if pd.notna(pos_x) else '',
                round(pos_y, 3) if pd.notna(pos_y) else '',
                round(g('sentPackets')[0], 3) if pd.notna(g('sentPackets')[0]) else '',
                round(g('sentPackets')[1], 3) if pd.notna(g('sentPackets')[1]) else '',
                round(g('packets_received')[0], 3) if pd.notna(g('packets_received')[0]) else '',
                round(g('packets_received')[1], 3) if pd.notna(g('packets_received')[1]) else '',
                round(g('rssi_dBm')[0], 3) if pd.notna(g('rssi_dBm')[0]) else '',
                round(g('rssi_dBm')[1], 3) if pd.notna(g('rssi_dBm')[1]) else '',
                round(g('snir_dB')[0], 3) if pd.notna(g('snir_dB')[0]) else '',
                round(g('snir_dB')[1], 3) if pd.notna(g('snir_dB')[1]) else '',
                round(sf_mean, 3) if pd.notna(sf_mean) else '',
                round(sf_std, 3) if pd.notna(sf_std) else '',
                round(g('finalTP')[0], 3) if pd.notna(g('finalTP')[0]) else '',
                round(g('finalTP')[1], 3) if pd.notna(g('finalTP')[1]) else '',
                round(g('numSent')[0], 3) if pd.notna(g('numSent')[0]) else '',
                round(g('numSent')[1], 3) if pd.notna(g('numSent')[1]) else '',
                round(g('numReceived')[0], 3) if pd.notna(g('numReceived')[0]) else '',
                round(g('numReceived')[1], 3) if pd.notna(g('numReceived')[1]) else '',
                round(g('numCollisions')[0], 3) if pd.notna(g('numCollisions')[0]) else '',
                round(g('numCollisions')[1], 3) if pd.notna(g('numCollisions')[1]) else '',
                round(g('numRetry')[0], 3) if pd.notna(g('numRetry')[0]) else '',
                round(g('numRetry')[1], 3) if pd.notna(g('numRetry')[1]) else '',
                toa_ms,
            ])
    print(f"  → {out_nodo}  ({len(por_nodo)} nodos)")

    # --- Métricas por gateway ---
    por_gw = extraer_metricas_por_gw(df)
    out_gw = os.path.join(carpeta_salida, f'metricas_gw_{nombre_cfg}.csv')
    with open(out_gw, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'gw_id',
            'GW_forwardedDown_mean', 'GW_forwardedDown_std',
            'GW_droppedDC_mean', 'GW_droppedDC_std',
            'DER_radio_mean', 'DER_radio_std',
        ])
        for gid in sorted(por_gw.keys()):
            m = por_gw[gid]
            def g(k):
                return m.get(k, (np.nan, np.nan))
            w.writerow([
                gid,
                round(g('GW_forwardedDown')[0], 3) if pd.notna(g('GW_forwardedDown')[0]) else '',
                round(g('GW_forwardedDown')[1], 3) if pd.notna(g('GW_forwardedDown')[1]) else '',
                round(g('GW_droppedDC')[0], 3) if pd.notna(g('GW_droppedDC')[0]) else '',
                round(g('GW_droppedDC')[1], 3) if pd.notna(g('GW_droppedDC')[1]) else '',
                round(g('DER - Data Extraction Rate')[0], 4) if pd.notna(g('DER - Data Extraction Rate')[0]) else '',
                round(g('DER - Data Extraction Rate')[1], 4) if pd.notna(g('DER - Data Extraction Rate')[1]) else '',
            ])
    print(f"  {out_gw}  ({len(por_gw)} gateways)")

    # --- Métricas agregadas (server) ---
    agg = extraer_metricas_agregadas(df)
    out_agg = os.path.join(carpeta_salida, f'metricas_agregadas_{nombre_cfg}.csv')
    with open(out_agg, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std'])
        # Orden fijo para legibilidad
        for metric in sorted(agg.keys()):
            mean, std = agg[metric]
            w.writerow([metric, round(mean, 6), round(std, 6)])
    print(f"  → {out_agg}  ({len(agg)} métricas)")

    # --- Totales agregados por escenario (QoS) ---
    totales = extraer_totales_escenario(df)
    der_plr_runs = calcular_der_plr_por_run(df)
    out_der_plr = os.path.join(
        carpeta_salida, f'der_plr_por_run_{nombre_cfg}.csv'
    )
    der_plr_runs.to_csv(out_der_plr, index=False, float_format='%.9f')
    print(f"  → {out_der_plr}  ({len(der_plr_runs)} ejecuciones)")

    out_tot = os.path.join(carpeta_salida, f'totales_{nombre_cfg}.csv')
    with open(out_tot, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std'])
        for k in sorted(totales.keys()):
            mean, std = totales[k]
            w.writerow([k, round(mean, 6), round(std, 6)])
    print(f"  → {out_tot}  ({len(totales)} métricas)")

    guardar_topologia(nombre_cfg, df, carpeta_salida)
    toa(nombre_cfg, carpeta_salida, csv_in=archivo_data)
    delay(nombre_cfg, carpeta_salida, csv_in=archivo_data)

    return {'por_nodo': por_nodo, 'por_gw': por_gw,
            'agregadas': agg, 'totales': totales}

# =====================================================================
# TOTALES AGREGADOS POR ESCENARIO (QoS)
# =====================================================================
def calcular_der_plr_por_run(df):
    """
    Usa directamente el DER calculado por OMNeT++ para cada ejecución.
    """
    sub = df[
        df['name'].str.strip() == 'LoRa_NS_DER'
    ][['run', 'value']].dropna()
    if sub.empty:
        return pd.DataFrame(columns=['run', 'DER', 'PLR'])

    resultado = (
        sub.groupby('run', as_index=False)['value']
        .mean()
        .rename(columns={'value': 'DER'})
    )
    resultado['PLR'] = 1.0 - resultado['DER']
    return resultado[['run', 'DER', 'PLR']]


def extraer_totales_escenario(df):
    """
    Agregados de red por escenario. Devuelve dict[metric] = (mean, std).
    """
    out = {}

    def suma_sobre(df_in, filtro_module, filtro_name):
        sub = df_in[
            df_in['module'].str.contains(filtro_module, na=False, regex=False)
            & (df_in['name'].str.strip() == filtro_name)
        ][['run', 'value']].dropna()
        if sub.empty:
            return None
        por_run = sub.groupby('run')['value'].sum()
        return (por_run.mean(),
                por_run.std() if len(por_run) > 1 else 0.0)

    def media_sobre(df_in, filtro_name):
        sub = df_in[df_in['name'].str.strip() == filtro_name][['run', 'value']].dropna()
        if sub.empty:
            return None
        por_run = sub.groupby('run')['value'].mean()
        return (por_run.mean(),
                por_run.std() if len(por_run) > 1 else 0.0)

    def estadisticas(serie):
        return (
            serie.mean(),
            serie.std() if len(serie) > 1 else 0.0,
        )

    out['total_sent']          = suma_sobre(df, 'loRaNodes', 'sentPackets')
    out['total_collisions']      = suma_sobre(df, 'loRaNodes', 'numCollisions')
    out['total_dropped_DC']    = suma_sobre(df, 'loRaGW',    'GW_droppedDC')
    out['total_received_uniq'] = media_sobre(df, 'totalReceivedPackets')
    out['DER_global']          = media_sobre(df, 'LoRa_NS_DER')
    out['PDR_global']          = out['DER_global']

    der_plr_runs = calcular_der_plr_por_run(df)
    if not der_plr_runs.empty:
        out['PLR_global'] = estadisticas(der_plr_runs['PLR'])

    # Mean Delay end-to-end (viene en SEGUNDOS → a ms)
    sub = df[df['name'].str.startswith('endToEndDelay_node', na=False)][['run', 'name', 'value']].dropna()
    if not sub.empty:
        por_run = sub.groupby('run')['value'].mean() * 1000.0

        out['mean_delay_ms'] = (
            por_run.mean(),
            por_run.std() if len(por_run) > 1 else 0.0
        )
        out['n_nodos_con_delay'] = (sub['name'].nunique(), 0.0)

    sub2 = df[df['name'].str.startswith('meanToA_node', na=False)][['run', 'name', 'value']].dropna()
    if not sub2.empty:
        por_ru = sub2.groupby('run')['value'].mean() * 1000.0

        out['mean_toa'] = (
            por_ru.mean(),
            por_ru.std() if len(por_ru) > 1 else 0.0
        )
        out['n_nodos_toa'] = (sub2['name'].nunique(), 0.0)

        nodos_sf = df[
        df['module'].str.contains('loRaNodes', na=False, regex=False) & 
        (df['name'].str.strip() == 'finalSF')
    ][['run', 'value']].dropna()

    if not nodos_sf.empty:
        # 2. Promediamos el SF de todos los nodos para cada run
        sf_por_run = nodos_sf.groupby('run')['value'].mean()
        # 3. Calculamos la media y desviación estándar entre todas las runs
        out['mean_final_SF'] = (
            sf_por_run.mean(),
            sf_por_run.std() if len(sf_por_run) > 1 else 0.0
        )

    return {k: v for k, v in out.items() if v is not None}

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    os.makedirs(f'{OUT_DIR}/por_nodo', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/histogramas', exist_ok=True)

    for cfg in ESCENARIOS:
        # Mismo nombre para conRiesgo / sinRiesgo
        for sufijo in ["sinRiesgo", "conRiesgo"]:
            nombre_archivo = f'{cfg}{sufijo}'
            csv_in = f'data_base/results_{nombre_archivo}.csv'

            guardar_metricas_escenario(
                nombre_archivo, csv_in, f'{OUT_DIR}/por_nodo'
            )

            # Histograma de RSSI
            extraer_histograma(
                csv_in,
                f'{OUT_DIR}/histogramas/rssi_{nombre_archivo}.csv',
                nombre_hist='receivedRSSI'
            )
