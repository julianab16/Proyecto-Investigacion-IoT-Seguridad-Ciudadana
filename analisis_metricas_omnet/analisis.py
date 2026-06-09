import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv

def guardar_info_nodos_lorawan(archivo_salida, archivo_data):

    IMAGEN_MAPA = "images/mapa_calor_solo_cali.png"
    imagen_path = IMAGEN_MAPA

    CALI_MINX = 720414.01
    CALI_MAXX = 735029.27
    CALI_MINY = 860572.64
    CALI_MAXY = 879817.30

    minx = CALI_MINX
    maxx = CALI_MAXX
    miny = CALI_MINY
    maxy = CALI_MAXY

    # Cargar el archivo original para extraer nodos
    df = pd.read_csv(archivo_data, sep=',', engine='python', quotechar='"', skipinitialspace=True)
    imagen = Image.open(imagen_path)
    img_width, img_height = imagen.size

    # Filtrar filas de nodos
    nodos = df[df['module'].str.contains('loRaNodes', na=False)].copy()
    # Extraer node_id
    nodos['node_id'] = nodos['module'].str.extract(r'loRaNodes\[(\d+)\]')[0].astype(float)
    nodos = nodos.dropna(subset=['node_id'])
    nodos['node_id'] = nodos['node_id'].astype(int)
    nodos_ids = sorted(nodos['node_id'].unique().tolist())
    
    print(f" NODOS ENCONTRADOS: {len(nodos_ids)} nodos")
    print(f"   IDs: {nodos_ids[:10]}{'...' if len(nodos_ids) > 10 else ''}")

    matriz = df[df["name"].str.startswith("pkts_node", na=False)][["name", "value"]]

    # Separar en columnas nodo y gateway
    matriz["node"] = matriz["name"].str.extract(r"pkts_node(\d+)_gw")
    matriz["gw"] = matriz["name"].str.extract(r"_gw(\d+)")
    matriz["value"] = pd.to_numeric(matriz["value"], errors="coerce")
    matriz = matriz[["node", "gw", "value"]].sort_values(["node", "gw"])
    
    print(f"📡 MATRIZ pkts_node: {len(matriz)} registros")
    print(f"   Nodos únicos en matriz: {matriz['node'].nunique()}")
    print(f"   Gateways únicos: {matriz['gw'].nunique()}")

    # Extraer gateway_id por nodo 
    gateway_id_map = {}
    nodo_id_map = {}
    pkts_received_map = {}
    if not matriz.empty:
        grouped = matriz.dropna(subset=["node", "gw", "value"]).copy()
        if not grouped.empty:
            grouped["node"] = grouped["node"].astype(int)
            grouped["gw"] = grouped["gw"].astype(int)
            for node_id, group in grouped.groupby("node"):
                best_row = group.sort_values("value", ascending=False).iloc[0]
                gateway_id_map[node_id] = int(best_row["gw"])
                nodo_id_map[node_id] = int(best_row["node"])
                pkts_received_map[node_id] = best_row["value"]

    def per_node_value(metric, column='value'):
        sub = nodos[nodos['name'] == metric].copy()
        if sub.empty:
            return {}
        if sub[column].dtype == object:
            sub[column] = sub[column].astype(str).str.strip()
            sub[column] = sub[column].str.replace(r"\s+", "", regex=True)
            sub[column] = sub[column].str.replace(r"[^0-9+\-\.eE]+$", "", regex=True)

        sub[column] = pd.to_numeric(sub[column], errors='coerce')
        return sub.groupby('node_id')[column].mean().to_dict()

    def per_node_v(metric):
        sub = df[df["name"].str.startswith(metric, na=False)]
        if sub.empty:
            return {}
        values = pd.to_numeric(sub["value"], errors="coerce").dropna().tolist()
        if not values:
            return {}
        return {node_id: values[i] for i, node_id in enumerate(nodos_ids) if i < len(values)}

    def metric_to_nodes(metric):
        sub = df[df["name"].str.startswith(metric, na=False)]
        if sub.empty:
            return {}
        values = pd.to_numeric(sub["value"], errors="coerce").dropna().tolist()
        if not values:
            return {}
        return {node_id: values[i] for i, node_id in enumerate(nodos_ids) if i < len(values)}
    
    def metric_module(metric, name, column='value'):
        """
        Extrae valores SOLO si:
        - Module CONTIENE 'metric' (ej: 'LoRaWan.loRaGW')
        - Name ES EXACTO (ej: 'initialX')
        - El valor NO está vacío/nan
        
        Extrae el ID del módulo (ej: [0] de 'LoRaWan.loRaGW[0].mobility')
        Regresa diccionario {id_extraído: valor_promedio}
        """
        # Filtrar por AMBAS condiciones + valor no vacío
        sub = df[(df['module'].str.contains(metric, na=False, regex=False)) & 
                 (df['name'] == name) &
                 (df['type'] == 'param')].copy()
        
        # Extraer ID del módulo (ej: [0] de 'LoRaWan.loRaGW[0].mobility')
        sub['extracted_id'] = sub['module'].str.extract(r'\[(\d+)\]')[0]
        sub['extracted_id'] = pd.to_numeric(sub['extracted_id'], errors='coerce')
        
        
        # Limpiar y convertir valores - MEJORADO
        if sub[column].dtype == object:
            # Convertir a string
            sub[column] = sub[column].astype(str)
            # Remover espacios
            sub[column] = sub[column].str.strip()
            # Extraer SOLO números, signos y notación científica (elimina unidades como 'm', 'dBm', etc)
            sub[column] = sub[column].str.extract(r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', expand=False)
            
        sub[column] = pd.to_numeric(sub[column], errors='coerce')
        result = sub.groupby('extracted_id')[column].mean().to_dict()

        return result
    
    def pixeles_a_epsg3116(x_map, y_map):
        """Convierte diccionarios de píxeles a coordenadas EPSG3116"""
        x_conv = {}
        y_conv = {}
        for node_id in x_map:
            if node_id in y_map:
                x_pixel = x_map[node_id] / img_width
                y_pixel = y_map[node_id] / img_height
                
                x_norm = x_pixel
                y_norm = 1 - y_pixel
                
                x_conv[node_id] = (x_norm * (maxx - minx)) + minx
                y_conv[node_id] = (y_norm * (maxy - miny)) + miny
        return x_conv, y_conv
    
    # Métricas por nodo
    pos_x_map = per_node_value('positionX')
    pos_y_map = per_node_value('positionY')

    pos_x_map, pos_y_map = pixeles_a_epsg3116(pos_x_map, pos_y_map)

    sf_map = per_node_value('finalSF') #
    tp_map = per_node_value('finalTP') #
    pkts_sent_map = per_node_value('sentPackets') #
    retransmi_map = per_node_value('numRetry')
    transmission_map = per_node_value('LoRaTransmissionCreated:count')
    data_size_map = per_node_value('dataSize')
    energy = per_node_value('totalEnergyConsumed')

    # RSSI/SNR pueden venir como vectores globales
    rssi_map = metric_to_nodes("rssi_node")
    snr_map = metric_to_nodes("snir_node")

    # Latencia por nodo: Extraer de endToEndDelay_nodeXXX
    latency_data = df[df["name"].str.startswith("endToEndDelay_node", na=False)][["name", "value"]].copy()
    latency_map = {}
    if not latency_data.empty:
        # Extraer node_id de "endToEndDelay_nodeXXX"
        latency_data["node"] = latency_data["name"].str.extract(r"endToEndDelay_node(\d+)")[0]
        latency_data["value"] = pd.to_numeric(latency_data["value"], errors="coerce")
        latency_data = latency_data[["node", "value"]].sort_values(["node"])
        
        # Agrupar por nodo y obtener valores
        grouped = latency_data.dropna(subset=["node", "value"]).copy()
        if not grouped.empty:
            grouped["node"] = grouped["node"].astype(int)
            for node_id, group in grouped.groupby("node"):
                # Tomar el promedio de todos los valores para este nodo
                latency_map[node_id] = group["value"].mean()

    # Simulated time
    sim_time_series = df[df['name'] == 'simulated time']['value']
    sim_time = pd.to_numeric(sim_time_series, errors='coerce').dropna()
    sim_time = float(sim_time.iloc[0]) 

    # Throughput por nodo (bytes -> bps) usando la mejor métrica disponible
    bytes_metrics = [
        'packetReceived:sum(packetBytes)',
        'rxPkOk:sum(packetBytes)',
        'txPk:sum(packetBytes)',
        'packetSent:sum(packetBytes)'
    ]
    bytes_map = {}
    for metric in bytes_metrics:
        bytes_map = per_node_value(metric)
        if bytes_map:
            break
    
    # Llamadas separadas para debug
    gateway_x_raw = metric_module('LoRaWan.loRaGW', 'initialX')
    gateway_y_raw = metric_module('LoRaWan.loRaGW', 'initialY')
    
    
    # Convertir de píxeles a EPSG3116 si es necesario
    gateway_x_map, gateway_y_map = pixeles_a_epsg3116(gateway_x_raw, gateway_y_raw)

    # Guardar CSV
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id','x','y','gateway_id', 'gw_x','gw_y', 'distance_m','sf','tp','lambda','PDR','Energy_mJ','Latency_ms'])
        # Solo usar nodos que tienen información en los mapas de paquetes
        nodos_con_info = [n for n in nodos_ids if n in nodo_id_map]
        for node_id in nodos_con_info:
            x = pos_x_map.get(node_id, np.nan)
            y = pos_y_map.get(node_id, np.nan)

            # Obtener el gateway asignado a este nodo
            gw_id = gateway_id_map.get(node_id)
            gx_node = gateway_x_map.get(gw_id, np.nan)
            gy_node = gateway_y_map.get(gw_id, np.nan)
            gx = gx_node 
            gy = gy_node

            if not np.isnan(x) and not np.isnan(y) and not np.isnan(gx) and not np.isnan(gy):
                distance = float(np.sqrt((x - gx)**2 + (y - gy)**2))
            else:
                distance = ''

            sf_val = sf_map.get(node_id, np.nan)
            tp_val = tp_map.get(node_id, np.nan)
            rssi_val = rssi_map.get(node_id, np.nan)
            snr_val = snr_map.get(node_id, np.nan)

            if not np.isnan(tp_val) and not np.isnan(rssi_val):
                path_loss = tp_val - rssi_val
            else:
                path_loss = ''

            pkts_sent = pkts_sent_map.get(node_id, np.nan)
            pkts_received = pkts_received_map.get(node_id, '')
            #pdr = (pkts_received / pkts_sent * 100) if (pd.notna(pkts_sent) and pkts_sent > 0) else ''

            data_size = data_size_map.get(node_id, np.nan)
            if pd.notna(data_size) and pd.notna(pkts_sent) and sim_time > 0:
                throughput = (pkts_sent * data_size * 8) / sim_time
            else:
                bytes_val = bytes_map.get(node_id, np.nan)
                throughput = (bytes_val * 8 / sim_time) if pd.notna(bytes_val) and sim_time > 0 else ''

            transmission_count = transmission_map.get(node_id, np.nan)
            if pd.notna(sf_val):
                sf_int = int(sf_val)
                toa_map = {7: 0.041, 8: 0.082, 9: 0.165, 10: 0.330, 11: 0.659, 12: 1.318}
                toa = toa_map.get(sf_int, 0.5)
            else:
                toa = 0.5
            duty_cicle = (transmission_count * toa / sim_time) * 100 if pd.notna(transmission_count) and sim_time > 0 else ''

            retransmi = retransmi_map.get(node_id, '')
            latency_val = latency_map.get(node_id, '')
            
            # ===== CÁLCULOS SIMILARES A nsga.py (pero adaptados a datos del CSV) =====
            pdr = ''
            energy_mj = ''
            latency_ms = ''
            
            # Máximo energía teórico (SF12, TP27, λ=100)
            MAX_ENERGY_MJ = 51 * 0.001 * 2.0 * 2.0 * 100  # = 20.4 mJ
            
            # 1. PDR: Usar paquetes reales del CSV cuando esté disponible
            if pd.notna(pkts_sent) and pkts_sent > 0 and pkts_received != '':
                pdr = (pkts_received / pkts_sent)
            else:
                pdr = ''
            
            # 2. Energy_mJ
            energy_mj = energy.get(node_id, np.nan)

            # 3. Latency_ms: Usar del CSV cuando esté disponible, si no usar SF_latency como base
            if latency_val != '' and latency_val is not None:
                latency_ms = latency_val * 1000
            else:
                latency_ms = ''

            writer.writerow([
                nodo_id_map.get(node_id, ''),
                x,
                y,
                gateway_id_map.get(node_id, ''),
                gx,
                gy,
                distance,
                sf_val if pd.notna(sf_val) else '',
                tp_val if pd.notna(sf_val) else '',
                pkts_sent if pd.notna(pkts_sent) else '',
                round(pdr, 4) if isinstance(pdr, float) else pdr,
                round(energy_mj, 4) if isinstance(energy_mj, float) else energy_mj,
                round(latency_ms, 4) if isinstance(latency_ms, (float, int)) else latency_ms
            ])

if __name__ == "__main__":

    # Guardar solo la información de resultado1p
    guardar_info_nodos_lorawan('optimizacion_nodos/comparacion/nodos_lorawan_ga.csv', "data_base/resultsga.csv")

    