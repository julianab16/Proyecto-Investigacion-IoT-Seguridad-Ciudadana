import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main(archivo_csv):
    """
    Función principal que analiza métricas de LoRaWAN desde un archivo CSV
    
    Args:
        archivo_csv: Ruta al archivo CSV con los resultados de simulación
        
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    print(f"\n{'='*80}")
    print(f"ANALIZANDO: {archivo_csv}")
    print(f"{'='*80}\n")
    
    # Diccionario para almacenar resultados
    resultados = {
        'archivo': os.path.basename(archivo_csv)
    }
    
    # Cargar CSV
    df = pd.read_csv(
        archivo_csv,
        sep=",",
        engine="python",
        quotechar='"',
        skipinitialspace=True
    )

    # Ver todas las métricas disponibles
    #print(df['name'].dropna().unique())

    # Limpiar columnas numéricas
    numeric_cols = ["value", "mean", "count", "vectime", "vecvalue"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Funciones internas que usan df y archivo_csv
    def metric_mean(name):
        data = df[df["name"] == name]["mean"]
        data = data.dropna()
        return data.mean() if not data.empty else np.nan

    def metric_count(name):
        data = df[df["name"] == name]["count"]
        data = data.dropna()
        return data.sum() if not data.empty else 0

    def extract_metric(metric):
        values = []
        with open(archivo_csv, "r", encoding="utf-8") as f:
            for line in f:
                if metric in line:
                    parts = line.strip().split(",")
                    # última columna = vecvalue
                    vecvalue = parts[-1]

                    if vecvalue:
                        # quitar comillas y espacios
                        vecvalue = vecvalue.replace('"', '').strip()

                        for x in vecvalue.split():
                            try:
                                values.append(float(x))
                            except ValueError:
                                pass  # ignora basura si aparece

        return np.array(values)

    def metric_value(name):
        data = df[df["name"] == name]["value"]
        data = data.dropna()
        return data.mean() if not data.empty else np.nan

    def metric_series(name):
        data = df[df["name"] == name]["value"]
        return data.dropna()

    # Obtener prefijo para nombres de archivo
    base_name = os.path.splitext(os.path.basename(archivo_csv))[0]
    
    # RSSI
    rssi_mean = metric_mean("receivedRSSI")
    resultados['rssi_medio_dBm'] = rssi_mean
    print("RSSI medio:", rssi_mean, "dBm")

    # SNR
    snr = extract_metric("Vector of SNIR per node")
    snr_mean = snr.mean() if len(snr) > 0 else np.nan
    resultados['snr_medio_dB'] = snr_mean
    print("SNR medio:", snr_mean, "dB")

    # SF y TP (ADR)
    sf_mean = metric_value("finalSF")
    tp_mean = metric_value("finalTP")
    resultados['sf_promedio'] = sf_mean
    resultados['tp_promedio_dBm'] = tp_mean

    print("SF promedio:", sf_mean)
    print("TP promedio:", tp_mean, "dBm")

    # Path Loss (estimado)
    if not np.isnan(tp_mean) and not np.isnan(rssi_mean):
        path_loss = tp_mean - rssi_mean
    else:
        path_loss = np.nan
    resultados['path_loss_dB'] = path_loss

    print("Path Loss estimado:", path_loss, "dB")

    # PDR
    sent = metric_value("LoRa_AppPacketSent:count")
    received_gw = metric_value("LoRaGWRadioReceptionFinishedCorrect:count")
    received_ns = metric_value("LoRa_ServerPacketReceived:count")

    pdr_gw = (received_gw / sent * 100) if sent > 0 else np.nan
    pdr_ns = (received_ns / sent * 100) if sent > 0 else np.nan
    
    resultados['pdr_gateway_%'] = pdr_gw
    resultados['pdr_network_server_%'] = pdr_ns
    resultados['paquetes_enviados'] = sent
    resultados['paquetes_recibidos_gw'] = received_gw
    resultados['paquetes_recibidos_ns'] = received_ns

    print(f"PDR (Gateway): {pdr_gw:.2f}%")
    print(f"PDR (Network Server): {pdr_ns:.2f}%")

    # Duty Cycle (por nodo - correcto)
    # El Duty Cycle debe calcularse individualmente por nodo, no globalmente
    
    # Obtener transmisiones por cada nodo
    tx_per_node = df[df["module"].str.contains("loRaNodes", na=False) & 
                     (df["name"] == "LoRaTransmissionCreated:count")]["value"].dropna()
    
    # Tiempo on-air aproximado según SF (en segundos)
    # Basado en fórmula: ToA ≈ 2^SF / BW para BW=125kHz y payload típico
    toa_map = {7: 0.041, 8: 0.082, 9: 0.165, 10: 0.330, 11: 0.659, 12: 1.318}
    toa = toa_map.get(int(sf_mean), 0.5) if not np.isnan(sf_mean) else 0.5

    sim_time = df["vectime"].dropna().max()
    sim_time = sim_time if not np.isnan(sim_time) else 300  # valor por defecto de la simulación

    # Calcular duty cycle por nodo
    duty_cycles = (tx_per_node * toa / sim_time) * 100
    
    if len(duty_cycles) > 0:
        duty_cycle_mean = duty_cycles.mean()
        duty_cycle_min = duty_cycles.min()
        duty_cycle_max = duty_cycles.max()
        
        resultados['duty_cycle_promedio_%'] = duty_cycle_mean
        resultados['duty_cycle_min_%'] = duty_cycle_min
        resultados['duty_cycle_max_%'] = duty_cycle_max
        resultados['nodos_analizados'] = len(duty_cycles)
        
        print(f"Duty Cycle promedio por nodo: {duty_cycle_mean:.3f}% ({'✅ OK' if duty_cycle_mean <= 1.0 else '❌ EXCEDE LÍMITE'}')")
        print(f"  - Mínimo: {duty_cycle_min:.3f}%")
        print(f"  - Máximo: {duty_cycle_max:.3f}%")
        print(f"  - Nodos analizados: {len(duty_cycles)}")
    else:
        resultados['duty_cycle_promedio_%'] = np.nan
        resultados['duty_cycle_min_%'] = np.nan
        resultados['duty_cycle_max_%'] = np.nan
        resultados['nodos_analizados'] = 0
        print("Duty Cycle: No disponible (sin datos de transmisión)")

    # Throughput
    throughput = metric_value("throughput:last")
    resultados['throughput_bps'] = throughput
    print("Throughput:", throughput, "bps")

    bits = metric_value("packetReceived:sum(packetBytes)") * 8
    time = metric_value("simulated time")

    throughput_teorico = bits / time
    resultados['throughput_teorico_bps'] = throughput_teorico
    print("Throughput teorico:", throughput_teorico, "bps")

    # Latencia
    latency = extract_metric("queueingTime:vector")
    latency_mean = latency.mean() if len(latency) > 0 else np.nan
    resultados['latencia_promedio_s'] = latency_mean
    print("Latencia promedio:", latency_mean, "s")

    # Rango de Comunicación
    x = metric_series("positionX")
    y = metric_series("positionY")
    gx = df[df["name"] == "gatewayX"]["value"].dropna().iloc[0]
    gy = df[df["name"] == "gatewayY"]["value"].dropna().iloc[0]
    # Convertir a numpy arrays (IMPORTANTE)
    x = x.values
    y = y.values
    gx = float(gx)
    gy = float(gy)
    # Calcular distancias
    distance = np.sqrt((x - gx)**2 + (y - gy)**2)

    resultados['rango_promedio_m'] = distance.mean()
    resultados['rango_minimo_m'] = distance.min()
    resultados['rango_maximo_m'] = distance.max()
    
    #print(f"Distancias calculadas: {distance}")
    print(f"Rango promedio: {distance.mean():.2f} m")
    print(f"Rango mínimo: {distance.min():.2f} m")
    print(f"Rango máximo: {distance.max():.2f} m")

    # Re-transmisiones
    num_retry = metric_value("numRetry")
    num_sent = metric_value("numSent")

    avg_retx = num_retry / num_sent if num_sent > 0 else np.nan
    resultados['retransmisiones_promedio'] = avg_retx
    print("Re-transmisiones promedio:", avg_retx)

    # Histograma individual de RSSI
    plt.figure(figsize=(10, 6))
    rssi_data = extract_metric("Vector of RSSI per node")
    plt.hist(rssi_data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(rssi_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {rssi_data.mean():.2f} dBm')
    plt.axvline(np.median(rssi_data), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(rssi_data):.2f} dBm')
    plt.xlabel('RSSI (dBm)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(f'Distribución de RSSI - {base_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'histograma_rssi_{base_name}.png', dpi=300)
    plt.show()

    # Histograma individual de SNR
    plt.figure(figsize=(10, 6))
    snr_data = extract_metric("Vector of SNIR per node")
    plt.hist(snr_data, bins=30, color='crimson', edgecolor='black', alpha=0.7)
    plt.axvline(snr_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'Media: {snr_data.mean():.2f} dB')
    plt.axvline(np.median(snr_data), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(snr_data):.2f} dB')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(f'Distribución de SNR - {base_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'histograma_snr_{base_name}.png', dpi=300)
    plt.show()
    
    # Retornar el diccionario de resultados
    return resultados


def comparar_resultados(resultados1, resultados2, archivo_salida='comparacion_metricas.csv'):
    """
    Compara los resultados de dos análisis y guarda la comparación en un CSV
    
    Args:
        resultados1: Diccionario con métricas del primer análisis
        resultados2: Diccionario con métricas del segundo análisis
        archivo_salida: Nombre del archivo CSV de salida
    """
    print(f"\n{'='*80}")
    print("COMPARACIÓN DE RESULTADOS")
    print(f"{'='*80}\n")
    
    # Crear DataFrame de comparación
    df_comparacion = pd.DataFrame({
        'Métrica': [],
        resultados1['archivo']: [],
        resultados2['archivo']: [],
        'Diferencia': [],
        'Diferencia_%': []
    })
    
    # Lista de métricas a comparar (excluyendo 'archivo')
    metricas = [k for k in resultados1.keys() if k != 'archivo']
    
    filas = []
    for metrica in metricas:
        val1 = resultados1.get(metrica, np.nan)
        val2 = resultados2.get(metrica, np.nan)
        
        # Calcular diferencia
        if not np.isnan(val1) and not np.isnan(val2):
            diferencia = val2 - val1
            if val1 != 0:
                diferencia_pct = (diferencia / val1) * 100
            else:
                diferencia_pct = np.nan
        else:
            diferencia = np.nan
            diferencia_pct = np.nan
        
        filas.append({
            'Métrica': metrica,
            resultados1['archivo']: val1,
            resultados2['archivo']: val2,
            'Diferencia': diferencia,
            'Diferencia_%': diferencia_pct
        })
    
    df_comparacion = pd.DataFrame(filas)
    
    # Guardar en CSV
    df_comparacion.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f"✅ Comparación guardada en: {archivo_salida}\n")
    
    # Mostrar tabla comparativa en consola
    print("RESUMEN DE COMPARACIÓN:")
    print("-" * 120)
    
    # Formato de impresión
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A')
    
    print(df_comparacion.to_string(index=False))
    print("-" * 120)
    
    # Análisis de mejoras/empeoramientos
    print("\n📊 ANÁLISIS DE CAMBIOS SIGNIFICATIVOS:")
    print("-" * 80)
    
    metricas_mejora = [
        ('pdr_gateway_%', 'PDR Gateway', 'aumentó', True),
        ('pdr_network_server_%', 'PDR Network Server', 'aumentó', True),
        ('rssi_medio_dBm', 'RSSI medio', 'aumentó', True),
        ('snr_medio_dB', 'SNR medio', 'aumentó', True),
        ('duty_cycle_promedio_%', 'Duty Cycle', 'disminuyó', False),
        ('latencia_promedio_s', 'Latencia', 'disminuyó', False),
    ]
    
    for metrica, nombre, verbo, mejor_mayor in metricas_mejora:
        fila = df_comparacion[df_comparacion['Métrica'] == metrica]
        if not fila.empty:
            diff_pct = fila['Diferencia_%'].values[0]
            if not np.isnan(diff_pct):
                if abs(diff_pct) > 5:  # Solo cambios mayores a 5%
                    if (mejor_mayor and diff_pct > 0) or (not mejor_mayor and diff_pct < 0):
                        emoji = "✅"
                    else:
                        emoji = "⚠️"
                    print(f"{emoji} {nombre} {verbo} en {abs(diff_pct):.2f}%")
    
    print("-" * 80)


if __name__ == "__main__":
    # Ejecutar análisis para ambos archivos
    archivo1 = "data_base/results1.csv"
    archivo2 = "data_base/results2.csv"
    
    resultados1 = main(archivo1)
    
    # Verificar si existe el segundo archivo
    if os.path.exists(archivo2):
        resultados2 = main(archivo2)
        
        # Guardar resultados individuales
        df_resultados = pd.DataFrame([resultados1, resultados2])
        df_resultados.to_csv('metricas_lorawan.csv', index=False, encoding='utf-8')
        print(f"\n✅ Métricas individuales guardadas en: metricas_lorawan.csv")
        
        # Comparar resultados
        comparar_resultados(resultados1, resultados2)
    else:
        print(f"\n⚠️ Advertencia: No se encontró {archivo2}")
        print(f"   Solo se analizará {archivo1}")
        
        # Guardar solo el primer resultado
        df_resultados = pd.DataFrame([resultados1])
        df_resultados.to_csv('metricas_lorawan.csv', index=False, encoding='utf-8')
        print(f"\n✅ Métricas guardadas en: metricas_lorawan.csv")