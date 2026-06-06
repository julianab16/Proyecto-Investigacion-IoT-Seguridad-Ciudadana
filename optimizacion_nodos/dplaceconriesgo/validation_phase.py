"""
validation_phase.py : Validation Phase of DPLACE model.

Step 5 (LoRaWAN Simulation):
  - Reinforce gateway positions using LoRaWAN environment characteristics
  - Include LoRaWAN parameters: CR (Coding Rate), SF (Spreading Factor), 
    channels, frequency
  - Simulate urban environment with building attenuation
  - Calculate packet loss due to interference and channel unavailability
  - Output: packets sent, packets received, packets lost (interference),
    packets dropped (no channels)

Step 6 (Metrics Calculation):
  - CAPEX: CBs (gateway cost) + Cins (installation) + Cset (setup) + Txinst (comm)
  - OPEX: Cman (maintenance) + Clease (leasing) + Celet (electricity) + CTrans (transmission)
  - Coverage constraint: minimum % of covered devices
  - Shannon capacity: minimum CBS per device
  - PDR: Packet Delivery Ratio
"""
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# LoRaWAN CONFIGURATION
# ─────────────────────────────────────────────

# LoRaWAN Spreading Factors and sensitivities
SF_SENSITIVITIES = {
    7: -123,   # dBm @ 125 kHz BW
    8: -126,
    9: -129,
    10: -132,
    11: -135,
    12: -137
}

# LoRaWAN typical parameters
LORAWAN_CONFIG = {
    'tx_power_dbm': 14,              # Gateway TX power
    'default_sf': 12,                # Default Spreading Factor
    'n_channels': 8,                 # Number of concurrent channels
    'frequency_ghz': 0.915,          # LoRaWAN frequency band
    'path_loss_exp_urban': 3.5,      # Path loss exponent in urban areas
    'path_loss_exp_los': 2.7,        # Line-of-sight exponent
    'building_attenuation_db': 10,   # Urban building attenuation
    'ref_distance_m': 1.0
}

# Cost parameters (in USD or local currency)
COST_PARAMS = {
    'cbs': 300,              # Gateway hardware cost
    'cins': 100,             # Installation cost per gateway
    'cset': 50,              # Setup cost per gateway
    'txinst_lte': 80,        # LTE communication installation
    'txinst_fiber': 120,     # Fiber communication installation
    'clease_monthly': 50,    # Monthly leasing (gateway location)
    'celet_monthly': 30,     # Monthly electricity per gateway
    'ctrans_monthly': 20,    # Monthly transmission cost per gateway
    'cman_ratio': 0.12       # Maintenance = 12% of CAPEX annually
}


def euclidean_distance(p1, p2):
    """2D Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_rssi(distance, sf=12, tx_power_dbm=14, urban=True, ref_dist=1.0):
    """
    Log-distance path loss RSSI model with urban attenuation.
    
    RSSI(d) = tx_power - 10 * n * log10(d / d0) - building_loss
    
    Parameters
    ----------
    distance : float
        Distance from gateway (meters)
    sf : int
        Spreading Factor (7-12)
    tx_power_dbm : float
        Transmission power in dBm
    urban : bool
        True = urban environment (higher path loss), False = LoS
    ref_dist : float
        Reference distance for path loss calculation
    
    Returns
    -------
    tuple : (rssi_dbm, above_sensitivity)
    """
    if distance <= 0:
        return tx_power_dbm, True
    
    n = LORAWAN_CONFIG['path_loss_exp_urban'] if urban else LORAWAN_CONFIG['path_loss_exp_los']
    building_loss = LORAWAN_CONFIG['building_attenuation_db'] if urban else 0
    
    rssi = tx_power_dbm - 10 * n * np.log10(distance / ref_dist) - building_loss
    sensitivity = SF_SENSITIVITIES.get(sf, -137)
    above_threshold = rssi >= sensitivity
    
    return rssi, above_threshold


def simulate_channel_access(n_devices_cluster, n_channels=8, collision_prob=0.15):
    """
    Simulate channel access and collisions in LoRaWAN.
    
    Parameters
    ----------
    n_devices_cluster : int
        Number of devices trying to access gateway
    n_channels : int
        Number of available channels
    collision_prob : float
        Base collision probability when channels are congested
    
    Returns
    -------
    dict : success_rate, collisions, dropped_packets
    """
    if n_devices_cluster == 0:
        return {'success_rate': 0, 'collisions': 0, 'dropped': 0}
    
    # Simple model: if devices > channels, collision risk increases
    congestion_ratio = n_devices_cluster / n_channels
    
    if congestion_ratio <= 1:
        # All devices fit in channels
        success_rate = 1.0 - (collision_prob * 0.1)
        collisions = int(n_devices_cluster * 0.02)  # minimal collisions
        dropped = 0
    elif congestion_ratio <= 2:
        success_rate = 1.0 - (collision_prob * 0.5)
        collisions = int(n_devices_cluster * 0.1)
        dropped = int(n_devices_cluster * 0.05)
    else:
        # Heavy congestion
        success_rate = 1.0 - (collision_prob * 1.0)
        collisions = int(n_devices_cluster * 0.2)
        dropped = int(n_devices_cluster * 0.15)
    
    return {
        'success_rate': max(0, success_rate),
        'collisions': collisions,
        'dropped': dropped
    }


def calculate_shannon_capacity(rssi_dbm, noise_dbm=-99, bandwidth_khz=125):
    """
    Calculate Shannon capacity for a link.
    C = B * log2(1 + S/N)
    
    Parameters
    ----------
    rssi_dbm : float
        Received signal strength
    noise_dbm : float
        Noise floor
    bandwidth_khz : float
        Channel bandwidth
    
    Returns
    -------
    float : capacity in bps
    """
    snr_linear = 10 ** ((rssi_dbm - noise_dbm) / 10)
    bandwidth_hz = bandwidth_khz * 1000
    capacity_bps = bandwidth_hz * np.log2(1 + snr_linear)
    return capacity_bps


def validation(
    X_devices,
    gateways,
    coverage_radius=3000,
    tx_power_dbm=14,
    sf=12,
    min_coverage_ratio=0.80,
    min_shannon_bps=100,
    communication_tech='lte',  # 'lte' or 'fiber'
    years_opex=1,
    verbose=True
):
    """
    Validation Phase: comprehensive LoRaWAN network performance evaluation.
    
    Implements Steps 5-6 from paper:
    - Step 5: LoRaWAN simulation with urban environment
    - Step 6: Cost and performance metrics (CAPEX, OPEX, PDR)

    Parameters
    ----------
    X_devices : ndarray (N, 2)
        IoT device positions (x, y) in meters
    gateways : ndarray (K, 2)
        Final gateway positions from processing phase
    coverage_radius : float
        Maximum coverage radius per gateway (meters)
    tx_power_dbm : float
        Gateway transmission power in dBm
    sf : int
        Default Spreading Factor (7-12)
    min_coverage_ratio : float
        Minimum required fraction of covered devices (Eq. 13)
    min_shannon_bps : float
        Minimum Shannon capacity per device (Eq. 14)
    communication_tech : str
        'lte' or 'fiber' for Txinst cost
    years_opex : float
        Number of years to calculate OPEX (t in Eq. 15)
    verbose : bool

    Returns
    -------
    dict with comprehensive metrics:
        Step 5 (LoRaWAN Simulation):
            'packets_sent'          : total packets sent by devices
            'packets_received'      : successfully received packets
            'packets_lost_interference' : packets lost to interference/collision
            'packets_dropped_channels'  : packets dropped (no channel available)
            'pdr'                   : Packet Delivery Ratio
            
        Step 6 (Cost & Performance):
            'coverage_ratio'        : fraction of covered devices (constraint 13)
            'coverage_constraint_met' : bool, coverage_ratio >= min_coverage_ratio
            'min_shannon_constraint' : bool, all devices meet min capacity (constraint 14)
            'capex_total'           : total CAPEX in currency units (Eq. 12)
            'opex_annual'           : annual OPEX cost (Eq. 15)
            
        Supporting Data:
            'device_df'             : per-device analysis
            'gateway_df'            : per-gateway statistics
            'gateways'              : gateway positions used
    """
    N = len(X_devices)
    K = len(gateways)
    
    if N == 0 or K == 0:
        return _create_empty_validation_result(gateways)
    
    # ─── STEP 5: LoRaWAN SIMULATION ───────────────────────────────────────
    
    device_records = []
    packets_sent = 0
    packets_received = 0
    packets_lost_interference = 0
    packets_dropped_channels = 0
    
    # Assign each device to best gateway
    gateway_clusters = {j: [] for j in range(K)}
    
    for i, device_pos in enumerate(X_devices):
        best_gw = 0
        best_rssi = -np.inf
        best_dist = np.inf
        best_above_threshold = False
        
        # Find best gateway (closest + above sensitivity threshold)
        for j, gw_pos in enumerate(gateways):
            dist = euclidean_distance(device_pos, gw_pos)
            rssi, above_thresh = compute_rssi(dist, sf=sf, tx_power_dbm=tx_power_dbm, urban=True)
            
            if dist < best_dist:
                best_dist = dist
                best_gw = j
                best_rssi = rssi
                best_above_threshold = above_thresh
        
        gateway_clusters[best_gw].append(i)
        
        # Coverage assessment (Constraint 13)
        covered = (best_dist <= coverage_radius) and best_above_threshold
        
        # Shannon capacity assessment (Constraint 14)
        shannon_cap = calculate_shannon_capacity(best_rssi)
        shannon_ok = shannon_cap >= min_shannon_bps
        
        # Per-device packet generation and reception
        packets_per_device = 100  # Simulate 100 packets per device
        packets_sent += packets_per_device
        
        if covered and best_above_threshold:
            # Device can transmit
            channel_sim = simulate_channel_access(
                len(gateway_clusters[best_gw]),
                n_channels=LORAWAN_CONFIG['n_channels']
            )
            received = int(packets_per_device * channel_sim['success_rate'])
            lost_interference = channel_sim['collisions']
            dropped = channel_sim['dropped']
            
            packets_received += received
            packets_lost_interference += lost_interference
            packets_dropped_channels += dropped
        else:
            # Device out of range or below sensitivity
            packets_lost_interference += packets_per_device
        
        device_records.append({
            'device_id': i,
            'x': device_pos[0],
            'y': device_pos[1],
            'best_gateway': best_gw,
            'distance_m': best_dist,
            'rssi_dbm': best_rssi,
            'covered': covered,
            'shannon_capacity_bps': shannon_cap,
            'shannon_ok': shannon_ok,
            'packets_sent': packets_per_device,
            'packets_received': int(received if covered else 0),
            'packets_lost': packets_per_device - (int(received if covered else 0))
        })
    
    device_df = pd.DataFrame(device_records)
    
    # Calculate PDR
    pdr = packets_received / packets_sent if packets_sent > 0 else 0
    n_covered = device_df['covered'].sum()
    coverage_ratio = n_covered / N
    coverage_constraint_met = coverage_ratio >= min_coverage_ratio
    shannon_constraint_met = device_df['shannon_ok'].sum() == N
    
    # ─── STEP 6: COST CALCULATIONS (Equations 12, 15) ─────────────────────
    
    # CAPEX (Equation 12)
    cbs = COST_PARAMS['cbs']
    cins = COST_PARAMS['cins']
    cset = COST_PARAMS['cset']
    txinst = COST_PARAMS['txinst_lte'] if communication_tech == 'lte' else COST_PARAMS['txinst_fiber']
    
    capex_per_gateway = cbs + cins + cset + txinst
    capex_total = K * capex_per_gateway
    
    # OPEX (Equation 15) - annually
    cman = COST_PARAMS['cman_ratio'] * capex_total  # 12% of CAPEX
    clease = COST_PARAMS['clease_monthly'] * 12 * K
    celet = COST_PARAMS['celet_monthly'] * 12 * K
    ctrans = COST_PARAMS['ctrans_monthly'] * 12 * K
    
    opex_annual = (cman + clease + celet + ctrans)
    opex_total = opex_annual * years_opex
    
    # ─── GATEWAY STATISTICS ────────────────────────────────────────────────
    
    gateway_records = []
    for j in range(K):
        devices_in_cluster = device_df[device_df['best_gateway'] == j]
        n_dev = len(devices_in_cluster)
        n_cov = devices_in_cluster['covered'].sum()
        avg_dist = devices_in_cluster['distance_m'].mean() if n_dev > 0 else 0
        avg_rssi = devices_in_cluster['rssi_dbm'].mean() if n_dev > 0 else 0
        
        gateway_records.append({
            'gateway_id': j,
            'x': gateways[j][0],
            'y': gateways[j][1],
            'n_devices': int(n_dev),
            'n_covered': int(n_cov),
            'avg_distance_m': avg_dist,
            'avg_rssi_dbm': avg_rssi,
            'capex_unit': capex_per_gateway,
            'opex_annual_share': opex_annual / K if K > 0 else 0
        })
    
    gateway_df = pd.DataFrame(gateway_records)
    
    if verbose:
        print(f"\n  ═══ VALIDATION PHASE RESULTS ═══")
        print(f"  [Step 5] LoRaWAN Simulation:")
        print(f"    • Devices: {N}, Gateways: {K}")
        print(f"    • Packets sent: {packets_sent} | Received: {packets_received}")
        print(f"    • Lost (interference): {packets_lost_interference} | Dropped (no channel): {packets_dropped_channels}")
        print(f"    • PDR: {pdr*100:.2f}%")
        print(f"\n  [Step 6] Coverage & Cost Metrics:")
        print(f"    • Coverage: {n_covered}/{N} ({coverage_ratio*100:.1f}%) [constraint: {min_coverage_ratio*100:.0f}%] {'✓' if coverage_constraint_met else '✗'}")
        print(f"    • Shannon capacity constraint: {'✓' if shannon_constraint_met else '✗'}")
        print(f"    • CAPEX: ${capex_total:.2f} ({capex_per_gateway} per gateway)")
        print(f"    • OPEX/year: ${opex_annual:.2f} (Cman: ${cman:.0f}, Lease: ${clease:.0f}, Electricity: ${celet:.0f}, Trans: ${ctrans:.0f})")
    
    return {
        # Step 5 outputs
        'packets_sent': packets_sent,
        'packets_received': packets_received,
        'packets_lost_interference': packets_lost_interference,
        'packets_dropped_channels': packets_dropped_channels,
        'pdr': pdr,
        
        # Step 6 outputs
        'coverage_ratio': coverage_ratio,
        'n_covered': int(n_covered),
        'n_uncovered': int(N - n_covered),
        'coverage_constraint_met': coverage_constraint_met,
        'shannon_constraint_met': shannon_constraint_met,
        'capex_total': capex_total,
        'capex_per_gateway': capex_per_gateway,
        'opex_annual': opex_annual,
        'opex_total': opex_total,
        
        # Supporting data
        'device_df': device_df,
        'gateway_df': gateway_df,
        'gateways': gateways,
    }


def _create_empty_validation_result(gateways):
    """Create empty validation result when N or K is 0."""
    return {
        'packets_sent': 0,
        'packets_received': 0,
        'packets_lost_interference': 0,
        'packets_dropped_channels': 0,
        'pdr': 0,
        'coverage_ratio': 0,
        'n_covered': 0,
        'n_uncovered': 0,
        'coverage_constraint_met': False,
        'shannon_constraint_met': False,
        'capex_total': 0,
        'capex_per_gateway': 0,
        'opex_annual': 0,
        'opex_total': 0,
        'device_df': pd.DataFrame(),
        'gateway_df': pd.DataFrame(),
        'gateways': gateways,
    }
