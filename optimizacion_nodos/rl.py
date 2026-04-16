"""
Reinforcement Learning (RL) for IoT Node Optimization
Objective: Maximize Reward = α·PDR - β·Energy - γ·Latency ± Insecurity_Index
Variables: 
  - Spatial: cell_id (hexagonal grid) OR (x, y) grid cells
  - Parameters: SF, TP, λ (packets_sent)

Output: Single optimal solution with best node position and parameters
Algorithm: Q-Learning with ε-greedy exploration
Integration: Uses hexagonal grid discretization with insecurity indices from georeferencia.py
"""

import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
from math import sqrt, log10
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import traceback

try:
    from hexagonal_grid import HexagonalGridManager
    HAS_HEX_GRID = True
except ImportError:
    HAS_HEX_GRID = False

# =====================================================================
# REINFORCEMENT LEARNING Q-LEARNING NODE OPTIMIZER
# =====================================================================

class RLNodeOptimizer:
    """
    Q-Learning agent for optimizing single IoT node configuration.
    
    State space (grid mode): (grid_cell_x, grid_cell_y, SF, TP, lambda_packets)
    State space (hexagon mode): (cell_id, SF, TP, lambda_packets)
    
    Action space (grid mode): Move up/down/left/right + change SF/TP/lambda
    Action space (hexagon mode): Move to adjacent cells + change SF/TP/lambda
    
    Reward (grid): α·PDR - β·Energy_norm - γ·Latency_norm
    Reward (hexagon): α·PDR - β·Energy_norm - γ·Latency_norm + δ·Insecurity_boost
    """
    
    # RL Hyperparameters
    LEARNING_RATE = 0.1  # α
    DISCOUNT_FACTOR = 0.95  # γ
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    
    # Node Parameters
    SF_VALUES = list(range(7, 13))  # [7, 8, 9, 10, 11, 12]
    TP_VALUES = [14, 20, 27]  # dBm
    LAMBDA_VALUES = list(range(1, 6))  # [1, 2, 3, 4, 5]
    
    # Grid discretization
    GRID_SIZE = 10  # 10x10 grid for spatial discretization
    
    # Reward normalization
    MAX_ENERGY = 1.02  # mJ
    MAX_LATENCY = 4300  # ms
    
    def __init__(self, 
                 node_data: Dict,
                 gateway_data: Dict,
                 reward_weights: Optional[Dict] = None,
                 original_sf: Optional[int] = None,
                 original_tp: Optional[int] = None,
                 original_lambda: Optional[int] = None,
                 hex_grid_manager: Optional['HexagonalGridManager'] = None):
        """
        Initialize RL optimizer for a single node.
        
        Args:
            node_data: {'id': int, 'x': float, 'y': float, 'gateway_id': int}
            gateway_data: {'id': int, 'x': float, 'y': float}
            grid_size: Grid dimensions (grid_size × grid_size cells) - used if hex_grid_manager is None
            alpha: Learning rate
            gamma: Discount factor
            reward_weights: Dict with 'pdr', 'energy', 'latency' weights
            original_sf: Original SF value from CSV (for reproducible initialization)
            original_tp: Original TP value from CSV (for reproducible initialization)
            original_lambda: Original lambda value from CSV (for reproducible initialization)
            hex_grid_manager: Optional HexagonalGridManager for using hexagonal discretization
        """
        self.node_data = node_data
        self.gateway_data = gateway_data

        
        # Hexagonal grid support
        self.hex_grid_manager = hex_grid_manager
        self.use_hex_grid = hex_grid_manager is not None
        
        # Default reward weights: Equal weights (1.0, 1.0, 1.0)
        self.reward_weights = reward_weights or {
            'pdr': 1.0,
            'energy': 1.0,
            'latency': 1.0
        }
        
        # Import FitnessCalculator ONCE at initialization
        spec = importlib.util.spec_from_file_location("ga_module", str(Path(__file__).parent / "ga.py"))
        ga_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ga_module)
        self.FitnessCalculator = ga_module.FitnessCalculator
        
        # Store original configuration for reproducible greedy search
        self.original_sf = original_sf if original_sf is not None else np.random.choice(self.SF_VALUES)
        self.original_tp = original_tp if original_tp is not None else np.random.choice(self.TP_VALUES)
        self.original_lambda = original_lambda if original_lambda is not None else np.random.choice(self.LAMBDA_VALUES)
        
        # Extract initial position
        self.initial_x = node_data['x']
        self.initial_y = node_data['y']
        self.gateway_x = gateway_data['x']
        self.gateway_y = gateway_data['y']
        
        # Initialize Q-table: Q[(state, action)] = value
        self.Q_table = defaultdict(float)
        
        # Tracking
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_values': []
        }
        
        # Current state for simulation
        self.current_state = None
        self.current_distance = None
        
        # If using hexagonal grid, get initial cell
        if self.use_hex_grid:
            self.initial_cell_id = hex_grid_manager.position_to_cell(self.initial_x, self.initial_y)
            if self.initial_cell_id is None:
                # Fallback: find closest cell center
                min_dist = float('inf')
                for cell_id in hex_grid_manager.get_all_cells():
                    cx, cy = hex_grid_manager.cell_center(cell_id)
                    dist = sqrt((cx - self.initial_x)**2 + (cy - self.initial_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        self.initial_cell_id = cell_id
        
    # =====================================================================
    # STATE SPACE DISCRETIZATION
    # =====================================================================
    
    def _discretize_position(self, x: float, y: float) -> Tuple[int, int]:
        """
        Discretize continuous position (x, y) to grid cell indices.
        
        Grid cells are centered around initial node position with ±2km range.
        Each cell represents ~400m.
        """
        # Define grid bounds: ±2km from initial position
        x_offset = 2000  # meters
        y_offset = 2000  # meters
        
        x_min = self.initial_x - x_offset
        x_max = self.initial_x + x_offset
        y_min = self.initial_y - y_offset
        y_max = self.initial_y + y_offset
        
        # Clamp to grid bounds
        x_clamped = np.clip(x, x_min, x_max)
        y_clamped = np.clip(y, y_min, y_max)
        
        # Map to grid indices [0, grid_size-1]
        grid_x = int((x_clamped - x_min) / (x_max - x_min) * (self.grid_size - 1))
        grid_y = int((y_clamped - y_min) / (y_max - y_min) * (self.grid_size - 1))
        
        return (grid_x, grid_y)
    
    def _undiscretize_position(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid cell indices back to continuous coordinates.
        """
        x_offset = 2000
        y_offset = 2000
        
        x_min = self.initial_x - x_offset
        x_max = self.initial_x + x_offset
        y_min = self.initial_y - y_offset
        y_max = self.initial_y + y_offset
        
        x = x_min + (grid_x / (self.grid_size - 1)) * (x_max - x_min)
        y = y_min + (grid_y / (self.grid_size - 1)) * (y_max - y_min)
        
        return (x, y)
    
    def _encode_state(self, *args) -> Tuple:
        """
        Encode state as hashable tuple.
        
        Grid mode: (grid_x, grid_y, sf, tp, lambda_packets)
        Hex mode: (cell_id, sf, tp, lambda_packets)
        """
        return tuple(args)
    
    def _decode_state(self, state: Tuple) -> Tuple:
        """Decode state tuple. Returns same as input (it's already a tuple)."""
        return state
    
    def _get_initial_state(self) -> Tuple:
        """Get initial state for training."""
        if self.use_hex_grid:
            # Hex grid: start from node's initial cell
            return self._encode_state(
                self.initial_cell_id,
                self.original_sf,
                self.original_tp,
                self.original_lambda
            )
        else:
            # Regular grid: discretize initial position
            grid_x, grid_y = self._discretize_position(self.initial_x, self.initial_y)
            return self._encode_state(
                grid_x, grid_y,
                self.original_sf,
                self.original_tp,
                self.original_lambda
            )
    
    def _state_to_position(self, state: Tuple) -> Tuple[float, float]:
        """Convert state to continuous coordinates."""
        if self.use_hex_grid:
            _, sf, tp, lambda_packets = state
            cell_id = state[0]
            if cell_id is not None:
                x, y = self.hex_grid_manager.cell_center(cell_id)
                return (x, y)
            else:
                return (self.initial_x, self.initial_y)
        else:
            grid_x, grid_y, sf, tp, lambda_packets = state
            return self._undiscretize_position(grid_x, grid_y)
    
    def _state_to_cell_id(self, state: Tuple) -> Optional[int]:
        """Extract cell_id from state if using hex_grid."""
        if self.use_hex_grid:
            return state[0]
        return None
    
    # =====================================================================
    # ACTION SPACE
    # =====================================================================
    
    def _get_possible_actions(self, state: Tuple) -> List[Tuple]:
        """
        Get all possible actions from current state.
        
        For grid mode:
        1. Spatial movement: up, down, left, right (4 actions)
        2. Parameter changes: change SF, TP, lambda (each has multiple options)
        
        For hexagonal grid mode:
        1. Spatial movement: move to adjacent cells (variable, depends on neighbors)
        2. Parameter changes: change SF, TP, lambda (each has multiple options)
        """
        if self.use_hex_grid:
            return self._get_hex_actions(state)
        else:
            return self._get_grid_actions(state)
    
    def _get_grid_actions(self, state: Tuple) -> List[Tuple]:
        """Get actions for regular grid mode."""
        grid_x, grid_y, sf, tp, lambda_packets = state
        actions = []
        
        # Spatial actions (4 directions)
        directions = [
            (grid_x + 1, grid_y, sf, tp, lambda_packets),  # right
            (grid_x - 1, grid_y, sf, tp, lambda_packets),  # left
            (grid_x, grid_y + 1, sf, tp, lambda_packets),  # up
            (grid_x, grid_y - 1, sf, tp, lambda_packets),  # down
        ]
        
        # Validate spatial actions (stay within bounds)
        for new_x, new_y, s_sf, s_tp, s_lambda in directions:
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                actions.append((new_x, new_y, s_sf, s_tp, s_lambda))
        
        # Parameter change actions
        for new_sf in self.SF_VALUES:
            if new_sf != sf:
                actions.append((grid_x, grid_y, new_sf, tp, lambda_packets))
        
        for new_tp in self.TP_VALUES:
            if new_tp != tp:
                actions.append((grid_x, grid_y, sf, new_tp, lambda_packets))
        
        for new_lambda in self.LAMBDA_VALUES:
            if new_lambda != lambda_packets:
                actions.append((grid_x, grid_y, sf, tp, new_lambda))
        
        return actions
    
    def _get_hex_actions(self, state: Tuple) -> List[Tuple]:
        """Get actions for hexagonal grid mode."""
        cell_id, sf, tp, lambda_packets = state
        actions = []
        
        # Spatial actions: move to adjacent cells (Queen neighborhood)
        neighbors = self.hex_grid_manager.get_neighbors(cell_id)
        for neighbor_cell_id in neighbors:
            actions.append((neighbor_cell_id, sf, tp, lambda_packets))
        
        # If no neighbors (shouldn't happen), stay in current cell
        if not neighbors:
            actions.append(state)
        
        # Parameter change actions
        for new_sf in self.SF_VALUES:
            if new_sf != sf:
                actions.append((cell_id, new_sf, tp, lambda_packets))
        
        for new_tp in self.TP_VALUES:
            if new_tp != tp:
                actions.append((cell_id, sf, new_tp, lambda_packets))
        
        for new_lambda in self.LAMBDA_VALUES:
            if new_lambda != lambda_packets:
                actions.append((cell_id, sf, tp, new_lambda))
        
        return actions
    
    # =====================================================================
    # REWARD COMPUTATION
    # =====================================================================
    
    def compute_reward(self, x: float, y: float, sf: int, tp: int, 
                      lambda_packets: int, cell_id: Optional[int] = None) -> float:
        """
        Compute reward for state (x, y, SF, TP, λ).
        
        Reward: α·PDR - β·Energy_norm - γ·Latency_norm
        
        where:
        - PDR ∈ [0, 1]
        - Energy normalized by MAX_ENERGY
        - Latency normalized by MAX_LATENCY
        """
        # Calculate distance to gateway
        distance = sqrt((x - self.gateway_x)**2 + (y - self.gateway_y)**2)
        
        # Get metrics (use pre-loaded FitnessCalculator)
        pdr = self.FitnessCalculator.calculate_pdr(sf, tp, distance, lambda_packets)
        energy = self.FitnessCalculator.calculate_energy(sf, tp, lambda_packets)
        latency = self.FitnessCalculator.calculate_latency(sf, lambda_packets, distance)
        
        # Normalize
        pdr_norm = np.clip(pdr, 0, 1)
        energy_norm = np.clip(energy / self.MAX_ENERGY, 0, 1)
        latency_norm = np.clip(latency / self.MAX_LATENCY, 0, 1)
        
        # Compute reward (standard formula without insecurity bonus)
        reward = (
            self.reward_weights['pdr'] * pdr_norm -
            self.reward_weights['energy'] * energy_norm -
            self.reward_weights['latency'] * latency_norm
        )
        
        return float(reward)
    
    # =====================================================================
    # Q-LEARNING UPDATE
    # =====================================================================
    
    def select_action(self, state: Tuple, epsilon: float) -> Tuple:
        """
        Select action using ε-greedy strategy.
        
        With probability ε: random action
        With probability 1-ε: best action (max Q-value)
        """
        if np.random.random() < epsilon:
            # Explore: random action
            possible_actions = self._get_possible_actions(state)
            return possible_actions[np.random.randint(len(possible_actions))]
        else:
            # Exploit: best action
            possible_actions = self._get_possible_actions(state)
            action_values = [self.Q_table[(state, action)] for action in possible_actions]
            best_idx = np.argmax(action_values)
            return possible_actions[best_idx]
    
    def update_q_value(self, state: Tuple, action: Tuple, reward: float, 
                      next_state: Tuple) -> None:
        """
        Update Q-value using Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]
        """
        # Get max Q-value for next state
        next_actions = self._get_possible_actions(next_state)
        next_q_values = [self.Q_table[(next_state, action)] for action in next_actions]
        max_next_q = np.max(next_q_values) if next_q_values else 0.0
        
        # Update Q-table
        old_q = self.Q_table[(state, action)]
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q_table[(state, action)] = new_q
    
    # =====================================================================
    # TRAINING
    # =====================================================================
    
    def train(self, episodes: int = 1000, max_steps: int = 100, 
              verbose: bool = True) -> Dict:
        """
        Train RL agent for multiple episodes.
        
        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress every 100 episodes
            
        Returns:
            Training history dict
        """
        epsilon = self.EPSILON_START
        
        for episode in range(episodes):
            # Initialize episode with proper state format
            state = self._get_initial_state()
            
            episode_reward = 0.0
            step = 0
            
            # Episode loop
            for step in range(max_steps):
                # Select and execute action
                action = self.select_action(state, epsilon)
                next_state = action  # Action directly becomes next state
                
                # Convert next_state to continuous coordinates (reward is consequence of action)
                x, y = self._state_to_position(next_state)
                cell_id = self._state_to_cell_id(next_state)
                
                # Extract parameters from state
                if self.use_hex_grid:
                    _, sf, tp, lambda_packets = next_state
                else:
                    _, _, sf, tp, lambda_packets = next_state
                
                # Calculate reward for reaching next_state
                reward = self.compute_reward(x, y, sf, tp, lambda_packets, cell_id=cell_id)
                episode_reward += reward
                
                # Update Q-table: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
            
            # Decay epsilon
            epsilon = max(self.EPSILON_MIN, epsilon * self.EPSILON_DECAY)
            
            # Track history
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_steps'].append(step + 1)
            self.training_history['epsilon_values'].append(epsilon)
            
            # Verbose output
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                print(f"Episode {episode+1}/{episodes} | "
                      f"Avg Reward (last 100): {avg_reward:.4f} | "
                      f"Epsilon: {epsilon:.4f}")
        
        return self.training_history
    
    # =====================================================================
    # INFERENCE & OPTIMIZATION
    # =====================================================================
    
    def get_best_configuration(self, num_steps: int = 50) -> Dict:
        """
        Get best configuration for node using trained agent (greedy).
        
        Runs agent in greedy mode (ε=0) for specified steps and returns
        best configuration found.
        
        Args:
            num_steps: Number of steps to run greedy search
            
        Returns:
            Dict with optimized parameters and metrics (keys match cols_order in optimize_all_nodes)
        """
        # Initialize from original configuration (for reproducibility)
        state = self._get_initial_state()
        
        best_reward = -np.inf
        best_config = None
        
        # Greedy search
        for step in range(num_steps):
            # Always select best action
            action = self.select_action(state, epsilon=0.0)
            
            # Get configuration
            x, y = self._state_to_position(action)
            cell_id = self._state_to_cell_id(action)
            
            if self.use_hex_grid:
                _, sf, tp, lambda_packets = action
            else:
                grid_x, grid_y, sf, tp, lambda_packets = action
            
            distance = sqrt((x - self.gateway_x)**2 + (y - self.gateway_y)**2)
            
            # Calculate metrics
            reward = self.compute_reward(x, y, sf, tp, lambda_packets, cell_id=cell_id)
            pdr = self.FitnessCalculator.calculate_pdr(sf, tp, distance, lambda_packets)
            energy = self.FitnessCalculator.calculate_energy(sf, tp, lambda_packets)
            latency = self.FitnessCalculator.calculate_latency(sf, lambda_packets, distance)
            
            if reward > best_reward:
                best_reward = reward
                
                # Build config with exact key names expected by optimize_all_nodes
                best_config = {
                    'x': x,
                    'y': y,
                    'SF': sf,                    # ← UPPERCASE for consistency
                    'TP': tp,                    # ← UPPERCASE for consistency
                    'lambda': lambda_packets,
                    'distance': distance,
                    'PDR': pdr,                  # ← UPPERCASE for consistency
                    'Energy_mJ': energy,         # ← With units suffix
                    'Latency_ms': latency,       # ← With units suffix
                    'Reward': reward,            # ← Exact name match
                }
            
            state = action
        
        return best_config
    
    def predict_reward_for_config(self, sf: int, tp: int, 
                                  lambda_packets: int) -> float:
        """
        Predict reward for a specific configuration using current position.
        """
        return self.compute_reward(self.initial_x, self.initial_y, sf, tp, 
                                   lambda_packets)
    
    # =====================================================================
    # PERSISTENCE
    # =====================================================================
    
    def save_model(self, filepath: str) -> None:
        """Save Q-table and training history to file."""
        model_data = {
            'Q_table': dict(self.Q_table),
            'training_history': self.training_history,
            'hyperparameters': {
                'learning_rate': self.alpha,
                'discount_factor': self.gamma,
                'grid_size': self.grid_size,
                'reward_weights': self.reward_weights
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load Q-table and training history from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.Q_table = defaultdict(float, model_data['Q_table'])
        self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded {len(self.Q_table)} Q-values")
    
    # =====================================================================
    # VISUALIZATIONS
    # =====================================================================
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            save_path: If provided, save plot to this path
        """
        if not self.training_history['episode_rewards']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'], alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Training: Cumulative Reward per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moving average
        window = 50
        moving_avg = np.convolve(self.training_history['episode_rewards'], 
                                 np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg, label=f'{window}-episode MA', color='red')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title(f'Training: {window}-Episode Moving Average')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[1, 0].plot(self.training_history['epsilon_values'], color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon (Exploration Rate)')
        axes[1, 0].set_title('ε-Greedy Exploration Decay')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Steps per episode
        axes[1, 1].plot(self.training_history['episode_steps'], alpha=0.5, color='orange')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].set_title('Steps per Episode')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
    
    def get_q_table_stats(self) -> Dict:
        """Get statistics about Q-table."""
        if not self.Q_table:
            return {'total_entries': 0, 'mean_q_value': 0, 'min_q_value': 0, 
                   'max_q_value': 0}
        
        q_values = list(self.Q_table.values())
        return {
            'total_entries': len(self.Q_table),
            'mean_q_value': float(np.mean(q_values)),
            'min_q_value': float(np.min(q_values)),
            'max_q_value': float(np.max(q_values)),
            'std_q_value': float(np.std(q_values))
        }


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_nodes(nodes_csv: str) -> pd.DataFrame:
    """Load node data from CSV file."""
    return pd.read_csv(nodes_csv)


def load_georeferencia_grid(geojson_path: Optional[str] = None) -> Optional['HexagonalGridManager']:
    """
    Load hexagonal grid from georeferencia GeoJSON export.
    
    Args:
        geojson_path: Path to grid_cali.geojson (if None, tries to find it automatically)
        
    Returns:
        HexagonalGridManager instance if successful, None otherwise
    """
    if not HAS_HEX_GRID:
        print("⚠ HexagonalGridManager not available, using regular grid mode")
        return None
    
    try:
        # Import geopandas for reading GeoJSON
        import geopandas as gpd
        
        # Try default path if not provided
        if geojson_path is None:
            project_root = Path(__file__).parent.parent
            geojson_path = project_root / "images" / "grid_cali.geojson"
        
        geojson_path = Path(geojson_path)
        
        if geojson_path.exists():
            print(f"\n📍 Loading hexagonal grid from: {geojson_path}")
            
            # Read GeoJSON as GeoDataFrame
            grid_gdf = gpd.read_file(geojson_path)
            print(f"   ✓ Loaded {len(grid_gdf)} hexagonal cells")
            
            # Create HexagonalGridManager from GeoDataFrame
            manager = HexagonalGridManager.from_georeferencia_grid(grid_gdf)
            return manager
        else:
            print(f"⚠ GeoJSON not found at {geojson_path}")
            print("   To use hexagonal grid optimization, export grid_cali.geojson from georeferencia.py")
            print("   Run: python export_georeferencia_grid.py")
            return None
            
    except Exception as e:
        print(f"⚠ Error loading hexagonal grid: {str(e)}")
        import traceback
        traceback.print_exc()
        print("   Continuing with regular grid mode")
        return None


def optimize_single_node(node_id: int, nodes_csv: str,
                        episodes: int = 1000, 
                        reward_weights: Optional[Dict] = None,
                        hex_grid_manager: Optional['HexagonalGridManager'] = None) -> Tuple[RLNodeOptimizer, Dict]:
    """
    Optimize configuration for a single node.
    
    Args:
        node_id: ID of node to optimize
        nodes_csv: Path to nodes CSV file (debe contener: node_id, pos_x, pos_y, gateway_id, gw_x, gw_y)
        episodes: Number of training episodes
        reward_weights: Reward function weights
        hex_grid_manager: Optional HexagonalGridManager for insecurity-aware optimization
        
    Returns:
        Tuple of (optimizer, optimized configuration dict)
    """
    # Load data FROM NODES CSV ONLY
    nodes = load_nodes(nodes_csv)
    
    # Get node row
    node_col = 'node_id' if 'node_id' in nodes.columns else 'id'
    node = nodes[nodes[node_col] == node_id].iloc[0]
    
    # Get position columns
    x_col = 'pos_x' if 'pos_x' in nodes.columns else 'x'
    y_col = 'pos_y' if 'pos_y' in nodes.columns else 'y'
    
    # Get gateway info FROM SAME NODE ROW
    node_data = {
        'id': int(float(node[node_col])),
        'x': float(node[x_col]),
        'y': float(node[y_col]),
        'gateway_id': int(float(node['gateway_id']))
    }
    
    # Gateway data comes from the same CSV row
    gateway_data = {
        'id': int(float(node['gateway_id'])),
        'x': float(node['gw_x']),
        'y': float(node['gw_y'])
    }
    
    # Extract original configuration from CSV for reproducible greedy search
    # Safe conversion: float → int, with NaN handling
    original_sf = int(float(node['sf'])) if 'sf' in node.index and pd.notna(node['sf']) else None
    original_tp = int(float(node['tp'])) if 'tp' in node.index and pd.notna(node['tp']) else None
    original_lambda = int(float(node['packets_sent'])) if 'packets_sent' in node.index and pd.notna(node['packets_sent']) else None
    
    # Create and train optimizer with optional hex grid
    optimizer = RLNodeOptimizer(node_data, gateway_data, 
                               reward_weights=reward_weights,
                               original_sf=original_sf,
                               original_tp=original_tp,
                               original_lambda=original_lambda,
                               hex_grid_manager=hex_grid_manager)
    
    mode = "hexagonal" if hex_grid_manager else "grid (10x10)"
    print(f"Training Node {node_id} [{mode}]...", end=" ", flush=True)
    optimizer.train(episodes=episodes, verbose=False)
    best_config = optimizer.get_best_configuration(num_steps=50)
    print(f"✓ Reward={best_config['Reward']:.4f}")
    
    return optimizer, best_config


def optimize_all_nodes(nodes_csv: str,
                       episodes: int = 500,
                       output_csv: Optional[str] = None,
                       reward_weights: Optional[Dict] = None,
                       max_nodes: Optional[int] = None,
                       geojson_path: Optional[str] = None) -> pd.DataFrame:
    """
    Optimize configuration for all nodes in CSV using georeferencia hexagonal grid.
    
    Args:
        nodes_csv: Path to nodes CSV file (contiene: node_id, pos_x, pos_y, gateway_id, gw_x, gw_y)
        episodes: Number of training episodes per node
        output_csv: Path to save results (default: results_optimized_nodes.csv)
        reward_weights: Reward function weights
        max_nodes: Maximum number of nodes to optimize (None = all)
        geojson_path: Path to grid_cali.geojson from georeferencia (if None, uses default)
        
    Returns:
        DataFrame with optimized configurations
    """
    print(f"\n{'='*80}")
    print(f"LOADING GEOREFERENCIA HEXAGONAL GRID FOR INSECURITY-AWARE OPTIMIZATION")
    print(f"{'='*80}")
    
    # Load hexagonal grid from georeferencia
    hex_grid_manager = load_georeferencia_grid(geojson_path)
    
    # Load node data only
    nodes = load_nodes(nodes_csv)
    
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get column name for node ID
    node_col = 'node_id' if 'node_id' in nodes.columns else 'id'
    
    # Get list of unique nodes
    node_ids = sorted(nodes[node_col].dropna().unique())
    if max_nodes:
        node_ids = node_ids[:max_nodes]
    
    # Optimize each node
    results = []
    total_nodes = len(node_ids)
    
    print(f"\n{'='*80}")
    print(f"BATCH OPTIMIZATION: {total_nodes} nodes")
    print(f"Episodes per node: {episodes}")
    print(f"Input CSV: {nodes_csv}")
    print(f"Output file: {output_csv}")
    if hex_grid_manager:
        print(f"Grid mode: HEXAGONAL (with insecurity indices)")
    else:
        print(f"Grid mode: REGULAR 10x10 (georeferencia grid not found)")
    print(f"{'='*80}\n")
    
    for idx, node_id in enumerate(node_ids, 1):
        try:
            optimizer, best_config = optimize_single_node(
                node_id=int(node_id),
                nodes_csv=nodes_csv,
                episodes=episodes,
                reward_weights=reward_weights,
                hex_grid_manager=hex_grid_manager            )
            
            # Add node_id to config
            best_config['node_id'] = int(node_id)
            results.append(best_config)
            
        except Exception as e:
            print(f"\n✗ Error optimizing node {node_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols_order = ['node_id', 'x', 'y', 'SF', 'TP', 'lambda', 'distance', 
                  'PDR', 'Energy_mJ', 'Latency_ms', 'Reward']
    cols_available = [c for c in cols_order if c in results_df.columns]
    results_df = results_df[cols_available]
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"✓ OPTIMIZATION COMPLETE!")
    print(f"Saved {len(results_df)} optimized nodes to: {output_csv}")
    if hex_grid_manager:
        print(f"Mode: Hexagonal grid with insecurity indices from georeferencia")
    print(f"{'='*80}\n")
    
    
    return results_df


if __name__ == "__main__":
    # Paths
    base_path = Path(__file__).parent.parent / "data_base"
    nodes_csv = base_path / "info_nodos_lorawan.csv"
    
    # Output path
    output_csv = str(Path(__file__).parent / "resultados" / "rl_optimized_nodes.csv")

    # Reward weights: PDR prioritized
    reward_weights = {
        'pdr': 1.0,
        'energy': 1.0,
        'latency': 1.0
    }
    
    # Path to hexagonal grid GeoJSON (generated by export_georeferencia_grid.py)
    geojson_path = Path(__file__).parent.parent / "images" / "grid_cali.geojson"
    
    # Optimize all nodes using hexagonal grid from city map
    results_df = optimize_all_nodes(
        nodes_csv=str(nodes_csv),
        episodes=500,
        output_csv=str(output_csv),
        reward_weights=reward_weights,
        geojson_path=str(geojson_path),  # ← Usar mapa hexagonal de Cali
        max_nodes=None  # Change to 10 for testing with 10 nodes
    )
    
    print("="*80)
