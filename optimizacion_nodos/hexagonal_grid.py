"""
Hexagonal Grid Manager for IoT Optimization
Integrates with georeferencia.py to use actual map cells and insecurity indices
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import math
from shapely.geometry import Point, Polygon, box
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HexagonalGridManager:
    """
    Manages hexagonal grid discretization using actual city map geometry.
    Integrates insecurity index and crime density for optimization.
    """
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 cell_size: float = 500,
                 cali_boundary: Optional[gpd.GeoDataFrame] = None):
        """
        Initialize hexagonal grid manager.
        
        Args:
            bbox: Bounding box (minx, maxx, miny, maxy) in EPSG:3116
            cell_size: Approximate side length of hexagon in meters
            cali_boundary: Optional GeoDataFrame with city boundary (EPSG:3116)
        """
        self.bbox = bbox
        self.cell_size = cell_size  # Side length of hexagon
        self.cali_boundary = cali_boundary
        
        # Grid storage
        self.grid_gdf = None  # GeoDataFrame with all hexagons
        self.cell_data = {}   # {cell_id: {'center': (x,y), 'insecurity': 0-100, 'density': int}}
        self.cell_adjacency = {}  # {cell_id: [neighbor_cell_ids]}
        
        # Initialize grid
        self._create_hexagonal_grid()
    
    def _create_hexagonal_grid(self):
        """Create hexagonal grid over the bounding box"""
        minx, maxx, miny, maxy = self.bbox
        
        # Hexagon geometry calculations
        hex_side = self.cell_size
        hex_width = math.sqrt(3) * hex_side
        hex_height = 2 * hex_side
        
        # Step sizes for grid generation
        step_x = hex_width
        step_y = hex_height * 0.75  # Vertical spacing for offset rows
        
        # Generate hexagon centers
        centers = []
        x_vals = np.arange(minx, maxx + step_x, step_x)
        y_vals = np.arange(miny, maxy + step_y, step_y)
        
        for ix, x in enumerate(x_vals):
            y_offset = (step_y / 2.0) if (ix % 2 == 1) else 0.0
            for y in y_vals:
                centers.append((x, y + y_offset))
        
        # Create hexagon geometries
        def hexagon_polygon(center_x: float, center_y: float, size: float) -> Polygon:
            """Create hexagon polygon from center and size"""
            angles = [i * math.pi / 3.0 for i in range(6)]
            coords = [(center_x + size * math.cos(a), 
                      center_y + size * math.sin(a)) for a in angles]
            return Polygon(coords)
        
        # Filter hexagons: only keep those within bounding box
        polygons = []
        valid_centers = []
        for cx, cy in centers:
            hex_poly = hexagon_polygon(cx, cy, hex_side)
            # Check if hexagon intersects with bbox
            bbox_polygon = box(minx, miny, maxx, maxy)
            if hex_poly.intersects(bbox_polygon):
                polygons.append(hex_poly)
                valid_centers.append((cx, cy))
        
        # Create GeoDataFrame
        self.grid_gdf = gpd.GeoDataFrame(
            {'cell_id': range(len(polygons))},
            geometry=polygons,
            crs='EPSG:3116'
        )
        
        # Initialize cell data
        for idx, (cx, cy) in enumerate(valid_centers):
            self.cell_data[idx] = {
                'center': (cx, cy),
                'density': 0,
                'num_eventos': 0
            }
        
        # Build adjacency graph
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency graph for cells (Queen neighborhood)"""
        if self.grid_gdf is None or len(self.grid_gdf) == 0:
            return
        
        # Initialize all cells with empty neighbor lists
        for idx in self.grid_gdf.index:
            self.cell_adjacency[idx] = []
        
        # Use spatial index for efficiency (STRtree if available, otherwise brute force)
        try:
            from shapely.strtree import STRtree
            tree = STRtree(self.grid_gdf.geometry)
            
            for idx, cell in self.grid_gdf.iterrows():
                cell_geom = cell.geometry
                # Find potential neighbors using spatial index
                potential_neighbors_idx = tree.query(cell_geom.buffer(1e-6), predicate='intersects')
                
                neighbors = []
                for other_idx in potential_neighbors_idx:
                    if idx != other_idx:
                        other_cell = self.grid_gdf.iloc[other_idx]
                        # Double check: Queen neighborhood (touching or intersecting)
                        if cell_geom.touches(other_cell.geometry) or \
                           (cell_geom.intersects(other_cell.geometry) and \
                            not cell_geom.equals(other_cell.geometry)):
                            neighbors.append(other_idx)
                
                self.cell_adjacency[idx] = neighbors
        except ImportError:
            # Fallback to brute force if STRtree not available
            for idx, cell in self.grid_gdf.iterrows():
                neighbors = []
                cell_geom = cell.geometry
                
                for other_idx, other_cell in self.grid_gdf.iterrows():
                    if idx != other_idx:
                        # Queen neighborhood: touching or intersecting
                        if cell_geom.touches(other_cell.geometry) or \
                           (cell_geom.intersects(other_cell.geometry) and \
                            not cell_geom.equals(other_cell.geometry)):
                            neighbors.append(other_idx)
                
                self.cell_adjacency[idx] = neighbors
    
    def load_insecurity_data(self, insecurity_dict: Dict[int, float],
                            density_dict: Optional[Dict[int, int]] = None):
        """
        Load crime density into grid cells (insecurity data NOT stored).
        
        Args:
            insecurity_dict: Ignored (for compatibility)
            density_dict: Optional {cell_id: num_eventos}
        """
        # Only load density, ignore insecurity
        if density_dict:
            for cell_id, density in density_dict.items():
                if cell_id in self.cell_data:
                    self.cell_data[cell_id]['num_eventos'] = density
    
    def position_to_cell(self, x: float, y: float) -> Optional[int]:
        """
        Find which cell contains position (x, y).
        
        Args:
            x, y: Coordinates in EPSG:3116
            
        Returns:
            Cell ID (index) or None if position outside grid
        """
        if self.grid_gdf is None or len(self.grid_gdf) == 0:
            return None
        
        point = Point(x, y)
        
        # First try: find cell that contains the point
        for idx, cell in self.grid_gdf.iterrows():
            if point.within(cell.geometry):
                return idx
        
        # Second try: find cell that touches the point (on boundary)
        for idx, cell in self.grid_gdf.iterrows():
            if point.touches(cell.geometry):
                return idx
        
        return None
    
    def cell_center(self, cell_id: int) -> Optional[Tuple[float, float]]:
        """Get center coordinates of a cell"""
        if cell_id in self.cell_data:
            return self.cell_data[cell_id]['center']
        return None
    
    def get_neighbors(self, cell_id: int) -> List[int]:
        """Get neighboring cell IDs (Queen neighborhood)"""
        return self.cell_adjacency.get(cell_id, [])
    
    def get_insecurity(self, cell_id: int) -> float:
        """Get insecurity index for cell (always returns 0.0 - not stored)"""
        return 0.0
    
    def get_density(self, cell_id: int) -> int:
        """Get crime density (number of events) for cell"""
        if cell_id in self.cell_data:
            return self.cell_data[cell_id]['num_eventos']
        return 0
    
    def num_cells(self) -> int:
        """Total number of cells in grid"""
        return len(self.cell_data)
    
    def get_all_cells(self) -> List[int]:
        """Get list of all valid cell IDs"""
        return list(self.cell_data.keys())
    
    def valid_cell(self, cell_id: int) -> bool:
        """Check if cell ID is valid"""
        return cell_id in self.cell_data
    
    @staticmethod
    def from_georeferencia_grid(grid_gdf: gpd.GeoDataFrame,
                               bbox: Optional[Tuple] = None) -> 'HexagonalGridManager':
        """
        Create from existing GeoDataFrame generated by georeferencia.py
        
        Args:
            grid_gdf: GeoDataFrame from GeoreferenciaMapa.grid_cali
            bbox: Optional bounding box to extract
            
        Returns:
            HexagonalGridManager instance
        """
        # Create instance WITHOUT calling _create_hexagonal_grid (we'll use provided grid)
        bounds = grid_gdf.total_bounds
        manager = HexagonalGridManager.__new__(HexagonalGridManager)
        manager.bbox = tuple(bounds)
        manager.cell_size = 500  # Default, not used when loading from GeoDataFrame
        manager.cali_boundary = None
        
        # Initialize storage
        manager.grid_gdf = grid_gdf.copy()
        manager.cell_data = {}
        manager.cell_adjacency = {}
        
        # CRITICAL: Use DataFrame index as cell_id for consistency
        for idx, row in manager.grid_gdf.iterrows():
            # Use the DataFrame index as cell_id to maintain consistency
            cell_id = idx
            centroid = row.geometry.centroid
            
            manager.cell_data[cell_id] = {
                'center': (float(centroid.x), float(centroid.y)),
                'density': int(row.get('num_eventos', 0)),
                'num_eventos': int(row.get('num_eventos', 0))
            }
        
        # Build adjacency using consistent indexing
        manager._build_adjacency()
        
        return manager
    
    def export_to_geojson(self, filepath: str):
        """Export grid as GeoJSON for visualization (without insecurity data)"""
        if self.grid_gdf is not None:
            # Don't add insecurity data - it's not stored
            self.grid_gdf.to_file(filepath, driver='GeoJSON')
