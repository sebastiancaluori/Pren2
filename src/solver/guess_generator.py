# Update src/solver/guess_generator.py

from itertools import permutations, product
from typing import Sequence, Tuple, List
import numpy as np
import random


class GuessGenerator:
    """Generates candidate placements for puzzle pieces."""
    
    def __init__(self, rotation_step: int = 15):
        self.rotation_step = rotation_step
    
    def generate_grid_positions(self, 
                                target: np.ndarray, 
                                grid_spacing: int = 40,
                                include_edges: bool = True) -> List[Tuple[float, float]]:
        """
        Generate a grid of positions focused on the target area.
        
        Args:
            target: Target layout mask
            grid_spacing: Distance between grid points in pixels
            include_edges: Whether to include positions at target edges (for edge pieces)
            
        Returns:
            List of (x, y) positions to try
        """
        # Find target bounding box
        y_coords, x_coords = np.where(target > 0)
        
        if len(x_coords) == 0:
            # Fallback if no target
            return [(400.0, 400.0)]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Calculate target center
        target_center_x = (x_min + x_max) / 2
        target_center_y = (y_min + y_max) / 2
        
        # Reduce margin - stay closer to target
        margin = 50  # Reduced from 150
        x_min = max(0, x_min - margin)
        x_max = min(target.shape[1], x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(target.shape[0], y_max + margin)
        
        positions = []
        
        # Generate dense grid INSIDE and slightly outside target area
        for y in range(int(y_min), int(y_max), grid_spacing):
            for x in range(int(x_min), int(x_max), grid_spacing):
                positions.append((float(x), float(y)))
        
        # Add specific positions at target corners and center
        target_width = x_max - x_min
        target_height = y_max - y_min
        
        key_positions = [
            # Center
            (target_center_x, target_center_y),
            # Corners
            (x_min + margin, y_min + margin),
            (x_max - margin, y_min + margin),
            (x_min + margin, y_max - margin),
            (x_max - margin, y_max - margin),
            # Mid-edges
            (target_center_x, y_min + margin),
            (target_center_x, y_max - margin),
            (x_min + margin, target_center_y),
            (x_max - margin, target_center_y),
        ]
        
        positions.extend(key_positions)
        
        # Remove duplicates
        positions = list(set(positions))
        
        return positions
    
    def generate_guesses(self, 
                        num_pieces: int, 
                        target: np.ndarray,
                        max_guesses: int = 10000,
                        sample_positions: int = 8,
                        rotation_step: int = None) -> List[List[dict]]: # type: ignore
        """Generate smart guesses with configurable rotation step."""
        
        if rotation_step is None:
            rotation_step = self.rotation_step
        
        # Generate positions
        all_positions = self.generate_grid_positions(target, grid_spacing=50)
        
        rotations = list(range(0, 360, rotation_step))
        piece_ids = list(range(num_pieces))

        all_guesses = []
        
        # Strategy: Smart sampling
        # 1. For each piece arrangement
        # 2. For each rotation combo
        # 3. Try different position combinations, but SAMPLE intelligently
        
        # Limit piece permutations for speed
        max_permutations = min(24, len(list(permutations(piece_ids))))  # Max 24 permutations
        piece_permutations = list(permutations(piece_ids))
        random.shuffle(piece_permutations)
        piece_permutations = piece_permutations[:max_permutations]
        
        for piece_order in piece_permutations:
            # For each rotation combination
            for rotation_combo in product(rotations, repeat=num_pieces):
                # Sample positions intelligently
                # Each piece gets to try 'sample_positions' different locations
                sampled_positions = random.sample(all_positions, 
                                                 min(len(all_positions), sample_positions))
                
                # Try different combinations where pieces are at different positions
                for position_combo in product(sampled_positions, repeat=num_pieces):
                    guess = []
                    for piece_id, pos, theta in zip(piece_order, position_combo, rotation_combo):
                        guess.append({
                            'piece_id': piece_id,
                            'x': pos[0],
                            'y': pos[1],
                            'theta': theta
                        })
                    all_guesses.append(guess)
                    
                    # Limit total guesses
                    if len(all_guesses) >= max_guesses:
                        return all_guesses

        return all_guesses
    