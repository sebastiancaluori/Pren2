
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

from src.utils.geometry import rotate_and_crop

class MovementAnalyzer:
    """Calculate center of mass and movement data for puzzle pieces."""
    
    @staticmethod
    def calculate_piece_com(shape: np.ndarray, x: float, y: float, theta: float) -> Optional[Tuple[float, float]]:
        """
        Calculate center of mass for a piece at given position and rotation.
        
        Args:
            shape: Binary mask of the piece
            x, y: Top-left position of the piece
            theta: Rotation angle in degrees
            
        Returns:
            (com_x, com_y) in absolute coordinates, or None if invalid
        """
        if shape is None or np.sum(shape) == 0:
            return None
        
        # Rotate the shape
        rotated_shape = MovementAnalyzer._rotate_shape(shape, theta)
        
        # Find center of mass of rotated shape
        y_coords, x_coords = np.where(rotated_shape > 0)
        
        if len(x_coords) == 0:
            return None
        
        # COM relative to rotated shape's top-left
        com_x_rel = np.mean(x_coords)
        com_y_rel = np.mean(y_coords)
        
        # Add position offset to get absolute COM
        com_x = com_x_rel + x
        com_y = com_y_rel + y
        
        return (float(com_x), float(com_y))
    
    @staticmethod
    def _rotate_shape(shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees and crop to tight bounding box."""
        return rotate_and_crop(shape, angle)
    
    @staticmethod
    def analyze_best_solution_movements(
        puzzle_pieces: List, 
        best_guess: List[dict], 
        piece_shapes: Dict[int, np.ndarray],
        surfaces: dict
    ) -> Dict:
        """
        Analyze movements for the best solution with detailed robot movement calculations.
        
        Returns:
            {
                'source_coms': {piece_id: (x, y)},  # COM in source area
                'target_coms': {piece_id: (x, y)},  # COM in target area  
                'movements': {piece_id: {
                    'distance': float,      # Total distance to move
                    'rotation': float,      # Rotation change in degrees
                    'dx': float,           # X movement (+ = right, - = left)
                    'dy': float,           # Y movement (+ = down, - = up)
                    'x_mm': float,         # X movement in mm for robot
                    'y_mm': float,         # Y movement in mm for robot
                }}
            }
        """
        print("\n🔍 Analyzing robot movements for best solution...")
        
        source_coms = {}
        target_coms = {}
        movements = {}
        
        # Get surface offsets
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']
        target_offset_y = surfaces['target']['offset_y']
        
        # Scale factor (assuming 2 pixels per mm as mentioned in your code)
        pixels_per_mm = 2.0
        
        # Calculate source COMs (original positions)
        # pick_pose.x/y is already the centroid from parts.json — use it directly
        for piece in puzzle_pieces:
            piece_id = int(piece.id)

            if piece_id in piece_shapes:
                global_source_com = (
                    piece.pick_pose.x + source_offset_x,
                    piece.pick_pose.y + source_offset_y
                )
                source_coms[piece_id] = global_source_com
                print(f"  Source P{piece_id}: COM at {global_source_com}")
        
        # Calculate target COMs (best solution positions)
        for placement in best_guess:
            piece_id = placement['piece_id']
            
            if piece_id in piece_shapes:
                target_com = MovementAnalyzer.calculate_piece_com(
                    piece_shapes[piece_id],
                    placement['x'],
                    placement['y'],
                    placement['theta']
                )
                
                if target_com:
                    # Convert to global coordinates
                    global_target_com = (
                        target_com[0] + target_offset_x,
                        target_com[1] + target_offset_y
                    )
                    target_coms[piece_id] = global_target_com
                    print(f"  Target P{piece_id}: COM at {global_target_com}")
                    
                    # Calculate detailed movement if we have both source and target
                    if piece_id in source_coms:
                        source = source_coms[piece_id]
                        target = global_target_com
                        
                        # Calculate movement components
                        dx = target[0] - source[0]  # Positive = move right
                        dy = target[1] - source[1]  # Positive = move down
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Convert to robot coordinates (mm)
                        x_mm = dx / pixels_per_mm
                        y_mm = dy / pixels_per_mm
                        distance_mm = distance / pixels_per_mm
                        
                        # Calculate rotation change
                        original_piece = next(p for p in puzzle_pieces if int(p.id) == piece_id)
                        rotation_change = (placement['theta'] - original_piece.pick_pose.theta) % 360
                        if rotation_change > 180:
                            rotation_change -= 360  # Use shortest rotation
                        
                        movements[piece_id] = {
                            'distance': float(distance),      # Total distance in pixels
                            'rotation': float(rotation_change), # Rotation in degrees
                            'dx': float(dx),                   # X movement in pixels
                            'dy': float(dy),                   # Y movement in pixels
                            'x_mm': float(x_mm),              # X movement in mm
                            'y_mm': float(y_mm),              # Y movement in mm
                            'distance_mm': float(distance_mm)  # Total distance in mm
                        }
                        
                        # Determine movement directions
                        x_direction = "RIGHT" if dx > 0 else "LEFT" if dx < 0 else "NONE"
                        y_direction = "DOWN" if dy > 0 else "UP" if dy < 0 else "NONE"
                        rot_direction = "CLOCKWISE" if rotation_change > 0 else "COUNTER-CW" if rotation_change < 0 else "NONE"
                        
                        print(f"  Movement P{piece_id}:")
                        print(f"    Distance: {distance_mm:.1f}mm ({distance:.1f}px)")
                        print(f"    X: {x_mm:+.1f}mm ({x_direction})")
                        print(f"    Y: {y_mm:+.1f}mm ({y_direction})")
                        print(f"    Rotation: {rotation_change:+.0f}° ({rot_direction})")
        
        print(f"✅ Analyzed robot movements for {len(movements)} pieces")
        
        return {
            'source_coms': source_coms,
            'target_coms': target_coms,
            'movements': movements
        }


# Integration function for pipeline.py
def calculate_movement_data_for_visualizer(solution_data):
    """
    Call this in pipeline.py to pre-calculate movement data.
    Add the result to solver_data before launching UI.
    """
    if not solution_data.get('puzzle_pieces') or not solution_data.get('best_guess'):
        print("⚠️  Missing puzzle_pieces or best_guess - skipping movement analysis")
        return None
    
    movement_data = MovementAnalyzer.analyze_best_solution_movements(
        puzzle_pieces=solution_data['puzzle_pieces'],
        best_guess=solution_data['best_guess'],
        piece_shapes=solution_data['piece_shapes'],
        surfaces=solution_data['surfaces']
    )
    
    return movement_data