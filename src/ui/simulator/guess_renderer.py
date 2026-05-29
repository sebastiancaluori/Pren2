# src/ui/simulator/guess_renderer.py

import numpy as np
import cv2
from typing import List, Dict, Optional

from src.utils.geometry import rotate_and_crop

class GuessRenderer:
    def __init__(self, width: int = 1000, height: int = 1000):
        self.width = width
        self.height = height
        
        # Define colors for different pieces (BGR format)
        self.piece_colors = [
            (255, 100, 100),  # Blue-ish
            (100, 255, 100),  # Green-ish
            (100, 100, 255),  # Red-ish
            (255, 255, 100),  # Cyan-ish
            (255, 100, 255),  # Magenta-ish
            (100, 255, 255),  # Yellow-ish
        ]
    def render(self, guess: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Render a guess onto a canvas.
        Returns grayscale for scoring.
        Now uses TOP-LEFT corner positioning instead of center.
        """
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        for placement in guess:
            piece_id = placement['piece_id']
            x = int(placement['x'])  # Now this is TOP-LEFT x
            y = int(placement['y'])  # Now this is TOP-LEFT y
            theta = placement['theta']

            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]

                # Rotate shape
                rotated = self._rotate_shape(shape, theta)

                # Place on canvas using top-left positioning
                self._place_shape(canvas, rotated, x, y, value=1)

        return canvas

    def render_static(self, placements: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """Render a fixed set of placements into a uint8 canvas for use as a static background."""
        return self.render(placements, piece_shapes)

    def render_on_base(self, base: np.ndarray, rotated_shape: np.ndarray, x: int, y: int) -> np.ndarray:
        """Stamp a pre-rotated shape onto a copy of base. Used for incremental scoring."""
        canvas = base.copy()
        self._place_shape(canvas, rotated_shape, x, y, value=1)
        return canvas

    def render_color(self, guess: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Render a guess with different colors for each piece (for visualization).
        Now uses TOP-LEFT corner positioning instead of center.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for placement in guess:
            piece_id = placement['piece_id']
            x = int(placement['x'])  # Now this is TOP-LEFT x
            y = int(placement['y'])  # Now this is TOP-LEFT y
            theta = placement['theta']
            
            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]
                
                # Rotate shape
                rotated = self._rotate_shape(shape, theta)
                
                # Get color for this piece
                color = self.piece_colors[piece_id % len(self.piece_colors)]
                
                # Place on canvas with color using top-left positioning
                self._place_shape_color(canvas, rotated, x, y, color)
        
        return canvas

    def _place_shape(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, value: float = 1.0):
        """Place shape on canvas using TOP-LEFT corner positioning."""
        h, w = shape.shape[:2]
        
        # Calculate bounds - x,y is now TOP-LEFT
        y1 = max(0, y)
        y2 = min(canvas.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(canvas.shape[1], x + w)
        
        # Calculate corresponding region in shape
        shape_y1 = max(0, -y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, -x)
        shape_x2 = shape_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            canvas[y1:y2, x1:x2] += shape[shape_y1:shape_y2, shape_x1:shape_x2] * value

    def _place_shape_color(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, color: tuple):
        """Place colored shape on canvas using TOP-LEFT corner positioning."""
        h, w = shape.shape[:2]
        
        # Calculate bounds - x,y is now TOP-LEFT
        y1 = max(0, y)
        y2 = min(canvas.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(canvas.shape[1], x + w)
        
        # Calculate corresponding region in shape
        shape_y1 = max(0, -y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, -x)
        shape_x2 = shape_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            shape_region = shape[shape_y1:shape_y2, shape_x1:shape_x2]
            mask = shape_region > 0
            
            for c in range(3):
                canvas[y1:y2, x1:x2, c][mask] = color[c]
             
    def render_debug(self, guess: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Render a guess with debug information: bounding boxes, centers, and coordinates.
        Now uses TOP-LEFT corner positioning.
        """
        # First render in color
        canvas = self.render_color(guess, piece_shapes)
        
        for placement in guess:
            piece_id = placement['piece_id']
            x = int(placement['x'])  # TOP-LEFT x
            y = int(placement['y'])  # TOP-LEFT y
            theta = placement['theta']
            
            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]
                
                # Rotate shape to see its actual bounds
                rotated = self._rotate_shape(shape, theta)
                h, w = rotated.shape[:2]
                
                # Bounding box is simply from (x,y) to (x+w, y+h)
                x1 = max(0, x)
                x2 = min(canvas.shape[1], x + w)
                y1 = max(0, y)
                y2 = min(canvas.shape[0], y + h)
                
                # Draw bounding box
                cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), (255, 255, 255), 2)
                
                # Draw TOP-LEFT anchor point
                cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1)  # Yellow at top-left
                cv2.putText(canvas, "TL", (x + 8, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Draw all four corners
                corners = [
                    (x, y, 'TL', (0, 255, 255)),           # Top-left (anchor) - Yellow
                    (x + w - 1, y, 'TR', (255, 0, 255)),   # Top-right - Magenta
                    (x, y + h - 1, 'BL', (255, 0, 255)),   # Bottom-left - Magenta
                    (x + w - 1, y + h - 1, 'BR', (0, 0, 255)),  # Bottom-right - Red
                ]
                
                for cx, cy, label, color in corners:
                    if 0 <= cx < canvas.shape[1] and 0 <= cy < canvas.shape[0]:
                        cv2.circle(canvas, (cx, cy), 4, color, -1)
                        cv2.putText(canvas, label, (cx + 5, cy - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Add text with piece info
                text = f"P{piece_id}: ({x},{y}) {theta:.0f}° [{w}x{h}]"
                cv2.putText(canvas, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw canvas border
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1]-1, canvas.shape[0]-1), (0, 255, 0), 3)
        
        # Add canvas size info
        cv2.putText(canvas, f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw target corners for reference
        cv2.circle(canvas, (0, 0), 6, (0, 255, 0), 2)  # TL
        cv2.circle(canvas, (canvas.shape[1]-1, 0), 6, (0, 255, 0), 2)  # TR
        cv2.circle(canvas, (0, canvas.shape[0]-1), 6, (0, 255, 0), 2)  # BL
        cv2.circle(canvas, (canvas.shape[1]-1, canvas.shape[0]-1), 6, (0, 255, 0), 2)  # BR
        
        return canvas
    def _rotate_shape(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees and crop to tight bounding box."""
        return rotate_and_crop(shape, angle)
    