class Pose:
    """Represents a 2D pose (position + orientation) in pixels."""
    
    def __init__(self, x: float, y: float, theta: float):
        """
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            theta: Rotation angle in degrees
        """
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
    
    def __repr__(self) -> str:
        return f"Pose({self.x:.1f}px, {self.y:.1f}px, {self.theta:.1f}°)"