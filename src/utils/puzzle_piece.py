"""
Extended PuzzlePiece that stores corner and edge analysis data.
Drop-in replacement for src.utils.puzzle_piece.PuzzlePiece
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.utils.pose import Pose


@dataclass
class CornerData:
    """Data about a detected corner on a puzzle piece."""

    position: Tuple[int, int]
    angle: float  # Actual angle in degrees
    quality: float  # 0-1 score
    edge_lengths: Tuple[float, float]  # Lengths of adjacent edges
    rotation_to_align: float  # Rotation needed to place corner at bottom-right


@dataclass
class EdgeData:
    """Data about a detected straight edge on a puzzle piece."""

    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    midpoint: Tuple[int, int]
    length: float
    straightness: float  # 0-1 score
    angle: float  # Current angle (0-360°)
    quality: float  # 0-1 score
    rotations_to_align: Dict[str, float]  # {'bottom': 0, 'right': -90, ...}


class PuzzlePiece:
    """
    Extended PuzzlePiece with geometric analysis data.

    Usage:
        piece = PuzzlePiece(pid="0", pick=Pose(x=100, y=200, theta=0))

        # Analyzer populates these fields:
        piece.piece_type = "corner"
        piece.corners = [CornerData(...)]
        piece.edges = [EdgeData(...)]

        # Solver uses this data to make intelligent placement decisions
    """

    def __init__(self, pid: str, pick: Pose):
        self.id = pid
        self.pick_pose = pick
        self.place_pose: Optional[Pose] = None
        self.confidence = 0.0

        # === ANALYSIS DATA (populated by PieceAnalyzer) ===

        # Piece classification
        self.piece_type: str = "unknown"  # 'corner', 'edge', 'center', or 'unknown'
        self.analysis_confidence: float = 0.0  # How confident is the classification?

        # Corner detection results
        self.corners: List[CornerData] = []
        self.has_corner: bool = False
        self.primary_corner_rotation: Optional[float] = (
            None  # Best rotation to place corner at bottom-right
        )

        # Edge detection results
        self.edges: List[EdgeData] = []
        self.has_straight_edge: bool = False
        self.primary_edge_rotation: Optional[float] = (
            None  # Best rotation to place edge at bottom
        )
        self.edge_rotations: Dict[
            str, List[float]
        ] = {}  # {'bottom': [0, 45], 'right': [...], ...}

        # Piece geometry
        self.center: Optional[Tuple[float, float]] = None  # Piece centroid
        self.area: float = 0.0  # Piece area in pixels
        self.perimeter: float = 0.0  # Piece perimeter in pixels

    def __repr__(self) -> str:
        analysis_str = f", type={self.piece_type}"
        if self.corners:
            analysis_str += f", corners={len(self.corners)}"
        if self.edges:
            analysis_str += f", edges={len(self.edges)}"

        return (
            f"PuzzlePiece(id={self.id}, pick={self.pick_pose}, "
            f"place={self.place_pose}, conf={self.confidence:.2f}{analysis_str})"
        )

    def get_primary_rotation(self) -> Optional[float]:
        """
        Get the primary rotation for this piece based on its type.

        Returns:
            Rotation in degrees, or None if no clear rotation exists
        """
        if self.piece_type == "corner" and self.primary_corner_rotation is not None:
            return self.primary_corner_rotation
        elif self.piece_type == "edge" and self.primary_edge_rotation is not None:
            return self.primary_edge_rotation
        else:
            return None

    def get_rotations_for_edge(self, edge_direction: str) -> List[float]:
        """
        Get possible rotations to align this piece's edge to a specific direction.

        Args:
            edge_direction: 'bottom', 'right', 'top', or 'left'

        Returns:
            List of rotation angles in degrees
        """
        if edge_direction in self.edge_rotations:
            return self.edge_rotations[edge_direction]
        return []

    def summary(self) -> str:
        """Get a human-readable summary of the piece analysis."""
        lines = [f"Piece {self.id} ({self.piece_type.upper()})"]

        if self.corners:
            lines.append(f"  Corners: {len(self.corners)}")
            for i, corner in enumerate(self.corners):
                lines.append(
                    f"    Corner {i + 1}: quality={corner.quality:.3f}, "
                    f"angle={corner.angle:.1f}°, edges={corner.edge_lengths[0]:.1f}px/{corner.edge_lengths[1]:.1f}px"
                )
            if self.primary_corner_rotation is not None:
                lines.append(
                    f"  Primary corner rotation: {self.primary_corner_rotation:.1f}°"
                )

        if self.edges:
            lines.append(f"  Straight edges: {len(self.edges)}")
            for i, edge in enumerate(self.edges):
                lines.append(
                    f"    Edge {i + 1}: quality={edge.quality:.3f}, "
                    f"length={edge.length:.1f}px, angle={edge.angle:.1f}°"
                )
            if self.primary_edge_rotation is not None:
                lines.append(
                    f"  Primary edge rotation: {self.primary_edge_rotation:.1f}°"
                )

        if self.center:
            lines.append(f"  Center: ({self.center[0]:.1f}, {self.center[1]:.1f})")

        if self.area > 0:
            lines.append(f"  Area: {self.area:.1f}px²")

        return "\n".join(lines)
