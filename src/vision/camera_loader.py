"""
Loads puzzle pieces from the camera vision system output.

Expected input directory layout:
    input/
        parts.json      – metadata (px_per_mm, A4 dims, centroids, coordinate system)
        png_0.png       – binary mask for piece 0 (0/255, grayscale)
        png_1.png
        ...
"""

import json
import cv2
import numpy as np
from pathlib import Path

from src.utils.puzzle_piece import PuzzlePiece
from src.utils.pose import Pose


class CameraLoader:
    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)
        self._json: dict = {}

    # ------------------------------------------------------------------
    # Public API (mirrors MockPuzzleGenerator.load_pieces_for_solver)
    # ------------------------------------------------------------------

    def load_json(self) -> dict:
        """Read and return parts.json.  Raises if missing."""
        json_path = self.input_dir / "parts.json"
        with open(json_path, "r") as f:
            self._json = json.load(f)
        return self._json

    # --- Metadata properties ---

    @property
    def px_per_mm(self) -> float:
        return float(self._json["px_per_mm"])

    @property
    def a4_width_mm(self) -> int:
        return round(self._json["a4_size_mm"]["width"])

    @property
    def a4_height_mm(self) -> int:
        return round(self._json["a4_size_mm"]["height"])

    @property
    def origin(self) -> str:
        """Coordinate origin as declared in JSON."""
        return self._json.get("coordinate_system", {}).get("origin", "top_left")

    def validate(self) -> list[str]:
        """Return list of warning strings for any failed validation checks."""
        warnings = []

        if not self._json.get("part_count_is_valid", True):
            expected = self._json.get("expected_part_count", "?")
            actual = self._json.get("part_count", "?")
            warnings.append(
                f"Stückzahl ungültig: erwartet {expected}, erhalten {actual}"
            )

        area_val = self._json.get("area_validation", {})
        if area_val and not area_val.get("is_valid", True):
            err_pct = area_val.get("area_error_percent", "?")
            max_pct = area_val.get("max_allowed_error_percent", "?")
            warnings.append(
                f"Flächenvalidierung fehlgeschlagen: Fehler {err_pct:.2f}% > max {max_pct}%"
            )

        return warnings

    # --- Loading ---

    def load_pieces_for_solver(self, scale: float = 1.0):
        """
        Load piece masks, apply scale, return (piece_ids, piece_shapes).

        scale = solver_px_per_mm / native_px_per_mm
        """
        piece_shapes: dict[int, np.ndarray] = {}
        piece_ids: list[int] = []

        for part in self._json["parts"]:
            idx = part["index"] - 1  # JSON is 1-based
            mask_path = self.input_dir / part["mask_filename"]

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  ⚠️  Maske nicht gefunden: {mask_path}")
                continue

            mask = (mask > 127).astype(np.uint8)

            if scale != 1.0:
                new_h = max(1, int(round(mask.shape[0] * scale)))
                new_w = max(1, int(round(mask.shape[1] * scale)))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            piece_shapes[idx] = mask
            piece_ids.append(idx)

        return piece_ids, piece_shapes

    def create_puzzle_pieces(self, solver_px_per_mm: float) -> list:
        """
        Build PuzzlePiece objects with pick poses from centroid data.

        Converts centroid_mm from the JSON coordinate system to image
        coordinates (origin top-left, x right, y down):

          top_left    x→right, y→down   no flip needed
          bottom_left x→right, y→up     flip y: y_img = height - y
          top_right   x→left,  y→down   flip x: x_img = width  - x
          bottom_right x→left, y→up     flip both
        """
        pieces = []
        w_mm = self.a4_width_mm
        h_mm = self.a4_height_mm
        origin = self.origin

        flip_x = origin in ("top_right", "bottom_right")
        flip_y = origin in ("bottom_left", "bottom_right")

        for part in self._json["parts"]:
            idx = part["index"] - 1
            cx_mm = part["centroid_mm"]["x"]
            cy_mm = part["centroid_mm"]["y"]

            x_mm = (w_mm - cx_mm) if flip_x else cx_mm
            y_mm = (h_mm - cy_mm) if flip_y else cy_mm

            x_px = x_mm * solver_px_per_mm
            y_px = y_mm * solver_px_per_mm

            pick_pose = Pose(x=float(x_px), y=float(y_px), theta=0.0)
            piece = PuzzlePiece(pid=str(idx), pick=pick_pose)
            pieces.append(piece)

        return pieces

    @staticmethod
    def has_parts_json(directory: str | Path) -> bool:
        return (Path(directory) / "parts.json").exists()
