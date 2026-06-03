"""
Wall-align finetune step.

For each solved placement:
  - Corner pieces (near 2 walls): push flush to both walls.
  - Edge pieces (near 1 wall / have a 'side' field): push flush to their wall,
    then slide along it to maximise score.
  - Center pieces: left unchanged.

Rotations are never modified — only x/y positions are adjusted.
"""

import numpy as np

from src.utils.geometry import rotate_and_crop


class WallAlignFinetuner:
    def __init__(self, renderer, scorer, slide_positions: int = 20, wall_threshold: int = 60):
        self.renderer = renderer
        self.scorer = scorer
        self.slide_positions = slide_positions
        self.wall_threshold = wall_threshold

    def finetune(self, placements, piece_shapes, target):
        height, width = target.shape

        result_corners = []
        to_process_edges = []
        result_centers = []

        for p in placements:
            role = self._placement_role(p, piece_shapes, width, height)
            if role == "corner":
                result_corners.append(self._push_corner(p, piece_shapes, width, height))
            elif role == "edge":
                to_process_edges.append(p)
            else:
                result_centers.append(p)

        result_edges = []
        for p in to_process_edges:
            placed_so_far = result_corners + result_edges
            result_edges.append(self._push_and_slide_edge(p, piece_shapes, placed_so_far, target, width, height))

        result_centers_placed = []
        for p in result_centers:
            placed_so_far = result_corners + result_edges + result_centers_placed
            result_centers_placed.append(self._place_center(p, piece_shapes, placed_so_far, target, width, height))

        final = result_corners + result_edges + result_centers_placed
        print(f"  WallAlign: {len(result_corners)} corners pushed, "
              f"{len(result_edges)} edges slid, {len(result_centers_placed)} centers grid-placed")
        return final

    def _placement_role(self, placement, piece_shapes, width, height):
        if "side" in placement:
            return "edge"

        pid = placement["piece_id"]
        theta = placement["theta"]
        rotated = rotate_and_crop(piece_shapes[pid], theta)
        ph, pw = rotated.shape
        x, y = placement["x"], placement["y"]
        thr = self.wall_threshold

        near_left   = x <= thr
        near_right  = (x + pw) >= (width - thr)
        near_top    = y <= thr
        near_bottom = (y + ph) >= (height - thr)

        wall_count = sum([near_left, near_right, near_top, near_bottom])
        if wall_count >= 2:
            return "corner"
        if wall_count == 1:
            return "edge"
        return "center"

    def _push_corner(self, placement, piece_shapes, width, height):
        pid = placement["piece_id"]
        theta = placement["theta"]
        rotated = rotate_and_crop(piece_shapes[pid], theta)
        ph, pw = rotated.shape

        cx = placement["x"] + pw / 2.0
        cy = placement["y"] + ph / 2.0

        x = 0.0 if cx < width  / 2.0 else float(width  - pw)
        y = 0.0 if cy < height / 2.0 else float(height - ph)

        return {**placement, "x": x, "y": y}

    def _push_and_slide_edge(self, placement, piece_shapes, current_placements, target, width, height):
        pid = placement["piece_id"]
        theta = placement["theta"]
        rotated = rotate_and_crop(piece_shapes[pid], theta)
        ph, pw = rotated.shape

        side = placement.get("side") or self._infer_side(placement, pw, ph, width, height)

        if side in ("left", "right"):
            fixed_x = 0.0 if side == "left" else float(width - pw)
            slide_pos = self._center_of_valid_range(
                current_placements, piece_shapes, axis="y",
                fixed_coord=fixed_x, fixed_key="x", piece_size=ph, wall_size=height,
            )
            return {**placement, "x": fixed_x, "y": slide_pos, "side": side}
        else:
            fixed_y = 0.0 if side == "top" else float(height - ph)
            slide_pos = self._center_of_valid_range(
                current_placements, piece_shapes, axis="x",
                fixed_coord=fixed_y, fixed_key="y", piece_size=pw, wall_size=width,
            )
            return {**placement, "x": slide_pos, "y": fixed_y, "side": side}

    def _center_of_valid_range(self, current_placements, piece_shapes, axis, fixed_coord, fixed_key, piece_size, wall_size):
        """Findet den Mittelpunkt des kollisionsfreien Bereichs entlang der Gleitachse.

        Scannt alle Positionen, bestimmt den laengsten Bereich ohne Ueberlappung,
        gibt dessen Mitte zurueck.
        """
        first_good = None
        last_good = None

        for pos in range(int(wall_size - piece_size) + 1):
            if self._overlaps(current_placements, piece_shapes, axis, pos, piece_size):
                continue
            if first_good is None:
                first_good = pos
            last_good = pos

        if first_good is None:
            return float((wall_size - piece_size) / 2.0)
        return float((first_good + last_good) / 2.0)

    def _overlaps(self, placements, piece_shapes, axis, pos, piece_size):
        """True wenn das Teil an Position pos entlang axis mit einem platzierten Teil ueberlappt."""
        p_start = float(pos)
        p_end = p_start + piece_size
        for p in placements:
            pid = p["piece_id"]
            if pid not in piece_shapes:
                continue
            rotated = rotate_and_crop(piece_shapes[pid], p["theta"])
            ph, pw = rotated.shape
            if axis == "x":
                o_start, o_end = p["x"], p["x"] + pw
            else:
                o_start, o_end = p["y"], p["y"] + ph
            if o_start < p_end and o_end > p_start:
                return True
        return False

    def slide_edges(self, placements, piece_shapes, target):
        """Slide only edge pieces along their wall for best fit. Corners and centers unchanged."""
        height, width = target.shape
        result = []
        placed_so_far = []
        for p in placements:
            role = self._placement_role(p, piece_shapes, width, height)
            if role == "edge":
                slid = self._push_and_slide_edge(p, piece_shapes, placed_so_far, target, width, height)
                result.append(slid)
                placed_so_far.append(slid)
            else:
                result.append(p)
                placed_so_far.append(p)
        edge_count = sum(1 for p in placements if self._placement_role(p, piece_shapes, width, height) == "edge")
        print(f"  EdgeSlide: {edge_count} edge(s) slid along wall")
        return result

    def _place_center(self, placement, piece_shapes, current_placements, target, width, height):
        pid = placement["piece_id"]
        theta = placement["theta"]
        rotated = rotate_and_crop(piece_shapes[pid], theta)
        ph, pw = rotated.shape

        x = self._center_of_valid_range(
            current_placements, piece_shapes, axis="x",
            fixed_coord=None, fixed_key="y", piece_size=pw, wall_size=width,
        )
        y = self._center_of_valid_range(
            current_placements, piece_shapes, axis="y",
            fixed_coord=None, fixed_key="x", piece_size=ph, wall_size=height,
        )
        return {**placement, "x": x, "y": y}

    def _infer_side(self, placement, pw, ph, width, height):
        cx = placement["x"] + pw / 2.0
        cy = placement["y"] + ph / 2.0
        dists = {"left": cx, "right": width - cx, "top": cy, "bottom": height - cy}
        return min(dists, key=dists.get)
