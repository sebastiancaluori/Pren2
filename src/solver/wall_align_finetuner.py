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

        final = result_corners + result_edges + result_centers
        print(f"  WallAlign: {len(result_corners)} corners pushed, "
              f"{len(result_edges)} edges slid, {len(result_centers)} centers unchanged")
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
            positions = np.linspace(0, max(0, height - ph), num=self.slide_positions)
            candidates = [{**placement, "x": fixed_x, "y": float(pos), "side": side} for pos in positions]
        else:
            fixed_y = 0.0 if side == "top" else float(height - ph)
            positions = np.linspace(0, max(0, width - pw), num=self.slide_positions)
            candidates = [{**placement, "x": float(pos), "y": fixed_y, "side": side} for pos in positions]

        best_placement = candidates[0]
        best_score = -float("inf")
        for candidate in candidates:
            rendered = self.renderer.render(current_placements + [candidate], piece_shapes)
            score = self.scorer.score(rendered, target)
            if score > best_score:
                best_score = score
                best_placement = candidate

        return best_placement

    def _infer_side(self, placement, pw, ph, width, height):
        cx = placement["x"] + pw / 2.0
        cy = placement["y"] + ph / 2.0
        dists = {"left": cx, "right": width - cx, "top": cy, "bottom": height - cy}
        return min(dists, key=dists.get)
