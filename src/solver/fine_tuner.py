"""
Feinabstimmung der Puzzle-Platzierung durch koordinatenweises Optimieren (x, y, theta).
"""

import numpy as np
from copy import deepcopy
from typing import List, Optional

from src.utils.geometry import rotate_and_crop


class FineTuner:
    """
    Verbessert eine vorhandene Loesung durch lokale Suche:
    Fuer jedes Teil werden x, y und theta in kleinen Schritten variiert,
    um den Gesamtscore zu maximieren. Wiederholt bis keine Verbesserung mehr.

    Koordinaten sind in fine-Resolution. Wird coarse_ratio gesetzt,
    wird nach jedem Teilschritt ein herunterskalierter Snapshot an
    all_guesses angehaengt (fuer den Visualizer).
    """

    def __init__(
        self,
        renderer,
        scorer,
        xy_range: int,
        xy_step: int,
        theta_range: float,
        theta_step: float,
        max_passes: int,
    ):
        self.renderer = renderer
        self.scorer = scorer
        self.xy_range = xy_range
        self.xy_step = xy_step
        self.theta_range = theta_range
        self.theta_step = theta_step
        self.max_passes = max_passes

    def finetune(
        self,
        placements: List[dict],
        piece_shapes: dict,
        target: np.ndarray,
        all_guesses: Optional[List] = None,
        coarse_ratio: float = 1.0,
    ):
        """
        Gibt (verbesserte_placements_fine, score) zurueck.

        all_guesses: wenn angegeben, wird nach jeder Teilverbesserung ein
                     Snapshot in Coarse-Koordinaten angehaengt.
        coarse_ratio: fine→coarse Faktor (= solver_scale / finetune_scale).
        """
        placements = deepcopy(placements)

        initial_rendered = self.renderer.render(placements, piece_shapes)
        initial_score = self.scorer.score(initial_rendered, target)
        print(f"  [FineTuner] start={initial_score:.0f}")

        for pass_idx in range(self.max_passes):
            improved = False

            for i in range(len(placements)):
                new_score, new_x, new_y, new_theta = self._optimize_piece(
                    i, placements, piece_shapes, target
                )

                old = placements[i]
                if new_x != old["x"] or new_y != old["y"] or new_theta != old["theta"]:
                    placements[i] = {**old, "x": new_x, "y": new_y, "theta": new_theta}
                    improved = True

                    if all_guesses is not None:
                        all_guesses.append(_to_coarse(placements, coarse_ratio))

            if not improved:
                print(f"  [FineTuner] converged after {pass_idx + 1} passes")
                break
        else:
            print(f"  [FineTuner] max passes reached ({self.max_passes})")

        final_rendered = self.renderer.render(placements, piece_shapes)
        final_score = self.scorer.score(final_rendered, target)
        print(f"  [FineTuner] end={final_score:.0f} (Δ{final_score - initial_score:+.0f})")
        return placements, final_score

    def _optimize_piece(self, piece_idx, placements, piece_shapes, target):
        """Optimiert ein einzelnes Teil gegenueber dem Hintergrund der anderen."""
        background = np.zeros(
            (self.renderer.height, self.renderer.width), dtype=np.float32
        )
        for i, p in enumerate(placements):
            if i == piece_idx:
                continue
            pid = p["piece_id"]
            if pid in piece_shapes:
                rotated = rotate_and_crop(piece_shapes[pid], p["theta"])
                _place(background, rotated, int(p["x"]), int(p["y"]))

        cur = placements[piece_idx]
        pid = cur["piece_id"]
        if pid not in piece_shapes:
            return -float("inf"), cur["x"], cur["y"], cur["theta"]

        best_score = -float("inf")
        best_x, best_y, best_theta = cur["x"], cur["y"], cur["theta"]

        xs = range(-self.xy_range, self.xy_range + 1, self.xy_step)
        ys = range(-self.xy_range, self.xy_range + 1, self.xy_step)
        n_steps = max(1, int(self.theta_range / self.theta_step))
        thetas = [k * self.theta_step for k in range(-n_steps, n_steps + 1)]

        for dtheta in thetas:
            theta = cur["theta"] + dtheta
            rotated = rotate_and_crop(piece_shapes[pid], theta)

            for dx in xs:
                x = int(cur["x"]) + dx
                for dy in ys:
                    y = int(cur["y"]) + dy
                    canvas = background.copy()
                    _place(canvas, rotated, x, y)
                    score = self.scorer.score(canvas, target)
                    if score > best_score:
                        best_score = score
                        best_x, best_y, best_theta = cur["x"] + dx, cur["y"] + dy, theta

        return best_score, best_x, best_y, best_theta


def _to_coarse(placements: List[dict], coarse_ratio: float) -> List[dict]:
    """Gibt eine Kopie der Placements in Coarse-Koordinaten zurueck."""
    if coarse_ratio == 1.0:
        return deepcopy(placements)
    return [{**p, "x": p["x"] * coarse_ratio, "y": p["y"] * coarse_ratio} for p in placements]


def _place(canvas: np.ndarray, shape: np.ndarray, x: int, y: int) -> None:
    h, w = shape.shape[:2]
    y1, y2 = max(0, y), min(canvas.shape[0], y + h)
    x1, x2 = max(0, x), min(canvas.shape[1], x + w)
    sy1, sx1 = max(0, -y), max(0, -x)
    sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
    if y2 > y1 and x2 > x1:
        canvas[y1:y2, x1:x2] += shape[sy1:sy2, sx1:sx2]
