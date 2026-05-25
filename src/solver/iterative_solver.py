"""
Iterative solver with MODE SWITCHING
Switches between CORNER_SEARCH and EDGE_REFINEMENT modes adaptively.
"""

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from src.solver.corner_fitter import CornerFit, CornerFitter
from src.solver.corner_placement import evaluate_corner_layouts, place_corners
from src.solver.edge_placement import try_edge_placement_on_corners
from src.utils.geometry import rotate_and_crop
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece


class SolverMode(Enum):
    """Solver operating modes."""

    CORNER_SEARCH = "corner_search"  # Rapidly evaluate corner layouts
    EDGE_REFINEMENT = "edge_refinement"  # Adaptive edge placement for best corners


@dataclass
class IterativeSolution:
    """Result from iterative solving process."""

    success: bool
    anchor_fit: Optional[CornerFit]
    remaining_placements: List[dict]
    score: float
    iteration: int
    total_iterations: int
    all_guesses: Optional[List[List[dict]]] = None


@dataclass
class SolverState:
    """Tracks solver state across mode switches."""

    mode: SolverMode
    corner_evaluations: List[tuple]  # (combo_idx, corner_indices, placements, score)
    best_score: float
    best_guess: Optional[List[dict]]
    iterations_since_improvement: int
    current_corner_rank: int  # Which corner layout we're currently refining
    refinement_attempts: int  # How many refinement attempts on current corner


class IterativeSolver:
    """
    Iterative puzzle solver with adaptive mode switching.

    MODES:
    - CORNER_SEARCH: Rapidly evaluate corner-only layouts in batches
    - EDGE_REFINEMENT: Adaptive edge placement on promising corners

    The solver switches between modes based on progress:
    - Starts in CORNER_SEARCH
    - Switches to EDGE_REFINEMENT when promising corners are found
    - Switches back to CORNER_SEARCH if refinement plateaus
    """

    def __init__(self, renderer, scorer, guess_generator, tuning=None):
        self.renderer = renderer
        self.scorer = scorer
        self.guess_generator = guess_generator
        self.tuning = tuning
        self.corner_fitter = None
        self.all_guesses = []
        self.all_scores = []
        self._px_per_mm = 1.0

    def solve_iteratively(
        self,
        piece_shapes: Dict[int, np.ndarray],
        target: np.ndarray,
        puzzle_pieces: list,
        score_max: float,
        score_accept: float | None = None,
        initial_corner_count: int = 60,
        max_corners_to_refine: int = 20,
        refinement_patience: int = 5,
        max_iterations: int = 600,
        px_per_mm: float = 2.0,
    ) -> IterativeSolution:
        """
        1. Evaluate many corners upfront (e.g., 100)
        2. Pick top N corners (e.g., top 10)
        3. Do edge refinement on each
        4. Switch to next corner when stuck
        """

        height, width = target.shape
        self._px_per_mm = px_per_mm

        # Dilate the scoring target once so small physical gaps between pieces
        # don't get penalised. Piece shapes and positions are untouched.
        dilation_px = int(round((self.tuning.gap_dilation_mm if self.tuning else 0) * px_per_mm))
        if dilation_px > 0:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
            target = cv2.dilate(target.astype(np.uint8), kernel).astype(target.dtype)
            print(f"  Gap dilation: {dilation_px}px ({self.tuning.gap_dilation_mm}mm) applied to target")
        self.corner_fitter = CornerFitter(
            width=width, height=height, tuning=self.tuning
        )

        # Reset state
        self.all_guesses = []
        self.all_scores = []

        # Pool all pieces that have corner detections upfront — corners, edges, and centers.
        # This avoids multi-round escalation: all candidates are ranked by corner quality
        # and the combination generator sorts by quality, so well-classified corners naturally
        # appear first and the threshold fires early without wasting a full round on wrong sets.
        corner_candidates = [
            piece for piece in puzzle_pieces
            if piece.corners and piece.piece_type in ("corner", "edge", "center")
        ]

        if len(corner_candidates) < 4:
            print(f"  ⚠️ Only {len(corner_candidates)} pieces with corner detections (need 4)!")
            return self._empty_solution()

        print(f"\n  Corner candidates (all pieces with corner detections): {[int(p.id) for p in corner_candidates]}")
        for p in corner_candidates:
            print(f"    Piece {p.id}: type={p.piece_type}, {len(p.corners)} corners")

        all_corner_combinations = self._generate_combinations(corner_candidates)

        if not all_corner_combinations:
            print("  ⚠️ Could not generate corner combinations!")
            return self._empty_solution()

        # Always evaluate all combinations in Phase 1 — corner-only scoring is cheap
        # and guarantees the best layout makes it into Phase 2 regardless of classifier quality.
        phase1_count = len(all_corner_combinations)

        early_exit_score = score_accept if score_accept is not None else score_max

        solution = self._solve_with_mode_switching(
            corner_candidates,
            all_corner_combinations,
            piece_shapes,
            target,
            puzzle_pieces,
            score_max,
            early_exit_score,
            phase1_count,
            max_corners_to_refine,
            refinement_patience,
            max_iterations,
        )

        best_solution = solution

        return best_solution

    def _generate_combinations(self, corner_candidates):
        """Generate sorted corner combinations (permutations × rotations) for given candidates."""
        candidates_with_corners = [p for p in corner_candidates if p.corners]
        if len(candidates_with_corners) < 4:
            print(f"  ⚠️ Only {len(candidates_with_corners)} candidates have corner data (need 4)")
            return []

        piece_permutations = list(itertools.permutations(candidates_with_corners, 4))

        all_corner_combinations = []
        for perm in piece_permutations:
            piece_corner_options = [list(range(len(p.corners))) for p in perm]
            for rotation_combo in itertools.product(*piece_corner_options):
                all_corner_combinations.append((perm, rotation_combo))

        def combo_quality(combo):
            perm, rotation_indices = combo
            return sum(piece.corners[corner_idx].quality
                       for piece, corner_idx in zip(perm, rotation_indices))

        all_corner_combinations.sort(key=combo_quality, reverse=True)
        print(f"  {len(candidates_with_corners)} pieces with corners → {len(all_corner_combinations)} combinations")
        return all_corner_combinations

    def _solve_with_mode_switching(
        self,
        corner_pieces,
        all_corner_combinations,
        piece_shapes,
        target,
        puzzle_pieces,
        score_max,
        score_accept,
        initial_corner_count,
        max_corners_to_refine,
        refinement_patience,
        max_iterations,
    ) -> IterativeSolution:
        """
        Phase 1: score all corner layouts (corner-only, cheap).
        Phase 2: try edge placement in batches. Exit early when score_accept is reached;
                 success is determined by score_max.
        """

        total = len(all_corner_combinations)

        # ====================================================================
        # PHASE 1: score every corner layout, no edge placement
        # ====================================================================
        print(f"  Phase 1: scoring all {total} corner layouts...")
        corner_evaluations = evaluate_corner_layouts(
            all_combinations=all_corner_combinations,
            initial_corner_count=total,
            renderer=self.renderer,
            scorer=self.scorer,
            piece_shapes=piece_shapes,
            target=target,
            all_guesses=self.all_guesses,
            all_scores=self.all_scores,
        )
        # corner_evaluations is sorted best→worst by corner-only score

        # ====================================================================
        # PHASE 2: full edge placement in batches, exit at threshold
        # ====================================================================
        batch_size = max_corners_to_refine
        print(f"  Phase 2: edge placement in batches of {batch_size} (accept={score_accept}, max={score_max})...")

        edge_kwargs = {}
        if self.tuning:
            edge_kwargs = dict(
                slide_positions=self.tuning.slide_positions,
                slide_patience=self.tuning.slide_patience,
                center_piece_margin=self.tuning.center_piece_margin,
            )

        best_overall_score = -float("inf")
        best_overall_solution = None
        layouts_tried = 0

        for batch_start in range(0, len(corner_evaluations), batch_size):
            batch = corner_evaluations[batch_start: batch_start + batch_size]
            print(f"  Batch {batch_start // batch_size + 1}: layouts {batch_start + 1}–{batch_start + len(batch)}")

            for entry in batch:
                _, piece_perm, _, corner_placements, corner_only_score = entry
                layouts_tried += 1
                guesses_before = len(self.all_guesses)

                solution_with_edges = try_edge_placement_on_corners(
                    corner_pieces=piece_perm,
                    corner_placements=corner_placements,
                    corner_only_score=corner_only_score,
                    piece_shapes=piece_shapes,
                    target=target,
                    puzzle_pieces=puzzle_pieces,
                    layout_number=layouts_tried,
                    renderer=self.renderer,
                    scorer=self.scorer,
                    all_guesses=self.all_guesses,
                    all_scores=self.all_scores,
                    **edge_kwargs,
                )

                final_score = solution_with_edges["final_score"]
                final_placements = solution_with_edges["final_placements"]
                layout_guesses = len(self.all_guesses) - guesses_before

                if final_score > best_overall_score:
                    best_overall_score = final_score
                    best_overall_solution = final_placements
                    print(f"    Layout {layouts_tried}: pieces {[int(p.id) for p in piece_perm]} score {final_score:.0f} [{layout_guesses} guesses] *** NEW BEST ***")
                else:
                    print(f"    Layout {layouts_tried}: pieces {[int(p.id) for p in piece_perm]} score {final_score:.0f} [{layout_guesses} guesses]")

                if final_score >= score_accept:
                    print(f"  ACCEPTED: {final_score:.0f} >= {score_accept} (layout {layouts_tried}, {len(self.all_guesses)} total guesses)")
                    self._update_piece_poses(puzzle_pieces, final_placements)
                    return IterativeSolution(
                        success=True,
                        anchor_fit=None,
                        remaining_placements=final_placements,
                        score=final_score,
                        iteration=layouts_tried,
                        total_iterations=total,
                        all_guesses=self.all_guesses,
                    )

        print(f"  Best: {best_overall_score:.0f} after {layouts_tried} layouts | guesses: {len(self.all_guesses)}")
        self._update_piece_poses(puzzle_pieces, best_overall_solution)
        return IterativeSolution(
            success=best_overall_score >= score_max,
            anchor_fit=None,
            remaining_placements=best_overall_solution,
            score=best_overall_score,
            iteration=layouts_tried,
            total_iterations=total,
            all_guesses=self.all_guesses,
        )

    def _pull_placements_to_center(self, placements, canvas_w, canvas_h):
        """Verschiebt alle Platzierungen um pull_to_center_mm in Richtung Puzzlemitte."""
        if not placements:
            return
        pull_mm = self.tuning.pull_to_center_mm if self.tuning else 0.0
        if pull_mm <= 0:
            return
        pull_px = pull_mm * self._px_per_mm
        cx, cy = canvas_w / 2.0, canvas_h / 2.0
        for p in placements:
            dx = cx - p["x"]
            dy = cy - p["y"]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < 1e-6:
                continue
            p["x"] += dx / dist * pull_px
            p["y"] += dy / dist * pull_px

    def _update_piece_poses(self, puzzle_pieces, placements):
        """Update PuzzlePiece objects with place_pose."""

        piece_lookup = {int(p.id): p for p in puzzle_pieces}

        for placement in placements:
            piece_id = placement["piece_id"]
            if piece_id in piece_lookup:
                piece_lookup[piece_id].place_pose = Pose(
                    x=placement["x"], y=placement["y"], theta=placement["theta"]
                )

    def _rotate_and_crop(self, shape, angle):
        """Rotate and crop piece shape."""
        return rotate_and_crop(shape, angle)

    def _empty_solution(self):
        """Return empty solution."""
        return IterativeSolution(
            success=False,
            anchor_fit=None,
            remaining_placements=[],
            score=-float("inf"),
            iteration=0,
            total_iterations=0,
            all_guesses=[],
        )
