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

    def solve_iteratively(
        self,
        piece_shapes: Dict[int, np.ndarray],
        target: np.ndarray,
        puzzle_pieces: list,
        score_threshold: float,
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

        # Inflate piece masks by gap_dilation_mm so the scorer penalises placements
        # that are spatially wrong even when physical gaps give extra wiggle room.
        dilation_px = int(round((self.tuning.gap_dilation_mm if self.tuning else 0) * px_per_mm))
        if dilation_px > 0:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
            piece_shapes = {
                pid: cv2.dilate(mask.astype(np.uint8), kernel).astype(mask.dtype)
                for pid, mask in piece_shapes.items()
            }
            print(f"  Gap dilation: {dilation_px}px ({self.tuning.gap_dilation_mm}mm)")
        self.corner_fitter = CornerFitter(
            width=width, height=height, tuning=self.tuning
        )

        # Reset state
        self.all_guesses = []
        self.all_scores = []

        # Find corner pieces — piece_analyzer re-evaluation has already corrected classifications
        corner_candidates = [
            piece for piece in puzzle_pieces if piece.piece_type == "corner"
        ]

        if not corner_candidates:
            print("  ⚠️ No corner pieces found!")
            return self._empty_solution()

        # Escalation: try with corner-only pieces first, then expand to edge/center pieces
        # that have corner features if no solution is found.
        best_solution = self._empty_solution()
        escalation_round = 0
        newly_added = []  # pieces added in the most recent escalation step

        while True:
            escalation_round += 1
            print(f"\n--- ESCALATION ROUND {escalation_round}: {len(corner_candidates)} candidates {[int(p.id) for p in corner_candidates]} ---")

            # In escalation rounds, only generate combinations that include at least one
            # of the newly added pieces — those are the only combinations not tried before.
            all_corner_combinations = self._generate_combinations(
                corner_candidates, required_pieces=newly_added if newly_added else None
            )

            if not all_corner_combinations:
                print("  ⚠️ Not enough candidates with corners for 4 corner slots — stopping")
                break

            # In escalation rounds, evaluate all filtered combinations in Phase 1
            # (already a reduced set), then refine more layouts since we haven't seen any of them.
            phase1_count = len(all_corner_combinations) if newly_added else initial_corner_count
            phase2_count = len(all_corner_combinations) if newly_added else max_corners_to_refine

            solution = self._solve_with_mode_switching(
                corner_candidates,
                all_corner_combinations,
                piece_shapes,
                target,
                puzzle_pieces,
                score_threshold,
                phase1_count,
                phase2_count,
                refinement_patience,
                max_iterations,
            )

            if solution.score > best_solution.score:
                best_solution = solution

            if solution.success:
                return best_solution

            # Escalate 1: add edge pieces that have corner detections
            edges_with_corners = [
                p for p in puzzle_pieces
                if p.piece_type == "edge" and p.corners and p not in corner_candidates
            ]
            if edges_with_corners:
                newly_added = edges_with_corners
                corner_candidates = corner_candidates + edges_with_corners
                print(f"  Escalating: adding {len(edges_with_corners)} edge piece(s) with corners: {[int(p.id) for p in edges_with_corners]}")
                continue

            # Escalate 2: add center pieces that have corner detections
            centers_with_corners = [
                p for p in puzzle_pieces
                if p.piece_type == "center" and p.corners and p not in corner_candidates
            ]
            if centers_with_corners:
                newly_added = centers_with_corners
                corner_candidates = corner_candidates + centers_with_corners
                print(f"  Escalating: adding {len(centers_with_corners)} center piece(s) with corners: {[int(p.id) for p in centers_with_corners]}")
                continue

            print(f"  Escalation exhausted after {escalation_round} round(s). Best score: {best_solution.score:.1f}")
            break

        return best_solution

    def _generate_combinations(self, corner_candidates, required_pieces=None):
        """Generate sorted corner combinations (permutations × rotations) for given candidates.

        If required_pieces is set, only combinations that include at least one of those
        pieces are returned — used in escalation rounds to skip already-explored territory.
        """
        candidates_with_corners = [p for p in corner_candidates if p.corners]
        if len(candidates_with_corners) < 4:
            print(f"  ⚠️ Only {len(candidates_with_corners)} candidates have corner data (need 4)")
            return []

        piece_permutations = list(itertools.permutations(candidates_with_corners, 4))

        required_ids = {int(p.id) for p in required_pieces} if required_pieces else None

        all_corner_combinations = []
        for perm in piece_permutations:
            if required_ids and not any(int(p.id) in required_ids for p in perm):
                continue
            piece_corner_options = [list(range(len(p.corners))) for p in perm]
            for rotation_combo in itertools.product(*piece_corner_options):
                all_corner_combinations.append((perm, rotation_combo))

        def combo_quality(combo):
            perm, rotation_indices = combo
            return sum(piece.corners[corner_idx].quality
                       for piece, corner_idx in zip(perm, rotation_indices))

        all_corner_combinations.sort(key=combo_quality, reverse=True)
        filter_note = f" (filtered to include pieces {list(required_ids)})" if required_ids else ""
        print(f"  {len(candidates_with_corners)} pieces with corners → {len(all_corner_combinations)} combinations{filter_note}")
        return all_corner_combinations

    def _solve_with_mode_switching(
        self,
        corner_pieces,
        all_corner_combinations,
        piece_shapes,
        target,
        puzzle_pieces,
        score_threshold,
        initial_corner_count,
        max_corners_to_refine,
        refinement_patience,
        max_iterations,
    ) -> IterativeSolution:
        """Main mode-switching solve loop with proper iteration through corner layouts."""

        # ========================================================================
        # PHASE 1: EVALUATE MANY CORNERS FIRST (no edge refinement yet!)
        # ========================================================================
        print(f"  Phase 1: evaluating {min(initial_corner_count, len(all_corner_combinations))} corner layouts...")

        guesses_before_phase1 = len(self.all_guesses)
        corner_evaluations = evaluate_corner_layouts(
            all_combinations=all_corner_combinations,
            initial_corner_count=initial_corner_count,
            renderer=self.renderer,
            scorer=self.scorer,
            piece_shapes=piece_shapes,
            target=target,
            all_guesses=self.all_guesses,
            all_scores=self.all_scores,
        )
        guesses_phase1 = len(self.all_guesses) - guesses_before_phase1

        initial_corners_to_evaluate = min(
            initial_corner_count, len(all_corner_combinations)
        )

        best_corner_score = corner_evaluations[0][4]
        worst_in_top10 = corner_evaluations[min(9, len(corner_evaluations) - 1)][4]
        print(f"  Corner scores: {worst_in_top10:.0f} → {best_corner_score:.0f}")
        print(f"  Phase 2: refining up to {min(max_corners_to_refine, len(corner_evaluations))} layouts (threshold {score_threshold})...")

        best_overall_score = -float("inf")
        best_overall_solution = None
        layouts_tried = 0
        guesses_phase2 = 0

        # Try multiple corner layouts
        corners_to_try = min(max_corners_to_refine, len(corner_evaluations))

        for layout_idx in range(corners_to_try):
            (
                _,
                current_piece_perm,
                current_rotation_indices,
                current_corner_placements,
                corner_only_score,
            ) = corner_evaluations[layout_idx]
            layouts_tried += 1
            guesses_before_layout = len(self.all_guesses)
            print(f"  Layout {layout_idx + 1}/{corners_to_try}: pieces {[int(p.id) for p in current_piece_perm]}, corner score {corner_only_score:.0f}")

            # Try edge placement on this corner layout
            edge_kwargs = {}
            if self.tuning:
                edge_kwargs = dict(
                    slide_positions=self.tuning.slide_positions,
                    slide_patience=self.tuning.slide_patience,
                    center_piece_margin=self.tuning.center_piece_margin,
                )
            solution_with_edges = try_edge_placement_on_corners(
                corner_pieces=current_piece_perm,
                corner_placements=current_corner_placements,
                corner_only_score=corner_only_score,
                piece_shapes=piece_shapes,
                target=target,
                puzzle_pieces=puzzle_pieces,
                layout_number=layout_idx + 1,
                renderer=self.renderer,
                scorer=self.scorer,
                all_guesses=self.all_guesses,
                all_scores=self.all_scores,
                **edge_kwargs,
            )

            layout_guesses = len(self.all_guesses) - guesses_before_layout
            guesses_phase2 += layout_guesses

            final_score = solution_with_edges["final_score"]
            final_placements = solution_with_edges["final_placements"]

            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_solution = final_placements
                print(f"    → score {final_score:.0f} ({final_score - corner_only_score:+.0f}) [{layout_guesses} guesses] *** NEW BEST ***")
            else:
                print(f"    → score {final_score:.0f} ({final_score - corner_only_score:+.0f}) [{layout_guesses} guesses]")

            if final_score >= score_threshold:
                print(f"  THRESHOLD REACHED: {final_score:.0f} >= {score_threshold} (layout {layout_idx + 1})")
                print(f"  [STATS] phase1={guesses_phase1} | phase2={guesses_phase2} | total={len(self.all_guesses)}")

                # Update piece poses and return success immediately
                self._update_piece_poses(puzzle_pieces, final_placements)

                return IterativeSolution(
                    success=True,
                    anchor_fit=None,
                    remaining_placements=final_placements,
                    score=final_score,
                    iteration=initial_corners_to_evaluate + layouts_tried,
                    total_iterations=len(all_corner_combinations),
                    all_guesses=self.all_guesses,
                )

        print(f"  Best: {best_overall_score:.0f} after {layouts_tried} layouts | guesses: {len(self.all_guesses)} | success: {best_overall_score >= score_threshold}")
        print(f"  [STATS] phase1={guesses_phase1} | phase2={guesses_phase2} | avg/layout={guesses_phase2 // max(1, layouts_tried)}")

        # Update piece poses with best solution
        self._update_piece_poses(puzzle_pieces, best_overall_solution)

        return IterativeSolution(
            success=best_overall_score >= score_threshold,
            anchor_fit=None,
            remaining_placements=best_overall_solution,
            score=best_overall_score,
            iteration=initial_corners_to_evaluate + layouts_tried,
            total_iterations=len(all_corner_combinations),
            all_guesses=self.all_guesses,
        )

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
