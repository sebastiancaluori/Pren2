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
    ) -> IterativeSolution:
        """
        1. Evaluate many corners upfront (e.g., 100)
        2. Pick top N corners (e.g., top 10)
        3. Do edge refinement on each
        4. Switch to next corner when stuck
        """

        height, width = target.shape
        self.corner_fitter = CornerFitter(
            width=width, height=height, tuning=self.tuning
        )

        # Reset state
        self.all_guesses = []
        self.all_scores = []

        # Find corner pieces - USE PIECE_TYPE CLASSIFICATION
        corner_pieces = [
            piece for piece in puzzle_pieces if piece.piece_type == "corner"
        ]

        if not corner_pieces:
            print("  ⚠️  No corner pieces found (piece_type == 'corner')!")
            # Fallback to has_corner if piece_type not set
            corner_pieces = [
                piece
                for piece in puzzle_pieces
                if piece.has_corner and len(piece.corners) > 0
            ]
            if not corner_pieces:
                return self._empty_solution()
            print(
                f"  ⚠️  Using fallback: found {len(corner_pieces)} pieces with corner features"
            )

        print(f"\n  Found {len(corner_pieces)} corner pieces (by piece_type):")
        for piece in corner_pieces:
            print(
                f"    Piece {piece.id}: type={piece.piece_type}, {len(piece.corners)} corners"
            )

        # Validate corner piece count
        if len(corner_pieces) != 4:
            print(
                f"\n  ⚠️  WARNING: Expected 4 corner pieces, found {len(corner_pieces)}!"
            )
            if len(corner_pieces) < 4:
                print(
                    f"      Not enough corner pieces - puzzle may not solve correctly"
                )
            else:
                print(f"      Too many corner pieces - will only use first 4")
                corner_pieces = corner_pieces[:4]

        # Generate combinations:
        # Step 1: Permutations of pieces (which piece in which corner)
        piece_permutations = list(itertools.permutations(corner_pieces))

        print(
            f"\n  Piece permutations: {len(piece_permutations)} (which piece → which corner)"
        )
        print(f"  Example: {[int(p.id) for p in piece_permutations[0]]}")
        print(f"           {[int(p.id) for p in piece_permutations[1]]}")

        # Step 2: For each permutation, generate corner rotation combinations
        all_corner_combinations = []

        for perm in piece_permutations:
            # For this permutation, get all rotation combinations
            piece_corner_options = [[i for i in range(len(p.corners))] for p in perm]
            rotation_combos = list(itertools.product(*piece_corner_options))

            # Store (piece_permutation, rotation_combo)
            for rotation_combo in rotation_combos:
                all_corner_combinations.append((perm, rotation_combo))

        print(f"\n  Total combinations: {len(all_corner_combinations)}")
        print(f"    = {len(piece_permutations)} permutations × avg rotations per perm")

        # Sort by quality (sum of corner qualities for each combo)
        def combo_quality(combo):
            perm, rotation_indices = combo
            total_quality = 0
            for piece, corner_idx in zip(perm, rotation_indices):
                total_quality += piece.corners[corner_idx].quality
            return total_quality

        all_corner_combinations.sort(key=combo_quality, reverse=True)

        print(f"  Total corner combinations: {len(all_corner_combinations)}")

        # Show piece distribution for verification
        all_edge_pieces = [p for p in puzzle_pieces if p.piece_type == "edge"]
        all_center_pieces = [p for p in puzzle_pieces if p.piece_type == "center"]

        print(f"\n  📋 Piece Distribution:")
        print(
            f"     Corners (→ placed in 4 corners): {[int(p.id) for p in corner_pieces]}"
        )
        print(
            f"     Edges (→ placed along sides):   {[int(p.id) for p in all_edge_pieces]}"
        )
        print(
            f"     Centers (→ placed in middle):   {[int(p.id) for p in all_center_pieces]}"
        )

        return self._solve_with_mode_switching(
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
        )

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
        print(f"\n  === PHASE 1: Evaluate corner layouts (no edges yet) ===")

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

        initial_corners_to_evaluate = min(
            initial_corner_count, len(all_corner_combinations)
        )

        best_corner_score = corner_evaluations[0][4]  # score is at index 4
        worst_in_top10 = corner_evaluations[min(9, len(corner_evaluations) - 1)][4]
        print(f"\n  Score range: {worst_in_top10:.1f} → {best_corner_score:.1f}")

        # ========================================================================
        # PHASE 2: ITERATE THROUGH CORNER LAYOUTS WITH SMART EDGE PLACEMENT
        # ========================================================================
        print(
            f"\n  === PHASE 2: Iterate through corner layouts with edge placement ==="
        )
        print(
            f"  Will try up to {max_corners_to_refine} corner layouts until score >= {score_threshold}"
        )

        best_overall_score = -float("inf")
        best_overall_solution = None
        layouts_tried = 0

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

            print(
                f"\n  → Layout {layout_idx + 1}/{corners_to_try}: Pieces {[int(p.id) for p in current_piece_perm]}"
            )
            print(f"    Corner-only score: {corner_only_score:.1f}")

            # Try edge placement on this corner layout
            edge_kwargs = {}
            if self.tuning:
                edge_kwargs = dict(
                    slide_positions=self.tuning.slide_positions,
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

            final_score = solution_with_edges["final_score"]
            final_placements = solution_with_edges["final_placements"]

            print(
                f"    Final score with edges: {final_score:.1f} ({final_score - corner_only_score:+.1f})"
            )

            # Track best solution
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_solution = final_placements
                print(f"    ✓ NEW BEST SOLUTION! Score: {final_score:.1f}")

            # Check if we've reached the threshold
            if final_score >= score_threshold:
                print(
                    f"\n🎯 THRESHOLD REACHED! Score {final_score:.1f} >= {score_threshold}"
                )
                print(f"   Used layout {layout_idx + 1}/{len(corner_evaluations)}")

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

        print(f"\n🏆 Final Results:")
        print(f"   Best score: {best_overall_score:.1f}")
        print(f"   Layouts tried: {layouts_tried}/{corners_to_try}")
        print(f"   Success: {best_overall_score >= score_threshold}")
        print(f"   Total guesses: {len(self.all_guesses)}")

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
