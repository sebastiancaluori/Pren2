"""
Edge placement logic for the iterative solver.
Handles edge piece positioning, sliding optimization, and center piece placement.
"""

import random
from typing import Dict, List, Optional

import numpy as np

from src.utils.geometry import rotate_and_crop
from src.utils.puzzle_piece import PuzzlePiece


def try_edge_placement_on_corners(
    corner_pieces,
    corner_placements,
    corner_only_score,
    piece_shapes,
    target,
    puzzle_pieces,
    layout_number,
    renderer,
    scorer,
    all_guesses,
    all_scores,
    slide_positions: int = 8,
    slide_patience: int = 3,
    center_piece_margin: int = 50,
) -> dict:
    """Try smart edge placement on a specific corner layout."""

    # Get edge and center pieces — exclude whichever pieces are already placed as corners.
    # Corner-classified pieces that weren't selected for the 4 corner slots (escalation
    # rounds can have more than 4 candidates) are treated as edge pieces here.
    corner_piece_ids = {int(p.id) for p in corner_pieces}
    edge_pieces = [
        p
        for p in puzzle_pieces
        if p.piece_type in ("edge", "corner") and int(p.id) not in corner_piece_ids
    ]
    center_pieces = [
        p
        for p in puzzle_pieces
        if p.piece_type == "center" and int(p.id) not in corner_piece_ids
    ]

    # Start with corner placements
    current_placements = corner_placements.copy()
    current_score = corner_only_score

    for edge_idx, edge_piece in enumerate(edge_pieces):
        guesses_before_edge = len(all_guesses)
        best_placement = find_best_edge_placement(
            edge_piece=edge_piece,
            piece_shapes=piece_shapes,
            current_placements=current_placements,
            target=target,
            current_score=current_score,
            renderer=renderer,
            scorer=scorer,
            all_guesses=all_guesses,
            all_scores=all_scores,
            slide_positions=slide_positions,
            slide_patience=slide_patience,
        )

        if best_placement:
            current_placements.append(best_placement)

            rendered = renderer.render(current_placements, piece_shapes)
            new_score = scorer.score(rendered, target)
            improvement = new_score - current_score
            current_score = new_score

            all_guesses.append(current_placements.copy())
            all_scores.append(new_score)
            edge_guesses = len(all_guesses) - guesses_before_edge
            print(f"    edge {edge_piece.id}: {best_placement['side']} score {new_score:.0f} ({improvement:+.0f}) [{edge_guesses} guesses]")

            # Early exit: if the first edge piece made things worse than the corner-only
            # baseline, no subsequent edge piece can rescue this layout — skip the rest.
            if edge_idx == 0 and new_score < corner_only_score:
                print(f"    → early exit: edge 1 degraded score, skipping remaining edges")
                break
        else:
            print(f"    edge {edge_piece.id}: no placement found")

    # Place center pieces (simple for now - just random)
    for center_piece in center_pieces:
        piece_id = int(center_piece.id)
        theta = 0
        rotated = rotate_and_crop(piece_shapes[piece_id], theta)
        piece_h, piece_w = rotated.shape

        x = random.uniform(center_piece_margin, target.shape[1] - piece_w - center_piece_margin)
        y = random.uniform(center_piece_margin, target.shape[0] - piece_h - center_piece_margin)

        current_placements.append(
            {"piece_id": piece_id, "x": x, "y": y, "theta": theta}
        )

    # Final render and score
    rendered = renderer.render(current_placements, piece_shapes)
    final_score = scorer.score(rendered, target)

    # Add final to visualizer
    all_guesses.append(current_placements.copy())
    all_scores.append(final_score)

    return {
        "final_score": final_score,
        "final_placements": current_placements,
        "improvement": final_score - corner_only_score,
    }


def find_best_edge_placement(
    edge_piece: PuzzlePiece,
    piece_shapes: Dict[int, np.ndarray],
    current_placements: List[dict],
    target: np.ndarray,
    current_score: float,
    renderer,
    scorer,
    all_guesses,
    all_scores,
    slide_positions: int = 8,
    slide_patience: int = 3,
) -> Optional[dict]:
    """
    Smart edge placement:
    Each rotation has exactly one correct side (derived from primary_edge_rotation).
    Slide immediately for each rotation on its correct side — no center-position gate
    that can falsely reject a good rotation.
    """

    piece_id = int(edge_piece.id)
    height, width = target.shape

    primary = edge_piece.primary_edge_rotation if edge_piece.primary_edge_rotation is not None else 0

    # primary_edge_rotation aligns flat edge to bottom → correct side is "bottom".
    # Each +90° turn rotates the flat edge to the next side.
    rotation_side_pairs = [
        (primary % 360,          "bottom"),
        ((primary + 90) % 360,   "right"),
        ((primary + 180) % 360,  "top"),
        ((primary + 270) % 360,  "left"),
    ]

    best_placement = None
    best_score = -float("inf")  # Always place the piece at its best position

    for rotation, side_name in rotation_side_pairs:
        rotated_mask = rotate_and_crop(piece_shapes[piece_id], rotation)
        piece_h, piece_w = rotated_mask.shape

        if side_name == "right":
            init_x, init_y, axis_type = float(width - piece_w), float((height - piece_h) / 2), "vertical"
        elif side_name == "left":
            init_x, init_y, axis_type = 0.0, float((height - piece_h) / 2), "vertical"
        elif side_name == "bottom":
            init_x, init_y, axis_type = float((width - piece_w) / 2), float(height - piece_h), "horizontal"
        else:  # top
            init_x, init_y, axis_type = float((width - piece_w) / 2), 0.0, "horizontal"

        initial_placement = {
            "piece_id": piece_id,
            "x": init_x,
            "y": init_y,
            "theta": rotation,
            "side": side_name,
            "axis_type": axis_type,
        }

        optimized = slide_along_axis(
            piece_id=piece_id,
            piece_shapes=piece_shapes,
            current_placements=current_placements,
            target=target,
            initial_placement=initial_placement,
            axis_type=axis_type,
            side_name=side_name,
            renderer=renderer,
            scorer=scorer,
            all_guesses=all_guesses,
            all_scores=all_scores,
            num_positions=slide_positions,
            patience=slide_patience,
        )

        if optimized["score"] > best_score:
            best_score = optimized["score"]
            best_placement = optimized["placement"]

    return best_placement


def slide_along_axis(
    piece_id: int,
    piece_shapes: Dict[int, np.ndarray],
    current_placements: List[dict],
    target: np.ndarray,
    initial_placement: dict,
    axis_type: str,
    side_name: str,
    renderer,
    scorer,
    all_guesses,
    all_scores,
    num_positions: int = 8,
    patience: int = 3,
) -> dict:
    """
    Slide piece along its axis (vertical or horizontal) to find best position.
    Uses a simple grid search with 20 test positions.

    IMPORTANT: Adds each test position to visualizer!
    """

    height, width = target.shape
    rotated_mask = rotate_and_crop(
        piece_shapes[piece_id], initial_placement["theta"]
    )
    piece_h, piece_w = rotated_mask.shape

    # Pre-render already-placed pieces once; stamp only the sliding piece per iteration
    static_canvas = renderer.render_static(current_placements, piece_shapes)

    best_placement = initial_placement.copy()
    best_score = -float("inf")
    no_improve_streak = 0

    # Determine search range
    if axis_type == "vertical":
        positions = np.linspace(0, max(0, height - piece_h), num=num_positions)

        for y_pos in positions:
            test_placement = initial_placement.copy()
            test_placement["y"] = float(y_pos)

            rendered = renderer.render_on_base(static_canvas, rotated_mask, int(initial_placement["x"]), int(y_pos))
            score = scorer.score(rendered, target)

            all_guesses.append((current_placements + [test_placement]).copy())
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_placement = test_placement.copy()
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                if no_improve_streak >= patience:
                    break

    else:  # horizontal
        positions = np.linspace(0, max(0, width - piece_w), num=num_positions)

        for x_pos in positions:
            test_placement = initial_placement.copy()
            test_placement["x"] = float(x_pos)

            rendered = renderer.render_on_base(static_canvas, rotated_mask, int(x_pos), int(initial_placement["y"]))
            score = scorer.score(rendered, target)

            all_guesses.append((current_placements + [test_placement]).copy())
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_placement = test_placement.copy()
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                if no_improve_streak >= patience:
                    break

    return {"placement": best_placement, "score": best_score}
