"""
Piece analyzer that detects corners and straight edges,
then populates PuzzlePiece objects with the analysis data.

This is a drop-in replacement for src.solver.piece_analyzer.PieceAnalyzer
that extends the existing corner detection with straight edge detection.
"""

import cv2
import numpy as np
from typing import Dict, List

from src.solver.corner_detector import detect_corners
from src.solver.edge_detector import calculate_edge_rotations, detect_edges
from src.utils.puzzle_piece import PuzzlePiece


class PieceAnalyzer:
    """
    Analyzes puzzle pieces to detect corners and straight edges.
    Enriches PuzzlePiece objects with analysis data in-place.

    Usage:
        # Create pieces
        pieces = [PuzzlePiece(pid="0", pick=Pose(x=100, y=200, theta=0)), ...]

        # Analyze all pieces
        PieceAnalyzer.analyze_all_pieces(pieces, piece_shapes)

        # Now pieces have .piece_type, .corners, .edges, etc populated
    """

    @staticmethod
    def analyze_all_pieces(puzzle_pieces: List[PuzzlePiece],
                          piece_shapes: Dict[int, np.ndarray],
                          tuning=None) -> None:
        """
        Analyze all pieces and populate their analysis data IN-PLACE.

        Args:
            puzzle_pieces: List of PuzzlePiece objects to enrich
            piece_shapes: Dict mapping piece_id to binary mask
        """
        print("\n[ANALYSIS] Analyzing all pieces for corners and edges...")

        for piece in puzzle_pieces:
            piece_id = int(piece.id)

            if piece_id not in piece_shapes:
                print(f"  ⚠️  Piece {piece_id} not found in piece_shapes")
                continue

            mask = piece_shapes[piece_id]

            # Analyze this piece
            PieceAnalyzer.analyze_piece(piece, mask, tuning=tuning)

            # Print summary
            if piece.piece_type == "corner":
                print(f"  [+] Piece {piece_id}: CORNER ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")
            elif piece.piece_type == "edge":
                print(f"  [+] Piece {piece_id}: EDGE ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")
            else:
                print(f"  [o] Piece {piece_id}: CENTER ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")

        print(f"\n[RE-EVALUATION] Checking corner piece categorization...")

        # Collect all pieces with any corner data (regardless of current classification)
        def _corner_score(piece):
            if not piece.corners:
                return (0.0, 0, 0.0)
            best = max(c.quality for c in piece.corners)
            total = sum(c.quality for c in piece.corners)
            return (total, len(piece.corners), best)

        # Always re-evaluate using all pieces with corner data — not just pre-classified corners.
        # This corrects misclassifications regardless of whether count is 4, >4, or <4.
        all_with_corners = [p for p in puzzle_pieces if p.corners]

        corner_scores = []
        for piece in all_with_corners:
            best_corner_quality = max(c.quality for c in piece.corners)
            total_corner_quality = sum(c.quality for c in piece.corners)
            best_edge_quality = max((e.quality for e in piece.edges), default=0.0)
            # Penalize pieces where edge dominates over corners
            corner_dominance = best_corner_quality - best_edge_quality
            corner_scores.append((piece, best_corner_quality, total_corner_quality, len(piece.corners), corner_dominance))

        corner_scores.sort(key=lambda x: (x[4], x[2], x[3], x[1]), reverse=True)

        print(f"      Corner piece rankings (all pieces with corner detections):")
        for i, (piece, best_qual, total_qual, count, dominance) in enumerate(corner_scores):
            print(f"        {i+1}. Piece {piece.id} ({piece.piece_type}): {count} corners, total_quality={total_qual:.2f}, best={best_qual:.2f}, dominance={dominance:.2f}")

        true_corners = corner_scores[:4]
        demoted_pieces = corner_scores[4:]

        for piece, _, _, _, _ in true_corners:
            if piece.piece_type != "corner":
                old_type = piece.piece_type
                piece.piece_type = "corner"
                print(f"        📈 Promoted piece {piece.id}: {old_type} → corner")

        for piece, _, _, _, _ in demoted_pieces:
            if piece.piece_type == "corner":
                piece.piece_type = "edge"
                print(f"        📉 Demoted piece {piece.id}: corner → edge")

        if len(all_with_corners) < 4:
            print(f"  ⚠️  Only {len(all_with_corners)} pieces with corner detections — puzzle may not solve correctly")
        else:
            print(f"  ✅ Final corner pieces: {[int(p[0].id) for p in true_corners]}")

        # Final summary
        final_corners = [p for p in puzzle_pieces if p.piece_type == "corner"]
        final_edges = [p for p in puzzle_pieces if p.piece_type == "edge"]
        final_centers = [p for p in puzzle_pieces if p.piece_type == "center"]

        print(f"\n[FINAL SUMMARY]")
        print(f"  Corner pieces: {len(final_corners)} - {[int(p.id) for p in final_corners]}")
        print(f"  Edge pieces:   {len(final_edges)} - {[int(p.id) for p in final_edges]}")
        print(f"  Center pieces: {len(final_centers)} - {[int(p.id) for p in final_centers]}")

    @staticmethod
    def analyze_piece(piece: PuzzlePiece, mask: np.ndarray, tuning=None) -> None:
        """
        Analyze a single piece and populate its analysis fields IN-PLACE.

        Args:
            piece: PuzzlePiece object to enrich
            mask: Binary mask of the piece (0s and 1s, or 0-255)
        """
        try:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            # Calculate piece center and geometry
            M = cv2.moments(mask)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                h, w = mask.shape
                cx, cy = w / 2, h / 2

            piece.center = (cx, cy)
            piece.area = float(M['m00'])

            # Find contour for perimeter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                piece.perimeter = cv2.arcLength(contours[0], True)

            # Detect corners
            corner_kwargs = {}
            if tuning:
                corner_kwargs = dict(
                    angle_tolerance=tuning.corner_angle_tolerance,
                    min_straightness=tuning.corner_min_straightness,
                    min_edge_length=tuning.corner_min_edge_length,
                    min_quality=tuning.corner_min_quality,
                    contour_epsilon=tuning.corner_contour_epsilon,
                    max_overhang=tuning.corner_max_overhang,
                    min_extent=tuning.corner_min_extent,
                )
            corner_data_list = detect_corners(mask, (cx, cy), **corner_kwargs)
            piece.corners = corner_data_list
            piece.has_corner = len(corner_data_list) > 0

            # Set primary corner rotation (best corner)
            if corner_data_list:
                best_corner = max(corner_data_list, key=lambda c: c.quality)
                piece.primary_corner_rotation = best_corner.rotation_to_align

            # Detect straight edges (excluding corner edges)
            edge_kwargs = {}
            if tuning:
                edge_kwargs = dict(
                    min_edge_length=tuning.edge_min_length,
                    min_edge_straightness=tuning.edge_min_straightness,
                    min_edge_score=tuning.edge_min_score,
                    contour_epsilon=tuning.edge_contour_epsilon,
                )
            edge_data_list = detect_edges(mask, (cx, cy), corner_data_list, **edge_kwargs)
            piece.edges = edge_data_list
            piece.has_straight_edge = len(edge_data_list) > 0

            # Calculate edge rotations for all cardinal directions
            if edge_data_list:
                piece.edge_rotations = calculate_edge_rotations(edge_data_list)

                # Set primary edge rotation (best edge, aligned to bottom)
                best_edge = max(edge_data_list, key=lambda e: e.quality)
                piece.primary_edge_rotation = best_edge.rotations_to_align.get('bottom')

            # Classify piece type - SMARTER LOGIC
            num_corners = len(corner_data_list)
            num_edges = len(edge_data_list)
            corner_thresh = tuning.classify_corner_threshold if tuning else 0.85
            edge_thresh = tuning.classify_edge_threshold if tuning else 0.8

            if (num_corners > 0 and corner_data_list[0].quality > corner_thresh):
                piece.piece_type = "corner"
                piece.analysis_confidence = corner_data_list[0].quality
            elif (num_edges > 0 and edge_data_list[0].quality > edge_thresh):
                piece.piece_type = "edge"
                piece.analysis_confidence = edge_data_list[0].quality
            elif (num_corners > 0 and num_edges > 0):
                best_corner = corner_data_list[0].quality
                best_edge = edge_data_list[0].quality
                corner_meets_thresh = best_corner >= corner_thresh
                edge_meets_thresh = best_edge >= edge_thresh
                if corner_meets_thresh and not edge_meets_thresh:
                    piece.piece_type = "corner"
                    piece.analysis_confidence = best_corner
                elif edge_meets_thresh and not corner_meets_thresh:
                    piece.piece_type = "edge"
                    piece.analysis_confidence = best_edge
                elif best_corner >= best_edge:
                    piece.piece_type = "corner"
                    piece.analysis_confidence = best_corner
                else:
                    piece.piece_type = "edge"
                    piece.analysis_confidence = best_edge
            elif (num_corners > 0):
                piece.piece_type = "corner"
                piece.analysis_confidence = corner_data_list[0].quality
            elif (num_edges > 0):
                piece.piece_type = "edge"
                piece.analysis_confidence = edge_data_list[0].quality
            else:
                piece.piece_type = "center"
                piece.analysis_confidence = 1.0



        except Exception as e:
            print(f"  ⚠️  Error analyzing piece {piece.id}: {e}")
            import traceback
            traceback.print_exc()

            # Set safe defaults
            piece.piece_type = "unknown"
            piece.analysis_confidence = 0.0

    @staticmethod
    def visualize_corners(mask: np.ndarray, piece: PuzzlePiece) -> np.ndarray:
        """
        Create visualization showing detected corners and edges on a piece.
        Compatible with your existing visualization code.
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = [255, 255, 255]

        # Draw piece center
        if piece.center:
            cx, cy = int(piece.center[0]), int(piece.center[1])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)

        # Draw corners (green)
        for i, corner in enumerate(piece.corners):
            radius = 12 if i == 0 else 8
            thickness = -1 if i == 0 else 2
            cv2.circle(vis, corner.position, radius, (0, 255, 0), thickness)

            if piece.center:
                cv2.line(vis, (int(piece.center[0]), int(piece.center[1])),
                        corner.position, (0, 255, 0), 2)

            # Add quality text
            text_pos = (corner.position[0] + 15, corner.position[1])
            cv2.putText(vis, f"{corner.quality:.2f}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw straight edges (cyan)
        for i, edge in enumerate(piece.edges):
            cv2.line(vis, edge.start_point, edge.end_point, (255, 255, 0), 3)
            cv2.circle(vis, edge.midpoint, 6, (255, 255, 0), -1)

            # Add quality text
            text_pos = (edge.midpoint[0] + 15, edge.midpoint[1])
            cv2.putText(vis, f"{edge.quality:.2f}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Add text with piece info
        y_pos = 30
        text = f"Type: {piece.piece_type.upper()}"
        cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_pos += 30
        if piece.corners:
            text = f"Corners: {len(piece.corners)}"
            if piece.primary_corner_rotation is not None:
                text += f", Rot: {piece.primary_corner_rotation:.1f}"
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

        if piece.edges:
            text = f"Edges: {len(piece.edges)}"
            if piece.primary_edge_rotation is not None:
                text += f", Rot: {piece.primary_edge_rotation:.1f}"
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return vis
