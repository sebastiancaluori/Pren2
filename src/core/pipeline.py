# src/core/pipeline.py

"""
Haupt-Pipeline orchestriert alle Schritte
"""

import os
from dataclasses import dataclass
from pathlib import Path
from time import time, sleep
from typing import Optional

import cv2
import numpy as np

from src.solver.fine_tuner import FineTuner
from src.solver.wall_align_finetuner import WallAlignFinetuner
from src.solver.iterative_solver import IterativeSolver
from src.solver.movement_analyzer import MovementAnalyzer, calculate_movement_data_for_visualizer
from src.solver.piece_analyzer import PieceAnalyzer
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece

from ..solver.guess_generator import GuessGenerator
from ..solver.validation.scorer import PlacementScorer
from ..ui.simulator.guess_renderer import GuessRenderer
from ..utils.logger import setup_logger
from ..vision.camera_loader import CameraLoader
from ..vision.cam_module import (main as runCameraModule, initCameraIfAvailable, stopCamera, isPiCameraAvailable)
from ..vision.mock_puzzle_creator import MockPuzzleGenerator
from .config import Config


@dataclass
class PipelineResult:
    """Ergebnis der Pipeline"""

    success: bool
    duration: float
    message: str
    solution: Optional[dict] = None


class PuzzlePipeline:
    """
    Haupt-Pipeline fuer Puzzle-Loesung

    Schritte:
    1. Bildaufnahme & Preprocessing
    2. Segmentierung
    3. Feature-Extraktion
    4. Puzzle loesen
    5. Validierung
    6. (PREN2) Hardware-Steuerung
    """

    def __init__(
        self, config: Config, show_ui: bool = False, puzzle_dir: str | None = None
    ):
        self.config = config
        self.logger = setup_logger("pipeline")
        self.show_ui = show_ui
        self.puzzle_dir = puzzle_dir  # Directory containing a saved puzzle

        self.guess_generator = GuessGenerator(rotation_step=90)
        self.renderer = None  # Will be created after we have target
        self._init_resolution_components()

    def _init_resolution_components(self):
        """(Re-)initialisiert alle auflösungsabhängigen Komponenten."""
        self.resolution = self.config.resolution
        self.tuning = self.config.tuning.scaled(self.resolution.solver_px_per_mm)
        self.analysis_tuning = self.config.tuning.scaled(self.resolution.effective_analysis_px_per_mm)

        # weight = score_max / target_area_px
        # → perfect coverage always scores exactly score_max, at any resolution
        target_area_px = max(1, self.resolution.a4_width * self.resolution.a4_height)
        fine_area_px = max(1, self.resolution.fine_a4_width * self.resolution.fine_a4_height)
        self._score_weight = self.config.tuning.score_max / target_area_px
        self._finetune_weight = self.config.tuning.score_max / fine_area_px

        self.scorer = PlacementScorer(
            overlap_penalty=self.tuning.overlap_penalty,
            coverage_reward=self.tuning.coverage_reward,
            gap_penalty=self.tuning.gap_penalty,
            weight_multiplier=self._score_weight,
        )

    def run(self) -> PipelineResult:
        """Fuehrt die komplette Pipeline aus"""
        self.logger.info("Pipeline gestartet...")
        start_time = time()
        cam = None

        try:
            cam= initCameraIfAvailable()
            if self.config.hardware.enabled:
                from src.hardware.motion_control.MotionControlCommunication import (
                    wait_for_robot_start,
                )

                self.logger.info(
                    "Phase 0: Warte auf Freigabe durch den Roboter (Hardware-Button)..."
                )
                try:
                    wait_for_robot_start(
                        port=self.config.hardware.serial_port,
                        baudrate=self.config.hardware.baud_rate,
                    )
                except Exception as e:
                    self.logger.error(f"Abbruch in Phase 0: {e}")
                    return PipelineResult(
                        success=False,
                        duration=0,
                        message="Start durch Hardware-Button fehlgeschlagen",
                    )

            # Phase 0.5: Bildaufnahme
            self.logger.info("Phase 0.5: Bildaufnahme")

            _camera_start = time()
            runCameraModule(cam)
            _time_camera = time() - _camera_start
            self.logger.info(f"Kameramodul abgeschossen {_time_camera:.1f}s")


            # Phase 1: Vision
            self.logger.info("Phase 1: Bildverarbeitung")
            _vision_start = time()
            pieces, piece_shapes, piece_shapes_fine, piece_shapes_display, corner_info, puzzle_pieces = (
                self._process_vision()
            )
            _time_vision= time() - _vision_start
            self.logger.info(f"Vision abgeschossen {_time_vision:.1f}s")

            # Phase 2: Solving
            self.logger.info("Phase 2: Puzzle loesen")
            _solve_start = time()
            solution = self._solve_puzzle(
                pieces, piece_shapes, piece_shapes_fine, piece_shapes_display, corner_info, puzzle_pieces
            )
            _solve_elapsed = time() - _solve_start

            # Phase 3: Validation
            self.logger.info("Phase 3: Validierung")
            is_valid = self._validate_solution(solution)

            # Print hardware payload before UI blocks
            self._print_hardware_payload(solution, solve_time=_solve_elapsed)

            # Launch UI even if validation failed (for debugging)
            if self.show_ui and solution:
                self._launch_ui(solution)

            if not is_valid:
                return PipelineResult(
                    success=False,
                    duration=time() - start_time,
                    message="Loesung konnte nicht validiert werden",
                    solution=solution,  # Still return solution for debugging
                )

            # Phase 4: Hardware (nur PREN2)
            if self.config.hardware.enabled:
                self.logger.info("Phase 4: Hardware-Steuerung")
                self._execute_hardware(solution)

            duration = time() - start_time
            self.logger.info(f"✓ Pipeline erfolgreich abgeschlossen ({duration:.2f}s)")

            return PipelineResult(
                success=True,
                duration=duration,
                message="Puzzle erfolgreich geloest",
                solution=solution,
            )

        except Exception as e:
            self.logger.exception(f"Pipeline-Fehler: {e}")
            return PipelineResult(
                success=False, duration=time() - start_time, message=f"Fehler: {str(e)}"
            )
        finally:
            stopCamera(cam)

    def _process_vision(self):
        """Bildverarbeitung - load and analyze puzzle pieces"""
        self.logger.info("  → Bildaufnahme...")

        # Determine which directory to use
        if self.puzzle_dir:
            output_dir = self.puzzle_dir
            self.logger.info(f"  → Loading puzzle from: {output_dir}")
        else:
            output_dir = "data/mock_pieces"
            self.logger.info(f"  → Using default directory: {output_dir}")

        # Kamera-Eingang hat Vorrang vor Mock-Pieces
        if CameraLoader.has_parts_json(output_dir):
            return self._process_vision_camera(output_dir)

        generator = MockPuzzleGenerator(
            output_dir=output_dir,
            num_cuts=self.config.vision.num_cuts,
            a4_width=self.resolution.a4_width,
            a4_height=self.resolution.a4_height,
            a5_width=self.resolution.a5_width,
            a5_height=self.resolution.a5_height,
        )

        # Check if we already have saved pieces
        all_piece_files = list(generator.output_dir.glob("piece_*.png"))
        existing_pieces = [
            p for p in all_piece_files if not p.stem.endswith("_corners")
        ]

        puzzle_pieces = []

        if not existing_pieces or (
            not self.puzzle_dir and self.config.vision.regenerate_mock
        ):
            # Only regenerate if not using a specific puzzle_dir and regenerate flag is set
            self.logger.info("  → Generiere Mock-Puzzle...")

            # Generate new puzzle WITH positions - returns PuzzlePiece objects
            full_image, piece_images, debug_image, puzzle_pieces = (
                generator.generate_puzzle_with_positions()
            )

            # Save debug image
            cv2.imwrite(str(generator.output_dir / "debug_cuts.png"), debug_image)
            self.logger.info(f"  → Mock-Puzzle gespeichert in {generator.output_dir}")
        else:
            self.logger.info(
                f"  → Lade {len(existing_pieces)} existierende Mock-Teile..."
            )

            # A5 dimensions (aus ResolutionConfig skaliert)
            a5_width = self.resolution.a5_width
            a5_height = self.resolution.a5_height
            margin = max(1, int(round(80 * self.resolution.solver_scale)))

            corner_positions = [
                (margin, margin),
                (a5_width - margin, margin),
                (margin, a5_height - margin),
                (a5_width - margin, a5_height - margin),
            ]

            for idx, piece_path in enumerate(sorted(existing_pieces)):
                piece_id = int(piece_path.stem.split("_")[1])

                # Load to get dimensions (Pixel an Aufloesung anpassen)
                img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    piece_h = max(
                        1, int(round(img.shape[0] * self.resolution.solver_scale))
                    )
                    piece_w = max(
                        1, int(round(img.shape[1] * self.resolution.solver_scale))
                    )

                    # Assign corner
                    corner_idx = idx % len(corner_positions)
                    base_x, base_y = corner_positions[corner_idx]

                    # Clamp position
                    x = max(margin, min(a5_width - piece_w - margin, base_x))
                    y = max(margin, min(a5_height - piece_h - margin, base_y))

                    pick_pose = Pose(x=float(x), y=float(y), theta=0.0)
                    piece = PuzzlePiece(pid=str(piece_id), pick=pick_pose)
                    puzzle_pieces.append(piece)

        self.logger.info("  → Segmentierung...")
        self.logger.info("  → Feature-Extraktion...")

        # Coarse copy for solver render+score loop (fast, many iterations)
        piece_ids, piece_shapes = generator.load_pieces_for_solver(
            scale=self.resolution.solver_scale
        )

        # Full-res copy kept separately for fine-tuning
        _, piece_shapes_fine = generator.load_pieces_for_solver(
            scale=self.resolution.finetune_scale
        )

        # Native-resolution copy for display only
        _, piece_shapes_display = generator.load_pieces_for_solver(scale=1.0)

        # Analysis at higher resolution for cleaner corner/edge detection (runs once)
        _, piece_shapes_analysis = generator.load_pieces_for_solver(
            scale=self.resolution.analysis_scale
        )
        PieceAnalyzer.analyze_all_pieces(
            puzzle_pieces, piece_shapes_analysis, tuning=self.analysis_tuning
        )

        # Print analysis results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIECE ANALYSIS RESULTS")
        self.logger.info("=" * 80)

        corner_count = sum(1 for p in puzzle_pieces if p.piece_type == "corner")
        edge_count = sum(1 for p in puzzle_pieces if p.piece_type == "edge")
        center_count = sum(1 for p in puzzle_pieces if p.piece_type == "center")

        self.logger.info(f"\n[STATS] Classification:")
        self.logger.info(f"    Corner pieces: {corner_count}")
        self.logger.info(f"    Edge pieces: {edge_count}")
        self.logger.info(f"    Center pieces: {center_count}")

        # Save visualizations (skip if directory is read-only)
        debug_dir = generator.output_dir / "debug"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            save_debug = True
        except OSError:
            save_debug = False

        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            if piece_id in piece_shapes_analysis:
                self.logger.info(f"\n{piece.summary()}")
                if save_debug:
                    vis = PieceAnalyzer.visualize_corners(piece_shapes_analysis[piece_id], piece)
                    cv2.imwrite(str(debug_dir / f"piece_{piece_id}_analysis.png"), vis)

        self.logger.info(f"\n  → {len(piece_ids)} Teile geladen und analysiert")
        self.logger.info("=" * 80 + "\n")

        return piece_ids, piece_shapes, piece_shapes_fine, piece_shapes_display, {}, puzzle_pieces

    def _process_vision_camera(self, input_dir: str):
        """Kamera-Pfad: lädt Teile aus parts.json + PNG-Masken."""
        loader = CameraLoader(input_dir)
        json_data = loader.load_json()

        self.logger.info(
            f"  → Kamera-Eingabe erkannt: {loader.px_per_mm} px/mm, "
            f"A4 {loader.a4_width_mm}×{loader.a4_height_mm} mm, "
            f"Koordinatenursprung: {loader.origin}"
        )

        for warning in loader.validate():
            self.logger.warning(f"  ⚠  {warning}")

        # Auflösung aus JSON übernehmen und alle abhängigen Komponenten neu initialisieren
        self.config.resolution.native_px_per_mm = loader.px_per_mm
        # solver_px_per_mm darf native nicht überschreiten — Upscaling bringt keine Information.
        self.config.resolution.solver_px_per_mm = min(
            self.config.resolution.solver_px_per_mm,
            self.config.resolution.native_px_per_mm,
        )
        # Quellbereich (A5 in code = physisches A4-Blatt) kommt aus dem Kamera-JSON
        self.config.resolution.a5_width_mm = loader.a4_width_mm
        self.config.resolution.a5_height_mm = loader.a4_height_mm
        # Zielbereich (A4 in code = physisches A5-Blatt) ist fix — kommt nicht aus dem Kamera-JSON
        self._init_resolution_components()

        self.logger.info(f"  → solver_scale={self.resolution.solver_scale:.4f}, "
                         f"score_weight={self.resolution.score_weight_multiplier:.1f}")

        puzzle_pieces = loader.create_puzzle_pieces(self.resolution.solver_px_per_mm)

        piece_ids, piece_shapes = loader.load_pieces_for_solver(
            scale=self.resolution.solver_scale
        )
        _, piece_shapes_fine = loader.load_pieces_for_solver(
            scale=self.resolution.finetune_scale
        )
        _, piece_shapes_display = loader.load_pieces_for_solver(scale=1.0)
        _, piece_shapes_analysis = loader.load_pieces_for_solver(
            scale=self.resolution.analysis_scale
        )

        self.logger.info(f"  → {len(piece_ids)} Kamera-Teile geladen")
        self.logger.info("  → Segmentierung...")
        self.logger.info("  → Feature-Extraktion...")

        PieceAnalyzer.analyze_all_pieces(puzzle_pieces, piece_shapes_analysis, tuning=self.analysis_tuning)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIECE ANALYSIS RESULTS")
        self.logger.info("=" * 80)

        corner_count = sum(1 for p in puzzle_pieces if p.piece_type == "corner")
        edge_count = sum(1 for p in puzzle_pieces if p.piece_type == "edge")
        center_count = sum(1 for p in puzzle_pieces if p.piece_type == "center")

        self.logger.info(f"\n[STATS] Classification:")
        self.logger.info(f"    Corner pieces: {corner_count}")
        self.logger.info(f"    Edge pieces: {edge_count}")
        self.logger.info(f"    Center pieces: {center_count}")

        debug_dir = Path(input_dir) / "debug"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            save_debug = True
        except OSError:
            save_debug = False

        for piece in puzzle_pieces:
            pid = int(piece.id)
            self.logger.info(f"\n{piece.summary()}")
            if save_debug and pid in piece_shapes_analysis:
                vis = PieceAnalyzer.visualize_corners(piece_shapes_analysis[pid], piece)
                cv2.imwrite(str(debug_dir / f"piece_{pid}_analysis.png"), vis)

        self.logger.info(f"\n  → {len(piece_ids)} Teile geladen und analysiert")
        self.logger.info("=" * 80 + "\n")

        return piece_ids, piece_shapes, piece_shapes_fine, piece_shapes_display, {}, puzzle_pieces

    def _solve_puzzle(
        self, pieces, piece_shapes, piece_shapes_fine, piece_shapes_display, piece_corner_info, puzzle_pieces
    ):
        """Puzzle loesen mit iterativem Ansatz"""
        self.logger.info("  → Layout berechnen...")

        # Create surfaces with global coordinate system
        surfaces = self._create_surface_layout(len(pieces))
        target = surfaces["target"]["mask"]
        source = surfaces["source"]["mask"]

        self.logger.info(
            f"  → Global surface: {surfaces['global']['width']}x{surfaces['global']['height']}"
        )
        self.logger.info(
            f"  → Target (A4) at ({surfaces['target']['offset_x']}, {surfaces['target']['offset_y']}): {surfaces['target']['width']}x{surfaces['target']['height']}"
        )
        self.logger.info(
            f"  → Source (A5) at ({surfaces['source']['offset_x']}, {surfaces['source']['offset_y']}): {surfaces['source']['width']}x{surfaces['source']['height']}"
        )

        height, width = target.shape
        self.renderer = GuessRenderer(width=width, height=height)

        iterative_solver = IterativeSolver(
            renderer=self.renderer,
            scorer=self.scorer,
            guess_generator=self.guess_generator,
            tuning=self.tuning,
        )

        # Create initial placements from PuzzlePiece objects
        initial_placements = self._create_initial_placements_from_pieces(puzzle_pieces)

        solution = iterative_solver.solve_iteratively(
            piece_shapes=piece_shapes,
            target=target,
            puzzle_pieces=puzzle_pieces,
            score_max=self.config.tuning.score_max,
            score_accept=self.config.tuning.score_accept,
            initial_corner_count=self.tuning.initial_corner_count,
            max_corners_to_refine=self.tuning.max_corners_to_refine,
            max_iterations=self.tuning.max_iterations,
            px_per_mm=self.resolution.solver_px_per_mm,
        )
        if not solution.success:
            self.logger.warning("  ! Keine gute Loesung gefunden")
        else:
            self.logger.info(f"  ✓ Loesung gefunden mit Score: {solution.score:.2f}")

        # Phase 2b-pre: Wall-align finetune (push corners/edges flush to walls)
        if not self.config.tuning.skip_wall_align and solution.remaining_placements:
            self.logger.info("Phase 2b-pre: Wandausrichtung...")
            dilation_px = int(round(self.config.tuning.gap_dilation_mm * self.resolution.solver_px_per_mm))
            align_target = target
            if dilation_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
                )
                align_target = cv2.dilate(target.astype(np.uint8), kernel).astype(target.dtype)
            wall_aligner = WallAlignFinetuner(
                renderer=self.renderer,
                scorer=self.scorer,
                slide_positions=self.config.tuning.wall_align_slide_positions,
            )
            solution.remaining_placements = wall_aligner.finetune(
                solution.remaining_placements, piece_shapes, align_target
            )

        # Phase 2b: Fine-tuning auf voller Aufloesung
        all_guesses_for_finetune = (
            solution.all_guesses if solution.all_guesses is not None else []
        )
        if self.config.tuning.skip_finetune:
            self.logger.info("Phase 2b: Feinabstimmung übersprungen (skip_finetune=True)")
            ratio = self.resolution.finetune_ratio
            fine_placements = [
                {**p, "x": p["x"] * ratio, "y": p["y"] * ratio}
                for p in solution.remaining_placements
            ]
            fine_score = solution.score
        else:
            self.logger.info("Phase 2b: Feinabstimmung (volle Aufloesung)")
            fine_placements, fine_score = self._finetune_solution(
                solution.remaining_placements,
                piece_shapes_fine,
                target,
                all_guesses_for_finetune,
            )
        fine_placements = self._pull_to_center(fine_placements, piece_shapes_fine)
        # fine_placements are in fine-coordinate space.
        # Convert back to coarse for the visualizer/pipeline dict.
        ratio = self.resolution.finetune_ratio
        coarse_fine_placements = [
            {**p, "x": p["x"] / ratio, "y": p["y"] / ratio} for p in fine_placements
        ]
        solution.score = fine_score

        # All guesses (coarse coords) — fine-tuner already appended its steps
        all_guesses = all_guesses_for_finetune

        # Append final coarse result if not already there
        if coarse_fine_placements:
            if not all_guesses or all_guesses[-1] != coarse_fine_placements:
                all_guesses.append(coarse_fine_placements)

        self.logger.info(f"  → Collected {len(all_guesses)} guesses for visualization")

        # Normalise to weight=1.0 so the visualizer (which uses no weight_multiplier) can match it
        best_score = solution.score / self._score_weight
        best_guess = coarse_fine_placements
        best_guess_index = len(all_guesses) - 1 if all_guesses else 0

        # Populate place_pose with fine coordinates (precision output for robot).
        # The solver's placement (x, y) is the bounding-box top-left of the piece
        # at the given rotation.  The robot grabs each piece at its centroid, so
        # place_pose must also point to the centroid in the target area — computed
        # from the rotated piece shape at the solver position.
        self.logger.info(f"  → Populating place_pose on {len(puzzle_pieces)} pieces")
        piece_lookup = {int(p.id): p for p in puzzle_pieces}
        for placement in fine_placements:
            piece_id = placement["piece_id"]
            if piece_id in piece_lookup:
                piece = piece_lookup[piece_id]
                px, py, theta = placement["x"], placement["y"], placement["theta"]

                # Compute centroid of the rotated shape to get the grab point
                # the robot must reach in the target area.
                shape = piece_shapes_fine.get(piece_id)
                com = (
                    MovementAnalyzer.calculate_piece_com(shape, px, py, theta)
                    if shape is not None
                    else None
                )
                if com is None:
                    # Fallback: bounding-box top-left (pre-fix behaviour)
                    self.logger.warning(
                        f"    Piece {piece_id}: centroid unavailable, falling back to top-left"
                    )
                    com = (px, py)

                # Convert from fine pixels to mm (same unit as pick_pose)
                place_x_mm = com[0] / self.resolution.finetune_px_per_mm
                place_y_mm = com[1] / self.resolution.finetune_px_per_mm
                piece.place_pose = Pose(x=place_x_mm, y=place_y_mm, theta=theta)
                piece.confidence = (
                    1.0 if best_score > self.tuning.score_max else 0.5
                )
                self.logger.debug(
                    f"    Piece {piece_id}: bbox=({px:.1f},{py:.1f}) "
                    f"→ centroid=({place_x_mm:.1f},{place_y_mm:.1f})mm @ {theta:.1f}°"
                )

        # Print movement instructions using PuzzlePiece objects
        self._print_movement_instructions_from_pieces(puzzle_pieces, surfaces)

        return {
            "placements": solution.remaining_placements,
            "score": solution.score,
            "rendered": None,
            "target": target,
            "source": source,
            "surfaces": surfaces,
            "initial_placements": initial_placements,
            "guesses": all_guesses,
            "piece_shapes": piece_shapes,
            "piece_shapes_fine": piece_shapes_fine,
            "piece_shapes_display": piece_shapes_display,
            "display_ratio": self.resolution.native_px_per_mm / self.resolution.solver_px_per_mm,
            "finetune_ratio": self.resolution.finetune_ratio,
            "best_score": best_score,
            "best_guess": best_guess,
            "best_guess_index": best_guess_index,
            "final_fine_placements": fine_placements,
            "renderer": self.renderer,
            "puzzle_pieces": puzzle_pieces,
        }

    def _create_surface_layout(self, num_pieces):
        """
        Erstelle globale Oberflaeche mit Source (A5) und Target (A4) Bereichen.

        Returns dict with:
            - global: {width, height}
            - source: {width, height, offset_x, offset_y, mask}
            - target: {width, height, offset_x, offset_y, mask}
        """

        # A4 target dimensions (aus ResolutionConfig)
        target_width = self.resolution.a4_width
        target_height = self.resolution.a4_height

        # A5 source dimensions (doppelt so breit wie A4)
        source_width = self.resolution.a5_width
        source_height = self.resolution.a5_height

        # Global surface size (side by side with padding)
        padding = max(1, int(round(100 * self.resolution.solver_scale)))
        global_width = source_width + target_width + padding * 3
        global_height = max(source_height, target_height) + padding * 2

        # Calculate offsets (centered vertically)
        source_offset_x = padding
        source_offset_y = (global_height - source_height) // 2

        target_offset_x = source_width + padding * 2
        target_offset_y = (global_height - target_height) // 2

        # Create masks
        target_mask = np.ones((target_height, target_width), dtype=np.uint8)
        source_mask = np.ones((source_height, source_width), dtype=np.uint8)

        # Create global surface representation
        surfaces = {
            "global": {"width": global_width, "height": global_height},
            "source": {
                "width": source_width,
                "height": source_height,
                "offset_x": source_offset_x,
                "offset_y": source_offset_y,
                "mask": source_mask,
            },
            "target": {
                "width": target_width,
                "height": target_height,
                "offset_x": target_offset_x,
                "offset_y": target_offset_y,
                "mask": target_mask,
            },
        }

        return surfaces

    def _create_initial_placements_from_pieces(self, puzzle_pieces):
        """
        Create initial placements from PuzzlePiece objects.
        Uses pixel coordinates from pick_pose.
        """
        initial_placements = []

        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            initial_placements.append(
                {
                    "piece_id": piece_id,
                    "x": piece.pick_pose.x,
                    "y": piece.pick_pose.y,
                    "theta": piece.pick_pose.theta,
                }
            )

        self.logger.info(
            f"  → Created {len(initial_placements)} initial placements from PuzzlePiece objects"
        )
        return initial_placements

    def _print_movement_instructions_from_pieces(self, puzzle_pieces, surfaces):
        """Print movement instructions using PuzzlePiece objects directly."""

        self.logger.info("\n" + "=" * 80)
        self.logger.info("MOVEMENT INSTRUCTIONS (Global Coordinates)")
        self.logger.info("=" * 80)

        source_offset_x = surfaces["source"]["offset_x"]
        source_offset_y = surfaces["source"]["offset_y"]
        target_offset_x = surfaces["target"]["offset_x"]
        target_offset_y = surfaces["target"]["offset_y"]

        for piece in puzzle_pieces:
            if piece.place_pose is None:
                self.logger.warning(f"  ! Piece {piece.id} has no place_pose")
                continue

            # Convert to global coordinates
            # Initial (pick) position: source area coordinates
            initial_global_x = source_offset_x + piece.pick_pose.x
            initial_global_y = source_offset_y + piece.pick_pose.y

            # Final (place) position: target area coordinates
            final_global_x = target_offset_x + piece.place_pose.x
            final_global_y = target_offset_y + piece.place_pose.y

            # Calculate movement
            delta_x = final_global_x - initial_global_x
            delta_y = final_global_y - initial_global_y
            distance = np.sqrt(delta_x**2 + delta_y**2)

            # Rotation change
            rotation_change = (piece.place_pose.theta - piece.pick_pose.theta) % 360

            self.logger.info(f"\nPiece {piece.id}:")
            self.logger.info(f"  Pick:  {piece.pick_pose}")
            self.logger.info(f"  Place: {piece.place_pose}")
            self.logger.info(
                f"  Global pick:  ({initial_global_x:.1f}, {initial_global_y:.1f}) @ {piece.pick_pose.theta:.0f}°"
            )
            self.logger.info(
                f"  Global place: ({final_global_x:.1f}, {final_global_y:.1f}) @ {piece.place_pose.theta:.0f}°"
            )
            self.logger.info(
                f"  Movement: Δx={delta_x:.1f}, Δy={delta_y:.1f}, distance={distance:.1f}"
            )
            if rotation_change != 0:
                self.logger.info(f"  Rotation: {rotation_change:.0f}°")

            # Direction
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                angle = np.degrees(np.arctan2(delta_y, delta_x))
                self.logger.info(f"  Direction: {angle:.1f}° from horizontal")

        self.logger.info("\n" + "=" * 80 + "\n")

    def _pull_to_center(self, placements, piece_shapes_fine):
        """Zieht alle Platzierungen (fine-Koordinaten) um pull_to_center_mm zur Puzzlemitte.

        Richtung basiert auf der Klassifikation des Teils:
        - Edge-Teile (haben 'side'): senkrecht zur Wand einwärts.
        - Corner/Center-Teile: diagonal einwärts, bestimmt durch Bounding-Box-Mitte.
        """
        pull_mm = self.config.tuning.pull_to_center_mm
        if not placements or pull_mm <= 0:
            return placements
        fs = self.resolution.finetune_px_per_mm
        pull_px = pull_mm * fs
        canvas_cx = self.resolution.fine_a4_width / 2.0
        canvas_cy = self.resolution.fine_a4_height / 2.0
        from src.utils.geometry import rotate_and_crop as _rac
        result = []
        for p in placements:
            side = p.get("side")
            if side:
                # Edge piece: pull straight inward, perpendicular to its wall
                if side == "left":
                    shift_x, shift_y = pull_px, 0.0
                elif side == "right":
                    shift_x, shift_y = -pull_px, 0.0
                elif side == "top":
                    shift_x, shift_y = 0.0, pull_px
                else:  # bottom
                    shift_x, shift_y = 0.0, -pull_px
                result.append({**p, "x": p["x"] + shift_x, "y": p["y"] + shift_y})
                continue

            # Corner/center: check all four edges of the bounding box against walls.
            # top-left alone is unreliable for large pieces — a tall bottom piece
            # has its top-left y in the upper half of the canvas.
            shape = piece_shapes_fine.get(p["piece_id"])
            canvas_w = self.resolution.fine_a4_width
            canvas_h = self.resolution.fine_a4_height
            if shape is None:
                result.append(p)
                continue
            rotated = _rac(shape, p["theta"])
            ph, pw = rotated.shape
            thr = pull_px * 8
            x0, y0, x1, y1 = p["x"], p["y"], p["x"] + pw, p["y"] + ph
            new_x, new_y = p["x"], p["y"]
            # Snap to wall + pull_px gap — avoids rounding accumulation from wall-align
            if x0 <= thr:
                new_x = pull_px
            elif x1 >= canvas_w - thr:
                new_x = float(canvas_w - pw) - pull_px
            if y0 <= thr:
                new_y = pull_px
            elif y1 >= canvas_h - thr:
                new_y = float(canvas_h - ph) - pull_px
            result.append({**p, "x": new_x, "y": new_y})

        self.logger.info(f"  Center pull: {pull_px:.1f}px ({pull_mm}mm)")
        return result

    def _finetune_solution(
        self, placements, piece_shapes_fine, target_coarse, all_guesses
    ):
        """Feinabstimmung auf voller Aufloesung.

        Hochskalierte Koordinaten (fine-space), Suchschritte skaliert nach
        finetune_scale. Snapshots nach jeder Teilverbesserung werden als
        Coarse-Koordinaten an all_guesses angehaengt.
        """
        if not placements:
            return placements, 0.0

        ratio = self.resolution.finetune_ratio  # coarse→fine
        coarse_ratio = 1.0 / ratio  # fine→coarse
        fs = self.resolution.finetune_px_per_mm

        # Skaliere Koordinaten in fine-Aufloesung
        fine_placements = [
            {**p, "x": p["x"] * ratio, "y": p["y"] * ratio} for p in placements
        ]

        fine_target = np.ones(
            (self.resolution.fine_a4_height, self.resolution.fine_a4_width),
            dtype=np.uint8,
        )
        fine_renderer = GuessRenderer(
            width=self.resolution.fine_a4_width,
            height=self.resolution.fine_a4_height,
        )
        fine_scorer = PlacementScorer(
            overlap_penalty=self.config.tuning.overlap_penalty,
            coverage_reward=self.config.tuning.coverage_reward,
            gap_penalty=self.config.tuning.gap_penalty,
            weight_multiplier=self._finetune_weight,
        )

        # Dilation auf fine-Aufloesung anwenden (gleiche physikalische mm wie beim Solver)
        dilation_px = int(round(self.config.tuning.gap_dilation_mm * fs))
        if dilation_px > 0:
            import cv2 as _cv2
            kernel = _cv2.getStructuringElement(
                _cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
            )
            piece_shapes_fine = {
                pid: _cv2.dilate(mask.astype(np.uint8), kernel).astype(mask.dtype)
                for pid, mask in piece_shapes_fine.items()
            }

        raw = self.config.tuning  # unscaled defaults
        tuner = FineTuner(
            renderer=fine_renderer,
            scorer=fine_scorer,
            xy_range=max(1, int(round(raw.finetune_xy_range * fs))),
            xy_step=max(1, int(round(raw.finetune_xy_step * fs))),
            theta_range=raw.finetune_theta_range,
            theta_step=raw.finetune_theta_step,
            max_passes=raw.finetune_max_passes,
        )
        return tuner.finetune(
            fine_placements,
            piece_shapes_fine,
            fine_target,
            all_guesses=all_guesses,
            coarse_ratio=coarse_ratio,
        )

    def _validate_solution(self, solution):
        """Loesung validieren"""
        self.logger.info("  → Geometrie pruefen...")

        # Check if we have a solution
        if not solution or "score" not in solution:
            return False

        if solution["score"] < -10000:
            self.logger.warning(f"  ! Score zu niedrig: {solution['score']}")
            return False

        self.logger.info("  → Konfidenz berechnen...")
        # Better confidence calculation
        max_possible_score = 90000
        confidence = min(
            100, max(0, (solution["score"] + 10000) / max_possible_score * 100)
        )
        self.logger.info(f"  → Konfidenz: {confidence:.1f}%")

        return True

    def _print_hardware_payload(self, solution, solve_time: float = None):
        """Print the raw values that would be sent to the robot."""
        puzzle_pieces = (solution or {}).get("puzzle_pieces", [])
        if not puzzle_pieces:
            return
        px_per_mm = self.resolution.solver_px_per_mm
        a5_w_mm = self.resolution.a5_width_mm

        print("\n" + "=" * 60)
        if solve_time is not None:
            print(f"HARDWARE PAYLOAD  (solved in {solve_time:.1f}s)")
        else:
            print("HARDWARE PAYLOAD (raw values as sent to robot)")
        print("=" * 60)
        print(f"{'Piece':<8} {'pick_x_mm':>12} {'pick_y_mm':>12} {'place_x_mm':>12} {'place_y_mm':>12} {'rotation_deg':>14}")
        print("-" * 60)

        for p in puzzle_pieces:
            pick_x = a5_w_mm - (p.pick_pose.x / px_per_mm)
            pick_y = p.pick_pose.y / px_per_mm
            if p.place_pose:
                place_x = p.place_pose.y
                place_y = p.place_pose.x
                rotation = (90 - p.place_pose.theta) % 360
                if rotation > 180:
                    rotation -= 360
            else:
                place_x = place_y = rotation = 0.0
            print(f"{p.id:<8} {pick_x:>12.2f} {pick_y:>12.2f} {place_x:>12.2f} {place_y:>12.2f} {rotation:>14.1f}")
        print("=" * 60 + "\n")

    def _execute_hardware(self, solution):
        from src.hardware.motion_control.MotionControlCommunication import send_to_robot

        self.logger.info("  → Initialisiere UART-Verbindung...")

        puzzle_pieces = solution.get("puzzle_pieces", [])

        if not puzzle_pieces:
            self.logger.error("  ! Keine Puzzleteile zur Übertragung gefunden.")
            return

        self.logger.info(f"  → Sende {len(puzzle_pieces)} Teile an den Roboter...")

        success = send_to_robot(
            pieces=puzzle_pieces,
            port=self.config.hardware.serial_port,
            baudrate=self.config.hardware.baud_rate,
            pick_px_per_mm=self.config.resolution.solver_px_per_mm,
            timeout=5.0,
        )

        if success:
            self.logger.info("  ✓ Hardware-Befehl erfolgreich quittiert (ACK OK).")
            self.logger.info("  → Roboter führt Bewegungen nun aus.")
        else:
            self.logger.error("  X Fehler bei der Kommunikation mit dem STM32.")

    def _launch_ui(self, solution):
        """Launch Kivy UI to visualize the solution."""
        self.logger.info("🎬 Starte Visualisierung...")

        # Calculate movement data for best solution
        movement_data = None
        if solution.get("puzzle_pieces") and solution.get("best_guess"):
            movement_data = calculate_movement_data_for_visualizer(solution)

        solver_data = {
            "guesses": solution["guesses"],
            "piece_shapes": solution["piece_shapes"],
            "piece_shapes_fine": solution.get("piece_shapes_fine"),
            "piece_shapes_display": solution.get("piece_shapes_display"),
            "finetune_ratio": solution.get("finetune_ratio", 1.0),
            "display_ratio": solution.get("display_ratio", 1.0),
            "target": solution["target"],
            "source": solution["source"],
            "surfaces": solution["surfaces"],
            "initial_placements": solution["initial_placements"],
            "best_score": solution["best_score"],
            "best_guess": solution.get("best_guess"),
            "best_guess_index": solution.get("best_guess_index", 0),
            "renderer": solution["renderer"],
            "puzzle_pieces": solution["puzzle_pieces"],
            "movement_data": movement_data,
        }

        self.logger.info(
            f"  → Passing {len(solution['guesses'])} guesses to visualizer"
        )
        if movement_data:
            num_movements = len(movement_data.get("movements", {}))
            self.logger.info(f"  → Calculated movement data for {num_movements} pieces")

        from src.ui.simulator.solver_visualizer import SolverVisualizerApp

        app = SolverVisualizerApp(solver_data)
        app.run()
