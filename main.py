import sys
from pathlib import Path

# Parse and remove our flags before Kivy intercepts sys.argv
_regenerate = "--regenerate" in sys.argv
if _regenerate:
    sys.argv.remove("--regenerate")

_six_pieces = "--six-pieces" in sys.argv
if _six_pieces:
    sys.argv.remove("--six-pieces")

_no_camera = "--no-camera" in sys.argv
if _no_camera:
    sys.argv.remove("--no-camera")

from src.core.pipeline import PuzzlePipeline
from src.core.config import Config
from src.utils.logger import setup_logger
from src.vision import cam_module

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():

    # Logger initialisieren
    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("PREN Puzzle Solver gestartet")
    logger.info("=" * 60)

    try:

        config = Config()
        if _regenerate:
            config.vision.regenerate_mock = True
        if _six_pieces:
            config.vision.num_cuts = 3

        input_dir = project_root / "input"
        puzzle_dir = None

        if _no_camera:
            logger.info("--no-camera: Kameramodul wird übersprungen, verwende vorhandene Eingabe.")
        else:
            logger.info("Starte Kameramodul...")
            cam_module.main()
            logger.info("Kameramodul abgeschlossen")

        if (input_dir / "parts.json").exists():
            puzzle_dir = str(input_dir)
            logger.info(f"Kamera-Eingabe erkannt: {input_dir}")

        pipeline = PuzzlePipeline(config, show_ui=True, puzzle_dir=puzzle_dir)
        result = pipeline.run()
        
        if result.success:
            logger.info("✓ Puzzle erfolgreich gelöst!")
            logger.info(f"Zeit: {result.duration:.2f}s")
        else:
            logger.error("✗ Puzzle konnte nicht gelöst werden")

        # Print raw hardware payload (what would be sent to the robot)
        puzzle_pieces = (result.solution or {}).get("puzzle_pieces", [])
        px_per_mm = config.resolution.solver_px_per_mm
        if puzzle_pieces:
            print("\n" + "=" * 60)
            print("HARDWARE PAYLOAD (raw values as sent to robot)")
            print("=" * 60)
            print(f"{'Piece':<8} {'pick_x_mm':>12} {'pick_y_mm':>12} {'place_x_mm':>12} {'place_y_mm':>12} {'rotation_deg':>14}")
            print("-" * 60)
            for p in puzzle_pieces:
                pick_x = p.pick_pose.x / px_per_mm
                pick_y = p.pick_pose.y / px_per_mm
                if p.place_pose:
                    place_x = p.place_pose.x
                    place_y = p.place_pose.y
                    rotation = p.place_pose.theta % 360
                    if rotation > 180:
                        rotation -= 360
                else:
                    place_x = place_y = rotation = 0.0
                print(f"{p.id:<8} {pick_x:>12.2f} {pick_y:>12.2f} {place_x:>12.2f} {place_y:>12.2f} {rotation:>14.1f}")
            print("=" * 60)
            
    except KeyboardInterrupt:
        logger.info("\nProgramm durch Benutzer abgebrochen")
    except Exception as e:
        logger.exception(f"Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
