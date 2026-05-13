import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece
from src.vision.cut_patterns import (
    generate_sharp_cut,
    generate_square_cut,
    generate_wavy_cut,
)


class MockPuzzleGenerator:
    """Generate realistic mock puzzle pieces for testing."""

    def __init__(
        self,
        output_dir: str = "data/mock_pieces",
        num_cuts: int | None = None,
        a4_width: int = 420,
        a4_height: int = 594,
        a5_width: int = 840,
        a5_height: int = 594,
    ):
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        # Dimensionen (Default entspricht 2 px/mm). Werden ueber ResolutionConfig
        # skaliert, damit die gesamte Pipeline bei niedrigerer Aufloesung laeuft.
        self.a4_width = a4_width
        self.a4_height = a4_height
        self.a5_width = a5_width
        self.a5_height = a5_height

        self.num_cuts = num_cuts if num_cuts is not None else random.choice([2, 3])

        # Store piece positions (will be filled during save_pieces)
        self.piece_positions = {}

        print(f"Generating puzzle with {self.num_cuts} cuts")

    def generate_puzzle(self) -> tuple:
        """
        Generate a puzzle with 2-3 cuts (wavy or sharp).

        Returns:
            (full_image, piece_images, debug_image)
        """
        self.cleanup_old_pieces()
        # Create full A4 image (white paper)
        full_image = np.ones((self.a4_height, self.a4_width, 3), dtype=np.uint8) * 255

        # Add some texture to make it look like paper
        noise = np.random.randint(
            -10, 10, (self.a4_height, self.a4_width, 3), dtype=np.int16
        )
        full_image = np.clip(full_image.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        # Generate cuts
        cuts = []
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green

        if self.num_cuts == 2:
            # Generate two cuts (can be vertical/horizontal or diagonal)
            cuts.append(self._generate_random_cut(orientation="vertical"))
            cuts.append(self._generate_random_cut(orientation="horizontal"))
        else:  # 3 cuts
            # For 3 cuts, we can do various configurations
            # Option 1: Two verticals, one horizontal
            # Option 2: Two horizontals, one vertical
            # Option 3: One vertical, one horizontal, one diagonal

            config = "vhh"  # random.choice(["vvh", "vhh", "vhd"])

            if config == "vvh":
                # Two vertical cuts dividing into thirds, one horizontal
                left_x = self.a4_width // 3 + random.randint(-20, 20)
                right_x = 2 * self.a4_width // 3 + random.randint(-20, 20)

                cuts.append(
                    self._generate_cut_between_points(
                        (left_x, 0), (left_x, self.a4_height)
                    )
                )
                cuts.append(
                    self._generate_cut_between_points(
                        (right_x, 0), (right_x, self.a4_height)
                    )
                )
                cuts.append(self._generate_random_cut(orientation="horizontal"))

            elif config == "vhh":
                # Two horizontal cuts dividing into thirds, one vertical
                top_y = self.a4_height // 3 + random.randint(-20, 20)
                bottom_y = 2 * self.a4_height // 3 + random.randint(-20, 20)

                cuts.append(self._generate_random_cut(orientation="vertical"))
                cuts.append(
                    self._generate_cut_between_points(
                        (0, top_y), (self.a4_width, top_y)
                    )
                )
                cuts.append(
                    self._generate_cut_between_points(
                        (0, bottom_y), (self.a4_width, bottom_y)
                    )
                )

            else:  # vhd
                # One vertical, one horizontal, one diagonal
                cuts.append(self._generate_random_cut(orientation="vertical"))
                cuts.append(self._generate_random_cut(orientation="horizontal"))
                cuts.append(self._generate_random_cut(orientation="diagonal"))

        # Draw cuts on image for visualization
        debug_image = full_image.copy()
        for i, cut in enumerate(cuts):
            cv2.polylines(debug_image, [cut], False, colors[i], 3)

        # Create masks for each piece using the actual cut lines
        piece_masks = self._create_piece_masks_from_cuts(cuts)

        # Extract individual pieces
        piece_images = []
        for i, mask in enumerate(piece_masks):
            # Apply mask to get piece
            piece = cv2.bitwise_and(full_image, full_image, mask=mask)

            # Find bounding box
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                padding = 5
                x_min = max(0, x_min - padding)
                x_max = min(self.a4_width, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(self.a4_height, y_max + padding)

                # Crop piece
                cropped_piece = piece[y_min : y_max + 1, x_min : x_max + 1]
                cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

                piece_images.append(
                    {
                        "id": i,
                        "image": cropped_piece,
                        "mask": cropped_mask,
                        "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
                    }
                )

        return full_image, piece_images, debug_image

    def generate_puzzle_with_positions(self) -> tuple:
        """
        Generate a puzzle and assign initial positions in A5 source area.
        Places pieces in corners to avoid overlap.
        Saves pieces to disk automatically.

        Returns:
            (full_image, piece_images, debug_image, puzzle_pieces)
            where puzzle_pieces is a list of PuzzlePiece objects
        """

        # Generate puzzle as normal
        full_image, piece_images, debug_image = self.generate_puzzle()

        # Save pieces (this applies random rotation to each piece)
        piece_paths = self.save_pieces(piece_images)

        # Define corner positions (with margin from edges)
        # A5 is 840 x 594
        margin = 20  # Distance from corner

        # Four corners: top-left, top-right, bottom-left, bottom-right
        corner_positions = [
            (margin, margin),  # Top-left
            (self.a5_width + margin, margin),  # Top-right
            (margin, self.a5_height - margin),  # Bottom-left
            (self.a5_width + margin, self.a5_height - margin),  # Bottom-right
        ]

        # Create PuzzlePiece objects for each saved piece
        puzzle_pieces = []

        for idx, piece_path in enumerate(piece_paths):
            # Extract piece ID from filename
            piece_id = int(piece_path.stem.split("_")[1])

            # Load the SAVED piece (which is already rotated)
            saved_image = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)

            if saved_image is None:
                continue

            # Assign to a corner (cycle through corners)
            corner_idx = idx % len(corner_positions)
            base_x, base_y = corner_positions[corner_idx]

            # Get piece dimensions
            piece_h, piece_w = saved_image.shape[:2]

            # Clamp position to keep piece fully inside A5
            x = max(margin, min(self.a5_width - piece_w - margin, base_x))
            y = max(margin, min(self.a5_height - piece_h - margin, base_y))

            # Create PuzzlePiece with initial pick pose in pixels
            pick_pose = Pose(x=float(x), y=float(y), theta=0.0)
            piece = PuzzlePiece(pid=str(piece_id), pick=pick_pose)

            puzzle_pieces.append(piece)

            print(
                f"Piece {piece_id}: corner {corner_idx + 1}/4, position ({x:.1f}px, {y:.1f}px)"
            )

        return full_image, piece_images, debug_image, puzzle_pieces

    def _generate_random_cut(self, orientation="vertical") -> np.ndarray:
        """Generate a random cut with specified orientation."""
        if orientation == "vertical":
            center_x = self.a4_width // 2 + random.randint(-30, 30)
            angle = random.randint(-15, 15)
            offset = int(np.tan(np.radians(angle)) * self.a4_height / 2)

            return self._generate_cut_between_points(
                (center_x - offset, 0), (center_x + offset, self.a4_height)
            )

        elif orientation == "horizontal":
            center_y = self.a4_height // 2 + random.randint(-30, 30)
            angle = random.randint(-15, 15)
            offset = int(np.tan(np.radians(angle)) * self.a4_width / 2)

            return self._generate_cut_between_points(
                (0, center_y - offset), (self.a4_width, center_y + offset)
            )

        else:  # diagonal
            # Diagonal from one corner area to opposite corner area, but extended
            start_x = random.randint(
                -self.a4_width // 8, self.a4_width // 4
            )  # Start outside
            start_y = random.randint(
                -self.a4_height // 8, self.a4_height // 4
            )  # Start outside
            end_x = random.randint(
                3 * self.a4_width // 4, self.a4_width + self.a4_width // 8
            )  # End outside
            end_y = random.randint(
                3 * self.a4_height // 4, self.a4_height + self.a4_height // 8
            )  # End outside

            # Generate the cut line between these extended points
            cut_line = self._generate_cut_between_points(
                (start_x, start_y), (end_x, end_y)
            )

            # The cut line will naturally extend beyond the puzzle bounds due to extended start/end points
            return cut_line

    def _generate_cut_between_points(self, start: tuple, end: tuple) -> np.ndarray:
        """Generate a cut (wavy, sharp zigzag, or square wave) between two points."""
        cut_type = random.choice(["wavy", "sharp", "square"])

        if cut_type == "wavy":
            return self.generate_wavy_cut(
                start,
                end,
                num_waves=random.randint(3, 6),
                amplitude=random.randint(20, 40),
            )
        elif cut_type == "sharp":
            return self.generate_sharp_cut(
                start,
                end,
                num_angles=random.randint(4, 8),
                amplitude=random.randint(30, 60),
            )
        else:  # square
            return self.generate_square_cut(
                start,
                end,
                num_rectangles=random.randint(3, 6),
                amplitude=random.randint(25, 50),
            )

    def generate_wavy_cut(self, start, end, num_waves=3, amplitude=30):
        """Generate a wavy cut line between two points."""
        return generate_wavy_cut(start, end, num_waves, amplitude)

    def generate_sharp_cut(self, start, end, num_angles=5, amplitude=40):
        """Generate a sharp zigzag cut line between two points."""
        return generate_sharp_cut(start, end, num_angles, amplitude)

    def generate_square_cut(self, start, end, num_rectangles=4, amplitude=35):
        """Generate a square wave cut line between two points."""
        return generate_square_cut(start, end, num_rectangles, amplitude)

    def save_pieces(self, piece_images: list) -> list:
        """Save piece images to disk with random rotations and return file paths."""
        saved_paths = []

        for piece_data in piece_images:
            piece_id = piece_data["id"]
            image = piece_data["image"]
            mask = piece_data["mask"]

            # RANDOMLY ROTATE THE PIECE
            random_angle = random.randint(0, 359)
            print(f"Rotating piece {piece_id} by {random_angle} degrees")

            # Rotate both image and mask
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, random_angle, 1.0)

            # Calculate new bounding box after rotation
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Rotate image and mask
            rotated_image = cv2.warpAffine(
                image, M, (new_w, new_h), borderValue=(255, 255, 255)
            )
            rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))

            # Create RGBA image
            bgra = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2BGRA)

            # Set alpha channel from rotated mask
            bgra[:, :, 3] = rotated_mask

            # Save
            filepath = self.output_dir / f"piece_{piece_id}.png"
            cv2.imwrite(str(filepath), bgra)
            saved_paths.append(filepath)

            print(f"Saved piece {piece_id} to {filepath} (rotated {random_angle}°)")

        return saved_paths

    def load_pieces_for_solver(
        self, piece_paths: list = None, scale: float = 1.0
    ) -> tuple:  # type: ignore
        """
        Load saved pieces and prepare them for the solver.

        Args:
            piece_paths: Optional list of paths to piece images.
            scale: Resize factor applied to each mask (e.g. 0.5 halves both dims).
                   Use ResolutionConfig.scale so pieces match the solver canvas.

        Returns:
            (piece_ids, piece_shapes_dict)
        """
        if piece_paths is None:
            # Load all pieces from output directory
            piece_paths = sorted(self.output_dir.glob("piece_*.png"))

        piece_shapes = {}
        piece_ids = []

        for i, path in enumerate(piece_paths):
            # Load image with alpha channel
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

            if image is None:
                continue

            # Extract alpha channel as mask
            if image.shape[2] == 4:
                mask = image[:, :, 3]
            else:
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Normalize to 0 and 1
            mask = (mask > 127).astype(np.uint8)

            # An die Aufloesung anpassen, damit Stuecke zur Ziel-Canvas passen
            if scale != 1.0:
                new_h = max(1, int(round(mask.shape[0] * scale)))
                new_w = max(1, int(round(mask.shape[1] * scale)))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            piece_shapes[i] = mask
            piece_ids.append(i)

        return piece_ids, piece_shapes

    def _create_piece_masks_from_cuts(self, cuts: list) -> list:
        """Create binary masks for each piece using actual cut lines."""
        # Create a mask with all cuts drawn
        cut_image = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)

        # Draw all cuts as barriers with THICKER lines to ensure separation
        for cut in cuts:
            cv2.polylines(cut_image, [cut], False, 255, 6)

        # Invert so cuts are black (barriers)
        cut_image = 255 - cut_image

        # Optional: Apply morphological closing to ensure cuts are fully connected
        kernel = np.ones((5, 5), np.uint8)
        cut_image = cv2.morphologyEx(cut_image, cv2.MORPH_CLOSE, kernel)

        # Find all connected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cut_image, connectivity=4
        )

        # Create masks for each region (skip background label 0)
        masks = []

        # Calculate expected minimum area (should be roughly total_area / expected_pieces)
        total_area = self.a4_height * self.a4_width
        expected_pieces = self.num_cuts + 1  # 2 cuts = 3 pieces, 3 cuts = 4 pieces
        min_area_threshold = (
            total_area / expected_pieces
        ) * 0.2  # At least 20% of expected size

        print(f"\nDEBUG: Found {num_labels - 1} regions")

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            print(
                f"  Region {label}: area = {area}, min_threshold = {min_area_threshold:.0f}"
            )

            # Only keep masks with sufficient area
            if area > min_area_threshold:
                mask = (labels == label).astype(np.uint8) * 255
                masks.append(mask)
                print(f"    [+] Kept region {label}")
            else:
                print(f"    [-] Rejected region {label} (too small)")

        print(
            f"\nCreated {len(masks)} pieces from {self.num_cuts} cuts (expected {expected_pieces})"
        )

        # Add assertion to catch unexpected piece counts
        if len(masks) != expected_pieces:
            print(
                f"[!] WARNING: Expected {expected_pieces} pieces but got {len(masks)}!"
            )

            # Save debug image to see what's happening
            debug_path = self.output_dir / "debug_cut_regions.png"
            debug_img = cv2.cvtColor(cut_image, cv2.COLOR_GRAY2BGR)
            for i in range(1, num_labels):
                color = np.random.randint(0, 255, 3).tolist()
                debug_img[labels == i] = color
            cv2.imwrite(str(debug_path), debug_img)
            print(f"  Saved debug image to {debug_path}")

        return masks

    def cleanup_old_pieces(self):
        """Remove all existing piece files before generating new puzzle."""
        for old_piece in self.output_dir.glob("piece_*.png"):
            old_piece.unlink()
            print(f"Removed old piece: {old_piece.name}")
