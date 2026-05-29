import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from src.utils.geometry import rotate_and_crop
from ...solver.validation.scorer import PlacementScorer
from ...ui.simulator.guess_renderer import GuessRenderer
from .movement_renderer import MovementRenderer


class SolverVisualizer(BoxLayout):
    def __init__(self, solver_data, embedded=False, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.solver_data = solver_data
        self.embedded = embedded  # New parameter
        self.current_guess_index = 0
        self.is_running = False
        self.speed = 0.05
        self.show_movements = False

        # Only add background if not embedded
        if not self.embedded:
            with self.canvas.before:
                Color(0.9, 0.9, 0.9, 1)
                self.bg = Rectangle(size=self.size, pos=self.pos)
                self.bind(size=self._update_bg, pos=self._update_bg)

        # Top: Image display
        self.image_widget = Image(size_hint_y=0.75)
        self.add_widget(self.image_widget)

        # Bottom: Controls with better styling
        controls = BoxLayout(
            size_hint_y=0.25, orientation="vertical", padding=20, spacing=15
        )

        # Status label with better styling
        self.status_label = Label(
            text="Ready to visualize",
            size_hint_y=0.2,
            color=(0.2, 0.2, 0.2, 1),  # Dark grey text
            font_size="16sp",
            bold=True,
        )
        controls.add_widget(self.status_label)

        # Main controls - First row (Navigation & Playback)
        button_row1 = BoxLayout(orientation="horizontal", size_hint_y=0.4, spacing=10)

        # Navigation buttons
        self.first_button = Button(
            text="️First",
            background_color=(0.3, 0.5, 0.8, 1),  # Blue
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.first_button.bind(on_press=self.go_to_first)
        button_row1.add_widget(self.first_button)

        self.back_button = Button(
            text="Back",
            background_color=(0.7, 0.5, 0.2, 1),  # Orange
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.back_button.bind(on_press=self.go_back)
        button_row1.add_widget(self.back_button)

        self.step_button = Button(
            text="Next",
            background_color=(0.4, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.step_button.bind(on_press=self.step_guess)
        button_row1.add_widget(self.step_button)

        # Playback controls
        self.start_button = Button(
            text="Play",
            background_color=(0.3, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.start_button.bind(on_press=self.start_visualization)
        button_row1.add_widget(self.start_button)

        self.pause_button = Button(
            text="Pause",
            background_color=(0.8, 0.5, 0.2, 1),  # Orange-red
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.pause_button.bind(on_press=self.pause_visualization)
        button_row1.add_widget(self.pause_button)

        self.best_button = Button(
            text="Best",
            background_color=(0.8, 0.2, 0.8, 1),  # Purple
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.best_button.bind(on_press=self.show_best)
        button_row1.add_widget(self.best_button)

        self.show_movement_button = Button(
            text="Movement",
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.show_movement_button.bind(on_press=self.toggle_movement_view)
        button_row1.add_widget(self.show_movement_button)

        controls.add_widget(button_row1)

        self.add_widget(controls)

        # Show initial state with source+target
        self._show_initial_state()

    def _update_bg(self, instance, value):
        """Update background rectangle when widget size/position changes."""
        self.bg.pos = instance.pos
        self.bg.size = instance.size

    def go_to_first(self, instance):
        """Go to the first guess."""
        if self.is_running:
            self.pause_visualization(None)

        self.current_guess_index = 0
        self._show_initial_state()

    def go_back(self, instance):
        """Go back one guess."""
        if self.is_running:
            self.pause_visualization(None)

        if self.current_guess_index > 0:
            self.current_guess_index -= 1
            if self.current_guess_index == 0:
                self._show_initial_state()
            else:
                # Show the previous guess
                self._show_specific_guess(self.current_guess_index - 1)

    def _show_specific_guess(self, guess_index):
        """Show a specific guess by index."""
        if 0 <= guess_index < len(self.solver_data["guesses"]):
            guess = self.solver_data["guesses"][guess_index]

            renderer = self.solver_data["renderer"]
            scorer = PlacementScorer(
                overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5
            )

            rendered = renderer.render(guess, self.solver_data["piece_shapes"])
            score = scorer.score(rendered, self.solver_data["target"])
            rendered_color = self._render_guess_color(guess, debug=True)

            if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
                display = self._create_source_target_visualization(
                    rendered_color,
                    self.solver_data["puzzle_pieces"],
                    self.solver_data["surfaces"],
                )
            else:
                display = self._create_visualization(
                    rendered_color, self.solver_data["target"]
                )

            self._update_image(display)

            is_best = score >= self.solver_data["best_score"]
            best_marker = "BEST!" if is_best else ""

            self.status_label.text = (
                f"Guess {guess_index + 1}/{len(self.solver_data['guesses'])} | "
                f"Score: {score:.2f}{best_marker}"
            )

    def step_guess(self, instance):
        """Show the next guess."""
        if self.current_guess_index < len(self.solver_data["guesses"]):
            guess = self.solver_data["guesses"][self.current_guess_index]

            renderer = self.solver_data["renderer"]
            scorer = PlacementScorer(
                overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5
            )

            rendered = renderer.render(guess, self.solver_data["piece_shapes"])
            score = scorer.score(rendered, self.solver_data["target"])

            rendered_color = self._render_guess_color(guess, debug=True)

            if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
                display = self._create_source_target_visualization(
                    rendered_color,
                    self.solver_data["puzzle_pieces"],
                    self.solver_data["surfaces"],
                )
            else:
                display = self._create_visualization(
                    rendered_color, self.solver_data["target"]
                )

            self._update_image(display)

            is_best = score >= self.solver_data["best_score"]
            best_marker = " NEW BEST!" if is_best else ""

            self.status_label.text = (
                f"Guess {self.current_guess_index + 1}/{len(self.solver_data['guesses'])} | "
                f"Score: {score:.2f}{best_marker}"
            )

            self.current_guess_index += 1

    def show_best(self, instance):
        """Show the best solution found."""
        # Pause if running
        if self.is_running:
            self.pause_visualization(None)

        # Get the pre-calculated best guess
        best_guess = self.solver_data.get("final_fine_placements") or self.solver_data.get("best_guess")
        best_guess_index = self.solver_data.get("best_guess_index", 0)
        best_score = self.solver_data.get("best_score", 0)

        if best_guess is None:
            self.status_label.text = "No best solution found!"
            return

        rendered_color = self._render_guess_color(best_guess, debug=False)

        # Create side-by-side visualization
        if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
            display = self._create_source_target_visualization(
                rendered_color,
                self.solver_data["puzzle_pieces"],
                self.solver_data["surfaces"],
            )
        else:
            print("⚠️  Using fallback visualization for BEST solution")
            display = self._create_visualization(
                rendered_color, self.solver_data["target"]
            )

        # Update display
        self._update_image(display)

        self.status_label.text = (
            f" BEST SOLUTION | Guess #{best_guess_index + 1} | Score: {best_score:.2f}"
        )

        # Update current index
        self.current_guess_index = best_guess_index

    def _show_initial_state(self):
        """Show initial state with original positions and empty target."""
        if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
            # Create empty guess for target area
            empty_guess = []

            # Create empty rendered color (same size as target)
            target = self.solver_data["target"]
            empty_rendered = np.zeros(
                (target.shape[0], target.shape[1], 3), dtype=np.uint8
            )

            # Show source+target visualization with empty target
            display = self._create_source_target_visualization(
                empty_rendered,
                self.solver_data["puzzle_pieces"],
                self.solver_data["surfaces"],
            )
        else:
            print("⚠️  Using fallback initial state visualization")
            # Fallback to old target-only view
            target = self.solver_data["target"]
            display = (target * 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

            # Draw grid lines
            for i in range(0, display.shape[0], 100):
                cv2.line(display, (0, i), (display.shape[1], i), (50, 50, 50), 1)
            for i in range(0, display.shape[1], 100):
                cv2.line(display, (i, 0), (i, display.shape[0]), (50, 50, 50), 1)

        self._update_image(display)
        self.status_label.text = (
            f"Initial State | {len(self.solver_data['guesses'])} guesses to test"
        )

    def _create_source_target_visualization(
        self, rendered_color, puzzle_pieces, surfaces, show_movements=None
    ):
        """Create side-by-side visualization with optional COM dots."""
        # Use instance variable if not specified
        if show_movements is None:
            show_movements = getattr(self, "show_movements", False)

        print(f"📐 Creating visualization (movements: {show_movements})...")

        # Scale the entire canvas to the best available display resolution
        display_shapes, vis_scale = self._best_display_shapes()

        def _s(v):
            return int(round(v * vis_scale))

        global_width = _s(surfaces["global"]["width"])
        global_height = _s(surfaces["global"]["height"])
        canvas = np.full((global_height, global_width, 3), 200, dtype=np.uint8)

        source_offset_x = _s(surfaces["source"]["offset_x"])
        source_offset_y = _s(surfaces["source"]["offset_y"])
        target_offset_x = _s(surfaces["target"]["offset_x"])
        target_offset_y = _s(surfaces["target"]["offset_y"])

        # Fill areas white, draw borders
        source_w, source_h = _s(surfaces["source"]["width"]), _s(surfaces["source"]["height"])
        target_w, target_h = _s(surfaces["target"]["width"]), _s(surfaces["target"]["height"])

        canvas[
            source_offset_y : source_offset_y + source_h,
            source_offset_x : source_offset_x + source_w,
        ] = [255, 255, 255]
        canvas[
            target_offset_y : target_offset_y + target_h,
            target_offset_x : target_offset_x + target_w,
        ] = [255, 255, 255]

        cv2.rectangle(
            canvas,
            (source_offset_x, source_offset_y),
            (source_offset_x + source_w - 1, source_offset_y + source_h - 1),
            (0, 200, 0),
            4,
        )
        cv2.rectangle(
            canvas,
            (target_offset_x, target_offset_y),
            (target_offset_x + target_w - 1, target_offset_y + target_h - 1),
            (0, 150, 255),
            4,
        )

        piece_colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
        ]

        # Render original positions in source area
        for piece in puzzle_pieces:
            piece_id = int(piece.id)

            shapes_to_use = display_shapes
            if piece_id in shapes_to_use:
                shape = shapes_to_use[piece_id]
                rotated = self._rotate_shape(shape, piece.pick_pose.theta)
                # pick_pose is in solver pixels; scale to vis coordinates
                x = int(_s(piece.pick_pose.x)) - rotated.shape[1] // 2 + source_offset_x
                y = int(_s(piece.pick_pose.y)) - rotated.shape[0] // 2 + source_offset_y
                color = piece_colors[piece_id % len(piece_colors)]
                faded_color = tuple(int(c * 0.7) for c in color)

                self._place_shape_color_global(canvas, rotated, x, y, faded_color)
                cv2.putText(
                    canvas,
                    f"P{piece_id}",
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    canvas,
                    f"P{piece_id}",
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )

        # Overlay target area
        target_region = canvas[
            target_offset_y : target_offset_y + target_h,
            target_offset_x : target_offset_x + target_w,
        ]
        rc = rendered_color
        if rc.shape[:2] != target_region.shape[:2]:
            rc = cv2.resize(rc, (target_region.shape[1], target_region.shape[0]), interpolation=cv2.INTER_AREA)
        mask = np.any(rc > 0, axis=2)
        target_region[mask] = rc[mask]

        # Add COM dots if requested
        if show_movements and "movement_data" in self.solver_data:
            canvas = self._add_com_dots(canvas)

        return canvas

    def _rotate_shape(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees and crop to tight bounding box."""
        return rotate_and_crop(shape, angle)

    def _render_guess_color(self, guess, debug=False):
        """Render using the highest-resolution shapes available (native > fine > solver).

        If final_fine_placements is available and matches the passed guess, render
        directly from fine coordinates so the display exactly matches what the robot
        receives — no coarse round-trip.
        """
        fine_placements = self.solver_data.get("final_fine_placements")
        fine_shapes = self.solver_data.get("piece_shapes_fine")
        if fine_placements is not None and fine_shapes is not None and guess is fine_placements:
            # Render directly from fine coords: ratio = display_px / fine_px
            finetune_ratio = self.solver_data.get("finetune_ratio", 1.0)
            display_ratio = self.solver_data.get("display_ratio", finetune_ratio)
            fine_to_display = display_ratio / finetune_ratio
            target = self.solver_data["target"]
            solver_h, solver_w = target.shape
            disp_w = int(round(solver_w * display_ratio))
            disp_h = int(round(solver_h * display_ratio))
            disp_renderer = GuessRenderer(width=disp_w, height=disp_h)
            scaled = [{**p, "x": p["x"] * fine_to_display, "y": p["y"] * fine_to_display} for p in fine_placements]
            shapes = self.solver_data.get("piece_shapes_display") or fine_shapes
            if debug:
                return disp_renderer.render_debug(scaled, shapes)
            return disp_renderer.render_color(scaled, shapes)

        shapes, ratio = self._best_display_shapes()
        target = self.solver_data["target"]
        solver_h, solver_w = target.shape
        disp_w = int(round(solver_w * ratio))
        disp_h = int(round(solver_h * ratio))
        disp_renderer = GuessRenderer(width=disp_w, height=disp_h)
        scaled_guess = [{**p, "x": p["x"] * ratio, "y": p["y"] * ratio} for p in guess]
        if debug:
            return disp_renderer.render_debug(scaled_guess, shapes)
        return disp_renderer.render_color(scaled_guess, shapes)

    def _best_display_shapes(self):
        """Return (piece_shapes, ratio_vs_solver) for the best available resolution."""
        display = self.solver_data.get("piece_shapes_display")
        if display:
            return display, self.solver_data.get("display_ratio", 1.0)
        fine = self.solver_data.get("piece_shapes_fine")
        if fine:
            return fine, self.solver_data.get("finetune_ratio", 1.0)
        return self.solver_data["piece_shapes"], 1.0

    def _place_shape_color_global(
        self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, color: tuple
    ):
        """Place colored shape on global canvas using TOP-LEFT corner positioning."""
        h, w = shape.shape[:2]

        # Calculate bounds - x,y is TOP-LEFT in global coordinates
        y1 = max(0, y)
        y2 = min(canvas.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(canvas.shape[1], x + w)

        # Calculate corresponding region in shape
        shape_y1 = max(0, -y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, -x)
        shape_x2 = shape_x1 + (x2 - x1)

        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            shape_region = shape[shape_y1:shape_y2, shape_x1:shape_x2]
            mask = shape_region > 0

            for c in range(3):
                canvas[y1:y2, x1:x2, c][mask] = color[c]

    def start_visualization(self, instance):
        """Start the visualization."""
        if not self.is_running:
            self.is_running = True
            self.start_button.text = "Playing"
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)

    def pause_visualization(self, instance):
        """Pause the visualization."""
        if self.is_running:
            self.is_running = False
            self.start_button.text = "Play"
            if hasattr(self, "clock_event"):
                self.clock_event.cancel()

    def auto_step(self, dt):
        """Automatically step through guesses."""
        if self.current_guess_index < len(self.solver_data["guesses"]):
            self.step_guess(None)
        else:
            self.pause_visualization(None)
            self.status_label.text = (
                f"✅ DONE! Best score: {self.solver_data['best_score']:.2f}"
            )

    def _create_visualization(self, rendered_color, target):
        """Fallback: Create visualization - rendered_color is already in target space."""
        display = rendered_color.copy()

        # Draw target outline (which should match the canvas now)
        h, w = display.shape[:2]

        # Draw border around entire canvas (which IS the target)
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (255, 255, 100), 2)

        # Draw grid
        for i in range(0, h, 100):
            cv2.line(display, (0, i), (w, i), (80, 80, 80), 1)
        for i in range(0, w, 100):
            cv2.line(display, (i, 0), (i, h), (80, 80, 80), 1)

        return display

    def _update_image(self, array: np.ndarray):
        """Update the image widget with a numpy array."""
        max_w, max_h = 1400, 700
        h, w = array.shape[:2]
        scale = min(max_w / w, max_h / h, 4.0)
        if scale != 1.0:
            interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
            array = cv2.resize(
                array,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=interp,
            )

        # Flip vertically (Kivy uses bottom-left origin)
        display = np.flipud(array)

        # Create texture
        texture = Texture.create(
            size=(display.shape[1], display.shape[0]), colorfmt="rgb"
        )
        texture.blit_buffer(display.tobytes(), colorfmt="rgb", bufferfmt="ubyte")

        self.image_widget.texture = texture

    def _add_com_dots(self, canvas):
        """Add COM dots, movement arrows, and movement values to existing canvas."""
        renderer = MovementRenderer(self.solver_data)
        return renderer.add_com_dots(canvas)

    def _draw_movement_summary(self, canvas, movement_summary):
        """Draw detailed movement summary at bottom of canvas - clean format for robot."""
        renderer = MovementRenderer(self.solver_data)
        renderer.draw_movement_summary(canvas, movement_summary)

    def _draw_movement_legend(self, canvas):
        """Draw legend explaining movement visualization symbols."""
        renderer = MovementRenderer(self.solver_data)
        renderer.draw_movement_legend(canvas)

    def add_movement_button_to_init(self):
        """Add this button to your button_row1 in __init__"""
        self.show_movement_button = Button(
            text="🎯 Movement",
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.show_movement_button.bind(on_press=self.toggle_movement_view)
        # button_row1.add_widget(self.show_movement_button)
        self.show_movements = False  # Track state

    def toggle_movement_view(self, instance):
        """Toggle movement visualization and jump to best solution."""
        if not self.solver_data.get("movement_data"):
            self.status_label.text = "No movement data available!"
            return

        self.show_movements = not self.show_movements
        self.show_movement_button.text = (
            "🎯 Hide Movement" if self.show_movements else "🎯 Movement"
        )
        self.show_movement_button.background_color = (
            (0.8, 0.4, 0.2, 1) if self.show_movements else (0.2, 0.6, 0.8, 1)
        )
        self.show_best(None)  # Jump to best solution


# app
class SolverVisualizerApp(App):
    def __init__(self, solver_data, **kwargs):
        super().__init__(**kwargs)
        self.solver_data = solver_data

    def build(self):
        visualizer = SolverVisualizer(self.solver_data)
        visualizer.add_movement_button_to_init()
        return visualizer
