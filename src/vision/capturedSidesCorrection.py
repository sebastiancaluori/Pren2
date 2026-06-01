import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE_PATH = "input.png"
OUTPUT_IMAGE_PATH = "output_corrected.png"

DEBUG_CONTOURS_IMAGE_PATH = "debug_contours.png"
DEBUG_CORRECTION_POINTS_IMAGE_PATH = "debug_correction_points.png"
DEBUG_CORRECTED_OVERLAY_IMAGE_PATH = "debug_corrected_overlay.png"
DEBUG_FINAL_MASK_IMAGE_PATH = "debug_final_mask.png"

# Kamerahöhe über der A4-Fläche in mm. Nur für lokale Ausführung von Bedeutung
CAMERA_HEIGHT_MM = 500.0
# Höhe der Teile in mm.
PIECE_HEIGHT = 6.0
# Kleine Konturen ignorieren.
MIN_CONTOUR_AREA_PX = 2000

# Binärwerte für das aktuelle Inputbild:
# weisse Flächen auf schwarzem Hintergrund.
BACKGROUND_VALUE = 0     # schwarz
FOREGROUND_VALUE = 255   # weiss


# ============================================================
# GEOMETRY CORRECTION
# ============================================================

def move_points_towards_center(
    height_mm: float,
    points: np.ndarray,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Verschiebt mehrere Punkte gleichzeitig Richtung Zentrum.

    Formel:
        new_point = point + (center - point) * (3 / height_mm)

    Beispiel:
        height_mm = 500
        point = (130, 0)
        center = (0, 0)

        factor = 3 / 500 = 0.006
        new_x = 130 + (0 - 130) * 0.006
        new_x = 129.22
    """

    if height_mm == 0:
        raise ValueError("height_mm darf nicht 0 sein.")

    factor = PIECE_HEIGHT / height_mm

    points_float = points.astype(np.float32)
    center_array = np.array(center, dtype=np.float32)

    corrected_points = points_float + (center_array - points_float) * factor

    return np.round(corrected_points).astype(np.int32)


# ============================================================
# IMAGE LOADING / PREPARATION
# ============================================================

def load_binary_image(image_path: str) -> np.ndarray:
    """
    Lädt ein Bild als Graustufenbild und stellt sicher,
    dass es binär ist: schwarz/weiss.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")

    _, binary_image = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary_image


# ============================================================
# CONTOUR DETECTION
# ============================================================

def find_part_contours(
    binary_image: np.ndarray,
    center: tuple[int, int],
) -> list[np.ndarray]:
    """
    Findet weisse Flächen auf schwarzem Hintergrund.

    OpenCV findet helle Objekte. Da die Flächen bereits weiss sind,
    wird hier nicht invertiert.
    """

    search_image = binary_image.copy()

    all_contours, _ = cv2.findContours(
        search_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    filtered_contours = []

    for contour in all_contours:
        area_px = cv2.contourArea(contour)

        if area_px < MIN_CONTOUR_AREA_PX:
            continue

        filtered_contours.append(contour)

    print(f"Konturen total gefunden: {len(all_contours)}")
    print(f"Konturen nach Filter >= {MIN_CONTOUR_AREA_PX}px: {len(filtered_contours)}")

    # ------------------------------------------------------------
    # Debugbild: Konturen + Zentrum
    # ------------------------------------------------------------

    debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Gefilterte Konturen rot einzeichnen.
    cv2.drawContours(
        debug_image,
        filtered_contours,
        contourIdx=-1,
        color=(0, 0, 255),  # rot
        thickness=2
    )

    # Zentrum einzeichnen.
    center_x, center_y = center

    cv2.circle(
        debug_image,
        center=(int(center_x), int(center_y)),
        radius=8,
        color=(255, 0, 0),  # blau
        thickness=-1
    )

    cv2.drawMarker(
        debug_image,
        position=(int(center_x), int(center_y)),
        color=(0, 255, 255),  # gelb
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2
    )

    cv2.putText(
        debug_image,
        text=f"Zentrum ({int(center_x)}, {int(center_y)})",
        org=(int(center_x) + 10, int(center_y) - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    # cv2.imwrite(DEBUG_CONTOURS_IMAGE_PATH, debug_image)
    # print(f"Debugbild gespeichert: {DEBUG_CONTOURS_IMAGE_PATH}")

    return filtered_contours


# ============================================================
# POINT SELECTION
# ============================================================

def should_correct_contour_point(
    point: tuple[int, int],
    center: tuple[int, int],
    binary_image: np.ndarray,
) -> bool:
    """
    Entscheidungsregel für das aktuelle Bild:
        - weisse Flächen
        - schwarzer Hintergrund

    Vom Konturpunkt aus wird ein Pixel Richtung Zentrum geprüft.

    Wenn dieses Pixel weiss ist:
        Der Punkt wird korrigiert.

    Wenn dieses Pixel schwarz ist:
        Der Punkt bleibt unverändert.
    """

    x, y = point
    center_x, center_y = center

    vector_x = center_x - x
    vector_y = center_y - y

    # Punkt liegt exakt im Zentrum.
    if vector_x == 0 and vector_y == 0:
        return False

    # Grober Ein-Pixel-Schritt Richtung Zentrum.
    # Werte sind -1, 0 oder 1.
    step_x = int(np.sign(vector_x))
    step_y = int(np.sign(vector_y))

    check_x = x + step_x
    check_y = y + step_y

    height_px, width_px = binary_image.shape[:2]

    # Nicht ausserhalb des Bildes lesen.
    if check_x < 0 or check_x >= width_px:
        return False

    if check_y < 0 or check_y >= height_px:
        return False

    pixel_value = binary_image[check_y, check_x]

    # weiss -> korrigieren
    # schwarz -> unverändert lassen
    if pixel_value == FOREGROUND_VALUE:
        return True

    return False


# ============================================================
# SINGLE CONTOUR CORRECTION
# ============================================================

def correct_contour(
    working_image: np.ndarray,
    binary_image: np.ndarray,
    debug_image: np.ndarray,
    contour: np.ndarray,
    height_mm: float,
    center: tuple[int, int],
    contour_index: int,
) -> None:
    """
    Korrigiert eine einzelne weisse Fläche auf schwarzem Hintergrund.

    Ablauf:
        1. Komplette Konturpunkte auslesen.
        2. Jeden Punkt prüfen.
        3. Nur gewählte Punkte Richtung Zentrum korrigieren.
        4. Alte weisse Fläche schwarz löschen.
        5. Neue korrigierte weisse Fläche gefüllt einzeichnen.

    Debug:
        - Grün: Punkt wird korrigiert
        - Blau: Punkt bleibt unverändert
        - Rot: alte Kontur
        - Gelb: neue Kontur
    """

    points = contour.reshape(-1, 2)
    corrected_points_all = points.copy()

    indices_to_correct = []

    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])

        should_correct = should_correct_contour_point(
            point=(x, y),
            center=center,
            binary_image=binary_image,
        )

        if should_correct:
            indices_to_correct.append(index)

            # Debug: korrigierte Punkte grün.
            cv2.circle(
                debug_image,
                center=(x, y),
                radius=1,
                color=(0, 255, 0),
                thickness=-1
            )
        else:
            # Debug: unveränderte Punkte blau.
            cv2.circle(
                debug_image,
                center=(x, y),
                radius=1,
                color=(255, 0, 0),
                thickness=-1
            )

    print(f"Kontur {contour_index}: Punkte total: {len(points)}")
    print(f"Kontur {contour_index}: Punkte zu korrigieren: {len(indices_to_correct)}")

    if len(indices_to_correct) == 0:
        return

    indices_to_correct = np.array(indices_to_correct, dtype=np.int32)

    points_to_correct = points[indices_to_correct]

    corrected_points = move_points_towards_center(
        height_mm=height_mm,
        points=points_to_correct,
        center=center,
    )

    corrected_points_all[indices_to_correct] = corrected_points

    corrected_contour = corrected_points_all.reshape(-1, 1, 2)

    # Alte komplette weisse Fläche löschen -> schwarz.
    cv2.drawContours(
        working_image,
        [contour],
        contourIdx=-1,
        color=BACKGROUND_VALUE,
        thickness=cv2.FILLED
    )

    # Neue korrigierte weisse Fläche zeichnen -> weiss.
    cv2.drawContours(
        working_image,
        [corrected_contour],
        contourIdx=-1,
        color=FOREGROUND_VALUE,
        thickness=cv2.FILLED
    )

    # Debug: alte Kontur rot.
    cv2.drawContours(
        debug_image,
        [contour],
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=1
    )

    # Debug: neue Kontur gelb.
    cv2.drawContours(
        debug_image,
        [corrected_contour],
        contourIdx=-1,
        color=(0, 255, 255),
        thickness=1
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

def correct_binary_image_file(
    input_path: str,
    output_path: str,
    height_mm: float,
) -> None:
    """
    Test-/Standalone-Funktion:
    Lädt ein Bild von Datei, korrigiert es und speichert es wieder.
    """

    binary_image = load_binary_image(input_path)

    corrected_image = calculate_puzzle_piece_shape_without_sides(
        binary_image=binary_image,
        height_mm=height_mm,
    )

    success = cv2.imwrite(output_path, corrected_image)

    if not success:
        raise IOError(f"Bild konnte nicht gespeichert werden: {output_path}")

    print(f"Korrigiertes Bild gespeichert unter: {output_path}")


def calculate_puzzle_piece_shape_without_sides(
    binary_image: np.ndarray,
    height_mm: float,
) -> np.ndarray:
    """
    Korrigiert ein bereits geladenes Binärbild.

    Input:
        binary_image:
            np.ndarray, Graustufenbild mit 0 und 255.
            Erwartung:
                Hintergrund = 0  schwarz
                Fläche      = 255 weiss

        height_mm:
            Kamerahöhe / Geometriehöhe in mm.

    Output:
        Korrigiertes Binärbild als np.ndarray.
    """

    # Sicherheitskopie, damit das Originalbild nicht verändert wird.
    binary_image = binary_image.copy()

    # Sicherstellen, dass es wirklich binär ist.
    _, binary_image = cv2.threshold(
        binary_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    working_image = binary_image.copy()

    center_x = working_image.shape[1] // 2
    center_y = working_image.shape[0] // 2
    center = (center_x, center_y)

    print(f"Zentrum in Pixelkoordinaten: {center}")

    contours = find_part_contours(
        binary_image=binary_image,
        center=center,
    )

    print(f"Gefundene Konturen: {len(contours)}")


    debug_correction = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    cv2.drawMarker(
        debug_correction,
        position=center,
        color=(0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2
    )

    for contour_index, contour in enumerate(contours):
        print(f"Korrigiere Kontur {contour_index}")

        correct_contour(
            working_image=working_image,
            binary_image=binary_image,
            debug_image=debug_correction,
            contour=contour,
            height_mm=height_mm,
            center=center,
            contour_index=contour_index,
        )

    # Ergebnis wieder sauber binär machen.
    _, working_image = cv2.threshold(
        working_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Kleine Lücken schliessen.
    kernel = np.ones((3, 3), np.uint8)

    working_image = cv2.morphologyEx(
        working_image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    _, working_image = cv2.threshold(
        working_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Debugbilder weiterhin speichern.
    # cv2.imwrite(DEBUG_FINAL_MASK_IMAGE_PATH, working_image)
    # cv2.imwrite(DEBUG_CORRECTION_POINTS_IMAGE_PATH, debug_correction)

    # debug_overlay = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    changed_pixels = cv2.absdiff(binary_image, working_image)
    # debug_overlay[changed_pixels > 0] = (0, 0, 255)
    # cv2.imwrite(DEBUG_CORRECTED_OVERLAY_IMAGE_PATH, debug_overlay)

    return working_image



# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    correct_binary_image_file(
        input_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_IMAGE_PATH,
        height_mm=CAMERA_HEIGHT_MM,
    )
