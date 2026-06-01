# cam_module.py

# tbd Kameraeinstellungen von umgebung ableiten, benötigt in jedem fall zwei aufnahmen

# Kamera-/Datei-Eingang, ArUco-basierte A4-Entzerrung, Teile-Segmentierung
# und Export der Algorithmus-Eingaben.

import json
import shutil
import time
from pathlib import Path
from .capturedSidesCorrection import calculate_puzzle_piece_shape_without_sides

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except (ImportError, ModuleNotFoundError):
    Picamera2 = None


# ============================================================
# PROJEKT / DATEIEN
# ============================================================

# Projektwurzel. Von src/vision/cam_module.py zwei Ebenen nach oben.
PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)  # src/vision/cam_module.py -> project root

# Zielordner für die Daten, die an den Puzzle-Algorithmus gehen.
# In diesen Ordner werden parts.json und die Teilmasken geschrieben.
DESTINATION_TO_ALGO_INPUT_FOLDER = PROJECT_ROOT / "input"

# Eingabebild, falls IMAGE_SOURCE = "file" oder keine Pi Camera vorhanden ist.
INPUT_IMAGE_PATH = PROJECT_ROOT / "src" / "vision" / "1.png"

# Name der JSON-Datei, welche der Solver später einliest.
ALGO_INPUT_JSON_FILENAME = "parts.json"
# Präfix für die Teilmasken im Algorithmus-Input. Beispiel: piece_0.png.
ALGO_INPUT_MASK_PREFIX = "piece_"
# Der Input-Ordner wird bei jedem Start des Cam-Moduls geleert.
# Neue Algorithmus-Dateien werden nur nach gültiger Erkennung geschrieben.


# ============================================================
# BILDEINGABE
# ============================================================

# "camera" = neues Bild mit Pi Camera 3 aufnehmen
# "file"   = bestehendes Bild von Datei laden
IMAGE_SOURCE = "camera"

# Aufnahmebreite der Pi Camera. Grösser = mehr Details, aber langsamere Verarbeitung.
IMAGE_WIDTH = 4608
# Aufnahmehöhe der Pi Camera. Muss zur gewünschten Kameraauflösung passen.
IMAGE_HEIGHT = 2592


STARTUP_WAIT_SECONDS = 3.0

# Bild um 90 Grad im Uhrzeigersinn drehen, falls die Kamera mechanisch verdreht ist.
ROTATE_90_CLOCKWISE = False
# Bild um 180 Grad drehen. Nie gleichzeitig mit ROTATE_90_CLOCKWISE aktivieren.
ROTATE_180 = False
# Initial Kameraeinstellungen für den Start innerhalb der Pipeline
INIT_CAMERA_CONTROLS_FOR_PIPELINE = {
    # Automatische Belichtung. False = manueller ExposureTime-Wert wird verwendet.
    "AeEnable": False,
    # Automatischer Weissabgleich. False = ColourGains unten werden verwendet.
    "AwbEnable": False,
    # Belichtungszeit in Mikrosekunden. Höher = heller, zu hoch = A4 kann ausbrennen.
    "ExposureTime": 6000,  # Mikrosekunden
    # Analoge Verstärkung. Höher = heller, aber mehr Bildrauschen.
    "AnalogueGain": 1.0,
    # Rot-/Blau-Verstärkung für Weissabgleich. Höherer Wert = jeweiliger Farbkanal stärker.
    "ColourGains": (1.5, 1.5),
}
# Manuelle Kameraeinstellungen für Tests gegen Überbelichtung.
# Ohne Unterbeleuchtung, Kunstlicht, exposureTime rund 8000
# Mit Unterbeleuchtung, Kunstlicht 6000
CAMERA_CONTROLS_FOR_INTERNAL_USE = {
    # Automatische Belichtung. False = manueller ExposureTime-Wert wird verwendet.
    "AeEnable": False,
    # Automatischer Weissabgleich. False = ColourGains unten werden verwendet.
    "AwbEnable": False,
    # Belichtungszeit in Mikrosekunden. Höher = heller, zu hoch = A4 kann ausbrennen.
    "ExposureTime": 6000,  # Mikrosekunden
    # Analoge Verstärkung. Höher = heller, aber mehr Bildrauschen.
    "AnalogueGain": 1.0,
    # Rot-/Blau-Verstärkung für Weissabgleich. Höherer Wert = jeweiliger Farbkanal stärker.
    "ColourGains": (1.5, 1.5),
}


# ============================================================
# AUSGABE / DEBUG-DATEIEN
# ============================================================

# Hauptordner für Debug-Ausgaben der Vision-Pipeline.
OUTPUT_DIR = PROJECT_ROOT / "src" / "vision" / "output"
# Speichert Kopien der finalen Algorithmus-Teilemasken für Debug-Zwecke.
OUTPUT_PARTS_DIR = OUTPUT_DIR / "parts"
# Namenspräfix für alle Debug-Dateien dieses Laufs.
RUN_NAME = "debug_"

# Dateiname des gespeicherten Rohbildes. Beispiel: step_09_input.png.
OUTPUT_IMAGE_FILENAME = f"{RUN_NAME}_input.png"
# Debug-Bild mit erkannten Markern, Referenzpunkten und berechneten A4-Ecken.
OUTPUT_DEBUG_FILENAME = f"{RUN_NAME}_a4_corners_debug.png"
# Entzerrtes A4-Bild. Dieses Bild ist die Basis für die Teile-Erkennung.
OUTPUT_WARP_FILENAME = f"{RUN_NAME}_warp_a4.png"
# Binärmaske aller erkannten Teile. Weiss = Teil, schwarz = Hintergrund.
OUTPUT_MASK_FILENAME = f"{RUN_NAME}_parts_mask.png"
# Debug-Bild mit Konturen, Bounding Boxes, Schwerpunkten und Raster.
OUTPUT_PARTS_DEBUG_FILENAME = f"{RUN_NAME}_parts_debug.png"
# Debug-JSON mit Zusatzinfos, Pfaden und Einstellungen.
OUTPUT_JSON_FILENAME = f"{RUN_NAME}_parts.json"
# Speichert die Homographie-Matrix Bild -> entzerrtes A4 als NumPy-Datei.
OUTPUT_H_IMAGE_TO_WARP_PATH = f"{RUN_NAME}_h_image_to_warp.npy"

# False = nur Algorithmus-Input in DESTINATION_TO_ALGO_INPUT_FOLDER schreiben.
# True = Zusätzlich Debug-Dateien in src/vision/output speichern.
SAVE_DEBUG_FILES = True

# False = Kantenkorrektur wird nicht ausgeführt.
# True Kantenkorrektur wird durchgeführt aber Stand 1.Juni noch ziemlich wonky
CALCULATE_AREA_WITHOUT_SIDES = False

# ============================================================
# ARUCO / A4-GEOMETRIE
# ============================================================

# ArUco-Wörterbuch. Muss zu den gedruckten Markern passen.
ARUCO_DICT = cv2.aruco.DICT_4X4_50
# Diese Marker-IDs müssen erkannt werden, sonst kann die A4-Fläche nicht berechnet werden.
REQUIRED_IDS = [0, 1, 2, 3]

# A4 im Querformat
# Breite der A4-Fläche im Querformat in mm.
A4_WIDTH_MM = 297.0
# Höhe der A4-Fläche im Querformat in mm.
A4_HEIGHT_MM = 210.0
# Höhe der Kamera über der A4 Fläche in mm.
CAM_HEIGHT = 700.0
# Skalierung im entzerrten Bild. Grösser = mehr Pixel pro mm, genauer aber langsamer.
# Interne Auflösung mit der das Cam Modul arbeitet, so hoch wie möglich bzw sinnvoll
WORKING_INTERNAL_PX_PER_MM = 6.0
# Auflösung der Bilder die der Algorithmus erhält, so tief wie nötig
ALGO_INPUT_PX_PER_MM = 3.0


# Diese Punkte sind die gemessenen Referenzpunkte im Bild.
# Bei OFFSET = 0.0 sind diese Referenzpunkte direkt die echten A4-Ecken.


# A4 top_right    = ID 0 / Ecke 3
# A4 bottom_right = ID 1 / Ecke 2
# A4 bottom_left  = ID 2 / Ecke 1
# A4 top_left     = ID 3 / Ecke 0
REFERENCE_CORNER_FROM_MARKER = {
    # Echte A4-Ecke oben links wird aus Marker ID 3, Marker-Ecke 0 gelesen.
    "top_left": {"marker_id": 3, "corner_index": 0},
    # Echte A4-Ecke oben rechts wird aus Marker ID 0, Marker-Ecke 3 gelesen.
    "top_right": {"marker_id": 0, "corner_index": 3},
    # Echte A4-Ecke unten rechts wird aus Marker ID 1, Marker-Ecke 2 gelesen.
    "bottom_right": {"marker_id": 1, "corner_index": 2},
    # Echte A4-Ecke unten links wird aus Marker ID 2, Marker-Ecke 1 gelesen.
    "bottom_left": {"marker_id": 2, "corner_index": 1},
}

# ------------------------------------------------------------
# A4-Offset / Rahmen-Offset
# ------------------------------------------------------------
# Diese Werte beschreiben, wie weit die gemessenen Referenzpunkte ausserhalb
# der echten A4-Fläche liegen.
#
# Beispiel:
# Wenn ein Rahmen rundherum 20 mm breit ist und die ArUco-Referenzpunkte
# auf den Rahmenecken liegen, dann:
# FRAME_OFFSET_LEFT_MM = 20.0
# FRAME_OFFSET_RIGHT_MM = 20.0
# FRAME_OFFSET_TOP_MM = 20.0
# FRAME_OFFSET_BOTTOM_MM = 20.0
#
# Wenn aktuell die ArUco-Ecken direkt den A4-Ecken entsprechen:
# alle Werte auf 0.0 lassen.
# Abstand links zwischen Referenz-/Rahmenfläche und echter A4-Fläche. Grösser = A4 startet weiter rechts.
FRAME_OFFSET_LEFT_MM = 0.0
# Abstand rechts zwischen echter A4-Fläche und Referenz-/Rahmenfläche. Grösser = A4 endet weiter links.
FRAME_OFFSET_RIGHT_MM = 0.0
# Abstand oben zwischen Referenz-/Rahmenfläche und echter A4-Fläche. Grösser = A4 startet weiter unten.
FRAME_OFFSET_TOP_MM = 0.0
# Abstand unten zwischen echter A4-Fläche und Referenz-/Rahmenfläche. Grösser = A4 endet weiter oben.
FRAME_OFFSET_BOTTOM_MM = 0.0

# Optionaler Feintrimm pro echter A4-Ecke.
# Koordinaten in der Referenz-/Rahmenfläche:
# +x = nach rechts, +y = nach unten.
#
# Normalerweise alles 0.0 lassen.
# Nur verwenden, wenn einzelne Marker mechanisch anders sitzen.
A4_CORNER_EXTRA_OFFSETS_MM = {
    # Feintrimm oben links. x höher = nach rechts, y höher = nach unten.
    "top_left": {"x": 0.0, "y": 0.0},
    # Feintrimm oben rechts. x höher = nach rechts, y höher = nach unten.
    "top_right": {"x": 0.0, "y": 0.0},
    # Feintrimm unten rechts. x höher = nach rechts, y höher = nach unten.
    "bottom_right": {"x": 0.0, "y": 0.0},
    # Feintrimm unten links. x höher = nach rechts, y höher = nach unten.
    "bottom_left": {"x": 0.0, "y": 0.0},
}

# Optionaler Rand, der bei der Teile-Erkennung ignoriert wird.
# Rand ignorieren. Grösser = weniger Störungen am Rand, aber nutzbare Fläche wird kleiner.
IGNORE_BORDER_MM = 0.0


# ============================================================
# KOORDINATENSYSTEM FÜR OUTPUT
# ============================================================

# Fix: Ursprung oben rechts, x nach links, y nach unten.
# Fixes Output-Koordinatensystem: Ursprung oben rechts, x nach links, y nach unten.
COORDINATE_ORIGIN = "top_right"


# ============================================================
# TEILE-SEGMENTIERUNG
# ============================================================

# Segmentierungsmodus: fixed = fester Grenzwert, otsu = automatisch, adaptive = lokal anpassend.
SEGMENTATION_THRESHOLD_MODE = "otsu"  # "fixed", "otsu", "adaptive"
# Nur bei fixed relevant. Grösser = hellere Pixel werden eher als dunkles Teil erkannt.
THRESHOLD_VALUE = 150
# Nur bei adaptive relevant. Grösser = grössere lokale Umgebung, ruhiger aber weniger fein.
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 101
# Nur bei adaptive relevant. Grösser = Schwelle wird strenger, meist weniger erkannte Fläche.
ADAPTIVE_THRESHOLD_C = 8

# Weichzeichnung vor der Segmentierung. Grösser = weniger Rauschen, aber Kanten werden weicher.
GAUSSIAN_BLUR_KERNEL_SIZE = 7

# Kleinste erlaubte Teilfläche. Grösser = kleine Störungen werden eher ignoriert.
MIN_PART_AREA_MM2 = 500.0
# Grösste erlaubte Teilfläche. Kleiner = grosse Fehlblobs werden eher ignoriert.
MAX_PART_AREA_MM2 = 100000.0

# Entfernt kleine weisse Störungen in der Maske. Grösser = aggressiveres Entfernen.
MORPH_OPEN_KERNEL_SIZE = 5
# Schliesst kleine Löcher/Lücken in Teilen. Grösser = verbindet Flächen stärker.
MORPH_CLOSE_KERNEL_SIZE = 7
# True = Löcher innerhalb erkannter Teile füllen.
FILL_CONTOUR_HOLES = True

# Zusätzlicher Rand um ausgeschnittene Teile in Pixeln. Grösser = mehr Umgebung im Crop.
CROP_PADDING_PX = 0
# Erlaubte Anzahl Teile. Gültig ist nur 4 oder 6.
EXPECTED_PART_COUNT = [4, 5, 6]
# Hintergrundwert für Cutouts. 255 = weiss, 0 = schwarz.
CUTOUT_BACKGROUND_VALUE = 255


# ============================================================
# VALIDIERUNG
# ============================================================

# PREN-Puzzle ohne Rahmen: 18.9 x 12.6 cm
# Erwartete Gesamtfläche aller Puzzleteile in mm2.
# 20879 mm: Oberfläche des 6 Teile Puzzles von Silvan

EXPECTED_TOTAL_PART_AREA_MM2 = 20879
# Erlaubte Flächenabweichung. 0.03 = +-3 %.
MAX_TOTAL_AREA_ERROR_RATIO = 2.0


# ============================================================
# DEBUG-FARBEN UND DARSTELLUNG
# ============================================================

# Farben sind im OpenCV-Format BGR, nicht RGB.
# Farbe der erkannten Teilekontur im Debug-Bild. BGR: grün.
PART_CONTOUR_COLOR = (0, 255, 0)
# Farbe des Schwerpunktpunktes im Debug-Bild. BGR: rot.
PART_CENTROID_COLOR = (0, 0, 255)
# Farbe der Bounding Box um jedes Teil. BGR: cyan/hellblau.
PART_BOX_COLOR = (255, 255, 0)
# Farbe der Textbeschriftung im Teile-Debug-Bild. BGR: weiss.
PART_TEXT_COLOR = (255, 255, 255)

# Radius des Mittelpunkt-Punktes im Debug-Bild. Grösser = besser sichtbar.
PART_CENTROID_RADIUS_PX = 8
# Schriftgrösse der Teilbeschriftung im Debug-Bild.
PART_TEXT_FONT_SCALE = 0.7
# Schriftstärke der Teilbeschriftung. Grösser = dicker.
PART_TEXT_THICKNESS = 2

# Farbe der ArUco-Marker-Umrandung. BGR: grün.
COLOR_MARKER_OUTLINE = (0, 255, 0)
# Farbe des Marker-Mittelpunktes. BGR: blau.
COLOR_MARKER_CENTER = (255, 0, 0)
# Farbe der Marker-ID-Beschriftung. BGR: grün.
COLOR_MARKER_ID_TEXT = (0, 255, 0)

# Farbe für Marker-Ecke 0. BGR: rot.
COLOR_CORNER_0 = (0, 0, 255)
# Farbe für Marker-Ecke 1. BGR: gelb.
COLOR_CORNER_1 = (0, 255, 255)
# Farbe für Marker-Ecke 2. BGR: cyan/hellblau.
COLOR_CORNER_2 = (255, 255, 0)
# Farbe für Marker-Ecke 3. BGR: magenta.
COLOR_CORNER_3 = (255, 0, 255)

# Farbe der berechneten echten A4-Eckpunkte. BGR: orange.
COLOR_A4_POINT = (0, 165, 255)
# Farbe der gemessenen Referenz-/Rahmenpunkte. BGR: blau-orange.
COLOR_REFERENCE_POINT = (255, 128, 0)
# Farbe der A4-Ecken-Beschriftung. BGR: orange.
COLOR_A4_TEXT = (0, 165, 255)
# Farbe der A4-Umrandung. BGR: weiss.
COLOR_A4_POLYLINE = (255, 255, 255)
# Farbe der Referenz-/Rahmenumrandung. BGR: blau-orange.
COLOR_REFERENCE_POLYLINE = (255, 128, 0)
# Farbe der Status-Texte im Debug-Bild. BGR: weiss.
COLOR_STATUS_TEXT = (255, 255, 255)

# Radius für Marker-Ecken im Debug-Bild.
CORNER_CIRCLE_RADIUS_PX = 6
# Radius für Marker-Mittelpunkte im Debug-Bild.
MARKER_CENTER_RADIUS_PX = 6
# Radius der berechneten A4-Ecken im Debug-Bild.
A4_CORNER_RADIUS_PX = 10
# Radius der gemessenen Referenz-/Rahmenecken im Debug-Bild.
REFERENCE_CORNER_RADIUS_PX = 7

# Textversatz der Marker-Ecknummer nach rechts. Grösser = Text weiter rechts.
CORNER_TEXT_OFFSET_X = 8
# Textversatz der Marker-Ecknummer nach oben. Negativer = Text weiter oben.
CORNER_TEXT_OFFSET_Y = -8

# Allgemeine Schriftgrösse für ArUco-/A4-Debug-Texte. Grösser = grössere Schrift.
TEXT_FONT_SCALE = 0.8
# Allgemeine Schriftstärke für ArUco-/A4-Debug-Texte. Grösser = dicker.
TEXT_THICKNESS = 2

# Zeichnen des Koordinatensystems in parts_debug.png
# True = Koordinatenraster in parts_debug.png einzeichnen.
DEBUG_DRAW_COORDINATE_GRID = True
# Abstand der feinen Rasterlinien. 10.0 = 1 cm Raster, 1.0 = 1 mm Raster.
DEBUG_GRID_SPACING_MM = 1.0  # mm
# Abstand der stärkeren Rasterlinien. 10.0 = jede 1-cm-Linie stärker.
DEBUG_GRID_MAJOR_SPACING_MM = 10.0  # stärkere Linie alle x mm
# Transparenz des Rasters. Grösser = Raster sichtbarer, Bild darunter dunkler.
DEBUG_GRID_ALPHA = 0.35
# Länge der x-/y-Achsenpfeile in mm.
DEBUG_AXIS_LENGTH_MM = 50.0

# Farbe der feinen Rasterlinien im parts_debug-Bild. BGR: grau.
COLOR_GRID_MINOR = (120, 120, 120)
# Farbe der stärkeren Rasterlinien. BGR: hellgrau.
COLOR_GRID_MAJOR = (180, 180, 180)
# Farbe der x-Achse. BGR: rot.
COLOR_AXIS_X = (0, 0, 255)
# Farbe der y-Achse. BGR: grün.
COLOR_AXIS_Y = (0, 255, 0)
# Farbe des Ursprungspunktes. BGR: weiss.
COLOR_AXIS_ORIGIN = (255, 255, 255)
# Farbe der Achsenbeschriftung. BGR: weiss.
COLOR_AXIS_TEXT = (255, 255, 255)


# ============================================================
# DATEI- UND ALLGEMEINE HILFSFUNKTIONEN
# ============================================================


def buildOutputPath(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def buildDirPath(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def saveJson(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def savePngImage(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"cv2.imwrite konnte das Bild nicht speichern: {path}")


def clearAlgoInputFolder():
    # Leert den Algorithmus-Input-Ordner vollständig.
    # Der Ordner selbst bleibt bestehen.
    algoInputDirPath = buildDirPath(DESTINATION_TO_ALGO_INPUT_FOLDER)

    for path in algoInputDirPath.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    return algoInputDirPath


def clearDebugOutputFolder():
    # Leert den Output-Ordner vollständig.
    # Der Ordner selbst bleibt bestehen.
    outputDirPath = buildDirPath(OUTPUT_DIR)

    for path in outputDirPath.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    return outputDirPath


def rotateImageIfNeeded(imageBgr):
    if ROTATE_90_CLOCKWISE and ROTATE_180:
        raise ValueError(
            "Nur eine Rotation aktivieren: entweder ROTATE_90_CLOCKWISE oder ROTATE_180."
        )

    if ROTATE_90_CLOCKWISE:
        return cv2.rotate(imageBgr, cv2.ROTATE_90_CLOCKWISE)

    if ROTATE_180:
        return cv2.rotate(imageBgr, cv2.ROTATE_180)

    return imageBgr


# ============================================================
# BILDEINGABE
# ============================================================


def initCameraIfAvailable():
    if not isPiCameraAvailable():
        return None
    return initCamera()


def isPiCameraAvailable():
    return Picamera2 is not None


def initCamera():
    # Erstellt, konfiguriert und startet die Pi Camera.
    # Rückgabe ist eine gestartete Kamera, die später an captureImageFromInitializedCamera(cam)
    # übergeben werden kann.
    if Picamera2 is None:
        raise RuntimeError("Picamera2 ist auf dieser Maschine nicht verfügbar.")

    print("Initialisiere Kamera...")
    cam = Picamera2()

    # BGR888 = 8 Bit pro Farbkanal, Reihenfolge Blue/Green/Red.
    # Passt direkt zu OpenCV, da OpenCV standardmässig BGR erwartet.
    cameraConfig = cam.create_still_configuration(
        main={
            "size": (IMAGE_WIDTH, IMAGE_HEIGHT),
            "format": "BGR888",
        },
    )

    cam.configure(cameraConfig)
    cam.set_controls(INIT_CAMERA_CONTROLS_FOR_PIPELINE)

    print("Starte Kamera...")
    cam.start()
    cam._started_at = time.monotonic()

    return cam


def captureImageFromInitializedCamera(cam, controls=None, waitSecondsBetweeCapture=0.0):
    # Nimmt ein Bild mit einer bereits gestarteten Kamera auf.
    # Hier wird die Kamera nicht neu initialisiert und nicht gestoppt.
    if cam is None:
        # Platzhalter falls cam aus irgendeinem Grunde nicht bereit
        print("!!!!Keine Kamera gefunden obwohl sie vorhanden sein müsste")
        time.sleep(3)
        print("mache jetzt ohne Prüfung weiter")

    startedAt = getattr(cam, "_started_at", None)

    if STARTUP_WAIT_SECONDS > 0 and startedAt is not None:
        elapsedSeconds = time.monotonic() - startedAt
        remainingSeconds = STARTUP_WAIT_SECONDS - elapsedSeconds

        if remainingSeconds > 0:
            print(
                f"Kamera läuft erst seit {elapsedSeconds:.1f} Sekunden, "
                f"warte noch {remainingSeconds:.1f} Sekunden..."
            )
            time.sleep(remainingSeconds)

    print("Nehme Bild auf...")
    imageBgr = cam.capture_array()

    grayImage = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)
    overexposedPixels = np.sum(grayImage >= 250)
    totalPixels = grayImage.shape[0] * grayImage.shape[1]
    overexposedRatio = overexposedPixels / totalPixels

    print(f"überbelichtete Pixel: {overexposedRatio * 100.0:.2f} %")

    return imageBgr


def stopCamera(cam):
    # Stoppt eine gestartete Kamera.
    # Die Funktion ist absichtlich tolerant, damit Cleanup im finally-Block der pipeline.py robust bleibt.
    if cam is None:
        return

    try:
        cam.stop()
        print("Kamera gestoppt.")
    except Exception:
        pass


def captureImageFromCamera():
    # Standalone-Kompatibilität:
    # Für direkte Tests des Cam-Moduls wird die Kamera hier weiterhin selbst
    # gestartet, benutzt und danach wieder gestoppt.
    cam = None

    try:
        cam = initCamera()
        return captureImageFromInitializedCamera(cam)

    finally:
        stopCamera(cam)


def loadImageFromFile():
    inputPath = Path(INPUT_IMAGE_PATH)

    if not inputPath.exists():
        raise FileNotFoundError(f"Eingabebild nicht gefunden: {inputPath}")

    print(f"Lade Bild von Datei: {inputPath}")

    imageBgr = cv2.imread(str(inputPath), cv2.IMREAD_COLOR)

    if imageBgr is None:
        raise RuntimeError(f"cv2.imread konnte das Bild nicht laden: {inputPath}")

    return imageBgr


def getInputImage(cam=None):
    if IMAGE_SOURCE == "camera":
        if cam is not None:
            return captureImageFromInitializedCamera(cam)

        if isPiCameraAvailable():
            return captureImageFromCamera()

        print("Keine Picamera2-Unterstuetzung auf dieser Maschine, nutze Datei.")
        return loadImageFromFile()

    if IMAGE_SOURCE == "file":
        return loadImageFromFile()

    raise ValueError('IMAGE_SOURCE muss "camera" oder "file" sein.')


# ============================================================
# ARUCO / A4-ERKENNUNG MIT OFFSET
# ============================================================

def detectArucoMarkers(imageBgr):
    # Erkennt alle ArUco-Marker im Bild und behält nur die REQUIRED_IDS.
    arucoDictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    arucoParameters = cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    cornersList, ids, rejectedCandidates = arucoDetector.detectMarkers(imageBgr)
    detectedMarkers = {}

    if ids is None:
        return detectedMarkers, rejectedCandidates

    for i, markerId in enumerate(ids.flatten()):
        if markerId not in REQUIRED_IDS:
            continue

        markerCorners = cornersList[i].reshape(4, 2).astype(np.float32)

        detectedMarkers[int(markerId)] = {
            "id": int(markerId),
            "corners": markerCorners,
        }

    return detectedMarkers, rejectedCandidates


def validateDetectedMarkers(detectedMarkers):
    missingIds = [
        markerId for markerId in REQUIRED_IDS if markerId not in detectedMarkers
    ]

    if missingIds:
        raise RuntimeError(
            f"Nicht alle benoetigten Marker wurden erkannt. Fehlend: {missingIds}"
        )


def getReferenceCornersFromMarkers(detectedMarkers):
    # Holt genau jene Marker-Ecken, welche als Referenzpunkte für die A4-Geometrie dienen.
    validateDetectedMarkers(detectedMarkers)

    referenceCorners = {}

    for cornerName, mapping in REFERENCE_CORNER_FROM_MARKER.items():
        markerId = mapping["marker_id"]
        cornerIndex = mapping["corner_index"]

        markerCorners = detectedMarkers[markerId]["corners"]
        referenceCorners[cornerName] = markerCorners[cornerIndex].astype(np.float32)

    return referenceCorners


def getFrameSizeMm():
    # Referenz-/Rahmengrösse = echtes A4 plus Offset links/rechts/oben/unten.
    frameWidthMm = A4_WIDTH_MM + FRAME_OFFSET_LEFT_MM + FRAME_OFFSET_RIGHT_MM
    frameHeightMm = A4_HEIGHT_MM + FRAME_OFFSET_TOP_MM + FRAME_OFFSET_BOTTOM_MM

    if frameWidthMm <= 0 or frameHeightMm <= 0:
        raise ValueError("Rahmen-/A4-Groesse ungueltig. Offsets pruefen.")

    return frameWidthMm, frameHeightMm


def buildReferenceCornerArrayImage(referenceCorners):
    return np.array(
        [
            referenceCorners["top_left"],
            referenceCorners["top_right"],
            referenceCorners["bottom_right"],
            referenceCorners["bottom_left"],
        ],
        dtype=np.float32,
    )


def buildReferenceCornerArrayMm():
    frameWidthMm, frameHeightMm = getFrameSizeMm()

    return np.array(
        [
            [0.0, 0.0],
            [frameWidthMm, 0.0],
            [frameWidthMm, frameHeightMm],
            [0.0, frameHeightMm],
        ],
        dtype=np.float32,
    )


def buildA4CornerArrayInReferenceMm():
    # Berechnet die echten A4-Ecken innerhalb der grösseren Referenz-/Rahmenfläche.
    left = FRAME_OFFSET_LEFT_MM
    right = FRAME_OFFSET_LEFT_MM + A4_WIDTH_MM
    top = FRAME_OFFSET_TOP_MM
    bottom = FRAME_OFFSET_TOP_MM + A4_HEIGHT_MM

    basePoints = {
        "top_left": np.array([left, top], dtype=np.float32),
        "top_right": np.array([right, top], dtype=np.float32),
        "bottom_right": np.array([right, bottom], dtype=np.float32),
        "bottom_left": np.array([left, bottom], dtype=np.float32),
    }

    result = {}

    for cornerName, point in basePoints.items():
        extraOffset = A4_CORNER_EXTRA_OFFSETS_MM[cornerName]
        result[cornerName] = np.array(
            [
                point[0] + float(extraOffset["x"]),
                point[1] + float(extraOffset["y"]),
            ],
            dtype=np.float32,
        )

    return result


def transformPoints(points, homography):
    pointsArray = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pointsArray, homography)
    return transformed.reshape(-1, 2)


def extractA4Corners(detectedMarkers):
    # Diese Funktion ist der Kern der Offset-Logik:
    # Erst wird die gemessene Rahmenfläche aufgebaut, dann werden daraus
    # die echten A4-Ecken zurück ins Kamerabild projiziert.
    # 1. Aus den Markern die gemessene Referenz-/Rahmenfläche im Bild holen.
    referenceCorners = getReferenceCornersFromMarkers(detectedMarkers)

    # 2. Homographie: Referenz-/Rahmen-mm -> Bildpixel.
    referenceImagePoints = buildReferenceCornerArrayImage(referenceCorners)
    referenceMmPoints = buildReferenceCornerArrayMm()
    hReferenceMmToImage = cv2.getPerspectiveTransform(
        referenceMmPoints, referenceImagePoints
    )

    # 3. Echte A4-Ecken innerhalb der Referenz-/Rahmenfläche in mm definieren.
    a4CornersReferenceMm = buildA4CornerArrayInReferenceMm()
    a4ReferenceMmArray = np.array(
        [
            a4CornersReferenceMm["top_left"],
            a4CornersReferenceMm["top_right"],
            a4CornersReferenceMm["bottom_right"],
            a4CornersReferenceMm["bottom_left"],
        ],
        dtype=np.float32,
    )

    # 4. Echte A4-Ecken zurück ins Bild projizieren.
    a4ImageArray = transformPoints(a4ReferenceMmArray, hReferenceMmToImage)

    a4Corners = {
        "top_left": a4ImageArray[0].astype(np.float32),
        "top_right": a4ImageArray[1].astype(np.float32),
        "bottom_right": a4ImageArray[2].astype(np.float32),
        "bottom_left": a4ImageArray[3].astype(np.float32),
    }

    return a4Corners, referenceCorners


def calculate_native_a4_pixel_density(a4_corners_px):
    top_left = np.asarray(a4_corners_px["top_left"], dtype=np.float32)
    top_right = np.asarray(a4_corners_px["top_right"], dtype=np.float32)
    bottom_right = np.asarray(a4_corners_px["bottom_right"], dtype=np.float32)
    bottom_left = np.asarray(a4_corners_px["bottom_left"], dtype=np.float32)

    # Zwei Breiten messen: obere und untere A4-Kante.
    width_top_px = np.linalg.norm(top_right - top_left)
    width_bottom_px = np.linalg.norm(bottom_right - bottom_left)

    # Zwei Höhen messen: linke und rechte A4-Kante.
    height_left_px = np.linalg.norm(top_left - bottom_left)
    height_right_px = np.linalg.norm(top_right - bottom_right)

    # Mittelwert ist robuster, falls Kamera/A4 leicht schräg stehen.
    measured_width_px = (width_top_px + width_bottom_px) / 2.0
    measured_height_px = (height_left_px + height_right_px) / 2.0

    native_px_per_mm_x = measured_width_px / A4_WIDTH_MM
    native_px_per_mm_y = measured_height_px / A4_HEIGHT_MM
    native_px_per_mm_avg = (native_px_per_mm_x + native_px_per_mm_y) / 2.0

    return {
        "native_px_per_mm_x": float(native_px_per_mm_x),
        "native_px_per_mm_y": float(native_px_per_mm_y),
        "native_px_per_mm_avg": float(native_px_per_mm_avg),
        "measured_width_px": float(measured_width_px),
        "measured_height_px": float(measured_height_px),
    }


# ============================================================
# HOMOGRAPHIE / KOORDINATEN
# ============================================================


def getWarpSizePx():
    warpWidthPx = int(round(A4_WIDTH_MM * WORKING_INTERNAL_PX_PER_MM))
    warpHeightPx = int(round(A4_HEIGHT_MM * WORKING_INTERNAL_PX_PER_MM))
    return warpWidthPx, warpHeightPx


def buildImageCornerArray(a4Corners):
    return np.array(
        [
            a4Corners["top_left"],
            a4Corners["top_right"],
            a4Corners["bottom_right"],
            a4Corners["bottom_left"],
        ],
        dtype=np.float32,
    )


def buildWarpCornerArrayPx():
    warpWidthPx, warpHeightPx = getWarpSizePx()

    return np.array(
        [
            [0, 0],
            [warpWidthPx - 1, 0],
            [warpWidthPx - 1, warpHeightPx - 1],
            [0, warpHeightPx - 1],
        ],
        dtype=np.float32,
    )


def computeHomographyImageToWarp(a4Corners):
    imagePoints = buildImageCornerArray(a4Corners)
    warpPointsPx = buildWarpCornerArrayPx()
    return cv2.getPerspectiveTransform(imagePoints, warpPointsPx)


def warpImageToA4(imageBgr, hImageToWarp):
    warpWidthPx, warpHeightPx = getWarpSizePx()

    return cv2.warpPerspective(
        imageBgr,
        hImageToWarp,
        (warpWidthPx, warpHeightPx),
    )


def warpPxToOutputPxTopRight(xPx, yPx):
    # Warp-Pixel haben Ursprung oben links. Für den Output wird x gespiegelt, damit Ursprung oben rechts gilt.
    warpWidthPx, _ = getWarpSizePx()

    xA4Px = (warpWidthPx - 1) - float(xPx)
    yA4Px = float(yPx)

    return xA4Px, yA4Px


def outputPxToOutputMm(xPx, yPx):
    # Rechnet eine kontinuierliche Pixelkoordinate aus dem entzerrten A4-Bild
    # in echte A4-Millimeter um.
    # xPx/yPx sind keine diskreten Pixelnummern, sondern geometrische Koordinaten,
    # Zbsp ein Schwerpunkt aus cv2.moments() liefert kontinuierliche! Bildkoordinaten
    # Das Warp-Bild bildet die A4-Fläche von Pixelkoordinate 0 bis width-1
    # bzw. height-1 ab. Diese Spanne entspricht exakt 0..A4_WIDTH_MM und
    # 0..A4_HEIGHT_MM, also von einem Rand bis UND mit zum anderen Rand
    warpWidthPx, warpHeightPx = getWarpSizePx()

    xMm = float(xPx) * A4_WIDTH_MM / float(warpWidthPx - 1)
    yMm = float(yPx) * A4_HEIGHT_MM / float(warpHeightPx - 1)

    return xMm, yMm


def buildCoordinateOriginDescription():
    return "origin top_right, x left, y down"


# ============================================================
# TEILE-SEGMENTIERUNG
# ============================================================


def ensureOddKernelSize(value):
    value = int(value)

    if value < 3:
        value = 3

    if value % 2 == 0:
        value += 1

    return value


def applyIgnoreBorder(binaryMask):
    # Setzt einen Randbereich auf schwarz, damit Randartefakte nicht als Teile erkannt werden.
    if IGNORE_BORDER_MM <= 0:
        return binaryMask

    borderPx = int(round(IGNORE_BORDER_MM * WORKING_INTERNAL_PX_PER_MM))

    if borderPx <= 0:
        return binaryMask

    maskedBinaryMask = binaryMask.copy()
    imageHeight, imageWidth = maskedBinaryMask.shape[:2]

    maskedBinaryMask[0:borderPx, :] = 0
    maskedBinaryMask[imageHeight - borderPx : imageHeight, :] = 0
    maskedBinaryMask[:, 0:borderPx] = 0
    maskedBinaryMask[:, imageWidth - borderPx : imageWidth] = 0

    return maskedBinaryMask


def fillMaskContourHoles(binaryMask):
    # Füllt Löcher innerhalb externer Konturen, damit Teile als volle Flächen zählen.
    contours, _ = cv2.findContours(
        binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filledMask = np.zeros(binaryMask.shape, dtype=np.uint8)
    cv2.drawContours(filledMask, contours, -1, 255, -1)

    return filledMask


def buildPartsMask(warpedImageBgr):
    # Erst Grau/Blur, dann Schwellwert, dann Morphologie: daraus entsteht die Teile-Maske.
    # Aus dem entzerrten A4-Bild wird eine Binärmaske gebaut:
    # weiss = erkanntes Teil, schwarz = Hintergrund.
    grayImage = cv2.cvtColor(warpedImageBgr, cv2.COLOR_BGR2GRAY)
    blurKernelSize = ensureOddKernelSize(GAUSSIAN_BLUR_KERNEL_SIZE)

    blurredImage = cv2.GaussianBlur(grayImage, (blurKernelSize, blurKernelSize), 0)

    if SEGMENTATION_THRESHOLD_MODE == "fixed":
        _, binaryMask = cv2.threshold(
            blurredImage,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY_INV,
        )

    elif SEGMENTATION_THRESHOLD_MODE == "otsu":
        _, binaryMask = cv2.threshold(
            blurredImage,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    elif SEGMENTATION_THRESHOLD_MODE == "adaptive":
        adaptiveBlockSize = ensureOddKernelSize(ADAPTIVE_THRESHOLD_BLOCK_SIZE)

        binaryMask = cv2.adaptiveThreshold(
            blurredImage,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptiveBlockSize,
            ADAPTIVE_THRESHOLD_C,
        )

    else:
        raise ValueError(
            'SEGMENTATION_THRESHOLD_MODE muss "fixed", "otsu" oder "adaptive" sein.'
        )

    openKernel = np.ones((MORPH_OPEN_KERNEL_SIZE, MORPH_OPEN_KERNEL_SIZE), np.uint8)
    closeKernel = np.ones((MORPH_CLOSE_KERNEL_SIZE, MORPH_CLOSE_KERNEL_SIZE), np.uint8)

    binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_OPEN, openKernel)
    binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_CLOSE, closeKernel)

    if FILL_CONTOUR_HOLES:
        binaryMask = fillMaskContourHoles(binaryMask)

    binaryMask = applyIgnoreBorder(binaryMask)

    return binaryMask


def computeContourCentroid(contour):
    # Schwerpunkt über Bildmomente; Fallback ist die Mitte der Bounding Box.
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + (w / 2.0), y + (h / 2.0)

    centroidX = moments["m10"] / moments["m00"]
    centroidY = moments["m01"] / moments["m00"]

    return centroidX, centroidY


def findAllValidParts(binaryMask):
    # Sucht Konturen und filtert sie über minimale/maximale Fläche.
    # Konturen suchen und nur solche behalten, deren Fläche im erlaubten Bereich liegt.
    contours, _ = cv2.findContours(
        binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    minAreaPx = MIN_PART_AREA_MM2 * (WORKING_INTERNAL_PX_PER_MM**2)
    maxAreaPx = MAX_PART_AREA_MM2 * (WORKING_INTERNAL_PX_PER_MM**2)

    detectedParts = []

    for contour in contours:
        areaPx = cv2.contourArea(contour)

        if areaPx < minAreaPx:
            continue

        if areaPx > maxAreaPx:
            continue

        centroidX, centroidY = computeContourCentroid(contour)
        x, y, w, h = cv2.boundingRect(contour)

        detectedParts.append(
            {
                "contour": contour,
                "areaPx": float(areaPx),
                "centroidX": float(centroidX),
                "centroidY": float(centroidY),
                "bboxX": int(x),
                "bboxY": int(y),
                "bboxW": int(w),
                "bboxH": int(h),
            }
        )

    return detectedParts


def sortPartsByOutputYThenOutputX(detectedParts):
    def sortKey(partInfo):
        centroidXpxOutput, centroidYpxOutput = warpPxToOutputPxTopRight(
            partInfo["centroidX"],
            partInfo["centroidY"],
        )
        centroidXmm, centroidYmm = outputPxToOutputMm(
            centroidXpxOutput, centroidYpxOutput
        )
        return centroidYmm, centroidXmm

    return sorted(detectedParts, key=sortKey)


def addDerivedPartValues(detectedParts):
    # Ergänzt sortierten Teilen Index, Namen, Schwerpunkt in px/mm und Fläche in mm2.
    # Ergänzt berechnete Werte für JSON, Sortierung und Debug-Ausgabe.
    for i, partInfo in enumerate(detectedParts):
        centroidXpxOutput, centroidYpxOutput = warpPxToOutputPxTopRight(
            partInfo["centroidX"],
            partInfo["centroidY"],
        )
        centroidXmm, centroidYmm = outputPxToOutputMm(
            centroidXpxOutput, centroidYpxOutput
        )
        areaMm2 = partInfo["areaPx"] / (WORKING_INTERNAL_PX_PER_MM**2)

        partInfo["index"] = i + 1
        partInfo["partName"] = f"part_{i + 1:02d}"

        # Output-Pixel, Ursprung oben rechts
        partInfo["centroidXpx"] = float(centroidXpxOutput)
        partInfo["centroidYpx"] = float(centroidYpxOutput)

        # Output-mm, Ursprung oben rechts
        partInfo["centroidXmm"] = float(centroidXmm)
        partInfo["centroidYmm"] = float(centroidYmm)

        partInfo["areaMm2"] = float(areaMm2)


# ============================================================
# FLAECHENVALIDIERUNG
# ============================================================


def computeTotalPartsAreaMm2(detectedParts):
    return sum(partInfo["areaMm2"] for partInfo in detectedParts)


def buildAreaValidationData(detectedParts):
    # Vergleicht gemessene Gesamtfläche mit der erwarteten PREN-Puzzlefläche.
    totalAreaMm2 = computeTotalPartsAreaMm2(detectedParts)
    expectedAreaMm2 = EXPECTED_TOTAL_PART_AREA_MM2

    areaErrorMm2 = totalAreaMm2 - expectedAreaMm2
    areaErrorRatio = areaErrorMm2 / expectedAreaMm2
    areaErrorPercent = areaErrorRatio * 100.0
    isValid = abs(areaErrorRatio) <= MAX_TOTAL_AREA_ERROR_RATIO

    return {
        "expected_total_area_mm2": round(expectedAreaMm2, 6),
        "measured_total_area_mm2": round(totalAreaMm2, 6),
        "area_error_mm2": round(areaErrorMm2, 6),
        "area_error_percent": round(areaErrorPercent, 6),
        "max_allowed_error_percent": round(MAX_TOTAL_AREA_ERROR_RATIO * 100.0, 6),
        "is_valid": bool(isValid),
    }


def printAreaValidationInfo(areaValidationData):
    print()
    print("Flaechenvalidierung:")
    print(
        f"- Erwartete Gesamtflaeche: {areaValidationData['expected_total_area_mm2']:.0f} mm2"
    )
    print(
        f"- Gemessene Gesamtflaeche: {areaValidationData['measured_total_area_mm2']:.0f} mm2"
    )
    print(f"- Fehler: {areaValidationData['area_error_mm2']:.0f} mm2")
    print(f"- Fehler: {areaValidationData['area_error_percent']:.2f} %")
    print(f"- Erlaubt: +/- {areaValidationData['max_allowed_error_percent']:.2f} %")

    if areaValidationData["is_valid"]:
        print("- Ergebnis: OK")
    else:
        print("- Ergebnis: NICHT OK")


# ============================================================
# TEILE-AUSSCHNITTE SPEICHERN
# ============================================================


def buildCropBounds(imageWidth, imageHeight, bboxX, bboxY, bboxW, bboxH):
    x1 = max(0, bboxX - CROP_PADDING_PX)
    y1 = max(0, bboxY - CROP_PADDING_PX)
    x2 = min(imageWidth, bboxX + bboxW + CROP_PADDING_PX)
    y2 = min(imageHeight, bboxY + bboxH + CROP_PADDING_PX)

    return x1, y1, x2, y2


def cropPartImage(warpedImageBgr, bboxX, bboxY, bboxW, bboxH):
    imageHeight, imageWidth = warpedImageBgr.shape[:2]
    cropBounds = buildCropBounds(imageWidth, imageHeight, bboxX, bboxY, bboxW, bboxH)

    x1, y1, x2, y2 = cropBounds
    croppedImageBgr = warpedImageBgr[y1:y2, x1:x2].copy()

    return croppedImageBgr, cropBounds


def buildSinglePartMask(fullBinaryMask, contour, cropBounds):
    # Baut eine Maske für genau ein Teil und schneidet sie auf dessen Bounding Box zu.
    singleMask = np.zeros(fullBinaryMask.shape, dtype=np.uint8)
    cv2.drawContours(singleMask, [contour], -1, 255, -1)

    x1, y1, x2, y2 = cropBounds
    return singleMask[y1:y2, x1:x2].copy()


def getAlgoInputScaleFactor():
    return ALGO_INPUT_PX_PER_MM / WORKING_INTERNAL_PX_PER_MM


def resizeMaskForAlgoInput(maskImage):
    scaleFactor = getAlgoInputScaleFactor()

    if scaleFactor == 1.0:
        return maskImage

    newWidth = max(1, int(round(maskImage.shape[1] * scaleFactor)))
    newHeight = max(1, int(round(maskImage.shape[0] * scaleFactor)))

    return cv2.resize(
        maskImage,
        (newWidth, newHeight),
        interpolation=cv2.INTER_NEAREST,
    )


def saveAlgoInputFiles(binaryMask, detectedParts):
    # Speichert nur die Masken, welche der Algorithmus später als Input braucht.
    # Der Input-Ordner wird bewusst nicht hier geleert, sondern einmal zentral
    # am Anfang von main().
    # Intern wird mit WORKING_INTERNAL_PX_PER_MM gearbeitet.
    # Für den Algorithmus werden die Masken auf ALGO_INPUT_PX_PER_MM runterskaliert.
    algoInputDirPath = buildDirPath(DESTINATION_TO_ALGO_INPUT_FOLDER)
    debugPartsDirPath = buildDirPath(OUTPUT_PARTS_DIR) if SAVE_DEBUG_FILES else None

    for i, partInfo in enumerate(detectedParts):
        cropBounds = (
            partInfo["bboxX"],
            partInfo["bboxY"],
            partInfo["bboxX"] + partInfo["bboxW"],
            partInfo["bboxY"] + partInfo["bboxH"],
        )
        croppedSingleMaskInternal = buildSinglePartMask(
            binaryMask, partInfo["contour"], cropBounds
        )
        croppedSingleMaskAlgo = resizeMaskForAlgoInput(croppedSingleMaskInternal)

        algoMaskFilename = f"{ALGO_INPUT_MASK_PREFIX}{i}.png"
        algoMaskPath = algoInputDirPath / algoMaskFilename

        savePngImage(algoMaskPath, croppedSingleMaskAlgo)

        partInfo["algoInputMaskFilename"] = algoMaskFilename
        partInfo["algoInputMaskPath"] = str(algoMaskPath)

        if debugPartsDirPath is not None:
            debugMaskPath = debugPartsDirPath / algoMaskFilename
            shutil.copy2(algoMaskPath, debugMaskPath)
            partInfo["debugAlgoInputMaskPath"] = str(debugMaskPath)

        # Schwerpunkt in Pixeln passend zur Algorithmus-Skalierung.
        # Die mm-Werte bleiben die Wahrheit; daraus werden die Algo-Pixel berechnet.
        partInfo["algoInputCentroidXpx"] = float(
            partInfo["centroidXmm"] * ALGO_INPUT_PX_PER_MM
        )
        partInfo["algoInputCentroidYpx"] = float(
            partInfo["centroidYmm"] * ALGO_INPUT_PX_PER_MM
        )

        partInfo["algoInputMaskWidthPx"] = int(croppedSingleMaskAlgo.shape[1])
        partInfo["algoInputMaskHeightPx"] = int(croppedSingleMaskAlgo.shape[0])

    return algoInputDirPath


# ============================================================
# JSON-EXPORT
# ============================================================


# Die Json fungiert als Schnittstelle zwischen cam_modul und dem Solver.
# Dh: Der Konsistenz wegen sollten einmal eingetragene Keys ohne Absprache weder umbenannt noch gelöscht werden
# Ergänzungen sind aber möglich, idealerweise in buildAlgoInputJsonData() da die ganze Json Generierung ein Kandidat
# für Refactoring ist ( div Funktionen inkl. debug Infos in buildAlgoInputJsonData() integrieren bzw sammeln)
def buildGeometryJsonData():
    # JSON-Hilfsdaten zur Geometrie. JSON-Key-Namen bleiben bewusst stabil.
    return {
        "a4_size_mm": {
            "width": A4_WIDTH_MM,
            "height": A4_HEIGHT_MM,
            "area_mm2": round(A4_WIDTH_MM * A4_HEIGHT_MM, 6),
        },
        "px_per_mm": ALGO_INPUT_PX_PER_MM,
        "working_internal_px_per_mm": WORKING_INTERNAL_PX_PER_MM,
        "coordinate_system": {
            "origin": COORDINATE_ORIGIN,
            "description": buildCoordinateOriginDescription(),
        },
        "reference_corner_from_marker": REFERENCE_CORNER_FROM_MARKER,
        "frame_offsets_mm": {
            "left": FRAME_OFFSET_LEFT_MM,
            "right": FRAME_OFFSET_RIGHT_MM,
            "top": FRAME_OFFSET_TOP_MM,
            "bottom": FRAME_OFFSET_BOTTOM_MM,
        },
        "a4_corner_extra_offsets_mm": A4_CORNER_EXTRA_OFFSETS_MM,
    }


def buildPartsJsonList(detectedParts, includePaths):
    # Baut die JSON-Liste der Teile. includePaths=True ist nur für Debug-Ausgaben.
    partsJson = []

    for partInfo in detectedParts:
        if includePaths:
            centroidPx = {
                "x": round(partInfo["centroidXpx"], 6),
                "y": round(partInfo["centroidYpx"], 6),
            }
        else:
            centroidPx = {
                "x": round(partInfo["algoInputCentroidXpx"], 6),
                "y": round(partInfo["algoInputCentroidYpx"], 6),
            }

        partData = {
            "index": partInfo["index"],
            "part_name": partInfo["partName"],
            "centroid_mm": {
                "x": round(partInfo["centroidXmm"], 6),
                "y": round(partInfo["centroidYmm"], 6),
            },
            "centroid_px": centroidPx,
            "area_mm2": round(partInfo["areaMm2"], 6),
            "algo_input_mask_filename": partInfo.get("algoInputMaskFilename"),
        }

        if includePaths:
            partData.update(
                {
                    "bounding_box_px": {
                        "x": partInfo["bboxX"],
                        "y": partInfo["bboxY"],
                        "w": partInfo["bboxW"],
                        "h": partInfo["bboxH"],
                    },
                    "algo_input_mask_path": partInfo.get("algoInputMaskPath"),
                    "debug_algo_input_mask_path": partInfo.get(
                        "debugAlgoInputMaskPath"
                    ),
                }
            )

        partsJson.append(partData)

    return partsJson


def isExpectedPartCount(partCount):
    return partCount in EXPECTED_PART_COUNT


def buildDebugJsonData(detectedParts, areaValidationData):
    geometryData = buildGeometryJsonData()

    return {
        "run_name": RUN_NAME,
        "part_count": len(detectedParts),
        "expected_part_counts": EXPECTED_PART_COUNT,
        "part_count_is_valid": isExpectedPartCount(len(detectedParts)),
        **geometryData,
        "expected_total_part_area": {
            "description": "PREN puzzle area without frame",
            "area_mm2": round(EXPECTED_TOTAL_PART_AREA_MM2, 6),
        },
        "area_validation": areaValidationData,
        "segmentation": {
            "threshold_mode": SEGMENTATION_THRESHOLD_MODE,
            "fixed_threshold_value": THRESHOLD_VALUE,
            "adaptive_threshold_block_size": ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            "adaptive_threshold_c": ADAPTIVE_THRESHOLD_C,
            "gaussian_blur_kernel_size": GAUSSIAN_BLUR_KERNEL_SIZE,
            "morph_open_kernel_size": MORPH_OPEN_KERNEL_SIZE,
            "morph_close_kernel_size": MORPH_CLOSE_KERNEL_SIZE,
            "fill_contour_holes": FILL_CONTOUR_HOLES,
            "ignore_border_mm": IGNORE_BORDER_MM,
        },
        "sorting": "smallest_output_y_then_smallest_output_x",
        "parts": buildPartsJsonList(detectedParts, includePaths=True),
    }


def buildAlgoInputJsonData(detectedParts, areaValidationData):
    return {
        "part_count": len(detectedParts),
        "expected_part_count": EXPECTED_PART_COUNT,
        "part_count_is_valid": isExpectedPartCount(len(detectedParts)),
        "coordinate_system": {
            "origin": COORDINATE_ORIGIN,
            "description": buildCoordinateOriginDescription(),
        },
        "px_per_mm": ALGO_INPUT_PX_PER_MM,
        "working_internal_px_per_mm": WORKING_INTERNAL_PX_PER_MM,
        "a4_size_mm": {
            "width": A4_WIDTH_MM,
            "height": A4_HEIGHT_MM,
        },
        "frame_offsets_mm": {
            "left": FRAME_OFFSET_LEFT_MM,
            "right": FRAME_OFFSET_RIGHT_MM,
            "top": FRAME_OFFSET_TOP_MM,
            "bottom": FRAME_OFFSET_BOTTOM_MM,
        },
        "expected_total_part_area_mm2": round(EXPECTED_TOTAL_PART_AREA_MM2, 6),
        "area_validation": areaValidationData,
        "parts": buildPartsJsonList(detectedParts, includePaths=False),
    }


# ============================================================
# DEBUG-ZEICHNUNGEN
# ============================================================


def isMajorGridLine(index, spacingPx, majorSpacingPx):
    if majorSpacingPx <= 0:
        return False

    distancePx = index * spacingPx
    return abs(distancePx % majorSpacingPx) < 0.5


def drawCoordinateGridDebug(debugImageBgr):
    # Zeichnet ein Raster im gleichen Koordinatensystem wie die JSON-Ausgabe.
    # Zeichnet das feste Output-Koordinatensystem direkt ins entzerrte A4-Bild.
    if not DEBUG_DRAW_COORDINATE_GRID:
        return debugImageBgr

    imageHeight, imageWidth = debugImageBgr.shape[:2]
    overlay = debugImageBgr.copy()

    gridSpacingPx = int(round(DEBUG_GRID_SPACING_MM * WORKING_INTERNAL_PX_PER_MM))
    majorSpacingPx = int(
        round(DEBUG_GRID_MAJOR_SPACING_MM * WORKING_INTERNAL_PX_PER_MM)
    )

    if gridSpacingPx <= 0:
        return debugImageBgr

    # Raster passend zum Output-Koordinatensystem:
    # Ursprung oben rechts, x nach links, y nach unten.
    verticalLineIndex = 0
    x = imageWidth - 1
    while x >= 0:
        color = (
            COLOR_GRID_MAJOR
            if isMajorGridLine(verticalLineIndex, gridSpacingPx, majorSpacingPx)
            else COLOR_GRID_MINOR
        )
        thickness = 2 if color == COLOR_GRID_MAJOR else 1
        cv2.line(overlay, (x, 0), (x, imageHeight - 1), color, thickness)
        verticalLineIndex += 1
        x = imageWidth - 1 - verticalLineIndex * gridSpacingPx

    horizontalLineIndex = 0
    y = 0
    while y < imageHeight:
        color = (
            COLOR_GRID_MAJOR
            if isMajorGridLine(horizontalLineIndex, gridSpacingPx, majorSpacingPx)
            else COLOR_GRID_MINOR
        )
        thickness = 2 if color == COLOR_GRID_MAJOR else 1
        cv2.line(overlay, (0, y), (imageWidth - 1, y), color, thickness)
        horizontalLineIndex += 1
        y = horizontalLineIndex * gridSpacingPx

    debugImageBgr = cv2.addWeighted(
        overlay, DEBUG_GRID_ALPHA, debugImageBgr, 1.0 - DEBUG_GRID_ALPHA, 0
    )

    axisLengthPx = int(round(DEBUG_AXIS_LENGTH_MM * WORKING_INTERNAL_PX_PER_MM))
    axisLengthPx = max(gridSpacingPx, axisLengthPx)

    origin = (imageWidth - 1, 0)
    xAxisEnd = (max(0, imageWidth - 1 - axisLengthPx), 0)
    yAxisEnd = (imageWidth - 1, min(imageHeight - 1, axisLengthPx))

    cv2.circle(debugImageBgr, origin, 10, COLOR_AXIS_ORIGIN, -1)
    cv2.arrowedLine(
        debugImageBgr, origin, xAxisEnd, COLOR_AXIS_X, 4, cv2.LINE_AA, tipLength=0.08
    )
    cv2.arrowedLine(
        debugImageBgr, origin, yAxisEnd, COLOR_AXIS_Y, 4, cv2.LINE_AA, tipLength=0.08
    )

    cv2.putText(
        debugImageBgr,
        "origin (0,0)",
        (max(10, imageWidth - 230), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_AXIS_TEXT,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debugImageBgr,
        "+x",
        (max(10, xAxisEnd[0] - 10), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        COLOR_AXIS_X,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        debugImageBgr,
        "+y",
        (max(10, imageWidth - 70), min(imageHeight - 10, yAxisEnd[1] + 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        COLOR_AXIS_Y,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        debugImageBgr,
        "grid: 1 cm x 1 cm",
        (20, imageHeight - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_AXIS_TEXT,
        2,
        cv2.LINE_AA,
    )

    return debugImageBgr


def drawPartsDebug(warpedImageBgr, detectedParts):
    # Zeichnet finale Debug-Ansicht mit Raster, Konturen, Boxen und Schwerpunkten.
    debugImageBgr = warpedImageBgr.copy()
    debugImageBgr = drawCoordinateGridDebug(debugImageBgr)

    for partInfo in detectedParts:
        contour = partInfo["contour"]
        centroidXi = int(round(partInfo["centroidX"]))
        centroidYi = int(round(partInfo["centroidY"]))
        x = partInfo["bboxX"]
        y = partInfo["bboxY"]
        w = partInfo["bboxW"]
        h = partInfo["bboxH"]
        partName = partInfo["partName"]

        cv2.drawContours(debugImageBgr, [contour], -1, PART_CONTOUR_COLOR, 2)
        cv2.rectangle(debugImageBgr, (x, y), (x + w, y + h), PART_BOX_COLOR, 2)
        cv2.circle(
            debugImageBgr,
            (centroidXi, centroidYi),
            PART_CENTROID_RADIUS_PX,
            PART_CENTROID_COLOR,
            -1,
        )

        cv2.putText(
            debugImageBgr,
            partName,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            PART_TEXT_FONT_SCALE,
            PART_TEXT_COLOR,
            PART_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.putText(
            debugImageBgr,
            f"({partInfo['centroidXmm']:.1f}, {partInfo['centroidYmm']:.1f}) mm",
            (centroidXi + 12, centroidYi - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            PART_TEXT_FONT_SCALE,
            PART_TEXT_COLOR,
            PART_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    cv2.putText(
        debugImageBgr,
        f"parts found: {len(detectedParts)}   expected: {EXPECTED_PART_COUNT}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debugImageBgr,
        f"coord: {COORDINATE_ORIGIN}, x left, y down",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debugImageBgr,
        "sorting: smallest output-y, then smallest output-x",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    return debugImageBgr


def drawMarkerDebug(imageBgr, detectedMarkers):
    # Zeichnet Markerumrisse, Marker-IDs und die vier nummerierten Marker-Ecken.
    debugImageBgr = imageBgr.copy()
    cornerColors = [COLOR_CORNER_0, COLOR_CORNER_1, COLOR_CORNER_2, COLOR_CORNER_3]

    for markerId in sorted(detectedMarkers.keys()):
        markerCorners = detectedMarkers[markerId]["corners"]
        ptsInt = np.round(markerCorners).astype(int)

        cv2.polylines(
            debugImageBgr, [ptsInt.reshape((-1, 1, 2))], True, COLOR_MARKER_OUTLINE, 2
        )

        centerX = int(round(np.mean(markerCorners[:, 0])))
        centerY = int(round(np.mean(markerCorners[:, 1])))

        cv2.circle(
            debugImageBgr,
            (centerX, centerY),
            MARKER_CENTER_RADIUS_PX,
            COLOR_MARKER_CENTER,
            -1,
        )

        cv2.putText(
            debugImageBgr,
            f"ID {markerId}",
            (centerX + 10, centerY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_FONT_SCALE,
            COLOR_MARKER_ID_TEXT,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        for cornerIndex in range(4):
            x = int(round(markerCorners[cornerIndex, 0]))
            y = int(round(markerCorners[cornerIndex, 1]))

            cv2.circle(
                debugImageBgr,
                (x, y),
                CORNER_CIRCLE_RADIUS_PX,
                cornerColors[cornerIndex],
                -1,
            )

            cv2.putText(
                debugImageBgr,
                f"{cornerIndex}",
                (x + CORNER_TEXT_OFFSET_X, y + CORNER_TEXT_OFFSET_Y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cornerColors[cornerIndex],
                2,
                cv2.LINE_AA,
            )

    return debugImageBgr


def drawCornerPolygon(
    debugImageBgr, corners, colorPolyline, colorPoint, radius, labelPrefix
):
    ptTL = np.round(corners["top_left"]).astype(int)
    ptTR = np.round(corners["top_right"]).astype(int)
    ptBR = np.round(corners["bottom_right"]).astype(int)
    ptBL = np.round(corners["bottom_left"]).astype(int)

    polygon = np.array([ptTL, ptTR, ptBR, ptBL], dtype=np.int32)
    cv2.polylines(debugImageBgr, [polygon.reshape((-1, 1, 2))], True, colorPolyline, 2)

    labeledPoints = [
        (f"{labelPrefix} TL", ptTL),
        (f"{labelPrefix} TR", ptTR),
        (f"{labelPrefix} BR", ptBR),
        (f"{labelPrefix} BL", ptBL),
    ]

    for label, pt in labeledPoints:
        x = int(pt[0])
        y = int(pt[1])

        cv2.circle(debugImageBgr, (x, y), radius, colorPoint, -1)
        cv2.putText(
            debugImageBgr,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colorPoint,
            2,
            cv2.LINE_AA,
        )


def drawA4AndReferenceDebug(imageBgr, a4Corners, referenceCorners):
    debugImageBgr = imageBgr.copy()

    drawCornerPolygon(
        debugImageBgr,
        referenceCorners,
        COLOR_REFERENCE_POLYLINE,
        COLOR_REFERENCE_POINT,
        REFERENCE_CORNER_RADIUS_PX,
        "REF",
    )

    drawCornerPolygon(
        debugImageBgr,
        a4Corners,
        COLOR_A4_POLYLINE,
        COLOR_A4_POINT,
        A4_CORNER_RADIUS_PX,
        "A4",
    )

    return debugImageBgr


def buildReferenceStatusText():
    parts = []

    for cornerName in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        mapping = REFERENCE_CORNER_FROM_MARKER[cornerName]
        parts.append(
            f"{cornerName}=ID{mapping['marker_id']}/C{mapping['corner_index']}"
        )

    return "  ".join(parts)


def buildOffsetStatusText():
    return (
        f"offset L/R/T/B = "
        f"{FRAME_OFFSET_LEFT_MM:.1f}/"
        f"{FRAME_OFFSET_RIGHT_MM:.1f}/"
        f"{FRAME_OFFSET_TOP_MM:.1f}/"
        f"{FRAME_OFFSET_BOTTOM_MM:.1f} mm"
    )


def buildRotationStatusText():
    if ROTATE_90_CLOCKWISE:
        return "rotation: 90_cw"

    if ROTATE_180:
        return "rotation: 180"

    return "rotation: none"


def drawCombinedDebug(imageBgr, detectedMarkers, a4Corners, referenceCorners):
    # Kombiniert Markerdebug, Referenzrahmen, echte A4-Ecken und Status-Texte.
    debugImageBgr = drawMarkerDebug(imageBgr, detectedMarkers)
    debugImageBgr = drawA4AndReferenceDebug(debugImageBgr, a4Corners, referenceCorners)

    statusText1 = "A4 corners calculated from ArUco reference corners plus mm offsets"
    statusText2 = buildReferenceStatusText()
    statusText3 = buildOffsetStatusText()
    statusText4 = buildRotationStatusText()

    cv2.putText(
        debugImageBgr,
        statusText1,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_STATUS_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debugImageBgr,
        statusText2,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_STATUS_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debugImageBgr,
        statusText3,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_STATUS_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debugImageBgr,
        statusText4,
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_STATUS_TEXT,
        2,
        cv2.LINE_AA,
    )

    return debugImageBgr


# ============================================================
# KONSOLENAUSGABEN
# ============================================================


def printConsoleMarkerInfo(detectedMarkers):
    print()
    print("Erkannte Marker:")

    if len(detectedMarkers) == 0:
        print("- keine")
        return

    for markerId in sorted(detectedMarkers.keys()):
        markerCorners = detectedMarkers[markerId]["corners"]

        print(f"- ID {markerId}")
        for cornerIndex in range(4):
            x = markerCorners[cornerIndex, 0]
            y = markerCorners[cornerIndex, 1]
            print(f"  Ecke {cornerIndex}: x={x:.1f}, y={y:.1f}")


def printConsoleCornerInfo(title, corners):
    print()
    print(title)

    for name in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        x = corners[name][0]
        y = corners[name][1]
        print(f"- {name}: x={x:.1f}, y={y:.1f}")


def printConsoleGeometryInfo():
    frameWidthMm, frameHeightMm = getFrameSizeMm()

    print()
    print("Geometrie:")
    print(f"- A4: {A4_WIDTH_MM:.1f} x {A4_HEIGHT_MM:.1f} mm")
    print(f"- Referenz/Rahmen: {frameWidthMm:.1f} x {frameHeightMm:.1f} mm")
    print(f"- Offset links: {FRAME_OFFSET_LEFT_MM:.1f} mm")
    print(f"- Offset rechts: {FRAME_OFFSET_RIGHT_MM:.1f} mm")
    print(f"- Offset oben: {FRAME_OFFSET_TOP_MM:.1f} mm")
    print(f"- Offset unten: {FRAME_OFFSET_BOTTOM_MM:.1f} mm")


def printConsoleCoordinateSystemInfo():
    print()
    print("Koordinatensystem:")
    print(f"- Ursprung: {COORDINATE_ORIGIN}")
    print(f"- Beschreibung: {buildCoordinateOriginDescription()}")


def printConsoleHomographyInfo(hImageToWarp):
    print()
    print("Homographie Bild -> Warp-Pixel:")
    print(hImageToWarp)

    warpWidthPx, warpHeightPx = getWarpSizePx()

    print()
    print(f"Warp-Groesse: {warpWidthPx} x {warpHeightPx} px")
    print(f"PX_PER_MM: {WORKING_INTERNAL_PX_PER_MM}")


def printConsolePartsInfo(detectedParts):
    print()
    print("Erkannte Teile:")

    if len(detectedParts) == 0:
        print("- keine")
        return

    for partInfo in detectedParts:
        print(f"- {partInfo['partName']}")
        print(
            f"  Schwerpunkt A4-mm: x={partInfo['centroidXmm']:.3f}, y={partInfo['centroidYmm']:.3f}"
        )
        print(
            f"  Schwerpunkt A4-px: x={partInfo['centroidXpx']:.3f}, y={partInfo['centroidYpx']:.3f}"
        )
        print(
            f"  Schwerpunkt Warp-Pixel debug: x={partInfo['centroidX']:.3f}, y={partInfo['centroidY']:.3f}"
        )
        print(f"  Flaeche: {partInfo['areaMm2']:.3f} mm2")
        print(
            f"  Bounding Box: x={partInfo['bboxX']}, y={partInfo['bboxY']}, w={partInfo['bboxW']}, h={partInfo['bboxH']}"
        )
        if SAVE_DEBUG_FILES:
            print(
                f"  Debug-Kopie Algorithmus-Maske: {partInfo.get('debugAlgoInputMaskPath')}"
            )
        print(f"  Algorithmus-Maske: {partInfo.get('algoInputMaskPath')}")

    print()
    print(f"Anzahl Teile: {len(detectedParts)}")
    print(f"Erwartet: {EXPECTED_PART_COUNT}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================


def main(cam=None):
    try:
        # Bewusst am Anfang. Der Solver sieht innerhalb eines runs niemals zu keinem Zeitpunkt etwas,
        # das nicht explizit innerhalb des aktuellen runs abgesegnet wurde.
        clearAlgoInputFolder()
        print(f"Algorithmus-Input-Ordner geleert: {DESTINATION_TO_ALGO_INPUT_FOLDER}")
        clearDebugOutputFolder()
        print(f"Debug-Output-Ordner geleert: {OUTPUT_DIR}")
        imageBgr = getInputImage(cam)
        imageBgr = rotateImageIfNeeded(imageBgr)

        if SAVE_DEBUG_FILES:
            outputImagePath = buildOutputPath(OUTPUT_IMAGE_FILENAME)
            savePngImage(outputImagePath, imageBgr)
            print(f"Input-Bild gespeichert: {outputImagePath}")

        print(f"Bildgrösse: {imageBgr.shape[1]} x {imageBgr.shape[0]} Pixel")
        print(buildRotationStatusText())
        printConsoleGeometryInfo()
        printConsoleCoordinateSystemInfo()

        detectedMarkers, rejectedCandidates = detectArucoMarkers(imageBgr)
        printConsoleMarkerInfo(detectedMarkers)

        a4Corners, referenceCorners = extractA4Corners(detectedMarkers)
        printConsoleCornerInfo(
            "Referenz-/Rahmen-Bildecken aus ArUcos:", referenceCorners
        )
        printConsoleCornerInfo("Abgeleitete echte A4-Bildecken:", a4Corners)

        native_density = calculate_native_a4_pixel_density(a4Corners)
        print("Native Pixeldichte der A4-Fläche:")
        print(f"- x:   {native_density['native_px_per_mm_x']:.2f} px/mm")
        print(f"- y:   {native_density['native_px_per_mm_y']:.2f} px/mm")
        print(f"- avg: {native_density['native_px_per_mm_avg']:.2f} px/mm")

        if SAVE_DEBUG_FILES:
            outputDebugPath = buildOutputPath(OUTPUT_DEBUG_FILENAME)
            debugImageBgr = drawCombinedDebug(
                imageBgr, detectedMarkers, a4Corners, referenceCorners
            )
            savePngImage(outputDebugPath, debugImageBgr)
            print(f"Debug-Bild gespeichert: {outputDebugPath}")

        hImageToWarp = computeHomographyImageToWarp(a4Corners)
        printConsoleHomographyInfo(hImageToWarp)

        if SAVE_DEBUG_FILES:
            outputHImageToWarpPath = buildOutputPath(OUTPUT_H_IMAGE_TO_WARP_PATH)
            np.save(str(outputHImageToWarpPath), hImageToWarp)
            print(f"H gespeichert: {outputHImageToWarpPath}")

        warpedImageBgr = warpImageToA4(imageBgr, hImageToWarp)

        if SAVE_DEBUG_FILES:
            outputWarpPath = buildOutputPath(OUTPUT_WARP_FILENAME)
            savePngImage(outputWarpPath, warpedImageBgr)
            print(f"Warp-Bild gespeichert: {outputWarpPath}")

        binaryMask = buildPartsMask(warpedImageBgr)
        if CALCULATE_AREA_WITHOUT_SIDES:
            binaryMask = calculate_puzzle_piece_shape_without_sides(binaryMask, CAM_HEIGHT)
        if SAVE_DEBUG_FILES:
            outputMaskPath = buildOutputPath(OUTPUT_MASK_FILENAME)
            savePngImage(outputMaskPath, binaryMask)
            print(f"Maske gespeichert: {outputMaskPath}")

        detectedParts = findAllValidParts(binaryMask)
        detectedParts = sortPartsByOutputYThenOutputX(detectedParts)
        addDerivedPartValues(detectedParts)

        areaValidationData = buildAreaValidationData(detectedParts)

        partCountIsValid = isExpectedPartCount(len(detectedParts))
        areaIsValid = areaValidationData["is_valid"]
        detectionIsValid = partCountIsValid and areaIsValid

        printAreaValidationInfo(areaValidationData)

        if SAVE_DEBUG_FILES:
            outputPartsDebugPath = buildOutputPath(OUTPUT_PARTS_DEBUG_FILENAME)
            partsDebugImageBgr = drawPartsDebug(warpedImageBgr, detectedParts)
            savePngImage(outputPartsDebugPath, partsDebugImageBgr)
            print(f"Teile-Debug-Bild gespeichert: {outputPartsDebugPath}")

        if detectionIsValid:
            algoInputDirPath = saveAlgoInputFiles(binaryMask, detectedParts)

            algoJsonData = buildAlgoInputJsonData(detectedParts, areaValidationData)
            algoJsonPath = algoInputDirPath / ALGO_INPUT_JSON_FILENAME
            saveJson(algoJsonPath, algoJsonData)

            print(f"Algorithmus-Input gespeichert: {algoInputDirPath}")
        else:
            print()
            print("Algorithmus-Input wurde NICHT gespeichert.")
            print(
                "Der Input-Ordner bleibt leer, damit der Solver nicht mit alten Daten weiterläuft."
            )
            print(f"- Teileanzahl gültig: {partCountIsValid}")
            print(f"- Fläche gültig: {areaIsValid}")

        printConsolePartsInfo(detectedParts)

        if SAVE_DEBUG_FILES:
            outputJsonPath = buildOutputPath(OUTPUT_JSON_FILENAME)
            debugJsonData = buildDebugJsonData(detectedParts, areaValidationData)
            saveJson(outputJsonPath, debugJsonData)
            print(f"Debug-JSON gespeichert: {outputJsonPath}")

    except Exception as e:
        print("Fehler:")
        print(e)


if __name__ == "__main__":
    main()
