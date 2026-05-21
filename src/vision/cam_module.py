# cam_module.py
# Kamera-/Datei-Eingang, ArUco-basierte A4-Entzerrung, Teile-Segmentierung
# und Export der Algorithmus-Eingaben.

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except (ImportError, ModuleNotFoundError):
    Picamera2 = None


# ============================================================
# PROJEKT / DATEIEN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/vision/cam_module.py -> project root

# Zielordner fuer die Daten, die an den Puzzle-Algorithmus gehen.
# In diesen Ordner werden parts.json und die Teilmasken geschrieben.
DESTINATION_TO_ALGO_INPUT_FOLDER = PROJECT_ROOT / "input"

# Eingabebild, falls IMAGE_SOURCE = "file" oder keine Pi Camera vorhanden ist.
INPUT_IMAGE_PATH = PROJECT_ROOT / "1.png"

ALGO_INPUT_JSON_FILENAME = "parts.json"
ALGO_INPUT_MASK_PREFIX = "piece_"
CLEAR_ALGO_INPUT_FOLDER_BEFORE_SAVE = True


# ============================================================
# BILDEINGABE
# ============================================================

# "camera" = neues Bild mit Pi Camera 3 aufnehmen
# "file"   = bestehendes Bild von Datei laden
IMAGE_SOURCE = "camera"

IMAGE_WIDTH = 4608
IMAGE_HEIGHT = 2592
STARTUP_WAIT_SECONDS = 3.0

ROTATE_90_CLOCKWISE = False
ROTATE_180 = False

# Manuelle Kameraeinstellungen fuer Tests gegen Ueberbelichtung.
CAMERA_CONTROLS = {
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": 6000,  # Mikrosekunden
    "AnalogueGain": 1.0,
    "ColourGains": (1.5, 1.5),
}


# ============================================================
# AUSGABE / DEBUG-DATEIEN
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PARTS_DIR = OUTPUT_DIR / "parts"
OUTPUT_PART_MASKS_DIR = OUTPUT_DIR / "part_masks"
OUTPUT_PART_CUTOUTS_DIR = OUTPUT_DIR / "part_cutouts"

RUN_NAME = "step_09"

OUTPUT_IMAGE_FILENAME = f"{RUN_NAME}_input.png"
OUTPUT_DEBUG_FILENAME = f"{RUN_NAME}_a4_corners_debug.png"
OUTPUT_WARP_FILENAME = f"{RUN_NAME}_warp_a4.png"
OUTPUT_MASK_FILENAME = f"{RUN_NAME}_parts_mask.png"
OUTPUT_PARTS_DEBUG_FILENAME = f"{RUN_NAME}_parts_debug.png"
OUTPUT_JSON_FILENAME = f"{RUN_NAME}_parts.json"
OUTPUT_H_IMAGE_TO_WARP_PATH = f"{RUN_NAME}_h_image_to_warp.npy"

DEBUG_SHOW_IMAGES = False
DEBUG_WAIT_MS = 1000

INPUT_WINDOW_NAME = "Input Image"
DEBUG_WINDOW_NAME = "A4 Corner Debug Image"
WARP_WINDOW_NAME = "Warped A4 Image"
MASK_WINDOW_NAME = "Parts Mask"
PARTS_DEBUG_WINDOW_NAME = "Parts Debug"


# ============================================================
# ARUCO / A4-GEOMETRIE
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_4X4_50
REQUIRED_IDS = [0, 1, 2, 3]

# A4 im Querformat
A4_WIDTH_MM = 297.0
A4_HEIGHT_MM = 210.0
PX_PER_MM = 10.0

# ------------------------------------------------------------
# Marker-Ecken-Zuordnung
# ------------------------------------------------------------
# Diese Punkte sind die gemessenen Referenzpunkte im Bild.
# Bei OFFSET = 0.0 sind diese Referenzpunkte direkt die echten A4-Ecken.
#
# OpenCV-ArUco-Corner-Indizes normalerweise:
# 0 = oben links, 1 = oben rechts, 2 = unten rechts, 3 = unten links
# bezogen auf den Marker selbst im Bild.
#
# Deine aktuelle funktionierende Zuordnung:
# A4 top_right    = ID 0 / Ecke 3
# A4 bottom_right = ID 1 / Ecke 2
# A4 bottom_left  = ID 2 / Ecke 1
# A4 top_left     = ID 3 / Ecke 0
REFERENCE_CORNER_FROM_MARKER = {
    "top_left": {"marker_id": 3, "corner_index": 0},
    "top_right": {"marker_id": 0, "corner_index": 3},
    "bottom_right": {"marker_id": 1, "corner_index": 2},
    "bottom_left": {"marker_id": 2, "corner_index": 1},
}

# ------------------------------------------------------------
# A4-Offset / Rahmen-Offset
# ------------------------------------------------------------
# Diese Werte beschreiben, wie weit die gemessenen Referenzpunkte ausserhalb
# der echten A4-Flaeche liegen.
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
FRAME_OFFSET_LEFT_MM = 0.0
FRAME_OFFSET_RIGHT_MM = 0.0
FRAME_OFFSET_TOP_MM = 0.0
FRAME_OFFSET_BOTTOM_MM = 0.0

# Optionaler Feintrimm pro echter A4-Ecke.
# Koordinaten in der Referenz-/Rahmenflaeche:
# +x = nach rechts, +y = nach unten.
#
# Normalerweise alles 0.0 lassen.
# Nur verwenden, wenn einzelne Marker mechanisch anders sitzen.
A4_CORNER_EXTRA_OFFSETS_MM = {
    "top_left": {"x": 0.0, "y": 0.0},
    "top_right": {"x": 0.0, "y": 0.0},
    "bottom_right": {"x": 0.0, "y": 0.0},
    "bottom_left": {"x": 0.0, "y": 0.0},
}

# Optionaler Rand, der bei der Teile-Erkennung ignoriert wird.
IGNORE_BORDER_MM = 0.0


# ============================================================
# KOORDINATENSYSTEM FUER OUTPUT
# ============================================================

# Fix: Ursprung oben rechts, x nach links, y nach unten.
COORDINATE_ORIGIN = "top_right"


# ============================================================
# TEILE-SEGMENTIERUNG
# ============================================================

SEGMENTATION_THRESHOLD_MODE = "otsu"  # "fixed", "otsu", "adaptive"
THRESHOLD_VALUE = 150
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 101
ADAPTIVE_THRESHOLD_C = 8

GAUSSIAN_BLUR_KERNEL_SIZE = 7

MIN_PART_AREA_MM2 = 4500.0
MAX_PART_AREA_MM2 = 100000.0

MORPH_OPEN_KERNEL_SIZE = 5
MORPH_CLOSE_KERNEL_SIZE = 7
FILL_CONTOUR_HOLES = True

CROP_PADDING_PX = 0
EXPECTED_PART_COUNT = 4
CUTOUT_BACKGROUND_VALUE = 255


# ============================================================
# VALIDIERUNG
# ============================================================

# PREN-Puzzle ohne Rahmen: 18.9 x 12.6 cm
EXPECTED_TOTAL_PART_AREA_MM2 = 189 * 126
MAX_TOTAL_AREA_ERROR_RATIO = 0.01


# ============================================================
# DEBUG-FARBEN UND DARSTELLUNG
# ============================================================

PART_CONTOUR_COLOR = (0, 255, 0)
PART_CENTROID_COLOR = (0, 0, 255)
PART_BOX_COLOR = (255, 255, 0)
PART_TEXT_COLOR = (255, 255, 255)

PART_CENTROID_RADIUS_PX = 8
PART_TEXT_FONT_SCALE = 0.7
PART_TEXT_THICKNESS = 2

COLOR_MARKER_OUTLINE = (0, 255, 0)
COLOR_MARKER_CENTER = (255, 0, 0)
COLOR_MARKER_ID_TEXT = (0, 255, 0)

COLOR_CORNER_0 = (0, 0, 255)
COLOR_CORNER_1 = (0, 255, 255)
COLOR_CORNER_2 = (255, 255, 0)
COLOR_CORNER_3 = (255, 0, 255)

COLOR_A4_POINT = (0, 165, 255)
COLOR_REFERENCE_POINT = (255, 128, 0)
COLOR_A4_TEXT = (0, 165, 255)
COLOR_A4_POLYLINE = (255, 255, 255)
COLOR_REFERENCE_POLYLINE = (255, 128, 0)
COLOR_STATUS_TEXT = (255, 255, 255)

CORNER_CIRCLE_RADIUS_PX = 6
MARKER_CENTER_RADIUS_PX = 6
A4_CORNER_RADIUS_PX = 10
REFERENCE_CORNER_RADIUS_PX = 7

CORNER_TEXT_OFFSET_X = 8
CORNER_TEXT_OFFSET_Y = -8

TEXT_FONT_SCALE = 0.8
TEXT_THICKNESS = 2

# Koordinatensystem / Raster in der parts_debug-Ausgabe
DEBUG_DRAW_COORDINATE_GRID = True
DEBUG_GRID_SPACING_MM = 10.0       # 10 mm = 1 cm
DEBUG_GRID_MAJOR_SPACING_MM = 50.0 # staerkere Linie alle 5 cm
DEBUG_GRID_ALPHA = 0.35
DEBUG_AXIS_LENGTH_MM = 50.0

COLOR_GRID_MINOR = (120, 120, 120)
COLOR_GRID_MAJOR = (180, 180, 180)
COLOR_AXIS_X = (0, 0, 255)
COLOR_AXIS_Y = (0, 255, 0)
COLOR_AXIS_ORIGIN = (255, 255, 255)
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
    algoInputDirPath = buildDirPath(DESTINATION_TO_ALGO_INPUT_FOLDER)

    if not CLEAR_ALGO_INPUT_FOLDER_BEFORE_SAVE:
        return algoInputDirPath

    for path in algoInputDirPath.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    return algoInputDirPath


def showImage(windowName, image, waitMs):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, image)
    cv2.waitKey(waitMs)
    cv2.destroyWindow(windowName)


def showDebugImage(windowName, image):
    if DEBUG_SHOW_IMAGES:
        showImage(windowName, image, DEBUG_WAIT_MS)


def rotateImageIfNeeded(imageBgr):
    if ROTATE_90_CLOCKWISE and ROTATE_180:
        raise ValueError("Nur eine Rotation aktivieren: entweder ROTATE_90_CLOCKWISE oder ROTATE_180.")

    if ROTATE_90_CLOCKWISE:
        return cv2.rotate(imageBgr, cv2.ROTATE_90_CLOCKWISE)

    if ROTATE_180:
        return cv2.rotate(imageBgr, cv2.ROTATE_180)

    return imageBgr


# ============================================================
# BILDEINGABE
# ============================================================

def isPiCameraAvailable():
    return Picamera2 is not None


def captureImageFromCamera():
    picam2 = None

    try:
        print("Initialisiere Kamera...")
        picam2 = Picamera2()

        cameraConfig = picam2.create_still_configuration(
            main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)}
        )
        picam2.configure(cameraConfig)
        picam2.set_controls(CAMERA_CONTROLS)

        print("Starte Kamera...")
        picam2.start()

        print(f"Warte {STARTUP_WAIT_SECONDS:.1f} Sekunden...")
        time.sleep(STARTUP_WAIT_SECONDS)

        print("Nehme Bild auf...")
        imageBgr = picam2.capture_array()

        grayImage = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)
        overexposedPixels = np.sum(grayImage >= 250)
        totalPixels = grayImage.shape[0] * grayImage.shape[1]
        overexposedRatio = overexposedPixels / totalPixels

        print(f"Ueberbelichtete Pixel: {overexposedRatio * 100.0:.2f} %")

        return imageBgr

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
                print("Kamera gestoppt.")
            except Exception:
                pass


def loadImageFromFile():
    inputPath = Path(INPUT_IMAGE_PATH)

    if not inputPath.exists():
        raise FileNotFoundError(f"Eingabebild nicht gefunden: {inputPath}")

    print(f"Lade Bild von Datei: {inputPath}")

    imageBgr = cv2.imread(str(inputPath), cv2.IMREAD_COLOR)

    if imageBgr is None:
        raise RuntimeError(f"cv2.imread konnte das Bild nicht laden: {inputPath}")

    return imageBgr


def getInputImage():
    if IMAGE_SOURCE == "camera":
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
    missingIds = [markerId for markerId in REQUIRED_IDS if markerId not in detectedMarkers]

    if missingIds:
        raise RuntimeError(f"Nicht alle benoetigten Marker wurden erkannt. Fehlend: {missingIds}")


def getReferenceCornersFromMarkers(detectedMarkers):
    validateDetectedMarkers(detectedMarkers)

    referenceCorners = {}

    for cornerName, mapping in REFERENCE_CORNER_FROM_MARKER.items():
        markerId = mapping["marker_id"]
        cornerIndex = mapping["corner_index"]

        markerCorners = detectedMarkers[markerId]["corners"]
        referenceCorners[cornerName] = markerCorners[cornerIndex].astype(np.float32)

    return referenceCorners


def getFrameSizeMm():
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
    # 1. Aus den Markern die gemessene Referenz-/Rahmenflaeche im Bild holen.
    referenceCorners = getReferenceCornersFromMarkers(detectedMarkers)

    # 2. Homographie: Referenz-/Rahmen-mm -> Bildpixel.
    referenceImagePoints = buildReferenceCornerArrayImage(referenceCorners)
    referenceMmPoints = buildReferenceCornerArrayMm()
    hReferenceMmToImage = cv2.getPerspectiveTransform(referenceMmPoints, referenceImagePoints)

    # 3. Echte A4-Ecken innerhalb der Referenz-/Rahmenflaeche in mm definieren.
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

    # 4. Echte A4-Ecken zurueck ins Bild projizieren.
    a4ImageArray = transformPoints(a4ReferenceMmArray, hReferenceMmToImage)

    a4Corners = {
        "top_left": a4ImageArray[0].astype(np.float32),
        "top_right": a4ImageArray[1].astype(np.float32),
        "bottom_right": a4ImageArray[2].astype(np.float32),
        "bottom_left": a4ImageArray[3].astype(np.float32),
    }

    return a4Corners, referenceCorners


# ============================================================
# HOMOGRAPHIE / KOORDINATEN
# ============================================================

def getWarpSizePx():
    warpWidthPx = int(round(A4_WIDTH_MM * PX_PER_MM))
    warpHeightPx = int(round(A4_HEIGHT_MM * PX_PER_MM))
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
    warpWidthPx, _ = getWarpSizePx()

    xA4Px = (warpWidthPx - 1) - float(xPx)
    yA4Px = float(yPx)

    return xA4Px, yA4Px


def outputPxToOutputMm(xPx, yPx):
    return float(xPx) / PX_PER_MM, float(yPx) / PX_PER_MM


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
    if IGNORE_BORDER_MM <= 0:
        return binaryMask

    borderPx = int(round(IGNORE_BORDER_MM * PX_PER_MM))

    if borderPx <= 0:
        return binaryMask

    maskedBinaryMask = binaryMask.copy()
    imageHeight, imageWidth = maskedBinaryMask.shape[:2]

    maskedBinaryMask[0:borderPx, :] = 0
    maskedBinaryMask[imageHeight - borderPx:imageHeight, :] = 0
    maskedBinaryMask[:, 0:borderPx] = 0
    maskedBinaryMask[:, imageWidth - borderPx:imageWidth] = 0

    return maskedBinaryMask


def fillMaskContourHoles(binaryMask):
    contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filledMask = np.zeros(binaryMask.shape, dtype=np.uint8)
    cv2.drawContours(filledMask, contours, -1, 255, -1)

    return filledMask


def buildPartsMask(warpedImageBgr):
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
        raise ValueError('SEGMENTATION_THRESHOLD_MODE muss "fixed", "otsu" oder "adaptive" sein.')

    openKernel = np.ones((MORPH_OPEN_KERNEL_SIZE, MORPH_OPEN_KERNEL_SIZE), np.uint8)
    closeKernel = np.ones((MORPH_CLOSE_KERNEL_SIZE, MORPH_CLOSE_KERNEL_SIZE), np.uint8)

    binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_OPEN, openKernel)
    binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_CLOSE, closeKernel)

    if FILL_CONTOUR_HOLES:
        binaryMask = fillMaskContourHoles(binaryMask)

    binaryMask = applyIgnoreBorder(binaryMask)

    return binaryMask


def computeContourCentroid(contour):
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + (w / 2.0), y + (h / 2.0)

    centroidX = moments["m10"] / moments["m00"]
    centroidY = moments["m01"] / moments["m00"]

    return centroidX, centroidY


def findAllValidParts(binaryMask):
    contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minAreaPx = MIN_PART_AREA_MM2 * (PX_PER_MM ** 2)
    maxAreaPx = MAX_PART_AREA_MM2 * (PX_PER_MM ** 2)

    detectedParts = []

    for contour in contours:
        areaPx = cv2.contourArea(contour)

        if areaPx < minAreaPx:
            continue

        if areaPx > maxAreaPx:
            continue

        centroidX, centroidY = computeContourCentroid(contour)
        x, y, w, h = cv2.boundingRect(contour)

        detectedParts.append({
            "contour": contour,
            "areaPx": float(areaPx),
            "centroidX": float(centroidX),
            "centroidY": float(centroidY),
            "bboxX": int(x),
            "bboxY": int(y),
            "bboxW": int(w),
            "bboxH": int(h),
        })

    return detectedParts


def sortPartsByOutputYThenOutputX(detectedParts):
    def sortKey(partInfo):
        centroidXpxOutput, centroidYpxOutput = warpPxToOutputPxTopRight(
            partInfo["centroidX"],
            partInfo["centroidY"],
        )
        centroidXmm, centroidYmm = outputPxToOutputMm(centroidXpxOutput, centroidYpxOutput)
        return centroidYmm, centroidXmm

    return sorted(detectedParts, key=sortKey)


def addDerivedPartValues(detectedParts):
    for i, partInfo in enumerate(detectedParts):
        centroidXpxOutput, centroidYpxOutput = warpPxToOutputPxTopRight(
            partInfo["centroidX"],
            partInfo["centroidY"],
        )
        centroidXmm, centroidYmm = outputPxToOutputMm(centroidXpxOutput, centroidYpxOutput)
        areaMm2 = partInfo["areaPx"] / (PX_PER_MM ** 2)

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
    print(f"- Erwartete Gesamtflaeche: {areaValidationData['expected_total_area_mm2']:.0f} mm2")
    print(f"- Gemessene Gesamtflaeche: {areaValidationData['measured_total_area_mm2']:.0f} mm2")
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
    singleMask = np.zeros(fullBinaryMask.shape, dtype=np.uint8)
    cv2.drawContours(singleMask, [contour], -1, 255, -1)

    x1, y1, x2, y2 = cropBounds
    return singleMask[y1:y2, x1:x2].copy()


def buildPartCutout(croppedImageBgr, croppedSingleMask):
    cutoutImageBgr = np.full_like(croppedImageBgr, CUTOUT_BACKGROUND_VALUE)
    cutoutImageBgr[croppedSingleMask > 0] = croppedImageBgr[croppedSingleMask > 0]

    return cutoutImageBgr


def savePartOutputs(warpedImageBgr, binaryMask, detectedParts):
    partsDirPath = buildDirPath(OUTPUT_PARTS_DIR)
    partMasksDirPath = buildDirPath(OUTPUT_PART_MASKS_DIR)
    partCutoutsDirPath = buildDirPath(OUTPUT_PART_CUTOUTS_DIR)

    for partInfo in detectedParts:
        croppedImageBgr, cropBounds = cropPartImage(
            warpedImageBgr,
            partInfo["bboxX"],
            partInfo["bboxY"],
            partInfo["bboxW"],
            partInfo["bboxH"],
        )

        croppedSingleMask = buildSinglePartMask(binaryMask, partInfo["contour"], cropBounds)
        cutoutImageBgr = buildPartCutout(croppedImageBgr, croppedSingleMask)

        outputPartPath = partsDirPath / f"{partInfo['partName']}.png"
        outputPartMaskPath = partMasksDirPath / f"{partInfo['partName']}_mask.png"
        outputPartCutoutPath = partCutoutsDirPath / f"{partInfo['partName']}_cutout.png"

        savePngImage(outputPartPath, croppedImageBgr)
        savePngImage(outputPartMaskPath, croppedSingleMask)
        savePngImage(outputPartCutoutPath, cutoutImageBgr)

        partInfo["outputPath"] = str(outputPartPath)
        partInfo["maskPath"] = str(outputPartMaskPath)
        partInfo["cutoutPath"] = str(outputPartCutoutPath)


def saveAlgoInputFiles(binaryMask, detectedParts):
    algoInputDirPath = clearAlgoInputFolder()

    for i, partInfo in enumerate(detectedParts):
        cropBounds = (
            partInfo["bboxX"],
            partInfo["bboxY"],
            partInfo["bboxX"] + partInfo["bboxW"],
            partInfo["bboxY"] + partInfo["bboxH"],
        )

        croppedSingleMask = buildSinglePartMask(binaryMask, partInfo["contour"], cropBounds)

        algoMaskFilename = f"{ALGO_INPUT_MASK_PREFIX}{i}.png"
        algoMaskPath = algoInputDirPath / algoMaskFilename

        savePngImage(algoMaskPath, croppedSingleMask)

        partInfo["algoInputMaskFilename"] = algoMaskFilename
        partInfo["algoInputMaskPath"] = str(algoMaskPath)

    return algoInputDirPath


# ============================================================
# JSON-EXPORT
# ============================================================

def buildGeometryJsonData():
    return {
        "a4_size_mm": {
            "width": A4_WIDTH_MM,
            "height": A4_HEIGHT_MM,
            "area_mm2": round(A4_WIDTH_MM * A4_HEIGHT_MM, 6),
        },
        "px_per_mm": PX_PER_MM,
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
    partsJson = []

    for partInfo in detectedParts:
        partData = {
            "index": partInfo["index"],
            "part_name": partInfo["partName"],
            "centroid_mm": {
                "x": round(partInfo["centroidXmm"], 6),
                "y": round(partInfo["centroidYmm"], 6),
            },
            "centroid_px": {
                "x": round(partInfo["centroidXpx"], 6),
                "y": round(partInfo["centroidYpx"], 6),
            },
            "area_mm2": round(partInfo["areaMm2"], 6),
            "bounding_box_px": {
                "x": partInfo["bboxX"],
                "y": partInfo["bboxY"],
                "w": partInfo["bboxW"],
                "h": partInfo["bboxH"],
            },
            "algo_input_mask_filename": partInfo.get("algoInputMaskFilename"),
        }

        if includePaths:
            partData.update({
                "image_path": partInfo.get("outputPath"),
                "mask_path": partInfo.get("maskPath"),
                "cutout_path": partInfo.get("cutoutPath"),
                "algo_input_mask_path": partInfo.get("algoInputMaskPath"),
            })

        partsJson.append(partData)

    return partsJson


def buildDebugJsonData(detectedParts, areaValidationData):
    geometryData = buildGeometryJsonData()

    return {
        "run_name": RUN_NAME,
        "part_count": len(detectedParts),
        "expected_part_count": EXPECTED_PART_COUNT,
        "part_count_is_valid": len(detectedParts) == EXPECTED_PART_COUNT,
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
        "part_count_is_valid": len(detectedParts) == EXPECTED_PART_COUNT,
        "coordinate_system": {
            "origin": COORDINATE_ORIGIN,
            "description": buildCoordinateOriginDescription(),
        },
        "px_per_mm": PX_PER_MM,
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
    if not DEBUG_DRAW_COORDINATE_GRID:
        return debugImageBgr

    imageHeight, imageWidth = debugImageBgr.shape[:2]
    overlay = debugImageBgr.copy()

    gridSpacingPx = int(round(DEBUG_GRID_SPACING_MM * PX_PER_MM))
    majorSpacingPx = int(round(DEBUG_GRID_MAJOR_SPACING_MM * PX_PER_MM))

    if gridSpacingPx <= 0:
        return debugImageBgr

    # Raster passend zum Output-Koordinatensystem:
    # Ursprung oben rechts, x nach links, y nach unten.
    verticalLineIndex = 0
    x = imageWidth - 1
    while x >= 0:
        color = COLOR_GRID_MAJOR if isMajorGridLine(verticalLineIndex, gridSpacingPx, majorSpacingPx) else COLOR_GRID_MINOR
        thickness = 2 if color == COLOR_GRID_MAJOR else 1
        cv2.line(overlay, (x, 0), (x, imageHeight - 1), color, thickness)
        verticalLineIndex += 1
        x = imageWidth - 1 - verticalLineIndex * gridSpacingPx

    horizontalLineIndex = 0
    y = 0
    while y < imageHeight:
        color = COLOR_GRID_MAJOR if isMajorGridLine(horizontalLineIndex, gridSpacingPx, majorSpacingPx) else COLOR_GRID_MINOR
        thickness = 2 if color == COLOR_GRID_MAJOR else 1
        cv2.line(overlay, (0, y), (imageWidth - 1, y), color, thickness)
        horizontalLineIndex += 1
        y = horizontalLineIndex * gridSpacingPx

    debugImageBgr = cv2.addWeighted(overlay, DEBUG_GRID_ALPHA, debugImageBgr, 1.0 - DEBUG_GRID_ALPHA, 0)

    axisLengthPx = int(round(DEBUG_AXIS_LENGTH_MM * PX_PER_MM))
    axisLengthPx = max(gridSpacingPx, axisLengthPx)

    origin = (imageWidth - 1, 0)
    xAxisEnd = (max(0, imageWidth - 1 - axisLengthPx), 0)
    yAxisEnd = (imageWidth - 1, min(imageHeight - 1, axisLengthPx))

    cv2.circle(debugImageBgr, origin, 10, COLOR_AXIS_ORIGIN, -1)
    cv2.arrowedLine(debugImageBgr, origin, xAxisEnd, COLOR_AXIS_X, 4, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(debugImageBgr, origin, yAxisEnd, COLOR_AXIS_Y, 4, cv2.LINE_AA, tipLength=0.08)

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
        cv2.circle(debugImageBgr, (centroidXi, centroidYi), PART_CENTROID_RADIUS_PX, PART_CENTROID_COLOR, -1)

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
    debugImageBgr = imageBgr.copy()
    cornerColors = [COLOR_CORNER_0, COLOR_CORNER_1, COLOR_CORNER_2, COLOR_CORNER_3]

    for markerId in sorted(detectedMarkers.keys()):
        markerCorners = detectedMarkers[markerId]["corners"]
        ptsInt = np.round(markerCorners).astype(int)

        cv2.polylines(debugImageBgr, [ptsInt.reshape((-1, 1, 2))], True, COLOR_MARKER_OUTLINE, 2)

        centerX = int(round(np.mean(markerCorners[:, 0])))
        centerY = int(round(np.mean(markerCorners[:, 1])))

        cv2.circle(debugImageBgr, (centerX, centerY), MARKER_CENTER_RADIUS_PX, COLOR_MARKER_CENTER, -1)

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

            cv2.circle(debugImageBgr, (x, y), CORNER_CIRCLE_RADIUS_PX, cornerColors[cornerIndex], -1)

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


def drawCornerPolygon(debugImageBgr, corners, colorPolyline, colorPoint, radius, labelPrefix):
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
        parts.append(f"{cornerName}=ID{mapping['marker_id']}/C{mapping['corner_index']}")

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
    debugImageBgr = drawMarkerDebug(imageBgr, detectedMarkers)
    debugImageBgr = drawA4AndReferenceDebug(debugImageBgr, a4Corners, referenceCorners)

    statusText1 = "A4 corners calculated from ArUco reference corners plus mm offsets"
    statusText2 = buildReferenceStatusText()
    statusText3 = buildOffsetStatusText()
    statusText4 = buildRotationStatusText()

    cv2.putText(debugImageBgr, statusText1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)
    cv2.putText(debugImageBgr, statusText2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)
    cv2.putText(debugImageBgr, statusText3, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)
    cv2.putText(debugImageBgr, statusText4, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)

    return debugImageBgr


# ============================================================
# KONSOLENAUSGABEN
# ============================================================

def printMarkerInfo(detectedMarkers):
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


def printCornerInfo(title, corners):
    print()
    print(title)

    for name in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        x = corners[name][0]
        y = corners[name][1]
        print(f"- {name}: x={x:.1f}, y={y:.1f}")


def printGeometryInfo():
    frameWidthMm, frameHeightMm = getFrameSizeMm()

    print()
    print("Geometrie:")
    print(f"- A4: {A4_WIDTH_MM:.1f} x {A4_HEIGHT_MM:.1f} mm")
    print(f"- Referenz/Rahmen: {frameWidthMm:.1f} x {frameHeightMm:.1f} mm")
    print(f"- Offset links: {FRAME_OFFSET_LEFT_MM:.1f} mm")
    print(f"- Offset rechts: {FRAME_OFFSET_RIGHT_MM:.1f} mm")
    print(f"- Offset oben: {FRAME_OFFSET_TOP_MM:.1f} mm")
    print(f"- Offset unten: {FRAME_OFFSET_BOTTOM_MM:.1f} mm")


def printCoordinateSystemInfo():
    print()
    print("Koordinatensystem:")
    print(f"- Ursprung: {COORDINATE_ORIGIN}")
    print(f"- Beschreibung: {buildCoordinateOriginDescription()}")


def printHomographyInfo(hImageToWarp):
    print()
    print("Homographie Bild -> Warp-Pixel:")
    print(hImageToWarp)

    warpWidthPx, warpHeightPx = getWarpSizePx()

    print()
    print(f"Warp-Groesse: {warpWidthPx} x {warpHeightPx} px")
    print(f"PX_PER_MM: {PX_PER_MM}")


def printPartsInfo(detectedParts):
    print()
    print("Erkannte Teile:")

    if len(detectedParts) == 0:
        print("- keine")
        return

    for partInfo in detectedParts:
        print(f"- {partInfo['partName']}")
        print(f"  Schwerpunkt A4-mm: x={partInfo['centroidXmm']:.3f}, y={partInfo['centroidYmm']:.3f}")
        print(f"  Schwerpunkt A4-px: x={partInfo['centroidXpx']:.3f}, y={partInfo['centroidYpx']:.3f}")
        print(f"  Schwerpunkt Warp-Pixel debug: x={partInfo['centroidX']:.3f}, y={partInfo['centroidY']:.3f}")
        print(f"  Flaeche: {partInfo['areaMm2']:.3f} mm2")
        print(f"  Bounding Box: x={partInfo['bboxX']}, y={partInfo['bboxY']}, w={partInfo['bboxW']}, h={partInfo['bboxH']}")
        print(f"  Bild: {partInfo['outputPath']}")
        print(f"  Maske: {partInfo['maskPath']}")
        print(f"  Cutout: {partInfo['cutoutPath']}")
        print(f"  Algorithmus-Maske: {partInfo.get('algoInputMaskPath')}")

    print()
    print(f"Anzahl Teile: {len(detectedParts)}")
    print(f"Erwartet: {EXPECTED_PART_COUNT}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    outputImagePath = buildOutputPath(OUTPUT_IMAGE_FILENAME)
    outputDebugPath = buildOutputPath(OUTPUT_DEBUG_FILENAME)
    outputWarpPath = buildOutputPath(OUTPUT_WARP_FILENAME)
    outputMaskPath = buildOutputPath(OUTPUT_MASK_FILENAME)
    outputPartsDebugPath = buildOutputPath(OUTPUT_PARTS_DEBUG_FILENAME)
    outputJsonPath = buildOutputPath(OUTPUT_JSON_FILENAME)
    outputHImageToWarpPath = buildOutputPath(OUTPUT_H_IMAGE_TO_WARP_PATH)

    try:
        imageBgr = getInputImage()
        imageBgr = rotateImageIfNeeded(imageBgr)

        savePngImage(outputImagePath, imageBgr)

        print(f"Input-Bild gespeichert: {outputImagePath}")
        print(f"Bildgroesse: {imageBgr.shape[1]} x {imageBgr.shape[0]} Pixel")
        print(buildRotationStatusText())
        printGeometryInfo()
        printCoordinateSystemInfo()

        showDebugImage(INPUT_WINDOW_NAME, imageBgr)

        detectedMarkers, rejectedCandidates = detectArucoMarkers(imageBgr)
        printMarkerInfo(detectedMarkers)

        a4Corners, referenceCorners = extractA4Corners(detectedMarkers)
        printCornerInfo("Referenz-/Rahmen-Bildecken aus ArUcos:", referenceCorners)
        printCornerInfo("Abgeleitete echte A4-Bildecken:", a4Corners)

        debugImageBgr = drawCombinedDebug(imageBgr, detectedMarkers, a4Corners, referenceCorners)
        savePngImage(outputDebugPath, debugImageBgr)
        print(f"Debug-Bild gespeichert: {outputDebugPath}")

        showDebugImage(DEBUG_WINDOW_NAME, debugImageBgr)

        hImageToWarp = computeHomographyImageToWarp(a4Corners)
        printHomographyInfo(hImageToWarp)

        np.save(str(outputHImageToWarpPath), hImageToWarp)
        print(f"H gespeichert: {outputHImageToWarpPath}")

        warpedImageBgr = warpImageToA4(imageBgr, hImageToWarp)
        savePngImage(outputWarpPath, warpedImageBgr)
        print(f"Warp-Bild gespeichert: {outputWarpPath}")

        showDebugImage(WARP_WINDOW_NAME, warpedImageBgr)

        binaryMask = buildPartsMask(warpedImageBgr)
        savePngImage(outputMaskPath, binaryMask)
        print(f"Maske gespeichert: {outputMaskPath}")

        showDebugImage(MASK_WINDOW_NAME, binaryMask)

        detectedParts = findAllValidParts(binaryMask)
        detectedParts = sortPartsByOutputYThenOutputX(detectedParts)
        addDerivedPartValues(detectedParts)

        savePartOutputs(warpedImageBgr, binaryMask, detectedParts)
        algoInputDirPath = saveAlgoInputFiles(binaryMask, detectedParts)

        partsDebugImageBgr = drawPartsDebug(warpedImageBgr, detectedParts)
        savePngImage(outputPartsDebugPath, partsDebugImageBgr)
        print(f"Teile-Debug-Bild gespeichert: {outputPartsDebugPath}")

        showDebugImage(PARTS_DEBUG_WINDOW_NAME, partsDebugImageBgr)

        areaValidationData = buildAreaValidationData(detectedParts)

        debugJsonData = buildDebugJsonData(detectedParts, areaValidationData)
        saveJson(outputJsonPath, debugJsonData)
        print(f"Debug-JSON gespeichert: {outputJsonPath}")

        algoJsonData = buildAlgoInputJsonData(detectedParts, areaValidationData)
        algoJsonPath = algoInputDirPath / ALGO_INPUT_JSON_FILENAME
        saveJson(algoJsonPath, algoJsonData)
        print(f"Algorithmus-Input gespeichert: {algoInputDirPath}")

        printPartsInfo(detectedParts)
        printAreaValidationInfo(areaValidationData)

    except Exception as e:
        print("Fehler:")
        print(e)


if __name__ == "__main__":
    main()
