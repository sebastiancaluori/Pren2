#dezidierten ordner für algo input folder erstellt

import shutil
from pathlib import Path
import cv2
import time
import json
import numpy as np

try:
    from picamera2 import Picamera2
except (ImportError, ModuleNotFoundError):
    Picamera2 = None



PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/vision/cam_module.py → project root

DESTINATION_TO_ALGO_INPUT_FOLDER = PROJECT_ROOT / "input"

INPUT_IMAGE_PATH = PROJECT_ROOT / "1.png"

# Dateiname der JSON-Datei fuer den Algorithmus
ALGO_INPUT_JSON_FILENAME = "parts.json"

# Praefix fuer die Maskenbilder im Algorithmus-Ordner.
# Beispiel: "img" ergibt img1.png, img2.png, img3.png, ...
ALGO_INPUT_MASK_PREFIX = "piece_"

# True = inhalt des Ordners und aller unterordner löschen
CLEAR_ALGO_INPUT_FOLDER_BEFORE_SAVE = True

# Bildquelle:
# "camera" = neues Bild mit Pi Camera 3 aufnehmen
# "file"   = bestehendes Bild von Datei laden
IMAGE_SOURCE = "camera"

# Pfad zum Eingabebild, falls IMAGE_SOURCE = "file" oder falls keine Kamera gefunden wurde
INPUT_IMAGE_PATH = PROJECT_ROOT / "1.png"
# Bildgrösse für Kameraaufnahme
IMAGE_WIDTH = 4608
IMAGE_HEIGHT = 2592

# Wartezeit nach Kamerastart
STARTUP_WAIT_SECONDS = 2.0

# Rotation
ROTATE_90_CLOCKWISE = False
ROTATE_180 = False


# ============================================================
# AUSGABE
# ============================================================

OUTPUT_DIR = "output"
OUTPUT_PARTS_DIR = "output/parts"
OUTPUT_PART_MASKS_DIR = "output/part_masks"
OUTPUT_PART_CUTOUTS_DIR = "output/part_cutouts"

RUN_NAME = "step_09"

OUTPUT_IMAGE_FILENAME = f"{RUN_NAME}_input.png"
OUTPUT_DEBUG_FILENAME = f"{RUN_NAME}_a4_corners_debug.png"
OUTPUT_WARP_FILENAME = f"{RUN_NAME}_warp_a4.png"
OUTPUT_MASK_FILENAME = f"{RUN_NAME}_parts_mask.png"
OUTPUT_PARTS_DEBUG_FILENAME = f"{RUN_NAME}_parts_debug.png"
OUTPUT_JSON_FILENAME = f"{RUN_NAME}_parts.json"
OUTPUT_H_IMAGE_TO_WARP_PATH = f"{RUN_NAME}_h_image_to_warp.npy"
OUTPUT_H_IMAGE_TO_A4_MM_PATH = f"{RUN_NAME}_h_image_to_a4_mm.npy"


# ============================================================
# DEBUG-ANZEIGE
# ============================================================

# True  = Bilder mit cv2.imshow anzeigen
# False = keine Fenster anzeigen, Dateien werden trotzdem gespeichert
DEBUG_SHOW_IMAGES = False

# Wartezeit für jedes Debug-Fenster
DEBUG_WAIT_MS = 5000

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

# Position der Marker relativ zur A4-Fläche:
# "inside"  = Marker liegen innerhalb der A4-Fläche
# "outside" = Marker liegen ausserhalb der A4-Fläche
MARKER_POSITION_MODE = "outside"

# A4 im Querformat
A4_WIDTH_MM = 297.0
A4_HEIGHT_MM = 210.0

# Zielauflösung der Entzerrung
PX_PER_MM = 10.0


# ============================================================
# KOORDINATENSYSTEM
# ============================================================

# Ursprung des A4-Koordinatensystems für Teile-Schwerpunkte.
#
# Mögliche Werte:
# "bottom_left"  = Ursprung unten links,  x nach rechts, y nach oben
# "top_left"     = Ursprung oben links,   x nach rechts, y nach unten
# "top_right"    = Ursprung oben rechts,  x nach links,  y nach unten
# "bottom_right" = Ursprung unten rechts, x nach links,  y nach oben
COORDINATE_ORIGIN = "bottom_left"


# ============================================================
# TEILE-SEGMENTIERUNG
# ============================================================

# Dunkle Teile auf hellem Hintergrund
# "fixed"    = fester Threshold mit THRESHOLD_VALUE
# "otsu"     = automatischer Threshold, meistens gute erste Wahl
# "adaptive" = robuster bei ungleichmässiger Beleuchtung
SEGMENTATION_THRESHOLD_MODE = "otsu"

# Nur verwendet, wenn SEGMENTATION_THRESHOLD_MODE = "fixed"
THRESHOLD_VALUE = 150

# Nur verwendet, wenn SEGMENTATION_THRESHOLD_MODE = "adaptive"
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 101
ADAPTIVE_THRESHOLD_C = 8

# Leichte Glättung vor dem Threshold
GAUSSIAN_BLUR_KERNEL_SIZE = 7

# Minimale / maximale Fläche eines Teils
MIN_PART_AREA_MM2 = 3000.0
MAX_PART_AREA_MM2 = 100000.0

# Morphologie
MORPH_OPEN_KERNEL_SIZE = 5
MORPH_CLOSE_KERNEL_SIZE = 7

# Löcher in schwarzen Teilen füllen
FILL_CONTOUR_HOLES = True

# Optionaler Rand, der bei der Teile-Erkennung ignoriert wird.
# Hilft gegen Störungen am A4-Rand.
IGNORE_BORDER_MM = 0.0

# Zuschlag rund um Bounding Box beim Ausschneiden
CROP_PADDING_PX = 0

# Erwartete Anzahl Teile
EXPECTED_PART_COUNT = 4

# Weisser Hintergrund für freigestellte Teilbilder
CUTOUT_BACKGROUND_VALUE = 255


# ============================================================
# VALIDIERUNG
# ============================================================

# Erwartete Gesamtfläche der Teile:
# Die Teile bilden zusammen eine halbe A4-Fläche.
# EXPECTED_TOTAL_PART_AREA_MM2 = (A4_WIDTH_MM * A4_HEIGHT_MM) / 2.0

# Falls Teile gesamt eine andere Fläche als A5 haben:
# PREN-Puzzle ohne Rahmen: 18.9 x 12.6 cm
EXPECTED_TOTAL_PART_AREA_MM2 = 189 * 126

# 0.01 = 1 %
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
COLOR_A4_TEXT = (0, 165, 255)
COLOR_A4_POLYLINE = (255, 255, 255)
COLOR_STATUS_TEXT = (255, 255, 255)

CORNER_CIRCLE_RADIUS_PX = 6
MARKER_CENTER_RADIUS_PX = 6
A4_CORNER_RADIUS_PX = 10

CORNER_TEXT_OFFSET_X = 8
CORNER_TEXT_OFFSET_Y = -8

TEXT_FONT_SCALE = 0.8
TEXT_THICKNESS = 2


# ============================================================
# DATEI- UND ALLGEMEINE HILFSFUNKTIONEN
# ============================================================

def buildOutputPath(filename):
    outputDirPath = Path(OUTPUT_DIR)
    outputDirPath.mkdir(parents=True, exist_ok=True)
    return outputDirPath / filename


def buildDirPath(pathString):
    path = Path(pathString)
    path.mkdir(parents=True, exist_ok=True)
    return path


def saveJson(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def savePngImage(path, image):
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
    if not DEBUG_SHOW_IMAGES:
        return

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

        print("Starte Kamera...")
        picam2.start()

        print(f"Warte {STARTUP_WAIT_SECONDS:.1f} Sekunden...")
        time.sleep(STARTUP_WAIT_SECONDS)

        print("Nehme Bild auf...")
        imageBgr = picam2.capture_array()

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

        print("Keine Picamera2-Unterstützung auf dieser Maschine, nutze Datei.")
        return loadImageFromFile()

    if IMAGE_SOURCE == "file":
        return loadImageFromFile()

    raise ValueError('IMAGE_SOURCE muss "camera" oder "file" sein.')


# ============================================================
# ARUCO / A4-ERKENNUNG
# ============================================================

def detectArucoMarkers(imageBgr):
    arucoDictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    arucoParameters = cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    cornersList, ids, rejectedCandidates = arucoDetector.detectMarkers(imageBgr)

    detectedMarkers = {}

    if ids is None:
        return detectedMarkers, rejectedCandidates

    idsFlat = ids.flatten()

    for i, markerId in enumerate(idsFlat):
        if markerId not in REQUIRED_IDS:
            continue

        markerCorners = cornersList[i].reshape(4, 2).astype(np.float32)

        detectedMarkers[int(markerId)] = {
            "id": int(markerId),
            "corners": markerCorners
        }

    return detectedMarkers, rejectedCandidates


def getA4CornerIndexForMarker(markerId):
    if MARKER_POSITION_MODE == "inside":
        return markerId

    if MARKER_POSITION_MODE == "outside":
        return (markerId + 2) % 4

    raise ValueError('MARKER_POSITION_MODE muss "inside" oder "outside" sein.')


def extractA4Corners(detectedMarkers):
    missingIds = [markerId for markerId in REQUIRED_IDS if markerId not in detectedMarkers]

    if missingIds:
        raise RuntimeError(f"Nicht alle benötigten Marker wurden erkannt. Fehlend: {missingIds}")

    a4CornerBottomLeft = detectedMarkers[0]["corners"][getA4CornerIndexForMarker(0)]
    a4CornerTopLeft = detectedMarkers[1]["corners"][getA4CornerIndexForMarker(1)]
    a4CornerTopRight = detectedMarkers[2]["corners"][getA4CornerIndexForMarker(2)]
    a4CornerBottomRight = detectedMarkers[3]["corners"][getA4CornerIndexForMarker(3)]

    return {
        "bottom_left": a4CornerBottomLeft.astype(np.float32),
        "top_left": a4CornerTopLeft.astype(np.float32),
        "top_right": a4CornerTopRight.astype(np.float32),
        "bottom_right": a4CornerBottomRight.astype(np.float32),
    }


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
        dtype=np.float32
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
        dtype=np.float32
    )


def buildA4CornerArrayMm():
    return np.array(
        [
            [0.0, A4_HEIGHT_MM],
            [A4_WIDTH_MM, A4_HEIGHT_MM],
            [A4_WIDTH_MM, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32
    )


def computeHomographies(a4Corners):
    imagePoints = buildImageCornerArray(a4Corners)
    warpPointsPx = buildWarpCornerArrayPx()
    a4PointsMm = buildA4CornerArrayMm()

    hImageToWarp = cv2.getPerspectiveTransform(imagePoints, warpPointsPx)
    hImageToA4Mm = cv2.getPerspectiveTransform(imagePoints, a4PointsMm)

    return hImageToWarp, hImageToA4Mm


def warpImageToA4(imageBgr, hImageToWarp):
    warpWidthPx, warpHeightPx = getWarpSizePx()

    warpedImageBgr = cv2.warpPerspective(
        imageBgr,
        hImageToWarp,
        (warpWidthPx, warpHeightPx)
    )

    return warpedImageBgr


def validateCoordinateOrigin():
    validOrigins = ["bottom_left", "top_left", "top_right", "bottom_right"]

    if COORDINATE_ORIGIN not in validOrigins:
        raise ValueError(
            'COORDINATE_ORIGIN muss "bottom_left", "top_left", "top_right" oder "bottom_right" sein.'
        )


def warpPxToA4Mm(xPx, yPx):
    validateCoordinateOrigin()

    xFromLeftMm = float(xPx) / PX_PER_MM
    yFromTopMm = float(yPx) / PX_PER_MM

    if COORDINATE_ORIGIN == "bottom_left":
        xMm = xFromLeftMm
        yMm = A4_HEIGHT_MM - yFromTopMm

    elif COORDINATE_ORIGIN == "top_left":
        xMm = xFromLeftMm
        yMm = yFromTopMm

    elif COORDINATE_ORIGIN == "top_right":
        xMm = A4_WIDTH_MM - xFromLeftMm
        yMm = yFromTopMm

    elif COORDINATE_ORIGIN == "bottom_right":
        xMm = A4_WIDTH_MM - xFromLeftMm
        yMm = A4_HEIGHT_MM - yFromTopMm

    else:
        raise ValueError(
            'COORDINATE_ORIGIN muss "bottom_left", "top_left", "top_right" oder "bottom_right" sein.'
        )

    return xMm, yMm


def buildCoordinateOriginDescription():
    if COORDINATE_ORIGIN == "bottom_left":
        return "origin bottom_left, x right, y up"

    if COORDINATE_ORIGIN == "top_left":
        return "origin top_left, x right, y down"

    if COORDINATE_ORIGIN == "top_right":
        return "origin top_right, x left, y down"

    if COORDINATE_ORIGIN == "bottom_right":
        return "origin bottom_right, x left, y up"

    return "origin invalid"


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

    blurredImage = cv2.GaussianBlur(
        grayImage,
        (blurKernelSize, blurKernelSize),
        0
    )

    if SEGMENTATION_THRESHOLD_MODE == "fixed":
        _, binaryMask = cv2.threshold(
            blurredImage,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY_INV
        )

    elif SEGMENTATION_THRESHOLD_MODE == "otsu":
        _, binaryMask = cv2.threshold(
            blurredImage,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    elif SEGMENTATION_THRESHOLD_MODE == "adaptive":
        adaptiveBlockSize = ensureOddKernelSize(ADAPTIVE_THRESHOLD_BLOCK_SIZE)

        binaryMask = cv2.adaptiveThreshold(
            blurredImage,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptiveBlockSize,
            ADAPTIVE_THRESHOLD_C
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
        centroidX = x + (w / 2.0)
        centroidY = y + (h / 2.0)
        return centroidX, centroidY

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

        partInfo = {
            "contour": contour,
            "areaPx": float(areaPx),
            "centroidX": float(centroidX),
            "centroidY": float(centroidY),
            "bboxX": int(x),
            "bboxY": int(y),
            "bboxW": int(w),
            "bboxH": int(h),
        }

        detectedParts.append(partInfo)

    return detectedParts


def sortPartsByA4YThenA4X(detectedParts):
    def sortKey(partInfo):
        centroidXmm, centroidYmm = warpPxToA4Mm(partInfo["centroidX"], partInfo["centroidY"])
        return centroidYmm, centroidXmm

    return sorted(detectedParts, key=sortKey)


def addDerivedPartValues(detectedParts):
    for i, partInfo in enumerate(detectedParts):
        centroidXmm, centroidYmm = warpPxToA4Mm(partInfo["centroidX"], partInfo["centroidY"])
        areaMm2 = partInfo["areaPx"] / (PX_PER_MM ** 2)

        partInfo["index"] = i + 1
        partInfo["partName"] = f"part_{i + 1:02d}"
        partInfo["centroidXmm"] = float(centroidXmm)
        partInfo["centroidYmm"] = float(centroidYmm)
        partInfo["areaMm2"] = float(areaMm2)


# ============================================================
# FLÄCHENVALIDIERUNG
# ============================================================

def computeTotalPartsAreaMm2(detectedParts):
    totalAreaMm2 = 0.0

    for partInfo in detectedParts:
        totalAreaMm2 += partInfo["areaMm2"]

    return totalAreaMm2


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
    print("Flächenvalidierung:")
    print(f"- Erwartete Gesamtfläche: {areaValidationData['expected_total_area_mm2']:.0f} mm2")
    print(f"- Gemessene Gesamtfläche: {areaValidationData['measured_total_area_mm2']:.0f} mm2")
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
    croppedSingleMask = singleMask[y1:y2, x1:x2].copy()

    return croppedSingleMask


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
            partInfo["bboxH"]
        )

        croppedSingleMask = buildSinglePartMask(
            binaryMask,
            partInfo["contour"],
            cropBounds
        )

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
        croppedSingleMask = buildSinglePartMask(
            binaryMask,
            partInfo["contour"],
            (
                partInfo["bboxX"],
                partInfo["bboxY"],
                partInfo["bboxX"] + partInfo["bboxW"],
                partInfo["bboxY"] + partInfo["bboxH"],
            )
        )

        algoMaskFilename = f"{ALGO_INPUT_MASK_PREFIX}{i}.png"
        algoMaskPath = algoInputDirPath / algoMaskFilename

        savePngImage(algoMaskPath, croppedSingleMask)

        partInfo["algoInputMaskFilename"] = algoMaskFilename
        partInfo["algoInputMaskPath"] = str(algoMaskPath)

    return algoInputDirPath


# ============================================================
# JSON-EXPORT
# ============================================================

def buildPartsJsonData(detectedParts, areaValidationData):
    partsJson = []

    for partInfo in detectedParts:
        partsJson.append({
            "index": partInfo["index"],
            "part_name": partInfo["partName"],
            "centroid_mm": {
                "x": round(partInfo["centroidXmm"], 6),
                "y": round(partInfo["centroidYmm"], 6),
            },
            "centroid_px": {
                "x": round(partInfo["centroidX"], 6),
                "y": round(partInfo["centroidY"], 6),
            },
            "area_mm2": round(partInfo["areaMm2"], 6),
            "bounding_box_px": {
                "x": partInfo["bboxX"],
                "y": partInfo["bboxY"],
                "w": partInfo["bboxW"],
                "h": partInfo["bboxH"],
            },
            "image_path": partInfo["outputPath"],
            "mask_path": partInfo["maskPath"],
            "cutout_path": partInfo["cutoutPath"],
            "algo_input_mask_filename": partInfo.get("algoInputMaskFilename"),
            "algo_input_mask_path": partInfo.get("algoInputMaskPath"),
        })

    data = {
        "run_name": RUN_NAME,
        "part_count": len(detectedParts),
        "expected_part_count": EXPECTED_PART_COUNT,
        "part_count_is_valid": len(detectedParts) == EXPECTED_PART_COUNT,
        "px_per_mm": PX_PER_MM,
        "a4_size_mm": {
            "width": A4_WIDTH_MM,
            "height": A4_HEIGHT_MM,
            "area_mm2": round(A4_WIDTH_MM * A4_HEIGHT_MM, 6),
        },
        "coordinate_system": {
            "origin": COORDINATE_ORIGIN,
            "description": buildCoordinateOriginDescription(),
        },
        "expected_total_part_area": {
            "description": "PREN puzzle area without frame",
            "area_mm2": round(EXPECTED_TOTAL_PART_AREA_MM2, 6),
        },
        "area_validation": areaValidationData,
        "marker_position_mode": MARKER_POSITION_MODE,
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
        "sorting": "smallest_a4_y_then_smallest_a4_x",
        "parts": partsJson,
    }

    return data


def buildAlgoInputJsonData(detectedParts, areaValidationData):
    partsJson = []

    for partInfo in detectedParts:
        partsJson.append({
            "index": partInfo["index"],
            "mask_filename": partInfo["algoInputMaskFilename"],
            "centroid_mm": {
                "x": round(partInfo["centroidXmm"], 6),
                "y": round(partInfo["centroidYmm"], 6),
            },
            "area_mm2": round(partInfo["areaMm2"], 6),
        })

    data = {
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
        "expected_total_part_area_mm2": round(EXPECTED_TOTAL_PART_AREA_MM2, 6),
        "area_validation": areaValidationData,
        "parts": partsJson,
    }

    return data


# ============================================================
# DEBUG-ZEICHNUNGEN
# ============================================================

def drawPartsDebug(warpedImageBgr, detectedParts):
    debugImageBgr = warpedImageBgr.copy()

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
            cv2.LINE_AA
        )

        cv2.putText(
            debugImageBgr,
            f"({partInfo['centroidXmm']:.1f}, {partInfo['centroidYmm']:.1f}) mm",
            (centroidXi + 12, centroidYi - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            PART_TEXT_FONT_SCALE,
            PART_TEXT_COLOR,
            PART_TEXT_THICKNESS,
            cv2.LINE_AA
        )

    cv2.putText(
        debugImageBgr,
        f"parts found: {len(detectedParts)}   expected: {EXPECTED_PART_COUNT}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        debugImageBgr,
        f"coord: {COORDINATE_ORIGIN}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        debugImageBgr,
        "sorting: smallest A4-y, then smallest A4-x",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PART_TEXT_COLOR,
        2,
        cv2.LINE_AA
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
            cv2.LINE_AA
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
                cv2.LINE_AA
            )

    return debugImageBgr


def drawA4CornerDebug(imageBgr, a4Corners):
    debugImageBgr = imageBgr.copy()

    ptBL = np.round(a4Corners["bottom_left"]).astype(int)
    ptTL = np.round(a4Corners["top_left"]).astype(int)
    ptTR = np.round(a4Corners["top_right"]).astype(int)
    ptBR = np.round(a4Corners["bottom_right"]).astype(int)

    a4Polygon = np.array([ptBL, ptTL, ptTR, ptBR], dtype=np.int32)
    cv2.polylines(debugImageBgr, [a4Polygon.reshape((-1, 1, 2))], True, COLOR_A4_POLYLINE, 2)

    labeledPoints = [
        ("A4 BL", ptBL),
        ("A4 TL", ptTL),
        ("A4 TR", ptTR),
        ("A4 BR", ptBR),
    ]

    for label, pt in labeledPoints:
        x = int(pt[0])
        y = int(pt[1])

        cv2.circle(debugImageBgr, (x, y), A4_CORNER_RADIUS_PX, COLOR_A4_POINT, -1)

        cv2.putText(
            debugImageBgr,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_A4_TEXT,
            2,
            cv2.LINE_AA
        )

    return debugImageBgr


def buildStatusTextLine2():
    if MARKER_POSITION_MODE == "inside":
        return "BL=ID0/C0  TL=ID1/C1  TR=ID2/C2  BR=ID3/C3"

    return "BL=ID0/C2  TL=ID1/C3  TR=ID2/C0  BR=ID3/C1"


def buildRotationStatusText():
    if ROTATE_90_CLOCKWISE:
        return "rotation: 90_cw"

    if ROTATE_180:
        return "rotation: 180"

    return "rotation: none"


def drawCombinedDebug(imageBgr, detectedMarkers, a4Corners):
    debugImageBgr = drawMarkerDebug(imageBgr, detectedMarkers)
    debugImageBgr = drawA4CornerDebug(debugImageBgr, a4Corners)

    statusText1 = f"A4 corners extracted from marker corners, mode: {MARKER_POSITION_MODE}"
    statusText2 = buildStatusTextLine2()
    statusText3 = buildRotationStatusText()

    cv2.putText(debugImageBgr, statusText1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)
    cv2.putText(debugImageBgr, statusText2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)
    cv2.putText(debugImageBgr, statusText3, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STATUS_TEXT, 2, cv2.LINE_AA)

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


def printA4CornerInfo(a4Corners):
    print()
    print("Abgeleitete A4-Bildecken:")

    for name in ["bottom_left", "top_left", "top_right", "bottom_right"]:
        x = a4Corners[name][0]
        y = a4Corners[name][1]
        print(f"- {name}: x={x:.1f}, y={y:.1f}")


def printCoordinateSystemInfo():
    print()
    print("Koordinatensystem:")
    print(f"- Ursprung: {COORDINATE_ORIGIN}")
    print(f"- Beschreibung: {buildCoordinateOriginDescription()}")


def printHomographyInfo(hImageToWarp, hImageToA4Mm):
    print()
    print("Homographie Bild -> Warp-Pixel:")
    print(hImageToWarp)

    print()
    print("Homographie Bild -> A4-mm:")
    print(hImageToA4Mm)

    warpWidthPx, warpHeightPx = getWarpSizePx()

    print()
    print(f"Warp-Grösse: {warpWidthPx} x {warpHeightPx} px")
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
        print(f"  Schwerpunkt Warp-Pixel: x={partInfo['centroidX']:.3f}, y={partInfo['centroidY']:.3f}")
        print(f"  Fläche: {partInfo['areaMm2']:.3f} mm2")
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
    outputHImageToA4MmPath = buildOutputPath(OUTPUT_H_IMAGE_TO_A4_MM_PATH)

    try:
        validateCoordinateOrigin()

        imageBgr = getInputImage()
        imageBgr = rotateImageIfNeeded(imageBgr)

        savePngImage(outputImagePath, imageBgr)

        print(f"Input-Bild gespeichert: {outputImagePath}")
        print(f"Bildgrösse: {imageBgr.shape[1]} x {imageBgr.shape[0]} Pixel")
        print(f"Marker-Modus: {MARKER_POSITION_MODE}")
        print(buildRotationStatusText())
        printCoordinateSystemInfo()

        showDebugImage(INPUT_WINDOW_NAME, imageBgr)

        detectedMarkers, rejectedCandidates = detectArucoMarkers(imageBgr)
        printMarkerInfo(detectedMarkers)

        a4Corners = extractA4Corners(detectedMarkers)
        printA4CornerInfo(a4Corners)

        debugImageBgr = drawCombinedDebug(imageBgr, detectedMarkers, a4Corners)
        savePngImage(outputDebugPath, debugImageBgr)
        print(f"Debug-Bild gespeichert: {outputDebugPath}")

        showDebugImage(DEBUG_WINDOW_NAME, debugImageBgr)

        hImageToWarp, hImageToA4Mm = computeHomographies(a4Corners)
        printHomographyInfo(hImageToWarp, hImageToA4Mm)

        np.save(str(outputHImageToWarpPath), hImageToWarp)
        np.save(str(outputHImageToA4MmPath), hImageToA4Mm)

        print(f"H gespeichert: {outputHImageToWarpPath}")
        print(f"H gespeichert: {outputHImageToA4MmPath}")

        warpedImageBgr = warpImageToA4(imageBgr, hImageToWarp)
        savePngImage(outputWarpPath, warpedImageBgr)
        print(f"Warp-Bild gespeichert: {outputWarpPath}")

        showDebugImage(WARP_WINDOW_NAME, warpedImageBgr)

        binaryMask = buildPartsMask(warpedImageBgr)
        savePngImage(outputMaskPath, binaryMask)
        print(f"Maske gespeichert: {outputMaskPath}")

        showDebugImage(MASK_WINDOW_NAME, binaryMask)

        detectedParts = findAllValidParts(binaryMask)
        detectedParts = sortPartsByA4YThenA4X(detectedParts)
        addDerivedPartValues(detectedParts)

        savePartOutputs(warpedImageBgr, binaryMask, detectedParts)
        algoInputDirPath = saveAlgoInputFiles(binaryMask, detectedParts)

        partsDebugImageBgr = drawPartsDebug(warpedImageBgr, detectedParts)
        savePngImage(outputPartsDebugPath, partsDebugImageBgr)
        print(f"Teile-Debug-Bild gespeichert: {outputPartsDebugPath}")

        showDebugImage(PARTS_DEBUG_WINDOW_NAME, partsDebugImageBgr)

        areaValidationData = buildAreaValidationData(detectedParts)

        jsonData = buildPartsJsonData(detectedParts, areaValidationData)
        saveJson(outputJsonPath, jsonData)
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