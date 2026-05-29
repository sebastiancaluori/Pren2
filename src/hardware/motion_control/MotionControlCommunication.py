import logging
import struct
import time

import serial

from proto import puzzle_pb2
from src.utils.puzzle_piece import PuzzlePiece

logger = logging.getLogger(__name__)

# Default UART settings — match STM32 UART7 config
DEFAULT_PORT = "/dev/ttyAMA0"  # Raspberry Pi hardware UART
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 5.0  # seconds
PIXEL_TO_MM_SCALE = 2.0

def _send_frame(ser: serial.Serial, payload: bytes) -> None:
    """Sende ein laengenpraefixiertes Frame: [len_hi][len_lo][payload]"""
    length = len(payload)
    header = struct.pack(">H", length)
    ser.write(header + payload)
    ser.flush()


def _receive_frame(ser: serial.Serial) -> bytes | None:
    """Empfange ein laengenpraefixiertes Frame."""
    header = ser.read(2)
    if len(header) < 2:
        return None
    length = struct.unpack(">H", header)[0]
    data = ser.read(length)
    if len(data) < length:
        return None
    return data


def send_to_robot(
    pieces: list[PuzzlePiece],
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    pick_px_per_mm: float = PIXEL_TO_MM_SCALE,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """
    Sende Puzzle-Loesung ueber UART an den STM32.

    pick_pose.x/y: Zentroid in Solver-Pixeln → wird durch pick_px_per_mm geteilt → mm
    place_pose.x/y: Zentroid bereits in mm (umgerechnet in pipeline.py) → direkt verwenden

    Returns True bei Erfolg (STATUS_OK), False bei Fehler.
    """
    cmd = puzzle_pb2.PuzzleCommand()

    for p in pieces:
        piece = cmd.pieces.add()
        a4_x_mm = 297
        piece.piece_id = int(p.id)
        piece.pick_x = a4_x_mm - (p.pick_pose.x / pick_px_per_mm)
        piece.pick_y = p.pick_pose.y / pick_px_per_mm
        if p.place_pose:
            piece.place_x = p.place_pose.y  # already in mm
            piece.place_y = p.place_pose.x  # already in mm
            rotation = (90 - p.place_pose.theta) % 360
            if rotation > 180:
                rotation -= 360
            piece.rotation = rotation
        else:
            piece.place_x = 0.0
            piece.place_y = 0.0
            piece.rotation = 0.0

    payload = cmd.SerializeToString()
    logger.info(
        "Sende PuzzleCommand: %d Teile, %d Bytes", len(pieces), len(payload)
    )

    with serial.Serial(port, baudrate, timeout=timeout) as ser:
        time.sleep(0.1)  # STM32 UART settle time
        _send_frame(ser, payload)

        # Warte auf Ack
        ack_data = _receive_frame(ser)
        if ack_data is None:
            logger.error("Kein Ack vom STM32 erhalten (Timeout)")
            return False

        ack = puzzle_pb2.Ack()
        ack.ParseFromString(ack_data)

        if ack.status == puzzle_pb2.STATUS_OK:
            logger.info("STM32 Ack: OK")
            return True
        else:
            logger.error("STM32 Ack: Status=%d", ack.status)
            return False
        
        
def wait_for_robot_start(port: str, baudrate: int) -> bool:
    """
    Blockiert und lauscht auf UART, bis der Hardware-Knopf am STM32 gedrueckt 
    wurde (STM32 sendet STATUS_READY).
    """
    import serial
    
    logger.info("Warte auf Hardware-Start-Knopf am Roboter...")
    
    # timeout=None bedeutet: Endlos warten, bis Daten kommen
    with serial.Serial(port, baudrate, timeout=None) as ser:
        ser.reset_input_buffer() # Alte Daten verwerfen
        
        while True:
            # Deine existierende Funktion zum Einlesen eines Protobuf-Frames nutzen
            data = _receive_frame(ser) 
            if data is not None:
                try:
                    ack = puzzle_pb2.Ack()
                    ack.ParseFromString(data)
                    
                    if ack.status == puzzle_pb2.STATUS_READY: 
                        logger.info("✓ Roboter ist BEREIT (Button pressed)!")
                        return True
                except Exception as e:
                    logger.warning(f"Fehler beim Parsen des UART-Signals: {e}")
