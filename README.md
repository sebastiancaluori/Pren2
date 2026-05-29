# PREN Puzzle Solver
Automatisches Puzzle-Löse-System für PREN HS25/FS26 (HSLU)

## Architektur

### Module
- **vision**: Bildverarbeitung (Kamera, Segmentierung, Features)
- **solver**: Puzzle-Logik (Layout-Solver, Validierung)
- **ui**: User Interface (GUI, Simulator für PREN1)
- **hardware**: Hardware-Steuerung (Motion Control für PREN2)
- **core**: Kern-Pipeline und Konfiguration
- **utils**: Hilfsfunktionen

### Datenfluss
Kamera → Segmentierung → Feature-Extraktion → Solver → Validierung → Hardware

## Setup

### Development Setup
```bash
# Virtuelle Umgebung
python3.13 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Standalone Executable

Build a standalone executable that can run without Python installed:

```bash
# Make build script executable (first time only)
chmod +x build.sh

# Build executable
./build.sh
```

The executable will be created in `dist/PREN-Puzzle-Solver/PREN-Puzzle-Solver`


## Raspberry Pi Service Setup 

### 1. Service-Datei erstellen
Erstelle eine neue Konfigurationsdatei auf dem Raspberry Pi:

```bash
sudo vi /etc/systemd/system/puzzlesolver.service
```

#### Inhalt

```toml
[Unit]
Description=PREN Puzzle Solver Service
# Wartet, bis das Netzwerk und das System bereit sind
After=network.target multi-user.target

[Service]
# Der Benutzer, unter dem das Skript ausgeführt wird
User=pren
# Das Arbeitsverzeichnis deines Repositories
WorkingDirectory=/home/pren/pren2/pren-puzzle-solver


# Startbefehl (nutzt Python aus der virtuellen Umgebung)
ExecStart=/home/pren/pren2/pren-puzzle-solver/venv/bin/python /home/pren/pren2/pren-puzzle-solver/main.py

# Dauer-Loop: Startet das Programm bei jeder Beendigung neu (Fehler & normaler Exit)
Restart=always
# Wartezeit von 5 Sekunden vor dem nächsten Neustart
RestartSec=5

StandardOutput=inherit
StandardError=inherit

[Install]
WantedBy=multi-user.target
```

### 2. Dienst aktivieren und starten
```bash
sudo systemctl daemon-reload
sudo systemctl enable puzzlesolver.service
sudo systemctl start puzzlesolver.service
```

### 3. Status und Logs überwachen

```bash
# Status abfragen
sudo systemctl status puzzlesolver.service

# Live-Logs anzeigen
journalctl -u puzzlesolver.service -f
```

### 4. Restart Service

```bash
sudo systemctl restart puzzlesolver.service
```
