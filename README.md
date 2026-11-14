# CamQC

Image quality checks for camera modules at an end-of-line test station.

Load images (file, folder, or webcam), run configurable checks — brightness,
sharpness, dead/hot pixels, shading, color, noise, blemish — and get PASS/FAIL
against per-model thresholds. Results are stored in SQLite by serial number and
can be exported as reports. Available as a CLI and a PySide6 desktop app.

> Status: scaffolding only — logic not implemented yet.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

## Usage

```bash
camqc --help
pytest -q
```

Thresholds per camera model live in `config/models.yaml`.
