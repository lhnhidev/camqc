# CamQC

## Goal
Desktop + CLI tool that checks image quality of camera modules at an end-of-line
test station. Images come from files, folders, or a webcam. Each image runs through
configurable checks (brightness, sharpness, dead/hot pixels, shading, color, noise,
blemish), each gets PASS/FAIL against per-model thresholds (`config/models.yaml`),
results are stored in SQLite keyed by serial number, and reports can be exported.

## Stack
Python 3.11+, OpenCV, NumPy, PySide6, SQLite (`sqlite3`), PyYAML, pytest, matplotlib.

## Layout
- `src/camqc/checks/` — one module per image check, shared interface in `base.py`
- `src/camqc/pipeline.py` — runs configured checks, aggregates verdict
- `src/camqc/db.py` — SQLite persistence
- `src/camqc/report.py` — report export
- `src/camqc/cli.py` — command-line entry point
- `src/camqc/gui/` — PySide6 desktop app
- `config/models.yaml` — per camera-model check thresholds
- `tools/gen_samples.py` — synthetic test image generator
- `tests/` — pytest suite

## Coding conventions
- Type hints on every function signature (public and private).
- Docstrings on every module, class, and public function (Google style).
- Small, pure functions: checks take a NumPy image + params and return a result;
  no I/O inside check logic.
- No global mutable state; pass config/connections explicitly.
- Use `logging` (`logger = logging.getLogger(__name__)`), never `print`
  (except CLI user-facing output).
- Images are NumPy arrays in OpenCV BGR order unless stated otherwise.

## Commands
```bash
pip install -e ".[dev]"     # install with dev deps
pytest -q                   # run tests
camqc --help                # CLI (or: python -m camqc.cli --help)
camqc-gui                   # desktop app
camqc report --run-id 12 --out report.html   # self-contained HTML report of one run
camqc report --from 2026-11-01 --to 2026-11-30 --out runs.csv   # CSV of runs and checks
```

## Rules
- After each change, run `pytest -q` and fix any failures before moving on.
