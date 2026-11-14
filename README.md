# CamQC

CamQC checks the image quality of camera modules at an end-of-line test station in an
electronics factory. You give it images of a module (files, a folder, or a webcam frame); it
runs a configurable set of checks, compares each measurement with per-model thresholds,
decides **PASS / FAIL / INCOMPLETE**, stores the result in SQLite under the module's serial
number, and can export a report. It comes as a command-line tool (`camqc`) and a desktop
application (`camqc-gui`).

> The thresholds in `config/models.yaml` are tuned for the *synthetic* images produced by
> `tools/gen_samples.py`. They are a starting point, not a calibration for real hardware.

## Screenshots

Screenshots have not been added yet. Put them in `docs/screenshots/` and link them here.

> **[Screenshot placeholder]** Test tab: frames, results table with colored rows, verdict.
>
> **[Screenshot placeholder]** Test tab: overlay highlight after clicking a failing check.
>
> **[Screenshot placeholder]** History tab and the run-detail dialog.
>
> **[Screenshot placeholder]** Statistics tab: yield per day and failure Pareto.
>
> **[Screenshot placeholder]** HTML report opened in a browser.

## Features

**Checks and pipeline**
- Six image checks: brightness, sharpness, hot/dead pixels, lens shading, color, blemish
  (see [Checks](#checks)).
- Per-model configuration in YAML (two example models, `CAM_A` and `CAM_B`): which checks run,
  their thresholds, and which *frame type* each check needs (`normal`, `flat`, `dark`, `white`).
- One broken check never crashes a run: a missing frame makes a check `SKIPPED`, an exception
  makes it `ERROR`, and the unit is then `INCOMPLETE` instead of a false `PASS`.
- Config validation with error messages that name the offending key.

**Command line** (`camqc`): `init-db`, `run`, `batch`, `history`, `stats`, `report`; colored
PASS/FAIL tables (disabled when output is not a terminal); exit codes 0 / 1 / 2; rotating log file.

**Desktop app** (`camqc-gui`, PySide6)
- *Test* tab: model, serial and station, choose the Normal / Dark / White / Flat frames from
  disk, **capture a Normal frame from a webcam** (OpenCV `VideoCapture(0)`, friendly error if
  there is no camera), or **load a folder** of images (progress bar, cancel). Runs happen in a
  worker thread, so the UI does not freeze.
- Results table (green / red / amber rows), a large verdict label, and the image with
  **check overlays** (circles / rectangles); clicking a row highlights only that check's overlay.
- *History* tab: serial search (substring), date range and verdict filters, sortable table,
  double-click a run for its per-check results.
- *Statistics* tab: yield per day and a Pareto chart of failing checks (matplotlib), plus a
  data table. History and Statistics refresh when a run is saved.
- **Export report** button for the run that is shown.

**Storage and reports**
- SQLite (`sqlite3`, no ORM): runs, per-check results and, optionally, the input images.
  A retest of the same serial is simply another run.
- `camqc report`: a self-contained HTML report of one run (serial, model, verdict, results
  table, input images with the overlays drawn on them) and a CSV export of runs and checks.

**Tooling**: a synthetic sample generator (`tools/gen_samples.py`) and a pytest suite
(`pytest -q`), including headless GUI tests.

## Architecture

```
  images (files, folder, webcam)
            |
            v
  +--------------------+   config/models.yaml
  |    CLI  /  GUI     |   (models, checks, params, frame types)
  | cli.py   gui/*.py  |             |
  +----+----------+----+             v
       |          |        +-------------------+
       |          +------> |  pipeline.py      |
       |                   |  run_pipeline()   |
       |                   +---------+---------+
       |                             | frames by type
       |                             v
       |             +---------------------------------+
       |             | checks/*.py   run(image, params)|
       |             | brightness sharpness pixels     |
       |             | shading color blemish           |
       |             +---------------+-----------------+
       |                             | CheckResult
       |                             v
       |                   RunResult: PASS/FAIL/INCOMPLETE
       |                             |
       v                             v
  +--------------------------------------------+
  |  db.py  SQLite: runs, results, run_images  |
  +----------------------+---------------------+
                         |
                         v
          report.py (HTML report, CSV) / History / Statistics
```

| Module | Role |
|---|---|
| `checks/base.py` | `CheckResult`, `make_result`, shared image helpers |
| `checks/*.py` | one module per check, each exposing `run(image_bgr, params) -> CheckResult` |
| `checks/__init__.py` | registry of check names |
| `pipeline.py` | config loading and validation, `run_pipeline`, verdict rules |
| `db.py` | schema, `save_run`, queries (history, yield, Pareto, CSV rows) |
| `report.py` | HTML report and CSV export |
| `cli.py` | the `camqc` command |
| `gui/` | `main_window.py` (Test tab), `history_tab.py`, `stats_tab.py`, `viewer.py` (image + overlays), `workers.py` (threads), `common.py` |
| `tools/gen_samples.py` | synthetic test images with known defects |

## Checks

All images are NumPy arrays in OpenCV BGR order. A parameter that is left out of the config
disables its condition (except where a default is listed).

| Check | Frame | Measures | Algorithm | Key params |
|---|---|---|---|---|
| `brightness` | normal / flat | Mean luminance; % of clipped (> 250) and crushed (< 5) pixels | BGR to gray, mean and pixel percentages. Value = mean. Fails if the mean is out of range or clipping is too high. | `mean_low`, `mean_high`, `max_clipped_pct` |
| `sharpness` | normal (chart) | Focus score at the center and in the four corners | Variance of the Laplacian (`cv2.Laplacian`, computed once on the gray image) over the central 50 % region (the value) and four 20 % x 20 % corner regions (reported in the details, not thresholded). | `min_center`, `max_center` |
| `pixels` (`hot`) | dark | Count of hot pixels | 5x5 median (`cv2.medianBlur`); a pixel brighter than its median by more than the threshold is defective; each 8-connected component counts once. Up to 200 coordinates are listed. | `mode: hot`, `diff_threshold` (30), `max_defects` |
| `pixels` (`dead`) | white | Count of dead pixels | Same, for pixels darker than their median. | `mode: dead`, `diff_threshold` (30), `max_defects` |
| `shading` | flat | Corner-to-center brightness (vignetting) | Mean gray of the center patch and the four corner patches (each 10 % of the width and height). Value = min(corner) / center x 100. | `min_corner_pct` |
| `color` | flat | Color uniformity and white balance | 8x8 grid of mean colors converted to CIELAB; value = max CIE76 delta E of any cell against the mean of the central 2x2 cells. Also the global R/G and B/G channel ratios. Fails if delta E exceeds `max_delta_e` or either ratio is outside `ratio_low` .. `ratio_high`. | `max_delta_e`, `ratio_low`, `ratio_high` |
| `blemish` | flat | Dark spots (dust, stains) | Divide the image by a heavy Gaussian blur of itself (sigma = width / 20); pixels below `1 - contrast` form a mask that gets a 3x3 morphological opening; each connected component of at least `min_area` pixels is one blemish. Value = count. | `contrast` (0.05), `min_area` (30), `max_blobs` |
| `noise` | flat | - | **Not implemented.** `checks/noise.py` is an empty placeholder and the check is disabled in `config/models.yaml`. | - |

The `pixels` check appears twice in the config (`pixels_hot` and `pixels_dead`) because each
mode needs a different frame.

**Verdict rules.** `FAIL` if any check failed. Otherwise `INCOMPLETE` if any enabled check was
`SKIPPED` (its frame was not supplied) or `ERROR` (it raised, or is not implemented). Otherwise
`PASS`.

## Setup

Requires Python 3.11 or newer. The project was developed and tested on Windows 11 with
Python 3.14 (PySide6 6.12, matplotlib 3.11, OpenCV 5).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

This installs the `camqc` and `camqc-gui` commands.

## Usage

### Command line

```bash
python tools/gen_samples.py                 # writes synthetic images to samples/ (seed 42)
camqc init-db --db camqc.db

# one unit, up to four frames (--normal --flat --dark --white); exit code 0 = PASS
camqc run --model CAM_A --serial SN001 --station ST01 \
    --normal samples/sharp.png --flat samples/good_flat.png \
    --dark samples/dark_good.png --white samples/white_bad.png --db camqc.db

camqc batch --dir samples/ --model CAM_A --db camqc.db      # one unit per image, serial = file stem
camqc history --serial SN001 --db camqc.db [--detail]
camqc stats --from 2026-11-01 --to 2026-11-30 --db camqc.db
camqc report --run-id 1 --out report.html --db camqc.db     # HTML report of one run
camqc report --from 2026-11-01 --to 2026-11-30 --out runs.csv --db camqc.db
```

The `run` command above prints (the sample `white_bad.png` has 12 dead pixels, so the unit fails):

```
Serial SN001  Model CAM_A  Station ST01  Run #1  254 ms
check        status  value         limits      frame   note
-----------  ------  ------------  ----------  ------  --------------------------
brightness   PASS    124.94 gray   100 .. 160  normal
sharpness    PASS    3113.5 score  >= 100      normal
pixels_hot   PASS    0 count       <= 5        dark
pixels_dead  FAIL    12 count      <= 5        white   12 dead pixels above max 5
shading      PASS    91.856 %      >= 80       flat
color        PASS    3.7293 dE76   <= 10       flat
blemish      PASS    0 count       <= 2        flat
Verdict: FAIL
```

| Exit code | Meaning |
|---|---|
| 0 | PASS (and any successful query) |
| 1 | FAIL or INCOMPLETE (for `batch`: at least one unit did not pass) |
| 2 | Error: bad arguments, missing file, invalid config, database problem (for `batch`: at least one image could not be loaded or saved; this wins over 1) |

Useful options: `--config` (default: `config/models.yaml` in the current directory, else the
copy in the source tree) and `--no-images` (do not store input images; smaller database, but
the report then has no image), both for `run` and `batch` only; `--db`, `-v` for debug
logging and `--log-file` (default `logs/camqc.log`) work on every command.
`batch` gives every image the frame type chosen with `--frame` (default `normal`), so with a
config that needs several frames most checks end up `SKIPPED` and the unit `INCOMPLETE`.
`history`, `stats` and `report` need an existing database (`camqc init-db`).

### Desktop app

```bash
camqc-gui
```

Opens `camqc.db` in the current directory and writes `logs/camqc.log`. In the Test tab pick a
model, enter or scan the serial (Enter starts the run), choose the frames the model needs
(shown in the results table as `SKIPPED` if one is missing), and press **Run**. The chosen
frames are cleared after each run so a scan can never re-test the previous unit's images.

### Configuration

`config/models.yaml` lists models and, for each model, its checks:

```yaml
models:
  CAM_A:
    description: Standard module
    checks:
      brightness:
        frame: [normal, flat]          # a frame type, or a list tried in order
        params: {mean_low: 100, mean_high: 160, max_clipped_pct: 1.0}
      pixels_dead:
        check: pixels                  # the check module, if it differs from the key
        frame: white
        params: {mode: dead, diff_threshold: 50, max_defects: 5}
      noise:
        enabled: false                 # not implemented yet
        frame: flat
        params: {}
```

Unknown keys, unknown check names, bad frame types and models without an enabled check are
rejected with an error that names the key (for example
`models.CAM_A.checks.sharpness: missing required key 'frame'`).

## Adding a new check

1. **Write the module** `src/camqc/checks/saturation.py`. A check is one function that takes a
   BGR image and a params dict and returns a `CheckResult` built with `make_result`
   (`make_result(name, value, unit, low, high, details=None, overlay=None)`; it passes when
   `low <= value <= high`, and a `None` bound is unbounded):

   ```python
   """Saturation check: mean color saturation of a neutral target must stay below a limit."""

   from __future__ import annotations

   from typing import Any

   import cv2
   import numpy as np

   from camqc.checks.base import CheckResult, make_result


   def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
       """Measure the mean HSV saturation (0-255); fail above ``params["max_saturation"]``."""
       saturation = float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
       return make_result("saturation", saturation, "0-255", None, params.get("max_saturation"),
                          {"mean_saturation": saturation})
   ```

   To draw marks in the GUI and the HTML report, pass `overlay=[{"type": "circle", "x": ..,
   "y": .., "r": ..}, {"type": "rect", "x": .., "y": .., "w": .., "h": ..}]` in image pixel
   coordinates. Put a human-readable reason in `details["reason"]` when the check fails.
2. **Register it** in `src/camqc/checks/__init__.py`: import the module and add
   `"saturation": saturation` to `CHECK_MODULES`.
3. **Configure it** for the models that need it in `config/models.yaml`:

   ```yaml
   saturation:
     frame: flat
     params: {max_saturation: 20}
   ```
4. **Test it** in `tests/test_check_saturation.py` (see `tests/test_check_brightness.py`), using
   the images from `tools/gen_samples.py` (for example `good_flat.png` passes and
   `color_cast.png` fails with this limit).

The CLI, the GUI table, the database and the report need no changes.

## Development

```bash
pytest -q          # the whole suite, including headless GUI tests (QT_QPA_PLATFORM=offscreen)
```

Conventions are in `CLAUDE.md`: type hints, docstrings, small pure functions, no global
mutable state, `logging` instead of `print` (except CLI output).

## Limitations

- The `noise` check is not implemented.
- Thresholds are tuned for synthetic images only and must be calibrated with real camera data.
- The webcam capture uses OpenCV (`VideoCapture(0)`) and grabs one frame; there is no vendor
  camera SDK and no control over exposure or gain. It has been tested with a fake camera, not
  with real hardware.
- The GUI is covered by headless tests; it has not been run at a production station.
- `batch` and *Load folder* give every image a single frame type, so they only fit configs
  (or test setups) where one frame is enough.
- SQLite is a single local file: no server, no multi-station sharing, no user accounts.
- Input images are stored in the database as JPEG (about 0.3-1 MB per 1080p frame), the GUI
  always stores them, and nothing deletes old data, so the database grows over time.
  Stored images are JPEG-compressed and are meant as evidence, not for re-measurement.
- Timestamps are stored as local time without a time zone.
- English only.

## Future work

- Real camera SDK integration (exposure, gain, triggered capture).
- MTF / SFR measurement from a slanted-edge target.
- The missing `noise` check.
- PostgreSQL or SQL Server as the results database.
- MES integration through a REST API (push results, receive serials and work orders).
- A retention policy or prune command for stored images, and an option to turn image storage
  off in the GUI.
