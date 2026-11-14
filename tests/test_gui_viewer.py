"""Tests for overlay drawing (viewer.py) and the background workers (workers.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from camqc.checks.base import make_result  # noqa: E402
from camqc.gui import viewer, workers  # noqa: E402
from camqc.gui.viewer import ScaledImageLabel, StyledShape, overlay_shapes  # noqa: E402
from camqc.pipeline import (  # noqa: E402
    ERROR,
    FAIL,
    PASS,
    SKIPPED,
    CheckOutcome,
    Config,
    RunResult,
    parse_config,
)
from gui_support import png  # noqa: E402

RECT = {"type": "rect", "x": 100, "y": 40, "w": 200, "h": 120}
CIRCLE = {"type": "circle", "x": 700, "y": 240, "r": 100}


def colour_at(label: ScaledImageLabel, x: int, y: int) -> QColor:
    return label.pixmap().toImage().pixelColor(x, y)


def is_red(c: QColor) -> bool:
    return c.red() > 200 and c.green() < 90 and c.blue() < 90


def is_yellow(c: QColor) -> bool:
    return c.red() > 200 and c.green() > 160 and c.blue() < 90


def is_black(c: QColor) -> bool:
    return c.red() < 40 and c.green() < 40 and c.blue() < 40


def black_label(qapp: QApplication, label_size: tuple[int, int], image_size: tuple[int, int] = (960, 480)) -> ScaledImageLabel:
    label = ScaledImageLabel()
    label.show()  # resize events are only delivered to visible widgets
    label.resize(*label_size)
    pixmap = QPixmap(*image_size)
    pixmap.fill(Qt.GlobalColor.black)
    label.set_image(pixmap)
    return label


# --- drawing: correct scaling ----------------------------------------------------------


def test_overlays_are_scaled_with_the_image(qapp: QApplication) -> None:
    label = black_label(qapp, (480, 240))  # 960x480 image shown at scale 0.5
    label.set_overlays([StyledShape(RECT, "#ff0000", 3), StyledShape(CIRCLE, "#ffd400", 3)])
    QApplication.processEvents()
    assert label.pixmap().size().width() == 480
    assert is_red(colour_at(label, 50, 50))  # rect left edge: x 100 * 0.5
    assert is_red(colour_at(label, 100, 20))  # rect top edge: y 40 * 0.5
    assert is_red(colour_at(label, 150, 50))  # rect right edge: (100 + 200) * 0.5
    assert is_black(colour_at(label, 100, 50))  # rect interior is not filled
    assert is_black(colour_at(label, 120, 50)) and is_black(colour_at(label, 200, 120))  # nothing elsewhere
    assert is_yellow(colour_at(label, 400, 120))  # circle rightmost point: (700 + 100) * 0.5, 240 * 0.5
    assert is_yellow(colour_at(label, 300, 120))  # leftmost point
    assert is_yellow(colour_at(label, 350, 70))  # top point
    assert is_black(colour_at(label, 350, 120))  # center


def test_overlays_follow_resizes_and_keep_line_width(qapp: QApplication) -> None:
    label = black_label(qapp, (480, 240))
    label.set_overlays([StyledShape(RECT, "#ff0000", 3)])
    label.resize(960, 480)  # now scale 1.0
    QApplication.processEvents()
    assert label.pixmap().width() == 960
    assert is_red(colour_at(label, 100, 100)) and is_black(colour_at(label, 50, 100))
    # the pen stays ~3 screen pixels wide at any zoom (it is not scaled with the image)
    assert is_red(colour_at(label, 99, 100))
    assert is_black(colour_at(label, 97, 100)) and is_black(colour_at(label, 103, 100))


def test_aspect_ratio_uses_separate_x_and_y_scales(qapp: QApplication) -> None:
    label = black_label(qapp, (400, 400), image_size=(800, 600))  # scaled to 400x300 (0.5 both ways)
    label.set_overlays([StyledShape({"type": "rect", "x": 200, "y": 100, "w": 400, "h": 300}, "#ff0000", 3)])
    QApplication.processEvents()
    assert (label.pixmap().width(), label.pixmap().height()) == (400, 300)
    assert is_red(colour_at(label, 100, 100)) and is_red(colour_at(label, 200, 50))  # left x=100, top y=50
    assert is_red(colour_at(label, 300, 100)) and is_red(colour_at(label, 200, 200))  # right x=300, bottom y=200


def test_circle_centers_are_pixel_indices_so_they_are_drawn_at_the_pixel_middle(qapp: QApplication) -> None:
    pixmap = QPixmap(50, 50)
    pixmap.fill(Qt.GlobalColor.black)
    circle = {"type": "circle", "x": 2, "y": 2, "r": 1}  # the pixel (2, 2): its middle is (2.5, 2.5)
    viewer.draw_shapes(pixmap, [StyledShape(circle, "#ff0000", 3)], 10, 10)
    image = pixmap.toImage()
    assert is_red(image.pixelColor(35, 25)) and is_red(image.pixelColor(15, 25))  # (2.5 +- 1) * 10 = 35 / 15
    assert is_black(image.pixelColor(25, 25))  # centered on 25, not on 20


def test_drawing_never_modifies_the_source_image(qapp: QApplication) -> None:
    label = black_label(qapp, (480, 240))
    source = label._source
    assert source is not None
    label.set_overlays([StyledShape(RECT, "#ff0000", 3)])
    assert is_black(source.toImage().pixelColor(100, 100))  # the original pixmap is untouched


def test_set_image_clears_overlays_and_empty_list_removes_them(qapp: QApplication) -> None:
    label = black_label(qapp, (480, 240))
    label.set_overlays([StyledShape(RECT, "#ff0000", 3)])
    assert len(label.shapes) == 1
    label.set_overlays([])
    assert label.shapes == [] and is_black(colour_at(label, 50, 50))
    label.set_overlays([StyledShape(RECT, "#ff0000", 3)])
    pixmap = QPixmap(960, 480)
    pixmap.fill(Qt.GlobalColor.black)
    label.set_image(pixmap)
    assert label.shapes == [] and is_black(colour_at(label, 50, 50))


def test_unknown_shape_types_and_no_image_are_ignored(qapp: QApplication) -> None:
    label = black_label(qapp, (480, 240))
    label.set_overlays([StyledShape({"type": "polygon", "points": []}, "#ff0000", 3)])
    assert is_black(colour_at(label, 50, 50))
    empty = ScaledImageLabel()
    empty.set_overlays([StyledShape(RECT, "#ff0000", 3)])  # no image yet: nothing to draw on
    assert empty.pixmap().isNull() and empty.text() == "No image"


# --- which shapes are drawn --------------------------------------------------------------


def outcome(name: str, status: str, frame: str | None, overlay: list[dict[str, Any]] | None = None) -> CheckOutcome:
    if status in (PASS, FAIL):
        return CheckOutcome(name, status, make_result(name, 1.0, "u", None, None, None, overlay), None, frame, 1.0)
    return CheckOutcome(name, status, None, "boom", frame, 1.0)


@pytest.fixture
def run() -> RunResult:
    circle = {"type": "circle", "x": 5, "y": 6, "r": 7}
    outcomes = (
        outcome("a", FAIL, "dark", [circle]),
        outcome("b", PASS, "dark", [RECT, RECT]),
        outcome("c", PASS, "flat", [CIRCLE]),
        outcome("d", PASS, "dark", []),
        outcome("e", SKIPPED, None),
        outcome("f", ERROR, "dark"),
    )
    return RunResult("FAIL", outcomes, ("a",), 1.0)


def test_overlay_shapes_default_shows_all_checks_of_the_frame_thinly(run: RunResult) -> None:
    shapes = overlay_shapes(run, "dark")
    assert [s.shape["type"] for s in shapes] == ["circle", "rect", "rect"]
    assert [s.color for s in shapes] == [viewer.OVERLAY_COLORS[FAIL], viewer.OVERLAY_COLORS[PASS], viewer.OVERLAY_COLORS[PASS]]
    assert {s.width for s in shapes} == {viewer.THIN_WIDTH}
    assert [s.shape for s in overlay_shapes(run, "flat")] == [CIRCLE]
    assert overlay_shapes(run, "white") == []


def test_overlay_shapes_selected_check_only_and_highlighted(run: RunResult) -> None:
    shapes = overlay_shapes(run, "dark", selected="b")
    assert [s.shape for s in shapes] == [RECT, RECT]
    assert {(s.color, s.width) for s in shapes} == {(viewer.HIGHLIGHT_COLOR, viewer.HIGHLIGHT_WIDTH)}
    assert overlay_shapes(run, "dark", selected="a")[0].shape["type"] == "circle"


def test_overlay_shapes_selection_edge_cases(run: RunResult) -> None:
    assert overlay_shapes(run, "dark", selected="d") == []  # check without overlay
    assert overlay_shapes(run, "dark", selected="c") == []  # check used another frame
    assert overlay_shapes(run, "dark", selected="e") == overlay_shapes(run, "dark", selected="f") == []  # no result
    assert overlay_shapes(run, "dark", selected="nope") == []


# --- workers ---------------------------------------------------------------------------


class FakeCapture:
    """Stand-in for cv2.VideoCapture."""

    instances: list[FakeCapture] = []

    def __init__(self, opened: bool = True, frames: list[np.ndarray | None] | None = None) -> None:
        self.opened, self.frames, self.reads, self.released = opened, list(frames or []), 0, False
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        frame = self.frames.pop(0) if self.frames else None
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


def patch_camera(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    FakeCapture.instances = []
    monkeypatch.setattr(workers.cv2, "VideoCapture", lambda index: FakeCapture(**kwargs))


def test_capture_frame_returns_the_last_warmed_up_frame_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [np.full((4, 4, 3), v, np.uint8) for v in range(10)]
    patch_camera(monkeypatch, frames=frames)
    image = workers.capture_frame(warmup=3)
    cam = FakeCapture.instances[0]
    assert image[0, 0, 0] == 3 and cam.reads == 4 and cam.released  # 3 warm-up frames + the real one


def test_capture_frame_without_a_camera_gives_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_camera(monkeypatch, opened=False)
    with pytest.raises(workers.CameraError, match="No camera found") as info:
        workers.capture_frame()
    assert "not in use by another application" in str(info.value)
    assert FakeCapture.instances[0].released  # always released, even on failure


def test_capture_frame_when_the_camera_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_camera(monkeypatch, frames=[])
    with pytest.raises(workers.CameraError, match="did not return an image"):
        workers.capture_frame()
    assert FakeCapture.instances[0].released


def test_capture_worker_emits_frame_or_friendly_message(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = workers.CaptureWorker()
    got: list[Any] = []
    worker.captured.connect(lambda img: got.append(("ok", img.shape)))
    worker.failed.connect(lambda msg: got.append(("fail", msg)))
    monkeypatch.setattr(workers, "capture_frame", lambda: np.zeros((2, 3, 3), np.uint8))
    worker.run()
    monkeypatch.setattr(workers, "capture_frame", lambda: (_ for _ in ()).throw(workers.CameraError("No camera found.")))
    worker.run()
    monkeypatch.setattr(workers, "capture_frame", lambda: (_ for _ in ()).throw(OSError("backend exploded")))
    worker.run()
    assert got == [("ok", (2, 3, 3)), ("fail", "No camera found."), ("fail", "Camera error: OSError: backend exploded")]


@pytest.fixture
def brightness_config() -> Config:
    return parse_config({"models": {"M": {"checks": {"brightness": {
        "frame": "normal", "params": {"mean_low": 100, "mean_high": 160}}}}}})


def test_folder_worker_processes_every_file_and_survives_bad_ones(
    qapp: QApplication, tmp_path: Path, brightness_config: Config
) -> None:
    files = [png(tmp_path / "a.png", 128), png(tmp_path / "b.png", 20)]
    (tmp_path / "bad.png").write_bytes(b"garbage")
    files.insert(1, tmp_path / "bad.png")
    worker = workers.FolderWorker(files, "M", "normal", brightness_config)
    events: list[Any] = []
    worker.item_done.connect(lambda name, res, frames: events.append(("done", name, res.verdict, list(frames))))
    worker.item_failed.connect(lambda name, msg: events.append(("failed", name, msg)))
    worker.progress.connect(lambda done, total: events.append(("progress", done, total)))
    worker.finished.connect(lambda cancelled: events.append(("finished", cancelled)))
    worker.run()
    assert events[0] == ("done", "a.png", PASS, ["normal"])
    assert events[1] == ("progress", 1, 3)
    assert events[2][:2] == ("failed", "bad.png") and "cannot decode image" in events[2][2]
    assert events[4] == ("done", "b.png", FAIL, ["normal"])
    assert [e for e in events if e[0] == "progress"] == [("progress", 1, 3), ("progress", 2, 3), ("progress", 3, 3)]
    assert events[-1] == ("finished", False)


def test_folder_worker_cancel_stops_before_the_next_file(
    qapp: QApplication, tmp_path: Path, brightness_config: Config
) -> None:
    files = [png(tmp_path / f"{i}.png", 128) for i in range(5)]
    worker = workers.FolderWorker(files, "M", "normal", brightness_config)
    done: list[str] = []
    finished: list[bool] = []
    worker.item_done.connect(lambda name, *_: (done.append(name), worker.cancel() if len(done) == 2 else None))
    worker.finished.connect(finished.append)
    worker.run()
    assert done == ["0.png", "1.png"] and finished == [True]

    again = workers.FolderWorker(files, "M", "normal", brightness_config)
    again.cancel()  # cancelled before starting: nothing is processed
    out: list[Any] = []
    again.item_done.connect(lambda *a: out.append(a))
    again.finished.connect(out.append)
    again.run()
    assert out == [True]
