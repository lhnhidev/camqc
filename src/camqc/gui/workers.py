"""Background workers for the GUI: pipeline runs, webcam capture and folder batches.

Each worker is a ``QObject`` meant to be moved to a ``QThread``; results come back through
signals (queued to the GUI thread). Workers never touch widgets or SQLite connections.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from camqc.cli import CliError, load_image
from camqc.pipeline import Config, run_pipeline

logger = logging.getLogger(__name__)

CAMERA_INDEX = 0
CAMERA_WARMUP_FRAMES = 5  # cameras need a few frames for auto exposure / white balance
NO_CAMERA_MESSAGE = (
    "No camera found. Check that a camera is connected and not in use by another application."
)


class CameraError(RuntimeError):
    """Raised with a user-presentable message when a webcam frame cannot be captured."""


def capture_frame(index: int = CAMERA_INDEX, warmup: int = CAMERA_WARMUP_FRAMES) -> np.ndarray:
    """Grab one frame from the webcam ``index`` with ``cv2.VideoCapture``.

    Args:
        index: Camera index (0 = default camera).
        warmup: Frames read and discarded first so exposure can settle.

    Returns:
        The captured BGR frame.

    Raises:
        CameraError: If no camera can be opened or it returns no image.
    """
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise CameraError(NO_CAMERA_MESSAGE)
        frame = None
        for _ in range(max(warmup, 0) + 1):
            ok, grabbed = cap.read()
            if not ok or grabbed is None:
                raise CameraError("The camera did not return an image. Try again or reconnect it.")
            frame = grabbed
        return frame  # type: ignore[return-value]
    finally:
        cap.release()


class PipelineWorker(QObject):
    """Runs the pipeline off the GUI thread.

    Signals:
        finished(RunResult): the run completed.
        failed(str): the pipeline itself raised (individual check errors are not failures).
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, model: str, frames: Mapping[str, np.ndarray], config: Config) -> None:
        super().__init__()
        self._model, self._frames, self._config = model, dict(frames), config

    @Slot()
    def run(self) -> None:
        """Execute the pipeline and emit the outcome."""
        try:
            result = run_pipeline(self._model, self._frames, self._config)
        except Exception as exc:  # noqa: BLE001 - never let a worker exception kill the thread silently
            logger.exception("pipeline failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)


class CaptureWorker(QObject):
    """Grabs one webcam frame off the GUI thread (opening a camera can take seconds).

    Signals:
        captured(ndarray): the BGR frame.
        failed(str): a friendly error message.
    """

    captured = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        """Capture a frame and emit it, or emit a friendly error."""
        try:
            frame = capture_frame()
        except CameraError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - e.g. an OpenCV backend error
            logger.exception("camera capture failed")
            self.failed.emit(f"Camera error: {type(exc).__name__}: {exc}")
        else:
            self.captured.emit(frame)


class FolderWorker(QObject):
    """Runs the pipeline on every image of a folder, one unit per image.

    Signals:
        progress(done, total): emitted after each file (also after errors).
        item_done(file_name, RunResult, frames): one image processed; ``frames`` maps the
            frame type to the loaded image so the GUI can show it with overlays.
        item_failed(file_name, message): the image could not be loaded or processed.
        finished(cancelled): all files done, or :meth:`cancel` was called.
    """

    progress = Signal(int, int)
    item_done = Signal(str, object, object)
    item_failed = Signal(str, str)
    finished = Signal(bool)

    def __init__(self, files: Sequence[Path], model: str, frame_type: str, config: Config) -> None:
        super().__init__()
        self._files, self._model, self._frame_type, self._config = list(files), model, frame_type, config
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the worker to stop after the file it is processing (thread-safe)."""
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        """Process the files in order, emitting progress and per-file results."""
        total = len(self._files)
        for index, path in enumerate(self._files, start=1):
            if self._cancel.is_set():
                break
            try:
                frames = {self._frame_type: load_image(path)}
                result = run_pipeline(self._model, frames, self._config)
            except CliError as exc:
                self.item_failed.emit(path.name, str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                logger.exception("folder item %s failed", path)
                self.item_failed.emit(path.name, f"{type(exc).__name__}: {exc}")
            else:
                self.item_done.emit(path.name, result, frames)
            self.progress.emit(index, total)
        self.finished.emit(self._cancel.is_set())
