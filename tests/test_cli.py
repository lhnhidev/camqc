"""Tests for the command-line interface (direct main(argv) calls plus one subprocess check)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from camqc import cli, db
from camqc.checks.base import make_result
from camqc.pipeline import FAIL, PASS, CheckOutcome, RunResult

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
Sample = Callable[[str], np.ndarray]

MINI_CONFIG = """
models:
  M:
    checks:
      brightness:
        frame: normal
        params: {mean_low: 100, mean_high: 160}
"""


def write_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def solid(value: int) -> np.ndarray:
    return np.full((60, 80, 3), value, np.uint8)


@pytest.fixture(scope="session")
def frames(tmp_path_factory: pytest.TempPathFactory, make_sample: Sample) -> dict[str, Path]:
    """Full-size frame files for CAM_A: good ones plus dark_bad."""
    root = tmp_path_factory.mktemp("frames")
    noise = np.random.default_rng(0).normal(200, 2, (1080, 1920, 1)).clip(0, 255)
    return {
        "normal": write_png(root / "normal.png", make_sample("sharp.png")),
        "flat": write_png(root / "flat.png", make_sample("good_flat.png")),
        "dark": write_png(root / "dark.png", make_sample("dark_good.png")),
        "dark_bad": write_png(root / "dark_bad.png", make_sample("dark_bad.png")),
        "white": write_png(root / "white.png", np.repeat(noise, 3, axis=2).astype(np.uint8)),
    }


@pytest.fixture
def env(tmp_path: Path) -> list[str]:
    """Common options pointing the database and log into tmp_path."""
    return ["--db", str(tmp_path / "t.db"), "--log-file", str(tmp_path / "logs" / "t.log")]


@pytest.fixture
def mini_config(tmp_path: Path) -> list[str]:
    path = tmp_path / "mini.yaml"
    path.write_text(MINI_CONFIG, encoding="utf-8")
    return ["--config", str(path), "--model", "M"]


def run_args(frames: dict[str, Path], dark: str = "dark") -> list[str]:
    return ["run", "--model", "CAM_A", "--serial", "S001", "--station", "ST01",
            "--normal", str(frames["normal"]), "--flat", str(frames["flat"]),
            "--dark", str(frames[dark]), "--white", str(frames["white"])]


def test_load_image_handles_non_ascii_paths(tmp_path: Path) -> None:
    path = tmp_path / "ảnh mẫu" / "ảnh.png"
    path.parent.mkdir()
    ok, encoded = cv2.imencode(".png", solid(90))  # cv2.imwrite cannot write such paths on Windows
    assert ok
    encoded.tofile(str(path))
    assert cli.load_image(path).shape == (60, 80, 3)


def test_format_table_with_no_rows() -> None:
    assert cli.format_table(["a", "bb"], []).splitlines() == ["a  bb", "-  --"]


def test_format_table_alignment_is_unaffected_by_colors() -> None:
    headers = ["name", "status", "n"]
    rows = [["brightness", "PASS", 5], ["x", "FAIL", 1234]]
    plain = cli.format_table(headers, rows, colorize=False, align="llr")
    colored = cli.format_table(headers, rows, colorize=True, align="llr")
    assert ANSI.sub("", colored) == plain
    assert "\x1b[32mPASS\x1b[0m" in colored and "\x1b[31mFAIL\x1b[0m" in colored
    lines = plain.splitlines()
    assert lines[0].startswith("name        status") and lines[1].startswith("----------  ------")
    assert lines[2].endswith("   5") and lines[3].endswith("1234")


def test_use_color(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    monkeypatch.setattr(cli, "_enable_windows_ansi", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.use_color(Stream(True)) is True
    assert cli.use_color(Stream(False)) is False
    assert cli.use_color(object()) is False
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.use_color(Stream(True)) is False


@pytest.mark.parametrize(
    ("low", "high", "text"),
    [(100, 160, "100 .. 160"), (100, None, ">= 100"), (None, 5, "<= 5"), (None, None, "-"), (0.5, 1.25, "0.5 .. 1.25")],
)
def test_format_limits(low: float | None, high: float | None, text: str) -> None:
    assert cli.format_limits(low, high) == text


def test_exit_code_mapping() -> None:
    assert [cli.exit_code_for(v) for v in ("PASS", "FAIL", "INCOMPLETE")] == [0, 1, 1]
