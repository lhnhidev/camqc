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


def test_init_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "new" / "camqc.db"
    assert cli.main(["init-db", "--db", str(path), "--log-file", str(tmp_path / "l.log")]) == 0
    assert path.is_file() and "Initialized database" in capsys.readouterr().out
    db.init_db(path).close()  # idempotent


def test_default_log_file_is_logs_camqc_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init-db"]) == 0
    assert (tmp_path / "camqc.db").is_file() and (tmp_path / "logs" / "camqc.log").is_file()


def test_run_pass(frames: dict[str, Path], env: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([*run_args(frames), *env]) == 0
    out = capsys.readouterr().out
    assert "Verdict: PASS" in out and "Serial S001  Model CAM_A  Station ST01  Run #1" in out
    for name in ("brightness", "sharpness", "pixels_hot", "pixels_dead", "shading", "color", "blemish"):
        assert re.search(rf"^{name}\s+PASS", out, re.M), name
    assert "\x1b[" not in out  # not a TTY -> no colors
    conn = db.init_db(tmp_path / "t.db")
    (stored,) = db.find_by_serial(conn, "S001")
    assert stored["verdict"] == PASS and stored["station"] == "ST01" and stored["model"] == "CAM_A"
    assert len(db.get_run_detail(conn, stored["id"])["results"]) == 7  # type: ignore[index]
    conn.close()
    log = (tmp_path / "logs" / "t.log").read_text(encoding="utf-8")
    assert "run 1 serial=S001 model=CAM_A verdict=PASS" in log


def test_run_fail_exit_code_1(frames: dict[str, Path], env: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([*run_args(frames, dark="dark_bad"), *env]) == 1
    out = capsys.readouterr().out
    assert "Verdict: FAIL" in out
    assert re.search(r"^pixels_hot\s+FAIL .*hot pixels above max 5", out, re.M)


def test_run_incomplete_exit_code_1(frames: dict[str, Path], env: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", "--model", "CAM_A", "--serial", "S002", "--normal", str(frames["normal"]), *env])
    out = capsys.readouterr().out
    assert code == 1 and "Verdict: INCOMPLETE" in out
    assert re.search(r"^pixels_dead\s+SKIPPED .*required frame missing", out, re.M)


def test_run_colors_when_stdout_is_a_tty(
    tmp_path: Path, env: list[str], mini_config: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "use_color", lambda stream: True)
    good = write_png(tmp_path / "g.png", solid(128))
    assert cli.main(["run", "--serial", "X", "--normal", str(good), *mini_config, *env]) == 0
    assert "\x1b[32mPASS\x1b[0m" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--normal", "nope.png"], "image not found"),
        ([], "give at least one frame"),
        (["--normal", "{bad}"], "cannot decode image"),
        (["--normal", "{ok}", "--model", "CAM_Z"], "unknown model 'CAM_Z'"),
        (["--normal", "{ok}", "--config", "missing.yaml"], "cannot read config file"),
    ],
)
def test_run_errors_exit_2(extra: list[str], message: str, tmp_path: Path, env: list[str],
                           capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    ok = write_png(tmp_path / "ok.png", solid(128))
    extra = [e.replace("{bad}", str(bad)).replace("{ok}", str(ok)) for e in extra]
    base = ["run", "--model", "CAM_A", "--serial", "S1"]
    if "--model" in extra:  # let the test's own --model win
        base = ["run", "--serial", "S1"]
    assert cli.main([*base, *extra, *env]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err


def test_invalid_config_is_reported(tmp_path: Path, env: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("models:\n  M:\n    checks:\n      color: {}\n", encoding="utf-8")
    img = write_png(tmp_path / "i.png", solid(128))
    code = cli.main(["run", "--model", "M", "--serial", "S", "--normal", str(img), "--config", str(cfg), *env])
    assert code == 2 and "missing required key 'frame'" in capsys.readouterr().err


def test_load_image_handles_non_ascii_paths(tmp_path: Path) -> None:
    path = tmp_path / "ảnh mẫu" / "ảnh.png"
    path.parent.mkdir()
    ok, encoded = cv2.imencode(".png", solid(90))  # cv2.imwrite cannot write such paths on Windows
    assert ok
    encoded.tofile(str(path))
    assert cli.load_image(path).shape == (60, 80, 3)


def test_batch_uses_filename_stem_as_serial(
    tmp_path: Path, env: list[str], mini_config: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "imgs"
    write_png(folder / "S001.png", solid(128))
    write_png(folder / "S002.png", solid(20))
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    assert cli.main(["batch", "--dir", str(folder), *mini_config, *env]) == 1
    out = capsys.readouterr().out
    assert re.search(r"^S001\.png\s+S001\s+PASS", out, re.M)
    assert re.search(r"^S002\.png\s+S002\s+FAIL\s+brightness", out, re.M)
    assert "notes.txt" not in out and "2 images: 1 PASS, 1 not PASS, 0 errors" in out
    conn = db.init_db(tmp_path / "t.db")
    assert [db.find_by_serial(conn, s)[0]["verdict"] for s in ("S001", "S002")] == [PASS, FAIL]
    conn.close()


def test_batch_all_pass_exit_0(tmp_path: Path, env: list[str], mini_config: list[str]) -> None:
    write_png(tmp_path / "d" / "A.png", solid(120))
    write_png(tmp_path / "d" / "B.jpg", solid(130))
    assert cli.main(["batch", "--dir", str(tmp_path / "d"), *mini_config, *env]) == 0


def test_batch_continues_after_a_bad_file_and_exits_2(
    tmp_path: Path, env: list[str], mini_config: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "d"
    write_png(folder / "A.png", solid(128))
    (folder / "B.png").write_bytes(b"garbage")
    write_png(folder / "C.png", solid(128))
    assert cli.main(["batch", "--dir", str(folder), *mini_config, *env]) == 2
    captured = capsys.readouterr()
    assert "cannot decode image" in captured.err and "2 PASS" in captured.out and "1 errors" in captured.out
    conn = db.init_db(tmp_path / "t.db")
    assert db.find_by_serial(conn, "A") and db.find_by_serial(conn, "C") and not db.find_by_serial(conn, "B")
    conn.close()


def test_batch_survives_empty_and_blank_named_files(
    tmp_path: Path, env: list[str], mini_config: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "d"
    write_png(folder / "A.png", solid(128))
    (folder / "zero.png").write_bytes(b"")  # 0-byte file: OpenCV asserts on an empty buffer
    write_png(folder / " .png", solid(128))  # serial is blank after stripping
    write_png(folder / "Z.png", solid(128))
    assert cli.main(["batch", "--dir", str(folder), *mini_config, *env]) == 2
    captured = capsys.readouterr()
    assert "cannot decode image" in captured.err and "serial must not be empty" in captured.err
    assert "2 PASS" in captured.out and "2 errors" in captured.out
    conn = db.init_db(tmp_path / "t.db")
    assert db.find_by_serial(conn, "A") and db.find_by_serial(conn, "Z")
    conn.close()
    log = (tmp_path / "logs" / "t.log").read_text(encoding="utf-8")
    assert "cannot decode image" in log and "serial must not be empty" in log


def test_run_rejects_blank_serial(tmp_path: Path, env: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    img = write_png(tmp_path / "i.png", solid(128))
    assert cli.main(["run", "--model", "CAM_A", "--serial", " ", "--normal", str(img), *env]) == 2
    assert "--serial must not be blank" in capsys.readouterr().err


def test_run_reports_zero_byte_image(tmp_path: Path, env: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "e.png"
    empty.write_bytes(b"")
    assert cli.main(["run", "--model", "CAM_A", "--serial", "S", "--normal", str(empty), *env]) == 2
    assert "cannot decode image" in capsys.readouterr().err


def test_batch_frame_option_and_directory_errors(
    tmp_path: Path, env: list[str], mini_config: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_png(tmp_path / "d" / "A.png", solid(128))
    assert cli.main(["batch", "--dir", str(tmp_path / "d"), "--frame", "flat", *mini_config, *env]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out  # config needs a 'normal' frame
    (tmp_path / "empty").mkdir()
    assert cli.main(["batch", "--dir", str(tmp_path / "empty"), *mini_config, *env]) == 2
    assert "no images found" in capsys.readouterr().err
    assert cli.main(["batch", "--dir", str(tmp_path / "nope"), *mini_config, *env]) == 2
    assert "directory not found" in capsys.readouterr().err


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
