from types import SimpleNamespace

from vision_server.cli import (
    CAMERA_INDEX_FILE,
    parse_args,
    resolve_camera_index,
    resolve_show_preview,
    resolve_working_camera_index,
)


def test_default_args_leave_headless_unset():
    args = parse_args([])
    assert args.headless is None


def test_headless_flag():
    assert parse_args(["--headless"]).headless is True
    assert parse_args(["--no-headless"]).headless is False


def test_camera_index_flag():
    assert parse_args(["--camera-index", "2"]).camera_index == 2
    assert parse_args([]).camera_index is None


def test_list_cameras_flag():
    assert parse_args(["--list-cameras"]).list_cameras is True
    assert parse_args([]).list_cameras is False


def test_resolve_camera_index_prefers_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CAMERA_INDEX_FILE).write_text("3\n")
    monkeypatch.setenv("VISION_CAMERA_INDEX", "7")
    args = parse_args(["--camera-index", "1"])
    index, source = resolve_camera_index(args)
    assert index == 7
    assert "env var" in source


def test_resolve_camera_index_prefers_file_over_arg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISION_CAMERA_INDEX", raising=False)
    (tmp_path / CAMERA_INDEX_FILE).write_text("3\n")
    args = parse_args(["--camera-index", "1"])
    index, source = resolve_camera_index(args)
    assert index == 3
    assert CAMERA_INDEX_FILE in source


def test_resolve_working_camera_index_keeps_preferred(monkeypatch):
    monkeypatch.setattr(
        "vision_server.cli._camera_opens",
        lambda index: index == 2,
    )
    index, source = resolve_working_camera_index(2, "test")
    assert index == 2
    assert source == "test"


def test_resolve_working_camera_index_auto_fallback(monkeypatch):
    monkeypatch.setattr(
        "vision_server.cli._camera_opens",
        lambda index: index == 0,
    )
    index, source = resolve_working_camera_index(2, "test")
    assert index == 0
    assert "auto-detected" in source


def test_resolve_show_preview_respects_flags():
    assert resolve_show_preview(SimpleNamespace(headless=True)) is False
    assert resolve_show_preview(SimpleNamespace(headless=False)) is True
