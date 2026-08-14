from types import SimpleNamespace

from vision_server.cli import parse_args, resolve_show_preview


def test_default_args_leave_headless_unset():
    args = parse_args([])
    assert args.headless is None


def test_headless_flag():
    assert parse_args(["--headless"]).headless is True
    assert parse_args(["--no-headless"]).headless is False


def test_resolve_show_preview_respects_flags():
    assert resolve_show_preview(SimpleNamespace(headless=True)) is False
    assert resolve_show_preview(SimpleNamespace(headless=False)) is True
