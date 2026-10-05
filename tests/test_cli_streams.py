"""Piped or redirected output on Windows is cp1252; the CLI makes it UTF-8 before writing."""

import io
import sys

import pytest

from floorcall.cli import utf8_streams


def test_a_cp1252_stream_is_made_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(UnicodeEncodeError):
        stream.write("━━ ✓ ✗ ▸")
        stream.flush()
    utf8_streams()
    assert stream.encoding == "utf-8"
    stream.write("━━ ✓ ✗ ▸")
    stream.flush()
    assert raw.getvalue().decode("utf-8").endswith("━━ ✓ ✗ ▸")


def test_a_stream_that_is_not_a_text_wrapper_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.StringIO()  # what pytest's capture and many test harnesses install
    monkeypatch.setattr(sys, "stdout", buffer)
    utf8_streams()
    assert sys.stdout is buffer
