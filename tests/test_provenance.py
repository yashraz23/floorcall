"""Where a number came from, inside and outside a git checkout."""

import subprocess
from typing import Any

import pytest

from floorcall import provenance


def test_outside_git_the_bundle_commit_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.setenv("FLOORCALL_CODE", "abc1234-kaggle")
    assert provenance.git_head() == "abc1234-kaggle"
    monkeypatch.delenv("FLOORCALL_CODE")
    assert provenance.git_head() == "unknown"


def test_inside_the_repo_it_is_the_head_commit() -> None:
    head = provenance.git_head()
    assert head != "unknown" and len(head.removesuffix("-dirty")) >= 7
