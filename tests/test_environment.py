"""Guards on the environment itself, not on floorcall's code.

These fail loudly on the mistakes that would otherwise surface as a hang or a wrong number.
"""

import importlib.util
from importlib.metadata import version

# The adapter is written against this release's source; see src/floorcall/model/laya_adapter.py.
PINNED_LAYA = "0.3.21"


def test_tensorflow_is_not_installed() -> None:
    # transformers probes for TensorFlow at import, and TF's abseil runtime can deadlock model
    # construction (Laya CI config; docs/DECISIONS.md D-002). The failure mode is a hang, not an
    # error, so it is cheaper to refuse the environment outright.
    assert importlib.util.find_spec("tensorflow") is None, (
        "tensorflow is importable in this environment; remove it (see docs/DECISIONS.md D-002)"
    )


def test_laya_is_the_pinned_version() -> None:
    assert version("laya") == PINNED_LAYA, (
        f"laya {version('laya')} is installed but the adapter targets {PINNED_LAYA}; "
        "re-read the new version's source before changing the pin"
    )
