"""Raw sources, pinned by SHA256. A download that does not match its pin is refused."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from floorcall.data import clinc, swda


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    dest: str  # relative to data/raw
    licence: str
    unzip: bool = False


SOURCES: dict[str, Source] = {
    "swda": Source(swda.SOURCE_URL, swda.SOURCE_SHA256, "swda/swda.zip", swda.LICENCE, unzip=True),
    "clinc": Source(
        clinc.SOURCE_URL, clinc.SOURCE_SHA256, "clinc150/data_oos_plus.json", clinc.LICENCE
    ),
    "clinc_domains": Source(
        clinc.DOMAINS_URL, clinc.DOMAINS_SHA256, "clinc150/domains.json", clinc.LICENCE
    ),
}


class ChecksumError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, raw_root: Path) -> Path:
    """Download source `name` into raw_root if absent, verify its pin, unzip if needed."""
    src = SOURCES[name]
    path = raw_root / src.dest
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", src.url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.replace(path)
    actual = _sha256(path)
    if actual != src.sha256:
        raise ChecksumError(f"{name}: {path} has sha256 {actual}, pinned {src.sha256}")
    if src.unzip:
        marker = path.parent / ".unzipped"
        if not marker.exists():
            with zipfile.ZipFile(path) as z:
                members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
                z.extractall(path.parent, members=members)
            marker.write_text(src.sha256, encoding="utf-8")
    return path
