"""Download infrastructure: SHA-256 verification and idempotent file fetching."""

import hashlib
from pathlib import Path

import requests


def compute_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_current(path: Path, expected_sha256: str) -> bool:
    """Return True if the file exists and its SHA-256 matches expected_sha256."""
    if not path.exists():
        return False
    return compute_sha256(path) == expected_sha256


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download url to dest, skipping if is_current(dest, expected_sha256).

    Note: HTTP I/O is not unit-tested — see tests/ingest/download/test_download.py
    for the rationale. Correctness of SHA-256 verification is tested separately.
    """
    if expected_sha256 and is_current(dest, expected_sha256):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    with requests.get(url, stream=True, timeout=60) as r:  # pragma: no cover
        r.raise_for_status()  # pragma: no cover
        with open(dest, "wb") as f:  # pragma: no cover
            for chunk in r.iter_content(chunk_size=65536):  # pragma: no cover
                f.write(chunk)  # pragma: no cover
    return dest  # pragma: no cover
