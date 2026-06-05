"""Download infrastructure — strict TDD, one test at a time.

Tests for SHA-256 verification and idempotent download helpers.

Note on HTTP calls: download_file() makes real HTTP requests to external
services. These cannot produce a deterministic RED state in a unit test
without mocking the entire HTTP layer — which would test our mock, not the
behavior. HTTP I/O is therefore not unit-tested here. The SHA-256 verification
and idempotency logic — which are the correctness-critical parts — are tested.
"""

import hashlib

from src.ingest.download import compute_sha256, download_file, is_current


def test_compute_sha256_returns_64_char_hex_string(tmp_path):
    """SHA-256 digest must be a 64-character lowercase hex string."""
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello")
    result = compute_sha256(f)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_sha256_matches_known_digest(tmp_path):
    """SHA-256 of known content must equal the known digest."""
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert compute_sha256(f) == expected


def test_compute_sha256_differs_for_different_content(tmp_path):
    """Different file contents must produce different digests."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert compute_sha256(a) != compute_sha256(b)


def test_is_current_returns_false_when_file_absent(tmp_path):
    """is_current must return False when the file does not exist."""
    missing = tmp_path / "missing.csv"
    assert is_current(missing, "abc123") is False


def test_is_current_returns_false_when_hash_mismatch(tmp_path):
    """is_current must return False when file exists but hash differs."""
    f = tmp_path / "data.csv"
    f.write_bytes(b"old content")
    assert is_current(f, "wronghash") is False


def test_is_current_returns_true_when_file_exists_and_hash_matches(tmp_path):
    """is_current must return True only when file exists AND hash matches."""
    f = tmp_path / "data.csv"
    f.write_bytes(b"correct content")
    correct_hash = compute_sha256(f)
    assert is_current(f, correct_hash) is True


def test_download_file_skips_http_when_file_is_already_current(tmp_path):
    """download_file must return immediately when is_current — no HTTP call made.

    A deliberately invalid URL proves no request was attempted: if it were,
    requests would raise a connection error before we could assert anything.
    """
    f = tmp_path / "data.csv"
    f.write_bytes(b"correct content")
    correct_hash = compute_sha256(f)

    result = download_file("http://invalid.invalid/will-fail-if-called", f, correct_hash)

    assert result == f
    assert f.read_bytes() == b"correct content"
