"""Smoke test — confirms the package structure is importable.

This test exists so CI has something to run before any real
implementation is written. Delete it once substantive tests exist.
"""

import src


def test_src_package_is_importable():
    """src package must be importable from the repo root."""
    assert src is not None


def test_submodules_are_importable():
    """All top-level submodules must be importable."""
    import src.api
    import src.ingest
    import src.models
    import src.pipeline

    assert all(m is not None for m in [src.ingest, src.pipeline, src.models, src.api])
