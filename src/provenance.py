"""Fingerprint executable benchmark sources/assets for reproducible comparisons."""

import hashlib
from pathlib import Path


def source_provenance() -> dict:
    """Hash source files and game sprites without importing the legacy game.

    The root is anchored to this file, independent of the launch directory.
    Reports carry the file manifest; checkpoints carry its combined SHA-256.
    """
    root = Path(__file__).resolve().parents[1]
    paths = [root / "benchmark.py", root / "benchmark_suite.py",
             root / "requirements-benchmark.txt", *sorted((root / "src").glob("*.py")),
             *sorted((root / "assets/sprites").glob("*.png"))]
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    manifest = "\n".join(f"{name}:{digest}" for name, digest in files.items())
    return {"sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(), "files": files}
