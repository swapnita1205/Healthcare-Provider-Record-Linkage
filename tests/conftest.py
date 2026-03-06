"""Pytest configuration for stable cross-folder imports."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _ensure_path(path: Path) -> None:
    """Prepend a path to sys.path if missing."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_path(PROJECT_ROOT)
if SCRIPTS_DIR.exists():
    _ensure_path(SCRIPTS_DIR)
