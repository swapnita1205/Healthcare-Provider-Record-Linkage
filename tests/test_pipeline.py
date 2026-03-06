"""Integration tests for pipeline runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_main_steps_ingest_runs_if_sample_exists() -> None:
    """Run ingest step through main.py when sample data is available."""
    sample_path = PROJECT_ROOT / "data" / "Dataset_A_sample.csv"
    if not sample_path.exists():
        pytest.skip("Sample file data/Dataset_A_sample.csv not found; skipping ingest integration test.")

    result = subprocess.run(
        [sys.executable, "main.py", "--steps", "ingest"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"main.py ingest failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.slow
def test_ingest_creates_providers_a_parquet_if_sample_exists() -> None:
    """Validate providers_a parquet creation after ingest step."""
    sample_path = PROJECT_ROOT / "data" / "Dataset_A_sample.csv"
    if not sample_path.exists():
        pytest.skip("Sample file data/Dataset_A_sample.csv not found; skipping ingest artifact test.")

    result = subprocess.run(
        [sys.executable, "main.py", "--steps", "ingest"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ingest execution failed.\nSTDERR:\n{result.stderr}"
    assert (PROJECT_ROOT / "outputs" / "providers_a.parquet").exists()
