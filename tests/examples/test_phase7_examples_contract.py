"""
@file test_phase7_examples_contract.py
@brief Contract tests for Phase 7 release-ready examples.

These tests ensure the public API and benchmark smoke examples remain executable
from a source checkout. They are intentionally small and CPU-only.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# test_phase7_public_api...            Runs the public API quickstart example
# test_phase7_benchmark...             Runs the benchmark smoke example

# ✅ Design Patterns & Clean Code Notes
# - Contract Testing: examples shown to users must execute
# - Public API Boundary: examples import from FRsutils.api or benchmark facade
# - Optional Dependency Boundary: no CuPy required for these tests
##############################################
"""

import subprocess
import sys
from pathlib import Path


def test_phase7_public_api_quickstart_runs_from_source_checkout():
    """@brief Public API quickstart example should run successfully."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "phase7_public_api_quickstart.py"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FRsutils Phase 7 public API quickstart" in completed.stdout
    assert "dense/blockwise equivalence: OK" in completed.stdout


def test_phase7_benchmark_smoke_example_writes_outputs(tmp_path):
    """@brief Benchmark smoke example should write JSON/CSV artifacts."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "phase7_benchmark_smoke.py"
    output_dir = tmp_path / "phase7_benchmark_smoke"

    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FRsutils Phase 7 benchmark smoke complete" in completed.stdout
    assert (output_dir / "phase7_benchmark_smoke.json").exists()
    assert (output_dir / "phase7_benchmark_smoke.csv").exists()
