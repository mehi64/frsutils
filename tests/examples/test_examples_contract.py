# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for Phase 7 release-ready examples."""

import subprocess
import sys
from pathlib import Path


def test_phase7_public_api_quickstart_runs_from_source_checkout():
    """Public API quickstart example should run successfully."""
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
    """Benchmark smoke example should write JSON/CSV artifacts."""
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
