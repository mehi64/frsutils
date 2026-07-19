# SPDX-License-Identifier: BSD-3-Clause
"""Repository-level tests for JOSS paper and metadata readiness."""

from pathlib import Path

from scripts.validate_joss_submission import validate_repository


ROOT = Path(__file__).resolve().parents[2]


def test_joss_validator_passes_with_archive_doi_recorded():
    report = validate_repository(ROOT, require_archive_doi=False)

    assert report.errors == ()
    assert any("Software archive DOI is recorded" in item for item in report.passed)
    assert not any("archive DOI" in warning for warning in report.warnings)


def test_joss_validator_passes_final_acceptance_gate_with_archive_doi():
    report = validate_repository(ROOT, require_archive_doi=True)

    assert report.errors == ()
    assert any("Software archive DOI is recorded" in item for item in report.passed)
