# SPDX-License-Identifier: BSD-3-Clause
"""Repository-level tests for JOSS paper and metadata readiness."""

from pathlib import Path

from scripts.validate_joss_submission import validate_repository


ROOT = Path(__file__).resolve().parents[2]


def test_joss_validator_passes_before_archive_doi_is_minted():
    report = validate_repository(ROOT, require_archive_doi=False)

    assert report.errors == ()
    assert any("archive DOI" in warning for warning in report.warnings)


def test_joss_validator_requires_archive_doi_for_final_acceptance_gate():
    report = validate_repository(ROOT, require_archive_doi=True)

    assert any("archive DOI" in error for error in report.errors)
