# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the installed public API release validator."""

from __future__ import annotations

import json
from pathlib import Path
import stat

from scripts.validate_installed_public_api import (
    _find_writable_path,
    main,
    validate_public_api,
)


def test_public_api_release_validator_passes_from_source_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The validator covers all public models without writing to the CWD."""
    monkeypatch.chdir(tmp_path)

    report = validate_public_api()

    assert report["status"] == "success"
    assert report["checks"]["dense_blockwise_models_validated"] == 3
    assert report["checks"]["scorer_models_validated"] == 3
    assert report["checks"]["public_calls_silent"] is True
    assert list(tmp_path.iterdir()) == []


def test_public_api_release_validator_writes_json_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI writes a successful machine-readable report when requested."""
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "reports" / "public_api_validation.json"

    return_code = main(["--output-json", str(output_path)])

    assert return_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert {item["model"] for item in report["model_results"]} == {
        "itfrs",
        "vqrs",
        "owafrs",
    }


def test_find_writable_path_detects_and_accepts_permission_modes(
    tmp_path: Path,
) -> None:
    """Read-only validation relies on permission bits rather than user identity."""
    child = tmp_path / "child.txt"
    child.write_text("validation", encoding="utf-8")

    assert _find_writable_path(tmp_path) is not None

    child.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    tmp_path.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    try:
        assert _find_writable_path(tmp_path) is None
    finally:
        tmp_path.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
        child.chmod(stat.S_IRUSR | stat.S_IWUSR)
