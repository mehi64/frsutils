# SPDX-License-Identifier: BSD-3-Clause
"""Validate repository metadata and paper structure for a JOSS submission.

This script performs deterministic checks that can run locally and in continuous
integration. Account-level and platform-level checks, such as enabling GitHub
Issues or creating a Zenodo DOI, remain explicit manual checklist items.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
# import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    "paper.md",
    "paper.bib",
    "pyproject.toml",
    "studies/fuzzy_rough_reference_study/README.md",
    "studies/fuzzy_rough_reference_study/run_study.py",
    "studies/fuzzy_rough_reference_study/study_config.json",
    "studies/fuzzy_rough_reference_study/results/study_manifest.json",
)

REQUIRED_PAPER_SECTIONS = (
    "Summary",
    "Statement of need",
    "State of the field",
    "Software design",
    "Research impact statement",
    "AI usage disclosure",
    "Acknowledgements",
)

ORCID_PATTERN = re.compile(r"(?:https://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class ValidationReport:
    """Contain JOSS-readiness validation errors, warnings, and passed checks."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    passed: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the validation completed without errors."""

        return not self.errors


def _read_text(path: Path) -> str:
    """Read UTF-8 text from `path`."""

    return path.read_text(encoding="utf-8")


def _extract_project_table(pyproject_path: Path) -> dict[str, object]:
    """Return the PEP 621 project metadata table from `pyproject.toml`."""

    with pyproject_path.open("rb") as stream:
        data = tomllib.load(stream)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml does not contain a [project] table")
    return project


def _extract_cff_scalar(cff_text: str, key: str) -> str | None:
    """Return a simple top-level scalar value from CFF YAML text."""

    pattern = re.compile(rf"^{re.escape(key)}:\s*[\"']?([^\n\"']+)[\"']?\s*$", re.MULTILINE)
    match = pattern.search(cff_text)
    return match.group(1).strip() if match else None


def _extract_paper_sections(paper_text: str) -> set[str]:
    """Return level-one Markdown section headings from the paper."""

    return {
        match.group(1).strip()
        for match in re.finditer(r"^#\s+(.+?)\s*$", paper_text, re.MULTILINE)
    }


def _extract_citation_keys(paper_text: str) -> set[str]:
    """Return Pandoc citation keys used by the paper."""

    return set(re.findall(r"@([A-Za-z0-9:_-]+)", paper_text))


def _extract_bibliography_keys(bib_text: str) -> set[str]:
    """Return entry keys defined in a BibTeX bibliography."""

    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield repository Markdown files that may contain relative links."""

    yield root / "README.md"
    yield root / "CONTRIBUTING.md"
    yield root / "SUPPORT.md"
    yield from sorted((root / "docs").rglob("*.md"))
    yield from sorted((root / "studies").rglob("*.md"))


def _validate_relative_links(root: Path) -> list[str]:
    """Return errors for missing local targets in Markdown links."""

    errors: list[str] = []
    for markdown_path in _iter_markdown_files(root):
        if not markdown_path.exists():
            continue
        text = _read_text(markdown_path)
        for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
            target = raw_target.strip().split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            resolved = (markdown_path.parent / target).resolve()
            if not resolved.exists():
                relative_source = markdown_path.relative_to(root)
                errors.append(f"Broken link in {relative_source}: {raw_target}")
    return errors


def validate_repository(
    root: Path,
    *,
    require_archive_doi: bool = False,
) -> ValidationReport:
    """Validate the local repository against deterministic JOSS requirements.

    Parameters
    ----------
    root : pathlib.Path
        Repository root containing `paper.md` and `pyproject.toml`.
    require_archive_doi : bool, default=False
        Treat an absent software archive DOI as an error instead of a warning.

    Returns
    -------
    report : ValidationReport
        Collected errors, warnings, and successful checks.
    """

    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    missing_files = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing_files:
        errors.append(f"Missing required files: {', '.join(missing_files)}")
    else:
        passed.append("Required repository, paper, citation, and study files exist")

    lowercase_package = root / "frsutils"
    uppercase_package = root / "FRsutils"
    if not lowercase_package.is_dir():
        errors.append("Lowercase package directory `frsutils/` is missing")
    elif uppercase_package.exists():
        errors.append("Obsolete uppercase package directory `FRsutils/` exists")
    else:
        passed.append("Canonical lowercase package directory is present")

    if errors and not (root / "paper.md").exists():
        return ValidationReport(tuple(errors), tuple(warnings), tuple(passed))

    paper_text = _read_text(root / "paper.md")
    paper_sections = _extract_paper_sections(paper_text)
    missing_sections = [
        section for section in REQUIRED_PAPER_SECTIONS if section not in paper_sections
    ]
    if missing_sections:
        errors.append(f"Paper is missing required sections: {', '.join(missing_sections)}")
    else:
        passed.append("Paper contains all required JOSS sections")

    paper_body = re.sub(
        r"---.*?---",
        "",
        paper_text,
        count=1,
        flags=re.DOTALL,
    )
    word_count = len(re.findall(r"\b[\w'-]+\b", paper_body))
    if not 750 <= word_count <= 1750:
        errors.append(f"Paper word count {word_count} is outside the JOSS range 750-1750")
    else:
        passed.append(f"Paper word count is within the JOSS range ({word_count})")

    bibliography_text = _read_text(root / "paper.bib")
    used_keys = _extract_citation_keys(paper_text)
    defined_keys = _extract_bibliography_keys(bibliography_text)
    missing_keys = sorted(used_keys - defined_keys)
    if missing_keys:
        errors.append(f"Missing BibTeX entries: {', '.join(missing_keys)}")
    else:
        passed.append(f"All {len(used_keys)} paper citation keys resolve")

    ai_heading = "# AI usage disclosure"
    ai_text = paper_text.split(ai_heading, maxsplit=1)[1] if ai_heading in paper_text else ""
    next_heading = re.search(r"\n#\s+", ai_text)
    if next_heading:
        ai_text = ai_text[: next_heading.start()]
    ai_requirements = (
        "OpenAI ChatGPT",
        "GPT-5.5 Thinking",
        "GPT-5.6 Thinking",
        "reviewed",
        "validated",
        "design",
    )
    missing_ai_terms = [term for term in ai_requirements if term.lower() not in ai_text.lower()]
    if missing_ai_terms:
        errors.append(
            "AI disclosure is missing required specificity: " + ", ".join(missing_ai_terms)
        )
    else:
        passed.append("AI disclosure identifies tools, scope, and human verification")

    project = _extract_project_table(root / "pyproject.toml")
    cff_text = _read_text(root / "CITATION.cff")
    cff_version = _extract_cff_scalar(cff_text, "version")
    cff_license = _extract_cff_scalar(cff_text, "license")
    cff_title = _extract_cff_scalar(cff_text, "title")

    expected_version = str(project.get("version", ""))
    expected_license = str(project.get("license", ""))
    if cff_version != expected_version:
        errors.append(
            f"Version mismatch: pyproject.toml={expected_version!r}, CITATION.cff={cff_version!r}"
        )
    else:
        passed.append(f"Project and citation versions agree ({expected_version})")

    if cff_license != expected_license:
        errors.append(
            f"License mismatch: pyproject.toml={expected_license!r}, CITATION.cff={cff_license!r}"
        )
    else:
        passed.append(f"Project and citation licenses agree ({expected_license})")

    if cff_title != "frsutils: Fuzzy-Rough Set Utilities for Python":
        errors.append("CITATION.cff has an unexpected software title")
    else:
        passed.append("Citation title matches the released software identity")

    orcid_matches = ORCID_PATTERN.findall(cff_text + "\n" + paper_text)
    if orcid_matches:
        passed.append(f"Author metadata contains {len(set(orcid_matches))} ORCID identifier(s)")
    else:
        warnings.append("No verified author ORCID is recorded; ORCID is optional but recommended")

    active_cff_lines = "\n".join(
        line for line in cff_text.splitlines() if not line.lstrip().startswith("#")
    )
    doi_matches = sorted(set(DOI_PATTERN.findall(active_cff_lines)))
    archive_dois = [doi for doi in doi_matches if "zenodo" in doi.lower()]
    if archive_dois:
        passed.append(f"Software archive DOI is recorded: {archive_dois[0]}")
    elif require_archive_doi:
        errors.append("A version-specific software archive DOI is required but absent")
    else:
        warnings.append("Software archive DOI is pending the external Zenodo release step")

    placeholder_patterns = ("10.xxxx", "10.0000/zenodo", "10.5281/zenodo.x")
    active_cff_lower = active_cff_lines.lower()
    if any(pattern in active_cff_lower for pattern in placeholder_patterns):
        errors.append("CITATION.cff contains an active placeholder DOI")
    else:
        passed.append("Citation metadata contains no active placeholder DOI")

    link_errors = _validate_relative_links(root)
    if link_errors:
        errors.extend(link_errors)
    else:
        passed.append("Repository Markdown links resolve locally")

    return ValidationReport(tuple(errors), tuple(warnings), tuple(passed))


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for repository validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--require-archive-doi",
        action="store_true",
        help="Fail when a software archive DOI has not yet been recorded.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable validation report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run JOSS-readiness validation and return a process exit code."""

    args = _build_parser().parse_args(argv)
    report = validate_repository(
        args.root,
        require_archive_doi=args.require_archive_doi,
    )

    for item in report.passed:
        print(f"PASS: {item}")
    for item in report.warnings:
        print(f"WARN: {item}")
    for item in report.errors:
        print(f"ERROR: {item}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(asdict(report), indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if report.is_valid else 2


if __name__ == "__main__":
    sys.exit(main())
