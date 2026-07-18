# JOSS submission checklist

Use this concise gate after completing the full
[release procedure](release.md). Items involving GitHub, Zenodo, or the JOSS
submission form must be completed by the repository owner.

## Repository and software

- [ ] Version `0.1.1` is consistent across package, citation, README, changelog,
      release notes, and documentation metadata.
- [ ] The repository worktree is clean.
- [ ] The default and exhaustive slow test suites pass.
- [ ] Public API and core branch coverage remain at or above 95 percent.
- [ ] Wheel and source distribution build, pass `twine check`, install in clean
      environments, and execute real dense/blockwise computations.
- [ ] `mkdocs build --strict` passes.
- [ ] CI, documentation, extended-test, and paper workflows are green on the
      release commit.
- [ ] No cache directories, generated `site/`, local logs, `.vscode/`, virtual
      environments, benchmark outputs, or test reports are tracked.

## Reference study

- [ ] The study is regenerated after the release-preparation commit.
- [ ] `environment.json` records `frsutils_version: 0.1.1`.
- [ ] `git_commit` identifies the release commit and `git_worktree_dirty` is
      `false`.
- [ ] All dense/blockwise equivalence rows pass.
- [ ] The benchmark JSON contains no failed cases.
- [ ] The manifest checksums match the committed output files.

## Paper and metadata

- [ ] `python scripts/validate_joss_submission.py` passes.
- [ ] The paper PDF produced by GitHub Actions renders correctly.
- [ ] Author, corresponding-author, affiliation, date, funding, and AI-use
      disclosures are accurate.
- [ ] Every citation key resolves in `paper.bib`.
- [ ] Research-impact wording is limited to evidence provided by the public
      study and the documented ongoing FRSMOTE research use.
- [ ] An ORCID is included only when the exact author-owned identifier has been
      verified.

## Public project infrastructure

- [ ] The repository is publicly cloneable and source files are browsable
      without registration.
- [ ] GitHub Issues are enabled and an outside user can open an issue after
      signing in.
- [ ] Pull requests can be submitted by outside contributors.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, and issue templates
      are visible on the default branch.
- [ ] Public development history and tagged releases are visible.

## Release, archive, and submission

- [ ] Annotated tag `v0.1.1` and GitHub release are published.
- [ ] The GitHub release is archived as Software under BSD-3-Clause.
- [ ] The version-specific archive DOI is copied into `CITATION.cff` and README.
- [ ] `python scripts/validate_joss_submission.py --require-archive-doi` passes.
- [ ] The repository URL, release version, paper path, and author details are
      submitted through the JOSS form.
- [ ] The author participates personally in the public review discussion.
- [ ] After acceptance, the JOSS DOI is added as the preferred citation.

Do not invent an ORCID, archive DOI, JOSS submission URL, or article DOI. These
identifiers can only be added after the corresponding external action occurs.
