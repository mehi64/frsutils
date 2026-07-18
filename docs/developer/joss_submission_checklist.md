# Final JOSS submission checklist

This checklist separates repository checks that can be automated from external
account and platform checks that must be completed after the release commit is
pushed.

## Automated repository gate

Run from the repository root:

```bash
python scripts/validate_joss_submission.py
python -m pytest tests/repository_tests/test_joss_submission_readiness.py -q
```

Before final acceptance, require the archive DOI as well:

```bash
python scripts/validate_joss_submission.py --require-archive-doi
```

Current repository status:

- [x] Required JOSS paper sections are present.
- [x] The paper is within the 750–1750 word range.
- [x] Statement of need identifies the problem and target users.
- [x] State of the field compares `frsutils` with `RoughSets` and
  
      `fuzzy-rough-learn` and gives a build-versus-contribute justification.
- [x] Software design explains API, execution, memory, and backend trade-offs.
- [x] Research impact is supported by reproducible real-data and benchmark
  
      artifacts rather than aspirational language.
- [x] AI disclosure identifies the tools/models, scope of assistance, human
  
      verification, and author responsibility.
- [x] All paper citation keys resolve in `paper.bib`.
- [x] `CITATION.cff`, `pyproject.toml`, and source headers use BSD-3-Clause.
- [x] Version `0.1.0` is consistent in package and citation metadata.
- [x] No active placeholder DOI is present.
- [x] The lowercase `frsutils/` package is canonical.
- [x] A GitHub Action validates metadata and builds the draft JOSS PDF.

## Scientific and software validation

- [ ] Regenerate the reference study from a clean release checkout.
- [ ] Confirm `git_worktree_dirty` is `false` in the regenerated environment
  
      metadata.
- [ ] Run the default test suite on all supported Python versions through CI.
- [ ] Run the exhaustive slow suite and retain the local JUnit report outside
  
      the repository.
- [ ] Build wheel and source distribution and run `twine check`.
- [ ] Install the built wheel in an isolated environment and run an import and
  
      computation smoke test.
- [ ] Build documentation with `mkdocs build --strict`.
- [ ] Download and inspect the JOSS paper PDF artifact produced by GitHub
  
      Actions.

## Public repository checks

- [ ] The repository is publicly cloneable without registration.
- [ ] Source files are browsable without registration.
- [ ] GitHub Issues are enabled and outside users can create issues.
- [ ] Pull requests can be proposed by outside users.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, and the issue
  
      templates are visible from the default branch.
- [ ] Public history exceeds six months and shows iterative development.
- [ ] At least one tagged release and the changelog are visible.
- [ ] CI, documentation, and paper workflows are green on the release commit.
- [ ] A colleague or independent user has installed and exercised the release;
  
      record the resulting issue, discussion, or acknowledgement when possible.

## Author and paper metadata

- [x] Author name and independent-researcher affiliation are present.
- [ ] Add an ORCID only after the author confirms the exact identifier.
- [ ] Confirm whether any funding acknowledgement must be added.
- [ ] Confirm the final paper date matches the submitted revision.
- [ ] Confirm all software authors who meet the authorship criteria are listed.

## Release and archive

- [ ] Commit and push all release-preparation and reference-data migration changes.
- [ ] Confirm `main` is clean and all workflows pass.
- [ ] Create annotated tag `v0.1.0`.
- [ ] Create the GitHub release using `RELEASE_NOTES_v0.1.0.md`.
- [ ] Enable `mehi64/frsutils` in the linked Zenodo GitHub integration.
- [ ] Confirm that Zenodo archived the release as Software with BSD-3-Clause.
- [ ] Copy the version-specific Zenodo DOI into `CITATION.cff` and README.
- [ ] Run the validator with `--require-archive-doi`.
- [ ] If review changes the release, archive a new final version and provide
  
      that DOI to JOSS.

## Submission

- [ ] Submit the repository URL, release version, paper path, and author details
  
      through the JOSS submission form.
- [ ] Confirm the submission does not present new scientific results; the study
  
      is framed as software validation and reproducibility evidence.
- [ ] Participate personally in the public review issue; generative AI must not
  
      be used for author–editor or author–reviewer conversation except for
      translation.
- [ ] At acceptance, provide the final software archive DOI and release version
  
      requested by the editor.
- [ ] After publication, add the JOSS article DOI as the preferred citation in
  
      `CITATION.cff`.

## Current unresolved external items

The repository preparation is complete without inventing identifiers. The two
external items that remain are:

1. **ORCID:** no identifier has been added because an exact author-owned ORCID
   has not been verified.
2. **Archive DOI:** a DOI can only be minted through the author's Zenodo account
   after a release is available. Follow
   [Archive the software release and record its DOI](archive_and_doi.md).
