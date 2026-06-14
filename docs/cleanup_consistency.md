# Cleanup and consistency pass

This release-cleanup pass does **not** add a new fuzzy-rough
feature or execution path.

## Changes

- Fixed README links from removed `md_files/` paths to current `docs/` paths.
- Removed stale README text claiming VQRS still needs implementation.
- Updated `docs/backend_execution_status.md` after the release-hardening work.
- Rewrote `docs/vqrs_info.md` to remove OWAFRS copy/paste headings and typos.
- Updated old FRSMOTE example imports to use the standalone `frsampling` package.
- Moved KEEL Audit WIP variants to `archive/keel_audit_wip/` outside the
  importable package.
- Added a pytest `slow` marker and excluded slow exhaustive model-combination
  tests from the default pytest run.
- Marked real KEEL dataset integration tests as `slow`; mocked/unit KEEL tests
  remain in the default run.
- Removed generated cache artifacts from the packaged project zip.

## Test policy

Default local validation excludes slow tests:

```bash
python -m pytest tests -q
```

Run slow model-combination coverage explicitly:

```bash
python -m pytest tests/models_tests -m slow -q
```

Run release/backend smoke tests explicitly:

```bash
python -m pytest tests/api tests/benchmarks tests/examples -q
```
