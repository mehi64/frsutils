## Summary

Describe the problem and the implemented change.

## Scientific or API impact

Explain any mathematical convention, public API change, backend behavior, or
claim boundary affected by this pull request. Write "None" when not applicable.

## Validation

List the exact commands run and summarize their results.

```text
python -m pytest tests -ra
mkdocs build --strict
```

## Checklist

- [ ] The change is focused and contains no unrelated refactoring.
- [ ] Tests cover the new behavior or regression.
- [ ] Public docstrings and documentation are updated where needed.
- [ ] Dense/blockwise and backend contracts were considered where relevant.
- [ ] User-visible changes are recorded in `CHANGELOG.md`.
- [ ] Modified Python files follow the SPDX header and docstring guidelines.
- [ ] I reviewed and executed any materially AI-assisted code or text and
      disclosed that assistance when relevant.
