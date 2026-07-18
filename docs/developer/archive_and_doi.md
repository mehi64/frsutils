# Archive the software release and record its DOI

Use a version-specific software DOI for the exact `frsutils` release reviewed by
JOSS. The JOSS article receives a separate DOI after acceptance.

## Metadata source

`CITATION.cff` is the source of software citation metadata. Before archiving,
confirm the author, version `0.1.1`, release date, license, repository, abstract,
and keywords. Add an ORCID only after the exact identifier has been verified.

## GitHub–Zenodo workflow

1. Sign in to Zenodo and link the GitHub account that owns
   `mehi64/frsutils`.
2. In Zenodo's GitHub integration, synchronize repositories and enable
   `mehi64/frsutils`.
3. Push the clean release commit and wait for CI, documentation, extended-test,
   and paper workflows to pass.
4. Create and push the annotated tag:

   ```bash
   git switch main
   git pull --ff-only
   git status --short
   git tag -a v0.1.1 -m "frsutils 0.1.1"
   git push origin main
   git push origin v0.1.1
   ```

5. Create GitHub release `v0.1.1` and use
   `RELEASE_NOTES_v0.1.1.md` as its description.
6. Wait for Zenodo to ingest the release and confirm:

   - software resource type;
   - version `0.1.1`;
   - BSD-3-Clause license;
   - author and repository URL;
   - source archive;
   - version-specific DOI.

7. Add the version DOI to `CITATION.cff`:

   ```yaml
   identifiers:
     - type: doi
       value: 10.5281/zenodo.XXXXXXX
       description: Archived frsutils 0.1.1 software release
   ```

8. Add the DOI badge and citation text to README, then validate:

   ```bash
   cffconvert --validate
   python scripts/validate_joss_submission.py --require-archive-doi
   ```

9. Commit the DOI metadata update. Do not move or recreate the immutable
   `v0.1.1` tag.

## Manual-upload alternative

Use a manual deposit only if GitHub integration cannot be enabled:

```bash
git archive \
  --format=zip \
  --prefix=frsutils-0.1.1/ \
  --output=frsutils-0.1.1-source.zip \
  v0.1.1
sha256sum frsutils-0.1.1-source.zip
```

Upload the archive as **Software**, copy metadata from `CITATION.cff`, select
BSD-3-Clause, and publish the record. Zenodo can reserve a DOI before publication
when the identifier must be embedded in the uploaded files.

## JOSS timing

A software archive may be created before submission. If review changes software
behavior or public APIs, create a new final release and provide JOSS with the DOI
of that reviewed version.

After JOSS publishes the article, add a `preferred-citation` block to
`CITATION.cff` with the article DOI. Keep the software archive DOI as the
identifier of the executable artifact.
