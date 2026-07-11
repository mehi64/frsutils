# Archive the software release and record its DOI

This guide covers the account-level steps that cannot be completed by repository
code. Use a version-specific software DOI for the exact release reviewed by
JOSS. The JOSS article receives a separate DOI only after acceptance.

## Metadata source

`CITATION.cff` is the single source of software citation metadata for the first
archive. The repository intentionally does not include `.zenodo.json`: Zenodo
ignores `CITATION.cff` when both files exist, and FRsutils currently does not
need Zenodo-only funding or community fields.

Before archiving, confirm that `CITATION.cff` contains the correct author,
version, release date, license, repository, abstract, and keywords. Add an ORCID
only after the author has verified the identifier.

## Recommended GitHub–Zenodo workflow

1. Sign in to Zenodo and link the GitHub account that owns
   `mehi64/frsutils`.
2. In Zenodo, open **GitHub**, synchronize repositories, and enable
   `mehi64/frsutils`.
3. Push the release commit and wait for all CI, documentation, and JOSS-paper
   workflows to pass.
4. Create the annotated Git tag:

   ```bash
   git switch main
   git pull --ff-only
   git status --short
   git tag -a v0.1.0 -m "frsutils 0.1.0"
   git push origin v0.1.0
   ```

5. On GitHub, create release `v0.1.0`. Paste the contents of
   `RELEASE_NOTES_v0.1.0.md` into the release description.
6. Wait for Zenodo to ingest the release. Open the resulting software record
   and confirm:

   - title and author;
   - software resource type;
   - version `0.1.0`;
   - BSD-3-Clause license;
   - repository URL;
   - uploaded source archive;
   - version-specific DOI;
   - successful Software Heritage archival status when it becomes available.

7. Record the **version DOI** in `CITATION.cff`:

   ```yaml
   identifiers:
     - type: doi
       value: 10.5281/zenodo.XXXXXXX
       description: Archived frsutils 0.1.0 software release
   ```

8. Add the DOI badge and citation text to the README, then validate:

   ```bash
   cffconvert --validate
   python scripts/validate_joss_submission.py --require-archive-doi
   ```

9. Commit the DOI metadata update. Do not move or recreate the `v0.1.0` tag.
   The DOI identifies the already archived immutable release; the metadata
   update belongs to the next repository state.

## Manual Zenodo upload alternative

Use a manual deposit only when GitHub integration cannot be enabled. Create a
clean source archive from the immutable release tag:

```bash
git archive \
  --format=zip \
  --prefix=frsutils-0.1.0/ \
  --output=frsutils-0.1.0-source.zip \
  v0.1.0
sha256sum frsutils-0.1.0-source.zip
```

Upload the archive as **Software**, copy the metadata from `CITATION.cff`,
select BSD-3-Clause, and publish the record. Zenodo also allows a DOI to be
reserved before publication when the identifier must be embedded in files
inside the uploaded archive.

## JOSS timing

A software archive may be created before submission. JOSS also explicitly asks
for an archive DOI at acceptance, after review changes are complete. If the
review changes software behavior or public APIs, create a new release and give
JOSS the DOI of that final reviewed version instead of the earlier candidate.

## Article DOI after acceptance

After JOSS publishes the article, add a `preferred-citation` block to
`CITATION.cff` using the JOSS article DOI. Keep the software archive DOI as an
identifier for the executable research artifact; the two DOIs serve different
purposes.
