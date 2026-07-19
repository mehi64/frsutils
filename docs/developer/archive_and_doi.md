# Software archiving and DOI metadata

An immutable archive makes a specific `frsutils` release independently citable.
This page describes a general GitHub–Zenodo workflow and is not tied to a
particular package version or publication.

## Metadata source

`CITATION.cff` is the source of software citation metadata. Before creating an
archive, confirm that it contains the correct author, version, release date,
license, repository URL, abstract, and keywords. Add identifiers only after they
have been issued by the corresponding service.

## GitHub–Zenodo workflow

1. Sign in to Zenodo and link the GitHub account that owns the repository.
2. Enable the `mehi64/frsutils` repository in Zenodo's GitHub integration.
3. Publish a clean, annotated Git tag and matching GitHub release.
4. Wait for Zenodo to ingest the release.
5. Verify the Zenodo record:

   - resource type is **Software**;
   - version matches the Git tag;
   - license is BSD-3-Clause;
   - author and repository metadata are correct;
   - the archived source corresponds to the intended release.

6. Add the issued version-specific DOI to `CITATION.cff`:

   ```yaml
   identifiers:
     - type: doi
       value: 10.5281/zenodo.<record>
       description: Archived frsutils <version> software release
   ```

7. Add a DOI badge or citation link to the README when useful.
8. Validate the updated citation file:

   ```bash
   cffconvert --validate
   ```

9. Commit the DOI metadata update without moving or recreating the immutable
   release tag.

Use the version DOI when citing a specific executable release. The Zenodo
concept DOI may be used when a citation should resolve to the software project
across releases.

## Manual-upload alternative

Use a manual deposit when GitHub integration is unavailable:

```bash
git archive \
  --format=zip \
  --prefix=frsutils-<version>/ \
  --output=frsutils-<version>-source.zip \
  v<version>
sha256sum frsutils-<version>-source.zip
```

Upload the archive as **Software**, copy metadata from `CITATION.cff`, select the
BSD-3-Clause license, and verify the generated checksum before publishing the
record.

## Citation maintenance

Keep the software archive DOI as the identifier of the executable artifact. A
separate article, dataset, or methods publication may be recorded as a preferred
citation only after its final bibliographic metadata and DOI are available.
