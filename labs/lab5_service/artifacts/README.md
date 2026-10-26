# Runtime artifacts

This directory is populated locally and is intentionally excluded from Git.

- `acoustic/` is created by `prepare_lab4_artifacts.py`;
- `vocoder/` is created by `download_vocoder.py`.

The scripts record source paths and SHA-256 hashes in JSON manifests so that the
runtime can be reproduced without committing another copy of the model weights.

