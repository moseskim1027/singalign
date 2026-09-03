# Data

Datasets are stored locally and are never committed to this repository.

## Planned local layout

```text
data/
├── raw/         Immutable source files
├── interim/     Validated intermediate representations
├── processed/   Model-ready examples
└── manifests/   Versioned provenance and split metadata
```

Each dataset integration must document its source, version, retrieval date,
license, access conditions, checksums, preprocessing steps, and known
limitations. Evaluation splits must be isolated by singer and song where the
research question requires generalization to both.

Only provenance and split manifests that do not expose restricted data may be
committed.
