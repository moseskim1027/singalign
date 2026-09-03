# Reports

This directory will contain reproducible research reports and the source data
for published tables and figures. Generated assets should identify the
experiment manifests and code revision from which they were produced.

Paired comparison runs are written to `comparisons/<run-id>/`. The local UI
serves this directory through a read-only Docker mount and loads each run's
`manifest.json` and `summary.json`. Generated comparison reports remain ignored
by Git and should not be treated as source artifacts.
