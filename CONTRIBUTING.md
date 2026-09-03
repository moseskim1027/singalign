# Contributing to SingAlign

SingAlign is organized as a reproducible research project. Contributions should
make the research claim, evidence, and limitations easy to audit.

## Before opening a change

- State the hypothesis or repository need addressed by the change.
- Identify relevant baselines, controls, and evaluation measures.
- Document new data sources, licenses, and preprocessing assumptions.
- Keep exploratory results distinct from results used to support a claim.
- Update the research plan or evaluation protocol when methodology changes.

## Reproducibility

Reported experiments should include a versioned configuration, random seed,
environment information, data-split identifier, and an experiment manifest.
Do not overwrite a configuration associated with a reported result.

## Commit messages

Use Conventional Commit prefixes, such as `feat:`, `fix:`, `docs:`, `test:`,
`refactor:`, or `chore:`. Keep each commit focused on one coherent change.

## Data and artifacts

Do not commit datasets, checkpoints, generated audio, participant information,
or credentials. Record retrieval and provenance information in documentation
without redistributing restricted material.
