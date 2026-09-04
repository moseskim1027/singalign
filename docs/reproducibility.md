# Reproducibility package

SingAlign runs are intended to be reproducible on Apple Silicon and Linux
through Docker Compose. The repository records configurations, dataset split
fingerprints, experiment manifests, MLflow run identifiers, and the Docker
image build inputs used by the exploratory pilots.

## Verification

From the repository root:

```bash
docker compose build test
docker compose run --rm test
docker compose up --build -d mlflow api ui
curl -fsS http://localhost:8000/openapi.json >/dev/null
curl -fsS http://localhost:4173/ >/dev/null
```

The services are available at `http://localhost:5001` (MLflow),
`http://localhost:4173` (UI), and `http://localhost:8000` (training API).

## Provenance checklist

For each reported exploratory run, retain:

- the exact Git revision and clean/dirty state
- the Docker image build arguments and `uv.lock`
- the experiment configuration and manifest
- the dataset version and split fingerprint
- checkpoint hashes and MLflow run IDs
- generated reports and the command used to create them

Proxy metrics and learned reward models are simulation diagnostics. They do
not establish human preference or generalization beyond the declared data.
