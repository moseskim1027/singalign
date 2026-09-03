# Comparison UI

This plain TypeScript interface reads an existing paired comparison report. It
does not modify reports or communicate with MLflow.

## Run with Docker

From the repository root, generate a comparison report and start the UI:

```bash
docker compose up --build -d ui
```

Open <http://localhost:4173> and enter the comparison run ID. Docker mounts
`reports/comparisons/` into Nginx read-only.

## Develop locally

```bash
cd ui
npm ci
npm run check
npm run dev
```

The Vite development server does not mount the repository's report directory.
For report-backed inspection, use Docker Compose.

## Interpretation

The interface is an unblinded debugging and inspection tool. Baseline and
aligned examples are explicitly labeled, and their audio uses approximate mel
pseudoinversion with Griffin-Lim. Use a separately designed and preregistered
listening study for perceptual claims.
