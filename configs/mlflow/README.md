# MLflow Configuration

The local Docker workflow runs MLflow with a SQLite metadata backend and a
proxied filesystem artifact store. Both are persisted in the ignored local
directory `.mlflow/` so experiment state does not consume Docker's virtual-disk
allocation.

The tracking server binds to `127.0.0.1:5000` on the host and is not exposed to
the local network. Research containers connect through
`MLFLOW_TRACKING_URI=http://mlflow:5000`.

Experiment code should use `singalign.tracking` so every run records consistent
dataset, split, revision, seed, platform, and research-status metadata.

Stopping containers with `docker compose down` preserves tracked runs. Removing
`.mlflow/` permanently deletes the local database and artifacts and should be
done only through an intentional cleanup operation.
