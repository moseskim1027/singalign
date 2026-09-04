#!/usr/bin/env bash
set -euo pipefail

docker compose build test
docker compose run --rm test
docker compose up --build -d mlflow api ui
curl --fail --silent http://localhost:8000/openapi.json >/dev/null
curl --fail --silent http://localhost:4173/ >/dev/null
echo "SingAlign Docker smoke verification passed"
