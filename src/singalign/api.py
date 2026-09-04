"""Docker-backed training API for the local experiment sandbox."""

from __future__ import annotations

import os
from pathlib import Path

import docker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="SingAlign training API")

EXPERIMENTS = {
    "baseline": "singalign-train",
    "aligned": "singalign-align",
    "conditioned": "singalign-conditioned-train",
    "vocoder": "singalign-vocoder-train",
    "kto": "singalign-kto-train",
}
DEFAULTS = {"epochs": 10, "segment_seconds": 3.0, "learning_rate": 0.0001}


class TrainingRequest(BaseModel):
    experiment: str
    parameters: dict[str, float | int] = Field(default_factory=dict)


class TrainingResponse(BaseModel):
    job_id: str
    command: list[str]


@app.post("/training", response_model=TrainingResponse)
def start_training(request: TrainingRequest) -> TrainingResponse:
    """Validate and launch one detached research container."""
    if request.experiment not in EXPERIMENTS:
        raise HTTPException(status_code=400, detail="unsupported experiment")
    parameters = {**DEFAULTS, **request.parameters}
    if not 1 <= int(parameters["epochs"]) <= 100 or not 0 < float(parameters["segment_seconds"]) <= 30:
        raise HTTPException(status_code=400, detail="epochs or segment_seconds out of range")
    config = f"configs/training/{'alignment' if request.experiment == 'aligned' else request.experiment}.yaml"
    command = [EXPERIMENTS[request.experiment], "--config", config]
    if request.experiment in {"aligned", "kto"}:
        command.extend(["--checkpoint", "checkpoints/baseline/best.pt"])
    command.extend(["--index", "data/interim/pjs/index.jsonl", "--splits", "data/interim/pjs/splits.json"])
    for name, value in request.parameters.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    client = docker.from_env()
    root = Path(os.environ.get("SINGALIGN_HOST_ROOT", Path.cwd())).resolve()
    try:
        container = client.containers.run(
            "singalign-research", command=command, detach=True, remove=True,
            environment={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
            network=os.environ.get("SINGALIGN_DOCKER_NETWORK", "singalign_default"),
            volumes={str(root / "data"): {"bind": "/workspace/data", "mode": "ro"}, str(root / "checkpoints"): {"bind": "/workspace/checkpoints", "mode": "rw"}, str(root / "reports"): {"bind": "/workspace/reports", "mode": "rw"}},
        )
    except docker.errors.DockerException as error:
        raise HTTPException(status_code=503, detail="unable to launch research container") from error
    return TrainingResponse(job_id=container.id, command=command)


@app.get("/training/{job_id}")
def training_status(job_id: str) -> dict[str, str]:
    """Return the current Docker container status."""
    try:
        status = docker.from_env().containers.get(job_id).status
    except docker.errors.NotFound:
        return {"job_id": job_id, "status": "completed-or-unknown"}
    return {"job_id": job_id, "status": status}
