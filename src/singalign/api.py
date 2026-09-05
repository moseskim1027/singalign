"""Docker-backed training API for the local experiment sandbox."""

from __future__ import annotations

import os
import re
from pathlib import Path

import docker
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SingAlign training API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)

EXPERIMENTS = {
    "baseline": "singalign-train",
    "aligned": "singalign-align",
    "conditioned": "singalign-conditioned-train",
    "vocoder": "singalign-vocoder-train",
    "kto": "singalign-kto-train",
    "study2": "singalign-transfer",
}
DEFAULTS = {"epochs": 10, "segment_seconds": 3.0, "learning_rate": 0.0001}


class TrainingRequest(BaseModel):
    experiment: str
    parameters: dict[str, float | int] = Field(default_factory=dict)
    source: str | None = None
    target: str | None = None
    output: str | None = None
    aligned_output: str | None = None


class TrainingResponse(BaseModel):
    job_id: str
    command: list[str]


class EvaluationRequest(BaseModel):
    reference: str
    transferred: str
    output: str
    parent_run_id: str | None = None


class RenderRequest(BaseModel):
    score: str
    output: str
    bpm: float = 120.0


def launch_container(command: list[str]) -> TrainingResponse:
    """Launch a detached research command with the repository mounts."""
    client = docker.from_env()
    root = Path(os.environ.get("SINGALIGN_HOST_ROOT", Path.cwd())).resolve()
    try:
        container = client.containers.run(
            "singalign-research",
            command=command,
            detach=True,
            remove=False,
            environment={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
            network=os.environ.get("SINGALIGN_DOCKER_NETWORK", "singalign_default"),
            volumes={
                str(root / "data"): {"bind": "/workspace/data", "mode": "ro"},
                str(root / "checkpoints"): {
                    "bind": "/workspace/checkpoints",
                    "mode": "rw",
                },
                str(root / "reports"): {"bind": "/workspace/reports", "mode": "rw"},
            },
        )
    except docker.errors.DockerException as error:
        raise HTTPException(
            status_code=503, detail="unable to launch research container"
        ) from error
    return TrainingResponse(job_id=container.id, command=command)


@app.post("/training", response_model=TrainingResponse)
def start_training(request: TrainingRequest) -> TrainingResponse:
    """Validate and launch one detached research container."""
    if request.experiment not in EXPERIMENTS:
        raise HTTPException(status_code=400, detail="unsupported experiment")
    if request.experiment == "study2":
        if not request.source or not request.target or not request.output:
            raise HTTPException(
                status_code=400, detail="Study 2 requires source, target, and output"
            )
        command = [
            EXPERIMENTS[request.experiment],
            "--source",
            request.source,
            "--target",
            request.target,
            "--output",
            request.output,
            "--source-id",
            "api-source",
            "--target-id",
            "api-target",
        ]
        if request.aligned_output:
            command.extend(["--aligned-output", request.aligned_output])
    else:
        parameters = {**DEFAULTS, **request.parameters}
        if (
            not 1 <= int(parameters["epochs"]) <= 100
            or not 0 < float(parameters["segment_seconds"]) <= 30
        ):
            raise HTTPException(
                status_code=400, detail="epochs or segment_seconds out of range"
            )
        config_name = (
            "alignment" if request.experiment == "aligned" else request.experiment
        )
        config = f"configs/training/{config_name}.yaml"
        command = [EXPERIMENTS[request.experiment], "--config", config]
        if request.experiment in {"aligned", "kto"}:
            command.extend(["--checkpoint", "checkpoints/baseline/best.pt"])
        command.extend(
            [
                "--index",
                "data/interim/pjs/index.jsonl",
                "--splits",
                "data/interim/pjs/splits.json",
            ]
        )
        for name, value in request.parameters.items():
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    return launch_container(command)


@app.post("/study2/evaluation", response_model=TrainingResponse)
def start_study2_evaluation(request: EvaluationRequest) -> TrainingResponse:
    """Launch the objective Study 2 evaluation for one transfer output."""
    command = [
        "singalign-transfer-evaluate",
        "--reference",
        request.reference,
        "--transferred",
        request.transferred,
        "--output",
        request.output,
    ]
    if request.parent_run_id:
        command.extend(["--parent-run-id", request.parent_run_id])
    return launch_container(command)


@app.post("/study2/render", response_model=TrainingResponse)
def start_study2_render(request: RenderRequest) -> TrainingResponse:
    """Render a fixed MusicXML target instrumental for Study 2."""
    if request.bpm <= 0:
        raise HTTPException(status_code=400, detail="bpm must be positive")
    return launch_container([
        "singalign-render-instrumental", "--score", request.score,
        "--output", request.output, "--bpm", str(request.bpm),
    ])


@app.get("/training/{job_id}")
def training_status(job_id: str) -> dict[str, str]:
    """Return the current Docker container status."""
    try:
        container = docker.from_env().containers.get(job_id)
        status = container.status
        logs = container.logs(tail=20).decode("utf-8", errors="replace")
        match = re.findall(r'"mlflow_run_id":\s*"([^"]+)"', logs)
        state = container.attrs.get("State", {})
        if status == "exited":
            status = "completed" if state.get("ExitCode") == 0 else "failed"
        result = {"job_id": job_id, "status": status}
        if match:
            result["mlflow_run_id"] = match[-1]
        if status == "failed":
            errors = [line.strip() for line in logs.splitlines() if line.strip()]
            if errors:
                result["error"] = errors[-1]
        return result
    except docker.errors.NotFound:
        return {"job_id": job_id, "status": "completed-or-unknown"}
    return {"job_id": job_id, "status": status}
