# Experiments

Each reported experiment will have an immutable manifest containing its
hypothesis, configuration reference, code revision, data-split identifier,
random seed, environment, hardware assumptions, outputs, and result status.

Exploratory and confirmatory experiments must be labeled separately. Generated
logs and large artifacts remain local; compact manifests and auditable summaries
may be versioned.

## MLflow

Local runs are tracked through the Dockerized MLflow server. Experiment code
should use `singalign.tracking.tracked_run` and provide a `RunMetadata` object
containing the experiment name, run name, research classification, dataset
version, split fingerprint, and seed.

The tracking wrapper automatically records the Git revision, dirty state,
Python and platform versions, elapsed time, and standardized dataset tags.
Nested configuration mappings should be passed as run parameters; they are
flattened into stable dotted names.

Use `exploratory` for development, debugging, and hypothesis-generating work.
Use `confirmatory` only for experiments whose hypotheses, metrics, exclusions,
and success thresholds were registered before inspecting their results.

## Baseline training

`configs/training/baseline.yaml` defines the compact mel-autoencoder baseline.
Training and validation losses are logged once per epoch. The lowest
validation loss selects `best.pt`; `last.pt` records the final epoch. Both
checkpoints remain in ignored local storage and are attached to the MLflow run.

The trainer accepts only the `train` and `validation` partitions. The test
partition remains sealed until the evaluation pipeline is run against a
selected checkpoint.

## Held-out evaluation

`configs/evaluation/baseline.yaml` defines the evaluation seed, device policy,
bootstrap procedure, and output location. `singalign-evaluate` restores the
validation-selected checkpoint and evaluates only the immutable test IDs.

Reports are written beneath ignored `reports/evaluation/<run-id>/` directories
and attached to MLflow. Every report includes per-example measurements,
aggregate confidence intervals, checkpoint and split fingerprints, latency,
model size, and the training and evaluation configurations.

## Proxy preference alignment

`configs/training/alignment.yaml` defines deterministic synthetic preference
pairs and the DPO-style energy objective. Chosen candidates receive mild
controlled degradation; rejected candidates receive a stronger version of the
same degradation family. A frozen baseline supplies the reference margin and
an MSE anchor constrains reconstruction drift.

Runs log DPO loss, anchor loss, preference accuracy, relative margin, baseline
checkpoint hash, and split fingerprint. The resulting checkpoint is a research
instrument for studying post-training mechanics, not a human-aligned model.
