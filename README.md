# SingAlign

Reproducible music-generation sandbox for preference-aligned singing voice
synthesis.

> [!IMPORTANT]
> SingAlign is an early-stage research project. The methods, experiments, and
> conclusions described here are provisional and should not be treated as
> production-ready.

## Overview

SingAlign focuses on two PJS-supported research directions: same-singer,
score-conditioned singing synthesis, and preservation of vocal content and
melody when a source vocal is placed over a different instrumental track. It
is an engineering and simulation environment rather than a confirmatory
human-subjects study.

Reward modeling, candidate reranking, supervised fine-tuning, DPO, and KTO are
retained as exploratory alignment infrastructure. They are not currently the
primary research claim because PJS is small and contains one vocalist.

This repository is structured as a reproducible research artifact. It will
contain experiment definitions, evaluation protocols, statistical analyses,
and research documentation alongside the eventual implementation.

## Research questions

SingAlign is organized around two primary questions:

1. Can a compact model synthesize the PJS vocalist from lyrics, phonemes,
   musical score, timing, and pitch conditioning?
2. Can source vocal content, phoneme timing, and melody be preserved when the
   vocal is transferred onto a different instrumental track?

Reward-model and preference-optimization experiments remain optional
engineering diagnostics. Human-preference prediction and unseen-singer
generalization are deferred until a suitable multi-singer dataset and
evaluation design exist.

## Scope

The research scope has two focused directions. First, test same-singer,
score-conditioned synthesis: lyrics, phonemes, MIDI/MusicXML note timing, and
pitch are used to reconstruct the PJS vocalist. Second, test content-and-
melody transfer: preserve a source vocal's words, phoneme timing, and melody
while placing it over a different instrumental track.

The core transfer experiment does not depend on MusicGen. Use a fixed,
MIDI-rendered PJS instrumental so vocal alignment can be evaluated without
introducing a second generation variable. MusicGen is only an optional,
separately logged target-instrument extension.

The initial studies use the PJS corpus. Experiments will operate on short
mel-spectrogram segments so that data preparation, baseline development, and
pilot studies remain practical on Apple Silicon. They will address:

- vocal naturalness
- pitch and rhythm accuracy
- lyric intelligibility
- expressiveness
- audio fidelity

Singer similarity is not a generalization target because PJS contains one
vocalist; it is only a same-singer reconstruction diagnostic.

The first version will intentionally favor controlled, interpretable
experiments over model scale.

## Planned methodology

The primary research pipeline consists of:

1. Validate PJS phonemes, scores, lyrics, and deterministic song-disjoint
   splits.
2. Extract observed F0 and construct versioned conditioning records.
3. Train and evaluate a same-singer score-conditioned synthesis baseline.
4. Render reproducible target instrumentals from PJS MIDI/MusicXML.
5. Implement original-vocal content-and-melody remix controls, including
   tempo/key alignment and intentionally misaligned controls.
6. Add synthesized-vocal transfer using the Study 1 checkpoint.
7. Report lyric, pitch, timing, audio-quality, and mix diagnostics with full
   provenance.

Reward modeling, reranking, DPO, and KTO are optional exploratory extensions;
they should not be used to imply human preference alignment without suitable
data and evaluation.

The methodology may change as preliminary experiments reveal limitations. Any
material changes will be documented in the research plan and experiment logs.
The complete implementation sequence is: extract and validate observed F0;
complete same-singer synthesis; render target instrumentals from PJS MIDI;
create source/target pair manifests; implement tempo/key alignment and remix
controls; add synthesized-vocal transfer; then report lyric, pitch, timing,
audio-quality, and mix diagnostics.

### Score and lyric conditioning prototype

The repository now includes a dependency-light conditioning interface in
`singalign.conditioning`. It parses PJS MusicXML into deterministic note events
and phoneme label files into timed phoneme intervals. This is the first data
interface for the planned score/lyric-conditioned synthesis stage; it is not
yet a trained synthesizer or a candidate generator. The parser is covered by
unit tests and does not alter the immutable corpus.

Each conditioning record contains note events as `(onset, duration, MIDI
pitch)` tuples, with `MIDI pitch = null` for rests, plus phoneme intervals as
`(start, end, symbol)` tuples in the source label timebase. This schema is
deliberately model-independent so later candidate-generation experiments can
compare conditioning encoders without changing corpus parsing.
The next alignment layer expands these events to acoustic frames using explicit
frame rate, duration, and tempo inputs; no timing is inferred implicitly.
Windowed crops pass an explicit song-time offset so score and phoneme events are
aligned to the same crop rather than implicitly restarting at time zero.
The experimental `ScoreConditionedMelModel` consumes those frame-level MIDI
pitch and phoneme IDs and predicts mel frames. It is an architectural baseline
only; its first exploratory training run is not a synthesis or confirmatory
evaluation.
Its proposed training specification is frozen in
`configs/training/conditioned.yaml`: 16 kHz audio, 80-bin log-mel targets,
100-frame-per-second conditioning, a 3-second window, and a 10-epoch
exploratory budget. The training command is implemented and tested in Docker;
held-out synthesis evaluation remains intentionally deferred until a
decoder/candidate-generation protocol is specified.
The frame adapter emits integer MIDI pitch IDs with `0` for rests and integer
phoneme IDs with `0` reserved for unknown/padding symbols.
The exploratory conditioned-model trainer is available in Docker:

```bash
docker compose run --rm research \
  singalign-conditioned-train \
  --config configs/training/conditioned.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

It logs training/validation loss and a checkpoint to MLflow. This is an architectural
baseline, not yet a candidate-generation or confirmatory experiment.
The run also records the immutable split fingerprint through the shared MLflow
tracking contract.
The `PJSConditionedDataset` adapter pairs these tensors with deterministic
3-second mel targets using the same crop offset, tempo, and frame-count
convention. The first Docker run completed 10 exploratory epochs and logged
MLflow run `1fd53daa1f7e494abe16ceccf7daa3c1` in experiment
`singalign-score-conditioned-baseline`; it produced a checkpoint but no
reported synthesis result.
Exported records also include deterministic pitch metadata: note/rest counts and
the minimum, maximum, and mean voiced MIDI pitch. Score pitch is an intended
conditioning signal; observed performance F0 remains a training target or
diagnostic to avoid leaking the reference performance at inference time.

Export one conditioning record for inspection inside the reproducible Docker
environment:

```bash
docker compose run --rm research \
  singalign-data conditioning \
  --musicxml /workspace/data/raw/pjs/PJS_corpus_ver1.1/pjs001/pjs001.musicxml \
  --labels /workspace/data/raw/pjs/PJS_corpus_ver1.1/pjs001/pjs001.lab \
  --output /workspace/reports/conditioning/pjs001.json
```

## Dataset plan

The initial dataset is the PJS phoneme-balanced Japanese singing voice corpus.
PJS is an approximately 0.26 GB public dataset containing 100 short singing
recordings, their spoken counterparts, MIDI and MusicXML scores, phoneme
labels, and supporting metadata. This compact, paired design supports
score-conditioned modeling and low-resource preference experiments on a local
M1 machine.

Download PJS version 1.1 from the
[official corpus page](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus).
The corpus is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), which requires
attribution and ShareAlike distribution of adapted material.

Before training begins, the project will document:

- the dataset version and retrieval procedure
- applicable access conditions and licensing terms
- allowed uses and redistribution restrictions
- singer- and song-level split construction
- known demographic, linguistic, and recording limitations

Dataset files will not be committed to this repository. The raw corpus must
remain unchanged in the ignored local data directory, and any distributed
adaptations must comply with the corpus license.

See [`data/README.md`](data/README.md) for download, installation, verification,
and provenance instructions.

### Validate and index PJS

SingAlign provides a dependency-light CLI for validating the local corpus,
building a metadata-only index, and creating deterministic song-disjoint
splits. With Python 3.11 and
[`uv`](https://docs.astral.sh/uv/) installed, run:

```bash
uv sync
uv run singalign-data validate \
  --root data/raw/pjs/PJS_corpus_ver1.1
uv run singalign-data index \
  --root data/raw/pjs/PJS_corpus_ver1.1 \
  --output data/interim/pjs/index.jsonl
uv run singalign-data split \
  --index data/interim/pjs/index.jsonl \
  --output data/interim/pjs/splits.json \
  --seed 2026
```

The generated index and split files remain local and are ignored by Git. Run
the test suite without accessing the real corpus:

```bash
uv run python -m unittest discover -v
```

## Reproducible environment and tracking

The Docker workflow provides the same Python 3.11 environment for data
validation, future training, evaluation, and MLflow experiment tracking. It
supports Apple Silicon natively.

Build the research image and start the local tracking server:

```bash
SINGALIGN_GIT_REVISION="$(git rev-parse HEAD)" \
SINGALIGN_GIT_DIRTY="$(test -n "$(git status --porcelain --untracked-files=no)" \
  && echo true || echo false)" \
docker compose build
docker compose up -d mlflow
docker compose ps
```

Passing the revision at build time lets containerized runs retain their exact
source identity without copying repository Git metadata into the image.

MLflow is available at [http://localhost:5001](http://localhost:5001). Its
SQLite database and
artifacts persist in the ignored local directory `.mlflow/`.

Validate the locally mounted PJS corpus and run all tests inside Docker:

```bash
docker compose run --rm research \
  singalign-data validate \
  --root /workspace/data/raw/pjs/PJS_corpus_ver1.1
docker compose run --rm research \
  python -m unittest discover -v
```

Verify experiment tracking end to end with a minimal run:

```bash
docker compose run --rm research \
  singalign-track smoke \
  --experiment singalign-smoke
```

### Train the reconstruction baseline

The first baseline is a compact convolutional autoencoder trained on short
log-mel segments from the PJS singing recordings. It validates the training,
checkpointing, and experiment-tracking pipeline; it is not yet a
score-conditioned singing synthesizer.

The trainer uses the training split for parameter updates and the validation
split for checkpoint selection. It deliberately cannot load the held-out test
split, which is reserved for the separate evaluation workflow.

Run locally on MPS when available, with CPU as the automatic fallback:

```bash
uv run singalign-train \
  --config configs/training/baseline.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Local MPS execution requires an ARM64 build of Python and `uv`. The Docker
workflow below is the portable fallback when host tooling runs through Rosetta
or otherwise resolves incompatible wheels.

Start MLflow and run the same experiment in Docker:

```bash
docker compose up -d mlflow
docker compose run --rm research \
  singalign-train \
  --config configs/training/baseline.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Set the training and UI listening-window duration to any positive value up to
30 seconds with `--segment-seconds N`. The resolved value is logged to MLflow and
embedded in every checkpoint. For example, an M1-friendly three-second run is:

```bash
docker compose run --rm research \
  singalign-train \
  --config configs/training/baseline.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json \
  --segment-seconds 3 \
  --epochs 1 \
  --max-validation-items 2
```

Post-training inherits the resolved duration from this baseline checkpoint.
The comparison command also defaults to the checkpoint duration, so the UI
shows matching training and listening windows. Set
`comparison.audio_segment_seconds` only when intentionally inspecting a
different inference duration.

For a short end-to-end smoke run, add `--epochs 1 --max-train-items 4
--max-validation-items 2`. Checkpoints are written beneath the ignored
`checkpoints/` directory and also attached to the MLflow run.

Build and run the test target, whose dependencies are installed by the pinned
version of `uv` from the committed lockfile:

```bash
docker compose build test
docker compose run --rm lint
docker compose run --rm test
```

The root `Dockerfile` uses shared multi-stage targets. Production dependencies,
including PyTorch and MLflow, are installed once in `runtime-dependencies`;
the `research` and `test` targets inherit that layer. The test target adds only
development tools. Source-only edits therefore reuse the large locked
dependency layer instead of reinstalling it for every service.

### Evaluate the selected baseline

Held-out evaluation is a separate command so that test examples cannot be
loaded by the trainer. Select `best.pt` using validation loss, then evaluate it
once against the immutable test partition:

```bash
uv run singalign-evaluate \
  --config configs/evaluation/baseline.yaml \
  --checkpoint checkpoints/baseline/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Run the same evaluation through the tracked Docker environment:

```bash
docker compose up -d mlflow
docker compose run --rm research \
  singalign-evaluate \
  --config configs/evaluation/baseline.yaml \
  --checkpoint checkpoints/baseline/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Each run writes an ignored report beneath `reports/evaluation/<run-id>/` and
attaches the report to MLflow. The report contains aggregate metrics with
bootstrap confidence intervals, per-example metrics, latency, checkpoint and
split fingerprints, and the exact evaluation configuration.

### Run proxy preference alignment

The first post-training study constructs deterministic synthetic preference
pairs from training and validation log-mel segments. A chosen candidate has a
milder controlled degradation than its rejected counterpart. The trainer uses
a DPO-style energy objective relative to a frozen baseline and a reconstruction
anchor that limits fidelity loss:

```bash
docker compose run --rm research \
  singalign-align \
  --config configs/training/alignment.yaml \
  --checkpoint checkpoints/baseline/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

For a smoke run, add `--epochs 1 --max-train-items 4
--max-validation-items 2`. The best validation checkpoint is written beneath
`checkpoints/aligned/` and attached to MLflow. The test split remains sealed
during post-training.

This is an energy-based DPO proxy for controlled experimentation, not standard
autoregressive DPO and not evidence of alignment with human preferences.

### Compare baseline and aligned checkpoints

Generate paired validation metrics and local listening artifacts without using
the held-out test split:

```bash
docker compose run --rm research \
  singalign-compare \
  --config configs/evaluation/comparison.yaml \
  --baseline-checkpoint checkpoints/baseline/best.pt \
  --aligned-checkpoint checkpoints/aligned/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Reports are written beneath `reports/comparisons/<run-id>/`. They contain
paired deltas, bootstrap confidence intervals, win/tie/loss counts, and a
manifest referencing local reference, baseline, and aligned WAV files.
Generated model audio uses approximate mel pseudoinversion and Griffin-Lim and
must not be treated as a production-quality vocoder result.

The repository also includes `MelVocoder`, a trainable mel-to-waveform decoder
with an explicit frame hop length. It is the first differentiable vocoder
baseline for future generation experiments; it is untrained until a dedicated
vocoder dataset/training protocol is added, so Griffin-Lim remains the current
fallback for existing comparison reports.

Its reproducible exploratory trainer is available in Docker:

```bash
docker compose run --rm research \
  singalign-vocoder-train \
  --config configs/training/vocoder.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

This trains only on the training split, logs validation loss and the checkpoint
to MLflow, and is an engineering baseline rather than a production vocoder.
The first 10-epoch Docker pilot is MLflow run
`421229b14e3043bfb3d89e3d6d2ca209` in `singalign-mel-vocoder`.

Evaluate that checkpoint diagnostically on the sealed test split only after
the pilot is complete:

```bash
docker compose run --rm research \
  singalign-vocoder-evaluate \
  --config configs/training/vocoder.yaml \
  --checkpoint checkpoints/vocoder/last.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

The report prints the split fingerprint, waveform MSE, and generated peak
level. These are engineering diagnostics, not perceptual-quality claims.

The default comparison duration is read from the checkpoints, matching the
training window. An optional positive `comparison.audio_segment_seconds` value
up to 30 seconds can override it for deliberate out-of-window inspection.
Both durations and any mismatch are recorded in the report. Longer clips
increase inversion time and memory use.

For audible inspection, each clip is selected as the highest-RMS reference
window on the configured deterministic time grid. Selection never examines
model outputs. Reports record the method, grid spacing, and selected offset for
every example.

Changing the configured split to `test` additionally requires
`--confirm-test-use`. Test evaluation should occur only after the comparison
metrics and decision rules have been preregistered.

### Inspect a comparison in the UI

Start the minimal TypeScript comparison interface after generating a report:

```bash
docker compose up --build -d ui
```

Open [http://localhost:4173](http://localhost:4173), enter the comparison run
ID printed by `singalign-compare`, and select **Load comparison**. When no run
is specified, the latest successfully completed local comparison loads by
default. The report
directory is mounted read-only. The interface displays aggregate paired
metrics and side-by-side reference, baseline, and aligned audio for each
available example.

The listening view is an inspection aid, not a blinded perceptual study. Its
generated audio is approximate Griffin-Lim reconstruction and must not be used
alone to support claims about perceptual quality. See [`ui/README.md`](ui/README.md)
for development and troubleshooting instructions.

The UI still has deferred work around richer candidate selection, conditioning
metadata, cross-condition uncertainty summaries, and a separate blinded
listening-study interface.

The UI now includes a training interface for the implemented baseline, aligned,
conditioned, vocoder, and KTO experiments with default parameters. The
Docker-backed API is started with `docker compose up --build api`; it exposes
`POST /training` for allowlisted jobs and `GET /training/<job_id>` for status.
MLflow remains available at port 5001. The browser still displays the generated
command as a reproducibility fallback.
Launch requests forward only numeric allowlisted parameters and automatically
attach the supervised checkpoint for aligned/KTO jobs.
When the API is running, submitting the form launches the Docker job and shows
its container ID; if the API is unavailable, the same command remains visible
for manual execution.

The UI workflow is organized sequentially into separate tabs: **Training** for
launching model jobs, **Evaluation** for loading and inspecting evaluation
reports, and **Comparison** for paired or multi-condition result review.
Downstream tabs remain unavailable until the relevant upstream run or report is
loaded, making the dependency order visible during reproducible experiments.

### Candidate-generation sandbox

Candidate generation is supporting infrastructure for the two primary studies.
It creates deterministic variants of a synthesis or transfer condition so we
can compare pitch, timing, lyric intelligibility, audio quality, and alignment
failures. Each candidate records its seed, method, input condition, and output
provenance.

The sandbox can also apply transparent proxy scores and stable reranking, but
these scores are engineering diagnostics—not human-preference models. DPO, KTO,
and reward-model code remains optional exploratory infrastructure and is not a
primary research direction.

Example Docker invocation:

```bash
docker compose run --rm research \
  singalign-candidates \
  --input input.pt \
  --output reports/candidates/example.json
```

Candidate reports can be logged to MLflow with their condition metadata so
results remain reproducible.

Run the exploratory KTO condition from Docker with:

```bash
docker compose run --rm research \
  singalign-kto-train \
  --config configs/training/kto.yaml \
  --checkpoint checkpoints/baseline/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Stop the services without removing tracked runs:

```bash
docker compose down
```

Do not remove `.mlflow/` unless you intend to permanently delete the local
MLflow database and artifacts. See
[`configs/mlflow/README.md`](configs/mlflow/README.md) for storage and network
details.

## Evaluation plan

Evaluation will combine objective and perceptual evidence.

| Dimension | Candidate measurements |
| --- | --- |
| Pitch accuracy | F0 error, voiced/unvoiced error, note-level deviation |
| Rhythm accuracy | onset and duration deviation |
| Intelligibility | ASR error rate and human lyric recognition |
| Singer similarity | embedding similarity and human judgments |
| Audio quality | learned quality estimators and artifact analysis |
| Preference | blinded pairwise human comparisons |

Reported experiments will include confidence intervals, effect sizes, and
statistical tests where appropriate. Metrics will be treated as imperfect
proxies rather than interchangeable substitutes for human judgments.

The detailed protocol will live in
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Repository structure

```text
configs/       Versioned experiment and model configurations
data/          Dataset documentation and local data conventions
docs/          Research questions, protocols, and responsible-use analysis
experiments/   Experiment manifests and reproducibility records
reports/       Generated tables, figures, and research reports
src/           Data, tracking, model, and research utilities
ui/            Dockerized TypeScript comparison interface
```

Model code and training configurations will be introduced through separate
reviewed changes after their research claims and evaluation contracts are
registered.

## Reproducibility principles

The project will follow these practices:

- version all reported experiment configurations
- record random seeds, software versions, and hardware assumptions
- keep evaluation splits immutable after they are registered
- identify every reported result with an experiment manifest
- distinguish exploratory results from confirmatory results
- preserve failed experiments when they inform a conclusion
- report uncertainty instead of relying only on point estimates

## Responsible research

Singing voice generation creates risks involving consent, impersonation,
copyright, and misleading synthetic media. SingAlign will therefore:

- use only datasets with documented research permissions
- avoid presenting generated voices as real performances
- disclose synthetic audio in demonstrations
- document model and dataset limitations
- avoid releasing tools intended for unauthorized voice impersonation
- preserve dataset-specific attribution and usage restrictions

See [`docs/responsible-research.md`](docs/responsible-research.md) for the
evolving risk assessment.

## Roadmap

- [x] Define the PJS-supported singing synthesis and vocal-transfer direction
- [x] Review the PJS dataset terms and document local installation
- [ ] Validate the corpus and create immutable data splits
- [ ] Select and reproduce a baseline model
- [ ] Implement controlled preference-pair construction
- [ ] Establish objective evaluation baselines
- [ ] Extract and validate observed F0 conditioning
- [ ] Complete same-singer score-conditioned synthesis evaluation
- [ ] Implement MIDI-based target-instrument rendering
- [ ] Implement content-and-melody transfer baselines
- [ ] Evaluate synthesized-vocal transfer over target instrumentals
- [ ] Train and evaluate reward models as optional exploratory extensions
- [ ] Compare reranking, supervised fine-tuning, DPO, and KTO as optional
      diagnostics
- [x] Exclude participant-based listening claims from the simulation-sandbox scope
- [ ] Add a suitable multi-singer dataset before making identity-generalization claims
- [ ] Publish the final report and reproducibility package

The roadmap indicates intended work, not completed capabilities.

### Completed implementation milestone

The initial four-PR implementation milestone is complete: learned reward model
baselines, unified cross-condition analysis, sequential
Training/Evaluation/Comparison workflow safeguards, and the reproducibility
package are merged. Participant-based listening studies are intentionally
excluded from this completion plan.

The remaining simulation-sandbox work is consolidated into one umbrella PR:
tracked reward-model training/evaluation, the complete comparison matrix,
finished research UI views, optional informal listening notes, and the final
simulation report. This larger PR will retain separate commits and validation
records for each workstream while keeping the repository's final state easy to
reproduce from one review.

The first workstream adds `singalign-reward-train`, a Docker/MLflow command
that trains a learned reward model from saved chosen/rejected tensors and
records its checkpoint and pairwise diagnostic metrics.
The second workstream uses a versioned comparison-matrix manifest to run the
same paired analysis across available baseline, reranking, DPO, KTO,
conditioned, and vocoder outputs; missing outputs are reported as pending
rather than silently treated as results.
Run `singalign-matrix-status --config configs/evaluation/comparison-matrix.yaml`
to audit readiness before producing the aggregate report.
Matrix execution is data-dependent: use `singalign-matrix-status` first, and
only run `singalign-condition-analysis` when every declared condition is ready.
The complete matrix run is intentionally deferred to a follow-up PR so it can
be executed as a separately tracked, time-bounded experiment.

The remaining non-matrix tasks are consolidated into one follow-up PR: expose
the learned reward trainer and provenance in the UI, show candidate and
conditioning details, retain informal engineering notes, and populate the
simulation-sandbox report/checklists. Missing data-dependent results remain
explicitly pending.
The completion UI will consume these reports with condition filtering and show
available uncertainty/effect-size fields without inventing values for reports
that predate the aggregate schema.
The final workstreams include an informal engineering feedback template and a
simulation-sandbox report scaffold; neither is a participant study or a source
of population-level preference evidence.
Use [`docs/informal-feedback-template.md`](docs/informal-feedback-template.md)
for engineering notes and [`docs/simulation-sandbox-report.md`](docs/simulation-sandbox-report.md)
for the final report structure.

PR 1 is implementing learned reward baselines on top of the existing
deterministic preference-pair generator. The learned models are exploratory:
they provide scalar and multidimensional scoring baselines with reproducible
pairwise training, while proxy rewards remain available for transparent
diagnostics.

PR 2 will unify cross-condition analysis. It will apply one paired metric
contract, bootstrap uncertainty intervals, effect sizes, and stable condition
ordering to baseline, reranking, DPO, KTO, conditioned, and vocoder outputs.
The aggregate analysis command will consume a versioned JSON manifest of shared
references and condition tensors, so every condition is evaluated on the same
examples without relying on independently sampled datasets.
Use [`configs/evaluation/condition-analysis.example.json`](configs/evaluation/condition-analysis.example.json)
as the input shape for `singalign-condition-analysis --manifest ... --output ...`.

PR 3 will connect the UI tabs into a sequential workflow. A shared experiment
registry will define each method's required checkpoint, evaluation protocol,
and compatible comparison conditions; the UI will surface prerequisites and
prevent incompatible downstream selections.
The registry is intentionally declarative so the same experiment identity can
be reused by training, evaluation, and comparison controls.
Evaluation is a separate command-generation step: it consumes the selected
experiment's checkpoint and protocol, while Comparison consumes the resulting
report rather than directly reusing training settings.

PR 4 is the final reproducibility package: it will consolidate the Docker
workflow, experiment manifests, environment metadata, tracked run references,
and end-to-end reproduction commands. This package is intended to make the
simulation sandbox auditable without implying that its proxy metrics establish
human preference.
The verification workflow is documented in [`docs/reproducibility.md`](docs/reproducibility.md)
and can be run with `bash scripts/reproduce_smoke.sh`.
Use [`docs/run-record-template.md`](docs/run-record-template.md) to record each
experiment's provenance and MLflow lineage.

## Contributing

Research contributions should state the hypothesis being tested, describe the
experimental controls, and include a reproducible evaluation plan. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a change.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until the
project has a formal release, cite the repository and the exact commit used.

## License

Repository code and original documentation are licensed under the
[Apache License 2.0](LICENSE).

Datasets, pretrained models, third-party implementations, and generated
artifacts may be governed by separate terms. The Apache-2.0 license does not
override those terms.
