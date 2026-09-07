# SingAlign

SingAlign is a reproducible sandbox for training and evaluating singing-voice
systems on the PJS corpus. It focuses on objective signal analysis, traceable
experiments, and clear separation between working baselines and future research.

> [!IMPORTANT]
> The experimental sandbox is implemented, but the neural-model scaffolds and
> research conclusions are provisional. This is not a production singing
> synthesizer.

## Project at a glance

SingAlign supports two studies:

| Study | Goal | Current approach |
| --- | --- | --- |
| **1. Score-conditioned synthesis** | Reconstruct the PJS vocalist from phonemes, score timing, and pitch | Compact mel-model baseline |
| **2. Content-and-melody transfer** | Preserve a source vocal over a different instrumental | Deterministic pitch/tempo alignment and mixing |

Both studies follow the same evidence pipeline:

```text
PJS corpus
    |
    v
validate -> index -> song-disjoint split
    |
    v
conditioning records (phonemes + score + timing + F0)
    |
    +-----------------------------+
    |                             |
    v                             v
Study 1                       Study 2
mel-model baseline            deterministic transfer control
    |                             |
    +--------------+--------------+
                   |
                   v
       objective evaluation
                   |
                   v
      reports + audio + MLflow lineage
```

The held-out test split is not available to training. Checkpoints are selected
on validation data and evaluated separately.

## What is implemented?

```text
IMPLEMENTED                  SCAFFOLDED                   FUTURE RESEARCH
-----------                  ----------                   ---------------
PJS data pipeline            diffusion denoisers    ---> trained weights
conditioning contracts       diffusion schedules    ---> sampling loop
compact mel baseline         vocoder contract        ---> neural vocoder
deterministic transfer       preference utilities   ---> human evaluation
objective evaluation                                  multi-singer study
Docker + MLflow + UI
```

The diffusion and preference-learning code makes later interfaces concrete; it
does not imply trained voice conversion, human preference alignment, or
unseen-singer generalization. PJS contains one vocalist. See
[`experiments/diffusion-voice-conversion-v1.md`](experiments/diffusion-voice-conversion-v1.md)
for the future model contract.

## Quick start with Docker Compose

Docker is the recommended path and supports Apple Silicon. It provides a pinned
Python 3.11 environment and local MLflow tracking.

### 1. Get the data

Download PJS version 1.1 from the
[official corpus page](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus)
and place it at:

```text
data/raw/pjs/PJS_corpus_ver1.1/
```

The corpus is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Dataset files
remain local and are ignored by Git. See [`data/README.md`](data/README.md) for
provenance, verification, and licensing details.

### 2. Build and start tracking

```bash
SINGALIGN_GIT_REVISION="$(git rev-parse HEAD)" \
SINGALIGN_GIT_DIRTY="$(test -n "$(git status --porcelain --untracked-files=no)" \
  && echo true || echo false)" \
docker compose build

docker compose up -d mlflow
docker compose ps
```

MLflow is available at [http://localhost:5001](http://localhost:5001). Local
tracking data persists in the ignored `.mlflow/` directory.

### 3. Validate, index, and split PJS

```bash
docker compose run --rm research \
  singalign-data validate \
  --root /workspace/data/raw/pjs/PJS_corpus_ver1.1

docker compose run --rm research \
  singalign-data index \
  --root /workspace/data/raw/pjs/PJS_corpus_ver1.1 \
  --output /workspace/data/interim/pjs/index.jsonl

docker compose run --rm research \
  singalign-data split \
  --index /workspace/data/interim/pjs/index.jsonl \
  --output /workspace/data/interim/pjs/splits.json \
  --seed 2026
```

The generated metadata remains local and ignored by Git.

### 4. Run the baseline workflow

Train the reconstruction baseline:

```bash
docker compose run --rm research \
  singalign-train \
  --config configs/training/baseline.yaml \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Select `checkpoints/baseline/best.pt` using validation loss, then evaluate it on
the immutable test split:

```bash
docker compose run --rm research \
  singalign-evaluate \
  --config configs/evaluation/baseline.yaml \
  --checkpoint checkpoints/baseline/best.pt \
  --index data/interim/pjs/index.jsonl \
  --splits data/interim/pjs/splits.json
```

Reports are written under `reports/evaluation/<run-id>/` and attached to the
MLflow run. They include aggregate and per-example metrics, confidence
intervals, latency, configuration, and data/checkpoint fingerprints.

For the score-conditioned Study 1 model, use
`configs/training/conditioned.yaml` with `singalign-conditioned-train`.

### 5. Run tests and open the UI

```bash
docker compose build test
docker compose run --rm lint
docker compose run --rm test
docker compose up --build -d api ui
```

- Comparison UI: [http://localhost:4173](http://localhost:4173)
- API: [http://localhost:8000](http://localhost:8000)
- MLflow: [http://localhost:5001](http://localhost:5001)

The UI organizes work into **Training**, **Evaluation**, and **Comparison**.
Generated comparison audio uses approximate Griffin-Lim reconstruction and is
an inspection aid, not evidence of production audio quality or a blinded
listening study. See [`ui/README.md`](ui/README.md).

Stop services without deleting tracked runs:

```bash
docker compose down
```

Do not remove `.mlflow/` unless you intend to delete its local database and
artifacts.

## Study 2: deterministic transfer control

Study 2 uses a fixed MIDI/MusicXML-rendered instrumental so vocal alignment can
be measured without adding accompaniment generation as another variable.

```text
source vocal --------------------+
    |                             |
    v                             |
declared pitch shift              |
    |                             |
declared tempo alignment          |
    |                             |
    v                             v
aligned vocal + rendered target instrumental
                    |
                    v
             final mix + report
```

The pipeline preserves the original vocal, aligned vocal, and final mix as
separate artifacts. It does not estimate key or beat automatically and is not a
trained voice-conversion model. The exact experiment controls and required
lineage are documented in
[`docs/two-studies-experiments.md`](docs/two-studies-experiments.md).

## Evaluation

Objective signal-processing analysis is the primary evidence:

| Dimension | Examples |
| --- | --- |
| Pitch | F0 error, voiced/unvoiced error, note deviation |
| Timing | Onset and duration deviation |
| Content | ASR error and lyric recognition |
| Audio | Spectral/mel similarity, clipping, artifact analysis |
| Transfer | Content preservation and alignment against controls |

Metrics are imperfect proxies and are interpreted against explicit references
and controls. Human listening measures are complementary and must not be
implied by proxy preference experiments. See
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Repository map

```text
configs/          versioned training and evaluation settings
data/             dataset provenance and local-data conventions
docs/             research plans, protocols, and limitations
experiments/      registered experiment designs and manifests
reports/          generated evidence (mostly ignored)
src/singalign/    data, models, studies, evaluation, and tracking code
tests/            corpus-independent unit and integration tests
ui/               Dockerized experiment and comparison interface
```

Configuration says what a run means, source code executes it, MLflow records
its lineage, and reports hold its evidence. Raw data, checkpoints, generated
reports, and local MLflow state are not source artifacts.

## Research boundaries

SingAlign does not currently claim:

- production-quality singing synthesis or voice conversion
- human preference alignment
- unseen-singer or population-level generalization
- confirmatory listening-study results

Future work requires multi-singer song-disjoint data, trained diffusion models,
a sampling pipeline, a validated neural vocoder, automatic alignment, and an
appropriate human-evaluation design. Architecture references and detailed next
steps are in [`docs/research-plan.md`](docs/research-plan.md).

Singing-voice generation also raises consent, impersonation, copyright, and
synthetic-media risks. See
[`docs/responsible-research.md`](docs/responsible-research.md) for the project's
use and disclosure principles.

## Contributing

Contributions should state the hypothesis, controls, and reproducible evaluation
plan. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Repository code and
original documentation are licensed under the [Apache License 2.0](LICENSE).
Datasets, pretrained models, third-party implementations, and generated
artifacts may have separate terms.
