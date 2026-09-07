# Research Plan

## Purpose

SingAlign is a reproducible engineering sandbox for two PJS singing-voice
studies:

1. same-singer, score-conditioned synthesis; and
2. preservation of vocal content and melody over a different instrumental.

The project prioritizes objective signal analysis, explicit controls, immutable
data splits, and artifact lineage. It does not currently support production
singing synthesis, population-level conclusions, or confirmatory human-
preference claims.

This plan describes the current research boundary. Detailed commands belong in
the [root README](../README.md), while fixed experiment requirements belong in
the [two-study protocol](two-studies-experiments.md).

## Research questions

### Study 1: score-conditioned synthesis

Can a compact model reconstruct the single PJS vocalist from phonemes,
MusicXML/MIDI score timing, pitch, and aligned acoustic frames?

The study is a same-singer reconstruction diagnostic. PJS contains one
vocalist, so it cannot support unseen-singer generalization claims.

### Study 2: content-and-melody transfer

Can the words, phoneme timing, and melody of a source vocal be preserved when
the performance is placed over a different instrumental track?

The implemented baseline is a transparent signal-processing control. It uses
declared pitch and tempo transforms plus a deterministic MIDI/MusicXML-rendered
instrumental. It is not a learned voice-conversion system.

## Current system boundary

| Area | Status | Current boundary |
| --- | --- | --- |
| PJS validation, indexing, and split generation | Implemented | Deterministic 80/10/10 song-disjoint split |
| Score, phoneme, timing, and F0 conditioning | Implemented | Versioned records; missing F0 is not silently interpolated |
| Study 1 compact mel model | Implemented baseline | Runnable training and checkpointing; not a production synthesizer |
| Study 2 transfer control | Implemented baseline | Declared pitch/tempo alignment, deterministic accompaniment, mixing, and diagnostics |
| Objective evaluation | Implemented | Pitch, timing, spectral/mel, clipping, artifact, and transfer diagnostics |
| Experiment lineage | Implemented | Docker, MLflow, configs, manifests, fingerprints, and reports |
| Comparison UI and API | Implemented inspection tooling | Unblinded; not a listening-study interface |
| Preference alignment | Exploratory infrastructure | Synthetic pairs, reranking, reward models, DPO-style and KTO-style proxies |
| Diffusion voice conversion | Architecture scaffold only | Forward pass, schedule, loss, and config contracts; no weights or sampler |
| Neural vocoder | Exploratory contract | Diagnostic trainer exists; no validated production-quality decoder |
| Multi-singer generalization | Deferred | Requires a suitable licensed, song-disjoint dataset |
| Human evaluation | Deferred | Requires a separate protocol, recruitment plan, and preregistration |

The API exposes this boundary through `GET /capabilities`. Tooling must not
present scaffolded components as trained or available features.

## Shared experimental protocol

Both studies use PJS v1.1 and the split registered in
`configs/training/studies.yaml`.

Every reported run must:

- use the immutable song-disjoint split and record its fingerprint;
- keep raw corpus files unchanged and outside Git;
- record the dataset version, Git revision and dirty state, configuration,
  seed, hardware assumptions, and relevant input/checkpoint hashes;
- train on the training partition only;
- select checkpoints on validation data only;
- keep held-out test evaluation separate from training and model selection;
- log metrics and artifacts to MLflow from the reproducible Docker environment;
- export a committed manifest or report for any result cited in repository
  documentation; local MLflow run IDs alone are not reviewable evidence;
- distinguish exploratory diagnostics from research claims; and
- preserve failed runs when they affect interpretation.

Reports should include per-example values, aggregate estimates, sample counts,
uncertainty intervals where meaningful, exclusions, and known failure cases.

## Study 1 plan

### Inputs and baseline

The conditioning pipeline parses MusicXML note/rest events and timed phoneme
labels, extracts frame-level observed F0 for targets or diagnostics, and aligns
all signals to a common acoustic frame grid. Crop offsets, tempo, frame rate,
and duration are explicit.

`ScoreConditionedMelModel` is the runnable compact baseline. Its frozen
exploratory configuration uses 16 kHz audio, 80-bin log-mel targets, 100
conditioning frames per second, three-second windows, and a 10-epoch budget.

### Evaluation

Primary diagnostics are:

- mel and spectral reconstruction error;
- F0 and voiced/unvoiced error;
- score onset and duration deviation;
- clipping, peak level, and artifact checks; and
- checkpoint, split, and configuration lineage.

Generated waveform inspection currently depends on an exploratory vocoder or
approximate Griffin-Lim inversion. Neither supports a production audio-quality
claim.

### Next evidence milestone

Package a fully reproducible Study 1 run that links the conditioning record,
selected checkpoint, sealed-test report, representative audio artifacts, and
exported lineage metadata. Interpret the result as a same-singer baseline only.

## Study 2 plan

### Inputs and baseline

Each experiment uses a versioned, song-disjoint source/target pair manifest.
The source vocal supplies linguistic content, phoneme timing, and melody. The
target score supplies the musical context and is rendered to a fixed
instrumental WAV.

The deterministic control:

1. applies the declared semitone shift to the source vocal;
2. corrects duration introduced by pitch resampling;
3. applies the declared tempo scale;
4. mixes the aligned vocal with the rendered instrumental; and
5. preserves the source vocal, aligned vocal, instrumental, and final mix.

The first comparison set should include an unchanged key/tempo control, a
declared transposition or tempo change, and an intentionally misaligned
control. Automatic beat detection, key estimation, and generated accompaniment
remain outside the implemented baseline.

### Evaluation

Primary diagnostics are:

- content and lyric preservation;
- voiced-F0 error and correlation;
- onset and duration alignment;
- vocal activity, clipping, and peak level;
- stem and final-mix artifact analysis; and
- pair identity, transform settings, and artifact lineage.

Lyric intelligibility and musical fit remain separate axes. A technically
aligned vocal is not automatically a perceptually successful transfer.

### Next evidence milestone

Run the registered pair set across aligned and deliberately misaligned
conditions, then publish a single tracked comparison report with shared
references and deterministic ordering.

## Exploratory alignment infrastructure

Candidate generation, deterministic reranking, learned scalar and
multidimensional reward models, DPO-style post-training, and KTO-style
post-training remain available for engineering comparisons. Current preference
pairs and scores are synthetic proxies derived from controlled degradations.

These experiments may test software behavior and metric trade-offs, but they
must not be described as evidence of human preference alignment. A future
human-preference study would require suitable data, blinded evaluation,
listener eligibility criteria, a power rationale, stopping rules, and a
preregistered analysis.

## Committed research records

Repository documentation should cite only records that remote readers can
inspect. The frozen pilot manifest is
[`experiments/pilot-3s-10e-manifest.yaml`](../experiments/pilot-3s-10e-manifest.yaml).
Its associated [analysis plan](../experiments/analysis-plan-v1.md) remains a
draft; no external preregistration or confirmatory claim has been made.

Local `.mlflow/` state, checkpoints, and generated reports are intentionally
ignored. They do not count as remote evidence unless their relevant lineage,
configuration, metrics, and interpretation are exported into a committed
manifest or report.

## Future research

### Near-term: complete the current evidence package

- Run the full registered multi-condition matrix with shared references.
- Consolidate Study 1 and Study 2 artifacts into stable, tracked reports.
- Add qualitative failure-case examples without implying population-level
  preference.
- Keep the UI focused on reproducible inspection and explicit provenance.

### GPU-backed neural extension

A learned transfer system would use:

`phonemes/content + target F0/timing + singer embedding -> mel -> vocoder`

The repository currently contains `ScoreConditionedMelDiffusion`,
`ConditionalMelDiffusion`, `DiffusionSchedule`, and configuration contracts.
It does not contain trained diffusion weights, a reverse-sampling loop, or a
validated vocoder connection.

Before training this extension:

1. select and document a licensed multi-singer dataset;
2. define song- and singer-disjoint evaluation partitions;
3. freeze content, pitch, timing, and timbre conditioning contracts;
4. implement and test sampling and vocoder integration;
5. register compute, checkpoint-selection, and stopping policies; and
6. compare against the deterministic Study 2 control using the same metrics.

See the
[diffusion voice-conversion contract](../experiments/diffusion-voice-conversion-v1.md)
and `configs/training/diffusion.yaml`.

### Human evaluation extension

Human evaluation remains optional and separate from the current milestone. It
may begin only after playable systems and objective failure checks are stable.
Any participant study must address consent, recruitment, randomization,
blinding, listener exclusions, statistical power, data handling, and ethical
review requirements before responses are collected.

## Threats to validity

- **Single vocalist:** PJS cannot measure singer generalization or population
  diversity.
- **Small corpus:** model capacity and apparent improvements may reflect
  memorization or unstable estimates.
- **Proxy metrics:** spectral, pitch, ASR, and reward scores do not substitute
  for perception.
- **Approximate decoding:** Griffin-Lim and an exploratory vocoder can dominate
  audible artifacts independently of the acoustic model.
- **Declared alignment:** Study 2 currently depends on provided tempo and pitch
  transforms rather than automatic estimation.
- **Synthetic preferences:** controlled degradations may reward shortcuts that
  do not match listener priorities.
- **Exploratory reuse:** repeated inspection of held-out results can erode the
  meaning of the test boundary.

## Responsible research boundary

Only datasets and checkpoints with documented research permissions should be
used. Generated audio must be disclosed as synthetic, dataset attribution must
be preserved, and the project should not facilitate unauthorized voice
impersonation. See [responsible research](responsible-research.md) for the full
risk assessment.
