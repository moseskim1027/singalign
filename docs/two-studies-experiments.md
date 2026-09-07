# Two-Study Experiment Protocol

## Purpose

This document defines the shared experiment contract for SingAlign's two PJS
studies:

1. same-singer, score-conditioned synthesis; and
2. content-and-melody transfer over a different instrumental.

The current baselines are deliberately small and interpretable. The surrounding
data, execution, tracking, and evaluation contract is intended to remain stable
when stronger acoustic models, vocoders, or alignment methods replace them.

For project scope and future milestones, see the [research plan](research-plan.md).
For setup and commands, see the [root README](../README.md).

## Shared run contract

Both studies use PJS v1.1 and the defaults in
`configs/training/studies.yaml`:

| Setting | Contract |
| --- | --- |
| Dataset | PJS v1.1; raw files remain unchanged and outside Git |
| Split | Deterministic 80/10/10 song-disjoint split with seed `2026` |
| Audio | 16 kHz |
| F0 | 100 frames/second, 70–1000 Hz range, `0.35` voicing threshold |
| Execution | `research` Docker Compose service |
| Tracking | MLflow through `http://mlflow:5000` inside Compose |
| Run status | Exploratory unless a separate confirmatory protocol is frozen |

Every run must record:

- dataset name and version;
- split fingerprint;
- Git revision and dirty state;
- resolved configuration and random seed;
- relevant input, manifest, and checkpoint identities;
- hardware and software environment; and
- generated metrics and artifact paths.

Training may read only the training partition. Checkpoint selection may read
only validation data. Held-out test evaluation must be a separate command and
must not feed back into model selection.

MLflow state is local and ignored by Git. Any result cited in repository
documentation must therefore export its relevant configuration, lineage,
metrics, and interpretation into a committed manifest or report. A local
MLflow run ID by itself is not reviewable from the PR branch.

## Stable and replaceable stages

| Stable experiment layer | Replaceable research component |
| --- | --- |
| Dataset provenance and split contract | Dataset adapter for a future licensed corpus |
| Conditioning-record schema and timing conventions | Conditioning encoder |
| Docker execution and resolved configs | Acoustic or transfer model |
| MLflow logging and artifact naming | Optimizer and training method |
| Held-out evaluation and report schema | Vocoder, alignment method, or candidate generator |
| Deterministic baselines and controls | Learned model compared against those controls |

Changing a research component must not silently change the split, preprocessing,
metric definitions, or checkpoint-selection policy. If one of those contracts
must change, register it as a new experiment condition and report the difference.

## Replacement experiment registration contract

A baseline replacement is reproducible only when its resolved parameters are
stored in a committed configuration or manifest. MLflow may mirror that
configuration during execution, but local tracking state is not the canonical
record for reviewing the experiment.

### Training parameters

The following fields are required for every learned replacement in either
study. `Not applicable` is valid only with a short justification, such as for a
deterministic control with no optimizer.

| Parameter group | Required fields | Registration rule |
| --- | --- | --- |
| Data identity | Dataset, version, license reference, manifest path, split path, split fingerprint | Freeze before training; do not substitute local paths for dataset identity |
| Input representation | Sample rate, segment duration, feature type, FFT/hop settings, frame rate, normalization | Record resolved values and preprocessing implementation version |
| Conditioning | Inputs used, vocabulary or encoder version, dimensions, timing convention, missing-value policy | Distinguish training-only targets from inference-available inputs |
| Model | Architecture name, implementation revision, parameter count, layer/channel dimensions, initialization checkpoint | Record `none` when training from scratch |
| Optimization | Loss terms and weights, optimizer, learning rate, scheduler, weight decay, gradient clipping | Record all defaults explicitly rather than relying on library defaults |
| Training budget | Batch size, gradient accumulation, epochs or update steps, validation frequency | Declare the stopping unit and maximum budget before the run |
| Reproducibility | Seed, device, precision, worker count, deterministic settings | Record any operation known to be nondeterministic |
| Selection | Validation metric, direction, checkpoint frequency, early-stopping rule | Freeze before inspecting held-out test results |
| Resume policy | Source checkpoint, optimizer-state behavior, completed steps | Required whenever a run does not start from its declared initialization |
| Outputs | Checkpoint path, config snapshot, logs, report schema, artifact hashes | Export reviewable lineage into a committed manifest or report when cited |

### Generation and sampling parameters

Learned models must register inference separately from training. This prevents
changes in a sampler or decoder from being mistaken for changes in the acoustic
model.

| Component | Required fields |
| --- | --- |
| Acoustic generation | Checkpoint identity, input condition manifest, inference seed, requested duration, batching policy |
| Diffusion sampling | Schedule, number of inference steps, sampler, variance/noise policy, guidance method and scale |
| Mel processing | Value range, normalization/inversion parameters, padding and crop policy |
| Vocoder | Architecture, checkpoint hash, expected mel contract, hop length, sample rate, loudness/normalization policy |
| Waveform output | File format, bit depth, peak/limiter policy, silence trimming, artifact naming |

Values for the future diffusion sampler and validated vocoder are intentionally
not specified yet because neither component has been selected or trained. They
must be frozen in versioned configs before the first comparison run.

### Study-specific experiment parameters

| Study | Parameters that must be registered for every condition |
| --- | --- |
| Study 1 | Conditioning fields, acoustic-model config, training budget, decoder/vocoder identity, checkpoint-selection metric, synthesis seed, evaluation window |
| Study 2 deterministic control | Pair-manifest identity, source and target IDs, sample rate, tempo scale, transposition, vocal/instrument gains, renderer settings, alignment policy |
| Study 2 learned replacement | Content encoder, target-F0 source, timing/alignment method, singer/timbre representation, acoustic or conversion model, training contract, sampler, vocoder, mixing policy |

When comparing a replacement with a baseline, also register which parameters
are **held constant**, which are **intentionally changed**, and which are
**not applicable**. The comparison report must repeat that declaration.

## Study 1: score-conditioned synthesis

### Question

Can phonemes, score timing, and pitch reconstruct the single PJS vocalist under
a controlled, song-disjoint evaluation?

This is a same-singer diagnostic. It does not test voice cloning or unseen-
singer generalization.

### Inputs and conditioning

Each versioned conditioning record contains:

- MusicXML/MIDI note and rest events;
- phoneme symbols with start and end times;
- frame-level MIDI pitch and phoneme IDs;
- explicit frame rate, crop offset, duration, and tempo; and
- observed F0, voicing, and confidence when used as a target or diagnostic.

Unvoiced F0 frames remain `null`. Missing pitch must not be interpolated
silently. Observed performance F0 is a target or diagnostic; an inference path
must not depend on reference-performance information that would be unavailable
when synthesizing a new vocal.

### Current baseline

`ScoreConditionedMelModel` maps frame-aligned MIDI pitch and phoneme IDs to an
80-bin mel spectrogram. The exploratory configuration is frozen in
`configs/training/conditioned.yaml`.

The model validates conditioning, tensor, training, checkpoint, and tracking
interfaces. It is not a complete synthesizer. Playable synthesis still needs a
validated waveform decoder; the current Griffin-Lim path and exploratory
`MelVocoder` are engineering diagnostics.

For a replacement model, classify and register parameters as follows:

| Parameter | Classification | Replacement rule |
| --- | --- | --- |
| Dataset version and split | Shared experiment parameter | Hold fixed for a direct comparison |
| Sample rate, FFT, hop length, and mel bins | Input/output contract | Hold fixed unless the representation change is the registered experimental factor |
| Segment duration and conditioning frame rate | Experiment-specific | Declare in the condition config and apply consistently to training and evaluation |
| Phoneme vocabulary and pitch encoding | Conditioning-interface specific | Version the encoder and document unknown, rest, and padding conventions |
| Architecture, layers, channels, embeddings, and parameter count | Model-specific | Register the complete resolved model configuration |
| Loss terms and weights | Model-specific | Register every objective and coefficient |
| Optimizer, learning rate, scheduler, and regularization | Model-specific | Register explicit values; do not rely on framework defaults |
| Batch size and gradient accumulation | Model- and compute-specific | Register the effective batch size |
| Epochs, update steps, validation cadence, and stopping rule | Experiment-specific | Freeze the training budget before the run |
| Seed and checkpoint-selection metric | Shared comparison parameter | Hold fixed unless the experiment explicitly studies them |
| Device, precision, and worker count | Hardware-specific | Record the resolved execution values and known nondeterminism |
| Sampler and vocoder | Model-specific inference parameter | Register separately from acoustic-model training |

`configs/training/conditioned.yaml` remains authoritative for the current
baseline. Each replacement must receive its own committed configuration rather
than copying values from this documentation.

### Evaluation outputs

The report should contain:

- mel and spectral error;
- F0 and voiced/unvoiced error;
- onset and duration deviation;
- clipping, peak, and artifact diagnostics;
- per-example and aggregate values; and
- config, split, checkpoint, and source identities.

The first reviewable evidence package should link a conditioning record,
selected validation checkpoint, sealed-test report, representative audio, and
exported lineage metadata.

### Replacement path

A stronger Study 1 system may replace the compact mel model with a sequence or
diffusion acoustic model and replace approximate inversion with a trained neural
vocoder. It must keep the same conditioning and evaluation contract unless the
change is explicitly registered and compared as an ablation.

## Study 2: content-and-melody transfer

### Question

Can a source vocal's linguistic content, phoneme timing, and melody be preserved
when it is placed over a different instrumental context?

### Pair manifest

Every experiment uses a versioned, song-disjoint source/target pair manifest.
`experiments/study-2-pairs.example.json` defines the initial schema. Each pair
declares:

- source and target IDs;
- split membership;
- tempo scale;
- transposition in semitones; and
- condition name.

Pair identity and transforms must be recorded exactly as executed. Automatic
estimates must not replace declared values without creating a new registered
condition.

### Current deterministic control

The source vocal supplies the words, timing, and melody. The target score is
rendered to a fixed instrumental WAV with the deterministic renderer. Generated
accompaniment is outside this control because it would introduce another source
of variation.

The transfer command:

1. applies the declared pitch shift by resampling;
2. corrects the pitch-shift duration change;
3. applies the declared tempo scale;
4. mixes the aligned vocal and target instrumental at fixed gains; and
5. saves the aligned vocal and final mix with clipping diagnostics.

Preserve the original source vocal, aligned vocal, rendered instrumental, and
final mix as separate artifacts. The control does not estimate beat or key,
preserve formants with a production-grade algorithm, or perform learned timbre
conversion.

The control has no learned training parameters. Its experiment parameters are:

| Parameter | Registration source |
| --- | --- |
| Dataset, split, sample rate, and F0 policy | `configs/training/studies.yaml` |
| Source ID, target ID, condition, tempo scale, and transposition | Versioned pair manifest |
| Renderer BPM, sample rate, seed, and score identity | Renderer invocation and exported metadata |
| Vocal and instrumental gains | Resolved transfer configuration or invocation |
| Input/output artifact identities | Exported transfer metadata and report |

The example pair manifest defines the unchanged control. Transformed and
misaligned conditions must declare their own values rather than inheriting
undocumented command defaults.

### Required conditions

The initial comparison set contains:

| Condition | Purpose |
| --- | --- |
| Unchanged key and tempo | Verify the render/mix path without a declared transform |
| Transposed and/or tempo-scaled | Test the declared deterministic alignment control |
| Intentionally misaligned | Establish sensitivity of alignment metrics |
| Learned transfer | Future condition; compare only after a trained checkpoint exists |

All conditions for a pair must share the same source reference, target
instrumental, sample rate, evaluation window, and metric implementation.

### Evaluation outputs

Evaluate content and musical alignment separately. The report should contain:

- lyric/content preservation diagnostics;
- voiced-F0 error and correlation;
- onset and duration deviation;
- vocal activity, clipping, and peak level;
- stem and final-mix artifact diagnostics;
- per-pair and aggregate values; and
- pair manifest, transforms, config, and artifact identities.

The next evidence package should run the registered pair set across aligned and
misaligned conditions, using shared references and deterministic report order.

### Replacement path

A learned transfer system would replace the deterministic vocal transform with
a model conditioned on content or phonemes, target F0 and timing, and target
singer or timbre. Score-, beat-, or audio-derived alignment may replace declared
transforms when it is registered as its own component.

The fixed instrumental, deterministic transfer, and intentionally misaligned
condition remain controls. They make it possible to identify whether an
improvement comes from the vocal model, the alignment method, or accompaniment
variation.

## Comparison and reporting rules

- Use the same references and metric implementation for every compared
  condition.
- Preserve deterministic condition ordering and stable condition names.
- Report per-example values, aggregate estimates, sample counts, exclusions,
  and uncertainty intervals where appropriate.
- Keep objective dimensions separate rather than collapsing them into an
  unsupported universal quality score.
- Treat Griffin-Lim audio and UI playback as inspection aids, not perceptual
  evidence.
- Describe synthetic preference scores as engineering proxies, not human
  judgments.
- State limitations and failed conditions alongside successful outputs.

Human listening evaluation is a separate future protocol. It requires stable
playable systems, blinding, randomization, participant criteria, a power
rationale, stopping rules, and an analysis plan before responses are collected.

## Interpretation boundary

PJS contains one vocalist and is small. Results may support claims about the
behavior and reproducibility of these baselines on registered PJS conditions.
They do not support population-level preference, singer generalization,
production readiness, or authorized voice-imitation claims.
