# Research Plan

## Project framing

SingAlign is now an engineering and simulation sandbox for comparing music
generation methods. It does not require participant recruitment, confirmatory
hypothesis testing, or population-level human-preference claims. Metrics and
bootstrap intervals remain optional diagnostic tools for comparing reproducible
runs.

## Original research hypothesis

Preference-based post-training can improve perceived singing quality relative
to supervised fine-tuning and candidate reranking without materially degrading
score fidelity, lyric intelligibility, or singer similarity.

## Planned comparisons

- unmodified public baseline
- supervised fine-tuning
- reward-based candidate reranking
- Direct Preference Optimization (DPO)
- Kahneman-Tversky Optimization (KTO)

## Experimental factors

The study will vary preference-source composition, scalar versus
multidimensional rewards, post-training method, and preference-data scale.
Core ablations will isolate objective pseudo-labels, controlled degradations,
and human judgments.

## Success criteria

Success requires a statistically supported improvement in blinded human
preference without a practically meaningful regression in the registered
musical-accuracy, intelligibility, and identity-preservation measures.
Thresholds will be registered before confirmatory experiments begin.

## Threats to validity

Initial risks include preference-label shortcuts, reward hacking, dependence on
imperfect proxy metrics, limited listener diversity, dataset leakage, and weak
generalization beyond the language and vocal style represented by the data.

## Pilot post-training experiment

The first alignment pilot uses synthetic chosen/rejected pairs constructed by
controlled log-mel degradation. It tests whether the repository can reproduce
an energy-based DPO-style update relative to a frozen baseline while preserving
reconstruction through an anchor loss. It does not test human preference
alignment. Its purpose is to validate instrumentation and expose trade-offs
before collecting or modeling listener judgments.

## Paper-oriented methodology

The exploratory pilot validates the pipeline; it is not evidence for the
primary hypothesis. Paper-ready claims will use confirmatory runs whose
hypotheses, conditions, metrics, exclusions, and analysis thresholds are frozen
before the test split or listener responses are examined.

- Keep the song-disjoint split fingerprint immutable across comparisons.
- Use the same preprocessing, checkpoint-selection rule, and compute budget
  across conditions unless an ablation explicitly tests the difference.
- Separate exploratory, development, and confirmatory MLflow experiments.
- Select checkpoints using validation data only; evaluate the held-out test
  split once per registered condition and do not tune on test results.
- Report paired estimates, uncertainty intervals, effect sizes, sample counts,
  exclusions, and failed runs—not only the best point estimate.
- Treat synthetic preference labels and objective proxies as validation tools,
  not substitutes for human preference judgments.
- Preregister the blinded listening protocol, randomization, listener
  eligibility, power rationale, primary endpoint, stopping rule, and analysis.
- Maintain an experiment manifest linking config, code revision, data
  fingerprint, checkpoint hashes, Docker image, MLflow run, and report.
- Distinguish implementation validation from claims about quality,
  intelligibility, identity, or preference alignment.

## Execution status and next steps

This checklist is the working record for the current research cycle. Each
completed item should have a committed configuration, tracked MLflow run, and
reproducible report where applicable.

### Completed

- [x] Document PJS v1.1 provenance, licensing, and local-data conventions.
- [x] Validate/index/split pipeline with deterministic song-disjoint splits.
- [x] Implement the compact supervised log-mel reconstruction baseline.
- [x] Run the baseline for 10 epochs on 3-second segments in Docker.
- [x] Track parameters, metrics, artifacts, code revision, and data fingerprint
  in MLflow.
- [x] Implement the synthetic preference-pair generator and DPO-style proxy
  alignment with a reconstruction anchor.
- [x] Run proxy alignment for 10 epochs from the supervised baseline.
- [x] Generate paired validation comparisons and a local listening report.
- [x] Expose the comparison report in the Dockerized UI and link it to the
  corresponding MLflow evaluation run.
- [x] Define a deterministic, model-independent score/phoneme conditioning
  record and cover its parser with unit tests.
- [x] Add a Docker/CLI export path for inspecting conditioning records.
- [x] Include deterministic score-pitch metadata in conditioning exports.
- [x] Define deterministic frame-level alignment using explicit tempo and
  acoustic frame-rate parameters.
- [x] Add an untrained score/phoneme-conditioned mel-model interface with
  shape tests.
- [x] Freeze the initial score-conditioned baseline specification in
  `configs/training/conditioned.yaml`.
- [x] Add a fixed-shape tensor adapter with explicit pitch/rest and phoneme-ID
  conventions.
- [x] Add a deterministic dataset adapter pairing symbolic frames with mel
  targets using the same crop offset.
- [x] Add an exploratory Docker/MLflow training command for the conditioned mel
  model.
- [x] Run the conditioned model for the frozen exploratory 10-epoch budget in
  Docker; MLflow run `1fd53daa1f7e494abe16ceccf7daa3c1` is recorded in
  `singalign-score-conditioned-baseline`.
- [x] Support explicit crop offsets when aligning symbolic events to acoustic
  frames.
- [x] Run held-out baseline evaluation once on the sealed test split
  (MLflow run `9610bc68f175431b96e99f9812ca3197`).
- [x] Run held-out aligned evaluation using the same test split and metrics
  (MLflow run `0aa5ffc003d84eaea82259b1f4e45e1d`).
- [x] Freeze the exploratory 3-second/10-epoch pilot manifest in
  [`experiments/pilot-3s-10e-manifest.yaml`](../experiments/pilot-3s-10e-manifest.yaml).
- [x] Draft the confirmatory analysis plan in
  [`experiments/analysis-plan-v1.md`](../experiments/analysis-plan-v1.md);
  external preregistration remains pending.

### Next engineering sequence

- [x] Reframe the project from a confirmatory human-subjects study to a
  reproducible generation-method simulation sandbox.
- [x] Implement the first reproducible mel/waveform dataset adapter, vocoder
  configuration, and Docker training command; training the baseline remains
  an exploratory run.
- [x] Run the 10-epoch Docker vocoder pilot (MLflow run
  `421229b14e3043bfb3d89e3d6d2ca209`).
- [x] Add a held-out vocoder diagnostic command reporting waveform MSE and
  output peak level with the sealed split fingerprint.
- [x] Add candidate generation and deterministic reward-based reranking.
- [x] Define candidate-generation v1 with identity and controlled perturbation
  methods, deterministic seeds, and provenance records.
- [x] Implement deterministic proxy reranking using normalized mel error with
  stable tie-breaking; this is an engineering proxy, not a human-preference
  model.
- [x] Add scalar and multidimensional proxy-reward components with explicit
  weights and provenance.
- [x] Add deterministic candidate-report serialization, CLI execution, and
  optional MLflow artifact logging.
- [ ] Define and implement learned scalar and multidimensional reward-model
  baselines; current proxy rewards are complete but not learned reward models.
- [x] Add KTO as a separately tracked post-training condition.
- [x] Add and test a standalone KTO-style objective for proxy-score simulation.
- [x] Freeze exploratory DPO/KTO objective names, beta, KL baseline, and
  synthetic-preference provenance in `configs/training/preference-objectives.yaml`.
- [x] Add a tested chosen/rejected-to-KTO batch adapter shared by future
  preference trainers.
- [x] Freeze a separate exploratory KTO condition in `configs/training/kto.yaml`;
  trainer integration is complete.
- [x] Add a Docker/MLflow KTO trainer initialized from the supervised checkpoint;
  the trainer is exploratory and uses synthetic pairs.
- [x] Run the 10-epoch synthetic KTO pilot in Docker (MLflow run
  `3c9de2f603d2419683f6bfe2502fdc9d`).
- [x] Document the reproducible KTO Docker invocation and checkpoint lineage
  in the root README.
- [x] Add and run a sealed-test KTO diagnostic (10 examples; proxy accuracy
  `1.0`, mean proxy loss `0.3955`) with explicit evaluation opt-in.
- [x] Freeze the KTO held-out diagnostic settings in
  `configs/evaluation/kto.yaml`.
- [ ] Compare all conditions with paired bootstrap intervals and effect sizes.
- [x] Add a shared condition registry preserving stable names, methods,
  checkpoint paths, and declared order for future comparisons.
- [x] Add a generic multi-condition diagnostic engine using one metric contract
  and deterministic declared ordering.
- [x] Add a Docker/CLI multi-condition comparison command and freeze its
  exploratory metric/output configuration.
- [x] Add optional MLflow logging for the multi-condition report artifact.
- [x] Serialize multi-condition diagnostics as deterministic JSON artifacts with
  condition metadata preserved.
- [x] Add a stable MLflow artifact path for multi-condition reports.
- [ ] Add optional qualitative or informal listening feedback without making
  population-level claims.
- [ ] Package final manifests, reports, environment metadata, and reproduction
  instructions.

### Deferred UI work

The current UI is an unblinded reconstruction-inspection tool and should remain
stable during the score-conditioned model work. Revisit this section after
candidate generation exists:

- [ ] Add condition selection for baseline, reranking, DPO, and KTO.
- [x] Add a training interface with default parameters and a Docker-backed
  allowlisted job-launch API.
- [ ] Display multiple candidates with ranking scores and provenance.
- [ ] Display score/phoneme conditioning metadata alongside each example.
- [ ] Show cross-condition metrics, uncertainty intervals, and MLflow links.
- [ ] Build a separate blinded listening-study interface with randomized
  labels and no model-condition disclosure.

Do not use the current UI for perceptual claims: its model labels are visible,
and its generated audio uses approximate reconstruction.
