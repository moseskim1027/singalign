# Research Plan

## Primary hypothesis

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
- [x] Run held-out baseline evaluation once on the sealed test split
  (MLflow run `9610bc68f175431b96e99f9812ca3197`).
- [x] Run held-out aligned evaluation using the same test split and metrics
  (MLflow run `0aa5ffc003d84eaea82259b1f4e45e1d`).
- [x] Freeze the exploratory 3-second/10-epoch pilot manifest in
  [`experiments/pilot-3s-10e-manifest.yaml`](../experiments/pilot-3s-10e-manifest.yaml).
- [x] Draft the confirmatory analysis plan in
  [`experiments/analysis-plan-v1.md`](../experiments/analysis-plan-v1.md);
  external preregistration remains pending.

### Next research sequence

- [ ] Register confirmatory hypotheses, success thresholds, and exclusions
  before inspecting any future test or listener results.
- [ ] Add candidate generation and deterministic reward-based reranking.
- [ ] Define and implement scalar and multidimensional reward-model baselines.
- [ ] Add KTO as a separately tracked post-training condition.
- [ ] Compare all conditions with paired bootstrap intervals and effect sizes.
- [ ] Design and preregister a blinded listening study covering naturalness,
  pitch/rhythm, intelligibility, singer similarity, and expressiveness.
- [ ] Conduct the listening study, analyze failures, and report limitations.
- [ ] Package final manifests, reports, environment metadata, and reproduction
  instructions.
