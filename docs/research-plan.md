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
