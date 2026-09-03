# Evaluation Protocol

This document defines the evaluation contract for reported SingAlign results.
Exact metrics and decision thresholds will be registered before confirmatory
experiments begin.

## Evaluation dimensions

- pitch and note accuracy
- rhythm and duration accuracy
- lyric intelligibility
- singer similarity
- vocal naturalness and audio fidelity
- blinded pairwise human preference

## Split policy

Reported evaluations must name an immutable split manifest. Singer- and
song-disjoint partitions will be used whenever the claim concerns
generalization to unseen identities or musical content. Development results
must not be reported as held-out test results.

## Statistical reporting

Reports should include sample counts, effect sizes, confidence intervals, and
the statistical procedure used. Multiple comparisons must be identified and
handled explicitly. Both aggregate results and meaningful failure slices will
be reported.

## Metric governance

Each metric must document its implementation, version, input normalization,
directionality, and known limitations. No single automatic metric will be
treated as a substitute for perceptual evaluation.
