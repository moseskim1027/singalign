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

## Reconstruction baseline protocol

The initial mel-autoencoder is evaluated once after checkpoint selection. The
checkpoint is selected exclusively by validation loss; evaluation code then
loads only IDs listed in the immutable test partition. Each report records the
checkpoint SHA-256 and split fingerprint.

The initial objective metrics are mean squared error, mean absolute error, and
spectral convergence on per-segment normalized log-mel spectrograms. Reports
include per-example values, aggregate means, and deterministic percentile-
bootstrap 95% confidence intervals. These metrics measure reconstruction error
only and do not establish pitch accuracy, intelligibility, naturalness, or
human preference.

Inference latency is measured around the model forward pass after feature
extraction, with accelerator synchronization where applicable. Parameter count
and serialized checkpoint size are reported as efficiency descriptors.

## Paired model comparison

Development comparisons use identical validation examples for the baseline
and aligned checkpoints. Reports define delta as aligned minus baseline, so a
negative reconstruction-error delta is favorable. Paired bootstrap confidence
intervals and per-example win/tie/loss counts accompany aggregate means.

Test comparison requires an explicit command-line acknowledgement. Two test
examples used during pipeline smoke testing are considered inspected and may
not be described as pristine unseen observations in later confirmatory claims.

Listening artifacts are approximate Griffin-Lim inversions, not neural-vocoder
outputs. They support debugging and interface development but are not suitable
for claims about perceptual quality.

Comparison reports record the training segment duration embedded in each
checkpoint and the independently configured inference duration. Longer
listening clips are permitted for qualitative debugging because the baseline
architecture is fully convolutional, but results outside the training-window
duration must be labeled as such and must not be treated as evidence of
duration generalization.

Baseline training may override the versioned default with
`--segment-seconds`. The resolved value, rather than only the source-file
default, must be tracked and embedded in the checkpoint. Alignment inherits
that checkpoint duration, and comparison defaults to it unless an explicit
inference-only override is registered.

Listening windows are selected deterministically by maximum reference-audio
RMS on a fixed time grid. The selection rule cannot inspect baseline or aligned
outputs. Each manifest records the grid spacing and chosen offset so that the
audibility-oriented sampling remains reproducible and distinct from random or
cherry-picked evaluation.
