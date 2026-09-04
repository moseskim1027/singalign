# SingAlign Analysis Plan v1

Status: draft for review; no confirmatory claims are authorized until this
plan is externally preregistered and the registration timestamp is recorded in
the experiment manifest.

## Scope

This plan covers the first controlled comparison of the supervised baseline and
the proxy-aligned checkpoint. The current autoencoder reconstructs an
audio-derived log-mel input; it is not yet a score-conditioned singing
synthesizer. Therefore, this study evaluates reconstruction and pipeline
behavior only. It does not claim improved singing quality or human preference
alignment.

## Confirmatory question

For the same held-out PJS examples and preprocessing, does the aligned policy
change reconstruction error relative to the frozen baseline without a
practically meaningful degradation in reconstruction quality?

## Conditions

- Baseline: validation-selected supervised checkpoint.
- Aligned: checkpoint initialized from the baseline and trained with the
  frozen-baseline DPO-style proxy objective plus reconstruction anchor.
- Both conditions use the same 3-second audio window, model architecture,
  test IDs, and deterministic preprocessing.

## Primary endpoint

The primary endpoint is the paired per-example difference in normalized
log-mel MSE, defined as aligned minus baseline. Lower values are better. The
primary summary is the mean paired difference with a two-sided 95% bootstrap
confidence interval using the registered split and seed.

## Secondary endpoints

- normalized log-mel MAE
- spectral convergence
- inference latency
- per-example win/tie/loss counts

These are secondary diagnostic endpoints and will not replace the primary
endpoint after results are inspected.

## Decision rules

For a future confirmatory claim of improvement, the primary interval must lie
below zero and the mean improvement must be at least 2% of the baseline mean.
No secondary reconstruction metric may show a relative degradation greater than
5%. If these criteria are not met, report the result as no demonstrated
improvement. These thresholds are fixed before any future confirmatory test
run.

## Exclusions and analysis rules

- Do not exclude examples based on metric values or model outputs.
- Exclude only missing/corrupt input files identified before model inference;
  record the IDs and reason in the manifest.
- Do not tune checkpoints, hyperparameters, seeds, or preprocessing using test
  results.
- Report all registered conditions and failed runs.
- Use validation data for checkpoint selection and development decisions.
- Keep the test split sealed until the confirmatory command is run.

## Human evaluation boundary

No human-preference conclusion can be drawn from this plan. A later listening
study must separately register listener eligibility, randomization, blinding,
power rationale, primary perceptual endpoint, stopping rule, and analysis
before collecting responses.
