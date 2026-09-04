# Future track-transfer study: phoneme and pitch alignment

## Objective

Test whether SingAlign can preserve the linguistic content and sung melody of
a source vocal while placing the performance over a different instrumental
track. The target may differ in genre, instrumentation, tempo, key, and
production style. This is a controllable singing-transfer/remixing experiment,
not a participant study and not a claim that MusicGen can sing intelligible
lyrics.

## Working formulation

`source vocal -> phonemes + timing + F0 + optional energy -> target vocal -> target instrumental -> mix`

The source vocal supplies what is sung, when it is sung, and the melody. The
target instrumental supplies the new musical context. Keep these conditions
separate in every manifest and MLflow run.

## Required inputs and representations

- Source vocal WAV, preferably isolated from accompaniment.
- Source lyrics, language, and normalization policy.
- Phoneme sequence with word/phoneme timestamps.
- Frame-level F0 in Hz, voiced/unvoiced mask, and confidence.
- Optional energy envelope and breath/consonant regions.
- Target instrumental WAV, BPM, beat locations, time signature, and key.
- Target singer/voice identifier or declared voice-conversion checkpoint.
- Deterministic policy for tempo stretching and pitch transposition.

## Staged implementation instructions

1. Build an extraction-only CLI that writes a versioned conditioning record
   containing normalized phonemes, timestamps, F0, voiced mask, and confidence.
   Store tool versions and hashes; do not silently interpolate missing pitch.
2. Validate and visualize records: intervals must be ordered/non-overlapping,
   voiced F0 must be finite and positive, and duration must match audio within
   a documented tolerance.
3. Implement target beat alignment. Start with a fixed policy: preserve source
   note timing, resample the target instrumental to source BPM, and optionally
   transpose the vocal by declared semitones.
4. Establish a non-neural control using the original vocal with separation,
   time-stretch, pitch-shift, and mixing. This is the remix baseline.
5. Add a pretrained singing voice conversion or score-based SVS adapter using
   phonemes, durations, F0, and a target voice reference. Log checkpoint and
   license metadata.
6. Mix generated vocal and target instrumental with fixed loudness, sample
   rate, peak, and limiter settings. Export stems and final mixes.
7. Evaluate phoneme error rate, voiced-F0 RMSE/correlation, timing deviation,
   vocal activity, clipping, and stem/mix diagnostics.
8. Run ablations removing phoneme conditioning, F0, beat alignment, and
   singer conversion; vary tempo shift and transposition as separate factors.

## Evaluation design

Use song-disjoint examples and fixed source/target pair manifests. The first
diagnostic milestone is 10 pairs, one voice checkpoint, one unchanged-key/
tempo condition, and one transposed/tempo-shifted condition. Compare the
remix baseline, conversion output, and an intentionally misaligned control.
Report per-example values and aggregate means. Keep lyric intelligibility and
musical fit as separate axes.

## Deliberate initial exclusions

- Fully automatic lyric-to-new-melody composition.
- Joint end-to-end lyrics, vocals, and accompaniment generation.
- Human-preference or singer-identity claims without a dedicated evaluation.
- Training a new singing model before extraction and control baselines work.

## Continuation prompt for a new conversation

“Continue the SingAlign research plan section ‘Future track-transfer study:
phoneme and pitch alignment.’ Work in `/Users/moseskim/Portfolio/singalign`.
Inspect existing conditioning schemas, audio utilities, Docker services,
configs, tests, and MLflow conventions. Update the root README and this plan
before each task. Implement the extraction-only CLI and conditioning-record
schema first, with unit tests and a deterministic fixture. Do not add a
participant listening study. Use a feature branch and PR, run tests in Docker
on Apple Silicon, and log experiments with source/target manifests, checkpoint
hashes, and alignment settings.”
