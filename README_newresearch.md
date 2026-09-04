# New Research Directions: Singing Synthesis and Content-and-Melody Transfer

SingAlign should focus on two related but distinct experiments using the PJS
phoneme-balanced Japanese singing voice corpus. PJS contains approximately 100
recordings from one vocalist, so it supports controlled same-singer synthesis
and vocal-content transfer, but not general singer-to-singer identity transfer.

## Study 1: Same-singer score-conditioned synthesis

### Research question

Can a model reconstruct the PJS vocalist from lyrics, phonemes, musical score,
timing, and pitch conditioning?

### Current status

SingAlign already has PJS indexing and deterministic splits, MusicXML and
phoneme parsing, score/phoneme conditioning records, fixed-shape tensors, a
conditioned mel-model interface, an exploratory trainer, and a vocoder
diagnostic. This is a foundation, not yet a high-quality singing synthesizer.

### Input and output

```text
lyrics + phonemes + MIDI/MusicXML + note durations + pitch/F0
→ score-conditioned acoustic model → mel spectrogram
→ waveform decoder/vocoder → reconstructed PJS singing voice
```

Use symbolic MIDI pitch and score timing as the first inputs. Log observed F0
from the reference WAV as a diagnostic and later compare it as an explicit
conditioning or teacher signal. Keep intended score pitch separate from
observed performance pitch.

### Remaining work

- Extract and validate frame-level observed F0 from PJS singing WAV files.
- Define phoneme-to-note mapping for rests, held vowels, consonants, melismas,
  and notes with multiple phonemes.
- Compare symbolic-pitch conditioning against observed-F0 conditioning.
- Replace or isolate the approximate waveform reconstruction with a declared
  neural vocoder or reproducible DSP baseline.
- Evaluate lyric/phoneme correctness, pitch, timing, spectral quality, and
  waveform artifacts on the sealed test split.
- Keep the one-vocalist limitation explicit: this tests PJS reconstruction,
  not unseen-singer generalization.

## Study 2: Singing content-and-melody transfer

### Research question

Can SingAlign preserve the words, phoneme timing, and sung melody of a source
PJS vocal while placing that vocal performance over a different instrumental
track?

This is not singer-identity conversion. The initial study keeps the source
vocalist unchanged and tests content, melody, timing, and musical fit.

### Core pipeline

```text
source PJS vocal → phonemes + timestamps + score pitch + observed F0 + energy
→ aligned source vocal representation → existing target instrumental → mix
```

MusicGen is not required for this core experiment. A generated instrumental
would introduce a second uncontrolled variable. MusicGen may later be tested
as a separate target-instrument condition with model, prompt, seed, duration,
and artifact recorded in MLflow.

### Target instrumentals

PJS does not provide clean instrumental stems for every recording. Begin with
reproducible MIDI rendering:

```text
target PJS MIDI/MusicXML → fixed soundfont/synthesizer → target instrumental
```

This provides known notes, tempo, beat positions, and key. Retain source-vocal
separation as an optional artifact-prone remix baseline, not clean ground
truth.

### Initial conditions

For fixed, song-disjoint source/target pairs, compare:

1. Original source vocal over its original accompaniment.
2. Original vocal over a target MIDI-rendered accompaniment.
3. Time-stretched and/or transposed source vocal over the target.
4. A deliberately misaligned control with incorrect timing or pitch.
5. Later, a synthesized same-singer vocal conditioned on source content and
   melody over the target accompaniment.

Start with 10 diagnostic pairs, one unchanged-tempo/key condition, and one
declared tempo-shift or transposition condition. Do not select examples based
on post-hoc listening impressions.

### Required representations and metrics

Record Japanese lyrics, normalized phonemes, word/phoneme timestamps, MIDI
onset/duration/pitch/rests, observed F0, voiced mask, confidence, optional
energy, source/target BPM, beat grid, time signature, key, stretch, and
transposition. Report phoneme or lyric error rate, voiced-F0 RMSE/correlation,
note-level pitch deviation, timing deviation, vocal activity, clipping, and
stem/final-mix diagnostics. Keep lyric intelligibility, melody preservation,
and accompaniment fit as separate metrics.

## Relationship, methodology, and boundaries

Study 1 asks whether SingAlign can synthesize the PJS vocalist from structured
conditioning. Study 2 asks whether that conditioning preserves source content
and melody when the musical context changes. Study 1 should be stabilized
first, although Study 2's original-vocal remix baseline can begin independently.

- Use the existing deterministic PJS 80/10/10 song-disjoint split.
- Keep raw PJS files local and immutable; commit only metadata and manifests.
- Run preprocessing, training, and evaluation in Docker on Apple Silicon.
- Track dataset version, split fingerprint, code revision, config, checkpoint
  hash, audio settings, and MLflow run for every condition.
- Treat first runs as exploratory engineering diagnostics.
- Do not make unseen-singer or participant-preference claims.
- Do not combine MusicGen quality with vocal-transfer quality in one primary
  comparison.

## Recommended implementation order

1. Add observed-F0 extraction and validation to conditioning records.
2. Complete the same-singer synthesis baseline and compare vocoders.
3. Add deterministic MIDI/MusicXML target-instrument rendering.
4. Add source/target pair manifests and tempo/key alignment utilities.
5. Implement original-vocal remix and misalignment controls.
6. Add synthesized-vocal transfer using the Study 1 checkpoint.
7. Add metrics, MLflow artifacts, and reproducible comparison reports.
8. Evaluate MusicGen only as a separately tracked accompaniment generator.

## Continuation prompt

“Continue the work described in `/Users/moseskim/Portfolio/singalign/README_newresearch.md`.
Focus on Study 1 and Study 2 using PJS only. Inspect existing conditioning
records, PJS adapters, Docker services, configs, tests, MLflow helpers, and the
research checklist. Update the root README and this file before each task.
Begin with observed-F0 extraction and validation, then complete the same-singer
synthesis baseline. Next implement MIDI-based target-instrument rendering and
the original-vocal content-and-melody remix baseline. Keep MusicGen optional and
separate. Use a feature branch and PR, run checks in Docker on Apple Silicon,
preserve the song-disjoint split, and record experiments with manifests,
checkpoint hashes, and MLflow metadata. Do not add participant studies or make
unseen-singer claims.”
