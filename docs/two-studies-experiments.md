# Two-study experiment protocol

The two studies use the PJS corpus only and share the deterministic 80/10/10
song-disjoint split. Every run must use `configs/training/studies.yaml`, record
the split fingerprint, and be launched in the `research` Docker service with
`MLFLOW_TRACKING_URI=http://mlflow:5000`.

Study 1 is the same-singer score-conditioned synthesis baseline. Export a
versioned conditioning record containing MusicXML notes, phoneme intervals,
and frame-level observed F0. F0 records retain unvoiced frames as `null` and
include confidence; missing pitch must not be silently interpolated.

Study 2 is a controlled content-and-melody transfer/remix experiment. Use a
versioned, song-disjoint source/target pair manifest. The initial control uses
the original source vocal and a deterministic MIDI/MusicXML-rendered target
instrumental. Tempo scaling and transposition are explicit manifest fields.
MusicGen is a separate optional condition, never part of the primary control.

Runs are exploratory until the extraction, pair manifest, and baseline checks
are complete. MLflow tags and parameters must include dataset/version, split
fingerprint, Git revision and dirty state, seed, configuration, pair identity,
tempo scale, and transposition. Do not generalize beyond the single PJS singer.
