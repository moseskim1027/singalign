# Future diffusion voice-conversion placeholder

This file records the intended contract for a future GPU-backed experiment; it
is not a training recipe. A conditioned acoustic diffusion model would receive
source content/phonemes, target F0, score timing, and a singer/timbre embedding,
then generate mel frames for a compatible neural vocoder.

The experiment must use multi-singer, song-disjoint data, pinned seeds and
configs, MLflow lineage, and separate evaluation of content preservation,
target-pitch accuracy, timing, timbre, and artifacts. The current sandbox uses
the deterministic Study 2 transfer baseline instead.
