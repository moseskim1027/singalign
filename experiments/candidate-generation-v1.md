# Candidate Generation and Reranking v1

This sandbox stage produces several deterministic mel candidates from the same
reference input, scores them with configurable proxy rewards, and preserves all
candidate provenance. It is an engineering comparison, not a human-preference
claim.

Each candidate records its source method, seed, degradation parameters, reward
components, and selected rank. Candidate generation must never alter the
song-disjoint split or inspect held-out labels while selecting a candidate.

Initial candidate methods:

- identity/reference mel input;
- deterministic noise, time-mask, and frequency-mask perturbations;
- future learned decoder outputs, once a trained decoder is available.

The first reranker will use weighted normalized reconstruction error as a
diagnostic proxy. Scores are explicitly labeled proxy rewards and are not
treated as substitutes for listener judgments.
