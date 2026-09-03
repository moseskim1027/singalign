# SingAlign

Preference-aligned singing voice synthesis.

> [!IMPORTANT]
> SingAlign is an early-stage research project. The methods, experiments, and
> conclusions described here are provisional and should not be treated as
> production-ready.

## Overview

SingAlign studies whether preference-based post-training can improve singing
voice synthesis without sacrificing musical accuracy, lyric intelligibility,
or singer identity. Its initial experiments focus on low-resource preference
alignment: methods that can be investigated reproducibly without training a
large generative audio model from scratch.

The project will evaluate multidimensional reward modeling and preference
optimization methods for generative singing systems. Initial experiments will
compare supervised fine-tuning, candidate reranking, Direct Preference
Optimization (DPO), and Kahneman-Tversky Optimization (KTO).

This repository is structured as a reproducible research artifact. It will
contain experiment definitions, evaluation protocols, statistical analyses,
and research documentation alongside the eventual implementation.

## Research questions

SingAlign is organized around four primary questions:

1. Can a multidimensional reward model predict human preferences for generated
   singing?
2. Do preference-optimization methods improve perceived singing quality over
   supervised fine-tuning and candidate reranking?
3. What trade-offs arise between perceptual quality, score fidelity, lyric
   intelligibility, and singer similarity?
4. Do learned preferences generalize to unseen singers, songs, and generation
   conditions?

## Scope

The initial study will focus on compact, score-conditioned singing voice
synthesis using the PJS corpus. Experiments will operate on short
mel-spectrogram segments so that data preparation, baseline development, and
pilot studies remain practical on Apple Silicon. The study will address the
following perceptual dimensions:

- vocal naturalness
- pitch and rhythm accuracy
- lyric intelligibility
- singer similarity
- expressiveness
- audio fidelity

The first version will intentionally favor controlled, interpretable
experiments over model scale.

## Planned methodology

The planned research pipeline consists of:

1. Reproduce or adapt a public singing voice synthesis baseline.
2. Generate multiple candidates for each lyric and score condition.
3. Construct preference pairs using controlled degradations, objective
   measurements, and a limited human listening study.
4. Train scalar and multidimensional reward-model baselines.
5. Apply reward-based reranking and preference optimization.
6. Evaluate models using objective measurements and blinded human judgments.
7. Report uncertainty, failure cases, and relevant ablations.

The methodology may change as preliminary experiments reveal limitations. Any
material changes will be documented in the research plan and experiment logs.

## Dataset plan

The initial dataset is the PJS phoneme-balanced Japanese singing voice corpus.
PJS is an approximately 0.26 GB public dataset containing 100 short singing
recordings, their spoken counterparts, MIDI and MusicXML scores, phoneme
labels, and supporting metadata. This compact, paired design supports
score-conditioned modeling and low-resource preference experiments on a local
M1 machine.

Download PJS version 1.1 from the
[official corpus page](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus).
The corpus is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), which requires
attribution and ShareAlike distribution of adapted material.

Before training begins, the project will document:

- the dataset version and retrieval procedure
- applicable access conditions and licensing terms
- allowed uses and redistribution restrictions
- singer- and song-level split construction
- known demographic, linguistic, and recording limitations

Dataset files will not be committed to this repository. The raw corpus must
remain unchanged in the ignored local data directory, and any distributed
adaptations must comply with the corpus license.

See [`data/README.md`](data/README.md) for download, installation, verification,
and provenance instructions.

## Evaluation plan

Evaluation will combine objective and perceptual evidence.

| Dimension | Candidate measurements |
| --- | --- |
| Pitch accuracy | F0 error, voiced/unvoiced error, note-level deviation |
| Rhythm accuracy | onset and duration deviation |
| Intelligibility | ASR error rate and human lyric recognition |
| Singer similarity | embedding similarity and human judgments |
| Audio quality | learned quality estimators and artifact analysis |
| Preference | blinded pairwise human comparisons |

Reported experiments will include confidence intervals, effect sizes, and
statistical tests where appropriate. Metrics will be treated as imperfect
proxies rather than interchangeable substitutes for human judgments.

The detailed protocol will live in
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Repository structure

```text
configs/       Versioned experiment and model configurations
data/          Dataset documentation and local data conventions
docs/          Research questions, protocols, and responsible-use analysis
experiments/   Experiment manifests and reproducibility records
reports/       Generated tables, figures, and research reports
src/           Future implementation of models and research utilities
```

This initial repository contains documentation only. Source code, model
configuration files, and dataset tooling will be introduced through separate
reviewed changes.

## Reproducibility principles

The project will follow these practices:

- version all reported experiment configurations
- record random seeds, software versions, and hardware assumptions
- keep evaluation splits immutable after they are registered
- identify every reported result with an experiment manifest
- distinguish exploratory results from confirmatory results
- preserve failed experiments when they inform a conclusion
- report uncertainty instead of relying only on point estimates

## Responsible research

Singing voice generation creates risks involving consent, impersonation,
copyright, and misleading synthetic media. SingAlign will therefore:

- use only datasets with documented research permissions
- avoid presenting generated voices as real performances
- disclose synthetic audio in demonstrations
- document model and dataset limitations
- avoid releasing tools intended for unauthorized voice impersonation
- preserve dataset-specific attribution and usage restrictions

See [`docs/responsible-research.md`](docs/responsible-research.md) for the
evolving risk assessment.

## Roadmap

- [ ] Finalize the research hypotheses and success criteria
- [ ] Review dataset terms and create immutable data splits
- [ ] Select and reproduce a baseline model
- [ ] Implement controlled preference-pair construction
- [ ] Establish objective evaluation baselines
- [ ] Train and evaluate reward models
- [ ] Compare reranking, supervised fine-tuning, DPO, and KTO
- [ ] Conduct a blinded listening study
- [ ] Publish the final report and reproducibility package

The roadmap indicates intended work, not completed capabilities.

## Contributing

Research contributions should state the hypothesis being tested, describe the
experimental controls, and include a reproducible evaluation plan. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a change.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until the
project has a formal release, cite the repository and the exact commit used.

## License

Repository code and original documentation are licensed under the
[Apache License 2.0](LICENSE).

Datasets, pretrained models, third-party implementations, and generated
artifacts may be governed by separate terms. The Apache-2.0 license does not
override those terms.
