# Data

SingAlign uses the PJS phoneme-balanced Japanese singing voice corpus for its
initial low-resource experiments. Dataset files are stored locally and are
never committed to this repository.

## PJS corpus

PJS version 1.1 contains 100 short singing recordings and their spoken
counterparts from one Japanese vocalist. Each example includes a MIDI score,
a MusicXML score, phoneme labels, and descriptive metadata. The download is
approximately 0.26 GB.

The corpus is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Review the
terms distributed with the corpus before using it. Publications and released
adaptations must provide appropriate attribution, and adapted material is
subject to the license's ShareAlike requirement.

## Download

1. Visit the
   [official PJS corpus page](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus).
2. Follow the Google Drive link labeled `ver.1.1`.
3. Download the approximately 0.26 GB ZIP archive.
4. From the repository root, create the ignored destination directory:

   ```bash
   mkdir -p data/raw/pjs
   ```

5. Extract or move the corpus into that directory. The result should resemble:

   ```text
   data/raw/pjs/
   └── PJS_corpus_ver1.1/
       ├── background_noise/
       ├── pjs001/
       │   ├── pjs001_song.wav
       │   ├── pjs001_speech.wav
       │   ├── pjs001.lab
       │   ├── pjs001.mid
       │   ├── pjs001.musicxml
       │   └── pjs001.txt
       └── ...
   ```

The downloaded release determines the authoritative directory and filename
capitalization. If it differs from the illustrative layout above, preserve the
release layout rather than renaming files.

Keep the downloaded archive outside the repository or remove it after a
verified extraction. Do not commit the archive, extracted audio, annotations,
or locally processed derivatives.

## Verify the installation

Run these non-destructive checks from the repository root:

```bash
test -d data/raw/pjs/PJS_corpus_ver1.1
find data/raw/pjs/PJS_corpus_ver1.1 -name '*_song.wav' | wc -l
find data/raw/pjs/PJS_corpus_ver1.1 -name '*_speech.wav' | wc -l
find data/raw/pjs/PJS_corpus_ver1.1 -name '*.mid' | wc -l
find data/raw/pjs/PJS_corpus_ver1.1 \
  \( -name '*.musicxml' -o -name '*.xml' \) | wc -l
```

The principal recording and score categories are expected to contain 100
examples each. Confirm any discrepancy against the included release notes
before changing the data.

## Data handling

Raw corpus files are immutable inputs. Future preprocessing will write to
separate ignored directories:

```text
data/
├── raw/         Immutable source files
├── interim/     Validated intermediate representations
├── processed/   Model-ready examples
└── manifests/   Versioned provenance and split metadata
```

Record the corpus version, retrieval date, source URL, included license, and
archive checksum in the future provenance manifest. Every transformation must
also record its parameters and source revision so processed data can be
reproduced locally.

Only provenance and split manifests that do not expose corpus content may be
committed.

## Citation

Research using PJS should cite:

> Junya Koguchi and Shinnosuke Takamichi. “PJS: Phoneme-balanced Japanese
> singing voice corpus.” arXiv:2006.02959, 2020.

See the [PJS paper](https://arxiv.org/abs/2006.02959) for the corpus design and
the official corpus page for current citation guidance.
