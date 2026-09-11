# Public MVP Data Protocol

This document freezes the first runnable data protocol for the Selective Acoustic Relevance / DAA experiments.

## Research unit

Each statistical unit is one **waveform pair** with the same SHA-256 waveform and two queries:

- `ignore`: answer the spoken MMLU multiple-choice problem;
- `use`: answer a question about the later acoustic event/source.

Only the query changes between the two records.

## Stage 1 — controlled environmental event (primary MVP)

### Target reasoning content

- Hugging Face dataset: `cais/mmlu`
- config: `all`
- development split: `validation`
- confirmation split: `test`
- only four-choice items are used
- items are filtered to at most 60 spoken words (question + choices) by default to control audio length

### Target speech

The MMLU question and all four choices are synthesized locally with eSpeak-NG:

```text
Question. <question> Choices. A. <choice A> B. <choice B> C. <choice C> D. <choice D>
```

Default TTS settings:

- engine: eSpeak-NG
- voice: `en-us`
- speed: 190
- output sample rate: 16 kHz mono

The synthetic target speech is deliberate for the first controlled gate: it removes a private-data dependency and keeps the reasoning labels exact. Real-speech replication is a later robustness check.

### Acoustic event

- Hugging Face dataset: `ashraq/esc50`
- development: folds 1, 2, 3
- confirmation: folds 4, 5
- public dataset size: 2,000 five-second clips / 50 classes
- first MVP restricts to distinctive classes (dog, siren, rain, footsteps, alarm, horn, train, airplane, etc.)

The event is resampled to 16 kHz and placed after the target speech with a 0.35 s gap by default. The event interval therefore has an exact temporal mask and can be mapped to DAA audio-token blocks without pretending to solve source separation.

### Default acoustic ratio

`ratio_db = 0 dB` for the first mechanism gate. After Gate A/B, rerun at stronger/lower ratios as a stress test; do not tune DAA on confirmation data.

## Stage 2 — real human competing-speech stress

- Hugging Face dataset: `openslr/librispeech_asr`
- config: `all`
- development: `validation.clean`
- confirmation: `test.clean`
- only short 2–10 word utterances are sampled for the first stress test

The real human utterance is appended after the MMLU question in the initial source-addressable condition. `q_use` asks which phrase the later speaker said; `q_ignore` asks the MMLU question.

This is not yet overlapping cocktail-party speech. Overlapping speech is a separate stress regime because temporal DAA blocks cannot honestly isolate two sources that occupy the same time span.

## Development / confirmation separation

| Source | Development | Confirmation |
|---|---|---|
| MMLU | validation | test |
| ESC-50 | folds 1/2/3 | folds 4/5 |
| LibriSpeech | validation.clean | test.clean |

The `confirm` protocol should remain sealed until block strategy, focus budget, prompts, scoring, and stopping criteria are frozen.

## One-command preparation

Install:

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
pip install -e '.[dev,data,gpu]'
```

Primary MVP:

```bash
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp \
  --num-pairs 128 \
  --protocol dev \
  --event-source esc50 \
  --seed 0
```

Real-human stress:

```bash
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp_librispeech \
  --num-pairs 128 \
  --protocol dev \
  --event-source librispeech \
  --seed 0
```

## Generated files

```text
data/mvp/source.jsonl
data/mvp/public_mvp_metadata.json
data/mvp/sources/target/*.wav
data/mvp/sources/esc50/*.wav   # or librispeech
data/mvp/audio/*.wav
data/mvp/pairs.jsonl
```

`pairs.jsonl` is the path already consumed by the default Base / fixed-DAA / declared-DAA configs.

## Experimental order

1. Prepare `dev + esc50`.
2. Run Base relevance-switch diagnostic.
3. Run fixed-block DAA.
4. Run declared-block DAA.
5. Only if the mechanism is promising, prepare/run the LibriSpeech stress condition.
6. Freeze all settings.
7. Run `confirm` once.

## What this protocol does not claim

- eSpeak-NG target speech is not a real-human speech robustness benchmark.
- time-separated events are not cocktail-party source separation.
- ESC-50 class labels are used only to build a controlled relevance-switch diagnostic.
- DAA performance on this MVP is a mechanism gate, not a final external-benchmark result.

## Licensing notes

Researchers should review the upstream licenses before redistribution. In particular, ESC-50 is distributed under a Creative Commons Attribution-NonCommercial license; LibriSpeech is distributed under CC BY 4.0. Generated data should not be committed to this repository.
