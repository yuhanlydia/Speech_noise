from __future__ import annotations

import io
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


LETTERS = ("A", "B", "C", "D")

# Restrict the first controlled experiment to acoustically distinctive classes.
ESC50_MVP_CATEGORIES = (
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "sheep",
    "crow",
    "rain",
    "thunderstorm",
    "crying_baby",
    "footsteps",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "siren",
    "car_horn",
    "train",
    "airplane",
    "fireworks",
)

DISPLAY_EVENT = {
    "dog": "dog barking",
    "rooster": "a rooster",
    "pig": "a pig",
    "cow": "a cow",
    "frog": "a frog",
    "cat": "a cat",
    "sheep": "a sheep",
    "crow": "a crow",
    "rain": "rain",
    "thunderstorm": "a thunderstorm",
    "crying_baby": "a crying baby",
    "footsteps": "footsteps",
    "door_wood_knock": "a knock on a wooden door",
    "clock_alarm": "an alarm clock",
    "glass_breaking": "breaking glass",
    "siren": "a siren",
    "car_horn": "a car horn",
    "train": "a train",
    "airplane": "an airplane",
    "fireworks": "fireworks",
}


@dataclass(frozen=True)
class PublicProtocol:
    mmlu_split: str
    esc50_folds: tuple[int, ...]
    librispeech_split: str


def protocol_spec(protocol: str) -> PublicProtocol:
    """Return question/audio-disjoint development vs confirmation sources."""
    if protocol == "dev":
        return PublicProtocol(
            mmlu_split="validation",
            esc50_folds=(1, 2, 3),
            librispeech_split="validation.clean",
        )
    if protocol == "confirm":
        return PublicProtocol(
            mmlu_split="test",
            esc50_folds=(4, 5),
            librispeech_split="test.clean",
        )
    raise ValueError("protocol must be 'dev' or 'confirm'")


def format_mmlu_spoken_prompt(question: str, choices: Sequence[str]) -> str:
    if len(choices) != 4:
        raise ValueError("MMLU MVP requires exactly four choices")
    pieces = [f"Question. {question.strip()}", "Choices."]
    pieces.extend(
        f"{letter}. {str(choice).strip()}"
        for letter, choice in zip(LETTERS, choices)
    )
    return " ".join(pieces)


def ignore_query() -> str:
    # Deliberately do not say "ignore", "irrelevant", or "background". The query
    # itself must define relevance; otherwise the benchmark leaks its own label.
    return (
        "Which option correctly answers the spoken multiple-choice question? "
        "Answer with A, B, C, or D only."
    )


def _choice_query(
    stem: str, semantic_choices: Sequence[str], correct_index: int
) -> tuple[str, str]:
    if len(semantic_choices) != 4 or not 0 <= correct_index < 4:
        raise ValueError(
            "semantic multiple choice requires four choices and a valid correct index"
        )
    rendered = " ".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, semantic_choices)
    )
    return (
        f"{stem} {rendered} Answer with A, B, C, or D only.",
        LETTERS[correct_index],
    )


def esc50_use_query(category: str, *, rng: random.Random) -> tuple[str, str]:
    if category not in DISPLAY_EVENT:
        raise ValueError(f"unsupported ESC-50 MVP category: {category}")
    distractors = [c for c in ESC50_MVP_CATEGORIES if c != category]
    picked = rng.sample(distractors, 3) + [category]
    rng.shuffle(picked)
    choices = [DISPLAY_EVENT[c] for c in picked]
    return _choice_query(
        "Which of the following sounds can be heard in the recording?",
        choices,
        picked.index(category),
    )


def librispeech_use_query(
    transcript: str,
    distractor_transcripts: Sequence[str],
    *,
    rng: random.Random,
) -> tuple[str, str]:
    pool = [
        str(x).strip()
        for x in distractor_transcripts
        if str(x).strip() and str(x).strip() != transcript.strip()
    ]
    if len(pool) < 3:
        raise ValueError("need at least three distinct LibriSpeech distractor transcripts")
    choices = rng.sample(pool, 3) + [transcript.strip()]
    rng.shuffle(choices)
    return _choice_query(
        "Which of the following phrases can be heard in the recording?",
        choices,
        choices.index(transcript.strip()),
    )


def resample_linear(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("public MVP expects mono audio")
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=True)
    new_len = max(1, int(round(x.size * float(target_sr) / float(orig_sr))))
    old_pos = np.linspace(
        0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64
    )
    new_pos = np.linspace(
        0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64
    )
    return np.interp(new_pos, old_pos, x).astype(np.float32)


def decode_hf_audio_ref(audio_ref) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face Audio(decode=False) row with soundfile only."""
    if not isinstance(audio_ref, dict):
        raise TypeError("expected Hugging Face Audio(decode=False) dict")
    payload = audio_ref.get("bytes")
    path = audio_ref.get("path")
    if payload is not None:
        data, sr = sf.read(io.BytesIO(payload), dtype="float32")
    elif path:
        data, sr = sf.read(path, dtype="float32")
    else:
        raise ValueError("audio reference contains neither bytes nor path")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1:
        raise ValueError("decoded audio must be mono or stereo")
    return np.asarray(data, dtype=np.float32), int(sr)


def synthesize_espeak(
    text: str,
    output_wav: str | Path,
    *,
    sample_rate: int = 16000,
    voice: str = "en-us",
    speed: int = 190,
    executable: str = "espeak-ng",
) -> Path:
    exe = shutil.which(executable)
    if exe is None:
        raise RuntimeError(
            "espeak-ng is required for the public MVP. On Ubuntu run: "
            "sudo apt-get update && sudo apt-get install -y espeak-ng"
        )
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_wav = Path(tmp) / "tts.wav"
        subprocess.run(
            [
                exe,
                "-v",
                voice,
                "-s",
                str(int(speed)),
                "-w",
                str(raw_wav),
                text,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        audio, sr = sf.read(raw_wav, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = resample_linear(
        np.asarray(audio, dtype=np.float32), int(sr), sample_rate
    )
    sf.write(output_wav, audio, sample_rate, subtype="PCM_16")
    return output_wav


def _word_count(question: str, choices: Sequence[str]) -> int:
    return len((question + " " + " ".join(map(str, choices))).split())


def _load_datasets_api():
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:  # pragma: no cover - optional public-data dependency
        raise RuntimeError(
            "Install the public-data dependencies with: pip install -e '.[data]'"
        ) from exc
    return Audio, load_dataset


def load_mmlu_rows(
    *, protocol: str, num_pairs: int, seed: int, max_spoken_words: int
) -> list[dict]:
    _, load_dataset = _load_datasets_api()
    spec = protocol_spec(protocol)
    ds = load_dataset("cais/mmlu", "all", split=spec.mmlu_split)
    candidates = [
        dict(row)
        for row in ds
        if len(row["choices"]) == 4
        and _word_count(str(row["question"]), row["choices"]) <= max_spoken_words
    ]
    if len(candidates) < num_pairs:
        raise ValueError(
            f"only {len(candidates)} MMLU rows satisfy "
            f"max_spoken_words={max_spoken_words}; requested {num_pairs}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:num_pairs]


def load_esc50_rows(*, protocol: str, num_pairs: int, seed: int) -> list[dict]:
    Audio, load_dataset = _load_datasets_api()
    spec = protocol_spec(protocol)
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    candidates = [
        dict(row)
        for row in ds
        if int(row["fold"]) in spec.esc50_folds
        and str(row["category"]) in ESC50_MVP_CATEGORIES
    ]
    if len(candidates) < num_pairs:
        raise ValueError(
            f"only {len(candidates)} eligible ESC-50 rows; requested {num_pairs}"
        )
    rng = random.Random(seed + 17)
    rng.shuffle(candidates)
    return candidates[:num_pairs]


def load_librispeech_rows(
    *, protocol: str, num_pairs: int, seed: int
) -> list[dict]:
    Audio, load_dataset = _load_datasets_api()
    spec = protocol_spec(protocol)
    ds = load_dataset(
        "openslr/librispeech_asr", "all", split=spec.librispeech_split
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    candidates = []
    for row in ds:
        words = str(row["text"]).split()
        if 2 <= len(words) <= 10:
            candidates.append(dict(row))
    if len(candidates) < num_pairs + 3:
        raise ValueError(
            f"only {len(candidates)} eligible LibriSpeech rows; requested {num_pairs}"
        )
    rng = random.Random(seed + 29)
    rng.shuffle(candidates)
    return candidates[: max(num_pairs + 3, 8)]


def _write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")


def prepare_source_manifest(
    output_dir: str | Path,
    *,
    num_pairs: int = 128,
    protocol: str = "dev",
    event_source: str = "esc50",
    seed: int = 0,
    sample_rate: int = 16000,
    gap_s: float = 0.35,
    ratio_db: float = 0.0,
    max_spoken_words: int = 60,
    tts_voice: str = "en-us",
    tts_speed: int = 190,
) -> Path:
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    if event_source not in {"esc50", "librispeech"}:
        raise ValueError("event_source must be 'esc50' or 'librispeech'")
    if gap_s < 0:
        raise ValueError("gap_s must be non-negative")

    output_dir = Path(output_dir)
    source_dir = output_dir / "sources"
    target_dir = source_dir / "target"
    event_dir = source_dir / event_source
    target_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    mmlu_rows = load_mmlu_rows(
        protocol=protocol,
        num_pairs=num_pairs,
        seed=seed,
        max_spoken_words=max_spoken_words,
    )
    if event_source == "esc50":
        event_rows = load_esc50_rows(
            protocol=protocol, num_pairs=num_pairs, seed=seed
        )
    else:
        event_rows = load_librispeech_rows(
            protocol=protocol, num_pairs=num_pairs, seed=seed
        )

    rng = random.Random(seed)
    source_manifest = output_dir / "source.jsonl"
    with source_manifest.open("w", encoding="utf-8") as f:
        for idx, mmlu in enumerate(mmlu_rows):
            pair_id = f"{protocol}-{event_source}-{idx:04d}"
            target_path = target_dir / f"{pair_id}.wav"
            spoken = format_mmlu_spoken_prompt(
                str(mmlu["question"]), list(mmlu["choices"])
            )
            synthesize_espeak(
                spoken,
                target_path,
                sample_rate=sample_rate,
                voice=tts_voice,
                speed=tts_speed,
            )
            target_audio, target_sr = sf.read(target_path, dtype="float32")
            if int(target_sr) != sample_rate:
                raise RuntimeError("TTS target sample-rate normalization failed")

            event_row = event_rows[idx]
            event_audio, event_sr = decode_hf_audio_ref(event_row["audio"])
            event_audio = resample_linear(event_audio, event_sr, sample_rate)
            event_path = event_dir / f"{pair_id}.wav"
            _write_audio(event_path, event_audio, sample_rate)

            if event_source == "esc50":
                use_q, use_a = esc50_use_query(
                    str(event_row["category"]), rng=rng
                )
                event_type = str(event_row["category"])
                source_id = str(event_row.get("filename", pair_id))
            else:
                other_text = [
                    str(r["text"])
                    for j, r in enumerate(event_rows)
                    if j != idx
                ]
                use_q, use_a = librispeech_use_query(
                    str(event_row["text"]), other_text, rng=rng
                )
                event_type = "background_speech"
                source_id = str(event_row.get("id", pair_id))

            answer_idx = int(mmlu["answer"])
            offset_samples = int(
                len(target_audio) + round(gap_s * sample_rate)
            )
            row = {
                "pair_id": pair_id,
                "target_wav": str(target_path.resolve()),
                "event_wav": str(event_path.resolve()),
                "event_type": event_type,
                "source_id": source_id,
                "source_mask_valid": True,
                "offset_samples": offset_samples,
                "q_ignore": ignore_query(),
                "a_ignore": LETTERS[answer_idx],
                "options_ignore": list(LETTERS),
                "q_use": use_q,
                "a_use": use_a,
                "options_use": list(LETTERS),
                "seed": seed,
            }
            if event_source == "esc50":
                row["snr_db"] = float(ratio_db)
            else:
                row["tir_db"] = float(ratio_db)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    spec = protocol_spec(protocol)
    metadata = {
        "protocol": protocol,
        "event_source": event_source,
        "num_pairs": num_pairs,
        "seed": seed,
        "sample_rate": sample_rate,
        "gap_s": gap_s,
        "ratio_db": ratio_db,
        "max_spoken_words": max_spoken_words,
        "target_questions": {
            "dataset": "cais/mmlu",
            "config": "all",
            "split": spec.mmlu_split,
        },
        "event_dataset": (
            {
                "dataset": "ashraq/esc50",
                "split": "train",
                "folds": list(spec.esc50_folds),
            }
            if event_source == "esc50"
            else {
                "dataset": "openslr/librispeech_asr",
                "config": "all",
                "split": spec.librispeech_split,
            }
        ),
        "tts": {
            "engine": "espeak-ng",
            "voice": tts_voice,
            "speed": tts_speed,
        },
        "prompt_policy": "query defines relevance; no ignore/irrelevant label leakage",
    }
    (output_dir / "public_mvp_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return source_manifest
