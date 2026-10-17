"""Prepare MFA transcript sidecars and corpus vocabulary from normalized RUSLAN text."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from g2p_utils import extract_words, iter_metadata, normalize_word


def prepare_corpus(
    metadata_path: Path,
    audio_dir: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, int]:
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    vocabulary_path = output_dir / "vocabulary.txt"

    vocabulary: Counter[str] = Counter()
    utterance_count = 0
    word_count = 0
    missing_audio: list[str] = []
    empty_text: list[str] = []

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["utterance_id", "audio_path", "word_count", "transcript"])
        for utterance_id, text in iter_metadata(metadata_path):
            if limit is not None and utterance_count >= limit:
                break
            words = [normalize_word(word) for word in extract_words(text)]
            words = [word for word in words if word]
            if not words:
                empty_text.append(utterance_id)
                continue
            audio_path = audio_dir / f"{utterance_id}.wav"
            if not audio_path.is_file():
                missing_audio.append(utterance_id)
                continue

            transcript = " ".join(words)
            (transcript_dir / f"{utterance_id}.lab").write_text(
                transcript + "\n", encoding="utf-8"
            )
            writer.writerow([utterance_id, str(audio_path.resolve()), len(words), transcript])
            vocabulary.update(words)
            utterance_count += 1
            word_count += len(words)

    vocabulary_path.write_text("\n".join(sorted(vocabulary)) + "\n", encoding="utf-8")
    stats = {
        "utterances": utterance_count,
        "words": word_count,
        "unique_words": len(vocabulary),
        "missing_audio": len(missing_audio),
        "empty_text": len(empty_text),
    }
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing_audio or empty_text:
        details = []
        if missing_audio:
            details.append(f"missing audio: {missing_audio[:5]}")
        if empty_text:
            details.append(f"empty text: {empty_text[:5]}")
        raise RuntimeError("; ".join(details))
    return stats


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=project_root / "data" / "RUSLAN" / "RUSLAN",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "lab3_g2p" / "mfa_corpus",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    stats = prepare_corpus(args.metadata, args.audio_dir, args.output_dir, args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

