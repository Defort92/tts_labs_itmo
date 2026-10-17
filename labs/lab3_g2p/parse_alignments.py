"""Convert MFA TextGrid output into phoneme, word-pronunciation, and utterance tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from praatio import textgrid


SILENCE_LABELS = {"", "sil", "sp", "<eps>"}


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    label: str


def spans(tier: object) -> list[Span]:
    return [Span(float(entry.start), float(entry.end), str(entry.label)) for entry in tier.entries]


def select_tier(grid: object, fragment: str) -> object:
    names = list(grid.tierNames)
    matches = [name for name in names if fragment in name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"expected one {fragment!r} tier, found {matches!r} in {names!r}")
    return grid.getTier(matches[0])


def best_word(phone: Span, words: list[Span]) -> tuple[int | None, str]:
    best_index: int | None = None
    best_overlap = 0.0
    for index, word in enumerate(words):
        overlap = max(0.0, min(phone.end, word.end) - max(phone.start, word.start))
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    if best_index is None:
        return None, ""
    return best_index, words[best_index].label


def parse_textgrid(path: Path) -> tuple[list[Span], list[Span]]:
    grid = textgrid.openTextgrid(path, includeEmptyIntervals=True)
    word_spans = [span for span in spans(select_tier(grid, "word")) if span.label]
    phone_spans = spans(select_tier(grid, "phone"))
    return word_spans, phone_spans


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", type=Path, default=data_dir / "textgrids")
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phone_path = args.output_dir / "phoneme_alignments.csv"
    word_path = args.output_dir / "word_pronunciations.csv"
    utterance_path = args.output_dir / "utterance_phonemes.csv"
    phone_counts: Counter[str] = Counter()
    fallback_path = args.textgrid_dir / "_fallback_ids.txt"
    fallback_ids = (
        {line.strip() for line in fallback_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if fallback_path.is_file()
        else set()
    )
    utterance_count = 0
    phone_count = 0
    speech_phone_count = 0
    total_phone_duration = 0.0

    with (
        phone_path.open("w", encoding="utf-8", newline="") as phone_file,
        word_path.open("w", encoding="utf-8", newline="") as word_file,
        utterance_path.open("w", encoding="utf-8", newline="") as utterance_file,
    ):
        phone_writer = csv.writer(phone_file)
        word_writer = csv.writer(word_file)
        utterance_writer = csv.writer(utterance_file)
        phone_writer.writerow(
            ["utterance_id", "phone_index", "word_index", "word", "phone", "start_s", "end_s", "duration_s", "is_silence", "alignment_method"]
        )
        word_writer.writerow(
            ["utterance_id", "word_index", "word", "start_s", "end_s", "duration_s", "phones", "alignment_method"]
        )
        utterance_writer.writerow(["utterance_id", "phone_count", "phones", "alignment_method"])

        for textgrid_path in sorted(args.textgrid_dir.rglob("*.TextGrid")):
            utterance_id = textgrid_path.stem
            method = "lab2_uniform_fallback" if utterance_id in fallback_ids else "mfa"
            words, phones = parse_textgrid(textgrid_path)
            phones_by_word: dict[int, list[str]] = {index: [] for index in range(len(words))}
            utterance_phones: list[str] = []
            for phone_index, phone in enumerate(phones):
                is_silence = phone.label in SILENCE_LABELS
                word_index, word = best_word(phone, words)
                duration = phone.end - phone.start
                phone_writer.writerow(
                    [
                        utterance_id,
                        phone_index,
                        "" if word_index is None else word_index,
                        word,
                        phone.label or "sil",
                        f"{phone.start:.6f}",
                        f"{phone.end:.6f}",
                        f"{duration:.6f}",
                        int(is_silence),
                        method,
                    ]
                )
                phone_counts[phone.label or "sil"] += 1
                phone_count += 1
                total_phone_duration += duration
                if not is_silence:
                    speech_phone_count += 1
                    utterance_phones.append(phone.label)
                    if word_index is not None:
                        phones_by_word[word_index].append(phone.label)
            for word_index, word in enumerate(words):
                word_writer.writerow(
                    [
                        utterance_id,
                        word_index,
                        word.label,
                        f"{word.start:.6f}",
                        f"{word.end:.6f}",
                        f"{word.end - word.start:.6f}",
                        " ".join(phones_by_word[word_index]),
                        method,
                    ]
                )
            utterance_writer.writerow(
                [utterance_id, len(utterance_phones), " ".join(utterance_phones), method]
            )
            utterance_count += 1

    stats = {
        "utterances": utterance_count,
        "phone_intervals": phone_count,
        "speech_phone_intervals": speech_phone_count,
        "phone_inventory_size": len(phone_counts),
        "mfa_utterances": utterance_count - len(fallback_ids),
        "fallback_utterances": len(fallback_ids),
        "mean_phone_duration_ms": 1000 * total_phone_duration / phone_count if phone_count else 0.0,
        "phone_counts": dict(sorted(phone_counts.items())),
    }
    (args.output_dir / "alignment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
