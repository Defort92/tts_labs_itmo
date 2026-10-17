"""Create explicitly marked fallback TextGrids for utterances MFA could not align."""

from __future__ import annotations

import argparse
import csv
import wave
from collections import defaultdict
from pathlib import Path

from praatio import textgrid
from praatio.utilities.constants import Interval

from g2p_utils import normalize_word, read_dictionary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("utterance_ids", nargs="+")
    parser.add_argument(
        "--lab2", type=Path, default=project_root / "data" / "lab2_pauses" / "word_alignments.csv"
    )
    parser.add_argument("--dictionary", type=Path, default=data_dir / "corpus_mfa.dict")
    parser.add_argument(
        "--audio-dir", type=Path, default=project_root / "data" / "RUSLAN" / "RUSLAN"
    )
    parser.add_argument("--output-dir", type=Path, default=data_dir / "textgrids")
    args = parser.parse_args()

    requested = set(args.utterance_ids)
    word_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.lab2.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["utterance_id"] in requested:
                word_rows[row["utterance_id"]].append(row)
    lexicon = read_dictionary(args.dictionary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for utterance_id in sorted(requested):
        rows = sorted(word_rows.get(utterance_id, []), key=lambda row: int(row["word_index"]))
        if not rows:
            raise ValueError(f"no lab-2 word intervals for {utterance_id}")
        audio_path = args.audio_dir / f"{utterance_id}.wav"
        with wave.open(str(audio_path), "rb") as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()

        word_intervals: list[Interval] = []
        phone_intervals: list[Interval] = []
        cursor = 0.0
        for row in rows:
            start = float(row["start_s"])
            end = float(row["end_s"])
            word = normalize_word(row["word"])
            pronunciations = lexicon.get(word)
            if not pronunciations:
                raise ValueError(f"no pronunciation for {word!r} in {utterance_id}")
            phones = pronunciations[0]
            word_intervals.append(Interval(start, end, word))
            if start > cursor:
                phone_intervals.append(Interval(cursor, start, "sil"))
            phone_duration = (end - start) / len(phones)
            for index, phone in enumerate(phones):
                phone_start = start + index * phone_duration
                phone_end = end if index == len(phones) - 1 else start + (index + 1) * phone_duration
                phone_intervals.append(Interval(phone_start, phone_end, phone))
            cursor = end
        if cursor < duration:
            phone_intervals.append(Interval(cursor, duration, "sil"))

        grid = textgrid.Textgrid(minTimestamp=0.0, maxTimestamp=duration)
        grid.addTier(textgrid.IntervalTier("words", word_intervals, minT=0.0, maxT=duration))
        grid.addTier(textgrid.IntervalTier("phones", phone_intervals, minT=0.0, maxT=duration))
        grid.save(
            str(args.output_dir / f"{utterance_id}.TextGrid"),
            format="long_textgrid",
            includeBlankSpaces=True,
        )

    fallback_path = args.output_dir / "_fallback_ids.txt"
    existing = set()
    if fallback_path.is_file():
        existing = {line.strip() for line in fallback_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    fallback_path.write_text("\n".join(sorted(existing | requested)) + "\n", encoding="utf-8")
    print(f"Created {len(requested)} fallback TextGrid(s): {', '.join(sorted(requested))}")


if __name__ == "__main__":
    main()

