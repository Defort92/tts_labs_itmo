"""Build pause-boundary examples from word-level CTC alignments."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from align_words import WORD_RE, iter_metadata, read_wav_mono


def boundary_marks(text: str) -> list[str]:
    """Return the written material between each pair of adjacent words."""

    text = unicodedata.normalize("NFC", text)
    matches = list(WORD_RE.finditer(text))
    return [text[left.end() : right.start()].strip() for left, right in zip(matches, matches[1:])]


def normalize_punctuation(mark: str) -> str:
    punctuation = "".join(char for char in mark if not char.isspace())
    return punctuation or "NONE"


def longest_silence_duration(
    samples: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    threshold_dbfs: float = -40.0,
    frame_ms: float = 10.0,
    anchor_s: float | None = None,
    anchor_tolerance_ms: float = 40.0,
) -> float:
    """Measure the longest low-energy run in a CTC-localized boundary.

    CTC blank frames are not necessarily silence: the model also emits blanks
    while a phoneme is sustained.  RMS energy from the original waveform is
    therefore used for the actual pause duration.
    """

    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    start_index = max(0, math.floor(start_s * sample_rate))
    end_index = min(len(samples), math.ceil(end_s * sample_rate))
    segment = samples[start_index:end_index]
    frame_count = len(segment) // frame_size
    if frame_count == 0:
        return 0.0
    frames = segment[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-10))
    silent = dbfs <= threshold_dbfs

    runs: list[tuple[int, int]] = []
    current_run = 0
    run_start = 0
    for frame_index, is_silent in enumerate(silent):
        if is_silent:
            if current_run == 0:
                run_start = frame_index
            current_run += 1
        else:
            if current_run:
                runs.append((run_start, frame_index))
            current_run = 0
    if current_run:
        runs.append((run_start, frame_count))
    if not runs:
        return 0.0

    if anchor_s is not None:
        tolerance_s = anchor_tolerance_ms / 1000.0

        def distance_to_anchor(run: tuple[int, int]) -> float:
            run_start_s = start_index / sample_rate + run[0] * frame_size / sample_rate
            run_end_s = start_index / sample_rate + run[1] * frame_size / sample_rate
            if run_start_s <= anchor_s <= run_end_s:
                return 0.0
            return min(abs(anchor_s - run_start_s), abs(anchor_s - run_end_s))

        runs = [run for run in runs if distance_to_anchor(run) <= tolerance_s]
        if not runs:
            return 0.0
    longest_run = max(end - start for start, end in runs)
    return longest_run * frame_size / sample_rate


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv",
    )
    parser.add_argument(
        "--alignments",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "word_alignments.csv",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=project_root / "data" / "RUSLAN" / "RUSLAN",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_boundaries.csv",
    )
    parser.add_argument(
        "--min-pause-ms",
        type=float,
        default=80.0,
        help="provisional acoustic gap threshold; validate it on the pilot before full processing",
    )
    parser.add_argument(
        "--silence-dbfs",
        type=float,
        default=-40.0,
        help="a 10 ms RMS frame at or below this level is considered silence",
    )
    parser.add_argument(
        "--anchor-tolerance-ms",
        type=float,
        default=40.0,
        help="maximum distance from the CTC word delimiter to a silence run",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    texts = dict(iter_metadata(args.metadata))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.alignments.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            grouped[row["utterance_id"]].append(row)

    output_rows: list[dict[str, object]] = []
    threshold_s = args.min_pause_ms / 1000.0
    for utterance_id, words in grouped.items():
        words.sort(key=lambda item: int(item["word_index"]))
        marks = boundary_marks(texts[utterance_id])
        if len(marks) != max(0, len(words) - 1):
            raise ValueError(
                f"{utterance_id}: {len(words)} aligned words but {len(marks)} written boundaries"
            )
        boundary_count = len(words) - 1
        samples, _ = read_wav_mono(args.wav_dir / f"{utterance_id}.wav", target_sample_rate=22_050)
        for boundary_index, (left, right, mark) in enumerate(
            zip(words[:-1], words[1:], marks, strict=True)
        ):
            ctc_gap_s = max(0.0, float(right["start_s"]) - float(left["end_s"]))
            duration_s = longest_silence_duration(
                samples,
                22_050,
                float(left["end_s"]),
                float(right["start_s"]),
                threshold_dbfs=args.silence_dbfs,
                anchor_s=float(left["boundary_after_s"]),
                anchor_tolerance_ms=args.anchor_tolerance_ms,
            )
            output_rows.append(
                {
                    "utterance_id": utterance_id,
                    "boundary_index": boundary_index,
                    "left_word": left["word"],
                    "right_word": right["word"],
                    "punctuation": normalize_punctuation(mark),
                    "position_ratio": f"{(boundary_index + 1) / (boundary_count + 1):.6f}",
                    "ctc_gap_s": f"{ctc_gap_s:.6f}",
                    "ctc_gap_ms": f"{ctc_gap_s * 1000.0:.3f}",
                    "gap_s": f"{duration_s:.6f}",
                    "gap_ms": f"{duration_s * 1000.0:.3f}",
                    "has_pause": int(duration_s >= threshold_s),
                    "left_confidence": left["confidence"],
                    "right_confidence": right["confidence"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utterance_id",
        "boundary_index",
        "left_word",
        "right_word",
        "punctuation",
        "position_ratio",
        "ctc_gap_s",
        "ctc_gap_ms",
        "gap_s",
        "gap_ms",
        "has_pause",
        "left_confidence",
        "right_confidence",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    pause_durations = [float(row["gap_ms"]) for row in output_rows if row["has_pause"]]
    print(f"Boundaries: {len(output_rows)}")
    print(f"Pauses (>= {args.min_pause_ms:g} ms): {len(pause_durations)}")
    if pause_durations:
        print(f"Pause median: {statistics.median(pause_durations):.1f} ms")
        print(f"Pause mean: {statistics.fmean(pause_durations):.1f} ms")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
