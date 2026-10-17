"""Summarize and visualize the pause labels produced for RUSLAN."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(pauses_path: Path, status_path: Path) -> dict[str, object]:
    with pauses_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    with status_path.open("r", encoding="utf-8", newline="") as source:
        statuses = list(csv.DictReader(source))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["punctuation"]].append(row)

    by_punctuation: dict[str, dict[str, float | int]] = {}
    for punctuation, group in sorted(grouped.items()):
        durations = [float(row["gap_ms"]) for row in group]
        positive = [duration for row, duration in zip(group, durations, strict=True) if int(row["has_pause"])]
        by_punctuation[punctuation] = {
            "boundaries": len(group),
            "pauses": len(positive),
            "pause_rate": len(positive) / len(group),
            "median_pause_ms": statistics.median(positive) if positive else 0.0,
            "mean_pause_ms": statistics.fmean(positive) if positive else 0.0,
        }

    all_positive = [float(row["gap_ms"]) for row in rows if int(row["has_pause"])]
    return {
        "utterances_recorded": len(statuses),
        "utterances_aligned": sum(row["status"] == "ok" for row in statuses),
        "alignment_errors": sum(row["status"] != "ok" for row in statuses),
        "boundaries": len(rows),
        "pauses": len(all_positive),
        "pause_rate": len(all_positive) / len(rows) if rows else 0.0,
        "median_pause_ms": statistics.median(all_positive) if all_positive else 0.0,
        "mean_pause_ms": statistics.fmean(all_positive) if all_positive else 0.0,
        "by_punctuation": by_punctuation,
    }


def make_figure(pauses_path: Path, output_path: Path) -> None:
    with pauses_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    positive = [float(row["gap_ms"]) for row in rows if int(row["has_pause"])]
    rates: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        rates[row["punctuation"]].append(int(row["has_pause"]))
    ordered = sorted(rates.items(), key=lambda item: len(item[1]), reverse=True)[:10]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(positive, bins=40, color="#4472C4", edgecolor="white")
    axes[0].set_title("Распределение длительности пауз")
    axes[0].set_xlabel("Длительность, мс")
    axes[0].set_ylabel("Количество")
    axes[0].grid(axis="y", alpha=0.25)

    labels = ["без знака" if key == "NONE" else key for key, _ in ordered]
    values = [sum(items) / len(items) for _, items in ordered]
    axes[1].bar(labels, values, color="#70AD47")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Доля пауз по пунктуации")
    axes[1].set_ylabel("Доля границ с паузой")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pauses",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_boundaries.csv",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "alignment_status.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_analysis.json",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_distribution.png",
    )
    args = parser.parse_args()
    summary = summarize(args.pauses, args.status)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(args.pauses, args.output_figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
