"""Evaluate Russian MFA G2P predictions with PER and an explicit baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from g2p_utils import best_reference_distance, read_dictionary


LETTER_BASELINE = {
    "а": ("a",), "б": ("b",), "в": ("v",), "г": ("ɡ",),
    "д": ("d̪",), "е": ("e",), "ё": ("o",), "ж": ("ʐ",),
    "з": ("z̪",), "и": ("i",), "й": ("j",), "к": ("k",),
    "л": ("ɫ",), "м": ("m",), "н": ("n̪",), "о": ("o",),
    "п": ("p",), "р": ("r",), "с": ("s̪",), "т": ("t̪",),
    "у": ("u",), "ф": ("f",), "х": ("x",), "ц": ("t̪s̪",),
    "ч": ("tɕ",), "ш": ("ʂ",), "щ": ("ɕː",), "ы": ("ɨ",),
    "э": ("e",), "ю": ("u",), "я": ("a",), "ь": (), "ъ": (), "-": (),
}


def baseline_pronunciation(word: str) -> tuple[str, ...]:
    """Naive context-free letter-to-phone mapping used as a lower baseline."""

    return tuple(phone for letter in word for phone in LETTER_BASELINE.get(letter, ()))


def score_predictions(
    references: dict[str, list[tuple[str, ...]]],
    predictions: dict[str, list[tuple[str, ...]]],
) -> tuple[dict[str, float | int], list[dict[str, str | int]]]:
    phone_errors = 0
    reference_phones = 0
    exact = 0
    missing = 0
    details: list[dict[str, str | int]] = []
    for word in sorted(references):
        variants = predictions.get(word, [])
        if not variants:
            missing += 1
            details.append({"word": word, "reference": " ".join(references[word][0]), "prediction": "", "distance": -1})
            continue
        scored = [(*best_reference_distance(references[word], hypothesis), hypothesis) for hypothesis in variants]
        distance, reference, hypothesis = min(scored, key=lambda item: item[0])
        phone_errors += distance
        reference_phones += len(reference)
        exact += distance == 0
        details.append(
            {
                "word": word,
                "reference": " ".join(reference),
                "prediction": " ".join(hypothesis),
                "distance": distance,
            }
        )
    evaluated = len(references) - missing
    metrics: dict[str, float | int] = {
        "words": len(references),
        "evaluated_words": evaluated,
        "missing_words": missing,
        "phone_errors": phone_errors,
        "reference_phones": reference_phones,
        "per": phone_errors / reference_phones if reference_phones else 0.0,
        "word_accuracy": exact / evaluated if evaluated else 0.0,
    }
    return metrics, details


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=data_dir / "g2p_eval_reference.csv")
    parser.add_argument("--predictions", type=Path, default=data_dir / "g2p_eval_predictions.dict")
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    args = parser.parse_args()

    references: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    groups: dict[str, str] = {}
    with args.reference.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            references[row["word"]].append(tuple(row["reference_phones"].split()))
            groups[row["word"]] = row["length_group"]
    predictions = read_dictionary(args.predictions)
    model_metrics, details = score_predictions(dict(references), predictions)
    baseline = {word: [baseline_pronunciation(word)] for word in references}
    baseline_metrics, _ = score_predictions(dict(references), baseline)

    group_metrics: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(groups.values())):
        group_refs = {word: variants for word, variants in references.items() if groups[word] == group}
        group_metrics[group], _ = score_predictions(group_refs, predictions)

    metrics = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "by_length_group": group_metrics,
        "reference_note": (
            "Deterministic sample of single-pronunciation entries from the official Russian MFA "
            "dictionary intersected with the RUSLAN vocabulary; this is a reproducible lexical "
            "test, not an independent expert phonetic annotation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "g2p_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "g2p_errors.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["word", "reference", "prediction", "distance"])
        writer.writeheader()
        writer.writerows(sorted(details, key=lambda row: int(row["distance"]), reverse=True))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
