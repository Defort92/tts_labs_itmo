"""Measure MFA dictionary coverage and build a reproducible G2P evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from g2p_utils import read_dictionary


def length_bin(word: str) -> str:
    length = len(word.replace("-", ""))
    if length <= 4:
        return "short"
    if length <= 8:
        return "medium"
    return "long"


def choose_evaluation_words(
    vocabulary: set[str],
    dictionary: dict[str, list[tuple[str, ...]]],
    sample_size: int,
    seed: int,
) -> list[str]:
    """Sample known unambiguous words across three spelling-length groups."""

    candidates = [
        word
        for word in vocabulary
        if len(dictionary.get(word, [])) == 1 and word.isalpha() and len(word) >= 2
    ]
    groups: dict[str, list[str]] = {"short": [], "medium": [], "long": []}
    for word in sorted(candidates):
        groups[length_bin(word)].append(word)
    rng = random.Random(seed)
    selected: list[str] = []
    remaining = sample_size
    for index, group_name in enumerate(groups):
        group = groups[group_name]
        quota = remaining // (len(groups) - index)
        take = min(quota, len(group))
        selected.extend(rng.sample(group, take))
        remaining -= take
    if remaining:
        leftovers = sorted(set(candidates) - set(selected))
        selected.extend(rng.sample(leftovers, min(remaining, len(leftovers))))
    return sorted(selected)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=data_dir / "mfa_corpus" / "vocabulary.txt",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=data_dir
        / "mfa_root"
        / "pretrained_models"
        / "dictionary"
        / "russian_mfa.dict",
    )
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vocabulary = {
        line.strip() for line in args.vocabulary.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    dictionary = read_dictionary(args.dictionary)
    known = vocabulary & dictionary.keys()
    oov = vocabulary - dictionary.keys()
    token_counts = Counter()
    for word in vocabulary:
        token_counts[length_bin(word)] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oov_words.txt").write_text(
        "\n".join(sorted(oov)) + ("\n" if oov else ""), encoding="utf-8"
    )
    selected = choose_evaluation_words(vocabulary, dictionary, args.sample_size, args.seed)
    (args.output_dir / "g2p_eval_words.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "g2p_eval_reference.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target)
        writer.writerow(["word", "reference_phones", "length_group"])
        for word in selected:
            writer.writerow([word, " ".join(dictionary[word][0]), length_bin(word)])

    stats = {
        "vocabulary_size": len(vocabulary),
        "known_words": len(known),
        "oov_words": len(oov),
        "type_coverage": len(known) / len(vocabulary) if vocabulary else 0.0,
        "evaluation_words": len(selected),
        "seed": args.seed,
        "vocabulary_length_groups": dict(token_counts),
    }
    (args.output_dir / "lexicon_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

