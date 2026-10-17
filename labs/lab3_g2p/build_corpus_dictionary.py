"""Combine the Russian MFA lexicon with G2P pronunciations for corpus OOV words."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from g2p_utils import read_dictionary


# russian_mfa G2P 3.0.0 exposes /ɕ/, while the matching acoustic model uses
# one long alveolo-palatal fricative category /ɕː/.  This is the only symbol
# mismatch reported by MFA's own model validator for this bundle.
PHONE_MAP = {"ɕ": "ɕː"}


def acoustic_phone_inventory(acoustic_model: Path) -> set[str]:
    with zipfile.ZipFile(acoustic_model) as archive:
        phone_file = next(name for name in archive.namelist() if name.endswith("/phones.txt"))
        lines = archive.read(phone_file).decode("utf-8").splitlines()
    return {line.split()[0] for line in lines if line.strip()} - {"<eps>"}


def remap_phones(phones: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(PHONE_MAP.get(phone, phone) for phone in phones)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    model_root = data_dir / "mfa_root" / "pretrained_models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, default=data_dir / "mfa_corpus" / "vocabulary.txt")
    parser.add_argument(
        "--base-dictionary",
        type=Path,
        default=model_root / "dictionary" / "russian_mfa.dict",
    )
    parser.add_argument("--generated", type=Path, default=data_dir / "oov_predictions.dict")
    parser.add_argument(
        "--acoustic-model",
        type=Path,
        default=model_root / "acoustic" / "russian_mfa.zip",
    )
    parser.add_argument("--output", type=Path, default=data_dir / "corpus_mfa.dict")
    args = parser.parse_args()

    vocabulary = [
        line.strip() for line in args.vocabulary.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    base = read_dictionary(args.base_dictionary)
    generated = read_dictionary(args.generated)
    acoustic_phones = acoustic_phone_inventory(args.acoustic_model)

    entries: list[tuple[str, tuple[str, ...], str]] = []
    missing_words: list[str] = []
    unsupported: Counter[str] = Counter()
    remapped: Counter[str] = Counter()
    dehyphenated_fallbacks = 0
    for word in sorted(vocabulary):
        source = "dictionary" if word in base else "g2p"
        variants = base.get(word) or generated.get(word, [])
        # MFA's G2P text reader removes hyphens before writing its output key.
        # Preserve the corpus spelling in our dictionary but reuse the generated
        # pronunciation of the dehyphenated letter sequence.
        if not variants and "-" in word:
            variants = generated.get(word.replace("-", ""), [])
            dehyphenated_fallbacks += bool(variants)
        if not variants:
            missing_words.append(word)
            continue
        for original_phones in variants:
            phones = remap_phones(original_phones)
            for old, new in zip(original_phones, phones, strict=True):
                if old != new:
                    remapped[f"{old}->{new}"] += 1
            for phone in phones:
                if phone not in acoustic_phones:
                    unsupported[phone] += 1
            entries.append((word, phones, source))

    if missing_words or unsupported:
        raise RuntimeError(
            f"cannot build lexicon: missing_words={missing_words[:10]}, unsupported={dict(unsupported)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        for word, phones, _ in entries:
            target.write(f"{word}\t{' '.join(phones)}\n")

    phone_counts = Counter(phone for _, phones, _ in entries for phone in phones)
    with (args.output.parent / "phoneme_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target)
        writer.writerow(["phone", "lexicon_count", "is_special"])
        for phone, count in sorted(phone_counts.items()):
            writer.writerow([phone, count, int(phone in {"sil", "spn"})])

    stats_path = args.output.parent / "lexicon_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    stats.update(
        {
            "corpus_dictionary_entries": len(entries),
            "generated_words": sum(word not in base for word in vocabulary),
            "missing_generated_words": len(missing_words),
            "unsupported_phones": dict(unsupported),
            "remapped_phone_occurrences": dict(remapped),
            "dehyphenated_g2p_fallbacks": dehyphenated_fallbacks,
            "phoneme_inventory_size": len(phone_counts),
        }
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
