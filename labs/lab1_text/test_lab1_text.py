"""Regression tests for the first laboratory work."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
import unicodedata

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from preprocess_ruslan import preprocess
from text_filter import TextFilter
from text_normalizer import TextNormalizer


class TextFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text_filter = TextFilter()

    def test_accepts_literal_russian_text(self) -> None:
        self.assertEqual(self.text_filter.filter("Сегодня будет тепло."), 1)
        self.assertEqual(self.text_filter.filter("Стоимость — сто рублей."), 1)
        self.assertEqual(self.text_filter.filter("СЕРГЕЙ ДОВЛАТОВ"), 1)

    def test_rejects_non_literal_notation(self) -> None:
        for text in (
            "Стоимость — 100 рублей.",
            "Напишите на user@example.com.",
            "Офис находится на ул. Ленина.",
            "Справку выдали в МВД.",
            "Справку выдали в ФСБ.",
            "Спасибо , что подождали.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.text_filter.filter(text), 0)

    def test_dev_f1_is_above_target(self) -> None:
        with (LAB_DIR / "data" / "dev_sentences.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter="|"))
        expected = [int(row["is_normalized"]) for row in rows]
        predicted = [self.text_filter.filter(row["text"]) for row in rows]
        tp = sum(y == p == 1 for y, p in zip(expected, predicted))
        fp = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
        fn = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
        f1 = 2 * tp / (2 * tp + fp + fn)
        self.assertGreaterEqual(f1, 0.90)


class TextNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = TextNormalizer()

    def test_normalizes_safe_typography(self) -> None:
        self.assertEqual(
            self.normalizer.normalize("  Расстреливать  писателей!..  "),
            "Расстреливать писателей!",
        )
        self.assertEqual(self.normalizer.normalize("де‑факто"), "де-факто")
        self.assertEqual(self.normalizer.normalize("Спасибо ,что пришли."), "Спасибо, что пришли.")
        self.assertEqual(self.normalizer.normalize("«Начало фразы."), "Начало фразы.")
        self.assertEqual(self.normalizer.normalize("Конец фразы.»"), "Конец фразы.")
        self.assertEqual(self.normalizer.normalize("Текст (без конца."), "Текст без конца.")
        self.assertEqual(self.normalizer.normalize('Текст с лишней кавычкой."'), "Текст с лишней кавычкой.")

    def test_preserves_yo_case_and_stress(self) -> None:
        result = self.normalizer.normalize("Ёлка уже́ ЗЕЛЁНАЯ.")
        self.assertEqual(result, "Ёлка уже́ ЗЕЛЁНАЯ.")
        self.assertEqual(result, unicodedata.normalize("NFC", result))


class PreprocessTests(unittest.TestCase):
    def test_writes_exactly_three_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.csv"
            destination = Path(temp_dir) / "output.csv"
            source.write_text(
                "001|Спасибо ,что пришли.\n002|Стоимость 100 рублей.\n",
                encoding="utf-8",
            )
            stats = preprocess(source, destination)
            with destination.open(encoding="utf-8") as output_file:
                rows = list(csv.reader(output_file, delimiter="|"))
            self.assertEqual(rows, [["001", "Спасибо ,что пришли.", "Спасибо, что пришли."]])
            self.assertEqual(
                stats,
                {
                    "total": 2,
                    "kept": 1,
                    "changed": 1,
                    "rejected": 1,
                    "reviewed_overrides": 0,
                },
            )

    def test_applies_only_reviewed_corpus_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.csv"
            destination = Path(temp_dir) / "output.csv"
            source.write_text(
                "000880_RUSLAN|Я на букву О /библиография к Окуджаве/.\n"
                "001028_RUSLAN|Уфлянда зовут Владимир прим. автора.\n"
                "001257_RUSLAN|Александрову Г. П.\n",
                encoding="utf-8",
            )
            stats = preprocess(source, destination)
            with destination.open(encoding="utf-8") as output_file:
                rows = list(
                    csv.reader(output_file, delimiter="|", quoting=csv.QUOTE_NONE)
                )
            self.assertEqual(
                rows,
                [
                    [
                        "000880_RUSLAN",
                        "Я на букву О /библиография к Окуджаве/.",
                        "Я на букву О библиография к Окуджаве.",
                    ],
                    [
                        "001028_RUSLAN",
                        "Уфлянда зовут Владимир прим. автора.",
                        "Уфлянда зовут Владимир прим. автора.",
                    ],
                ],
            )
            self.assertEqual(stats["reviewed_overrides"], 2)
            self.assertEqual(stats["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
