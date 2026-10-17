from __future__ import annotations

import sys
import unittest
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from build_corpus_dictionary import remap_phones
from compare_word_boundaries import summarize
from evaluate_g2p import baseline_pronunciation, score_predictions
from g2p_utils import edit_distance, extract_words, normalize_word, parse_dictionary_line
from parse_alignments import Span, best_word


class TextPreparationTests(unittest.TestCase):
    def test_extracts_same_word_units_as_lab2(self) -> None:
        self.assertEqual(extract_words("Ёлка, кто-то зво́нит!"), ["Ёлка", "кто-то", "зво́нит"])

    def test_normalizes_case_dash_and_stress(self) -> None:
        self.assertEqual(normalize_word("КТО‑ТО"), "кто-то")
        self.assertEqual(normalize_word("зво́нит"), "звонит")


class DictionaryTests(unittest.TestCase):
    def test_parses_probability_dictionary(self) -> None:
        parsed = parse_dictionary_line("привет 0.99 0.48 3.55 0.55 p rʲ ɪ vʲ e t̪")
        self.assertEqual(parsed, ("привет", ("p", "rʲ", "ɪ", "vʲ", "e", "t̪")))

    def test_parses_plain_dictionary(self) -> None:
        self.assertEqual(parse_dictionary_line("мир\tmʲ i r"), ("мир", ("mʲ", "i", "r")))

    def test_remaps_known_bundle_mismatch(self) -> None:
        self.assertEqual(remap_phones(("ɕ", "a")), ("ɕː", "a"))


class MetricTests(unittest.TestCase):
    def test_phone_edit_distance(self) -> None:
        self.assertEqual(edit_distance(("a", "b", "c"), ("a", "c")), 1)

    def test_uses_closest_reference_variant(self) -> None:
        metrics, _ = score_predictions(
            {"тест": [("t", "e", "s", "t"), ("t", "ɛ", "s", "t")]},
            {"тест": [("t", "ɛ", "s", "t")]},
        )
        self.assertEqual(metrics["per"], 0.0)
        self.assertEqual(metrics["word_accuracy"], 1.0)

    def test_baseline_ignores_soft_and_hard_signs(self) -> None:
        self.assertEqual(baseline_pronunciation("объём"), ("o", "b", "o", "m"))

    def test_summarizes_boundary_errors(self) -> None:
        summary = summarize([10.0, 20.0, 30.0])
        self.assertEqual(summary["median_ms"], 20.0)


class AlignmentTests(unittest.TestCase):
    def test_assigns_phone_by_maximum_overlap(self) -> None:
        words = [Span(0.0, 0.5, "раз"), Span(0.5, 1.0, "два")]
        self.assertEqual(best_word(Span(0.45, 0.65, "d"), words), (1, "два"))
        self.assertEqual(best_word(Span(1.1, 1.2, "sil"), words), (None, ""))


if __name__ == "__main__":
    unittest.main()
