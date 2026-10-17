from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import numpy as np


LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from align_words import ctc_forced_alignment, prepare_transcript, word_segments
from extract_pauses import boundary_marks, longest_silence_duration, normalize_punctuation
from make_review_set import choose_examples
from pause_model import BoundaryFeatureBuilder, text_boundary_rows


class TranscriptPreparationTests(unittest.TestCase):
    def test_preserves_written_words_and_removes_non_acoustic_marks(self) -> None:
        vocab = {"<pad>": 0, "|": 1, "м": 2, "а": 3, "ё": 4, "е": 5, "л": 6}
        prepared = prepare_transcript("Ма-ма, ёл!", vocab)
        self.assertEqual(prepared.words, ["Ма-ма", "ёл"])
        self.assertEqual(prepared.acoustic_text, "мама|ёл")
        self.assertEqual(len(prepared.token_ids), len("мама|ёл"))
        self.assertEqual(prepared.token_to_word, [0, 0, 0, 0, None, 1, 1])

    def test_replaces_yo_only_when_model_has_no_yo(self) -> None:
        vocab = {"<pad>": 0, "|": 1, "е": 2, "л": 3}
        prepared = prepare_transcript("Ёл", vocab)
        self.assertEqual(prepared.acoustic_text, "ел")

    def test_combining_stress_mark_stays_inside_word(self) -> None:
        vocab = {"<pad>": 0, "|": 1, "з": 2, "в": 3, "о": 4, "н": 5, "и": 6, "т": 7}
        prepared = prepare_transcript("Он зво́нит", vocab)
        self.assertEqual(prepared.words, ["Он", "зво́нит"])
        self.assertEqual(prepared.acoustic_text, "он|звонит")


class CTCAlignmentTests(unittest.TestCase):
    def test_aligns_known_two_token_path(self) -> None:
        # Vocabulary: blank=0, a=1, b=2. The best path is blank,a,blank,b,blank.
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
                [0.90, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        segments, _ = ctc_forced_alignment(probabilities.log(), [1, 2], blank_id=0)
        self.assertEqual([segment.token_index for segment in segments], [0, 1])
        self.assertEqual([segment.emission_frame for segment in segments], [1, 3])
        self.assertLess(segments[0].start_frame, segments[0].end_frame)
        self.assertLessEqual(segments[0].end_frame, segments[1].start_frame)

    def test_merges_characters_into_words(self) -> None:
        vocab = {"<pad>": 0, "|": 1, "а": 2, "б": 3}
        prepared = prepare_transcript("А б", vocab)
        probabilities = torch.tensor(
            [
                [0.05, 0.05, 0.85, 0.05],
                [0.05, 0.85, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.85],
            ],
            dtype=torch.float32,
        )
        segments, _ = ctc_forced_alignment(probabilities.log(), prepared.token_ids, blank_id=0)
        words = word_segments(prepared, segments, audio_duration_s=0.3, frame_count=3)
        self.assertEqual([word.word for word in words], ["А", "б"])
        self.assertAlmostEqual(words[0].start_s, 0.0)
        self.assertAlmostEqual(words[1].end_s, 0.3)
        self.assertAlmostEqual(words[0].boundary_after_s, 0.15)
        self.assertIsNone(words[1].boundary_after_s)


class PauseExtractionTests(unittest.TestCase):
    def test_recovers_written_boundary_marks(self) -> None:
        self.assertEqual(boundary_marks("Раз, два — три."), [",", "—"])
        self.assertEqual(normalize_punctuation(" — "), "—")
        self.assertEqual(normalize_punctuation(" "), "NONE")

    def test_number_of_boundaries_is_one_less_than_number_of_words(self) -> None:
        words = ["раз", "два", "три"]
        marks = boundary_marks("Раз, два — три.")
        pairs = list(zip(words[:-1], words[1:], marks, strict=True))
        self.assertEqual(pairs, [("раз", "два", ","), ("два", "три", "—")])

    def test_measures_longest_acoustic_silence(self) -> None:
        # 30 ms sound, 90 ms silence, 20 ms sound, 40 ms silence at 1 kHz.
        samples = np.concatenate(
            [
                np.full(30, 0.1, dtype=np.float32),
                np.zeros(90, dtype=np.float32),
                np.full(20, 0.1, dtype=np.float32),
                np.zeros(40, dtype=np.float32),
            ]
        )
        duration = longest_silence_duration(
            samples,
            sample_rate=1000,
            start_s=0.0,
            end_s=len(samples) / 1000,
            threshold_dbfs=-40.0,
            frame_ms=10.0,
        )
        self.assertAlmostEqual(duration, 0.09)

        anchored_duration = longest_silence_duration(
            samples,
            sample_rate=1000,
            start_s=0.0,
            end_s=len(samples) / 1000,
            threshold_dbfs=-40.0,
            frame_ms=10.0,
            anchor_s=0.15,
            anchor_tolerance_ms=5.0,
        )
        self.assertAlmostEqual(anchored_duration, 0.04)


class PauseModelFeatureTests(unittest.TestCase):
    def test_builds_boundaries_for_new_text(self) -> None:
        rows = text_boundary_rows("Раз, два — три.")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["left_word"], "Раз")
        self.assertEqual(rows[0]["right_word"], "два")
        self.assertEqual(rows[0]["punctuation"], ",")
        self.assertEqual(rows[1]["punctuation"], "—")

    def test_features_include_punctuation_position_and_morphology(self) -> None:
        row = text_boundary_rows("Первая фраза.")[0]
        features = BoundaryFeatureBuilder().transform_row(row)
        self.assertEqual(features["punctuation"], "NONE")
        self.assertIn("left_pos", features)
        self.assertIn("right_pos", features)
        self.assertEqual(features["position_ratio"], 0.5)


class ReviewSetTests(unittest.TestCase):
    def test_stratified_examples_do_not_repeat_boundaries(self) -> None:
        rows = [
            {
                "utterance_id": f"u{index}",
                "boundary_index": "0",
                "punctuation": "," if index < 4 else "NONE",
                "gap_ms": str((index + 1) * 40),
            }
            for index in range(8)
        ]
        selected = choose_examples(rows, threshold_ms=80.0, per_group=1)
        keys = [(row["utterance_id"], row["boundary_index"]) for _, row in selected]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreaterEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
