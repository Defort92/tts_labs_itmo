from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from acoustic_data import MANIFEST_COLUMNS, SplitSettings, allocate_frames, split_manifest, validate_splits
from evaluate_fastspeech2 import parse_training_log
from train_fastspeech2 import quarantine_truncated_checkpoints, ruslan_phone_formatter


class DurationConversionTests(unittest.TestCase):
    def test_allocates_every_frame_and_every_phone(self) -> None:
        result = allocate_frames([0.01, 0.2, 0.03], total_frames=25)
        self.assertEqual(int(result.sum()), 25)
        self.assertTrue(np.all(result >= 1))

    def test_rejects_impossible_short_spectrogram(self) -> None:
        with self.assertRaises(ValueError):
            allocate_frames([0.1, 0.1, 0.1], total_frames=2)


class SplitTests(unittest.TestCase):
    @staticmethod
    def source_frame() -> pd.DataFrame:
        rows = []
        for index in range(200):
            rows.append(
                {
                    "utterance_id": f"{index:06d}_RUSLAN",
                    "audio_relpath": f"data/RUSLAN/RUSLAN/{index:06d}_RUSLAN.wav",
                    "phones": "a b",
                    "phone_count": 2,
                    "duration_s": 1.0 + index / 100,
                    "alignment_method": "mfa",
                }
            )
        return pd.DataFrame(rows)

    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        source = self.source_frame()
        settings = SplitSettings(seed=42)
        first = split_manifest(source, settings)
        second = split_manifest(source, settings)
        validate_splits(first, expected_ids=set(source["utterance_id"]))
        for name in first:
            self.assertEqual(list(first[name].columns), list(MANIFEST_COLUMNS))
            self.assertEqual(first[name]["utterance_id"].tolist(), second[name]["utterance_id"].tolist())
        self.assertEqual({name: len(frame) for name, frame in first.items()}, {"train": 180, "validation": 10, "test": 10})


class FormatterTests(unittest.TestCase):
    def test_formatter_uses_precomputed_phone_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "utterance_id": "000000_RUSLAN",
                        "audio_relpath": "audio/000000_RUSLAN.wav",
                        "phones": "s̪ t̪ rʲ ɪ",
                    }
                ]
            ).to_csv(manifest, index=False)
            items = ruslan_phone_formatter(root, manifest)
            self.assertEqual(items[0]["text"], "s̪ t̪ rʲ ɪ")
            self.assertEqual(Path(items[0]["audio_file"]).name, "000000_RUSLAN.wav")


class CheckpointRecoveryTests(unittest.TestCase):
    def test_quarantines_only_obviously_truncated_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            valid = run_dir / "checkpoint_10000.pth"
            truncated = run_dir / "checkpoint_10123.pth"
            best = run_dir / "best_model_9000.pth"
            valid.write_bytes(b"v" * 1000)
            best.write_bytes(b"b" * 1000)
            truncated.write_bytes(b"x" * 100)

            moved = quarantine_truncated_checkpoints(run_dir)

            self.assertTrue(valid.is_file())
            self.assertTrue(best.is_file())
            self.assertFalse(truncated.exists())
            self.assertEqual(len(moved), 1)
            self.assertEqual(moved[0].name, "checkpoint_10123.pth.incomplete")
            self.assertEqual(moved[0].read_bytes(), b"x" * 100)


class TrainingLogTests(unittest.TestCase):
    def test_parser_keeps_latest_value_for_repeated_step(self) -> None:
        log = """
 > EPOCH: 0/2
   --> TIME: now -- STEP: 10/10 -- GLOBAL_STEP: 10
     | > loss: 2.0  (2.2)
 > EVALUATION
     | > avg_loss_spec: 1.8 (+0.0)
     | > avg_loss:\x1b[92m 2.1 \x1b[0m(+0.0)
 > BEST MODEL : run/best_model_10.pth
 > EPOCH: 1/2
   --> TIME: now -- STEP: 1/10 -- GLOBAL_STEP: 10
     | > loss: 1.5  (1.6)
   --> TIME: now -- STEP: 10/10 -- GLOBAL_STEP: 20
     | > loss: 1.0  (1.2)
 > EVALUATION
     | > avg_loss_spec: 0.9 (-0.9)
     | > avg_loss: 1.1 (-1.0)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "trainer_0_log.txt"
            log_path.write_text(log, encoding="utf-8")
            train, validation = parse_training_log(log_path)

        self.assertEqual(train["global_step"].tolist(), [10, 20])
        self.assertEqual(train["loss"].tolist(), [1.5, 1.0])
        self.assertEqual(validation["avg_loss"].tolist(), [2.1, 1.1])
        self.assertEqual(validation["global_step"].tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()
