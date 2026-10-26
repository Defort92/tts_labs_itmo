from __future__ import annotations

import unittest

import pandas as pd
import torch

from labs.lab4_acoustic_repair.repair_training import freeze_for_variance_repair, select_smoke_ids


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)
        self.duration_predictor = torch.nn.Linear(2, 1)
        self.pitch_predictor = torch.nn.Linear(2, 1)
        self.energy_predictor = torch.nn.Linear(2, 1)


class RepairTrainingTests(unittest.TestCase):
    def test_only_variance_predictors_remain_trainable(self) -> None:
        model = _TinyModel()
        summary = freeze_for_variance_repair(model)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("duration_predictor.", "pitch_predictor.", "energy_predictor.")) for name in trainable))
        self.assertGreater(summary["frozen_parameters"], 0)

    def test_smoke_ids_are_deterministic_and_disjoint(self) -> None:
        splits = {
            "train": pd.DataFrame({"utterance_id": ["003", "001", "002"]}),
            "validation": pd.DataFrame({"utterance_id": ["102", "101"]}),
        }
        self.assertEqual(select_smoke_ids(splits, train_count=2, validation_count=1), {"001", "002", "101"})


if __name__ == "__main__":
    unittest.main()

