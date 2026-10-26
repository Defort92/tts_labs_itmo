"""Evaluate free-inference variance predictors on the frozen Lab 4 test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .repair_training import find_repo_root


def _metrics(predicted: list[np.ndarray], target: list[np.ndarray]) -> dict[str, float]:
    pred = np.concatenate(predicted).astype(np.float64)
    true = np.concatenate(target).astype(np.float64)
    correlation = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) and np.std(true) else float("nan")
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "bias": float(np.mean(pred - true)),
        "correlation": correlation,
        "predicted_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(true)),
    }


def evaluate(checkpoint: str | Path, *, batch_size: int = 16, num_workers: int = 0) -> dict:
    import torch
    from TTS.tts.utils.helpers import sequence_mask

    repo_root = find_repo_root()
    lab4_dir = repo_root / "labs/lab4_acoustic"
    if str(lab4_dir) not in sys.path:
        sys.path.insert(0, str(lab4_dir))
    from acoustic_data import create_or_load_splits  # pylint: disable=import-outside-toplevel
    from evaluate_fastspeech2 import (  # pylint: disable=import-outside-toplevel
        _move_batch_to_device,
        load_test_evaluator,
    )

    splits, _ = create_or_load_splits(repo_root)
    model, _, _, _, loader, details = load_test_evaluator(
        repo_root,
        splits,
        Path(checkpoint).resolve(),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    values = {name: [] for name in ("duration_pred", "duration_true", "pitch_pred", "pitch_true", "energy_pred", "energy_true")}
    frame_ratios = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch_to_device(model.format_batch(raw_batch), model.device)
            inputs = batch["text_input"]
            text_lengths = batch["text_lengths"]
            true_durations = batch["durations"]
            x_mask = torch.unsqueeze(sequence_mask(text_lengths, inputs.shape[1]), 1).float()
            encoded, x_mask, _, _ = model._forward_encoder(inputs, x_mask, None)

            duration_logs = model.duration_predictor(encoded, x_mask)
            predicted_durations = model.format_durations(duration_logs, x_mask).squeeze(1)
            pitch_embedding_gt, pitch_predicted, pitch_true = model._forward_pitch_predictor(
                encoded,
                x_mask,
                batch["pitch"],
                true_durations,
            )
            del pitch_embedding_gt
            predicted_pitch_embedding = model.pitch_emb(pitch_predicted)
            encoded_with_predicted_pitch = encoded + predicted_pitch_embedding
            _, energy_predicted = model._forward_energy_predictor(encoded_with_predicted_pitch, x_mask)
            _, _, energy_true = model._forward_energy_predictor(
                encoded + model.pitch_emb(pitch_true),
                x_mask,
                batch["energy"],
                true_durations,
            )

            for row in range(inputs.shape[0]):
                length = int(text_lengths[row])
                duration_pred = predicted_durations[row, :length].cpu().numpy()
                duration_true = true_durations[row, :length].cpu().numpy()
                values["duration_pred"].append(duration_pred)
                values["duration_true"].append(duration_true)
                values["pitch_pred"].append(pitch_predicted[row, 0, :length].cpu().numpy())
                values["pitch_true"].append(pitch_true[row, 0, :length].cpu().numpy())
                values["energy_pred"].append(energy_predicted[row, 0, :length].cpu().numpy())
                values["energy_true"].append(energy_true[row, 0, :length].cpu().numpy())
                frame_ratios.append(float(duration_pred.sum() / duration_true.sum()))

    ratios = np.asarray(frame_ratios)
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": details["device"],
        "test_samples": details["test_samples"],
        "duration": _metrics(values["duration_pred"], values["duration_true"]),
        "pitch": _metrics(values["pitch_pred"], values["pitch_true"]),
        "energy_free_pitch": _metrics(values["energy_pred"], values["energy_true"]),
        "frame_ratio": {
            "mean": float(ratios.mean()),
            "p10": float(np.quantile(ratios, 0.10)),
            "p50": float(np.quantile(ratios, 0.50)),
            "p90": float(np.quantile(ratios, 0.90)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

