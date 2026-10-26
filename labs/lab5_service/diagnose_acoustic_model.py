"""Local diagnostic for separating FastSpeech 2 predictor/decoder errors.

This script is intentionally independent from the HTTP service.  It compares
free-inference variance-adaptor predictions with the held-out MFA targets used
by Lab 4 and reports teacher-forced versus free mel errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "labs/lab4_acoustic").is_dir() and (candidate / "data/lab4_acoustic").is_dir():
            return candidate
    raise FileNotFoundError("Не найден корень репозитория")


def masked_metrics(prediction, target, valid):
    pred = prediction.detach().float().cpu().numpy()[valid]
    true = target.detach().float().cpu().numpy()[valid]
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "correlation": float(np.corrcoef(pred, true)[0, 1]),
        "predicted_sum": float(pred.sum()),
        "target_sum": float(true.sum()),
        "predicted_min": float(pred.min()),
        "predicted_max": float(pred.max()),
        "target_min": float(true.min()),
        "target_max": float(true.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance-id", default="013640_RUSLAN")
    parser.add_argument("--checkpoint", default="labs/lab4_acoustic/artifacts/best_model_742028.pth")
    parser.add_argument("--write-audio", action="store_true")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd().resolve())
    lab4_dir = repo_root / "labs/lab4_acoustic"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(lab4_dir))

    import torch

    from acoustic_data import create_or_load_splits
    from evaluate_fastspeech2 import _move_batch_to_device, load_test_evaluator

    splits, _ = create_or_load_splits(repo_root)
    model, _, tokenizer, ap, loader, _ = load_test_evaluator(
        repo_root,
        splits,
        repo_root / args.checkpoint,
        batch_size=128,
        num_workers=0,
    )

    row = splits["test"].loc[splits["test"]["utterance_id"] == args.utterance_id]
    if len(row) != 1:
        raise ValueError(f"В test split ожидалась одна запись {args.utterance_id}, найдено {len(row)}")
    expected_phones = row.iloc[0]["phones"].split()

    matching_indices = [
        index
        for index, sample in enumerate(loader.dataset.samples)
        if Path(sample["audio_file"]).stem == args.utterance_id
    ]
    if len(matching_indices) != 1:
        raise RuntimeError(f"В Dataset ожидалась одна запись {args.utterance_id}, найдено {len(matching_indices)}")
    raw_batch = loader.collate_fn([loader.dataset[matching_indices[0]]])
    selected = _move_batch_to_device(model.format_batch(raw_batch), model.device)

    batch = selected
    token_count = int(batch["text_lengths"][0])
    frame_count = int(batch["mel_lengths"][0])
    true_durations = batch["durations"][0, :token_count]
    if int(true_durations.sum()) != frame_count:
        raise AssertionError("Сумма эталонных длительностей не совпадает с длиной mel")

    inputs = batch["text_input"][:, :token_count]
    x_lengths = batch["text_lengths"]
    with torch.inference_mode():
        free = model.inference(inputs)
        from TTS.tts.utils.helpers import sequence_mask

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, inputs.shape[1]), 1).float()
        encoded, x_mask, _, _ = model._forward_encoder(inputs, x_mask, None)
        if model.args.use_pitch:
            pitch_embedding, _ = model._forward_pitch_predictor(encoded, x_mask)
            encoded = encoded + pitch_embedding
        if model.args.use_energy:
            energy_embedding, _ = model._forward_energy_predictor(encoded, x_mask)
            encoded = encoded + energy_embedding
        predicted_variance_gt_duration_mel, _ = model._forward_decoder(
            encoded,
            batch["durations"][:, :token_count],
            x_mask,
            batch["mel_lengths"],
            g=None,
        )
        teacher = model(
            x=inputs,
            x_lengths=x_lengths,
            y_lengths=batch["mel_lengths"],
            y=batch["mel_input"],
            dr=batch["durations"][:, :token_count],
            pitch=batch["pitch"],
            energy=batch["energy"],
        )

    free_x_mask = torch.ones((1, 1, token_count), dtype=torch.float32, device=model.device)
    predicted_durations = model.format_durations(free["durations_log"], free_x_mask).squeeze(1)[0, :token_count]
    valid = np.ones(token_count, dtype=bool)
    duration_metrics = masked_metrics(predicted_durations, true_durations, valid)
    pitch_metrics = masked_metrics(
        teacher["pitch_avg"][0, 0, :token_count],
        teacher["pitch_avg_gt"][0, 0, :token_count],
        valid,
    )
    energy_metrics = masked_metrics(
        teacher["energy_avg"][0, 0, :token_count],
        teacher["energy_avg_gt"][0, 0, :token_count],
        valid,
    )

    reference_mel = batch["mel_input"][0, :frame_count]
    teacher_mel = teacher["model_outputs"][0, :frame_count]
    free_mel = free["model_outputs"][0]
    report = {
        "utterance_id": args.utterance_id,
        "batch_audio_unique_name": str(batch["audio_unique_names"][0]),
        "batch_item_idx": str(batch["item_idx"][0]),
        "decoded_tokens": tokenizer.ids_to_text(inputs[0].tolist()),
        "phones_match_manifest": inputs[0].tolist() == tokenizer.text_to_ids(" ".join(expected_phones)),
        "phone_count": token_count,
        "reference_frames": frame_count,
        "free_frames": int(free_mel.shape[0]),
        "duration": duration_metrics,
        "pitch": pitch_metrics,
        "energy": energy_metrics,
        "teacher_forced_mel_mae": float(torch.mean(torch.abs(teacher_mel - reference_mel)).cpu()),
        "gt_duration_predicted_variance_mel_mae": float(
            torch.mean(torch.abs(predicted_variance_gt_duration_mel[0, :frame_count] - reference_mel)).cpu()
        ),
        "gt_duration_predicted_variance_mel_std": float(
            predicted_variance_gt_duration_mel[0, :frame_count].std().cpu()
        ),
        "teacher_forced_mel_std": float(teacher_mel.std().cpu()),
        "reference_mel_std": float(reference_mel.std().cpu()),
        "free_mel_std": float(free_mel.std().cpu()),
        "per_phone": [
            {
                "phone": phone,
                "target": int(target),
                "predicted": int(prediction),
            }
            for phone, target, prediction in zip(
                expected_phones,
                true_durations.detach().cpu().tolist(),
                predicted_durations.detach().cpu().tolist(),
            )
        ],
    }
    if args.write_audio:
        import soundfile as sf

        from labs.lab5_service.vocoder_runtime import VocoderRuntime

        vocoder_dir = (
            repo_root
            / "labs/lab5_service/artifacts/vocoder/tts/vocoder_models--en--ljspeech--hifigan_v2"
        )
        vocoder = VocoderRuntime(vocoder_dir / "model.pth", vocoder_dir / "config.json", device=str(model.device))
        acoustic_view = SimpleNamespace(audio_processor=ap)
        output_dir = repo_root / "data/lab5_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        mel_variants = {
            "free": free_mel,
            "gt_duration_predicted_variance": predicted_variance_gt_duration_mel[0, :frame_count],
            "teacher_forced": teacher_mel,
            "reference": reference_mel,
        }
        audio_paths = {}
        for label, mel_tensor in mel_variants.items():
            waveform = vocoder.infer(mel_tensor.detach().cpu().numpy(), acoustic_view)
            output_path = output_dir / f"{args.utterance_id}_{label}.wav"
            sf.write(output_path, waveform, vocoder.sample_rate, subtype="PCM_16")
            audio_paths[label] = str(output_path)
        report["audio"] = audio_paths
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
