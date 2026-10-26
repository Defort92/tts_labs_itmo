"""Final held-out evaluation helpers for Lab 4.

The training notebook deliberately never exposes the test split to Trainer.
This module loads the frozen best checkpoint and evaluates it exactly once on
that split.  Imports from the course Coqui fork remain lazy so lightweight log
parsing and unit tests do not require a CUDA environment.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_fastspeech2 import build_config, register_ruslan_formatter


TRAIN_STEP_RE = re.compile(r"GLOBAL_STEP:\s*(\d+)")
EPOCH_RE = re.compile(r"> EPOCH:\s*(\d+)/")
TRAIN_LOSS_RE = re.compile(r"\| > loss:\s*([-+0-9.eE]+)\s*\(([-+0-9.eE]+)\)")
EVAL_VALUE_RE = re.compile(r"\| > (avg_[a-z_]+):\s*([-+0-9.eE]+)")
BEST_MODEL_RE = re.compile(r"best_model_(\d+)\.pth")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_training_log(log_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse restart-safe train and validation curves from Trainer's text log.

    Repeated steps can occur after a restart from a checkpoint.  The last
    occurrence is retained because it belongs to the final training history.
    """

    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    current_epoch: int | None = None
    current_step: int | None = None
    in_evaluation = False
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    eval_values: dict[str, float] = {}

    def finish_eval() -> None:
        nonlocal eval_values
        if "avg_loss" in eval_values and current_epoch is not None and current_step is not None:
            eval_rows.append(
                {"epoch": current_epoch, "global_step": current_step, "is_best_checkpoint": False, **eval_values}
            )
        eval_values = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            # Trainer colors improvements green; remove terminal escapes before
            # applying numeric regular expressions.
            line = ANSI_RE.sub("", line)
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                if in_evaluation:
                    finish_eval()
                current_epoch = int(epoch_match.group(1))
                in_evaluation = False
                continue

            if "> EVALUATION" in line:
                if in_evaluation:
                    finish_eval()
                in_evaluation = True
                continue

            best_match = BEST_MODEL_RE.search(line)
            if best_match and in_evaluation:
                finish_eval()
                in_evaluation = False
                eval_rows[-1]["global_step"] = int(best_match.group(1))
                eval_rows[-1]["is_best_checkpoint"] = True
                continue

            step_match = TRAIN_STEP_RE.search(line)
            if step_match:
                if in_evaluation:
                    finish_eval()
                current_step = int(step_match.group(1))
                in_evaluation = False
                continue

            if in_evaluation:
                value_match = EVAL_VALUE_RE.search(line)
                if value_match:
                    eval_values[value_match.group(1)] = float(value_match.group(2))
                continue

            loss_match = TRAIN_LOSS_RE.search(line)
            if loss_match and current_epoch is not None and current_step is not None:
                train_rows.append(
                    {
                        "epoch": current_epoch,
                        "global_step": current_step,
                        "loss": float(loss_match.group(1)),
                        "epoch_running_avg_loss": float(loss_match.group(2)),
                    }
                )

    if in_evaluation:
        finish_eval()

    train = pd.DataFrame(train_rows)
    validation = pd.DataFrame(eval_rows)
    if train.empty or validation.empty:
        raise ValueError(f"Не удалось извлечь train/validation loss из {log_path}")

    # Restarts can repeat epochs and global steps.  The log is chronological,
    # therefore keep='last' reconstructs the history that led to the final run.
    train = train.drop_duplicates("global_step", keep="last").sort_values("global_step").reset_index(drop=True)
    validation = validation.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    return train, validation


def plot_loss_curves(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Plot logged train loss and the comparable held-out validation loss."""

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        train["global_step"],
        train["epoch_running_avg_loss"],
        color="#4472c4",
        linewidth=1.1,
        alpha=0.8,
        label="train: среднее внутри эпохи",
    )
    axis.plot(
        validation["global_step"],
        validation["avg_loss"],
        color="#c00000",
        linewidth=1.5,
        label="validation",
    )
    best = validation.loc[validation["avg_loss"].idxmin()]
    axis.scatter([best["global_step"]], [best["avg_loss"]], color="#008000", zorder=3)
    axis.annotate(
        f"best={best['avg_loss']:.4f}\nstep={int(best['global_step'])}",
        (best["global_step"], best["avg_loss"]),
        xytext=(8, 12),
        textcoords="offset points",
    )
    axis.set(title="FastSpeech 2: динамика функции потерь", xlabel="Global step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def _test_samples(config, repo_root: Path):
    """Load only the persistent test manifest, including MFA attention masks."""

    from TTS.tts.datasets import load_tts_samples

    dataset = config.datasets[0]
    dataset.meta_file_train = str((repo_root / "data/lab4_acoustic/splits/test.csv").relative_to(repo_root))
    dataset.meta_file_val = None
    samples, unused_eval = load_tts_samples(config.datasets, eval_split=False)
    if unused_eval:
        raise AssertionError("При итоговой оценке не должно создаваться дополнительное разбиение")
    return samples


def load_test_evaluator(
    repo_root: str | Path,
    splits: dict[str, pd.DataFrame],
    checkpoint_path: str | Path,
    *,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """Load the frozen best model and construct a DataLoader for all test rows."""

    import torch
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    repo_root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    register_ruslan_formatter()
    config = build_config(
        repo_root,
        splits,
        mode="full",
        batch_size=batch_size,
        epochs=300,
        num_workers=num_workers,
    )
    config.shuffle = False
    config.drop_last = False
    config.start_by_longest = False
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    samples = _test_samples(config, repo_root)
    expected = len(splits["test"])
    if len(samples) != expected:
        raise ValueError(f"Ожидалось {expected} test-записей, загружено {len(samples)}")

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    model.load_checkpoint(config, checkpoint_path, eval=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = model.get_criterion().to(device)
    loader = model.get_data_loader(
        config,
        assets={"audio_processor": ap},
        is_eval=True,
        samples=samples,
        verbose=True,
        num_gpus=1,
    )
    details = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "test_samples": len(samples),
        "batches": len(loader),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    return model, criterion, tokenizer, ap, loader, details


def _move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    import torch

    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate_test_set(model, criterion, loader, *, progress_every: int = 25) -> dict[str, float | int]:
    """Compute sample-weighted FastSpeech 2 losses over the complete test split."""

    import torch

    totals: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    model.eval()
    device = model.device
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = model.format_batch(raw_batch)
            batch = _move_batch_to_device(batch, device)
            _, losses = model.eval_step(batch, criterion)
            batch_count = int(batch["text_input"].shape[0])
            sample_count += batch_count
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * batch_count
            if progress_every and (batch_index % progress_every == 0 or batch_index == len(loader)):
                print(f"test: {sample_count}/{len(loader.dataset)} записей")

    if sample_count != len(loader.dataset):
        raise AssertionError(f"Оценено {sample_count}, ожидалось {len(loader.dataset)}")
    metrics: dict[str, float | int] = {name: total / sample_count for name, total in totals.items()}
    metrics["samples"] = sample_count
    metrics["batches"] = len(loader)
    return metrics


def select_representative_examples(test_frame: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    """Select deterministic duration quantiles instead of convenient cherry-picked rows."""

    if count < 1 or count > len(test_frame):
        raise ValueError("Некорректное число примеров")
    ordered = test_frame.sort_values(["duration_s", "utterance_id"]).reset_index(drop=True)
    positions = np.linspace(0.2, 0.8, count)
    indices = sorted({int(round(position * (len(ordered) - 1))) for position in positions})
    return ordered.iloc[indices].reset_index(drop=True)


def create_spectrogram_examples(
    model,
    tokenizer,
    ap,
    test_frame: pd.DataFrame,
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    count: int = 3,
) -> pd.DataFrame:
    """Run free inference from phones and save predicted/reference mel plots."""

    import torch

    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_representative_examples(test_frame, count=count)
    rows = []
    model.eval()
    with torch.inference_mode():
        for row in chosen.itertuples(index=False):
            token_ids = tokenizer.text_to_ids(str(row.phones))
            inputs = torch.tensor(token_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            prediction = model.inference(inputs)["model_outputs"][0].detach().cpu().numpy()
            wav = ap.load_wav(str(repo_root / row.audio_relpath))
            reference = ap.melspectrogram(wav).T
            if prediction.ndim != 2 or prediction.shape[1] != reference.shape[1]:
                raise ValueError(
                    f"{row.utterance_id}: формы mel несовместимы: pred={prediction.shape}, ref={reference.shape}"
                )

            low, high = np.percentile(np.concatenate([reference.ravel(), prediction.ravel()]), [1, 99])
            figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
            axes[0].imshow(reference.T, origin="lower", aspect="auto", vmin=low, vmax=high, cmap="magma")
            axes[0].set_title(f"Эталон: {row.utterance_id} ({reference.shape[0]} кадров)")
            axes[1].imshow(prediction.T, origin="lower", aspect="auto", vmin=low, vmax=high, cmap="magma")
            axes[1].set_title(f"Свободный inference ({prediction.shape[0]} кадров)")
            for axis in axes:
                axis.set(ylabel="Mel-канал", xlabel="Кадр")
            plot_path = output_dir / f"mel_{row.utterance_id}.png"
            figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            rows.append(
                {
                    "utterance_id": row.utterance_id,
                    "duration_s": float(row.duration_s),
                    "phones": row.phones,
                    "reference_frames": int(reference.shape[0]),
                    "predicted_frames": int(prediction.shape[0]),
                    "duration_ratio": float(prediction.shape[0] / reference.shape[0]),
                    "plot": str(plot_path),
                }
            )
    return pd.DataFrame(rows)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
    return path
