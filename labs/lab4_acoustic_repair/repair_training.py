"""Initialize a fresh variance-predictor fine-tuning run for FastSpeech 2.

The Lab 4 checkpoint has a usable decoder under teacher forcing, but its
duration, pitch and energy predictors are not accurate enough for free
inference.  This module deliberately keeps the original run immutable and
starts a new optimizer in a separate output directory.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAINABLE_PREFIXES = (
    "duration_predictor.",
    "pitch_predictor.",
    "energy_predictor.",
)
BEST_MODEL_RE = re.compile(r"best_model_(\d+)\.pth$")


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "labs/lab4_acoustic").is_dir() and (candidate / "data/lab4_acoustic").is_dir():
            return candidate
    raise FileNotFoundError("Не найден корень репозитория")


def _import_lab4(repo_root: Path):
    lab4_dir = repo_root / "labs/lab4_acoustic"
    path = str(lab4_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    from train_fastspeech2 import (  # pylint: disable=import-outside-toplevel
        _filter_samples,
        build_config,
        phone_inventory,
        quarantine_truncated_checkpoints,
        register_ruslan_formatter,
    )

    return {
        "filter_samples": _filter_samples,
        "build_config": build_config,
        "phone_inventory": phone_inventory,
        "quarantine": quarantine_truncated_checkpoints,
        "register_formatter": register_ruslan_formatter,
    }


def select_smoke_ids(
    splits: dict[str, pd.DataFrame],
    *,
    train_count: int = 128,
    validation_count: int = 32,
) -> set[str]:
    """Choose deterministic, disjoint train/validation records for a smoke run."""

    if train_count < 1 or validation_count < 1:
        raise ValueError("Размеры smoke-подвыборок должны быть положительными")
    train_ids = splits["train"].sort_values("utterance_id").head(train_count)["utterance_id"]
    validation_ids = splits["validation"].sort_values("utterance_id").head(validation_count)["utterance_id"]
    selected = set(train_ids.astype(str)) | set(validation_ids.astype(str))
    if len(selected) != len(train_ids) + len(validation_ids):
        raise AssertionError("Train и validation пересекаются")
    return selected


def freeze_for_variance_repair(model) -> dict[str, Any]:
    """Freeze the acoustic path and leave only variance predictors trainable."""

    trainable_names = []
    frozen_names = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        trainable = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        count = parameter.numel()
        if trainable:
            trainable_names.append(name)
            trainable_parameters += count
        else:
            frozen_names.append(name)
            frozen_parameters += count

    missing = [prefix for prefix in TRAINABLE_PREFIXES if not any(name.startswith(prefix) for name in trainable_names)]
    if missing:
        raise RuntimeError(f"В модели не найдены обязательные блоки: {missing}")
    return {
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_fraction": trainable_parameters / (trainable_parameters + frozen_parameters),
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": len(frozen_names),
        "trainable_names": trainable_names,
    }


def configure_repair(
    config,
    repo_root: Path,
    *,
    mode: str,
    batch_size: int,
    epochs: int,
    num_workers: int,
    learning_rate: float,
):
    if mode not in {"smoke", "full"}:
        raise ValueError("mode должен быть 'smoke' или 'full'")
    if epochs < 1 or batch_size < 1 or num_workers < 0 or learning_rate <= 0:
        raise ValueError("Некорректные параметры fine-tuning")

    config.run_name = f"ruslan_fastspeech2_variance_repair_{mode}"
    config.output_path = str((repo_root / "data/lab4_acoustic_repair/runs").resolve())
    config.epochs = epochs
    config.batch_size = batch_size
    config.eval_batch_size = batch_size
    config.num_loader_workers = num_workers
    config.num_eval_loader_workers = num_workers
    config.precompute_num_workers = num_workers

    # The decoder is deliberately frozen. Its teacher-forced output is already
    # usable, therefore mel losses would only obscure predictor progress.
    config.spec_loss_alpha = 0.0
    config.ssim_loss_alpha = 0.0
    config.dur_loss_alpha = 1.0
    config.pitch_loss_alpha = 1.0
    config.energy_loss_alpha = 1.0

    # A resumed Lab 4 optimizer has an almost exhausted Noam schedule. A new,
    # constant-rate optimizer gives the predictor heads meaningful updates.
    config.lr = learning_rate
    config.lr_scheduler = None
    config.lr_scheduler_params = {}
    config.save_step = 100 if mode == "smoke" else 2500
    config.save_n_checkpoints = 3
    config.print_step = 10 if mode == "smoke" else 50
    config.plot_step = 10_000_000
    config.test_sentences = []
    return config


def initialize_repair_training(
    repo_root: str | Path,
    splits: dict[str, pd.DataFrame],
    *,
    source_checkpoint: str | Path,
    mode: str = "smoke",
    selected_ids: set[str] | None = None,
    batch_size: int = 8,
    epochs: int = 2,
    num_workers: int = 0,
    learning_rate: float = 1e-4,
    continue_path: str | Path | None = None,
):
    """Build a predictor-only Trainer without touching the original run.

    A fresh run loads only model weights from ``source_checkpoint``. A resumed
    run uses ``continue_path`` to restore its optimizer, step and best loss.
    """

    import torch
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    repo_root = find_repo_root(repo_root)
    source_checkpoint = Path(source_checkpoint).resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    lab4 = _import_lab4(repo_root)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    lab4["register_formatter"]()
    config = lab4["build_config"](
        repo_root,
        splits,
        mode="full",
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
    )
    config = configure_repair(
        config,
        repo_root,
        mode=mode,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        learning_rate=learning_rate,
    )
    fresh_characters = config.characters
    audio_processor = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)
    train_samples = lab4["filter_samples"](train_samples, selected_ids)
    eval_samples = lab4["filter_samples"](eval_samples, selected_ids)
    if not train_samples or not eval_samples:
        raise ValueError("После выбора подмножества train или validation оказался пустым")

    model = ForwardTTS(config, audio_processor, tokenizer, speaker_manager=None)
    continue_dir = Path(continue_path).resolve() if continue_path else None
    quarantined = []
    if continue_dir is None:
        model.load_checkpoint(config, source_checkpoint, eval=False)
        trainer_args = TrainerArgs()
    else:
        if not continue_dir.is_dir() or not (continue_dir / "config.json").is_file():
            raise FileNotFoundError(f"Некорректный каталог продолжения: {continue_dir}")
        quarantined = lab4["quarantine"](continue_dir)
        trainer_args = TrainerArgs(continue_path=str(continue_dir))

    parameter_summary = freeze_for_variance_repair(model)
    trainer = Trainer(
        trainer_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    if continue_dir is not None:
        # Coqpit cannot deserialize the list-valued IPA inventory correctly.
        trainer.config.characters = fresh_characters
        trainer.config.num_loader_workers = num_workers
        trainer.config.num_eval_loader_workers = num_workers
        trainer.config.precompute_num_workers = num_workers
        trainer.config.save_json(str(trainer.output_path / "config.json"))
        if trainer.args.best_path:
            trainer.args.restore_path = trainer.args.best_path

    details = {
        "mode": mode,
        "source_checkpoint": str(source_checkpoint),
        "continue_path": str(continue_dir) if continue_dir else None,
        "output_path": str(trainer.output_path),
        "train_samples": len(train_samples),
        "validation_samples": len(eval_samples),
        "batch_size": trainer.config.batch_size,
        "epochs": trainer.config.epochs,
        "learning_rate": trainer.config.lr,
        "scheduler": trainer.config.lr_scheduler,
        "loss_weights": {
            "spec": trainer.config.spec_loss_alpha,
            "duration": trainer.config.dur_loss_alpha,
            "pitch": trainer.config.pitch_loss_alpha,
            "energy": trainer.config.energy_loss_alpha,
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "parameter_summary": parameter_summary,
        "resume_epoch": trainer.epochs_done,
        "resume_global_step": trainer.total_steps_done,
        "quarantined_checkpoints": [str(path) for path in quarantined],
        "vocabulary_size": len(lab4["phone_inventory"](splits)),
    }
    return trainer, config, tokenizer, details


def select_best_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    numbered = []
    for path in run_dir.glob("best_model_*.pth"):
        match = BEST_MODEL_RE.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]
    fallback = run_dir / "best_model.pth"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"В {run_dir} нет best_model*.pth")

