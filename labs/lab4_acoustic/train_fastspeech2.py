"""Coqui FastSpeech 2 configuration used by the Lab 4 notebook.

Imports from the course fork are intentionally lazy: the data audit and split
cells remain runnable even before the GPU environment is installed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd

from acoustic_data import AcousticSettings


COURSE_TTS_COMMIT = "d8f03ab5f23411842cc429d0c93a506b772f5b1f"
MULTICHAR_CLASS = "TTS.tts.utils.text.characters.MulticharPhonemes"


def ruslan_phone_formatter(root_path, meta_file, ignored_speakers=None, **kwargs):
    """Read our persistent manifest and pass precomputed phones to Coqui."""

    del ignored_speakers, kwargs
    root = Path(root_path).resolve()
    manifest_path = Path(meta_file)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    items = []
    with manifest_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            items.append(
                {
                    "text": " ".join(row["phones"].split()),
                    "audio_file": str((root / Path(row["audio_relpath"])).resolve()),
                    "speaker_name": "RUSLAN",
                    "root_path": str(root),
                }
            )
    return items


def register_ruslan_formatter() -> None:
    from TTS.tts.datasets.formatters import register_formatter

    try:
        register_formatter("ruslan_phones", ruslan_phone_formatter)
    except ValueError as error:
        # Re-running a notebook cell should be idempotent.
        if "already exists" not in str(error):
            raise


def phone_inventory(splits: dict[str, pd.DataFrame]) -> list[str]:
    inventory = sorted({phone for frame in splits.values() for phones in frame["phones"] for phone in str(phones).split()})
    if "sil" not in inventory:
        raise ValueError("В алфавите нет токена sil; паузы потеряны")
    return inventory


def build_config(
    repo_root: Path,
    splits: dict[str, pd.DataFrame],
    *,
    mode: str = "smoke",
    batch_size: int | None = None,
    epochs: int | None = None,
    num_workers: int = 0,
):
    """Create the exact FastSpeech 2 configuration for smoke or full training."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode должен быть 'smoke' или 'full'")
    import torch
    from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
    from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
    from TTS.tts.configs.shared_configs import CharactersConfig
    from TTS.tts.models.forward_tts import ForwardTTSArgs

    audio_settings = AcousticSettings()
    data_dir = repo_root / "data" / "lab4_acoustic"
    split_dir = data_dir / "splits"
    output_dir = data_dir / "runs"
    inventory = phone_inventory(splits)

    dataset_config = BaseDatasetConfig(
        formatter="ruslan_phones",
        dataset_name="ruslan_mfa_phones",
        path=str(repo_root.resolve()),
        meta_file_train=str((split_dir / "train.csv").relative_to(repo_root)),
        meta_file_val=str((split_dir / "validation.csv").relative_to(repo_root)),
        meta_file_attn_mask=str((data_dir / "metadata_attn_mask.txt").resolve()),
        language="",
    )
    audio_config = BaseAudioConfig(
        sample_rate=audio_settings.sample_rate,
        fft_size=audio_settings.fft_size,
        win_length=audio_settings.win_length,
        hop_length=audio_settings.hop_length,
        num_mels=audio_settings.num_mels,
        do_trim_silence=audio_settings.trim_silence,
        signal_norm=False,
        mel_fmin=audio_settings.mel_fmin,
        mel_fmax=audio_settings.mel_fmax,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
        pitch_fmin=45.0,
        pitch_fmax=500.0,
    )
    model_args = ForwardTTSArgs(
        use_aligner=False,
        use_pitch=True,
        use_energy=True,
        hidden_channels=384,
        max_duration=400,
    )
    characters = CharactersConfig(
        characters_class=MULTICHAR_CLASS,
        characters=inventory,
        punctuations="",
        pad="<PAD>",
        eos=None,
        bos=None,
        blank=None,
        is_unique=True,
        is_sorted=True,
    )

    is_smoke = mode == "smoke"
    config = Fastspeech2Config(
        run_name=f"ruslan_fastspeech2_{mode}",
        model_args=model_args,
        audio=audio_config,
        characters=characters,
        datasets=[dataset_config],
        output_path=str(output_dir.resolve()),
        training_seed=42,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        batch_size=batch_size or (4 if is_smoke else 8),
        eval_batch_size=batch_size or (4 if is_smoke else 8),
        num_loader_workers=num_workers,
        num_eval_loader_workers=num_workers,
        precompute_num_workers=num_workers,
        epochs=epochs or (2 if is_smoke else 300),
        run_eval=True,
        test_delay_epochs=-1,
        print_step=10 if is_smoke else 50,
        plot_step=50 if is_smoke else 500,
        save_step=100 if is_smoke else 5000,
        save_n_checkpoints=2,
        mixed_precision=bool(torch.cuda.is_available()),
        use_phonemes=False,
        text_cleaner="multicharacter_phoneme_cleaners",
        add_blank=False,
        enable_eos_bos_chars=False,
        compute_input_seq_cache=False,
        compute_f0=True,
        # Smoke statistics must never normalize the full run.
        f0_cache_path=str((data_dir / f"f0_cache_{mode}").resolve()),
        compute_energy=True,
        energy_cache_path=str((data_dir / f"energy_cache_{mode}").resolve()),
        min_text_len=1,
        max_text_len=1000,
        min_audio_len=1,
        # The persistent manifest already excludes >15 s outliers.  Keep a
        # defensive loader limit with one STFT-window of numerical headroom.
        max_audio_len=int(15.0 * audio_settings.sample_rate + audio_settings.win_length),
        start_by_longest=True,
        shuffle=True,
        drop_last=True,
        max_seq_len=10_000,
        test_sentences=splits["test"]["phones"].head(3).astype(str).tolist(),
    )
    return config


def _filter_samples(samples: list[dict], selected_ids: set[str] | None) -> list[dict]:
    if selected_ids is None:
        return samples
    return [sample for sample in samples if Path(sample["audio_file"]).stem in selected_ids]


def quarantine_truncated_checkpoints(run_dir: Path, *, minimum_ratio: float = 0.8) -> list[Path]:
    """Move obviously truncated checkpoints out of Trainer's discovery glob.

    An interrupted ``torch.save`` can leave a small ``checkpoint_*.pth`` whose
    step number is newer than every valid checkpoint.  Trainer then selects it
    blindly and fails inside ``torch.load``.  Checkpoints for one run have
    nearly identical sizes, so a file below 80% of the largest checkpoint/best
    model is safely treated as incomplete.  It is preserved for inspection.
    """

    run_dir = Path(run_dir).resolve()
    checkpoints = list(run_dir.glob("checkpoint_*.pth"))
    references = checkpoints + list(run_dir.glob("best_model*.pth"))
    if not checkpoints or len(references) < 2:
        return []
    reference_size = max(path.stat().st_size for path in references)
    minimum_size = int(reference_size * minimum_ratio)
    truncated = [path for path in checkpoints if path.stat().st_size < minimum_size]
    if not truncated:
        return []

    quarantine_dir = run_dir / "incomplete_checkpoints"
    quarantine_dir.mkdir(exist_ok=True)
    moved = []
    for path in truncated:
        target = quarantine_dir / f"{path.name}.incomplete"
        suffix = 1
        while target.exists():
            target = quarantine_dir / f"{path.name}.{suffix}.incomplete"
            suffix += 1
        path.replace(target)
        moved.append(target)
    return moved


def initialize_training(
    repo_root: Path,
    splits: dict[str, pd.DataFrame],
    *,
    mode: str = "smoke",
    selected_ids: set[str] | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    num_workers: int = 0,
    restore_path: str | None = None,
    continue_path: str | None = None,
):
    """Build config, tokenizer, samples, model and Trainer without starting fit()."""

    if restore_path and continue_path:
        raise ValueError("restore_path и continue_path нельзя использовать одновременно")
    quarantined_checkpoints: list[Path] = []
    if continue_path:
        continue_dir = Path(continue_path).resolve()
        if not continue_dir.is_dir() or not (continue_dir / "config.json").is_file():
            raise FileNotFoundError(f"Каталог продолжения обучения некорректен: {continue_dir}")
        quarantined_checkpoints = quarantine_truncated_checkpoints(continue_dir)

    import numpy as np
    import torch
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    register_ruslan_formatter()
    config = build_config(
        repo_root,
        splits,
        mode=mode,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
    )
    # Keep the freshly constructed multi-character inventory.  Coqpit 0.0.17
    # cannot deserialize the list-valued `characters` field from config.json
    # back into CharactersConfig and otherwise replaces it with None.
    fresh_characters = config.characters
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # A complete token audit catches Unicode/OOV errors before hours of training.
    for split_name, frame in splits.items():
        for row in frame.itertuples(index=False):
            ids = tokenizer.text_to_ids(str(row.phones))
            if len(ids) != int(row.phone_count):
                raise ValueError(
                    f"{split_name}/{row.utterance_id}: ожидалось {row.phone_count} токенов, получено {len(ids)}"
                )
    if tokenizer.not_found_characters:
        raise ValueError(f"Фонемы вне словаря: {tokenizer.not_found_characters}")

    train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)
    train_samples = _filter_samples(train_samples, selected_ids)
    eval_samples = _filter_samples(eval_samples, selected_ids)
    if not train_samples or not eval_samples:
        raise ValueError("После выбора подмножества train или validation оказался пустым")

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    if continue_path:
        # `continue_path` restores the model, optimizer, scheduler, AMP scaler,
        # epoch and global step.  In contrast, `restore_path` intentionally
        # loads only model weights and starts a new run from epoch zero.
        trainer_args = TrainerArgs(continue_path=str(continue_dir))
    elif restore_path:
        trainer_args = TrainerArgs(restore_path=restore_path)
    else:
        trainer_args = TrainerArgs()
    trainer = Trainer(
        trainer_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    resume_checkpoint = trainer.args.restore_path or None
    if continue_path:
        # Trainer reloads the original config.json for a continued run.  Apply
        # the requested DataLoader concurrency afterwards so a CPU-bound run
        # can be resumed with faster input preparation.
        trainer.config.characters = fresh_characters
        trainer.config.num_loader_workers = num_workers
        trainer.config.num_eval_loader_workers = num_workers
        trainer.config.precompute_num_workers = num_workers
        trainer.config.save_json(str(trainer.output_path / "config.json"))
        # trainer 0.3.3 prints best_path in _restore_best_loss(), but loads
        # restore_path by mistake.  The model/optimizer have already been
        # restored above, so point this argument at the real best checkpoint
        # before fit() restores only the historical best-loss value.
        if trainer.args.best_path:
            trainer.args.restore_path = trainer.args.best_path
    details = {
        "mode": mode,
        "train_samples": len(train_samples),
        "validation_samples": len(eval_samples),
        "vocabulary_size": len(phone_inventory(splits)),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mixed_precision": config.mixed_precision,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "num_workers": config.num_loader_workers,
        "output_path": str(trainer.output_path),
        "continue_path": str(continue_dir) if continue_path else None,
        "resume_checkpoint": resume_checkpoint,
        "resume_epoch": trainer.epochs_done,
        "resume_global_step": trainer.total_steps_done,
        "quarantined_checkpoints": [str(path) for path in quarantined_checkpoints],
        "course_tts_commit": COURSE_TTS_COMMIT,
    }
    return trainer, config, tokenizer, details

def main() -> None:
    """Run the upstream course CLI without changing the notebook-facing API."""
    from train_fastspeech2_course import main as course_main

    course_main()


if __name__ == "__main__":
    main()
