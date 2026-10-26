"""Inference-only wrapper around the FastSpeech 2 model trained in Lab 4."""


import json
import warnings
from pathlib import Path

import numpy as np

from .errors import ArtifactError


def select_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ArtifactError("В конфигурации выбрана CUDA, но torch.cuda.is_available() == False")
    return device


class AcousticRuntime:
    """Load the frozen model without requiring the RUSLAN corpus or split files."""

    def __init__(self, checkpoint_path: str | Path, config_path: str | Path, *, device: str = "auto") -> None:
        import torch
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import CharactersConfig
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_path = Path(config_path).resolve()
        for path in (self.checkpoint_path, self.config_path):
            if not path.is_file():
                raise ArtifactError(f"Не найден артефакт акустической модели: {path}")

        self.device = select_device(device)
        raw_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Type mismatch in Fastspeech2Config")
            self.config = load_config(self.config_path)
        # Coqpit declares CharactersConfig.characters as ``str`` although the
        # course's MulticharPhonemes tokenizer intentionally stores a list of
        # IPA tokens. Generic JSON deserialization therefore drops the list.
        # Recreate this one nested config exactly as it was constructed in Lab 4.
        self.config.characters = CharactersConfig(**raw_config["characters"])
        self.audio_processor = AudioProcessor.init_from_config(self.config)
        self.tokenizer, self.config = TTSTokenizer.init_from_config(self.config)
        self.model = ForwardTTS(self.config, self.audio_processor, self.tokenizer, speaker_manager=None)
        self.model.load_checkpoint(self.config, self.checkpoint_path, eval=True)
        self.model = self.model.to(self.device).eval()
        self.torch = torch

        self.phone_inventory = frozenset(raw_config["characters"]["characters"])
        self.sample_rate = int(raw_config["audio"]["sample_rate"])
        self.hop_length = int(raw_config["audio"]["hop_length"])
        self.num_mels = int(raw_config["audio"]["num_mels"])
        compatibility_keys = (
            "sample_rate",
            "fft_size",
            "win_length",
            "hop_length",
            "num_mels",
            "mel_fmin",
            "mel_fmax",
            "log_func",
            "signal_norm",
            "spec_gain",
            "ref_level_db",
            "preemphasis",
        )
        self.audio_settings = {key: raw_config["audio"].get(key) for key in compatibility_keys}

    def infer(self, phones: tuple[str, ...] | list[str], duration_overrides: dict[int, int]) -> tuple[np.ndarray, list[int]]:
        """Generate a mel and replace selected token durations before decoding."""

        torch = self.torch
        from TTS.tts.utils.helpers import sequence_mask

        phone_list = list(phones)
        unknown = sorted(set(phone_list) - self.phone_inventory)
        if unknown:
            raise ValueError(f"Неизвестные фонемы FastSpeech 2: {unknown}")
        token_ids = self.tokenizer.text_to_ids(" ".join(phone_list))
        if len(token_ids) != len(phone_list):
            raise RuntimeError(
                f"Токенизатор изменил длину последовательности: phones={len(phone_list)}, ids={len(token_ids)}"
            )
        inputs = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        x_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            x_mask = torch.unsqueeze(sequence_mask(x_lengths, inputs.shape[1]), 1).to(inputs.dtype).float()
            encoded, x_mask, _, _ = self.model._forward_encoder(inputs, x_mask, None)
            duration_logs = self.model.duration_predictor(encoded.squeeze(), x_mask)
            durations = self.model.format_durations(duration_logs, x_mask).squeeze(1).to(dtype=torch.long)
            for token_index, frame_count in duration_overrides.items():
                if token_index < 0 or token_index >= durations.shape[1]:
                    raise IndexError(f"Некорректный индекс длительности: {token_index}")
                if frame_count < 1 or frame_count > int(self.model.max_duration):
                    raise ValueError(
                        f"Длительность токена {token_index} равна {frame_count}; "
                        f"допустимо 1..{int(self.model.max_duration)} кадров"
                    )
                durations[0, token_index] = int(frame_count)
            y_lengths = durations.sum(dim=1)
            if self.model.args.use_pitch:
                pitch_embedding, _ = self.model._forward_pitch_predictor(encoded, x_mask)
                encoded = encoded + pitch_embedding
            if self.model.args.use_energy:
                energy_embedding, _ = self.model._forward_energy_predictor(encoded, x_mask)
                encoded = encoded + energy_embedding
            decoded, _ = self.model._forward_decoder(encoded, durations, x_mask, y_lengths, g=None)
            mel = decoded[0].detach().cpu().numpy().astype(np.float32, copy=False)

        if mel.ndim != 2 or mel.shape[1] != self.num_mels:
            raise RuntimeError(f"Некорректная форма mel: {mel.shape}, ожидалось [T, {self.num_mels}]")
        if not np.isfinite(mel).all():
            raise RuntimeError("FastSpeech 2 вернул NaN или бесконечные значения")
        return mel, durations[0].detach().cpu().tolist()
