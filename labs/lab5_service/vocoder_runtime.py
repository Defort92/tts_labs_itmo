"""Coqui neural-vocoder loading and mel-to-waveform conversion."""


from pathlib import Path

import numpy as np

from .acoustic_runtime import select_device
from .errors import ArtifactError


class VocoderRuntime:
    def __init__(self, checkpoint_path: str | Path, config_path: str | Path, *, device: str = "auto") -> None:
        from TTS.config import load_config
        from TTS.utils.audio import AudioProcessor
        from TTS.vocoder.models import setup_model as setup_vocoder_model

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_path = Path(config_path).resolve()
        for path in (self.checkpoint_path, self.config_path):
            if not path.is_file():
                raise ArtifactError(f"Не найден артефакт вокодера: {path}")
        self.device = select_device(device)
        self.config = load_config(self.config_path)
        self.audio_processor = AudioProcessor(**self.config.audio)
        self.model = setup_vocoder_model(self.config)
        self.model.load_checkpoint(self.config, self.checkpoint_path, eval=True)
        self.model = self.model.to(self.device).eval()
        self.sample_rate = int(self.config.audio["sample_rate"])
        self.num_mels = int(self.config.audio["num_mels"])

    def validate_compatibility(self, acoustic: "AcousticRuntime") -> None:
        mismatches = []
        for key, acoustic_value in acoustic.audio_settings.items():
            vocoder_value = self.config.audio.get(key)
            if acoustic_value != vocoder_value:
                mismatches.append(f"{key}: acoustic={acoustic_value!r}, vocoder={vocoder_value!r}")
        if mismatches:
            raise ArtifactError("Несовместимые параметры mel:\n" + "\n".join(mismatches))

    def infer(self, mel: np.ndarray, acoustic: "AcousticRuntime") -> np.ndarray:
        import torch

        # Convert from the acoustic model's representation to the normalization
        # expected by the independently trained vocoder, following Coqui's own
        # Synthesizer implementation.
        denormalized = acoustic.audio_processor.denormalize(mel.T).T
        normalized = self.audio_processor.normalize(denormalized.T)
        vocoder_input = torch.as_tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            waveform = self.model.inference(vocoder_input)
        waveform = waveform.detach().cpu().numpy().squeeze().astype(np.float32, copy=False)
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError(f"Вокодер вернул некорректную форму: {waveform.shape}")
        if not np.isfinite(waveform).all():
            raise RuntimeError("Вокодер вернул NaN или бесконечные значения")
        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak * 0.99
        return np.clip(waveform, -1.0, 1.0)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acoustic_runtime import AcousticRuntime
