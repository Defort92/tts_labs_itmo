"""End-to-end text-to-WAV synthesis pipeline."""


import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .acoustic_runtime import AcousticRuntime
from .frontend import FrontendResult, TextFrontend
from .runtime_g2p import RuntimeG2P
from .settings import ServiceSettings
from .vocoder_runtime import VocoderRuntime


@dataclass(frozen=True)
class SynthesisResult:
    wav_bytes: bytes
    sample_rate: int
    audio_duration_s: float
    synthesis_time_s: float
    real_time_factor: float
    mel_frames: int
    frontend: FrontendResult
    phone_durations: tuple[int, ...]
    timings: dict[str, float]

    def metadata(self) -> dict[str, Any]:
        return {
            "original_text": self.frontend.original_text,
            "normalized_text": self.frontend.normalized_text,
            "words": [
                {"word": word.spelling, "normalized": word.normalized, "phones": list(word.phones)}
                for word in self.frontend.words
            ],
            "phones": list(self.frontend.phones),
            "pauses": list(self.frontend.pauses),
            "duration_overrides": self.frontend.duration_overrides,
            "phone_durations_frames": list(self.phone_durations),
            "mel_frames": self.mel_frames,
            "sample_rate": self.sample_rate,
            "audio_duration_s": self.audio_duration_s,
            "synthesis_time_s": self.synthesis_time_s,
            "real_time_factor": self.real_time_factor,
            "timings": self.timings,
        }


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Для записи WAV ожидается конечный непустой одномерный сигнал")
    target = io.BytesIO()
    sf.write(target, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return target.getvalue()


class TTSPipeline:
    """Load every model once and serialize access to the shared GPU models."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        settings.validate_artifacts()
        repo_root = Path(__file__).resolve().parents[2]
        self.acoustic = AcousticRuntime(
            settings.acoustic_checkpoint, settings.acoustic_config, device=settings.device
        )
        self.g2p = RuntimeG2P(settings.pronunciation_dictionary, set(self.acoustic.phone_inventory))
        self.frontend = TextFrontend(
            repo_root,
            settings.pause_model,
            self.g2p,
            sample_rate=self.acoustic.sample_rate,
            hop_length=self.acoustic.hop_length,
            max_text_chars=settings.max_text_chars,
            edge_silence_ms=settings.edge_silence_ms,
        )
        self.vocoder = VocoderRuntime(
            settings.vocoder_checkpoint, settings.vocoder_config, device=settings.device
        )
        self.vocoder.validate_compatibility(self.acoustic)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "TTSPipeline":
        return cls(ServiceSettings.load(config_path))

    @property
    def device(self) -> str:
        return str(self.acoustic.device)

    def analyze(self, text: str) -> FrontendResult:
        return self.frontend.prepare(text)

    def synthesize(self, text: str) -> SynthesisResult:
        started = time.perf_counter()
        with self._lock:
            frontend_started = time.perf_counter()
            prepared = self.frontend.prepare(text)
            frontend_seconds = time.perf_counter() - frontend_started

            acoustic_started = time.perf_counter()
            mel, phone_durations = self.acoustic.infer(prepared.phones, prepared.duration_overrides)
            acoustic_seconds = time.perf_counter() - acoustic_started

            vocoder_started = time.perf_counter()
            waveform = self.vocoder.infer(mel, self.acoustic)
            vocoder_seconds = time.perf_counter() - vocoder_started

        wav_bytes = waveform_to_wav_bytes(waveform, self.vocoder.sample_rate)
        total_seconds = time.perf_counter() - started
        audio_duration = len(waveform) / self.vocoder.sample_rate
        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=self.vocoder.sample_rate,
            audio_duration_s=audio_duration,
            synthesis_time_s=total_seconds,
            real_time_factor=total_seconds / audio_duration,
            mel_frames=int(mel.shape[0]),
            frontend=prepared,
            phone_durations=tuple(int(value) for value in phone_durations),
            timings={
                "frontend_s": frontend_seconds,
                "acoustic_s": acoustic_seconds,
                "vocoder_s": vocoder_seconds,
            },
        )

