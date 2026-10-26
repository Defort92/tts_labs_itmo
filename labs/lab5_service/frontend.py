"""Join normalization, validation, pause prediction and G2P."""


import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InputTextError
from .runtime_g2p import PronouncedWord, RuntimeG2P


@dataclass(frozen=True)
class FrontendResult:
    original_text: str
    normalized_text: str
    words: tuple[PronouncedWord, ...]
    phones: tuple[str, ...]
    duration_overrides: dict[int, int]
    pauses: tuple[dict[str, Any], ...]


def milliseconds_to_frames(duration_ms: float, sample_rate: int, hop_length: int) -> int:
    if duration_ms < 0:
        raise ValueError("Длительность паузы не может быть отрицательной")
    return max(1, round(duration_ms * sample_rate / (1000 * hop_length)))


def _load_previous_lab_components(repo_root: Path):
    # Labs 1 and 2 were written as standalone scripts with local imports. Adding
    # their directories once preserves their public classes without duplicating
    # training-time feature extraction in the service.
    for relative in ("labs/lab1_text", "labs/lab2_pauses"):
        path = str((repo_root / relative).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    from pause_model import PausePredictor
    from text_filter import TextFilter
    from text_normalizer import TextNormalizer

    return TextNormalizer, TextFilter, PausePredictor


class TextFrontend:
    """Prepare the exact phone sequence and pause-duration controls for FastSpeech 2."""

    def __init__(
        self,
        repo_root: str | Path,
        pause_model_path: str | Path,
        g2p: RuntimeG2P,
        *,
        sample_rate: int,
        hop_length: int,
        max_text_chars: int = 500,
        edge_silence_ms: float = 80.0,
    ) -> None:
        TextNormalizer, TextFilter, PausePredictor = _load_previous_lab_components(Path(repo_root).resolve())
        self.normalizer = TextNormalizer()
        self.filter = TextFilter()
        self.pause_predictor = PausePredictor(Path(pause_model_path))
        self.g2p = g2p
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_text_chars = max_text_chars
        self.edge_silence_ms = edge_silence_ms

    def prepare(self, text: str) -> FrontendResult:
        if not isinstance(text, str):
            raise InputTextError("Поле text должно быть строкой")
        if not text.strip():
            raise InputTextError("Текст не должен быть пустым")
        if len(text) > self.max_text_chars:
            raise InputTextError(
                f"Текст содержит {len(text)} символов; максимально допустимо {self.max_text_chars}"
            )

        normalized = self.normalizer.normalize(text)
        reasons = self.filter.rejection_reasons(normalized)
        if reasons:
            raise InputTextError("Текст содержит неподдерживаемые элементы: " + ", ".join(reasons))

        words = self.g2p.transcribe(normalized)
        pauses = self.pause_predictor.predict_text(normalized)
        if len(pauses) != max(0, len(words) - 1):
            raise RuntimeError(
                f"Число межсловных границ не совпало: words={len(words)}, boundaries={len(pauses)}"
            )

        phones: list[str] = ["sil"]
        edge_frames = milliseconds_to_frames(self.edge_silence_ms, self.sample_rate, self.hop_length)
        overrides: dict[int, int] = {0: edge_frames}
        for word_index, word in enumerate(words):
            phones.extend(word.phones)
            if word_index < len(pauses) and bool(pauses[word_index]["has_pause"]):
                phones.append("sil")
                overrides[len(phones) - 1] = milliseconds_to_frames(
                    float(pauses[word_index]["duration_ms"]), self.sample_rate, self.hop_length
                )
        phones.append("sil")
        overrides[len(phones) - 1] = edge_frames

        unknown = sorted(set(phones) - self.g2p.phone_inventory)
        if unknown:
            raise RuntimeError(f"После подготовки появились неизвестные фонемы: {unknown}")
        return FrontendResult(
            original_text=text,
            normalized_text=normalized,
            words=tuple(words),
            phones=tuple(phones),
            duration_overrides=overrides,
            pauses=tuple(pauses),
        )

