"""Configuration loading for the synthesis service."""


from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import ArtifactError


@dataclass(frozen=True)
class ServiceSettings:
    """Resolved, immutable runtime settings."""

    config_path: Path
    acoustic_checkpoint: Path
    acoustic_config: Path
    pause_model: Path
    pronunciation_dictionary: Path
    vocoder_checkpoint: Path
    vocoder_config: Path
    device: str
    max_text_chars: int
    edge_silence_ms: float
    preload_models: bool

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ServiceSettings":
        lab_root = Path(__file__).resolve().parent
        path = Path(config_path).resolve() if config_path else lab_root / "config.yaml"
        if not path.is_file():
            raise ArtifactError(f"Не найден файл конфигурации: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()

        try:
            acoustic = payload["acoustic"]
            frontend = payload["frontend"]
            vocoder = payload["vocoder"]
            service = payload["service"]
            return cls(
                config_path=path,
                acoustic_checkpoint=resolve(acoustic["checkpoint"]),
                acoustic_config=resolve(acoustic["config"]),
                pause_model=resolve(frontend["pause_model"]),
                pronunciation_dictionary=resolve(frontend["pronunciation_dictionary"]),
                vocoder_checkpoint=resolve(vocoder["checkpoint"]),
                vocoder_config=resolve(vocoder["config"]),
                device=str(service.get("device", "auto")),
                max_text_chars=int(service.get("max_text_chars", 500)),
                edge_silence_ms=float(service.get("edge_silence_ms", 80.0)),
                preload_models=bool(service.get("preload_models", False)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"Некорректная конфигурация {path}: {error}") from error

    def required_artifacts(self, *, include_vocoder: bool = True) -> list[Path]:
        paths = [
            self.acoustic_checkpoint,
            self.acoustic_config,
            self.pause_model,
            self.pronunciation_dictionary,
        ]
        if include_vocoder:
            paths.extend([self.vocoder_checkpoint, self.vocoder_config])
        return paths

    def validate_artifacts(self, *, include_vocoder: bool = True) -> None:
        missing = [path for path in self.required_artifacts(include_vocoder=include_vocoder) if not path.is_file()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise ArtifactError(f"Не найдены артефакты сервиса:\n{formatted}")

