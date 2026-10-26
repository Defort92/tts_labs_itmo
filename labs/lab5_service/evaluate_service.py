"""Synthesize deterministic unseen sentences and save Lab 5 timing metrics."""

import argparse
import json
from pathlib import Path

from .pipeline import TTSPipeline

DEFAULT_SENTENCES = (
    "Сегодня хорошая погода.",
    "Он остановился, потом медленно продолжил путь.",
    "Вечером над городом поднялся холодный ветер.",
)


def main() -> None:
    lab_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=lab_root / "config.yaml")
    parser.add_argument("--output", type=Path, default=lab_root / "examples")
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Своя тестовая фраза; можно повторить.",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TTSPipeline.from_config(args.config)
    rows = []
    for index, text in enumerate(args.texts or DEFAULT_SENTENCES, start=1):
        result = pipeline.synthesize(text)
        wav_path = output_dir / f"sample_{index:02d}.wav"
        wav_path.write_bytes(result.wav_bytes)
        metadata = result.metadata() | {"wav": wav_path.name}
        rows.append(metadata)
        print(
            f"{wav_path.name}: audio={result.audio_duration_s:.3f}s, "
            f"synthesis={result.synthesis_time_s:.3f}s, RTF={result.real_time_factor:.3f}"
        )
    (output_dir / "synthesis_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
