"""Download the pinned Coqui HiFi-GAN vocoder into Lab 5 runtime artifacts."""


import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "vocoder_models/en/ljspeech/hifigan_v2"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(output_dir: Path, model_name: str = DEFAULT_MODEL) -> dict:
    from TTS.utils.manage import ModelManager

    output_dir = output_dir.resolve()
    manager = ModelManager(output_prefix=output_dir, progress_bar=True)
    checkpoint, config, model_item = manager.download_model(model_name)
    if checkpoint is None or config is None:
        raise RuntimeError(f"Coqui не вернул checkpoint/config для {model_name}")
    checkpoint = Path(checkpoint).resolve()
    config = Path(config).resolve()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "checkpoint": str(checkpoint.relative_to(output_dir)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "config": str(config.relative_to(output_dir)),
        "config_sha256": sha256(config),
        "license": model_item.get("license"),
        "source": model_item.get("github_rls_url") or model_item.get("hf_url"),
    }
    (output_dir / "vocoder_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    lab_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=lab_root / "artifacts" / "vocoder")
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

