"""Copy the frozen acoustic model from laboratory 4 into the service directory."""


import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


NUMBERED_BEST_RE = re.compile(r"best_model_(\d+)\.pth$")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_checkpoint(source_dir: Path) -> Path:
    """Select the frozen checkpoint from a Lab 4 artifact or run directory."""

    manifest_path = source_dir / "model_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = source_dir / str(manifest.get("checkpoint", ""))
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint из model_manifest.json не найден: {candidate}")
        expected_hash = manifest.get("sha256")
        if expected_hash and sha256(candidate) != expected_hash:
            raise ValueError(f"SHA-256 checkpoint не совпадает с model_manifest.json: {candidate}")
        return candidate

    numbered: list[tuple[int, Path]] = []
    for candidate in source_dir.glob("best_model_*.pth"):
        match = NUMBERED_BEST_RE.fullmatch(candidate.name)
        if match:
            numbered.append((int(match.group(1)), candidate))
    if numbered:
        # Coqui keeps improving best checkpoints under increasing global-step names.
        return max(numbered, key=lambda item: item[0])[1]
    fallback = source_dir / "best_model.pth"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"В {source_dir} не найден best_model*.pth")


def select_config(source_dir: Path) -> Path:
    for name in ("training_config.json", "config.json"):
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"В {source_dir} не найден training_config.json или config.json")


def copy_if_needed(source: Path, destination: Path, *, force: bool) -> tuple[str, bool]:
    source_hash = sha256(source)
    if destination.is_file():
        destination_hash = sha256(destination)
        if destination_hash == source_hash:
            return source_hash, False
        if not force:
            raise FileExistsError(
                f"{destination} уже существует и отличается. Проверьте источник и добавьте --force для замены."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise OSError(f"Ошибка проверки скопированного файла: {destination}")
    temporary.replace(destination)
    return source_hash, True


def prepare(source_dir: Path, destination_dir: Path, *, force: bool = False) -> dict:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    checkpoint = select_checkpoint(source_dir)
    config = select_config(source_dir)
    destination_checkpoint = destination_dir / "model.pth"
    destination_config = destination_dir / "config.json"
    checkpoint_hash, checkpoint_copied = copy_if_needed(checkpoint, destination_checkpoint, force=force)
    config_hash, config_copied = copy_if_needed(config, destination_config, force=force)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_dir),
        "checkpoint": {
            "source": checkpoint.name,
            "destination": destination_checkpoint.name,
            "sha256": checkpoint_hash,
            "size_bytes": destination_checkpoint.stat().st_size,
            "copied": checkpoint_copied,
        },
        "config": {
            "source": config.name,
            "destination": destination_config.name,
            "sha256": config_hash,
            "size_bytes": destination_config.stat().st_size,
            "copied": config_copied,
        },
    }
    destination_dir.mkdir(parents=True, exist_ok=True)
    (destination_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    lab_root = Path(__file__).resolve().parent
    repo_root = lab_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "labs" / "lab4_acoustic" / "artifacts",
        help="Каталог artifacts Lab 4 или каталог успешного Coqui run.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=lab_root / "artifacts" / "acoustic",
    )
    parser.add_argument("--force", action="store_true", help="Заменить отличающиеся файлы назначения.")
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

