"""Data preparation helpers for laboratory work 4.

The module deliberately has no Coqui-TTS imports.  Dataset splitting and MFA
duration conversion can therefore be tested before the GPU environment is
installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_SCHEMA_VERSION = 2
MANIFEST_NAMES = ("train", "validation", "test")
MANIFEST_COLUMNS = (
    "utterance_id",
    "audio_relpath",
    "phones",
    "phone_count",
    "duration_s",
    "alignment_method",
    "split",
)


@dataclass(frozen=True)
class SplitSettings:
    """Parameters that uniquely define the persistent dataset split."""

    seed: int = 42
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    duration_bins: int = 10
    exclude_non_mfa: bool = True
    max_duration_s: float = 15.0


@dataclass(frozen=True)
class AcousticSettings:
    """Audio parameters shared by attention masks and FastSpeech 2."""

    sample_rate: int = 22_050
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    center: bool = True
    trim_silence: bool = False


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root independently of the notebook working directory."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "lab_reqs.md").is_file() and (candidate / "labs").is_dir():
            return candidate
    raise FileNotFoundError("Не найден корень репозитория с lab_reqs.md")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lab3_source_fingerprint(repo_root: Path) -> str:
    """Fingerprint both the speech-only summary and intervals containing sil."""

    paths = (
        repo_root / "data" / "lab3_g2p" / "utterance_phonemes.csv",
        repo_root / "data" / "lab3_g2p" / "phoneme_alignments.csv",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _wav_info(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()


def build_source_manifest(repo_root: Path, settings: SplitSettings) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Join Lab 3 phone sequences with audio and audit the immutable inputs."""

    phone_path = repo_root / "data" / "lab3_g2p" / "utterance_phonemes.csv"
    alignment_path = repo_root / "data" / "lab3_g2p" / "phoneme_alignments.csv"
    audio_dir = repo_root / "data" / "RUSLAN" / "RUSLAN"
    if not phone_path.is_file():
        raise FileNotFoundError(f"Нет результата лабораторной №3: {phone_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Нет каталога WAV: {audio_dir}")
    if not alignment_path.is_file():
        raise FileNotFoundError(f"Нет интервальной фонемной разметки: {alignment_path}")

    source = pd.read_csv(phone_path, dtype={"utterance_id": "string", "phones": "string"})
    required = {"utterance_id", "phone_count", "phones", "alignment_method"}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(f"В utterance_phonemes.csv нет столбцов: {sorted(missing_columns)}")
    if source["utterance_id"].duplicated().any():
        duplicates = source.loc[source["utterance_id"].duplicated(), "utterance_id"].tolist()[:5]
        raise ValueError(f"Повторяющиеся utterance_id: {duplicates}")

    # utterance_phonemes.csv intentionally omits sil.  The acoustic model must
    # retain those tokens, so its actual input sequence is rebuilt from the
    # ordered interval table and checked against the speech-only summary.
    intervals = pd.read_csv(
        alignment_path,
        usecols=["utterance_id", "phone_index", "phone", "alignment_method"],
        dtype={"utterance_id": "string", "phone": "string", "alignment_method": "string"},
    ).sort_values(["utterance_id", "phone_index"])
    interval_groups = {str(key): value for key, value in intervals.groupby("utterance_id", sort=False)}

    rows: list[dict] = []
    missing_audio: list[str] = []
    invalid_audio: list[str] = []
    for record in source.itertuples(index=False):
        utterance_id = str(record.utterance_id)
        phone_rows = interval_groups.get(utterance_id)
        if phone_rows is None or phone_rows.empty:
            raise ValueError(f"Нет интервальной разметки для {utterance_id}")
        methods = phone_rows["alignment_method"].astype(str).unique().tolist()
        if len(methods) != 1:
            raise ValueError(f"{utterance_id}: смешаны методы выравнивания {methods}")
        interval_phones = phone_rows["phone"].astype(str).tolist()
        speech_phones = [phone for phone in interval_phones if phone != "sil"]
        summary_phones = str(record.phones).split()
        if speech_phones != summary_phones:
            raise ValueError(f"{utterance_id}: речевые фонемы не совпадают между файлами лабораторной №3")
        relpath = Path("data") / "RUSLAN" / "RUSLAN" / f"{utterance_id}.wav"
        wav_path = repo_root / relpath
        if not wav_path.is_file():
            missing_audio.append(utterance_id)
            continue
        sample_rate, channels, sample_width, frame_count = _wav_info(wav_path)
        if sample_rate != 22_050 or channels != 1 or sample_width != 2 or frame_count <= 0:
            invalid_audio.append(
                f"{utterance_id}: sr={sample_rate}, channels={channels}, width={sample_width}, frames={frame_count}"
            )
            continue
        phones = " ".join(interval_phones)
        actual_phone_count = len(phones.split())
        if len(speech_phones) != int(record.phone_count):
            raise ValueError(
                f"{utterance_id}: speech phone_count={record.phone_count}, фактически {len(speech_phones)}"
            )
        rows.append(
            {
                "utterance_id": utterance_id,
                "audio_relpath": relpath.as_posix(),
                "phones": phones,
                "phone_count": actual_phone_count,
                "duration_s": frame_count / sample_rate,
                "alignment_method": methods[0],
            }
        )

    if missing_audio:
        raise FileNotFoundError(f"Не найдены WAV ({len(missing_audio)}), первые: {missing_audio[:5]}")
    if invalid_audio:
        raise ValueError(f"Некорректные WAV ({len(invalid_audio)}), первые: {invalid_audio[:5]}")

    complete = pd.DataFrame(rows).sort_values("utterance_id").reset_index(drop=True)
    reasons = pd.Series("", index=complete.index, dtype="string")
    if settings.exclude_non_mfa:
        reasons.loc[complete["alignment_method"].ne("mfa")] = "non_mfa_alignment"
    too_long = complete["duration_s"].gt(settings.max_duration_s)
    reasons.loc[too_long & reasons.eq("")] = f"duration_above_{settings.max_duration_s:g}s"
    reasons.loc[too_long & reasons.ne("")] += f";duration_above_{settings.max_duration_s:g}s"
    excluded_mask = reasons.ne("")
    excluded = complete.loc[excluded_mask].copy()
    excluded["reason"] = reasons.loc[excluded_mask].to_numpy()
    eligible = complete.loc[~excluded_mask].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("После фильтрации не осталось записей")

    fingerprint_payload = "\n".join(
        f"{row.utterance_id}|{row.phones}|{row.alignment_method}|{row.duration_s:.9f}"
        for row in eligible.itertuples(index=False)
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return eligible, excluded, fingerprint


def _duration_labels(frame: pd.DataFrame, bins: int) -> pd.Series:
    actual_bins = min(bins, max(2, len(frame) // 20))
    labels = pd.qcut(frame["duration_s"], q=actual_bins, labels=False, duplicates="drop")
    if labels.nunique() < 2:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    return labels.astype(int)


def split_manifest(frame: pd.DataFrame, settings: SplitSettings) -> dict[str, pd.DataFrame]:
    """Create a deterministic duration-stratified 90/5/5 style split."""

    holdout_fraction = settings.validation_fraction + settings.test_fraction
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("Сумма validation_fraction и test_fraction должна быть между 0 и 1")

    labels = _duration_labels(frame, settings.duration_bins)
    train, holdout = train_test_split(
        frame,
        test_size=holdout_fraction,
        random_state=settings.seed,
        shuffle=True,
        stratify=labels,
    )
    holdout_labels = _duration_labels(holdout, settings.duration_bins)
    test_share = settings.test_fraction / holdout_fraction
    validation, test = train_test_split(
        holdout,
        test_size=test_share,
        random_state=settings.seed,
        shuffle=True,
        stratify=holdout_labels,
    )

    result = {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part = part.copy().sort_values("utterance_id").reset_index(drop=True)
        part["split"] = name
        result[name] = part.loc[:, MANIFEST_COLUMNS]
    validate_splits(result, expected_ids=set(frame["utterance_id"]))
    return result


def validate_splits(splits: dict[str, pd.DataFrame], expected_ids: set[str] | None = None) -> None:
    """Fail loudly on leakage, duplicates, missing audio or malformed phones."""

    if set(splits) != set(MANIFEST_NAMES):
        raise ValueError(f"Ожидаются разбиения {MANIFEST_NAMES}, получено {sorted(splits)}")
    id_sets: dict[str, set[str]] = {}
    for name, frame in splits.items():
        missing = set(MANIFEST_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: отсутствуют столбцы {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name}: пустое разбиение")
        if frame["utterance_id"].duplicated().any():
            raise ValueError(f"{name}: повторяющиеся utterance_id")
        if not frame["split"].eq(name).all():
            raise ValueError(f"{name}: неверное значение столбца split")
        if (frame["phone_count"].astype(int) <= 0).any() or frame["phones"].isna().any():
            raise ValueError(f"{name}: пустая фонемная последовательность")
        id_sets[name] = set(frame["utterance_id"].astype(str))

    for index, left in enumerate(MANIFEST_NAMES):
        for right in MANIFEST_NAMES[index + 1 :]:
            overlap = id_sets[left].intersection(id_sets[right])
            if overlap:
                raise ValueError(f"Утечка между {left} и {right}: {sorted(overlap)[:5]}")
    union = set().union(*id_sets.values())
    if expected_ids is not None and union != expected_ids:
        missing = sorted(expected_ids.difference(union))[:5]
        unexpected = sorted(union.difference(expected_ids))[:5]
        raise ValueError(f"Разбиение не покрывает исходные данные: missing={missing}, unexpected={unexpected}")


def _split_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / f"{name}.csv" for name in MANIFEST_NAMES}


def create_or_load_splits(
    repo_root: Path,
    settings: SplitSettings = SplitSettings(),
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Reuse a valid split or create it once.

    Existing files are never silently regenerated.  If the source/configuration
    changed, the caller must explicitly pass ``force=True``.
    """

    output_dir = repo_root / "data" / "lab4_acoustic" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = _split_paths(output_dir)
    metadata_path = output_dir / "split_metadata.json"
    excluded_path = output_dir / "excluded.csv"
    any_existing = metadata_path.exists() or any(path.exists() for path in manifest_paths.values())
    all_existing = metadata_path.is_file() and all(path.is_file() for path in manifest_paths.values())

    if all_existing and not force:
        actual_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_lab3_fingerprint = lab3_source_fingerprint(repo_root)
        if actual_meta.get("schema_version") != SPLIT_SCHEMA_VERSION or actual_meta.get("settings") != asdict(settings):
            raise RuntimeError(
                "Сохранённое разбиение создано из других данных или параметров. "
                "Проверьте изменения и только затем вызовите create_or_load_splits(..., force=True)."
            )
        stored_lab3_fingerprint = actual_meta.get("lab3_source_fingerprint")
        if stored_lab3_fingerprint != current_lab3_fingerprint:
            raise RuntimeError(
                "Фонемные последовательности или интервалы изменились после создания разбиения. "
                "Проверьте лабораторную №3 и затем явно пересоздайте split с force=True."
            )
        splits = {name: pd.read_csv(path, dtype={"utterance_id": "string"}) for name, path in manifest_paths.items()}
        validate_splits(splits)
        if sum(len(frame) for frame in splits.values()) != int(actual_meta["source_count"]):
            raise RuntimeError("Количество строк в split-файлах не совпадает с split_metadata.json")
        actual_meta["cache_status"] = "loaded"
        return splits, actual_meta

    if any_existing and not force:
        raise RuntimeError(
            "Найден неполный кэш разбиения. Удалять его автоматически небезопасно; "
            "после проверки вызовите create_or_load_splits(..., force=True)."
        )

    eligible, excluded, source_fingerprint = build_source_manifest(repo_root, settings)
    expected_meta = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "settings": asdict(settings),
        "source_fingerprint": source_fingerprint,
        "lab3_source_fingerprint": lab3_source_fingerprint(repo_root),
        "source_count": int(len(eligible)),
    }
    splits = split_manifest(eligible, settings)
    for name, frame in splits.items():
        frame.to_csv(manifest_paths[name], index=False, encoding="utf-8")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8")
    metadata = {
        **expected_meta,
        "counts": {name: int(len(frame)) for name, frame in splits.items()},
        "excluded_count": int(len(excluded)),
        "excluded_ids": excluded["utterance_id"].astype(str).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["cache_status"] = "created"
    return splits, metadata


def allocate_frames(durations_s: Iterable[float], total_frames: int) -> np.ndarray:
    """Convert positive phone durations to integer mel frames with exact sum.

    Every phone receives at least one frame.  Remaining rounding error is
    distributed by the largest-remainder principle.
    """

    durations = np.asarray(list(durations_s), dtype=np.float64)
    if durations.ndim != 1 or len(durations) == 0 or not np.isfinite(durations).all():
        raise ValueError("Некорректный массив длительностей")
    if (durations <= 0).any():
        raise ValueError("Все длительности фонем должны быть положительными")
    if total_frames < len(durations):
        raise ValueError(f"Кадров ({total_frames}) меньше, чем фонем ({len(durations)})")

    exact = durations / durations.sum() * total_frames
    allocated = np.maximum(1, np.floor(exact).astype(np.int64))
    delta = int(total_frames - allocated.sum())
    if delta > 0:
        order = np.argsort(-(exact - np.floor(exact)), kind="stable")
        for step in range(delta):
            allocated[order[step % len(order)]] += 1
    elif delta < 0:
        for _ in range(-delta):
            candidates = np.flatnonzero(allocated > 1)
            if len(candidates) == 0:
                raise ValueError("Невозможно скорректировать округление длительностей")
            excess = allocated[candidates] - exact[candidates]
            allocated[candidates[int(np.argmax(excess))]] -= 1
    if int(allocated.sum()) != int(total_frames) or (allocated < 1).any():
        raise AssertionError("Ошибка распределения кадров")
    return allocated


def _attention_fingerprint(repo_root: Path, settings: AcousticSettings, split_meta: dict) -> str:
    alignment_path = repo_root / "data" / "lab3_g2p" / "phoneme_alignments.csv"
    payload = {
        "settings": asdict(settings),
        "split_source_fingerprint": split_meta["source_fingerprint"],
        "alignment_sha256": sha256_file(alignment_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_attention_masks(
    repo_root: Path,
    splits: dict[str, pd.DataFrame],
    split_meta: dict,
    settings: AcousticSettings = AcousticSettings(),
    *,
    only_ids: set[str] | None = None,
) -> dict:
    """Create/resume one-hot MFA attention masks expected by Coqui ForwardTTS.

    Masks are stored as uint8 to keep the full corpus cache reasonably small.
    The metadata file always lists every sample, so a smoke subset and the full
    run use exactly the same paths.
    """

    validate_splits(splits)
    output_dir = repo_root / "data" / "lab4_acoustic"
    mask_dir = output_dir / "attention_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "attention_index.csv"
    metadata_path = output_dir / "metadata_attn_mask.txt"
    cache_meta_path = output_dir / "attention_metadata.json"

    manifest = pd.concat(splits.values(), ignore_index=True).sort_values("utterance_id")
    manifest_ids = set(manifest["utterance_id"].astype(str))
    target_ids = manifest_ids if only_ids is None else manifest_ids.intersection(map(str, only_ids))
    unknown = set() if only_ids is None else set(map(str, only_ids)).difference(manifest_ids)
    if unknown:
        raise ValueError(f"only_ids содержит неизвестные записи: {sorted(unknown)[:5]}")

    fingerprint = _attention_fingerprint(repo_root, settings, split_meta)
    if cache_meta_path.is_file():
        cached = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Параметры аудио или фонемные границы изменились. "
                "Старые attention masks нельзя смешивать с новыми данными."
            )

    align_path = repo_root / "data" / "lab3_g2p" / "phoneme_alignments.csv"
    alignments = pd.read_csv(
        align_path,
        usecols=["utterance_id", "phone_index", "phone", "start_s", "end_s", "alignment_method"],
        dtype={"utterance_id": "string", "phone": "string", "alignment_method": "string"},
    )
    alignments = alignments[alignments["utterance_id"].isin(target_ids)].copy()
    alignments.sort_values(["utterance_id", "phone_index"], inplace=True)
    grouped = {str(key): value for key, value in alignments.groupby("utterance_id", sort=False)}

    previous_index: dict[str, dict] = {}
    if index_path.is_file():
        for row in pd.read_csv(index_path, dtype={"utterance_id": "string"}).to_dict("records"):
            previous_index[str(row["utterance_id"])] = row

    built = 0
    reused = 0
    index_records = dict(previous_index)
    for row in manifest[manifest["utterance_id"].isin(target_ids)].itertuples(index=False):
        utterance_id = str(row.utterance_id)
        phone_rows = grouped.get(utterance_id)
        if phone_rows is None or phone_rows.empty:
            raise ValueError(f"Нет интервальной разметки для {utterance_id}")
        if not phone_rows["alignment_method"].eq("mfa").all():
            raise ValueError(f"{utterance_id}: для обучения разрешено только MFA-выравнивание")
        phones = phone_rows["phone"].astype(str).tolist()
        if phones != str(row.phones).split():
            raise ValueError(f"{utterance_id}: последовательности фонем в двух файлах не совпадают")

        wav_path = repo_root / Path(row.audio_relpath)
        sample_rate, channels, sample_width, audio_frames = _wav_info(wav_path)
        if (sample_rate, channels, sample_width) != (settings.sample_rate, 1, 2):
            raise ValueError(f"{utterance_id}: параметры WAV не соответствуют AcousticSettings")
        mel_frames = 1 + audio_frames // settings.hop_length if settings.center else 1 + max(
            0, (audio_frames - settings.fft_size) // settings.hop_length
        )
        mask_path = mask_dir / f"{utterance_id}_attn.npy"
        record = previous_index.get(utterance_id)
        if mask_path.is_file() and record is not None:
            cached_mask = np.load(mask_path, mmap_mode="r")
            if cached_mask.shape == (mel_frames, len(phones)) and cached_mask.dtype == np.uint8:
                reused += 1
                continue

        durations = phone_rows["end_s"].to_numpy(float) - phone_rows["start_s"].to_numpy(float)
        frame_durations = allocate_frames(durations, mel_frames)
        mask = np.zeros((mel_frames, len(phones)), dtype=np.uint8)
        cursor = 0
        for phone_index, duration in enumerate(frame_durations):
            mask[cursor : cursor + duration, phone_index] = 1
            cursor += int(duration)
        if cursor != mel_frames or not np.all(mask.sum(axis=1) == 1):
            raise AssertionError(f"{utterance_id}: некорректная attention mask")
        np.save(mask_path, mask, allow_pickle=False)
        index_records[utterance_id] = {
            "utterance_id": utterance_id,
            "mask_relpath": mask_path.relative_to(repo_root).as_posix(),
            "mel_frames": mel_frames,
            "phone_count": len(phones),
            "max_phone_frames": int(frame_durations.max()),
        }
        built += 1

    index_frame = pd.DataFrame(index_records.values()).sort_values("utterance_id")
    index_frame.to_csv(index_path, index=False, encoding="utf-8")

    # Coqui looks up masks using the exact audio path emitted by the formatter.
    with metadata_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in manifest.itertuples(index=False):
            wav_path = (repo_root / Path(row.audio_relpath)).resolve()
            mask_path = (mask_dir / f"{row.utterance_id}_attn.npy").resolve()
            output.write(f"{wav_path}|{mask_path}\n")

    complete = len(index_records.keys() & manifest_ids) == len(manifest_ids) and all(
        (mask_dir / f"{utterance_id}_attn.npy").is_file() for utterance_id in manifest_ids
    )
    cache_meta = {
        "fingerprint": fingerprint,
        "complete": complete,
        "generated_count": len(index_records.keys() & manifest_ids),
        "expected_count": len(manifest_ids),
        "settings": asdict(settings),
    }
    cache_meta_path.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**cache_meta, "built_now": built, "reused_now": reused, "metadata_path": str(metadata_path)}


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in MANIFEST_NAMES:
        frame = splits[name]
        rows.append(
            {
                "split": name,
                "utterances": len(frame),
                "hours": frame["duration_s"].sum() / 3600,
                "mean_duration_s": frame["duration_s"].mean(),
                "mean_phones": frame["phone_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def choose_smoke_ids(
    splits: dict[str, pd.DataFrame],
    *,
    train_count: int = 128,
    validation_count: int = 32,
    seed: int = 42,
) -> set[str]:
    """Choose a deterministic small subset without touching the test split."""

    validate_splits(splits)
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for name, count in (("train", train_count), ("validation", validation_count)):
        ids = splits[name]["utterance_id"].astype(str).to_numpy()
        if count <= 0 or count > len(ids):
            raise ValueError(f"{name}: размер smoke-подвыборки должен быть от 1 до {len(ids)}")
        selected.update(rng.choice(ids, size=count, replace=False).tolist())
    return selected
