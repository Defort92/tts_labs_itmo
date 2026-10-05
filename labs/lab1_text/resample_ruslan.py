"""Resample a directory of WAV files while keeping the source untouched."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import wave


def has_sample_rate(path: Path, sample_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getframerate() == sample_rate
    except (OSError, EOFError, wave.Error):
        return False


def convert(ffmpeg: Path, source: Path, destination: Path, sample_rate: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and has_sample_rate(destination, sample_rate):
        return "reused"
    temporary = destination.with_suffix(".tmp.wav")
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source.name}: {completed.stderr.strip()}")
    if not has_sample_rate(temporary, sample_rate):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unexpected sample rate for {source.name}")
    temporary.replace(destination)
    return "converted"


def resample(
    ffmpeg: Path, source_directory: Path, destination_directory: Path,
    sample_rate: int, workers: int,
) -> dict[str, int]:
    sources = sorted(source_directory.glob("*.wav"))
    if not sources:
        raise FileNotFoundError(f"No WAV files found in {source_directory}")
    counts = {"converted": 0, "reused": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {
            executor.submit(
                convert, ffmpeg, source, destination_directory / source.name, sample_rate
            ): source
            for source in sources
        }
        for number, future in enumerate(as_completed(jobs), start=1):
            result = future.result()
            counts[result] += 1
            if number % 500 == 0:
                print(f"Processed {number}/{len(sources)}", flush=True)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    statistics = resample(
        arguments.ffmpeg,
        arguments.source,
        arguments.destination,
        arguments.sample_rate,
        arguments.workers,
    )
    print(f"Done: {statistics}", flush=True)
