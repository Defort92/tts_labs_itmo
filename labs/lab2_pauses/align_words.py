"""Forced alignment of RUSLAN words with a Russian CTC acoustic model.

The script does not use an ASR hypothesis as the transcript.  It takes the
verified normalized text produced by lab 1 and finds the most likely CTC path
for that exact text in the audio.  The result is written both as one CSV file
and as Praat TextGrid files that are convenient for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import unicodedata
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from praatio import textgrid
from praatio.utilities.constants import Interval
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, Wav2Vec2Processor


DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
WORD_PART_RE = r"[А-Яа-яЁё][А-Яа-яЁё\u0301]*"
WORD_RE = re.compile(rf"{WORD_PART_RE}(?:[-‐‑–—]{WORD_PART_RE})*")


@dataclass(frozen=True)
class TokenPoint:
    token_index: int
    frame_index: int
    score: float
    emitted_token: bool


@dataclass(frozen=True)
class TokenSegment:
    token_index: int
    start_frame: int
    end_frame: int
    score: float
    emission_frame: int


@dataclass(frozen=True)
class PreparedTranscript:
    words: list[str]
    token_ids: list[int]
    token_to_word: list[int | None]
    acoustic_text: str


@dataclass(frozen=True)
class WordAlignment:
    word_index: int
    word: str
    start_s: float
    end_s: float
    confidence: float
    boundary_after_s: float | None


def extract_words(text: str) -> list[str]:
    """Return spoken Cyrillic words while preserving their written labels."""

    return [match.group(0) for match in WORD_RE.finditer(unicodedata.normalize("NFC", text))]


def _letter_case_for_vocab(vocab: dict[str, int]) -> str:
    lower_count = sum(char in vocab for char in "абвгдежзийклмнопрстуфхцчшщыэюя")
    upper_count = sum(char in vocab for char in "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ")
    return "lower" if lower_count >= upper_count else "upper"


def prepare_transcript(text: str, vocab: dict[str, int], delimiter: str = "|") -> PreparedTranscript:
    """Convert lab-1 text to model tokens and retain token-to-word mapping.

    Punctuation and hyphens are not acoustic CTC targets.  They remain in the
    metadata and are recovered when pause examples are built.
    """

    words = extract_words(text)
    if not words:
        raise ValueError("text contains no Cyrillic words")
    if delimiter not in vocab:
        raise ValueError(f"word delimiter {delimiter!r} is absent from model vocabulary")

    letter_case = _letter_case_for_vocab(vocab)
    token_ids: list[int] = []
    token_to_word: list[int | None] = []
    acoustic_words: list[str] = []

    for word_index, word in enumerate(words):
        acoustic_word = re.sub(r"[-‐‑–—\u0301]", "", word)
        acoustic_word = acoustic_word.lower() if letter_case == "lower" else acoustic_word.upper()
        if "ё" not in vocab and "Ё" not in vocab:
            acoustic_word = acoustic_word.replace("ё", "е").replace("Ё", "Е")

        missing = sorted({char for char in acoustic_word if char not in vocab})
        if missing:
            raise ValueError(f"unsupported characters {missing!r} in word {word!r}")

        if word_index:
            token_ids.append(vocab[delimiter])
            token_to_word.append(None)
        for char in acoustic_word:
            token_ids.append(vocab[char])
            token_to_word.append(word_index)
        acoustic_words.append(acoustic_word)

    return PreparedTranscript(words, token_ids, token_to_word, delimiter.join(acoustic_words))


def ctc_forced_alignment(
    emission: torch.Tensor,
    target_ids: Sequence[int],
    blank_id: int,
) -> tuple[list[TokenSegment], float]:
    """Find the maximum-probability CTC path for an exact target sequence."""

    if emission.ndim != 2:
        raise ValueError("emission must have shape [frames, vocabulary]")
    frame_count = emission.size(0)
    token_count = len(target_ids)
    if token_count == 0:
        raise ValueError("target sequence is empty")
    if frame_count < token_count:
        raise ValueError(f"not enough acoustic frames: {frame_count} < {token_count}")

    target = torch.tensor(target_ids, dtype=torch.long, device=emission.device)
    trellis = torch.full(
        (frame_count + 1, token_count + 1),
        -float("inf"),
        dtype=emission.dtype,
        device=emission.device,
    )
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)

    for frame_index in range(frame_count):
        stay = trellis[frame_index, 1:] + emission[frame_index, blank_id]
        advance = trellis[frame_index, :-1] + emission[frame_index, target]
        trellis[frame_index + 1, 1:] = torch.maximum(stay, advance)

    end_frame = int(torch.argmax(trellis[:, token_count]).item())
    path_score = float(trellis[end_frame, token_count].item())
    frame_index = end_frame
    token_index = token_count
    path: list[TokenPoint] = []

    while token_index > 0:
        if frame_index <= 0:
            raise ValueError("failed to backtrack the CTC alignment")
        stay_score = trellis[frame_index - 1, token_index] + emission[frame_index - 1, blank_id]
        advance_score = (
            trellis[frame_index - 1, token_index - 1]
            + emission[frame_index - 1, target[token_index - 1]]
        )
        advanced = bool(advance_score > stay_score)
        emitted_id = int(target[token_index - 1].item()) if advanced else blank_id
        probability = float(emission[frame_index - 1, emitted_id].exp().item())
        path.append(TokenPoint(token_index - 1, frame_index - 1, probability, advanced))
        frame_index -= 1
        if advanced:
            token_index -= 1

    path.reverse()
    segments: list[TokenSegment] = []
    segment_start = 0
    while segment_start < len(path):
        segment_end = segment_start + 1
        while segment_end < len(path) and path[segment_end].token_index == path[segment_start].token_index:
            segment_end += 1
        points = path[segment_start:segment_end]
        token_points = [point for point in points if point.emitted_token]
        if len(token_points) != 1:
            raise ValueError(
                f"target token {points[0].token_index} has {len(token_points)} emission frames"
            )
        segments.append(
            TokenSegment(
                token_index=points[0].token_index,
                start_frame=points[0].frame_index,
                end_frame=points[-1].frame_index + 1,
                score=token_points[0].score,
                emission_frame=token_points[0].frame_index,
            )
        )
        segment_start = segment_end

    if len(segments) != token_count:
        raise ValueError(f"alignment produced {len(segments)} token segments for {token_count} tokens")
    return segments, path_score


def word_segments(
    prepared: PreparedTranscript,
    segments: Sequence[TokenSegment],
    audio_duration_s: float,
    frame_count: int,
) -> list[WordAlignment]:
    """Merge character-level CTC segments into word intervals."""

    if len(segments) != len(prepared.token_to_word):
        raise ValueError("token segment count does not match prepared transcript")
    seconds_per_frame = audio_duration_s / frame_count
    result: list[WordAlignment] = []
    for word_index, word in enumerate(prepared.words):
        own_positions = [
            position
            for position, mapped_word in enumerate(prepared.token_to_word)
            if mapped_word == word_index
        ]
        own_segments = [segments[position] for position in own_positions]
        if not own_segments:
            raise ValueError(f"word {word!r} has no aligned tokens")
        boundary_after_s: float | None = None
        if word_index < len(prepared.words) - 1:
            delimiter_position = own_positions[-1] + 1
            if prepared.token_to_word[delimiter_position] is not None:
                raise ValueError(f"word {word!r} is not followed by a delimiter token")
            boundary_after_s = (
                segments[delimiter_position].emission_frame + 0.5
            ) * seconds_per_frame
        result.append(
            WordAlignment(
                word_index=word_index,
                word=word,
                # CTC blank frames represent transitions or silence.  Using the
                # token-emission frames here deliberately leaves those frames
                # between words instead of swallowing them into the preceding
                # word, which is essential for measuring pauses.
                start_s=own_segments[0].emission_frame * seconds_per_frame,
                end_s=(own_segments[-1].emission_frame + 1) * seconds_per_frame,
                confidence=sum(segment.score for segment in own_segments) / len(own_segments),
                boundary_after_s=boundary_after_s,
            )
        )
    return result


def read_wav_mono(path: Path, target_sample_rate: int = 16_000) -> tuple[np.ndarray, float]:
    """Read PCM WAV, convert to mono float32 and resample for Wav2Vec2."""

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got sample width {sample_width}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    duration_s = len(samples) / sample_rate
    if sample_rate != target_sample_rate:
        ratio = Fraction(target_sample_rate, sample_rate)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float32)
    return samples, duration_s


def write_textgrid(path: Path, alignments: Sequence[WordAlignment], duration_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = textgrid.Textgrid(minTimestamp=0.0, maxTimestamp=duration_s)
    intervals = [Interval(item.start_s, item.end_s, item.word) for item in alignments]
    grid.addTier(textgrid.IntervalTier("words", intervals, minT=0.0, maxT=duration_s))
    grid.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def iter_metadata(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter="|", quoting=csv.QUOTE_NONE)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number}: expected 3 columns, got {len(row)}")
            yield row[0], row[2]


class CTCWordAligner:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: str = "cpu",
        local_files_only: bool = False,
        dynamic_quantization: bool = False,
        dtype: str = "float32",
    ) -> None:
        self.device = torch.device(device)
        # Load the plain processor deliberately.  Some model repositories also
        # publish an optional LM decoder; AutoProcessor then requires
        # pyctcdecode, although a language model is irrelevant for aligning a
        # transcript that is already known.
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCTC.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(self.device)
        if dtype == "float16":
            if self.device.type != "cuda":
                raise ValueError("float16 inference is supported only on CUDA")
            self.model = self.model.half()
        elif dtype != "float32":
            raise ValueError(f"unsupported dtype: {dtype}")
        if dynamic_quantization:
            if self.device.type != "cpu":
                raise ValueError("dynamic quantization is supported only for CPU inference")
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
        self.model.eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.delimiter = self.processor.tokenizer.word_delimiter_token or "|"
        self.blank_id = self.processor.tokenizer.pad_token_id
        if self.blank_id is None:
            raise ValueError("the model tokenizer does not define the CTC blank/pad token")

    def align(self, samples: np.ndarray, duration_s: float, text: str) -> tuple[list[WordAlignment], float]:
        return self.align_batch([samples], [duration_s], [text])[0]

    def align_batch(
        self,
        sample_batch: Sequence[np.ndarray],
        durations_s: Sequence[float],
        texts: Sequence[str],
    ) -> list[tuple[list[WordAlignment], float]]:
        """Align a small padded batch and trim logits to each real audio length."""

        if not (len(sample_batch) == len(durations_s) == len(texts)) or not sample_batch:
            raise ValueError("batch inputs must be non-empty and have equal lengths")
        prepared_batch = [prepare_transcript(text, self.vocab, self.delimiter) for text in texts]
        inputs = self.processor(
            list(sample_batch),
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        model_dtype = next(self.model.parameters()).dtype
        input_values = inputs.input_values.to(self.device, dtype=model_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        input_lengths = torch.tensor([len(samples) for samples in sample_batch], dtype=torch.long)
        with torch.inference_mode():
            # The acoustic network can run in FP16 on CUDA to reduce memory,
            # while the dynamic-programming trellis remains FP32 for stable
            # accumulation over long utterances.
            logits_batch = self.model(input_values, attention_mask=attention_mask).logits.float()
            output_lengths = self.model._get_feat_extract_output_lengths(input_lengths).tolist()

        results: list[tuple[list[WordAlignment], float]] = []
        for batch_index, (prepared, duration_s, output_length) in enumerate(
            zip(prepared_batch, durations_s, output_lengths, strict=True)
        ):
            emission = torch.log_softmax(logits_batch[batch_index, : int(output_length)], dim=-1)
            segments, path_score = ctc_forced_alignment(
                emission, prepared.token_ids, self.blank_id
            )
            words = word_segments(prepared, segments, duration_s, emission.size(0))
            normalized_score = math.exp(path_score / max(1, emission.size(0)))
            results.append((words, normalized_score))
        return results


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=project_root / "data" / "RUSLAN" / "RUSLAN",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "lab2_pauses")
    parser.add_argument("--cache-dir", type=Path, default=project_root / "data" / "models")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load an already downloaded model without any Hugging Face network checks",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None, help="optional utterance IDs to align")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append only utterances absent from an existing alignment_status.csv",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print aggregate progress after this many newly processed utterances",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of padded WAV files per acoustic forward pass; use 1 on low-memory hosts",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="CPU inference threads",
    )
    parser.add_argument(
        "--dynamic-quantization",
        action="store_true",
        help="quantize Linear layers to int8 for lower-memory CPU inference",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="acoustic-model precision; float16 reduces CUDA memory use",
    )
    parser.add_argument(
        "--sort-by-duration",
        action="store_true",
        help="sort remaining WAVs by file size to reduce padding inside batches",
    )
    parser.add_argument(
        "--max-batch-mb",
        type=float,
        default=0.0,
        help=(
            "optional maximum sum of source WAV sizes per batch; useful for "
            "automatically shrinking CUDA batches for long recordings"
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    textgrid_dir = args.output_dir / "textgrids"
    selected_ids = set(args.ids) if args.ids else None
    rows = [row for row in iter_metadata(args.metadata) if selected_ids is None or row[0] in selected_ids]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no metadata rows selected")

    alignment_path = args.output_dir / "word_alignments.csv"
    status_path = args.output_dir / "alignment_status.csv"
    processed_ids: set[str] = set()
    if args.resume and status_path.is_file():
        with status_path.open("r", encoding="utf-8", newline="") as source:
            processed_ids = {
                row["utterance_id"] for row in csv.DictReader(source) if row.get("utterance_id")
            }
        rows = [row for row in rows if row[0] not in processed_ids]
        print(f"Resume: {len(processed_ids)} utterances already recorded; {len(rows)} remaining", flush=True)
    if not rows:
        print("Nothing to do: every selected utterance is already recorded.", flush=True)
        return
    if args.sort_by_duration:
        rows.sort(key=lambda row: (args.wav_dir / f"{row[0]}.wav").stat().st_size)

    if args.batch_size < 1 or args.torch_threads < 1 or args.max_batch_mb < 0:
        raise SystemExit("batch size, thread count and maximum batch size must be valid")
    torch.set_num_threads(args.torch_threads)
    aligner = CTCWordAligner(
        args.model,
        args.cache_dir,
        args.device,
        local_files_only=args.offline,
        dynamic_quantization=args.dynamic_quantization,
        dtype=args.dtype,
    )
    alignment_fields = [
        "utterance_id",
        "word_index",
        "word",
        "start_s",
        "end_s",
        "confidence",
        "boundary_after_s",
    ]
    status_fields = ["utterance_id", "status", "word_count", "duration_s", "path_score", "error"]
    append = args.resume and alignment_path.is_file() and status_path.is_file()
    alignment_mode = "a" if append else "w"
    status_mode = "a" if append else "w"
    ok_count = 0

    with alignment_path.open(alignment_mode, encoding="utf-8", newline="") as alignment_file, \
            status_path.open(status_mode, encoding="utf-8", newline="") as status_file:
        alignment_writer = csv.DictWriter(alignment_file, fieldnames=alignment_fields)
        status_writer = csv.DictWriter(status_file, fieldnames=status_fields)
        if not append:
            alignment_writer.writeheader()
            status_writer.writeheader()

        def record_success(
            utterance_id: str,
            duration_s: float,
            alignments: Sequence[WordAlignment],
            path_score: float,
        ) -> None:
            nonlocal ok_count
            write_textgrid(textgrid_dir / f"{utterance_id}.TextGrid", alignments, duration_s)
            for item in alignments:
                alignment_writer.writerow(
                    {
                        "utterance_id": utterance_id,
                        "word_index": item.word_index,
                        "word": item.word,
                        "start_s": f"{item.start_s:.6f}",
                        "end_s": f"{item.end_s:.6f}",
                        "confidence": f"{item.confidence:.6f}",
                        "boundary_after_s": (
                            f"{item.boundary_after_s:.6f}" if item.boundary_after_s is not None else ""
                        ),
                    }
                )
            alignment_file.flush()
            status_writer.writerow(
                {
                    "utterance_id": utterance_id,
                    "status": "ok",
                    "word_count": len(alignments),
                    "duration_s": f"{duration_s:.6f}",
                    "path_score": f"{path_score:.6f}",
                    "error": "",
                }
            )
            status_file.flush()
            ok_count += 1

        def record_error(utterance_id: str, error: Exception, index: int) -> None:
            status_writer.writerow(
                {
                    "utterance_id": utterance_id,
                    "status": "error",
                    "word_count": 0,
                    "duration_s": "",
                    "path_score": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            status_file.flush()
            print(f"[{index}/{len(rows)}] {utterance_id}: ERROR: {error}", flush=True)

        max_batch_bytes = round(args.max_batch_mb * 1024 * 1024)
        batches: list[list[tuple[str, str]]] = []
        current_batch: list[tuple[str, str]] = []
        current_batch_bytes = 0
        for row in rows:
            wav_size = (args.wav_dir / f"{row[0]}.wav").stat().st_size
            exceeds_memory_budget = (
                max_batch_bytes > 0
                and current_batch
                and current_batch_bytes + wav_size > max_batch_bytes
            )
            if len(current_batch) >= args.batch_size or exceeds_memory_budget:
                batches.append(current_batch)
                current_batch = []
                current_batch_bytes = 0
            current_batch.append(row)
            current_batch_bytes += wav_size
        if current_batch:
            batches.append(current_batch)

        completed = 0
        last_reported = 0
        for batch_rows in batches:
            batch_start = completed
            loaded: list[tuple[str, str, np.ndarray, float, int]] = []
            for offset, (utterance_id, normalized_text) in enumerate(batch_rows, start=1):
                index = batch_start + offset
                wav_path = args.wav_dir / f"{utterance_id}.wav"
                try:
                    samples, duration_s = read_wav_mono(wav_path)
                    loaded.append((utterance_id, normalized_text, samples, duration_s, index))
                except Exception as error:
                    record_error(utterance_id, error, index)
                    completed += 1

            if loaded:
                try:
                    results = aligner.align_batch(
                        [item[2] for item in loaded],
                        [item[3] for item in loaded],
                        [item[1] for item in loaded],
                    )
                    for item, (alignments, path_score) in zip(loaded, results, strict=True):
                        record_success(item[0], item[3], alignments, path_score)
                        completed += 1
                except Exception as batch_error:
                    error_message = str(batch_error)
                    print(
                        f"Batch starting at {batch_start + 1} failed ({error_message}); retrying individually",
                        flush=True,
                    )
                    if aligner.device.type == "cuda":
                        batch_error.__traceback__ = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    for utterance_id, normalized_text, samples, duration_s, index in loaded:
                        try:
                            alignments, path_score = aligner.align(samples, duration_s, normalized_text)
                            record_success(utterance_id, duration_s, alignments, path_score)
                        except Exception as error:
                            record_error(utterance_id, error, index)
                        completed += 1

            if (
                completed == len(rows)
                or completed - last_reported >= args.progress_every
            ):
                last_reported = completed
                print(
                    f"Progress: {completed}/{len(rows)} new; {ok_count} aligned; "
                    f"total recorded: {len(processed_ids) + completed}",
                    flush=True,
                )

    print(f"Finished: {ok_count}/{len(rows)} new utterances aligned; output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
