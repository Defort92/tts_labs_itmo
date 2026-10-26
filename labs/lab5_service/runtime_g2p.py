"""Dictionary-backed runtime G2P compatible with the Lab 4 phone inventory."""


import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .errors import ArtifactError, UnknownWordError


WORD_PART_RE = r"[А-Яа-яЁё][А-Яа-яЁё\u0301]*"
WORD_RE = re.compile(rf"{WORD_PART_RE}(?:[-‐‑–—]{WORD_PART_RE})*")
STRESS_MARKS = {"\u0300", "\u0301"}


def normalize_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.casefold())
    without_stress = "".join(char for char in decomposed if char not in STRESS_MARKS)
    return unicodedata.normalize("NFC", without_stress).translate(
        str.maketrans({"‐": "-", "‑": "-", "–": "-", "—": "-"})
    )


def extract_words(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(unicodedata.normalize("NFC", text))]


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class PronouncedWord:
    spelling: str
    normalized: str
    phones: tuple[str, ...]


class RuntimeG2P:
    """Choose a dictionary pronunciation containing only acoustic-model phones."""

    def __init__(self, dictionary_path: str | Path, phone_inventory: set[str]) -> None:
        self.dictionary_path = Path(dictionary_path).resolve()
        if not self.dictionary_path.is_file():
            raise ArtifactError(f"Не найден словарь G2P: {self.dictionary_path}")
        self.phone_inventory = frozenset(phone_inventory)
        variants: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        with self.dictionary_path.open(encoding="utf-8") as source:
            for line in source:
                fields = line.strip().split()
                if len(fields) < 2 or fields[0].startswith("#"):
                    continue
                phone_start = 1
                while phone_start < len(fields) and _is_number(fields[phone_start]):
                    phone_start += 1
                phones = tuple(fields[phone_start:])
                if phones and phones not in variants[normalize_word(fields[0])]:
                    variants[normalize_word(fields[0])].append(phones)
        self.pronunciations = dict(variants)

    def pronounce_word(self, word: str) -> PronouncedWord:
        normalized = normalize_word(word)
        variants = self.pronunciations.get(normalized, [])
        for phones in variants:
            if set(phones).issubset(self.phone_inventory):
                return PronouncedWord(word, normalized, phones)
        if variants:
            unsupported = sorted({phone for phones in variants for phone in phones} - self.phone_inventory)
            raise UnknownWordError(
                f"Для слова «{word}» нет произношения в алфавите акустической модели; "
                f"неизвестные фонемы: {', '.join(unsupported)}"
            )
        raise UnknownWordError(
            f"Слово «{word}» отсутствует в словаре лабораторной №3. "
            "Добавьте его произношение через MFA G2P и повторите запрос."
        )

    def transcribe(self, text: str) -> list[PronouncedWord]:
        words = extract_words(text)
        if not words:
            raise UnknownWordError("В тексте не найдено ни одного русского слова")
        return [self.pronounce_word(word) for word in words]

