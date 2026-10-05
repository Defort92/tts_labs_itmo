"""Rule-based detector of text that is safe for TTS training.

The positive class means that the written text can be pronounced literally. This
is intentionally conservative: losing a small number of utterances is preferable
to teaching the acoustic model a wrong text/audio alignment.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import unicodedata

DEV_SET_PATH = Path(__file__).resolve().parent / "data" / "dev_sentences.csv"


class TextFilter:
    """Decide whether an utterance can be used as aligned training data."""

    def __init__(self) -> None:
        # Includes real foreign words and visually similar substitutions.
        self._latin = re.compile(r"[A-Za-z\u00c0-\u024f]")
        self._digit = re.compile(r"\d")
        self._forbidden_symbols = re.compile(r"[@#$%&*+=/\\<>\[\]{}_\|~^№€£¥₽°©®™]")
        self._invisible = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

        abbreviations = (
            r"т\s*\.\s*д\.", r"т\s*\.\s*п\.", r"и\s*\.\s*т\s*\.\s*[дп]\.",
            r"г-н", r"г-ж[аи]", r"ул\.", r"д\.", r"стр\.", r"кв\.",
            r"рис\.", r"табл\.", r"прим\.", r"им\.", r"г\.",
        )
        self._abbreviation = re.compile(
            rf"(?<![А-Яа-яЁё])(?:{'|'.join(abbreviations)})(?![А-Яа-яЁё])",
            re.IGNORECASE,
        )
        self._initial = re.compile(r"(?<![А-Яа-яЁё])[А-ЯЁ]\.(?:\s*[А-ЯЁ]\.)?(?=\s|$)")
        self._known_acronym = re.compile(
            r"\b(?:АХЧ|ВЛКСМ|ГИБДД|ЖКХ|КГБ|КПСС|МВД|МЧС|НКВД|ОК|ООН|РФ|СМИ|"
            r"СССР|ФСБ|ЦДЛ|ЦК|Минюст(?:е|а|ом|у)?)\b"
        )
        # A consonant-only uppercase token cannot be read as an ordinary Russian
        # word and is therefore almost certainly an initialism (КПСС, ФСБ, ЦДЛ).
        # Ordinary headings such as ``СЕРГЕЙ ДОВЛАТОВ`` must remain valid.
        self._consonant_acronym = re.compile(r"\b[БВГДЖЗЙКЛМНПРСТФХЦЧШЩЪЬ]{2,}\b")
        self._interjection = re.compile(
            r"^\s*(?:м+|хм+|гм+|мда+|ы+|хе(?:хе)+|ха(?:ха)+)\s*[,!.?]",
            re.IGNORECASE,
        )
        self._space_before_punctuation = re.compile(r"\s+[,.!?;:…»]")
        self._missing_space_after = re.compile(r"[,;:](?=[А-Яа-яЁё])")
        self._repeated_exclamation = re.compile(r"!{2,}")
        self._repeated_question = re.compile(r"\?{2,}")
        self._two_dots = re.compile(r"(?<!\.)\.\.(?!\.)")
        # ``?!`` and ``?!.`` are accepted expressive question endings in the
        # supplied annotation; a bare ``!.`` is a punctuation error.
        self._bad_terminal_mix = re.compile(r"(?<!\?)!\.$|\.\!$")
        self._emoticon = re.compile(r"(?:[:;=8xX][-^']?[()DPpРр]|[()]{2,}|<3)")

        self._named_patterns = (
            ("latin_letters", self._latin),
            ("digits", self._digit),
            ("technical_symbols", self._forbidden_symbols),
            ("invisible_characters", self._invisible),
            ("abbreviations", self._abbreviation),
            ("initials", self._initial),
            ("known_acronyms", self._known_acronym),
            ("consonant_acronyms", self._consonant_acronym),
            ("interjections", self._interjection),
            ("space_before_punctuation", self._space_before_punctuation),
            ("missing_space_after_punctuation", self._missing_space_after),
            ("repeated_exclamation_marks", self._repeated_exclamation),
            ("repeated_question_marks", self._repeated_question),
            ("double_dots", self._two_dots),
            ("invalid_terminal_punctuation", self._bad_terminal_mix),
            ("emoticons", self._emoticon),
        )

    @staticmethod
    def _balanced(text: str, left: str, right: str) -> bool:
        depth = 0
        for char in text:
            if char == left:
                depth += 1
            elif char == right:
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    def rejection_reasons(self, text: str) -> list[str]:
        """Return stable machine-readable reasons why ``text`` is rejected."""
        if not isinstance(text, str) or not text.strip():
            return ["empty_or_non_string"]

        reasons: list[str] = []
        if text != unicodedata.normalize("NFC", text):
            reasons.append("not_nfc")
        if any(char in text for char in "\r\n\t"):
            reasons.append("line_break_or_tab")
        if any(unicodedata.category(char) == "Cc" for char in text):
            reasons.append("control_characters")

        for name, pattern in self._named_patterns:
            if pattern.search(text):
                reasons.append(name)
        if any(mark in text for mark in "“”„‟"):
            reasons.append("nonstandard_quotes")
        if not self._balanced(text, "(", ")"):
            reasons.append("unbalanced_parentheses")
        if not self._balanced(text, "«", "»"):
            reasons.append("unbalanced_guillemets")
        if text.count('"') % 2:
            reasons.append("unbalanced_straight_quotes")
        return reasons

    def filter(self, text: str) -> int:
        """Return ``1`` for literal normalized Russian text, otherwise ``0``."""
        return int(not self.rejection_reasons(text))


if __name__ == "__main__":
    textfilter = TextFilter()
    with DEV_SET_PATH.open(encoding="utf-8", newline="") as dev_file:
        rows = list(csv.DictReader(dev_file, delimiter="|", quoting=csv.QUOTE_NONE))

    expected = [int(row["is_normalized"]) for row in rows]
    predicted = [textfilter.filter(row["text"]) for row in rows]
    true_positive = sum(y == p == 1 for y, p in zip(expected, predicted))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    print(
        f"F1 Score is {f1:.4f}, Precision is {precision:.4f}, "
        f"Recall is {recall:.4f}"
    )
