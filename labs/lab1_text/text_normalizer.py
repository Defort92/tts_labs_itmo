"""Conservative Russian text normalization for aligned speech data."""

from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    """Normalize typography without changing words pronounced in the audio."""

    def __init__(self) -> None:
        self._spaces = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
        self._space_before = re.compile(r"\s+([,.!?;:…])")
        self._space_after = re.compile(r"([,;:])(?=[А-Яа-яЁё])")
        self._many_questions = re.compile(r"\?{2,}")
        self._many_exclamations = re.compile(r"!{2,}")
        self._two_dots = re.compile(r"(?<!\.)\.\.(?!\.)")
        self._bad_bang_dot = re.compile(r"!\.+(?=\s|$)")
        self._translation = str.maketrans(
            {
                "‑": "-", "‐": "-", "−": "-",
                "–": "—", "―": "—",
                "“": '"', "”": '"', "„": '"', "‟": '"',
            }
        )

    @staticmethod
    def _remove_unmatched_pairs(text: str, left: str, right: str) -> str:
        """Remove only unmatched paired punctuation, preserving valid pairs."""
        stack: list[int] = []
        remove: set[int] = set()
        for index, char in enumerate(text):
            if char == left:
                stack.append(index)
            elif char == right:
                if stack:
                    stack.pop()
                else:
                    remove.add(index)
        remove.update(stack)
        return "".join(char for index, char in enumerate(text) if index not in remove)

    def normalize(self, text: str) -> str:
        """Normalize a single line.

        Args:
            text: Raw utterance text, exactly as stored in the corpus metadata.

        Returns:
            The normalized text. Returning the input unchanged is valid and common —
            most lines need nothing done to them.

        Note:
            Do not strip the combining acute accent ``U+0301``. It looks like part of
            the letter and is easily lost to "unicode cleanup", but it marks explicit
            stress and becomes labelled data for stress placement in lab 3.

            Normalize to NFC. Strings in NFC and NFD render identically in a terminal
            and compare unequal.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result = unicodedata.normalize("NFC", text)
        result = result.translate(self._translation)
        result = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", result)
        result = self._spaces.sub(" ", result).strip()
        result = self._space_before.sub(r"\1", result)
        result = self._space_after.sub(r"\1 ", result)
        result = self._many_questions.sub("?", result)
        result = self._many_exclamations.sub("!", result)
        result = self._bad_bang_dot.sub("!", result)
        result = self._two_dots.sub("…", result)
        result = self._remove_unmatched_pairs(result, "(", ")")
        result = self._remove_unmatched_pairs(result, "«", "»")
        if result.count('"') % 2:
            # An odd number means exactly one quote is unmatched. Quotation marks
            # are silent, so dropping the last unmatched mark preserves alignment.
            unmatched = result.rfind('"')
            result = result[:unmatched] + result[unmatched + 1:]
        # Removing an unmatched opening mark can expose a space immediately
        # before punctuation (for example ``«…`` -> ``…``).
        result = self._spaces.sub(" ", result).strip()
        result = self._space_before.sub(r"\1", result)
        return unicodedata.normalize("NFC", result)
