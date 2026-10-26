from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from labs.lab5_service.errors import UnknownWordError
from labs.lab5_service.frontend import milliseconds_to_frames
from labs.lab5_service.pipeline import waveform_to_wav_bytes
from labs.lab5_service.prepare_lab4_artifacts import prepare, select_checkpoint
from labs.lab5_service.runtime_g2p import RuntimeG2P, normalize_word
from labs.lab5_service.settings import ServiceSettings


class ArtifactPreparationTests(unittest.TestCase):
    def test_selects_highest_numbered_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "best_model_10.pth").write_bytes(b"old")
            (root / "best_model_25.pth").write_bytes(b"new")
            self.assertEqual(select_checkpoint(root).name, "best_model_25.pth")

    def test_copies_checkpoint_config_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "best_model_7.pth").write_bytes(b"checkpoint")
            (source / "config.json").write_text('{"audio": {}}', encoding="utf-8")
            first = prepare(source, destination)
            second = prepare(source, destination)
            self.assertTrue(first["checkpoint"]["copied"])
            self.assertFalse(second["checkpoint"]["copied"])
            self.assertEqual((destination / "model.pth").read_bytes(), b"checkpoint")
            self.assertTrue((destination / "manifest.json").is_file())


class RuntimeG2PTests(unittest.TestCase):
    def test_selects_only_supported_pronunciation_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dictionary = Path(temporary) / "dictionary.dict"
            dictionary.write_text("слово bad a\nслово s l o v a\n", encoding="utf-8")
            g2p = RuntimeG2P(dictionary, {"s", "l", "o", "v", "a", "sil"})
            self.assertEqual(g2p.pronounce_word("Сло́во").phones, ("s", "l", "o", "v", "a"))

    def test_unknown_word_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dictionary = Path(temporary) / "dictionary.dict"
            dictionary.write_text("слово s l o v a\n", encoding="utf-8")
            g2p = RuntimeG2P(dictionary, {"s", "l", "o", "v", "a", "sil"})
            with self.assertRaisesRegex(UnknownWordError, "отсутствует в словаре"):
                g2p.pronounce_word("неологизм")

    def test_normalizes_stress_and_dash(self) -> None:
        self.assertEqual(normalize_word("Сло́во–то"), "слово-то")


class SignalAndSettingsTests(unittest.TestCase):
    def test_pause_milliseconds_are_converted_to_mel_frames(self) -> None:
        self.assertEqual(milliseconds_to_frames(100.0, 22_050, 256), 9)
        self.assertEqual(milliseconds_to_frames(0.0, 22_050, 256), 1)

    def test_wav_bytes_are_pcm16_mono(self) -> None:
        payload = waveform_to_wav_bytes(np.zeros(2205, dtype=np.float32), 22_050)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.wav"
            path.write_bytes(payload)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 22_050)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getnframes(), 2205)

    def test_settings_resolve_paths_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text(
                """
service: {device: cpu, max_text_chars: 100, edge_silence_ms: 50, preload_models: false}
acoustic: {checkpoint: a/model.pth, config: a/config.json}
frontend: {pause_model: f/pause.joblib, pronunciation_dictionary: f/dictionary.dict}
vocoder: {checkpoint: v/model.pth, config: v/config.json}
""".strip(),
                encoding="utf-8",
            )
            settings = ServiceSettings.load(config)
            self.assertEqual(settings.acoustic_checkpoint, (root / "a/model.pth").resolve())
            self.assertEqual(settings.max_text_chars, 100)

    def test_api_declares_required_routes_without_loading_models(self) -> None:
        from labs.lab5_service.api import app

        routes = {route.path for route in app.routes}
        self.assertTrue({"/health", "/analyze", "/synthesize"}.issubset(routes))


if __name__ == "__main__":
    unittest.main()
