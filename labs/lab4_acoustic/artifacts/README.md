# Артефакты FastSpeech 2

`best_model_742028.pth` — лучший checkpoint полного обучения по минимальному
validation loss. Из-за размера файл отслеживается через Git LFS, правило задано
в корневом `.gitattributes`.

Перед первым commit необходимо один раз выполнить:

```bash
git lfs install
git add .gitattributes labs/lab4_acoustic/artifacts/
```

Обычный GitHub не принимает файлы больше 100 МБ; добавлять этот checkpoint с
отключённым Git LFS нельзя. Точная контрольная сумма и метрики выбора находятся
в `model_manifest.json`.
