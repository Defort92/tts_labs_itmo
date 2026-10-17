### Лабораторный практикум по курсу «Синтез речи», Университет ИТМО, 2026

**Лабораторная работа №3. Фонетическая транскрипция (G2P)**

> Основной файл для просмотра и сдачи: [`lab3_g2p.ipynb`](lab3_g2p.ipynb).
> Блокнот содержит постановку задачи, описание входных признаков, метод,
> baseline, метрики, анализ фонемной разметки, ограничения и выводы.

## Постановка задачи

1. Преобразовать нормализованный русский текст в последовательности фонем.
2. Зафиксировать единый фонемный алфавит для следующей лабораторной.
3. Получить временные границы фонем для всего принятого корпуса RUSLAN.
4. Оценить G2P по Phone Error Rate (PER), сравнить с простым baseline и
   проверить полноту и корректность временной разметки.

## Входные данные

- `data/metadata_RUSLAN_22200_normalized.csv` — результат лабораторной №1;
- `data/RUSLAN/RUSLAN/*.wav` — 22 100 принятых WAV с частотой 22 050 Гц;
- `data/lab2_pauses/word_alignments.csv` — независимые CTC-границы слов из
  лабораторной №2, используемые для проверки согласованности;
- официальный русский словарь, G2P- и акустическая модель
  `MontrealCorpusTools/russian_mfa`.

Текстовыми входными признаками G2P являются последовательности Unicode-графем
внутри слова. Ударение, редукции, мягкость, оглушение/озвончение и варианты
произношения кодируются обученной WFST-моделью и словарём, а не набором ручных
`if`-правил в этой работе.

## Почему выбран Montreal Forced Aligner

Посимвольная замена не учитывает контекст русского произношения и не может
получить временные границы из аудио. MFA объединяет произносительный словарь,
G2P для незнакомых слов и HMM-GMM forced alignment. Это даёт единый IPA-подобный
алфавит и согласованные уровни `words`/`phones` в Praat TextGrid.

Модели: [Russian MFA bundle](https://huggingface.co/MontrealCorpusTools/russian_mfa),
метод: [MFA alignment workflow](https://montreal-forced-aligner.readthedocs.io/en/v3.3.5/user_guide/workflows/alignment.html).

## Окружение

Python-утилиты и блокнот используют основное `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r labs\lab3_g2p\requirements.txt
```

MFA устанавливается отдельно, потому что содержит Kaldi и бинарные зависимости:

```powershell
conda env create -f labs\lab3_g2p\environment.yml
conda activate tts-lab3-mfa
$env:MFA_ROOT_DIR = "$PWD\data\lab3_g2p\mfa_root"
mfa model download dictionary russian_mfa
mfa model download acoustic russian_mfa
mfa model download g2p russian_mfa
```

В `environment.yml` намеренно зафиксированы Python 3.12, MFA 3.4.3 и CPU-сборка
Kaldi. GPU для этой работы не требуется.

## Воспроизведение

Подготовка текстов, анализ покрытия и генерация произношений:

```powershell
.\.venv\Scripts\python.exe labs\lab3_g2p\prepare_corpus.py
.\.venv\Scripts\python.exe labs\lab3_g2p\analyze_lexicon.py
mfa g2p data\lab3_g2p\oov_words.txt russian_mfa data\lab3_g2p\oov_predictions.dict --sorted --clean --overwrite -j 10
mfa g2p data\lab3_g2p\g2p_eval_words.txt russian_mfa data\lab3_g2p\g2p_eval_predictions.dict --sorted --clean --overwrite -j 10
.\.venv\Scripts\python.exe labs\lab3_g2p\build_corpus_dictionary.py
.\.venv\Scripts\python.exe labs\lab3_g2p\evaluate_g2p.py
```

Полное фонемное выравнивание:

```powershell
mfa align data\lab3_g2p\mfa_corpus\transcripts data\lab3_g2p\corpus_mfa.dict russian_mfa data\lab3_g2p\textgrids --audio_directory data\RUSLAN\RUSLAN --single_speaker --clean --overwrite -j 10
.\.venv\Scripts\python.exe labs\lab3_g2p\parse_alignments.py
.\.venv\Scripts\python.exe labs\lab3_g2p\compare_word_boundaries.py
.\.venv\Scripts\python.exe labs\lab3_g2p\audit_lab3.py --require-complete
.\.venv\Scripts\python.exe -m unittest discover -s labs\lab3_g2p -p "test_*.py" -v
```

## Алфавит и обработка OOV

Работа использует IPA/MFA-обозначения русской модели: например, `t̪`, `rʲ`,
`ɐ`, `ɪ`, `ɕː`. Полный список и частоты находятся в
`data/lab3_g2p/phoneme_inventory.csv`. Специальные обозначения:

- `sil` — акустическая тишина;
- `spn` — неизвестный речевой фрагмент (в финальной разметке не встретился).

Словарь покрывает 39 090 из 48 121 уникального слова (81,23%). Для 9 031 OOV
произношения получены G2P-моделью. У официального комплекта обнаружено одно
несовпадение алфавитов: G2P выдаёт `ɕ`, тогда как акустическая модель использует
единственную категорию `ɕː`; преобразование `ɕ → ɕː` применено 17 раз и явно
проверяется кодом. Для 584 слов G2P удалил дефис из ключа, поэтому произношение
дефисной формы восстановлено по той же последовательности букв.

## Полученные результаты

- 48 121 уникальное слово и 254 972 словоупотребления;
- словарь корпуса: 58 386 записей с вариантами произношения;
- G2P: PER = **0,2153%**, точность целого слова = **98,7%**;
- посимвольный baseline: PER = **48,601%**, точность = **5,5%**;
- размечено **22 100 из 22 100** фраз;
- **1 500 906** фонемных интервалов, из них 1 442 428 речевых;
- в разметке встретилось 89 символов, средняя длительность интервала 75,19 мс;
- все интервалы положительны и монотонны, пропусков TextGrid нет;
- медианное отличие границ слов от независимого CTC-aligner лабораторной №2:
  52,49 мс для начала и 37,29 мс для конца.

Ошибки G2P обнаружены в 13 из 1 000 слов. В десяти случаях они связаны с
выбором гласного аллофона и редукции (`а/ɐ/ə`, `е/ɪ`, `и/ɪ`), то есть прежде
всего с лексическим ударением. Остальные случаи относятся к палатализации
заимствованного имени (`джованни`), сложному стыку согласных (`бесчисленных`) и
нестандартному междометию `тсс`. В 11 случаях расстояние равно одной фонеме и
только в двух — двум.

MFA самостоятельно выровнял 22 099 фраз. Для `005321_RUSLAN` (1,68 с, пять
слов, CTC confidence 0–0,30) акустическое выравнивание не построилось. Эта одна
фраза не скрыта и не выдана за обычный результат: её word-level границы взяты
из лабораторной №2, а фонемы равномерно распределены внутри слов. Во всех CSV
она помечена `alignment_method=lab2_uniform_fallback`.

## Файлы результатов

Все тяжёлые результаты находятся в игнорируемом Git каталоге `data/lab3_g2p/`:

- `corpus_mfa.dict` — словарь всего корпуса;
- `phoneme_inventory.csv` — фонемный алфавит;
- `g2p_metrics.json`, `g2p_errors.csv` — метрики и ошибки G2P;
- `textgrids/*.TextGrid` — уровни слов и фонем;
- `phoneme_alignments.csv` — один фонемный интервал в строке;
- `word_pronunciations.csv` — произношение и время каждого слова;
- `utterance_phonemes.csv` — фонемная последовательность каждой фразы;
- `alignment_stats.json` — статистика полного выравнивания;
- `word_boundary_metrics.json` — согласованность с лабораторной №2.

## Ограничения

PER посчитан на детерминированной выборке из 1 000 однозначных слов официального
словаря, пересекающихся со словарём RUSLAN. Это воспроизводимый лексический тест,
но не независимая экспертная фонетическая разметка, поэтому оценка оптимистична.
Сравнение границ с лабораторной №2 также измеряет согласие двух автоматических
методов, а не ошибку относительно ручного gold standard.
