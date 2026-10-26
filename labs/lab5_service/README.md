### Лабораторный практикум по курсу «Синтез речи», Университет ИТМО, 2026

# Лабораторная работа №5. Сервис синтеза

## Постановка задачи

Необходимо объединить компоненты предыдущих лабораторных работ в полный тракт
синтеза русской речи и предоставить HTTP API, которое принимает текст и
возвращает WAV:

```text
текст → нормализация → модель пауз → G2P → FastSpeech 2 → HiFi-GAN → WAV
```

Сервис использует только inference: повторное обучение FastSpeech 2 не
выполняется. Модели загружаются один раз и затем повторно используются всеми
запросами.

## Входные данные и признаки

- вход сервиса — нормализуемый русский текст длиной до 500 символов;
- нормализатор и фильтр взяты из лабораторной №1;
- модель лабораторной №2 использует лексические, пунктуационные,
  морфологические и позиционные признаки межсловной границы;
- словарь лабораторной №3 преобразует слова в IPA-подобные фонемы MFA;
- FastSpeech 2 лабораторной №4 получает фонемы и токены `sil`, предсказывает
  длительности, F0, энергию и mel-спектрограмму;
- HiFi-GAN преобразует 80-канальную mel-спектрограмму в mono WAV 22 050 Гц.

Длительность паузы из лабораторной №2 переводится из миллисекунд в кадры mel и
заменяет предсказанную FastSpeech 2 длительность соответствующего токена `sil`.
Таким образом используются и классификация наличия паузы, и регрессия её
длительности.

## Подготовка

Команды выполняются из корня репозитория. Сначала установите course fork Coqui
TTS и зависимости сервиса:

```powershell
.\.venv\Scripts\python.exe -m pip install "TTS @ git+https://github.com/Arbaletos/coqui-ai-TTS.git@tts-course-itmo-2026-m1"
.\.venv\Scripts\python.exe -m pip install -r labs\lab5_service\requirements.txt
```

Перенесите итоговую модель лабораторной №4 в собственный каталог сервиса:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.prepare_lab4_artifacts
```

По умолчанию источником служит `labs/lab4_acoustic/artifacts`. Можно передать
каталог исходного полного прогона напрямую:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.prepare_lab4_artifacts `
  --source "data\lab4_acoustic\runs\ruslan_fastspeech2_full-..."
```

Скрипт копирует лучший checkpoint и `config.json`, проверяет копию по SHA-256 и
создаёт `labs/lab5_service/artifacts/acoustic/manifest.json`. Повторный запуск с
теми же файлами безопасен и не выполняет копирование заново.

Загрузите зафиксированный вокодер:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.download_vocoder
```

Также должны существовать результаты лабораторных №2 и №3:

- `data/lab2_pauses/pause_model.joblib`;
- `data/lab3_g2p/corpus_mfa.dict`.

Все пути и параметры находятся в `config.yaml`.

## Запуск API

```powershell
.\.venv\Scripts\python.exe -m uvicorn labs.lab5_service.api:app --host 127.0.0.1 --port 8000
```

Проверка состояния:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Синтез:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8000/synthesize `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"Сегодня хорошая погода."}' `
  -OutFile synthesis.wav
```

`POST /analyze` принимает тот же JSON, но возвращает нормализованный текст,
фонемы, паузы и управляемые длительности без сохранения WAV.

Неподдерживаемый текст (цифры, латиница, технические символы, неизвестные слова)
возвращает HTTP 422 с причиной. Отсутствующие модели возвращают HTTP 503.

## Оценка

Итоговый прогон на детерминированном наборе новых предложений:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.evaluate_service
```

WAV сохраняются в `examples/`, а `synthesis_metrics.json` содержит фонемы,
паузы, длительности, время каждого этапа и Real-Time Factor:

```text
RTF = время синтеза / длительность полученного аудио
```

Тесты:

```powershell
.\.venv\Scripts\python.exe -m pytest labs\lab5_service\test_lab5_service.py -q
```

### Диагностика акустической модели

Обычный test loss FastSpeech 2 вычисляется с эталонными длительностями, F0 и
энергией, поэтому сам по себе не гарантирует качество свободного синтеза. Для
разделения ошибок акустической модели и вокодера используется запись из
закрытого test split:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.diagnose_acoustic_model --write-audio
```

Скрипт сравнивает предсказания duration, pitch и energy с MFA-целями и создаёт
в `data/lab5_diagnostics` четыре варианта: свободный inference, inference с
эталонными длительностями, полный teacher forcing и HiFi-GAN от эталонной mel.
Если последний вариант разборчив, а свободный inference нет, проблема находится
в акустической модели, а не в HTTP API или вокодере.

## Ограничения

- словарь G2P покрывает лексику корпуса RUSLAN; незнакомые слова необходимо
  предварительно добавить с помощью MFA G2P;
- нормализатор консервативен и намеренно не раскрывает неоднозначные числа и
  сокращения;
- HiFi-GAN обучен отдельно от RUSLAN, поэтому возможен domain mismatch между
  предсказанными mel-спектрограммами и входом вокодера;
- текущий checkpoint лабораторной №4 необходимо принимать только после
  прослушивания свободного inference: teacher-forced test loss не проверяет
  самостоятельную работу предикторов duration, pitch и energy;
- перцептивное качество оценивается отдельно в лабораторной №6 с помощью MOS.
