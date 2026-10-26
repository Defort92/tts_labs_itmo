# Восстановление FastSpeech 2 после лабораторной №4

Эта папка изолирует корректирующее обучение от исходной лабораторной №4 и от
сервиса лабораторной №5. Старый checkpoint не перезаписывается.

## Причина отдельного fine-tuning

HiFi-GAN хорошо восстанавливает речь из эталонной mel-спектрограммы, а полный
teacher forcing FastSpeech 2 заметно лучше свободного inference. Значит,
основная неисправность находится в предикторах вариативности FastSpeech 2:
duration, pitch и energy.

Честная проверка исходного `best_model_742028.pth` на всех 1093 закрытых
test-записях дала:

| Показатель | Исходное значение |
|---|---:|
| Duration MAE, кадров на фонему | 3.554 |
| Duration correlation | 0.427 |
| Pitch correlation | 0.224 |
| Energy correlation при предсказанном pitch | 0.278 |
| Среднее отношение предсказанной длины к истинной | 0.730 |

Обычный test loss лабораторной №4 не показывал проблему полностью, потому что
mel-декодер проверялся с истинными duration, pitch и energy.

## Метод

Первый безопасный этап корректировки:

1. загрузить из лабораторной №4 только веса модели;
2. создать новый optimizer с learning rate `1e-4` без старого Noam scheduler;
3. заморозить encoder, decoder, фонемные и variance embeddings;
4. обучать только `duration_predictor`, `pitch_predictor` и
   `energy_predictor`;
5. увеличить веса трёх predictor-loss с `0.1` до `1.0`;
6. отключить mel-loss, потому что замороженный decoder уже принят по результату
   teacher forcing;
7. выбирать checkpoint по validation loss предикторов и отдельно проверять
   свободный inference.

Такой этап не может испортить веса decoder. Если его окажется недостаточно,
следующим этапом будет контролируемое частичное размораживание encoder, но его
нельзя выполнять до анализа первого результата.

## Запуск

Основной сценарий находится в
`lab4_fastspeech2_variance_repair.ipynb`. Сначала оставьте:

```python
MODE = "smoke"
RUN_TRAINING = False
```

Выполните подготовительные ячейки, проверьте сводку, затем установите
`RUN_TRAINING = True` и запустите две smoke-эпохи. После успешного smoke-запуска
перезапустите kernel и используйте:

```python
MODE = "full"
EPOCHS = 15
RUN_TRAINING = True
```

Полные прогоны сохраняются только в:

```text
data/lab4_acoustic_repair/runs/
```

## Остановка и продолжение

Останавливайте обучение между эпохами или обычным Interrupt Kernel. Для
продолжения укажите каталог конкретного repair-run:

```python
CONTINUE_PATH = r"C:\...\data\lab4_acoustic_repair\runs\ruslan_fastspeech2_variance_repair_full-..."
```

При `CONTINUE_PATH` восстанавливаются веса, optimizer, номер эпохи, global step
и best loss. При новом запуске `CONTINUE_PATH = None`: загружаются только веса
исходной модели и создаётся свежий optimizer.

## Независимая оценка

До и после обучения используется одна команда:

```powershell
.\.venv\Scripts\python.exe -m labs.lab4_acoustic_repair.evaluate_variance_predictors `
  --checkpoint "путь\к\best_model_....pth" `
  --output "data\lab4_acoustic_repair\candidate_metrics.json"
```

Затем создаются четыре аудиоварианта для контрольной test-записи:

```powershell
.\.venv\Scripts\python.exe -m labs.lab5_service.diagnose_acoustic_model `
  --checkpoint "путь\к\best_model_....pth" --write-audio
```

Ориентировочная цель первого этапа: заметно превзойти baseline по всем трём
корреляциям, получить median frame ratio около `1.0` и разборчивую свободную
речь. Численные пороги не заменяют прослушивание.

Новый checkpoint не переносится в лабораторную №5 автоматически. Это делается
только после сравнения метрик и аудио со старой моделью.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest labs\lab4_acoustic_repair -q
```

