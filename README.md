# Классификация обращений из PDF

ML/NLP-проект: извлечение текста из русскоязычных обращений и классификация по
справочнику из **1220 тематик**, **9 видов обращения** и **39 записей подтипов**
(включая «не применяется»). Веб-страница, HTTP API и CLI используют общий pipeline.

**Стек:** Python 3.10+, FastAPI, Pydantic 2, LangChain/OpenAI SDK,
мультимодальная GPT через Hydra API, BM25, FAISS, NumPy, razdel/pymorphy3,
PyMuPDF/Pillow, pytest, scikit-learn, Docker Compose. LangGraph и автономных
агентов в реализации нет; это pipeline с LLM-вызовами и поиском кандидатов.

## Вход и выход

Вход: PDF (до 25 MiB) или текст (до 20 000 символов через API).
Выход: темы с официальными кодами, вид, подтип, confidence и метаданные.
PDF endpoint также возвращает кандидатов retrieval/пар и информацию об OCR.
Retrieval score — оценка ранжирования, а LLM confidence не калиброванная вероятность.

Безопасный пример из существующего [seed_v1.json](data/evaluation/seed_v1.json), `seed-001`:

> Предлагаю создать муниципальный приют для бездомных животных и организовать их безопасное содержание.

Его **эталонная разметка** в сокращённой структуре ответа (это пример контракта,
а не результат нового обращения к API):

```json
{
  "themes": [{"code": "0003.0011.0127.0865", "name": "Гуманное отношение к животным. Создание приютов для животных", "section": "0003"}],
  "questionType": {"code": "1", "name": "Предложение"},
  "questionSubtype": {"code": "1.3.4", "officialCode": "П.3.4", "name": "Рекомендации автора обращения по улучшению иных сфер деятельности общества"},
  "classifierVersion": "2025-10-31"
}
```

## Архитектура и инженерные решения

```text
PDF → изображения страниц → мультимодальный OCR → текст
текст → LLM preprocessing → decisionText / typeDecisionText / retrievalQueries
      → BM25 + embeddings/FAISS → объединение кандидатов → ThemeSelector
      → классификатор вида → классификатор подтипа [→ выбор пары]
      → валидация кодов и допустимых связей → JSON
```

- Preprocessing в одном LLM-вызове сохраняет контекст для вида/подтипа и строит
  до четырёх запросов через [тематические якоря](data/classifiers/retrieval_topic_anchors.json).
  Первый запрос отражает центральный предмет, остальные — дополнительные аспекты.
- Hybrid retrieval нормализует русский текст, сочетает BM25 **0.45** и vector **0.55**.
  Для каждого запроса извлекаются 30 кандидатов, затем они объединяются с бонусами
  за повторные попадания и совпадение секции. Секция здесь — мягкий boost.
- Быстрый путь: top-1 вида ≥ **0.8**, второй кандидат ≤ **0.75**. Подтипы ищутся
  только для первого вида, берётся первый подтип; `QuestionPairSelector` пропускается.
  Иначе подтипы двух видов вычисляются параллельно в потоках, затем LLM выбирает пару.
  Для видов 5–9 подтип `-` назначается без LLM-вызова.
- ThemeSelector выбирает темы из кандидатов; prompt просит обычно 1–2, допускает 3.
  Схемы и валидаторы проверяют коды, версии и совместимость вида с подтипом.
- Индексы проверяются по SHA-256 справочника. Evaluation сохраняет ошибки,
  fallback и диагностику, поддерживает resume/retry и offline-пересчёт.

Подробнее: [архитектура](docs/architecture.md).

## Подтверждённые результаты

Каждая строка — отдельный сохранённый полный прогон, все обращения обработаны.
Это небольшие исследовательские выборки, а не независимый production benchmark.

| Прогон | Датасет / размер | Expected in top-30 | Accuracy вида | Accuracy подтипа | Themes microRecall | Themes micro-F1 |
|---|---|---:|---:|---:|---:|---:|
| [real-v3-repeat-1](docs/results/real-v3-repeat-1.json) | real_v3, 3.0 / 16 | 0.938 | 0.875 | 0.375 | 0.667 | 0.509 |
| [real-v3-repeat-2](docs/results/real-v3-repeat-2.json) | real_v3, 3.0 / 16 | 0.938 | 0.938 | 0.625 | 0.667 | 0.509 |
| [real-v3-repeat-3](docs/results/real-v3-repeat-3.json) | real_v3, 3.0 / 16 | 1.000 | 0.938 | 0.688 | 0.619 | 0.473 |
| [real-v3-repeat-4](docs/results/real-v3-repeat-4.json) | real_v3, 3.0 / 16 | 0.875 | 0.938 | 0.563 | 0.619 | 0.464 |
| [real-plus-synth-throttled](docs/results/real-plus-synth-throttled.json) | plus-synth-1.0 / 30 (20 реальных + 10 синтетических) | 0.933 | 0.833 | 0.533 | 0.750 | 0.537 |

`expectedInTop30` — доля обращений с хотя бы одной ожидаемой темой в выдаче;
это не microRecall всех тематических меток. Финальный выбор тем слабее retrieval:
есть пропуски, соседние темы и лишние метки. Few-shot дорабатывался на части
этих примеров, поэтому возможна переоценка обобщающей способности.
Сохранённые полные отчёты не фиксируют конфигурацию моделей/prompts; названия
моделей не восстановлены по памяти. [Методика и ограничения](docs/evaluation.md).

## Quick start: локально

Команды из корня репозитория. Рекомендуется Python 3.10 или 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

В Linux/macOS: `source .venv/bin/activate` и `cp .env.example .env`.
Заполните `HYDRA_API_KEY`. **Локальный API/CLI автоматически `.env` не читает**:
задайте значения в той же сессии до запуска Python:

```powershell
$env:HYDRA_API_KEY="ваш_ключ"
$env:HYDRA_BASE_URL="https://api.hydraai.ru/v1"
$env:HYDRA_MODEL="gpt-5-mini"
$env:HYDRA_PREPROCESSOR_MODEL="gpt-5.4-mini"
$env:HYDRA_EMBEDDING_MODEL="text-embedding-3-small"
```

В Bash используйте `export HYDRA_API_KEY="ваш_ключ"` и аналогично остальные переменные.
Основная модель обслуживает OCR и классификаторы; preprocessor имеет отдельную модель.

### С готовыми индексами

Есть JSON-справочники редакции `2025-10-31` и комплект
`data/indexes/themes/{bm25.pkl,faiss.index,embeddings.npy,metadata.json,index_report.json}`.
При неизменном справочнике пересборка не нужна. Используйте доверенные индексы:
BM25 загружается из pickle. Ключ нужен для embeddings запросов и LLM.

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Сайт: `http://127.0.0.1:8000/`, OpenAPI — `/docs`.
`/health` проверяет процесс; `/ready` — локальные артефакты, не доступность провайдера.

### Индексы с нуля

При отсутствии/устаревании индексов выполните **до запуска API**:

```powershell
python -m src.retrieval.build_theme_indexes --themes data/classifiers/themes_leaf.json --out-dir data/indexes/themes --embedding-model text-embedding-3-small
```

Это платный embedding API-вызов для справочника. После изменения справочника
или embedding-модели требуется новый комплект. Импорт Word описан в
[operations](docs/operations.md); для обычного запуска он не нужен.

### Запросы

Безопасный текстовый пример (PowerShell):

```powershell
$body = @{text="Предлагаю создать муниципальный приют для бездомных животных и организовать их безопасное содержание."} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/classify -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

PDF отправляется **сырым телом**, не multipart. Используйте свой обезличенный PDF:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/classify/pdf -H "Content-Type: application/pdf" -H "X-Filename: demo.pdf" --data-binary "@demo.pdf"
```

В Bash замените `curl.exe` на `curl`. Изображения PDF передаются внешнему API;
локальной деперсонализации нет. Загрузки/OCR сохраняются в `data/web_uploads/`.

## Quick start: Docker Compose

Нужны Docker Engine/Desktop и Compose v2. Compose читает корневой `.env`:

```bash
cp .env.example .env
# Заполнить HYDRA_API_KEY в .env
docker compose up --build -d
docker compose ps
curl http://127.0.0.1/ready
```

Сайт: `http://localhost/` (порт хоста **80**, контейнера **8000**).
Для запросов выше уберите `:8000`. Если порт 80 занят, измените mapping в compose.
Справочники и индексы берутся из bind mount `./data:/app/data`.

Если индексов нет или `/ready` сообщает о stale indexes:

```bash
docker compose run --rm classifier-api python -m src.retrieval.build_theme_indexes --themes data/classifiers/themes_leaf.json --out-dir data/indexes/themes --embedding-model text-embedding-3-small
docker compose restart classifier-api
```

`up -d` не перезапускает уже работающий процесс после пересборки индексов.
После изменения `.env`: `docker compose up -d --force-recreate classifier-api`.
Логи: `docker compose logs -f classifier-api`; остановка: `docker compose down`.

## Тесты и эксперименты

Существующие offline-тесты (без ключа):

```bash
python -m pytest tests -m "not external"
```

`pytest.ini` включает coverage gate 85%; `requirements.txt` содержит pytest-cov.
Без coverage-плагина проверка поведения отдельно:
`python -m pytest tests -o addopts='' -p no:cacheprovider -m "not external"`.

Evaluation на существующем публичном seed-наборе (50 примеров; требует API):

```bash
python -m src.evaluation.run --dataset data/evaluation/seed_v1.json --themes data/classifiers/themes_leaf.json --subtypes data/classifiers/question_subtypes.json --index-dir data/indexes/themes --out-dir data/evaluation/runs/seed-demo
```

Для пересчёта сохранённого прогона добавьте `--metrics-only`; для продолжения —
`--resume --retry-failed`. Новый prompt/модель требует нового `--out-dir`.
Retrieval-only и сборщики приватных датасетов: [tools/evaluation](tools/evaluation/README.md).
OCR-исследования: [tools/ocr_lab](tools/ocr_lab/README.md).

## Ограничения и данные

Нет обучения собственной модели, очереди задач, аутентификации и production-мониторинга.
PDF-запрос синхронный; индикатор этапов веб-страницы работает по таймеру,
а не по backend-событиям. Проценты в UI — эвристика, не вероятность правильного ответа.
Зависимости заданы диапазонами, lock-файла пока нет.

Реальные PDF, OCR и датасеты исключены из текущего публичного дерева; локальные
копии сохранены и игнорируются. Старые коммиты содержат эти данные, история
не очищена. Опубликованы безопасный seed и агрегированные отчёты.
Подробная документация: [docs/README.md](docs/README.md).
