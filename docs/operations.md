# Запуск и эксплуатация

Все команды выполняются из корня проекта. Требуется Python 3.11+.

## Установка

```bash
python -m pip install -r requirements.txt
```

Для LLM, embeddings и OCR используются настройки окружения. Секреты и URL
провайдера должны задаваться безопасным способом через окружение или
секрет-хранилище. Не следует хранить действующие API-ключи в Git.

Базовая конфигурация текущего контура:

```powershell
$env:HYDRA_API_KEY="..."
$env:HYDRA_BASE_URL="https://api.hydraai.ru/v1"
$env:HYDRA_MODEL="gpt-5-mini"
$env:OPENAI_BASE_URL="https://api.hydraai.ru/v1"
$env:OPENAI_MODEL="gpt-5-mini"
$env:OPENAI_PREPROCESSOR_MODEL="gpt-5.4-mini"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

`OPENAI_MODEL` используется основными классификаторами. `OPENAI_PREPROCESSOR_MODEL`
используется только для `LLMTextPreprocessor`, который готовит `decisionText`,
`typeDecisionText` и retrieval queries. Это позволяет оставить основной контур
на более дешёвой модели, а генерацию retrieval-запросов выполнять более сильной
моделью.

## Импорт Word-справочника

Для `.docx`:

```bash
python -m src.classifiers.import_themes \
  --input "data/raw/Классификаторы на 31.10.2025.docx" \
  --out-dir data/classifiers \
  --leaf-mode four-block-only
```

Для старого `.doc` нужен LibreOffice в `PATH`. Подозрительные коды со сносками
по умолчанию только регистрируются. Для применения предлагаемых исправлений
используется `--fix-footnotes`.

После импорта необходимо проверить
`data/classifiers/themes_import_report.json`, особенно:

- `duplicates`;
- `invalidCodes`;
- `fiveBlockCodes`;
- `suspiciousFootnoteCodes`;
- `warnings`.

## Построение индексов

```bash
python -m src.retrieval.build_theme_indexes \
  --themes data/classifiers/themes_leaf.json \
  --out-dir data/indexes/themes \
  --embedding-model text-embedding-3-small
```

Команда вызывает embedding API пакетами, поэтому требует доступного провайдера.
После смены справочника, embedding-модели или логики поискового документа
индексы нужно перестроить.
`index_report.json` содержит SHA-256 файла тематик; CLI/API откажутся загружать
несогласованный комплект.

## Проверка retrieval

```bash
python -m src.retrieval.hybrid_theme_retriever \
  --themes data/classifiers/themes_leaf.json \
  --index-dir data/indexes/themes \
  --query "Жалуюсь на перебои с горячей водой и отоплением" \
  --top-k 10
```

Флаг `--json` выводит машиночитаемый массив. Флаг `--section 0005`
ограничивает поиск разделом.

## Полная классификация строки

```bash
python -m src.classification.pipeline \
  --themes data/classifiers/themes_leaf.json \
  --index-dir data/indexes/themes \
  --text "Жалуюсь на перебои с горячей водой и отоплением"
```

## Evaluation

Текущий рабочий baseline для полного pipeline:

```powershell
$env:OPENAI_MODEL="gpt-5-mini"
$env:OPENAI_PREPROCESSOR_MODEL="gpt-5.4-mini"
```

```bash
python -m src.evaluation.run \
  --dataset data/evaluation/real_v3.json \
  --themes data/classifiers/themes_leaf.json \
  --subtypes data/classifiers/question_subtypes.json \
  --index-dir data/indexes/themes \
  --out-dir data/evaluation/runs/real-v3-current
```

Для повторного расчёта метрик без LLM используется `--metrics-only`.
После изменения prompt, модели или индексов запускайте evaluation в новый `--out-dir`
или очистите старый `predictions.jsonl`: режим `--resume` намеренно пропускает уже
обработанные ID и не пересчитывает старые ответы.

Основные артефакты evaluation:

- `evaluation_report.json` — сводные метрики themes, themeSelection, retrieval, type и subtype;
- `predictions.jsonl` — по одному результату на обращение, включая fallback/error,
  `decisionText`, `typeDecisionText`, retrieval queries и diagnostics;
- `retrieval_diagnostics.csv` — ранги ожидаемых тематик в top-1/3/10/20/30/50;
- `theme_per_label_metrics.csv` и `theme_top_confusions.csv` — ошибки выбора тематик;
- `question_type_confusion.csv/.png` и `question_subtype_confusion.csv/.png` — confusion matrix.

Для анализа тем смотрите не только `themes.subsetAccuracy`: она считает ответ
неверным, если модель нашла правильную тему, но добавила лишнюю. Блок
`themeSelection` показывает, включена ли ожидаемая тема в ответ, а блок
`retrieval` показывает, была ли ожидаемая тема вообще доступна LLM-selector среди
retrieval-кандидатов.

Для сравнения только качества retrieval-запросов без полного pipeline:

```powershell
python tools/evaluation/compare_retrieval_query_models.py `
  --dataset data/evaluation/real_v3.json `
  --themes data/classifiers/themes_leaf.json `
  --index-dir data/indexes/themes `
  --out-dir data/evaluation/retrieval_query_model_compare/real-v3 `
  --models gpt-5-mini gpt-5.4-mini
```

Скрипт сохраняет generated queries, diagnostics и token usage по моделям.

## OCR PDF через мультимодальную LLM

Лабораторный запуск OCR одного PDF:

```powershell
python tools/ocr_lab/hydra_vision_ocr.py `
  --pdf "Обращения\1.pdf" `
  --out-dir "tools/ocr_lab/runs/appeal-001-hydra" `
  --dpi 140 `
  --preprocess grayscale-contrast `
  --pages-per-request 5
```

Текущий OCR prompt извлекает не весь OCR подряд, а тело обращения заявителя и
ключевые предметные слова. Если документ выглядит как запрос прокуратуры,
запрос органа или служебный запрос, prompt требует явно сохранить это в начале
текста, чтобы классификатор не путал такой документ с обычной жалобой
гражданина.

## API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

OpenAPI доступен по `/docs`. CORS origins задаются переменной
`CLASSIFIER_CORS_ORIGINS` через запятую.

Основные endpoints:

- `GET /health` — процесс отвечает;
- `GET /ready` — pipeline, справочники и индексы загружены и согласованы;
- `POST /api/v1/classify` — классификация текста;
- `POST /api/v1/classify/pdf` — загрузка PDF, OCR через мультимодальную LLM и классификация;
- `GET /api/v1/catalog` — агрегированный справочник для web;
- `GET /api/v1/catalog/themes` — дерево тематик и количество leaf-тематик;
- `GET /api/v1/catalog/question-types` — 9 видов вопроса;
- `GET /api/v1/catalog/question-subtypes` — subtype-справочник.

API возвращает и принимает `X-Request-ID`: если клиент не передал заголовок,
backend сгенерирует его сам. Тот же requestId возвращается в header ответа и в
`ClassificationResult.meta.requestId`.

Ошибки возвращаются в едином формате:

```json
{
  "requestId": "...",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "...",
    "details": {}
  }
}
```

Эта команда выполняет embedding запроса и несколько LLM-вызовов. Fallback
обеспечивает структурно валидный ответ при большинстве ошибок LLM, но низкая
fallback-confidence должна рассматриваться как сигнал деградации.

## Тесты

```bash
python -m pytest tests -q -p no:cacheprovider
```

Проверка покрытия после установки `pytest-cov`:

```bash
python -m pytest tests --cov=src.classification --cov=src.retrieval \
  --cov=src.evaluation --cov=src.api --cov-fail-under=85
```

Текущий набор не обращается к внешнему LLM. Он проверяет импорт, нормализацию,
retrieval на искусственных embeddings, fallback, валидаторы и orchestration
pipeline.

## Обновление классификатора

Рекомендуемый порядок:

1. сохранить новый Word-файл отдельно от предыдущей версии;
2. выполнить импорт в новую временную директорию;
3. проверить отчёт и сравнить добавленные/удалённые коды;
4. обновить `classifierVersion`;
5. перестроить BM25 и FAISS;
6. прогнать regression/evaluation набор;
7. атомарно переключить JSON и индексы одной версии.

Смешивать `themes_leaf.json` одной версии с индексами другой нельзя: загрузчики
проверяют последовательность кодов и отклоняют несовместимые артефакты.
