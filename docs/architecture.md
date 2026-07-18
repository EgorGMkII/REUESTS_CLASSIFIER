# Архитектура

## Поток данных

```text
Word DOC/DOCX
  → import_themes
  → themes_all.json / themes_leaf.json / themes_tree.json
  → BM25 + OpenAI embeddings + FAISS
PDF или текст обращения
  → OCR через мультимодальную LLM, если вход PDF
  → LLMTextPreprocessor
  → decisionText + typeDecisionText + retrievalQueries
  → top-K ThemeCandidate
  → LLM question type по typeDecisionText
  → LLM/rule question subtype по typeDecisionText
  → LLM pair selector для спорных пар вид+subtype
  → LLM theme reranker
  → validators
  → ClassificationResult
```

Pipeline принимает строку. PDF/OCR находится в API/service-слое: PDF сначала
преобразуется в изображения, затем мультимодальная LLM извлекает тело обращения,
после чего текст передаётся в обычный `ClassificationPipeline`.

## 1. Импорт классификатора

Пакет `src.classifiers`:

- читает `.docx` через `python-docx`;
- для `.doc` использует конвертацию LibreOffice, если она доступна;
- извлекает четырёх- и пятиблочные коды;
- отделяет рабочие разделы `0001`–`0005` от методических перечней `0000`;
- нормализует названия, строит родителей, пути и дерево;
- отмечает leaf-узлы;
- регистрирует дубли, ошибочные коды и вероятные артефакты сносок.

Основной источник для retrieval — `data/classifiers/themes_leaf.json`.
Пятиблочные коды в API-leaf набор не входят.

## 2. Retrieval

Пакет `src.retrieval` содержит два независимых поиска:

- `BM25ThemeRetriever` — лексический поиск по лемматизированному русскому
  тексту;
- `FaissThemeRetriever` — cosine similarity по embeddings;
- `HybridThemeRetriever` — объединение результатов с весами BM25 `0.45`,
  vector `0.55` и бонусом `0.05` за совпадение обоих каналов.

Поисковый документ включает название тематики, родителя, раздел и полный путь.
Название повторяется три раза, родитель — два раза. Нормализация одинакова для
справочника и запроса: lowercase, `ё → е`, очистка разделителей, токенизация
`razdel`, лемматизация `pymorphy3`. При отсутствии морфологических библиотек
есть regex fallback.

Артефакты индекса:

```text
data/indexes/themes/
  bm25.pkl
  faiss.index
  embeddings.npy
  metadata.json
  index_report.json
```

Сохранённый индекс использует `text-embedding-3-small`, размерность 1536.

Перед поиском тем `LLMTextPreprocessor` формирует 2–4 retrieval query. Модель
не просто сжимает обращение, а сначала выбирает тематический якорь из
`data/classifiers/retrieval_topic_anchors.json`, выделяет главный предмет,
основную проблему, вторичные детали и детали для отбрасывания.

Правила query:

- первый query — центральный: главный предмет обращения языком справочника;
- остальные query — дополнительные: альтернативные формулировки или
  сопутствующие проблемы;
- адреса, ФИО, даты, номера документов, номера знаков и частные технические
  детали не должны попадать в query;
- section используется как мягкий boost, а не как фильтр.

Pipeline вызывает retrieval для каждого query, берёт top-30, дедуплицирует темы,
добавляет небольшой bonus за повторное попадание и совпадение section, затем
передаёт top-30 кандидатов в `ThemeSelector`.

## 3. Classification pipeline

Пакет `src.classification` выполняет шаги последовательно:

1. готовит `decisionText`, `typeDecisionText` и `retrievalQueries`;
2. нормализует `decisionText` для retrieval/theme selector;
3. ищет кандидатов тематик по нескольким retrieval query;
4. определяет top-2 вида вопроса через LLM по `typeDecisionText`;
5. определяет subtype через LLM для видов 1–4 и подставляет официальные
   `name`/`officialCode` из `question_subtypes.json`;
6. для видов 5–9 выставляет subtype `-` без LLM;
7. в спорных случаях выбирает финальную пару вид+subtype через
   `QuestionPairSelector`;
8. просит LLM выбрать до трёх тематик только из retrieval-кандидатов;
9. подставляет официальные данные тематик и вида вопроса;
10. валидирует итоговые коды и связи;
11. возвращает `ClassificationResult`.

Если top-1 вида вопроса увереннее `0.9`, а вторая позиция ниже `0.6`, pipeline
использует быстрый путь и не вызывает pair selector. Исключение: если top-1 —
`5 Запрос информации`, но среди кандидатов есть `2 Заявление`, включается
долгий путь, потому что сведения часто нужны как часть решения конкретной
проблемы.

LLM вызывается только через существующий `llm_module.py`. Компоненты принимают
внедряемый callable, поэтому тестируются без сети.

Fallback-поведение:

- невалидный вид вопроса → `2 / Заявление`, confidence `0.3`;
- невалидный subtype → `- / Не применяется`, confidence `0.3`;
- невалидный reranking → первый retrieval-кандидат, confidence `0.4`;
- отсутствие любых тематик → `ClassificationValidationError`.

## Выходная схема

```json
{
  "themes": [
    {
      "code": "0005.0005.0056.1156",
      "name": "Перебои в теплоснабжении",
      "confidence": 0.88,
      "section": "0005"
    }
  ],
  "questionType": {
    "code": "3",
    "name": "Жалоба",
    "confidence": 0.86
  },
  "questionSubtype": {
    "code": "-",
    "officialCode": "-",
    "name": "Не применяется",
    "confidence": 1.0
  },
  "classifierVersion": "2025-10-31",
  "meta": {
    "requestId": "uuid-without-hyphens",
    "processedAt": "UTC ISO-8601",
    "sourceTextLength": 73,
    "extractedFromFiles": false
  }
}
```

## 4. Evaluation и API

`src.evaluation` принимает версионированный JSON-набор, записывает результаты
каждого обращения в `predictions.jsonl`, поддерживает resume и строит метрики
themes/themeSelection/retrieval/type/subtype. Для type/subtype создаются CSV/PNG
confusion matrix, для retrieval — CSV с рангами ожидаемых тематик в top-50.

`src.api` загружает pipeline и read-only catalog один раз при старте FastAPI.
Backend-слой разделён на:

- `app.py` — сборка приложения, lifespan, middleware, exception handlers и routers;
- `routes/` — HTTP endpoints и привязка request/response к сервисам;
- `src.services` — бизнес-логика классификации и справочника;
- `src.repositories` — чтение справочников из JSON, подготовленное к будущей замене на БД.

API предоставляет:

- `POST /api/v1/classify`;
- `POST /api/v1/classify/pdf`;
- `GET /api/v1/catalog`;
- `GET /api/v1/catalog/themes`;
- `GET /api/v1/catalog/question-types`;
- `GET /api/v1/catalog/question-subtypes`;
- `GET /health`;
- `GET /ready`.
