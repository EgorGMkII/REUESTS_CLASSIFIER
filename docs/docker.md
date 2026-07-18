# Docker запуск

Контейнер запускает FastAPI backend и web-страницу на стандартном HTTP-порту `80`.
Внутри контейнера приложение слушает `8000`.

## Подготовка

Создать `.env` из примера:

```powershell
Copy-Item .env.example .env
```

В `.env` указать реальный `HYDRA_API_KEY`.

Минимальный `.env`:

```env
HYDRA_API_KEY=...
HYDRA_BASE_URL=https://api.hydraai.ru/v1
HYDRA_MODEL=gpt-5-mini
HYDRA_PREPROCESSOR_MODEL=gpt-5.4-mini
HYDRA_EMBEDDING_MODEL=text-embedding-3-small
```

## Запуск

```powershell
docker compose up --build
```

Открыть:

```text
http://127.0.0.1/
```

Проверка API:

```text
http://127.0.0.1/health
http://127.0.0.1/ready
```

## Что входит в Docker-среду

- backend API;
- web-страница загрузки PDF;
- LangChain/OpenAI-зависимости;
- PyMuPDF/Pillow для преобразования PDF в изображения;
- OCR через мультимодальную LLM по Hydra/OpenAI-compatible API.

`./data` проброшена в контейнер как volume, поэтому индексы, справочники и OCR-артефакты остаются на хосте.

Локальный PaddleOCR в Docker runtime больше не используется. Старые PaddleOCR-скрипты
остаются только в `tools/ocr_lab` как лабораторный/диагностический контур.

## Модели

- `HYDRA_MODEL` — основная модель для OCR PDF и классификаторов. По умолчанию `gpt-5-mini`.
- `HYDRA_PREPROCESSOR_MODEL` — более сильная модель для `LLMTextPreprocessor`.
  По умолчанию `gpt-5.4-mini`.
- `HYDRA_EMBEDDING_MODEL` — модель embeddings. По умолчанию `text-embedding-3-small`.

Если нужно удешевить запуск, можно поставить `HYDRA_PREPROCESSOR_MODEL=gpt-5-mini`,
но качество retrieval-запросов может просесть.
