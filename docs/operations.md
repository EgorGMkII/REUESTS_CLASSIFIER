# Эксплуатация

Установка, окружение, API/PDF запросы и Docker описаны в [README](../README.md).
Команды ниже выполняются из корня.

## Импорт справочника

Для запуска готовых JSON импорт не требуется. Получение их из исходного DOCX:

```bash
python -m src.classifiers.import_themes --input "data/raw/Классификаторы на 31.10.2025.docx" --out-dir data/classifiers --leaf-mode four-block-only
```

Старый `.doc` требует LibreOffice в PATH. Проверьте `themes_import_report.json`:
дубли, invalid codes и подозрительные сноски. `--fix-footnotes` применяется явно.
После изменения JSON используйте команду построения индексов из README.

## CLI

```bash
python -m src.retrieval.hybrid_theme_retriever --themes data/classifiers/themes_leaf.json --index-dir data/indexes/themes --query "Создание муниципального приюта для животных" --top-k 10
python -m src.classification.pipeline --themes data/classifiers/themes_leaf.json --index-dir data/indexes/themes --text "Предлагаю создать муниципальный приют для бездомных животных."
```

Retrieval CLI имеет `--json` и `--section`; последний задаёт фильтр CLI.
В classification pipeline секция retrieval-запроса используется как мягкий boost.

## API

- `POST /api/v1/classify`: JSON `{"text":"..."}`.
- `POST /api/v1/classify/pdf`: PDF в сыром теле, `X-Filename: demo.pdf`.
- `GET /api/v1/catalog`, `/api/v1/catalog/themes`, `/api/v1/catalog/question-types`,
  `/api/v1/catalog/question-subtypes`.
- `GET /health`, `/ready`; OpenAPI — `/docs`.

`X-Request-ID` нормализуется; path-like значения заменяются безопасным ID.
PDF ограничен 25 MiB, текст — 20 000 символов. Readiness проверяет артефакты,
не доступность LLM. Локальный процесс читает окружение при импорте;
после смены переменных процесс надо перезапустить.

## Артефакты и приватность

`data/web_uploads/<requestId>/` содержит PDF, изображения и OCR.
OCR CLI сохраняет `text.txt`, `hydra_ocr_debug.json` и страницы в своём `--out-dir`.
Автоматического удаления/деперсонализации нет; реальные обращения не предназначены
для публичной демонстрации. Данные передаются внешнему API до классификации.

Исторические реальные датасеты и отчёты остаются на диске, исключены
из текущего git-дерева и Docker-контекста. Это не удаляет их из старых коммитов.
Секреты задавайте в окружении или локальном `.env`; не печатайте ключи.

## Evaluation

Команды и трактовка результатов: [evaluation.md](evaluation.md).
Сборщики и retrieval-only сравнение: [tools/evaluation](../tools/evaluation/README.md).
