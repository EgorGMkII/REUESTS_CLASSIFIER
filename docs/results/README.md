# Источники агрегированных отчётов

JSON скопированы без изменения метрик из локальных существующих артефактов:

| Публичный файл | Локальный источник |
|---|---|
| `real-v3-repeat-1.json` | `data/evaluation/runs/real-v3-repeat-1/evaluation_report.json` |
| `real-v3-repeat-2.json` | `data/evaluation/runs/real-v3-repeat-2/evaluation_report.json` |
| `real-v3-repeat-3.json` | `data/evaluation/runs/real-v3-repeat-3/evaluation_report.json` |
| `real-v3-repeat-4.json` | `data/evaluation/runs/real-v3-repeat-4/evaluation_report.json` |
| `real-plus-synth-throttled.json` | `data/evaluation/runs/real-plus-synth-throttled/evaluation_report.json` |
| `real-plus-synth-gpt54-retrieval.json` | `data/evaluation/retrieval_query_model_compare/real-plus-synth-gpt54/comparison_report.json` |

Это архив результатов, а не запуск текущей версии кода. Тексты, predictions,
queries и персональные данные не включены. Full-pipeline JSON содержит версию
датасета/справочника, но не модели и prompts. Retrieval-only JSON фиксирует
модель и веса. Методика и ограничения: [evaluation](../evaluation.md).
