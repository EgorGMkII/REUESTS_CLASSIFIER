# Evaluation и подготовка данных

Команды из корня. Окружение: [README](../../README.md). Реальная разметка/PDF
приватные; инструменты сохранены для воспроизводимости исследования.

| Инструмент | Назначение |
|---|---|
| `compare_retrieval_query_models.py` | Preprocessing выбранными моделями, retrieval и ранги; без ThemeSelector/вида/subtype |
| `build_real_v3_from_hydra_ocr.py` | Обновляет тексты через мультимодальный OCR, сохраняет разметку |
| `build_real_v2_from_ocr.py` | Исторический PaddleOCR + cleanup, отдельная среда |
| `build_real_plus_synth.py` | Историческая сборка mixed-набора из приватного real_v3 и локальных OCR |

Сборщики v2/v3 перенесены из корня; default paths остаются относительно корня.
Mixed-сборщик сохранён без изменения ранее подготовленных примеров/разметки.

Retrieval-only на существующем seed (API-эксперимент):

```bash
python tools/evaluation/compare_retrieval_query_models.py --dataset data/evaluation/seed_v1.json --out-dir data/evaluation/retrieval_query_model_compare/seed --models gpt-5-mini gpt-5.4-mini --bm25-weight 0.30 --vector-weight 0.70
```

Есть `--limit 1`/`--case-id seed-001`. Сохраняются comparison_report.json,
generated_queries.jsonl, retrieval_diagnostics.csv. Queries содержат текст;
публикуйте только безопасные данные. Веса эксперимента не меняют pipeline (0.45/0.55).

Приватная пересборка текстов:

```bash
python tools/evaluation/build_real_v3_from_hydra_ocr.py --input data/evaluation/real_v2.json --output data/evaluation/real_v3.json --pdf-dir Обращения --dpi 140 --pages-per-request 5 --limit 1
```

`--force` повторяет OCR даже при наличии текста. Исходный датасет/PDF должны быть
предоставлены локально. Не запускайте сборку ради просмотра документации.
