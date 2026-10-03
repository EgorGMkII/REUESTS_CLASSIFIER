# Evaluation: сохранённые результаты

В `results/` скопированы агрегированные отчёты существующих прогонов, без текстов
и предсказаний отдельных обращений. Они не пересчитаны при оформлении проекта.
Таблица полного pipeline: [README](../README.md#подтверждённые-результаты).

- `real-v3-repeat-1..4`: датасет 3.0, 16 реальных обращений; successful=16, failed=0.
  Вид 0.875–0.9375, subtype 0.375–0.6875, microRecall тем 0.619–0.667,
  micro-F1 0.464–0.509, expected in top-30 0.875–1.0.
- `real-plus-synth-throttled`: plus-synth-1.0, 30 (20 реальных + 10 синтетических),
  successful=30. MicroRecall 0.75, micro-F1 0.5366, вид 0.8333, subtype 0.5333,
  retrieval top-30 0.9333.
- [Retrieval-only](results/real-plus-synth-gpt54-retrieval.json): тот же mixed-набор,
  30 обращений; отчёт фиксирует gpt-5.4-mini, BM25 0.30 / vector 0.70,
  queryTopK=30, finalTopK=50. Top-1 0.60, top-3 0.8333, top-10 0.9333,
  top-30 0.9667, top-50 1.0. Финальные классификаторы не запускались.

Все пять обсуждавшихся максимумов подтверждены, но относятся к разным прогонам:
top-30 1.0 и accuracy вида 0.9375/подтипа 0.6875 есть в repeat-3;
microRecall 0.75 и micro-F1 0.5366 — в mixed throttled.

## Трактовка и ограничения

`expectedInTopK` считает обращения с хотя бы одной expected-темой в выдаче.
MicroRecall объединяет все истинные тематические метки; subsetAccuracy требует
точного совпадения наборов. Высокая accuracy вида при низком macro-F1 отражает,
в частности, дисбаланс классов. MeanBestRank/MRR нельзя трактовать как recall.

Малый объём, спорные соседние leaf-темы и few-shot на части evaluation ограничивают
выводы. Это development evaluation, не независимый holdout. Full-pipeline отчёты
не содержат model/prompt manifest; модель не приписывается по имени каталога.
Latency включает ожидание API/retry/throttling. OCR не входит в эту latency:
evaluation получает подготовленный текст. Resume может объединять результаты
разного времени. Исходные real-датасеты приватные, публично есть старый seed.

## Команды

```bash
python -m src.evaluation.run --dataset data/evaluation/seed_v1.json --themes data/classifiers/themes_leaf.json --subtypes data/classifiers/question_subtypes.json --index-dir data/indexes/themes --out-dir data/evaluation/runs/seed-demo
```

Seed содержит 50 ранее существовавших синтетических примеров. Это API-эксперимент.
`--resume --retry-failed` продолжает прогон; `--metrics-only` пересчитывает метрики
по predictions.jsonl (CLI всё равно загружает pipeline/индексы).
`--llm-min-interval-seconds`, `--llm-max-retries`, `--llm-retry-seconds` учитывают
ограничения ключа. Новый prompt/модель требует нового `--out-dir`.
Predictions, queries и диагностика могут содержать ПДн и игнорируются git.
