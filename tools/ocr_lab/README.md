# OCR lab

Текущий API/PDF сервис использует функции `hydra_vision_ocr.py`: PDF → изображения
через PyMuPDF → опциональный контраст через Pillow → мультимодальная LLM.
Hydra — роутер API, а не отдельная OCR-модель. Модель: `HYDRA_MODEL`/`--model`.

```bash
python tools/ocr_lab/hydra_vision_ocr.py --pdf demo.pdf --out-dir tools/ocr_lab/runs/demo --dpi 140 --preprocess grayscale-contrast --pages-per-request 5
```

Используйте обезличенный PDF. `HYDRA_API_KEY` задаётся в окружении.
Сохраняются страницы, `text.txt`, `hydra_ocr_debug.json` и token usage.
Default max-tokens=2500; проверяйте обрезание длинного текста.
PDF web-сервис имеет свой default dpi=120; параметры CLI не меняют сервис.

Prompt извлекает тело обращения и значимый административный контекст.
Признаки исходящего запроса прокуратуры сохраняются для классификации.
Extraction может ошибаться; проверяйте debug-артефакты.

## Сохранённые альтернативы

- `paddleocr_pdf.py` — локальный OCR и постраничные строки. Зависимости отдельно
  в `requirements-paddleocr.txt`, не входят в API runtime.
- `ocr_llm_cleanup.py` — отдельная очистка распознанного текста. Текущий
  мультимодальный сценарий объединяет extraction/cleanup в одном этапе.
- [lift_lab](../lift_lab/README.md) — исследование schema-guided extraction.
- [ocrTest](../../ocrTest/README.md) — ранний прототип, не точка входа приложения.

Документы, изображения и OCR могут содержать ПДн, поэтому `runs/` игнорируется git.
Сборщики датасетов: [tools/evaluation](../evaluation/README.md).
