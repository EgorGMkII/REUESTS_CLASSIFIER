# OCR lab

Мини-песочница для проверки OCR отдельно от классификатора.

Цель первого теста: прогнать PaddleOCR на `Обращения\1.pdf` и сохранить:

- распознанный текст;
- постраничные изображения;
- сырые OCR-строки с confidence, чтобы можно было глазами понять, где мусор.

## 1. Проверить текущую среду

В PowerShell:

```powershell
conda info --envs
$env:CONDA_DEFAULT_ENV
python --version
where python
python -m pip --version
```

Если есть NVIDIA GPU:

```powershell
nvidia-smi
```

## 2. Рекомендуемая среда

PaddleOCR/PaddlePaddle может плохо ставиться на Python 3.13, поэтому для OCR лучше отдельная conda-среда на Python 3.10 или 3.11.

Если хочешь использовать существующую среду:

```powershell
conda activate myenv
```

Если нужна отдельная:

```powershell
conda create -n ocrtest python=3.10 -y
conda activate ocrtest
```

## 3. Установка PaddleOCR для CPU

Сначала:

```powershell
python -m pip install -U pip
python -m pip install paddleocr pymupdf pillow
```

Потом поставить PaddlePaddle CPU-версию по официальной инструкции для твоей ОС/Python:

```powershell
python -m pip install paddlepaddle
```

Если обычная установка `paddlepaddle` не сработает, лучше открыть официальный install selector PaddlePaddle и выбрать:

- Windows;
- pip;
- CPU;
- свою версию Python.

### Если PaddleOCR 3.x падает на Windows CPU

Если видишь ошибку вида:

```text
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ...
onednn_instruction.cc
```

это похоже на проблему связки свежего PaddleOCR/PaddlePaddle с Windows CPU/oneDNN.
Для первичного теста проще откатиться на более старую стабильную ветку:

```powershell
python -m pip uninstall -y paddleocr paddlepaddle paddlex
python -m pip install paddlepaddle==2.6.2
python -m pip install paddleocr==2.7.3 pymupdf pillow
```

Потом снова проверить импорт:

```powershell
python -c "import paddle; print('paddle', paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('paddleocr ok')"
```

## 4. Проверить импорт

```powershell
python -c "import paddle; print('paddle', paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('paddleocr ok')"
python -c "import fitz; print('pymupdf ok')"
```

## 5. Запустить OCR на первом PDF

Из корня проекта:

```powershell
python tools/ocr_lab/paddleocr_pdf.py `
  --pdf "Обращения\1.pdf" `
  --out-dir "tools/ocr_lab/runs/appeal-001" `
  --lang ru `
  --dpi 200
```

Более быстрый тестовый вариант для CPU:

```powershell
python tools/ocr_lab/paddleocr_pdf.py `
  --pdf "Обращения\1.pdf" `
  --out-dir "tools/ocr_lab/runs/appeal-001-dpi100-prep" `
  --lang ru `
  --dpi 100 `
  --textline-orientation `
  --preprocess grayscale-contrast
```

Результаты:

```text
tools/ocr_lab/runs/appeal-001/
  pages/
    page_001.png
    ...
  pages_preprocessed/
    page_001.png
    ...
  text.txt
  raw_lines.json
```

## 6. Что смотреть

Сначала открыть `text.txt`.

Если текст совсем плохой:

- попробовать `--dpi 250` или `--dpi 300`;
- попробовать `--dpi 100` / `--dpi 150`;
- попробовать `--preprocess grayscale-contrast`;
- попробовать `--textline-orientation`;
- проверить, не повернуты ли страницы;
- посмотреть картинки в `pages/`;
- посмотреть обработанные картинки в `pages_preprocessed/`;
- сравнить с другим OCR.

Если текст в целом нормальный, но много служебного мусора — это уже задача следующего слоя: LLM-нормализация/очистка перед классификатором.

## 7. Очистить OCR-текст через LLM

После OCR можно выделить тело обращения и исправить очевидные OCR-артефакты:

```powershell
python tools/ocr_lab/ocr_llm_cleanup.py `
  --input "tools/ocr_lab/runs/appeal-001-dpi100-prep/text.txt" `
  --out-dir "tools/ocr_lab/runs/appeal-001-dpi100-prep/llm-cleaned"
```

## 8. Альтернатива: OCR сразу через Hydra vision API

Если хочется обойти PaddleOCR и сразу получить очищенный текст обращения из PDF:

```powershell
$env:HYDRA_API_KEY="..."
$env:HYDRA_BASE_URL="https://api.hydraai.ru/v1"
$env:HYDRA_MODEL="gpt-5.4-mini"

python tools/ocr_lab/hydra_vision_ocr.py `
  --pdf "Обращения\1.pdf" `
  --out-dir "tools/ocr_lab/runs/appeal-001-hydra" `
  --dpi 120 `
  --preprocess grayscale-contrast
```

Результаты:

```text
tools/ocr_lab/runs/appeal-001-hydra/
  pages/
  pages_preprocessed/
  text.txt
  hydra_ocr_debug.json
```

Этот вариант делает PDF → картинки → vision-запрос к роутеру и просит модель вернуть только тело обращения, без служебного мусора.

Результаты:

```text
tools/ocr_lab/runs/appeal-001-dpi100-prep/llm-cleaned/
  cleaned_text.txt
  cleanup_debug.json
```

`cleaned_text.txt` — текст, который можно дальше передавать в classification pipeline через `--text`.
