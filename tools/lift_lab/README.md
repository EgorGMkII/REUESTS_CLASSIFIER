# Lift lab

Мини-песочница для проверки `datalab-to/lift` отдельно от основного классификатора.

`lift` — это не обычный OCR. Он принимает PDF/изображение и JSON Schema, а на выходе пытается сразу вернуть структурированный JSON.

Для нас первый тест:

```text
Обращения\1.pdf
→ lift
→ {"appeal_text": "...", "summary": "..."}
```

## Важное про среду

По `pyproject.toml` пакет `lift-pdf` требует Python `>=3.12`.

Проверь текущую среду:

```powershell
python --version
where python
nvidia-smi
```

Если Python ниже 3.12:

```powershell
conda create -n lifttest python=3.12 -y
conda activate lifttest
```

## Вариант A: vLLM backend

Это рекомендуемый вариант в README `lift`, но он рассчитан на GPU и Docker.

Установка:

```powershell
python -m pip install -U pip
python -m pip install lift-pdf
```

Запуск vLLM-сервера:

```powershell
lift_vllm --gpu 3090
```

Возможные значения GPU из README:

```text
h100, a100-80, a100, a100-40, l40s, a10, l4, 4090, 3090, t4
```

Если у тебя другая GPU, начни с ближайшего варианта по VRAM. Например:

```powershell
lift_vllm --gpu 3090
```

После запуска сервера extraction:

```powershell
lift_extract "Обращения\1.pdf" "tools/lift_lab/runs/appeal-001-vllm" `
  --schema "tools/lift_lab/schemas/appeal_text.schema.json"
```

## Вариант B: HuggingFace backend

Это проще концептуально, но тяжелее по зависимостям: ставит `torch`, `transformers`, `accelerate`.

Установка:

```powershell
python -m pip install -U pip
python -m pip install "lift-pdf[hf]"
```

Запуск:

```powershell
lift_extract "Обращения\1.pdf" "tools/lift_lab/runs/appeal-001-hf" `
  --schema "tools/lift_lab/schemas/appeal_text.schema.json" `
  --method hf
```

Если HF backend не видит GPU, проверь:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Ограничить страницы для первого теста

Чтобы не ждать вечность:

```powershell
lift_extract "Обращения\1.pdf" "tools/lift_lab/runs/appeal-001-page0" `
  --schema "tools/lift_lab/schemas/appeal_text.schema.json" `
  --page-range "0"
```

Для HF:

```powershell
lift_extract "Обращения\1.pdf" "tools/lift_lab/runs/appeal-001-page0-hf" `
  --schema "tools/lift_lab/schemas/appeal_text.schema.json" `
  --page-range "0" `
  --method hf
```

## Что смотреть

`lift_extract` должен сохранить в output directory:

```text
<filename>.json
<filename>_metadata.json
```

Нас интересует:

- есть ли `appeal_text`;
- не выдумывает ли модель факты;
- лучше ли результат, чем `PaddleOCR + LLM cleanup`;
- сколько времени занял документ;
- сколько VRAM/RAM занято.

## Источники

- GitHub README: https://github.com/datalab-to/lift
- Установка: `pip install lift-pdf`, `pip install lift-pdf[hf]`
- CLI: `lift_extract input.pdf ./output --schema schema.json`
- vLLM server: `lift_vllm --gpu 3090`

