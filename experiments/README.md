# Experiments

Локальное окружение (uv) для исследовательских ноутбуков — отдельно от `app/`,
т.к. сюда идут тяжёлые/специфичные ML-зависимости (torch, transformers), которые
не нужны продовому FastAPI-сервису.

## Setup

```bash
cd experiments
uv sync
uv run python -m ipykernel install --user --name bag-counting-experiments --display-name "bag-counting-experiments"
```

Затем открыть `notebooks/` в Jupyter/VS Code и выбрать kernel `bag-counting-experiments`.

## Notebooks

- `01_sam3_segmentation_test.ipynb` — интерактивный тест SAM 3 (text-prompt сегментация,
  `facebookresearch/sam3` native API) на кадрах из `../storage/input/input.mp4`. Требует
  CUDA GPU, пакет `sam3` (`uv pip install git+https://github.com/facebookresearch/sam3.git`)
  и локально скачанный checkpoint + BPE vocab — see cells in the notebook. Сэмплирование
  кадров/визуализация работают без GPU.

  ⚠️ На момент последнего обновления в ноутбуке был захардкожен HF-токен в открытом виде —
  если репозиторий публикуется на GitHub, токен нужно отозвать на
  huggingface.co/settings/tokens и заменить на чтение из переменной окружения.

## Scripts

- `scripts/label_with_sam3.py` — та же SAM3-логика, что и в ноутбуке, но прогоняется по
  всему видео (не по горстке кадров) и сразу пишет COCO-датасет для `../training/`. См.
  `../training/README.md` за полным пайплайном разметка → обучение → инференс.
