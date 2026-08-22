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

- `01_sam3_segmentation_test.ipynb` — тест SAM 3 (text-prompt сегментация) на кадрах из
  `../storage/input/input.mp4` как bootstrap-разметка под дообучение RTMDet. Требует
  одобренный доступ к `facebook/sam3` на HuggingFace (`HF_TOKEN`) и CUDA GPU — see markdown
  cells in the notebook. Frame-sampling/visualization cells работают без GPU.
