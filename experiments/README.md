# Experiments

Local uv environment for research notebooks — kept separate from `app/`
because it pulls in heavy/specific ML dependencies (torch, transformers)
that the production FastAPI service doesn't need.

## Setup

```bash
cd experiments
uv sync
uv run python -m ipykernel install --user --name bag-counting-experiments --display-name "bag-counting-experiments"
```

Then open `notebooks/` in Jupyter/VS Code and select the
`bag-counting-experiments` kernel.

## Notebooks
  Поставить sam3 было не просто. Официальная модель от meta требует разрешения от компании на huggingface. Скачал веса из зеркала, но пришлось повозиться с зависимости, возможно ноутбук сразу не запуститься. 
  sam3 может работает по текстовому промту. Baseline промпт bag или package были недостаточно эффективны. Оказывается мешки больше похожи на подушки для модели, чем на мешки
Пример
![alt text](<Screenshot from 2026-08-23 03-54-26.png>)

  Тут захватывается мешки лежащие на полу. Как я понял, в задании нужно их игнорировать. Простым понижением trethhold это не починить. Поэтому подбирался другой промпт. 
![alt text](<Screenshot from 2026-08-23 03-56-15.png>)

  Вот этот достаточно хорошо задлеял мешки на конвейере и лежащие на полу. Самый высокий confidence у мешков на полу за все кадры 0.7. А на конвейере >0.9. 
![alt text](<Screenshot from 2026-08-23 03-55-55.png>)

  Взял threthhold 0.75 и разметил датасет для обучения. С обучением проблем не было - все обучилось с первого раза за 5 минут (использовалась A100 на 40gb).

## Scripts

- `scripts/label_with_sam3.py` — the same SAM3 logic as the notebook, but
  run over the entire video (not a handful of frames) and writes a COCO
  dataset straight to `../training/`. See `../training/README.md` for the
  full labeling → training → inference pipeline.
