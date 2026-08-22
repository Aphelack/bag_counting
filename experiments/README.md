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

- `01_sam3_segmentation_test.ipynb` — interactive SAM 3 test (text-prompt
  segmentation, `facebookresearch/sam3` native API) on frames from
  `../storage/input/input.mp4`. Requires a CUDA GPU, the `sam3` package
  (`uv pip install git+https://github.com/facebookresearch/sam3.git`), and
  a locally downloaded checkpoint + BPE vocab — see the cells in the
  notebook. Frame sampling/visualization work without a GPU.

  ⚠️ At one point this notebook had a hardcoded HF token in plain text —
  if you're reusing history from this repo, revoke that token at
  huggingface.co/settings/tokens and read it from an environment variable
  instead.

## Scripts

- `scripts/label_with_sam3.py` — the same SAM3 logic as the notebook, but
  run over the entire video (not a handful of frames) and writes a COCO
  dataset straight to `../training/`. See `../training/README.md` for the
  full labeling → training → inference pipeline.
