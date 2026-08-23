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

SAM 3 itself is not a trivial install. The official Meta weights require
Hugging Face to approve access; I used a mirrored checkpoint instead, which
still needed some dependency wrangling — the notebook may not run clean on
the first try.

## Notebooks

- `01_sam3_segmentation_test.ipynb` — prompt tuning for the labeling step.

SAM 3 segments from a text prompt. The obvious prompts didn't work:

**Baseline prompt `"bags"`** — weak and inconsistent. On some frames it
barely clears 0.2–0.3 confidence on the actual belt bag:

![Baseline "bags" prompt: low, inconsistent confidence](<Screenshot from 2026-08-23 03-54-26.png>)

The model responds much better to a description of *shape and material*
than to the word "bag" — these sacks read to SAM 3 as pillows, not bags.
Rewording the prompt to **`"A white soft pillow positioned on an
industrial roller conveyor system."`** fixed the belt bag reliably
(consistently >0.9 on the same frame):

![Tuned prompt: >0.9 confidence on the belt bag](<Screenshot from 2026-08-23 03-55-55.png>)

That doesn't solve everything: the assignment only cares about bags
actually on the conveyor, but there's a pile of bags on the floor in
frame, and any prompt broad enough to catch "bag-shaped object" catches
those too. Lowering the threshold doesn't fix it — it makes the floor
bags worse, not better. Confidence turned out to be a good separator on
its own: across every frame I checked, floor bags topped out around 0.7,
belt bags stayed above 0.9:

![Same tuned prompt, floor pile: confidence caps around 0.7](<Screenshot from 2026-08-23 03-56-15.png>)

I used **`confidence_threshold=0.75`** to label the dataset — high enough
to exclude the floor pile, low enough to keep every real belt bag.

Training on the resulting labels worked on the first attempt, no
hyperparameter iteration needed: 50 epochs, ~5 minutes on an A100 40GB.

⚠️ At one point this notebook had a hardcoded HF token in plain text —
if you're reusing history from this repo, revoke that token at
huggingface.co/settings/tokens and read it from an environment variable
instead.

## Scripts

- `scripts/label_with_sam3.py` — the same SAM3 logic as the notebook, but
  run over the entire video (not a handful of frames) and writes a COCO
  dataset straight to `../training/`. See `../training/README.md` for the
  full labeling → training → inference pipeline.
