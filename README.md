# Anthropic Economic Index Analysis

Anthropic released the Economic Index dataset in March 2026 — about 2 million anonymized conversations from the week of February 5–12, classified against O*NET occupational tasks. Both Claude.ai and the 1P API are in there.

The question I kept coming back to: the same model, deployed two different ways, with a 20-point gap in task success rates. Claude.ai sits around 70%; the API around 50%. That's not a model capability story. So what is it?

This repo is my attempt to work that out.

## What I found

For 258 O*NET tasks that appear in both platforms, Claude.ai outperforms the API in 84.9% of cases. The mean gap is 15.9 percentage points. And it tracks pretty cleanly with how directive the API interactions are — tasks where the API skips the back-and-forth and goes straight to instruction-following tend to show the widest gaps.

| Metric | Claude.ai | 1P API |
|--------|-----------|--------|
| Task success | 69.9% | 50.5% |
| Augmentation share | 52.8% | 17.2% |
| Directive share | 32.6% | 58.2% |
| Learning share | 22.4% | 4.4% |
| Speed vs human baseline | 12.8× | 24.9× |

The API is roughly twice as fast as Claude.ai relative to a human doing the same work. More speed, less human involvement, lower success rate. The tradeoff is real and measurable.

Per-task summary (258 shared tasks):
- Claude.ai outperforms in 84.9% of cases
- Mean success gap: +15.9pp for Claude.ai
- API directive share → success gap: r = −0.263, p < 0.001

## Data

**Source**: [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex), released 2026-03-24  
**Coverage**: Feb 5–12, 2026  
**License**: CC-BY-4.0

Long-format dataset — each row is one metric value for a geography × facet × variable combination:

- `aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv` — ~1M Claude.ai conversations (425K rows)
- `aei_raw_1p_api_2026-02-05_to_2026-02-12.csv` — ~1M API conversations (195K rows)
- `job_exposure.csv` — AI exposure scores for 756 O*NET occupations
- `task_penetration.csv` — AI penetration scores for ~18K O*NET tasks

## Notebooks

| Notebook | What it covers |
|----------|----------------|
| `01_core_analysis.ipynb` | Collaboration modes, use case breakdown, geographic distribution, task complexity |
| `02_advanced_analysis.ipynb` | Augmentation–success correlation, partial correlations, platform divergence |
| `03_platform_oversight.ipynb` | The cross-platform natural experiment — same tasks, two deployment contexts |

Each notebook has a corresponding build script (`notebooks/build_notebook_0[1-3].py`) that regenerates it from scratch.

## Getting started

```bash
pip install -r requirements.txt
python scripts/download_data.py     # pulls all 4 files from HuggingFace
jupyter notebook notebooks/
```

Python 3.9+. Data files aren't in the repo — gitignored, download separately. Figures are committed so you can browse the results without running anything.

## Repo layout

```
├── data/          # data files (gitignored — download via script)
├── figures/       # all generated figures
├── notebooks/     # notebooks + build scripts
├── scripts/       # download_data.py, setup_repo.sh
└── src/           # data.py, style.py
```
