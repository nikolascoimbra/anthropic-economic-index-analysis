# Anthropic Economic Index Analysis

Analysis of the Anthropic Economic Index public dataset, examining how deployment context shapes AI task outcomes.

**Central finding**: The same model family achieves 70% task success on Claude.ai and 50% on the 1P API. For 258 tasks present in both platforms, Claude.ai outperforms in 84.9% of cases (mean gap: 15.9pp). The gap is predicted by directive interaction share on the API (r = −0.263, p < 0.001) and reflects a measurable tradeoff between speed and human oversight.

## Dataset

**Source**: [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex), release 2026-03-24  
**Coverage**: Conversations sampled February 5–12, 2026  
**License**: CC-BY-4.0

Four files:
- `aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv` — 1M Claude.ai conversations, 425K rows (long format)
- `aei_raw_1p_api_2026-02-05_to_2026-02-12.csv` — 1M API conversations, 195K rows (long format)
- `job_exposure.csv` — AI exposure scores for 756 O*NET occupations
- `task_penetration.csv` — AI penetration scores for ~18,000 O*NET tasks

Data is in long format: each row is one metric value for a geography × facet × variable combination.

## Key numbers

| Metric | Claude.ai | 1P API |
|--------|-----------|--------|
| Task success | 69.9% | 50.5% |
| Augmentation share | 52.8% | 17.2% |
| Directive share | 32.6% | 58.2% |
| Learning share | 22.4% | 4.4% |
| Speed vs baseline | 12.8× | 24.9× |

**Per-task comparison (258 shared tasks)**:
- Claude.ai outperforms API in 84.9% of tasks
- Mean success gap: +15.9pp for Claude.ai
- API directive share → success gap: r = −0.263, p < 0.001

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_core_analysis.ipynb` | Collaboration modes, use cases, geographic distribution, task complexity |
| `02_advanced_analysis.ipynb` | Augmentation–success correlation, partial correlations, platform divergence, oversight erosion index |
| `03_platform_oversight.ipynb` | Cross-platform natural experiment, per-task success comparison, mechanism analysis, speed tradeoff |

Build scripts: `notebooks/build_notebook_0[1-3].py`

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py     # downloads all 4 files from HuggingFace
jupyter notebook notebooks/
```

Requires Python 3.9+. Data files are excluded from the repo (see `.gitignore`); download separately.

## Structure

```
├── data/                    # downloaded data files (gitignored)
├── figures/                 # all generated figures
├── notebooks/               # analysis notebooks + build scripts
├── scripts/                 # download_data.py, setup_repo.sh
└── src/                     # data.py, style.py, report_data.py
```

## Figures

New in notebook 03:
- `p01_cross_platform_scatter.png` — per-task success: Claude.ai vs API (natural experiment)
- `p02_mode_by_platform.png` — mode distribution by platform
- `p03_directive_success_gap.png` — API directive share vs success gap
- `p04_speed_success_tradeoff.png` — speed, success, and oversight tradeoff

From notebooks 01–02:
- `01–07_*.png` — core analysis figures
- `10–15_*.png` — advanced analysis figures
