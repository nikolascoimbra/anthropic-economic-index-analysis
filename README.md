# The AI Learning Curve

An independent analysis of Anthropic's [Economic Index](https://www.anthropic.com/research/the-anthropic-economic-index) dataset. The core finding: how people use AI matters more than whether they use it — and the signature of experience is a shift from delegation to collaboration.

## What this is

Anthropic published a dataset of 2 million anonymized conversations (1M Claude.ai, 1M first-party API) classified against the O\*NET occupational task database, alongside supplementary labor market files covering 756 occupations and ~18,000 tasks. They released the data on [HuggingFace](https://huggingface.co/datasets/Anthropic/EconomicIndex) alongside three reports ([Sep 2025](https://www.anthropic.com/research/the-anthropic-economic-index), [Jan 2026](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report), [Mar 2026](https://www.anthropic.com/research/economic-index-march-2026-report)) focused on productivity and labor economics.

This repo runs the same data through a different lens. The question: **what does it look like when people get better at working with AI?**

Anthropic's March 2026 report introduced a tenure analysis showing experienced users (6+ months) achieve higher success rates and more collaborative interaction styles. This repo extends that finding by identifying the task-level mechanism behind it:

- **Augmentation predicts success.** Across 1,923 O\*NET tasks, the share of augmentative (collaborative) interactions correlates with task success rate at r = 0.489 (p < 10⁻¹¹⁶). Tasks in the most-augmented quartile succeed 81.6% of the time vs. 65.2% in the least — a 16.4 percentage-point gap.
- **The correlation survives controls.** Controlling for education requirements: r = 0.644. Controlling for volume: r = 0.636. Both controls together: r = 0.640. This is not a confound — it is a structural relationship.
- **Task penetration reveals which tasks scale.** Joining the labor market supplementary data, high-penetration tasks show distinct automation/augmentation profiles and different success ceilings.

The learning curve, then, is not just "people get faster." Experienced users shift toward augmentation-dominant interaction patterns — and those patterns are the ones that actually work.

All statistics are computed directly from the dataset. Nothing is hardcoded from the reports — the reports are referenced only for cross-validation.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/anthropic-economic-index-analysis.git
cd anthropic-economic-index-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

The dataset is ~140MB across four files. Two options:

**Option 1: Download script** (requires `huggingface_hub`)
```bash
python scripts/download_data.py
```

**Option 2: Manual download** from [HuggingFace](https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/release_2026_03_24)

Download these files into `data/`:

| File | Size | Contents |
|---|---|---|
| `aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv` | 96 MB | Claude.ai conversations (Free/Pro/Max) |
| `aei_raw_1p_api_2026-02-05_to_2026-02-12.csv` | 44 MB | First-party API conversations |
| `job_exposure.csv` | 37 KB | 756 occupations with AI exposure scores |
| `task_penetration.csv` | 1.9 MB | ~18,000 tasks with penetration scores |

The first two are under `release_2026_03_24/data/`, the latter two under `release_2026_03_24/labor_market_impacts/`.

Then open the notebooks:

```bash
jupyter notebook notebooks/01_core_analysis.ipynb
```

## Repo structure

```
├── notebooks/
│   ├── 01_core_analysis.ipynb       # Task distribution, collaboration modes,
│   │                                # success rates, education, geographic patterns
│   └── 02_advanced_analysis.ipynb   # Augmentation–success correlation, labor market
│                                    # integration, geographic inequality, OEI
├── src/
│   ├── data.py            # Data loader and query functions for the AEI CSVs
│   ├── report_data.py     # Published report statistics (for cross-validation)
│   └── style.py           # Plot styling (Tufte-inspired, muted palette)
├── scripts/
│   └── download_data.py   # Downloads all four files from HuggingFace
├── figures/                # Generated figures from the notebooks
├── data/                   # CSV files (gitignored — download via script)
├── requirements.txt
└── LICENSE
```

## Key numbers

All computed from the February 5–12, 2026 release (2M conversations).

| Metric | Claude.ai | 1P API |
|---|---|---|
| Augmentation share | 52.8% | 17.2% |
| Automation share | 44.2% | 67.6% |
| Task success rate | 69.9% | 50.5% |
| Human-only time (mean) | 3.1 hours | 1.7 hours |
| Human+AI time (mean) | 14 minutes | 4 minutes |
| Implied speedup | 12.8× | 24.9× |

**Task-level augmentation–success correlation:** r = 0.489, p < 10⁻¹¹⁶ (N = 1,923 tasks, Claude.ai). Partial r = 0.640 controlling for education + volume.

**Tenure effects** (from March 2026 report): 6+ month users show +5pp bivariate, +3pp complexity-controlled, +4pp all-controls success advantage over new users.

## Data

The dataset is published under CC-BY by Anthropic. Each row represents one metric value for a specific geography × facet × variable combination. The facets include O\*NET task mappings, collaboration patterns (directive, feedback loop, task iteration, learning, validation), use case, task success, education requirements, time estimates, and AI autonomy scores. Full schema in [Anthropic's data documentation](https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/release_2026_03_24/data_documentation.md).

## License

Code: MIT. Data: CC-BY (Anthropic).
