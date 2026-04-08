#!/usr/bin/env python3
"""
Download the Anthropic Economic Index dataset from HuggingFace.

Uses the 2026-03-24 release (February 2026 data).
Downloads all four files:
  - Two core conversation datasets (Claude.ai + 1P API)
  - Two labor market supplementary files (job exposure + task penetration)

Files are saved to data/ in the project root.

Usage:
    python scripts/download_data.py

Requirements:
    pip install huggingface_hub
"""

from pathlib import Path

REPO_ID = "Anthropic/EconomicIndex"
RELEASE = "release_2026_03_24"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    # Core conversation data
    f"{RELEASE}/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv",
    f"{RELEASE}/data/aei_raw_1p_api_2026-02-05_to_2026-02-12.csv",
    # Labor market supplementary data
    f"{RELEASE}/labor_market_impacts/job_exposure.csv",
    f"{RELEASE}/labor_market_impacts/task_penetration.csv",
]


def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in FILES:
        filename = Path(file_path).name
        dest = DATA_DIR / filename

        if dest.exists():
            print(f"Already exists: {dest}")
            continue

        print(f"Downloading {filename}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=str(DATA_DIR / "_hf_cache"),
        )
        # Move from cache to data/
        Path(downloaded).rename(dest)
        print(f"  Saved to {dest}")

    # Clean up cache directory
    cache_dir = DATA_DIR / "_hf_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    print("\nDone. Files in data/:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download()
