"""
Data loading and query utilities for the Anthropic Economic Index dataset.

The dataset is published on Hugging Face:
https://huggingface.co/datasets/Anthropic/EconomicIndex

Release used: 2026-03-24 (data collected February 5-12, 2026).
Four source files:
  - aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv  (Claude.ai Free/Pro/Max)
  - aei_raw_1p_api_2026-02-05_to_2026-02-12.csv     (First-party API)
  - job_exposure.csv                                  (756 occupations, AI exposure)
  - task_penetration.csv                              (~18,000 tasks, penetration scores)

Each row in the core files is one metric value for a specific geography × facet
× variable combination.  See data_documentation.md for full schema.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# File names for the 2026-03-24 release
_CAI_FILE = "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv"
_API_FILE = "aei_raw_1p_api_2026-02-05_to_2026-02-12.csv"
_JOB_EXPOSURE_FILE = "job_exposure.csv"
_TASK_PENETRATION_FILE = "task_penetration.csv"

# Collaboration mode groupings (from Anthropic's classification)
AUGMENTATION_MODES = {"task iteration", "learning", "validation"}
AUTOMATION_MODES = {"directive", "feedback loop"}


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_claude_ai(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the Claude.ai usage dataset."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return pd.read_csv(d / _CAI_FILE)


def load_api(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the 1P API usage dataset."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return pd.read_csv(d / _API_FILE)


def load_both(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate both platforms with a 'platform' column."""
    cai = load_claude_ai(data_dir)
    cai["platform"] = "Claude.ai"
    api = load_api(data_dir)
    api["platform"] = "API"
    return pd.concat([cai, api], ignore_index=True)


def load_job_exposure(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the job exposure supplementary file (756 occupations)."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return pd.read_csv(d / _JOB_EXPOSURE_FILE)


def load_task_penetration(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the task penetration supplementary file (~18,000 tasks).

    Normalizes task names to lowercase for joining with core data.
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / _TASK_PENETRATION_FILE)
    # Core data uses lowercase task names; task_penetration uses title case
    if "task" in df.columns:
        df["task_lower"] = df["task"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Query helpers (all operate on the long-format DataFrames)
# ---------------------------------------------------------------------------

def query_facet(
    df: pd.DataFrame,
    facet: str,
    variable: Optional[str] = None,
    geography: str = "global",
    level: Optional[int] = None,
) -> pd.DataFrame:
    """Filter to a specific facet, optionally by variable and geography."""
    mask = (df["facet"] == facet) & (df["geography"] == geography)
    if variable is not None:
        mask &= df["variable"] == variable
    if level is not None:
        mask &= df["level"] == level
    return df[mask].copy()


def get_collaboration_split(df: pd.DataFrame, geography: str = "global") -> dict:
    """Compute augmentation / automation / none percentages from raw data.

    Returns dict with keys: augmentation_pct, automation_pct, none_pct,
    and a 'detail' DataFrame with per-mode percentages.
    """
    collab = query_facet(df, "collaboration", "collaboration_pct", geography)
    detail = collab[["cluster_name", "value"]].rename(
        columns={"cluster_name": "mode", "value": "pct"}
    )
    aug = detail[detail["mode"].isin(AUGMENTATION_MODES)]["pct"].sum()
    auto = detail[detail["mode"].isin(AUTOMATION_MODES)]["pct"].sum()
    none_pct = detail[detail["mode"] == "none"]["pct"].sum()
    return {
        "augmentation_pct": aug,
        "automation_pct": auto,
        "none_pct": none_pct,
        "detail": detail.reset_index(drop=True),
    }


def get_global_stats(df: pd.DataFrame) -> dict:
    """Extract key global summary statistics."""
    g = "global"
    stats = {}

    # Education
    ed = query_facet(df, "human_education_years", "human_education_years_mean", g)
    if len(ed):
        stats["education_years_mean"] = ed["value"].iloc[0]

    ed_med = query_facet(df, "human_education_years", "human_education_years_median", g)
    if len(ed_med):
        stats["education_years_median"] = ed_med["value"].iloc[0]

    # AI autonomy
    aa = query_facet(df, "ai_autonomy", "ai_autonomy_mean", g)
    if len(aa):
        stats["ai_autonomy_mean"] = aa["value"].iloc[0]

    # Human-only time (hours)
    ht = query_facet(df, "human_only_time", "human_only_time_mean", g)
    if len(ht):
        stats["human_only_time_mean"] = ht["value"].iloc[0]
    ht_med = query_facet(df, "human_only_time", "human_only_time_median", g)
    if len(ht_med):
        stats["human_only_time_median"] = ht_med["value"].iloc[0]

    # Human-with-AI time (minutes)
    hat = query_facet(df, "human_with_ai_time", "human_with_ai_time_mean", g)
    if len(hat):
        stats["human_with_ai_time_mean"] = hat["value"].iloc[0]
    hat_med = query_facet(df, "human_with_ai_time", "human_with_ai_time_median", g)
    if len(hat_med):
        stats["human_with_ai_time_median"] = hat_med["value"].iloc[0]

    # Task success
    ts = query_facet(df, "task_success", "task_success_pct", g)
    success_row = ts[ts["cluster_name"] == "yes"]
    if len(success_row):
        stats["task_success_pct"] = success_row["value"].iloc[0]

    # Use case
    uc = query_facet(df, "use_case", "use_case_pct", g)
    for _, row in uc.iterrows():
        if row["cluster_name"] != "none":
            stats[f"use_case_{row['cluster_name']}_pct"] = row["value"]

    # Collaboration split
    collab = get_collaboration_split(df, g)
    stats["augmentation_pct"] = collab["augmentation_pct"]
    stats["automation_pct"] = collab["automation_pct"]

    return stats


def get_top_tasks(
    df: pd.DataFrame,
    n: int = 20,
    level: int = 0,
    geography: str = "global",
) -> pd.DataFrame:
    """Return top N O*NET tasks by percentage share."""
    tasks = query_facet(df, "onet_task", "onet_task_pct", geography, level)
    tasks = tasks[~tasks["cluster_name"].isin(["none", "not_classified"])]
    return (
        tasks[["cluster_name", "value"]]
        .rename(columns={"cluster_name": "task", "value": "pct"})
        .sort_values("pct", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_request_categories(
    df: pd.DataFrame,
    level: int = 2,
    geography: str = "global",
) -> pd.DataFrame:
    """Return request categories (broadest occupational groupings at level 2)."""
    req = query_facet(df, "request", "request_pct", geography, level)
    req = req[~req["cluster_name"].isin(["not_classified"])]
    return (
        req[["cluster_name", "value"]]
        .rename(columns={"cluster_name": "category", "value": "pct"})
        .sort_values("pct", ascending=False)
        .reset_index(drop=True)
    )


def get_task_collaboration(
    df: pd.DataFrame,
    task_name: str,
    geography: str = "global",
) -> pd.DataFrame:
    """Get collaboration breakdown for a specific O*NET task."""
    tc = query_facet(df, "onet_task::collaboration", "onet_task_collaboration_pct", geography)
    mask = tc["cluster_name"].str.startswith(task_name + "::")
    result = tc[mask].copy()
    result["collab_mode"] = result["cluster_name"].str.split("::").str[1]
    return result[["collab_mode", "value"]].rename(columns={"value": "pct"}).reset_index(drop=True)


def get_task_success_rates(
    df: pd.DataFrame,
    n: int = 20,
    geography: str = "global",
) -> pd.DataFrame:
    """Get success rates for top tasks."""
    # Get top tasks first
    top = get_top_tasks(df, n=n, geography=geography)
    ts = query_facet(df, "onet_task::task_success", "onet_task_task_success_pct", geography)

    rows = []
    for task in top["task"]:
        success = ts[(ts["cluster_name"] == f"{task}::yes")]
        if len(success):
            rows.append({"task": task, "success_pct": success["value"].iloc[0]})
    return pd.DataFrame(rows)


def get_country_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Get country-level usage percentages."""
    countries = query_facet(df, "country", "usage_pct", "country")
    countries = countries[countries["geo_id"] != "not_classified"]
    return (
        countries[["geo_id", "value"]]
        .rename(columns={"geo_id": "country", "value": "usage_pct"})
        .sort_values("usage_pct", ascending=False)
        .reset_index(drop=True)
    )


def get_us_state_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Get US state-level usage percentages."""
    states = df[
        (df["facet"] == "country-state")
        & (df["geography"] == "country-state")
        & (df["variable"] == "usage_pct")
        & (df["geo_id"].str.startswith("US-"))
    ].copy()
    return (
        states[["geo_id", "value"]]
        .rename(columns={"geo_id": "state", "value": "usage_pct"})
        .sort_values("usage_pct", ascending=False)
        .reset_index(drop=True)
    )


def get_task_education(
    df: pd.DataFrame,
    n: int = 20,
    geography: str = "global",
) -> pd.DataFrame:
    """Get mean education years per task for top tasks."""
    top = get_top_tasks(df, n=n, geography=geography)
    ed = query_facet(df, "onet_task::human_education_years",
                     "onet_task_human_education_years_mean", geography)

    rows = []
    for task in top["task"]:
        # Intersection cluster_names use format "task_name::value"
        task_ed = ed[ed["cluster_name"] == f"{task}::value"]
        if len(task_ed) == 0:
            task_ed = ed[ed["cluster_name"] == task]
        if len(task_ed):
            rows.append({"task": task, "education_years": task_ed["value"].iloc[0]})
    return pd.DataFrame(rows)


def get_task_time(
    df: pd.DataFrame,
    n: int = 20,
    geography: str = "global",
) -> pd.DataFrame:
    """Get mean human-only and human-with-AI time per task."""
    top = get_top_tasks(df, n=n, geography=geography)
    hot = query_facet(df, "onet_task::human_only_time",
                      "onet_task_human_only_time_mean", geography)
    hat = query_facet(df, "onet_task::human_with_ai_time",
                      "onet_task_human_with_ai_time_mean", geography)

    rows = []
    for task in top["task"]:
        # Intersection cluster_names use format "task_name::value"
        ho = hot[hot["cluster_name"] == f"{task}::value"]
        if len(ho) == 0:
            ho = hot[hot["cluster_name"] == task]
        ha = hat[hat["cluster_name"] == f"{task}::value"]
        if len(ha) == 0:
            ha = hat[hat["cluster_name"] == task]
        if len(ho) and len(ha):
            rows.append({
                "task": task,
                "human_only_time": ho["value"].iloc[0],
                "human_with_ai_time": ha["value"].iloc[0],
            })
    return pd.DataFrame(rows)


def get_task_autonomy(
    df: pd.DataFrame,
    n: int = 20,
    geography: str = "global",
) -> pd.DataFrame:
    """Get mean AI autonomy per task for top tasks."""
    top = get_top_tasks(df, n=n, geography=geography)
    aa = query_facet(df, "onet_task::ai_autonomy",
                     "onet_task_ai_autonomy_mean", geography)

    rows = []
    for task in top["task"]:
        # Intersection cluster_names use format "task_name::value"
        task_aa = aa[aa["cluster_name"] == f"{task}::value"]
        if len(task_aa) == 0:
            task_aa = aa[aa["cluster_name"] == task]
        if len(task_aa):
            rows.append({"task": task, "autonomy_mean": task_aa["value"].iloc[0]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print(f"Shape: {df.shape}")
    print(f"Platform: {df['platform_and_product'].unique()}")
    print(f"Date range: {df['date_start'].unique()} to {df['date_end'].unique()}")
    print(f"Geographies: {df['geography'].unique()}")
    print(f"Facets ({df['facet'].nunique()}): {sorted(df['facet'].unique())}")
    print(f"Variables ({df['variable'].nunique()})")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        nuniq = df[col].nunique()
        print(f"  {col:<40} {str(dtype):<12} nulls={nulls:<6} unique={nuniq}")
