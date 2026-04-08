"""
Published aggregate statistics from the Anthropic Economic Index reports.

These numbers come directly from the three public reports and are used for
figures and cross-checks when the full HF dataset is unavailable.

Sources
-------
[1] "The Anthropic Economic Index," Sep 2025 (data: early 2025, Free+Pro Claude.ai only).
    https://www.anthropic.com/research/the-anthropic-economic-index
[2] "Economic Primitives," Jan 2026 (data: November 13-20 2025, adds 1P API).
    https://www.anthropic.com/research/anthropic-economic-index-january-2026-report
[3] "Learning Curves," Mar 2026 (data: February 5-12 2026, adds tenure analysis).
    https://www.anthropic.com/research/economic-index-march-2026-report
"""

import pandas as pd

# ── Automation vs Augmentation ─────────────────────────────────────────────
# The split evolved across reports. [1] covered Claude.ai only (Free+Pro).
# [2] added 1P API data and showed an augmentation shift.
# [3] noted augmentation continued increasing on Claude.ai, API automation
#     "decreased sharply."

# Initial report [1] — Claude.ai only (early 2025 data)
INTERACTION_MODES_INITIAL = pd.DataFrame({
    "platform": ["Claude.ai", "Claude.ai"],
    "mode":     ["Augmentation", "Automation"],
    "share":    [57.4, 42.6],
    "period":   ["early 2025", "early 2025"],
})

# January 2026 report [2] — November 2025 data, first API comparison
INTERACTION_MODES = pd.DataFrame({
    "platform": ["Claude.ai", "Claude.ai", "First-party API", "First-party API"],
    "mode":     ["Augmentation", "Automation", "Augmentation", "Automation"],
    "share":    [52.0, 45.0, 25.0, 75.0],
    "period":   ["Nov 2025", "Nov 2025", "Nov 2025", "Nov 2025"],
})
# Note: Claude.ai remaining 3% classified as "Neither" [2]

# Sub-breakdown from [1] (matches 57.4/42.6 initial split, not Nov 2025)
AUGMENTATION_DETAIL = pd.DataFrame({
    "subcategory":  ["Task Iteration", "Learning", "Validation"],
    "share_of_all": [31.3, 23.3, 2.8],
    "source":       ["[1]", "[1]", "[1]"],
})

AUTOMATION_DETAIL = pd.DataFrame({
    "subcategory":  ["Directive", "Feedback Loop"],
    "share_of_all": [27.8, 14.8],
    "source":       ["[1]", "[1]"],
})

# Updated November 2025 sub-categories from [2]
# Directive dropped from 39% to 32% on Claude.ai; task iteration increased
AUTOMATION_DETAIL_NOV2025 = pd.DataFrame({
    "subcategory":  ["Directive"],
    "claude_ai_pct": [32.0],
    "api_pct":       [64.0],
    "source":       ["[2]"],
})

# ── Occupational distribution ──────────────────────────────────────────────
# Source: [1] initial report, [2] updated November 2025 snapshot

OCCUPATION_SHARES = pd.DataFrame({
    "occupation": [
        "Computer & Mathematical",
        "Arts, Design, Entertainment, Media",
        "Educational Instruction & Library",
        "Office & Administrative Support",
        "Life Sciences",
        "Business & Financial",
        "Management",
        "Other",
    ],
    "claude_ai_pct": [34.0, 11.0, 15.0, 8.0, 6.4, 5.9, 5.0, 15.7],
    "api_pct":       [46.0, 3.0, 4.0, 13.0, 2.0, 5.0, 3.0, 24.0],
    "us_workforce_pct": [3.4, 2.0, 6.2, 12.0, 0.8, 5.5, 7.8, 62.3],
})

# ── Education / deskilling metrics ─────────────────────────────────────────
# Source: [2] section on education primitives

EDUCATION_STATS = {
    "mean_all_tasks_yrs":          13.2,
    "mean_claude_covered_yrs":     14.4,
    "high_school_success_rate":    0.70,
    "college_success_rate":        0.66,
    "high_school_speedup":         9,   # x faster
    "college_speedup":             12,  # x faster
}

# ── Task complexity and success frontier ───────────────────────────────────
# Source: [2] section on task horizons

TASK_FRONTIER = pd.DataFrame({
    "platform":         ["Claude.ai (Opus 4.5)", "API (Sonnet)"],
    "fifty_pct_hours":  [19.0, 3.5],
})

# Software dev vs personal tasks (Table from [2])
TASK_COMPLEXITY = pd.DataFrame({
    "category":             ["Software Development", "Personal Life Mgmt", "Global Average"],
    "human_time_hours":     [3.3, 1.8, 3.1],
    "education_years":      [13.8, 9.25, None],
    "work_use_pct":         [64, 17, 46],
    "success_rate":         [0.61, 0.78, 0.67],
})

# ── Geographic patterns ────────────────────────────────────────────────────
# Source: [2] and [3]

GEO_CONVERGENCE = {
    # US state-level concentration
    "us_top5_share_aug2025":    0.50,
    "us_top5_share_feb2026":    None,  # [3] says "30% -> 24%" for top 5 states
    "us_gini_aug2025":          0.37,
    "us_gini_nov2025":          0.32,
    "us_convergence_years":     "2-5 initially, revised to 5-9",

    # Global
    "global_top20_share_aug2025": 0.45,
    "global_top20_share_feb2026": 0.48,
    "global_trend":               "inequality increasing",
}

# ── Tenure / learning effects ──────────────────────────────────────────────
# Source: [3]

TENURE_EFFECTS = {
    "high_tenure_threshold_months":   6,
    "baseline_success_advantage_pp":  5,
    "controlled_success_advantage_pp": 3,  # with task controls
    "full_controls_advantage_pp":     4,
    "personal_use_new":               0.44,
    "personal_use_experienced":       0.38,
    "education_increase_per_year":    1.0,  # years of ed per year of adoption
}

# ── Macro productivity estimates ───────────────────────────────────────────
# Source: [2]

PRODUCTIVITY = {
    "unadjusted_annual_pp":                1.8,
    "success_adjusted_claude_ai_pp":       1.2,
    "success_adjusted_api_pp":             1.0,
    "complementary_sigma_0_5_pp":          "0.7-0.9",
    "substitutes_sigma_1_5_pp":            "2.2-2.6",
}

# ── Temporal trends ────────────────────────────────────────────────────────
# Source: [3]

TRENDS = pd.DataFrame({
    "metric": [
        "Top-10 task concentration (Claude.ai)",
        "Top-10 task concentration (API)",
        "Avg task value ($/hr, Claude.ai)",
        "Education required (yrs, Claude.ai)",
        "Coursework share",
        "Personal use share",
        "Management occupation share",
    ],
    "earlier": [
        "24% (Nov 2025)", "28% (Aug 2025)",
        "$49.30", "12.2",
        "19%", "35%", "3%",
    ],
    "later": [
        "19% (Feb 2026)", "33% (Feb 2026)",
        "$47.90", "11.9",
        "12%", "42%", "5%",
    ],
    "direction": [
        "declining", "rising",
        "declining", "declining",
        "declining", "rising", "rising",
    ],
})
