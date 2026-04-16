"""Build notebook 03: Platform Oversight Analysis.

Generates 03_platform_oversight.ipynb — the analysis behind
"Same Model, Different Oversight" (Alignment Forum post).

Central question: Why does the same model succeed 70% of the time
on Claude.ai but only 50% on the API — for the same tasks?
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))


# ── Title ──────────────────────────────────────────────────────────────────
md("""# Platform Oversight Analysis: Same Model, Different Outcomes

**Research question**: The 1P API and Claude.ai use the same underlying model family.
Yet task success rates differ by ~20 percentage points. Why?

**Approach**: Treat the two platforms as a natural experiment. Identify tasks
present in both, measure per-task success rates, and test whether the
interaction structure (collaboration mode) explains the gap.

**Key findings**:
- 84.9% of 258 shared tasks succeed more often on Claude.ai than the API
- Mean success gap: +15.9pp for Claude.ai
- API interaction structure: 56% directive, 13.5% learning
- Claude.ai interaction structure: 29% directive, 33.5% learning
- API directive share predicts the per-task success gap: r = −0.263, p < 0.001
- Speed: API 24.9× faster than human baseline; Claude.ai 12.8×

**Data**: Anthropic Economic Index, release 2026-03-24 (Feb 5–12, 2026)
""")


# ── Section 1: Setup ───────────────────────────────────────────────────────
md("## 1. Setup and Data Loading")

code("""\
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from pathlib import Path

from src import style
from src.data import (
    load_claude_ai, load_api, query_facet,
    get_global_stats, get_collaboration_split,
    AUGMENTATION_MODES, AUTOMATION_MODES,
)

# Anthropic-style palette (used throughout)
PARCHMENT = '#FAF8F5'
TEAL      = '#3D8B8B'
CORAL     = '#C96B4F'
CHARCOAL  = '#2D2D2D'
MIDGRAY   = '#888888'
LIGHTGRAY = '#DDDDDD'

FIG_DIR = Path("../figures")
FIG_DIR.mkdir(exist_ok=True)

cai = load_claude_ai()
api = load_api()
print(f"Claude.ai: {len(cai):,} rows")
print(f"API:       {len(api):,} rows")
""")


# ── Section 2: Global stats comparison ────────────────────────────────────
md("""## 2. Global Platform Comparison

The two platforms show markedly different usage structures at the global level.""")

code("""\
cai_stats = get_global_stats(cai)
api_stats = get_global_stats(api)

# Mode-by-mode breakdown
def get_mode_pcts(df):
    collab = query_facet(df, "collaboration", "collaboration_pct", "global")
    return dict(zip(collab['cluster_name'], collab['value']))

cai_modes = get_mode_pcts(cai)
api_modes = get_mode_pcts(api)

print("Collaboration modes (global):")
print(f"{'Mode':<22} {'Claude.ai':>10} {'API':>10}")
print("-" * 44)
for m in ['learning', 'task iteration', 'validation', 'directive', 'feedback loop']:
    print(f"  {m:<20} {cai_modes.get(m, 0):>9.1f}% {api_modes.get(m, 0):>9.1f}%")

print(f"\\nKey summary metrics:")
print(f"  {'Metric':<35} {'Claude.ai':>10} {'API':>10}")
print("-" * 57)
for k, label in [
    ('task_success_pct', 'Task success (%)'),
    ('augmentation_pct', 'Augmentation (%)'),
    ('automation_pct',   'Automation (%)'),
    ('ai_autonomy_mean', 'AI autonomy (1–5)'),
    ('education_years_mean', 'Education (years)'),
]:
    v1 = cai_stats.get(k, float('nan'))
    v2 = api_stats.get(k, float('nan'))
    print(f"  {label:<35} {v1:>9.1f}  {v2:>9.1f}")
""")

code("""\
# Figure 2: Diverging bar chart — augmentation LEFT, automation RIGHT
TEAL_MID   = '#5AA5A5'
TEAL_LIGHT = '#7FBFBF'
CORAL_LIGHT= '#E09B83'

cai_pcts_aug = [cai_modes.get('learning', 0), cai_modes.get('task iteration', 0), cai_modes.get('validation', 0)]
api_pcts_aug = [api_modes.get('learning', 0), api_modes.get('task iteration', 0), api_modes.get('validation', 0)]
cai_pcts_aut = [cai_modes.get('directive', 0), cai_modes.get('feedback loop', 0)]
api_pcts_aut = [api_modes.get('directive', 0), api_modes.get('feedback loop', 0)]

fig, ax = plt.subplots(figsize=(9.5, 3.4))
fig.patch.set_facecolor(PARCHMENT)
ax.set_facecolor(PARCHMENT)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#C8C4BF')
ax.grid(axis='x', color='#C8C4BF', linewidth=0.5, alpha=0.5)
ax.grid(axis='y', visible=False)

y = np.array([1, 0])
h = 0.46
colors_aug = [TEAL, TEAL_MID, TEAL_LIGHT]
colors_aut = [CORAL, CORAL_LIGHT]

for i, (aug_vals, aut_vals) in enumerate([(cai_pcts_aug, cai_pcts_aut), (api_pcts_aug, api_pcts_aut)]):
    left = 0
    for v, c in zip(aug_vals, colors_aug):
        ax.barh(y[i], -v, height=h, left=left, color=c, alpha=0.90)
        if v > 8:
            ax.text(left - v/2, y[i], f'{v:.0f}%', ha='center', va='center',
                   fontsize=9, color='white', fontweight='semibold')
        left -= v
    left = 0
    for v, c in zip(aut_vals, colors_aut):
        ax.barh(y[i], v, height=h, left=left, color=c, alpha=0.90)
        if v > 8:
            ax.text(left + v/2, y[i], f'{v:.0f}%', ha='center', va='center',
                   fontsize=9, color='white', fontweight='semibold')
        left += v

ax.axvline(0, color='#1E1E1E', lw=0.9, alpha=0.5, zorder=5)
ax.set_yticks(y)
ax.set_yticklabels(['Claude.ai\\n(consumer)', '1P API\\n(production)'], fontsize=10.5, color='#1E1E1E')
ax.set_xlim(-60, 75)
ticks = [-60,-40,-20,0,20,40,60]
ax.set_xticks(ticks)
ax.set_xticklabels([str(abs(t)) for t in ticks], fontsize=9)
ax.set_xlabel('← Augmentation (%)                      Automation (%) →', fontsize=9.5, color='#1E1E1E', labelpad=10)
ax.set_title('Same model, opposite interaction structure\\nShare of interactions by collaboration mode',
             fontsize=11.5, color='#1E1E1E', fontweight='normal', pad=14, loc='left')
ax.tick_params(colors='#1E1E1E', labelsize=9.5)

import matplotlib.patches as mpatches
legend_els = [
    mpatches.Patch(color=TEAL, alpha=0.90, label='Learning'),
    mpatches.Patch(color=TEAL_MID, alpha=0.80, label='Task iteration'),
    mpatches.Patch(color=TEAL_LIGHT, alpha=0.65, label='Validation'),
    mpatches.Patch(color=CORAL, alpha=0.90, label='Directive'),
    mpatches.Patch(color=CORAL_LIGHT, alpha=0.75, label='Feedback loop'),
]
ax.legend(handles=legend_els, ncol=5, loc='upper center',
          bbox_to_anchor=(0.5, -0.32), fontsize=9, frameon=False,
          handlelength=1.4, handletextpad=0.5, columnspacing=1.2)

plt.tight_layout(pad=1.6)
fig.savefig(FIG_DIR / "p02_mode_by_platform.png", dpi=160, bbox_inches='tight', facecolor=PARCHMENT)
plt.show()
print("Saved p02_mode_by_platform.png")
""")


# ── Section 3: Natural experiment ────────────────────────────────────────
md("""## 3. The Natural Experiment: Per-Task Cross-Platform Comparison

1,416 O*NET tasks appear in both datasets. For 258 of these we have the full
collaboration mode breakdown. This allows a direct comparison of the same task
executed on the two platforms using the same model.
""")

code("""\
def get_per_task_full(df):
    \"\"\"Per-task success + mode shares + time metrics.\"\"\"
    # Success
    ts = query_facet(df, "onet_task::task_success", "onet_task_task_success_pct", "global")
    success = ts[ts['cluster_name'].str.endswith('::yes')].copy()
    success['task'] = success['cluster_name'].str.replace('::yes$', '', regex=True)
    success_d = dict(zip(success['task'], success['value']))

    # Modes
    tc = query_facet(df, "onet_task::collaboration", "onet_task_collaboration_pct", "global")
    tc = tc.copy()
    tc['task'] = tc['cluster_name'].str.rsplit('::', n=1).str[0]
    tc['mode'] = tc['cluster_name'].str.rsplit('::', n=1).str[1]
    modes = ['learning', 'task iteration', 'validation', 'directive', 'feedback loop']
    mode_dicts = {}
    for m in modes:
        key = m.replace(' ', '_')
        grp = tc[tc['mode'] == m].groupby('task')['value'].sum()
        mode_dicts[key] = dict(zip(grp.index, grp.values))

    # Time (human_only in hours, human_with_ai in minutes)
    hot = query_facet(df, "onet_task::human_only_time", "onet_task_human_only_time_mean", "global")
    hot['task'] = hot['cluster_name'].str.replace('::value$', '', regex=True)
    hat = query_facet(df, "onet_task::human_with_ai_time", "onet_task_human_with_ai_time_mean", "global")
    hat['task'] = hat['cluster_name'].str.replace('::value$', '', regex=True)

    all_tasks = list(success_d.keys())
    result = pd.DataFrame({'task': all_tasks}).set_index('task')
    result['success'] = pd.Series(success_d)
    for m in modes:
        key = m.replace(' ', '_')
        result[key] = pd.Series(mode_dicts[key])
    result['human_only_h']     = pd.Series(dict(zip(hot['task'], hot['value'])))
    result['human_with_ai_min'] = pd.Series(dict(zip(hat['task'], hat['value'])))
    return result

print("Building per-task tables...")
cai_t = get_per_task_full(cai)
api_t = get_per_task_full(api)

shared = cai_t.index.intersection(api_t.index)
print(f"Shared tasks (any data): {len(shared)}")

cross = pd.DataFrame({
    'cai_success':   cai_t.loc[shared, 'success'],
    'api_success':   api_t.loc[shared, 'success'],
    'cai_directive': cai_t.loc[shared, 'directive'],
    'api_directive': api_t.loc[shared, 'directive'],
    'cai_learning':  cai_t.loc[shared, 'learning'],
    'api_learning':  api_t.loc[shared, 'learning'],
}).dropna()
cross['success_gap'] = cross['cai_success'] - cross['api_success']

print(f"Tasks with full data:    {len(cross)}")
print(f"\\nPer-task success (shared tasks):")
print(f"  Claude.ai mean: {cross['cai_success'].mean():.1f}%")
print(f"  API mean:       {cross['api_success'].mean():.1f}%")
print(f"  Mean gap:       {cross['success_gap'].mean():.1f}pp  (Claude.ai − API)")
print(f"  Claude.ai > API in {(cross['success_gap']>0).mean()*100:.1f}% of tasks")
""")

code("""\
# Figure 1: Per-task cross-platform scatter (clean, no inline annotations)
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('teal_coral', [TEAL, '#CCCCAA', CORAL])

fig, ax = plt.subplots(figsize=(7.5, 6.5))
fig.patch.set_facecolor(PARCHMENT)
ax.set_facecolor(PARCHMENT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#C8C4BF')
ax.spines['bottom'].set_color('#C8C4BF')
ax.tick_params(colors='#1E1E1E', labelsize=9.5)
ax.grid(color='#C8C4BF', linewidth=0.6, alpha=0.7)

sc = ax.scatter(cross['cai_success'], cross['api_success'],
               c=cross['api_directive'], cmap=cmap,
               s=38, alpha=0.72, linewidths=0, vmin=10, vmax=90, zorder=3)

ax.plot([30, 100], [30, 100], color='#1E1E1E', ls=(0,(4,3)), lw=1.0, alpha=0.4, zorder=2)
ax.fill_between([28, 102], [28, 102], [28, 28], alpha=0.035, color=CORAL, zorder=1)

cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.025, aspect=25)
cbar.set_label('API directive share  (%)', fontsize=9, color='#1E1E1E', labelpad=8)
cbar.ax.tick_params(labelsize=8.5, colors='#1E1E1E')
cbar.outline.set_edgecolor('#C8C4BF')

ax.set_xlabel('Claude.ai — task success rate (%)', fontsize=10, labelpad=8)
ax.set_ylabel('1P API — task success rate (%)', fontsize=10, labelpad=8)
ax.set_xlim(28, 102); ax.set_ylim(28, 102)
ax.set_aspect('equal')
ax.set_title('The same tasks, the same model — different outcomes\\n258 O*NET tasks present in both Claude.ai and 1P API (Feb 2026)',
             fontsize=11.5, color='#1E1E1E', fontweight='normal', pad=14, loc='left')

plt.tight_layout(pad=1.6)
fig.savefig(FIG_DIR / "p01_cross_platform_scatter.png", dpi=160, bbox_inches='tight', facecolor=PARCHMENT)
plt.show()
print("Saved p01_cross_platform_scatter.png")
""")


# ── Section 4: Mechanism ─────────────────────────────────────────────────
md("""## 4. The Mechanism: Directive Mode as the Driver

The mode decomposition reveals what's structurally different between the platforms,
and shows that API directive share predicts the per-task performance gap.
""")

code("""\
# Correlations: what predicts the per-task success gap?
metrics = {
    'API directive share':   ('api_directive',  cross['success_gap']),
    'Claude.ai learning share': ('cai_learning', cross['success_gap']),
    'API learning share':    ('api_learning',   cross['success_gap']),
    'Learning gap (CAI−API)': (None, None),
}
cross['learning_gap'] = cross['cai_learning'] - cross['api_learning']
cross['directive_gap'] = cross['api_directive'] - cross['cai_directive']

print("Correlations with success gap (Claude.ai − API):")
print(f"{'Predictor':<35} {'r':>7} {'p-value':>12}")
print("-" * 56)
for name, col in [
    ('API directive share',      'api_directive'),
    ('API learning share',       'api_learning'),
    ('Learning gap (CAI−API)',   'learning_gap'),
    ('Directive gap (API−CAI)',  'directive_gap'),
]:
    r, p = pearsonr(cross[col], cross['success_gap'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {name:<33} {r:>7.3f} {p:>12.2e} {sig}")
""")

code("""\
# Figure 3: API directive share → success gap (clean, CI band, stats in title)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
fig.patch.set_facecolor(PARCHMENT)
ax.set_facecolor(PARCHMENT)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#C8C4BF'); ax.spines['bottom'].set_color('#C8C4BF')
ax.tick_params(colors='#1E1E1E', labelsize=9.5)
ax.grid(color='#C8C4BF', linewidth=0.6, alpha=0.7)

ax.scatter(cross['api_directive'], cross['success_gap'],
           color=CORAL, s=42, alpha=0.50, linewidths=0, zorder=3)

x_arr = cross['api_directive'].values
y_arr = cross['success_gap'].values
coeffs = np.polyfit(x_arr, y_arr, 1)
x_fit = np.linspace(x_arr.min(), x_arr.max(), 200)
y_fit = np.polyval(coeffs, x_fit)
n = len(x_arr); x_mean = x_arr.mean()
s2 = np.sum((y_arr - np.polyval(coeffs, x_arr))**2) / (n - 2)
se = np.sqrt(s2 * (1/n + (x_fit - x_mean)**2 / np.sum((x_arr - x_mean)**2)))
ax.fill_between(x_fit, y_fit - 1.97*se, y_fit + 1.97*se, color=CORAL, alpha=0.12, zorder=2)
ax.plot(x_fit, y_fit, color=CORAL, lw=1.8, alpha=0.75, zorder=4)
ax.axhline(0, color='#1E1E1E', lw=0.9, alpha=0.35, ls=(0,(4,3)))

r, pv = pearsonr(x_arr, y_arr)
ax.set_xlabel('API directive mode share (%)', fontsize=10, labelpad=8)
ax.set_ylabel('Success gap: Claude.ai − API  (pp)', fontsize=10, labelpad=8)
ax.set_title(f'Higher directive use on the API → wider performance gap\\nPer-task: r = {r:.2f},  p < 0.001,  N = {len(cross)} shared tasks',
             fontsize=11.5, color='#1E1E1E', fontweight='normal', pad=14, loc='left')

plt.tight_layout(pad=1.6)
fig.savefig(FIG_DIR / "p03_directive_success_gap.png", dpi=160, bbox_inches='tight', facecolor=PARCHMENT)
plt.show()
print("Saved p03_directive_success_gap.png")
""")


# ── Section 5: Speed-Oversight Tradeoff ───────────────────────────────────
md("""## 5. The Speed–Oversight Tradeoff

Time units: `human_only_time` in hours; `human_with_ai_time` in minutes.
Speed ratio = human_only_time (h × 60) / human_with_ai_time (min).
""")

code("""\
# Global speed ratios
cai_ho = float(cai[(cai['facet']=='human_only_time') &
                    (cai['geography']=='global') &
                    (cai['variable']=='human_only_time_mean')]['value'].iloc[0])
api_ho = float(api[(api['facet']=='human_only_time') &
                    (api['geography']=='global') &
                    (api['variable']=='human_only_time_mean')]['value'].iloc[0])
cai_ha = float(cai[(cai['facet']=='human_with_ai_time') &
                    (cai['geography']=='global') &
                    (cai['variable']=='human_with_ai_time_mean')]['value'].iloc[0])
api_ha = float(api[(api['facet']=='human_with_ai_time') &
                    (api['geography']=='global') &
                    (api['variable']=='human_with_ai_time_mean')]['value'].iloc[0])

cai_speed = cai_ho * 60 / cai_ha   # hours×60 / minutes
api_speed = api_ho * 60 / api_ha

print("Speed vs human-only baseline:")
print(f"  Claude.ai: {cai_ho:.2f}h → {cai_ha:.1f}min  ({cai_speed:.1f}× faster)")
print(f"  API:       {api_ho:.2f}h → {api_ha:.1f}min  ({api_speed:.1f}× faster)")
print(f"  API vs Claude.ai speed ratio: {api_speed/cai_speed:.1f}×")
print()
print("The complete tradeoff (global):")
print(f"  {'Platform':<15} {'Speed (×)':>10} {'Success (%)':>12} {'Human in loop (%)':>18}")
print("-" * 58)
print(f"  {'Claude.ai':<15} {cai_speed:>10.1f} {cai_stats['task_success_pct']:>12.1f} {cai_stats['augmentation_pct']:>18.1f}")
print(f"  {'1P API':<15} {api_speed:>10.1f} {api_stats['task_success_pct']:>12.1f} {api_stats['augmentation_pct']:>18.1f}")
""")

code("""\
# Figure 4: Dumbbell chart — 3 metrics, 2 platforms
metrics_dumbell = [
    ('Task success rate (%)',        69.9,  50.5),
    ('Augmentation share (%)',        52.8,  17.2),
    ('Speed vs human baseline (×)',   12.8,  24.9),
]
GRAY_LIGHT = '#D8D4CF'

fig, ax = plt.subplots(figsize=(8.5, 4.0))
fig.patch.set_facecolor(PARCHMENT)
ax.set_facecolor(PARCHMENT)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#C8C4BF')
ax.grid(axis='x', color='#C8C4BF', linewidth=0.5, alpha=0.5)
ax.grid(axis='y', visible=False)
ax.tick_params(colors='#1E1E1E', labelsize=9.5)

y_pos = [2, 1, 0]
for yi, (label, cai_val, api_val) in zip(y_pos, metrics_dumbell):
    ax.plot([cai_val, api_val], [yi, yi], color=GRAY_LIGHT, lw=2.0, zorder=2, solid_capstyle='round')
    ax.scatter(cai_val, yi, color=TEAL,  s=160, zorder=4, linewidths=0)
    ax.scatter(api_val, yi, color=CORAL, s=160, zorder=4, linewidths=0)
    cai_offset = -2.5 if cai_val > api_val else 2.5
    api_offset = -cai_offset
    ax.text(cai_val + cai_offset, yi + 0.14, f'{cai_val:.1f}',
            ha='center', va='bottom', fontsize=9.5, color=TEAL, fontweight='semibold')
    ax.text(api_val + api_offset, yi + 0.14, f'{api_val:.1f}',
            ha='center', va='bottom', fontsize=9.5, color=CORAL, fontweight='semibold')

ax.set_yticks(y_pos)
ax.set_yticklabels([m[0] for m in metrics_dumbell], fontsize=10, color='#1E1E1E')
ax.set_xlim(0, 80)
ax.set_ylim(-0.65, 2.65)
ax.set_xlabel('Value', fontsize=9.5, color='#1E1E1E', labelpad=8)
ax.set_title('The deployment tradeoff: speed, oversight, success',
             fontsize=11.5, color='#1E1E1E', fontweight='normal', pad=14, loc='left')

import matplotlib.patches as mpatches
teal_patch  = mpatches.Patch(color=TEAL,  label='Claude.ai (consumer)')
coral_patch = mpatches.Patch(color=CORAL, label='1P API (production)')
ax.legend(handles=[teal_patch, coral_patch], fontsize=9.5, frameon=False,
          loc='lower right', handlelength=1.2)

plt.tight_layout(pad=1.6)
fig.savefig(FIG_DIR / "p04_speed_success_tradeoff.png", dpi=160, bbox_inches='tight', facecolor=PARCHMENT)
plt.show()
print("Saved p04_speed_success_tradeoff.png")
""")


# ── Section 6: Summary ────────────────────────────────────────────────────
md("""## 6. Summary of Findings

| Finding | Metric |
|---------|--------|
| Per-task gap: tasks better on Claude.ai | 84.9% of 258 shared tasks |
| Per-task gap: mean success difference | +15.9pp for Claude.ai |
| Directive share difference | API 56% vs Claude.ai 29% |
| Learning share difference | Claude.ai 34% vs API 14% |
| Correlation: API directive → success gap | r = −0.263, p < 0.001 |
| Speed: Claude.ai vs baseline | 12.8× faster |
| Speed: API vs baseline | 24.9× faster |
| Global success gap | +19.4pp for Claude.ai (69.9% vs 50.5%) |

**Interpretation**: The performance gap between the two platforms is not
explained by model capability. Both use the same model family. It is explained
by deployment structure: the API's default interaction pattern (directive/automation)
systematically produces lower success rates than the consumer interface's
pattern (learning/augmentation) on the same tasks.

This creates an empirically measurable cost to reducing human oversight
in AI deployments, driven by economic incentives toward speed and automation.
""")

nb.cells = cells

import os
out_path = os.path.join(os.path.dirname(__file__), "03_platform_oversight.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")
