"""
Plot styling for consistent, publication-quality figures.

Loosely inspired by Edward Tufte's principles: high data-ink ratio,
no chartjunk, muted palette, serif type.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Palette ────────────────────────────────────────────────────────────────
NAVY       = "#2C3E6B"
CLAY       = "#C0504D"
SAND       = "#D4A574"
SAGE       = "#6B8E6B"
SLATE      = "#5B6770"
PARCHMENT  = "#F5F5F0"

PALETTE = [NAVY, CLAY, SAND, SAGE, SLATE]

# ── Style application ──────────────────────────────────────────────────────

def apply() -> None:
    """Apply the project style globally."""
    mpl.rcParams.update({
        # Canvas
        "figure.facecolor":    PARCHMENT,
        "axes.facecolor":      PARCHMENT,
        "savefig.facecolor":   PARCHMENT,
        "figure.dpi":          150,
        "savefig.dpi":         150,
        "savefig.bbox":        "tight",

        # Typography
        "font.family":         "serif",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.labelsize":      11,

        # Spines
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.6,

        # Grid
        "axes.grid":           False,

        # Legend
        "legend.frameon":      False,
        "legend.fontsize":     10,

        # Ticks
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
    })
