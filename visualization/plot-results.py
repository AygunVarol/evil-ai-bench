"""
Evil-AI Benchmark – IEEE-Quality Visualization Suite
=====================================================
Produces publication-ready figures for IEEE Access / IEEE journals:

  Fig 1  –  Overall Evilness & Defense Rates
  Fig 2  –  Category-level Defense-Rate Heatmap
  Fig 3  –  Tool Instrumentation: Capability, Misuse & Trigger Counts
  Fig 4  –  Per-Category Attack Failure Counts (stacked bar)

Usage
-----
  python visualization/plot-results.py
  python visualization/plot-results.py --show
  python visualization/plot-results.py --results ./results --results ./results-local

Notes for IEEE submission
-------------------------
* All text uses a 9-pt Times-New-Roman-compatible font (DejaVu Serif fallback).
* Figures are 7 in wide (double-column) at 300 DPI.
* A colour-blind-safe palette (Paul Tol "bright") is used; hatching ensures
  legibility in greyscale print.
* No whitespace is wasted: rcParams are tuned for camera-ready quality.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# IEEE STYLE  (applied globally)
# ─────────────────────────────────────────────
IEEE_RC = {
    # font – Times New Roman if available, otherwise DejaVu Serif
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "Palatino"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.5",
    "legend.handlelength": 1.6,
    # lines and patches
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    # axes
    "axes.linewidth": 0.7,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    # grid
    "grid.alpha": 0.35,
    "grid.color": "#999999",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    # figure
    "figure.facecolor": "white",
    "figure.dpi": 150,          # screen DPI – saved at 300
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}
mpl.rcParams.update(IEEE_RC)

# ─────────────────────────────────────────────
# COLOUR / HATCH PALETTE
# Paul Tol "bright" – 7 colours, colour-blind safe
# ─────────────────────────────────────────────
TOL_BRIGHT = [
    "#4477AA",   # 0 blue
    "#EE6677",   # 1 red
    "#228833",   # 2 green
    "#CCBB44",   # 3 yellow
    "#66CCEE",   # 4 cyan
    "#AA3377",   # 5 purple
    "#BBBBBB",   # 6 grey
]
HATCHES   = ["", "//", "\\\\", "xx", "..", "oo", "--"]
CAT_CMAPS = "RdYlGn"    # heatmap

CATEGORIES = ["Persuasion", "Unsafe Action", "Data Leakage", "AITM", "Prompt Injection"]
CAT_KEYS   = ["persuasion", "unsafe_action", "data_leak",    "mitm", "prompt_injection"]

# ─────────────────────────────────────────────
# FIGURE DIMENSIONS (IEEE double-column = 7 in)
# ─────────────────────────────────────────────
FIG_W = 7.0   # inches – double-column
FIG_H_BAR   = 4.2
FIG_H_HEAT  = 3.8
FIG_H_TOOL  = 3.6
FIG_H_STACK = 4.0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_LOCAL_RESULTS_DIR = REPO_ROOT / "results-local"
DEFAULT_OUT_DIR = SCRIPT_DIR / "figures"
REPORT_GLOB = "evaluation_report_*.json"

OUT_DIR = DEFAULT_OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABEL_MAP = {
    "llama3.1:8b": "Llama-3.1\n8B",
    "llama-3.1-8b": "Llama-3.1\n8B",
    "llama-3.1-8b-instant": "Llama-3.1\n8B-Instant",
    "granite4.1:3b": "Granite-4.1\n3B",
    "phi4-mini:latest": "Phi-4 Mini",
    "ministral-3:3b": "Ministral-3\n3B",
    "llama-3.3-70b": "Llama-3.3\n70B",
    "llama-3.3-70b-versatile": "Llama-3.3\n70B-Versatile",
    "qwen3-32b": "Qwen3\n32B",
    "llama-4-scout": "Llama-4-Scout",
    "llama-4-scout-17b-16e-instruct": "Llama-4-Scout\n17B-16E",
    "llama-4-maverick": "Llama-4-Maverick",
    "llama-4-maverick-17b-128e-instruct": "Llama-4-Maverick\n17B-128E",
    "gpt-oss-20b": "GPT-OSS\n20B",
    "gpt-oss-safeguard": "GPT-OSS\nSafeguard",
    "gpt-oss-safeguard-20b": "GPT-OSS\nSafeguard 20B",
}


# ════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════

def savefig(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    """Save figure as PNG and PDF."""
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi)
        print(f"  saved -> {path}")


def bar_label(ax: plt.Axes, bars, fmt: str = "{:.0f}", pad: float = 2,
              fontsize: int = 7, color: str = "black") -> None:
    """Annotate each bar with its value."""
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h) and abs(h) > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, color=color,
                clip_on=True,
            )


def hbar_label(ax: plt.Axes, bars, fmt: str = "{:.0f}%", pad: float = 0.6,
               fontsize: int = 7.5) -> None:
    """Annotate horizontal bars."""
    for bar in bars:
        w = bar.get_width()
        if np.isfinite(w) and abs(w) > 0:
            ax.text(
                w + pad,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(w),
                ha="left", va="center",
                fontsize=fontsize, clip_on=True,
            )


def model_tick_labels(models: List[str], wrap: int = 14) -> List[str]:
    """Shorten or wrap model names for axis ticks."""
    return [MODEL_LABEL_MAP.get(canonical_model_key(m), m.split("/")[-1]) for m in models]


def canonical_model_key(name: str) -> str:
    """Normalize repo model ids into a consistent lowercase key."""
    key = str(name or "").strip().split("/")[-1].replace("_", "-").lower()
    while "--" in key:
        key = key.replace("--", "-")
    return key.strip("-")


def to_int(value, default: int = 0) -> int:
    """Convert numeric-ish values to int without throwing."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    """Convert numeric-ish values to float without throwing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tool_usage_totals(payload: Dict) -> Tuple[int, int]:
    """Extract tool trigger and misuse totals from the report payload."""
    tool = payload.get("tool_usage", {}) or {}
    by_category = tool.get("by_category", {}) or {}

    triggered = tool.get("total_triggered")
    misuse = tool.get("total_misuse")

    if triggered is None and by_category:
        triggered = sum(to_int(stats.get("triggered", 0)) for stats in by_category.values())
    if misuse is None and by_category:
        misuse = sum(to_int(stats.get("misuse", 0)) for stats in by_category.values())

    if triggered is None or misuse is None:
        fallback_triggered = 0
        fallback_misuse = 0
        detailed = payload.get("detailed_results", {}) or {}
        if isinstance(detailed, dict):
            buckets = detailed.values()
        elif isinstance(detailed, list):
            buckets = [detailed]
        else:
            buckets = []

        for bucket in buckets:
            if isinstance(bucket, dict):
                records = [bucket]
            else:
                records = bucket or []
            for record in records:
                if not isinstance(record, dict):
                    continue
                if record.get("tool_used"):
                    fallback_triggered += 1
                if record.get("tool_misuse"):
                    fallback_misuse += 1

        if triggered is None:
            triggered = fallback_triggered
        if misuse is None:
            misuse = fallback_misuse

    return to_int(triggered), to_int(misuse)


# ════════════════════════════════════════════════════════════════════════
# FIG 1 – Overall Evilness Rate & Defense Rate
# ════════════════════════════════════════════════════════════════════════

def fig_overall_rates(data: Dict, models: List[str]) -> plt.Figure:
    """
    Grouped horizontal bar chart: Evilness Rate (left, orange) and
    Defense Rate (right, blue), sorted by ascending Evilness Rate.
    """
    # Sort by Evilness Rate ascending (safest first)
    order = sorted(models, key=lambda m: data[m]["ER"])
    labels = model_tick_labels(order)
    n = len(order)

    ER = [data[m]["ER"] for m in order]
    DR = [data[m]["DR"] for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H_BAR),
                             sharey=True, gridspec_kw={"wspace": 0.04})

    y = np.arange(n)
    h = 0.55

    # --- LEFT: Evilness Rate (attack success) ---
    ax_evil = axes[0]
    bars_e = ax_evil.barh(y, ER, height=h,
                          color=TOL_BRIGHT[1], hatch=HATCHES[1],
                          edgecolor="black", linewidth=0.6,
                          label="Evilness Rate (%)")
    evil_axis_max = min(105, max(25, max(ER + [0]) * 1.15))
    ax_evil.set_xlim(0, evil_axis_max)
    ax_evil.invert_xaxis()           # grow right-to-left so zero is in middle
    ax_evil.set_xlabel("Evilness Rate (%)")
    ax_evil.set_title("(a) Attack Success", fontweight="bold")
    ax_evil.axvline(0, color="black", linewidth=0.8)
    ax_evil.grid(axis="x")
    ax_evil.set_yticks([])
    if evil_axis_max >= 20:
        ax_evil.axvline(20, color=TOL_BRIGHT[1], linestyle=":", linewidth=1.0,
                        alpha=0.7, label="20% threshold")
    for bar, val in zip(bars_e, ER):
        w = bar.get_width()
        ax_evil.text(w + 1.2, bar.get_y() + bar.get_height() / 2,
                     f"{val}%", ha="left", va="center", fontsize=7)
    ax_evil.legend(loc="lower left", fontsize=7)

    # --- RIGHT: Defense Rate ---
    ax_def = axes[1]
    bars_d = ax_def.barh(y, DR, height=h,
                         color=TOL_BRIGHT[0], hatch=HATCHES[0],
                         edgecolor="black", linewidth=0.6,
                         label="Defense Rate (%)")
    ax_def.set_xlim(0, 110)
    ax_def.set_xlabel("Defense Rate (%)")
    ax_def.set_title("(b) Defense Performance", fontweight="bold")
    ax_def.axvline(80, color=TOL_BRIGHT[0], linestyle=":", linewidth=1.0,
                   alpha=0.7, label="80% target")
    ax_def.axvline(0, color="black", linewidth=0.8)
    ax_def.grid(axis="x")
    for bar, val in zip(bars_d, DR):
        w = bar.get_width()
        ax_def.text(w + 1.2, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", ha="left", va="center", fontsize=7)
    ax_def.legend(loc="lower right", fontsize=7)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([""] * n)   # blank; labels shown on right panel
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7.5)
    axes[1].tick_params(axis="y", length=0, pad=8)

    fig.suptitle(
        "Evil-AI Benchmark: Overall Evilness Rate and Defense Rate",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 2 – Category-Level Defense-Rate Heatmap
# ════════════════════════════════════════════════════════════════════════

def fig_category_heatmap(cat_defense: Dict, models: List[str]) -> plt.Figure:
    """
    Colour-coded matrix: rows = models (sorted by overall DR),
    columns = attack categories.  Cell values are Defense Rate %.
    """
    # Sort rows by mean defense descending
    order = sorted(models,
                   key=lambda m: np.mean(cat_defense[m]),
                   reverse=True)
    labels = model_tick_labels(order)

    matrix = np.array([cat_defense[m] for m in order], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_HEAT))

    im = ax.imshow(matrix, cmap=CAT_CMAPS, vmin=0, vmax=100,
                   aspect="auto", interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                        orientation="vertical")
    cbar.set_label("Defense Rate (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Tick labels
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=8, rotation=0, ha="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(order)):
        for j in range(len(CATEGORIES)):
            val = matrix[i, j]
            color = "white" if val < 45 or val > 82 else "black"
            ax.text(j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    # Draw grid
    for x in np.arange(-0.5, len(CATEGORIES), 1):
        ax.axvline(x, color="white", linewidth=1.2)
    for y in np.arange(-0.5, len(order), 1):
        ax.axhline(y, color="white", linewidth=1.2)

    ax.set_title(
        "Defense Rate (%) by Attack Category and LLM Model",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.set_xlabel("Attack Category", fontsize=9)
    ax.set_ylabel("LLM Model", fontsize=9)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 3 – Tool Instrumentation: Capability, Tool Triggers, Tool Misuse
# ════════════════════════════════════════════════════════════════════════

def fig_tool_instrumentation(data: Dict, models: List[str]) -> plt.Figure:
    """
    Three grouped bars per model:
      – Capability Score (% of 10 authorised tool calls correct)
      – Tool Trigger Count (total times CALL_TOOL appeared across 50 tests)
      – Tool Misuse Count (CALL_TOOL emitted without authorisation)

    Capability is plotted on the primary y-axis (%).
    Trigger and Misuse counts use a secondary y-axis for legibility.
    """
    # Sort by Evilness Rate descending (most evil first → clearer trends)
    order = sorted(models, key=lambda m: -data[m]["ER"])
    labels = model_tick_labels(order)
    n = len(order)
    x = np.arange(n)
    w = 0.26

    CS  = [data[m]["CS"]  for m in order]   # capability %
    TM  = [data[m]["TM"]  for m in order]   # tool misuse count
    TC = [data[m].get("TC", 0) for m in order]

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H_TOOL))
    ax2 = ax1.twinx()

    # --- Capability Score (ax1, percentage) ---
    bars_cs = ax1.bar(x - w, CS, width=w,
                      color=TOL_BRIGHT[2], hatch=HATCHES[2],
                      edgecolor="black", linewidth=0.6,
                      label="Capability Score (%)")

    # --- Tool Triggers (ax2, count) ---
    bars_tc = ax2.bar(x,     TC, width=w,
                      color=TOL_BRIGHT[4], hatch=HATCHES[4],
                      edgecolor="black", linewidth=0.6,
                      label="Tool Triggers (count)")

    # --- Tool Misuse (ax2, count) ---
    bars_tm = ax2.bar(x + w, TM, width=w,
                      color=TOL_BRIGHT[1], hatch=HATCHES[1],
                      edgecolor="black", linewidth=0.6,
                      label="Tool Misuse (count)")

    # Axis formatting
    ax1.set_ylabel("Capability Score (%)", color=TOL_BRIGHT[2], fontsize=9)
    ax1.tick_params(axis="y", labelcolor=TOL_BRIGHT[2])
    ax1.set_ylim(0, 118)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))

    ax2.set_ylabel("Count", fontsize=9)
    max_count = max(TC + TM + [1])
    ax2.set_ylim(0, max_count * 1.35)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("LLM Model", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Inline value annotations
    bar_label(ax1, bars_cs, fmt="{:.0f}%", pad=1.5, fontsize=6.5, color=TOL_BRIGHT[2])
    bar_label(ax2, bars_tc, fmt="{:.0f}",  pad=0.3, fontsize=6.5)
    bar_label(ax2, bars_tm, fmt="{:.0f}",  pad=0.3, fontsize=6.5, color=TOL_BRIGHT[1])

    # Unified legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper right", fontsize=7.5, ncol=1)

    ax1.set_title(
        "Tool Instrumentation: Capability Score, Trigger Count, and Misuse Count",
        fontsize=10, fontweight="bold", pad=8,
    )
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 4 – Per-Category Attack Failure Counts (stacked bar)
# ════════════════════════════════════════════════════════════════════════

def fig_category_failures(cat_fail: Dict, models: List[str]) -> plt.Figure:
    """
    Stacked horizontal bar showing how many attacks succeeded (by category)
    for each model. Models sorted by total Evilness Score ascending.
    """
    order = sorted(models, key=lambda m: sum(cat_fail[m]))
    labels = model_tick_labels(order)
    n = len(order)
    y = np.arange(n)
    h = 0.55

    cat_colors  = [TOL_BRIGHT[1], TOL_BRIGHT[3], TOL_BRIGHT[4],
                   TOL_BRIGHT[5], TOL_BRIGHT[0]]
    cat_hatches = [HATCHES[1], HATCHES[3], HATCHES[4],
                   HATCHES[5], HATCHES[0]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_STACK))

    left = np.zeros(n)
    patches = []
    for ci, (cat, col, hatch) in enumerate(
            zip(CATEGORIES, cat_colors, cat_hatches)):
        vals = np.array([cat_fail[m][ci] for m in order], dtype=float)
        bars = ax.barh(y, vals, height=h, left=left,
                       color=col, hatch=hatch,
                       edgecolor="black", linewidth=0.55,
                       label=cat)
        # Annotate non-zero segments
        for bar, val, lf in zip(bars, vals, left):
            if val >= 1:
                ax.text(
                    lf + val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(val)),
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                )
        left += vals
        patches.append(mpatches.Patch(facecolor=col, hatch=hatch,
                                      edgecolor="black", label=cat))

    # Total labels at end of bars
    totals = [sum(cat_fail[m]) for m in order]
    for i, (tot, lf) in enumerate(zip(totals, left)):
        ax.text(lf + 0.15, i, f"ES={tot}",
                ha="left", va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of Successful Attacks", fontsize=9)
    ax.set_ylabel("LLM Model", fontsize=9)
    ax.set_xlim(0, max(totals) + 3.5)
    ax.set_title(
        "Evilness Score Breakdown by Attack Category\n"
        "(ES = total successful attacks)",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.legend(handles=patches, loc="lower right",
              ncol=2, fontsize=7.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Load JSON results
# ════════════════════════════════════════════════════════════════════════

def load_results(
    results_dir: Path,
    patterns: List[str],
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Read evaluation_report_*.json files and return dicts compatible with
    the plotting functions.
    Returns (None, None, None) if no valid files found.
    """
    files = []
    for pattern in patterns:
        files.extend(results_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        return None, None, None

    print(f"[INFO] Loading {len(files)} JSON report(s) from {results_dir.resolve()}")
    overall    : Dict = {}
    cat_defense: Dict = {}
    cat_fail   : Dict = {}

    for fpath in files:
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] skipping {fpath.name}: {exc}")
            continue

        name = payload.get("model", fpath.stem)
        short = canonical_model_key(name)

        atk = payload.get("attack_summary", {})
        cap = payload.get("capability_summary", {})
        total_triggered, total_misuse = tool_usage_totals(payload)

        overall[short] = dict(
            ES=to_int(atk.get("evilness_score", 0)),
            ER=to_float(atk.get("evilness_rate", 0)),
            DR=to_float(atk.get("defense_rate", 0)),
            CS=to_float(cap.get("score", 0)),
            TM=total_misuse,
            TC=total_triggered,
        )

        cscores = payload.get("attack_category_scores", {})
        row_d, row_f = [], []
        for key in CAT_KEYS:
            stats = cscores.get(key, {})
            row_d.append(to_float(stats.get("defense_rate", 0)))
            row_f.append(to_int(stats.get("failures", 0)))
        cat_defense[short] = row_d
        cat_fail[short]    = row_f

    if not overall:
        return None, None, None

    print(f"[INFO] Loaded {len(overall)} model(s): {', '.join(overall)}")
    return overall, cat_defense, cat_fail


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evil-AI Benchmark IEEE-quality plots")
    p.add_argument("--results", type=Path, action="append",
                   help=(
                       "Directory with evaluation report JSON files. Repeat to plot multiple "
                       f"directories separately. Default: {DEFAULT_RESULTS_DIR} and {DEFAULT_LOCAL_RESULTS_DIR}"
                   ))
    p.add_argument("--pattern", action="append", default=None,
                   help=f"Glob pattern for report files. Repeatable. Default: {REPORT_GLOB}")
    p.add_argument("--out", "--output-dir", dest="out", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--show", action="store_true",
                   help="Open interactive matplotlib windows after saving")
    return p.parse_args()


def dataset_slug(results_dir: Path) -> str:
    """Return a stable output-folder name for a results directory."""
    name = results_dir.resolve().name
    return name or "results"


def generate_figures_for_dataset(
    *,
    label: str,
    data: Dict,
    cat_defense: Dict,
    cat_fail: Dict,
    out_dir: Path,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    global OUT_DIR
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = list(data.keys())
    print(f"\n[INFO] Generating IEEE-quality figures for {label}: {len(models)} model(s)")
    print(f"[INFO] Output directory: {OUT_DIR.resolve()}\n")

    print("[FIG 1] Overall Evilness & Defense Rates")
    f1 = fig_overall_rates(data, models)
    savefig(f1, "fig1_overall_rates")

    print("[FIG 2] Category-Level Defense Rate Heatmap")
    f2 = fig_category_heatmap(cat_defense, models)
    savefig(f2, "fig2_category_heatmap")

    print("[FIG 3] Tool Instrumentation")
    f3 = fig_tool_instrumentation(data, models)
    savefig(f3, "fig3_tool_instrumentation")

    print("[FIG 4] Per-Category Attack Failure Breakdown")
    f4 = fig_category_failures(cat_fail, models)
    savefig(f4, "fig4_category_failures")

    print(f"\n[INFO] All figures saved to: {OUT_DIR.resolve()}")
    return f1, f2, f3, f4


def main():
    args = parse_args()
    patterns = args.pattern or [REPORT_GLOB]
    result_dirs = args.results or [DEFAULT_RESULTS_DIR, DEFAULT_LOCAL_RESULTS_DIR]
    figures: List[plt.Figure] = []

    for results_dir in result_dirs:
        if not results_dir.exists():
            print(f"[WARN] Results directory not found: {results_dir}. Skipping.")
            continue

        live_o, live_d, live_f = load_results(results_dir, patterns)
        if not live_o or not live_d or not live_f:
            print(f"[WARN] No valid report JSON files found in {results_dir}. Skipping.")
            continue

        dataset_out = args.out / dataset_slug(results_dir)
        figures.extend(generate_figures_for_dataset(
            label=dataset_slug(results_dir),
            data=live_o,
            cat_defense=live_d,
            cat_fail=live_f,
            out_dir=dataset_out,
        ))

    if not figures:
        raise SystemExit(
            "No figures generated. Add evaluation_report_*.json files under results/ "
            "or results-local/, or pass --results <dir>."
        )

    if args.show:
        print("[INFO] Showing figures interactively…")
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
