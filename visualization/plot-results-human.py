"""
Evil-AI Benchmark - IEEE-Quality Visualization Suite (HUMAN-ANNOTATED results)
=============================================================================
Companion to plot-results.py, but it plots the HUMAN-VERIFIED safety outcomes
(from `human_evaluation_summary.by_evaluator.<id>.human_attack_summary`) rather
than the raw automated verdicts. Tool-instrumentation quantities (capability,
tool triggers, tool misuse) are objective and are read from the automated
telemetry unchanged.

Differences from plot-results.py (per revision request):
  * Reads human-verified ES / ER / DR and per-category outcomes.
  * ALL in-figure titles/subtitles removed (moved into the LaTeX captions).
  * Fig 1: Evilness-Rate value labels pushed further left so they no longer
    merge with the bars (mirrors the Defense-Rate panel).
  * Fig 3: right-hand "Count" axis uses coarser ticks (0, 50, 100, ...).
  * Fig 4: wider x-limit so the largest ES label (e.g. ES=135) fits.

Usage
-----
  python visualization/plot-results-human.py
  python visualization/plot-results-human.py --results ./results-human --evaluator Aygun
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
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "grid.alpha": 0.35,
    "grid.color": "#999999",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}
mpl.rcParams.update(IEEE_RC)

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
CAT_CMAPS = "RdYlGn"

CATEGORIES = ["Persuasion", "Unsafe Action", "Data Leakage", "AITM", "Prompt Injection"]
CAT_KEYS   = ["persuasion", "unsafe_action", "data_leak",    "mitm", "prompt_injection"]

FIG_W = 7.0
FIG_H_BAR   = 4.2
FIG_H_HEAT  = 3.8
FIG_H_TOOL  = 3.6
FIG_H_STACK = 4.0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results-human"
DEFAULT_OUT_DIR = SCRIPT_DIR / "figures"
REPORT_GLOB = "evaluation_report_*.json"

OUT_DIR = DEFAULT_OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluator id whose human labels are plotted (overridable via --evaluator)
EVALUATOR_ID = "Aygun"

MODEL_LABEL_MAP = {
    "llama-3.1-8b-instant": "Llama-3.1\n8B-Instant",
    "llama-3.3-70b-versatile": "Llama-3.3\n70B-Versatile",
    "qwen3-32b": "Qwen3\n32B",
    "qwen3.6-27b": "Qwen3.6\n27B",
    "llama-4-scout-17b-16e-instruct": "Llama-4-Scout\n17B-16E",
    "gpt-oss-20b": "GPT-OSS\n20B",
    "gpt-oss-120b": "GPT-OSS\n120B",
    "gpt-oss-safeguard-20b": "GPT-OSS\nSafeguard 20B",
}


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def savefig(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi)
        print(f"  saved -> {path}")


def bar_label(ax, bars, fmt="{:.0f}", pad=2, fontsize=7, color="black"):
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h) and abs(h) > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + pad, fmt.format(h),
                    ha="center", va="bottom", fontsize=fontsize, color=color, clip_on=True)


def canonical_model_key(name: str) -> str:
    key = str(name or "").strip().split("/")[-1].replace("_", "-").lower()
    while "--" in key:
        key = key.replace("--", "-")
    return key.strip("-")


def model_tick_labels(models: List[str]) -> List[str]:
    return [MODEL_LABEL_MAP.get(canonical_model_key(m), m.split("/")[-1]) for m in models]


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tool_usage_totals(payload: Dict) -> Tuple[int, int]:
    tool = payload.get("tool_usage", {}) or {}
    by_category = tool.get("by_category", {}) or {}
    triggered = tool.get("total_triggered")
    misuse = tool.get("total_misuse")
    if triggered is None and by_category:
        triggered = sum(to_int(s.get("triggered", 0)) for s in by_category.values())
    if misuse is None and by_category:
        misuse = sum(to_int(s.get("misuse", 0)) for s in by_category.values())
    return to_int(triggered), to_int(misuse)


def human_attack_summary(payload: Dict, evaluator: str) -> Optional[Dict]:
    """Return the human_attack_summary for the requested evaluator (or first)."""
    hes = payload.get("human_evaluation_summary", {}) or {}
    by_eval = hes.get("by_evaluator", {}) or {}
    if not by_eval:
        return None
    ev = by_eval.get(evaluator) or next(iter(by_eval.values()))
    return ev.get("human_attack_summary")


# ════════════════════════════════════════════════════════════════════════
# FIG 1 – Overall Evilness Rate & Defense Rate  (no titles)
# ════════════════════════════════════════════════════════════════════════

def fig_overall_rates(data: Dict, models: List[str]) -> plt.Figure:
    order = sorted(models, key=lambda m: data[m]["ER"])
    labels = model_tick_labels(order)
    n = len(order)
    ER = [data[m]["ER"] for m in order]
    DR = [data[m]["DR"] for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H_BAR),
                             sharey=True, gridspec_kw={"wspace": 0.04})
    y = np.arange(n)
    h = 0.55

    # --- LEFT: Evilness Rate ---
    ax_evil = axes[0]
    bars_e = ax_evil.barh(y, ER, height=h, color=TOL_BRIGHT[1], hatch=HATCHES[1],
                          edgecolor="black", linewidth=0.6, label="Evilness Rate (%)")
    evil_axis_max = min(105, max(25, max(ER + [0]) * 1.28))  # extra headroom for left labels
    ax_evil.set_xlim(0, evil_axis_max)
    ax_evil.invert_xaxis()
    ax_evil.set_xlabel("Evilness Rate (%)")
    ax_evil.axvline(0, color="black", linewidth=0.8)
    ax_evil.grid(axis="x")
    ax_evil.set_yticks([])
    if evil_axis_max >= 20:
        ax_evil.axvline(20, color=TOL_BRIGHT[1], linestyle=":", linewidth=1.0,
                        alpha=0.7, label="20% threshold")
    # Labels sit LEFT of the bar tip (ha="right"), so they never overlap the
    # bars -- mirroring the Defense-Rate panel spacing.
    lab_pad = max(1.8, evil_axis_max * 0.03)
    for bar, val in zip(bars_e, ER):
        w = bar.get_width()
        ax_evil.text(w + lab_pad, bar.get_y() + bar.get_height() / 2,
                     f"{val}%", ha="right", va="center", fontsize=7)
    ax_evil.legend(loc="lower left", fontsize=7)

    # --- RIGHT: Defense Rate ---
    ax_def = axes[1]
    bars_d = ax_def.barh(y, DR, height=h, color=TOL_BRIGHT[0], hatch=HATCHES[0],
                         edgecolor="black", linewidth=0.6, label="Defense Rate (%)")
    ax_def.set_xlim(0, 126)
    ax_def.set_xlabel("Defense Rate (%)")
    ax_def.axvline(80, color=TOL_BRIGHT[0], linestyle=":", linewidth=1.0,
                   alpha=0.7, label="80% target")
    ax_def.axvline(0, color="black", linewidth=0.8)
    ax_def.grid(axis="x")
    for bar, val in zip(bars_d, DR):
        w = bar.get_width()
        ax_def.text(w + 1.2, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", ha="left", va="center", fontsize=7)
    # upper-right corner is empty (top model has the lowest defense), so the
    # legend there does not overlap any value label.
    ax_def.legend(loc="upper right", fontsize=7)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([""] * n)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7.5)
    axes[1].tick_params(axis="y", length=0, pad=8)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 2 – Category-Level Defense-Rate Heatmap  (no title)
# ════════════════════════════════════════════════════════════════════════

def fig_category_heatmap(cat_defense: Dict, models: List[str]) -> plt.Figure:
    order = sorted(models, key=lambda m: np.mean(cat_defense[m]), reverse=True)
    labels = model_tick_labels(order)
    matrix = np.array([cat_defense[m] for m in order], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_HEAT))
    im = ax.imshow(matrix, cmap=CAT_CMAPS, vmin=0, vmax=100,
                   aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, orientation="vertical")
    cbar.set_label("Defense Rate (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=8, rotation=0, ha="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(order)):
        for j in range(len(CATEGORIES)):
            val = matrix[i, j]
            color = "white" if val < 45 or val > 82 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    for x in np.arange(-0.5, len(CATEGORIES), 1):
        ax.axvline(x, color="white", linewidth=1.2)
    for yy in np.arange(-0.5, len(order), 1):
        ax.axhline(yy, color="white", linewidth=1.2)

    ax.set_xlabel("Attack Category", fontsize=9)
    ax.set_ylabel("LLM Model", fontsize=9)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 3 – Tool Instrumentation  (no title; coarse Count ticks)
# ════════════════════════════════════════════════════════════════════════

def fig_tool_instrumentation(data: Dict, models: List[str]) -> plt.Figure:
    order = sorted(models, key=lambda m: -data[m]["ER"])
    labels = model_tick_labels(order)
    n = len(order)
    x = np.arange(n)
    w = 0.26

    CS = [data[m]["CS"] for m in order]
    TM = [data[m]["TM"] for m in order]
    TC = [data[m].get("TC", 0) for m in order]

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H_TOOL))
    ax2 = ax1.twinx()

    bars_cs = ax1.bar(x - w, CS, width=w, color=TOL_BRIGHT[2], hatch=HATCHES[2],
                      edgecolor="black", linewidth=0.6, label="Capability Score (%)")
    bars_tc = ax2.bar(x,     TC, width=w, color=TOL_BRIGHT[4], hatch=HATCHES[4],
                      edgecolor="black", linewidth=0.6, label="Tool Triggers (count)")
    bars_tm = ax2.bar(x + w, TM, width=w, color=TOL_BRIGHT[1], hatch=HATCHES[1],
                      edgecolor="black", linewidth=0.6, label="Tool Misuse (count)")

    ax1.set_ylabel("Capability Score (%)", color=TOL_BRIGHT[2], fontsize=9)
    ax1.tick_params(axis="y", labelcolor=TOL_BRIGHT[2])
    ax1.set_ylim(0, 118)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))

    ax2.set_ylabel("Count", fontsize=9)
    max_count = max(TC + TM + [1])
    # round the top up to the next multiple of 50 and add one tick of headroom
    top = (int(max_count / 50) + 2) * 50
    ax2.set_ylim(0, top)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(50))  # 0, 50, 100, 150, ...

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.6)  # smaller so long names do not collide
    ax1.set_xlabel("LLM Model", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    bar_label(ax1, bars_cs, fmt="{:.0f}%", pad=1.5, fontsize=6.5, color=TOL_BRIGHT[2])
    bar_label(ax2, bars_tc, fmt="{:.0f}",  pad=top * 0.01, fontsize=6.5)
    bar_label(ax2, bars_tm, fmt="{:.0f}",  pad=top * 0.01, fontsize=6.5, color=TOL_BRIGHT[1])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5, ncol=1)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# FIG 4 – Per-Category Attack Failure Counts  (no title; wider x-limit)
# ════════════════════════════════════════════════════════════════════════

def fig_category_failures(cat_fail: Dict, models: List[str]) -> plt.Figure:
    order = sorted(models, key=lambda m: sum(cat_fail[m]))
    labels = model_tick_labels(order)
    n = len(order)
    y = np.arange(n)
    h = 0.55

    cat_colors  = [TOL_BRIGHT[1], TOL_BRIGHT[3], TOL_BRIGHT[4], TOL_BRIGHT[5], TOL_BRIGHT[0]]
    cat_hatches = [HATCHES[1], HATCHES[3], HATCHES[4], HATCHES[5], HATCHES[0]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_STACK))
    left = np.zeros(n)
    patches = []
    for ci, (cat, col, hatch) in enumerate(zip(CATEGORIES, cat_colors, cat_hatches)):
        vals = np.array([cat_fail[m][ci] for m in order], dtype=float)
        bars = ax.barh(y, vals, height=h, left=left, color=col, hatch=hatch,
                       edgecolor="black", linewidth=0.55, label=cat)
        for bar, val, lf in zip(bars, vals, left):
            if val >= 1:
                ax.text(lf + val / 2, bar.get_y() + bar.get_height() / 2,
                        str(int(val)), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        left += vals
        patches.append(mpatches.Patch(facecolor=col, hatch=hatch, edgecolor="black", label=cat))

    totals = [sum(cat_fail[m]) for m in order]
    for i, (tot, lf) in enumerate(zip(totals, left)):
        ax.text(lf + max(totals) * 0.012 + 0.4, i, f"ES={tot}",
                ha="left", va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of Successful Attacks", fontsize=9)
    ax.set_ylabel("LLM Model", fontsize=9)
    # Wider x-limit so the largest "ES=NNN" label is not clipped.
    ax.set_xlim(0, max(totals) * 1.16 + 8)
    ax.legend(handles=patches, loc="lower right", ncol=2, fontsize=7.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Load HUMAN-annotated results
# ════════════════════════════════════════════════════════════════════════

def load_results(results_dir: Path, patterns: List[str], evaluator: str):
    files = []
    for pattern in patterns:
        files.extend(results_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        return None, None, None

    print(f"[INFO] Loading {len(files)} human report(s) from {results_dir.resolve()}")
    overall: Dict = {}
    cat_defense: Dict = {}
    cat_fail: Dict = {}

    for fpath in files:
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] skipping {fpath.name}: {exc}")
            continue

        name = payload.get("model", fpath.stem)
        short = canonical_model_key(name)

        hsum = human_attack_summary(payload, evaluator)
        if hsum is None:
            print(f"  [WARN] {fpath.name}: no human_attack_summary; skipping")
            continue

        cap = payload.get("capability_summary", {})       # objective
        total_triggered, total_misuse = tool_usage_totals(payload)  # objective

        overall[short] = dict(
            ES=to_int(hsum.get("evilness_score", 0)),
            ER=to_float(hsum.get("evilness_rate", 0)),
            DR=to_float(hsum.get("defense_rate", 0)),
            CS=to_float(cap.get("score", 0)),
            TM=total_misuse,
            TC=total_triggered,
        )

        by_cat = hsum.get("by_category", {}) or {}
        row_d, row_f = [], []
        for key in CAT_KEYS:
            stats = by_cat.get(key, {})
            row_d.append(to_float(stats.get("defense_rate", 0)))
            row_f.append(to_int(stats.get("failures", 0)))
        cat_defense[short] = row_d
        cat_fail[short] = row_f

    if not overall:
        return None, None, None

    print(f"[INFO] Loaded {len(overall)} model(s): {', '.join(overall)}")
    return overall, cat_defense, cat_fail


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evil-AI Benchmark IEEE plots (human-annotated)")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"Directory with human evaluation reports (default: {DEFAULT_RESULTS_DIR})")
    p.add_argument("--pattern", action="append", default=None,
                   help=f"Glob for report files (default: {REPORT_GLOB})")
    p.add_argument("--evaluator", default=EVALUATOR_ID,
                   help=f"Human evaluator id to plot (default: {EVALUATOR_ID})")
    p.add_argument("--out", dest="out", type=Path, default=DEFAULT_OUT_DIR / "results-human",
                   help="Output directory")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    global OUT_DIR
    args = parse_args()
    patterns = args.pattern or [REPORT_GLOB]

    if not args.results.exists():
        raise SystemExit(f"Results directory not found: {args.results}")

    data, cat_defense, cat_fail = load_results(args.results, patterns, args.evaluator)
    if not data:
        raise SystemExit("No valid human report JSON files found.")

    OUT_DIR = args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = list(data.keys())
    print(f"\n[INFO] Generating title-less human-annotated figures: {len(models)} model(s)")
    print(f"[INFO] Output directory: {OUT_DIR.resolve()}\n")

    print("[FIG 1] Overall Evilness & Defense Rates")
    savefig(fig_overall_rates(data, models), "fig1_overall_rates")
    print("[FIG 2] Category-Level Defense Rate Heatmap")
    savefig(fig_category_heatmap(cat_defense, models), "fig2_category_heatmap")
    print("[FIG 3] Tool Instrumentation")
    savefig(fig_tool_instrumentation(data, models), "fig3_tool_instrumentation")
    print("[FIG 4] Per-Category Attack Failure Breakdown")
    savefig(fig_category_failures(cat_fail, models), "fig4_category_failures")
    print(f"\n[INFO] All human-annotated figures saved to: {OUT_DIR.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
