"""
Generates evaluation plots for the matched-task pairwise
evaluation (SAQ + SBQ, Raw + Production).

Outputs (saved to <workspace_root>/plots/):
  main_figure.pdf/png       — Main paper (1 figure, 4 subplots):
                              2x2 grouped win-rate bar chart
                              Rows = Raw | Production, Columns = SAQ | SBQ
                              "Ours" bars highlighted with bold outline.

  appendix_stacked.pdf/png  — Appendix: 2x2 stacked win / tie / loss bar chart
                              (rates averaged across models, one bar per pipeline)

  appendix_delta.pdf/png    — Appendix: Δ win-rate (production - raw) line chart
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, Any

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE  = _SCRIPT_DIR.parent.parent
RESULTS_DIR = _WORKSPACE / "evaluation_results"
PLOTS_DIR   = _WORKSPACE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "figure.dpi":         300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Label mappings ─────────────────────────────────────────────────────────────
PIPELINE_ORDER = [
    "zero_shot",
    "zero_shot_bloom",
    "multi_stage_zero_shot",
    "multi_stage_zero_shot_bloom",
    "proposed_pipeline",
]

PIPELINE_LABELS = {
    "zero_shot":                   "ZS",
    "zero_shot_bloom":             "ZS+B",
    "multi_stage_zero_shot":       "MZS",
    "multi_stage_zero_shot_bloom": "MZS+B",
    "proposed_pipeline":           "Ours",
}

MODEL_ORDER = ["gemini-2.5-flash", "gpt-4.1", "gpt-4o-mini"]

MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini",
    "gpt-4.1":          "GPT-4.1",
    "gpt-4o-mini":      "GPT-4o-mini",
}

# Short names for compact annotations
MODEL_SHORT = {
    "gemini-2.5-flash": "Gemini",
    "gpt-4.1":          "GPT-4.1",
    "gpt-4o-mini":      "GPT-4o",
}

# Colour-blind friendly palette (Wong 2011)
MODEL_COLORS = {
    "gemini-2.5-flash": "#0072B2",
    "gpt-4.1":          "#E69F00",
    "gpt-4o-mini":      "#009E73",
}

WIN_COLOR  = "#4CAF50"
TIE_COLOR  = "#BDBDBD"
LOSS_COLOR = "#EF5350"

_BAR_W     = 0.21
_GROUP_GAP = 0.28


def _group_layout(n_pipelines: int, n_models: int):
    """Return (group_centers, model_offsets) for grouped bar charts."""
    group_width   = n_models * _BAR_W + _GROUP_GAP
    group_centers = np.arange(n_pipelines) * group_width
    model_offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * _BAR_W
    return group_centers, model_offsets


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_results(qt: str) -> Dict[str, Any]:
    path = RESULTS_DIR / f"matched_task_results_{qt}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rates(
        results: Dict[str, Any],
        mode: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return pipeline → model → {win_rate, tie_rate, loss_rate, bootstrap_95ci}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pipeline, stats in results[mode]["summary"].items():
        out[pipeline] = {}
        for model, ms in stats.get("per_model", {}).items():
            out[pipeline][model] = {
                "win_rate":       ms.get("win_rate",  0.0),
                "tie_rate":       ms.get("tie_rate",  0.0),
                "loss_rate":      ms.get("loss_rate", 0.0),
                "bootstrap_95ci": ms.get("bootstrap_95ci", [None, None]),
            }
    return out


def extract_global_rates(
        results: Dict[str, Any],
        mode: str,
) -> Dict[str, Dict[str, float]]:
    """Return pipeline → {win_rate, tie_rate, loss_rate} from the global summary.

    Because the new matched-task evaluation builds a single global set T where
    every pipeline has the same denominator, the top-level summary rates are the
    correct aggregated rates and should be used in preference to averaging
    per-model rates (which can differ when model groups have unequal sizes).
    """
    out: Dict[str, Dict[str, float]] = {}
    for pipeline, stats in results[mode]["summary"].items():
        out[pipeline] = {
            "win_rate":  stats.get("win_rate",  0.0),
            "tie_rate":  stats.get("tie_rate",  0.0),
            "loss_rate": stats.get("loss_rate", 0.0),
        }
    return out


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main paper: 2x2 win-rate grouped bar chart
#   axes[0,0] = (a) SAQ (Raw)        |  axes[0,1] = (c) SBQ (Raw)
#   axes[1,0] = (b) SAQ (Production) |  axes[1,1] = (d) SBQ (Production)
# ══════════════════════════════════════════════════════════════════════════════
def plot_main_figure(saqs: Dict, sbqs: Dict) -> None:
    """One combined figure with 4 subplots — win rate only (main paper)."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), constrained_layout=True)

    subplots = [
        (axes[0, 0], saqs, "raw",        "(a) SAQs (Raw)"),
        (axes[1, 0], saqs, "production", "(b) SAQs (Production)"),
        (axes[0, 1], sbqs, "raw",        "(c) SBQs (Raw)"),
        (axes[1, 1], sbqs, "production", "(d) SBQs (Production)"),
    ]

    centers, offsets = _group_layout(len(PIPELINE_ORDER), len(MODEL_ORDER))

    for ax, data, mode, title in subplots:
        rates = extract_rates(data, mode)

        # Draw bars pipeline-by-pipeline so "Ours" gets a distinct outline
        for mi, model in enumerate(MODEL_ORDER):
            for pi, p in enumerate(PIPELINE_ORDER):
                wr = rates.get(p, {}).get(model, {}).get("win_rate", 0.0)
                is_ours = (p == "proposed_pipeline")
                ax.bar(
                    centers[pi] + offsets[mi],
                    wr,
                    width=_BAR_W,
                    color=MODEL_COLORS[model],
                    edgecolor="black" if is_ours else "white",
                    linewidth=1.2 if is_ours else 0.4,
                    zorder=3,
                )
                # Win-rate label above every bar
                ax.text(
                    centers[pi] + offsets[mi],
                    wr + 0.008,
                    f"{wr:.2f}",
                    ha="center", va="bottom",
                    fontsize=4.0,
                    color="black",
                    clip_on=True,
                )

        ax.set_title(title, pad=3)
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [PIPELINE_LABELS[p] for p in PIPELINE_ORDER],
            rotation=25, ha="right",
        )
        ax.set_ylim(0, 1.09)
        ax.set_ylabel("Win Rate")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.suptitle(
        "Matched-Task Win Rate Across Pipelines and Models",
        fontsize=9.5, fontweight="bold",
    )

    _save(fig, "main_figure")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Appendix: 2x2 stacked win / tie / loss bar chart
# Rates are averaged across models — one clean bar per pipeline.
# ══════════════════════════════════════════════════════════════════════════════
def plot_appendix_stacked(saqs: Dict, sbqs: Dict) -> None:
    """2x2 stacked bar charts — win / tie / loss (averaged across models)."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), constrained_layout=True)

    subplots = [
        (axes[0, 0], saqs, "raw",        "(a) SAQs (Raw)"),
        (axes[1, 0], saqs, "production", "(b) SAQs (Production)"),
        (axes[0, 1], sbqs, "raw",        "(c) SBQs (Raw)"),
        (axes[1, 1], sbqs, "production", "(d) SBQs (Production)"),
    ]

    x     = np.arange(len(PIPELINE_ORDER))
    bar_w = 0.55

    for ax, data, mode, title in subplots:
        # Use global pipeline-level rates directly. The new evaluation guarantees
        # every pipeline has the same denominator (global matched set), so these
        # rates are the correct aggregated values — more accurate than a simple
        # per-model average when model groups have unequal sizes.
        global_rates = extract_global_rates(data, mode)
        for pi, p in enumerate(PIPELINE_ORDER):
            wr = global_rates.get(p, {}).get("win_rate",  0.0)
            tr = global_rates.get(p, {}).get("tie_rate",  0.0)
            lr = global_rates.get(p, {}).get("loss_rate", 0.0)
            is_ours = (p == "proposed_pipeline")
            ec = "black" if is_ours else "white"
            lw = 1.1 if is_ours else 0.3

            ax.bar(pi, wr, width=bar_w, color=WIN_COLOR,  edgecolor=ec, linewidth=lw, alpha=0.9, zorder=3)
            ax.bar(pi, tr, width=bar_w, color=TIE_COLOR,  edgecolor=ec, linewidth=lw, alpha=0.9, bottom=wr, zorder=3)
            ax.bar(pi, lr, width=bar_w, color=LOSS_COLOR, edgecolor=ec, linewidth=lw, alpha=0.9, bottom=wr + tr, zorder=3)
            # "W=0.74" label above the full stacked bar
            ax.text(
                pi, 1.013,
                f"W={wr:.2f}",
                ha="center", va="bottom",
                fontsize=4.8,
                color="black",
                fontweight="bold" if is_ours else "normal",
                clip_on=True,
            )

        ax.set_title(title, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PIPELINE_LABELS[p] for p in PIPELINE_ORDER],
            rotation=25, ha="right",
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Rate (global matched set)")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    stack_handles = [
        mpatches.Patch(facecolor=WIN_COLOR,  alpha=0.9, label="Win"),
        mpatches.Patch(facecolor=TIE_COLOR,  alpha=0.9, label="Tie"),
        mpatches.Patch(facecolor=LOSS_COLOR, alpha=0.9, label="Loss"),
    ]
    fig.legend(
        handles=stack_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.suptitle(
        "Matched-Task Outcome Distribution (Win / Tie / Loss)",
        fontsize=9, fontweight="bold",
    )

    _save(fig, "appendix_stacked")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Appendix: Δ win-rate (production − raw) per pipeline / model
# ══════════════════════════════════════════════════════════════════════════════
def plot_appendix_delta(saqs: Dict, sbqs: Dict) -> None:
    """Dumbbell plot: raw (open) → production (filled) win rate per pipeline & model.

    Each dumbbell shows where a pipeline started (raw) and ended (production),
    with the connecting segment coloured by evaluation model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)

    subplots = [
        (axes[0], saqs, "SAQs"),
        (axes[1], sbqs, "SBQs"),
    ]

    centers, offsets = _group_layout(len(PIPELINE_ORDER), len(MODEL_ORDER))
    ours_i = PIPELINE_ORDER.index("proposed_pipeline")

    for ax, data, qt_label in subplots:
        raw_rates  = extract_rates(data, "raw")
        prod_rates = extract_rates(data, "production")

        # Shade the "Ours" group column
        ours_x = centers[ours_i]
        half_w  = (_BAR_W * len(MODEL_ORDER) / 2.0) + 0.06
        ax.axvspan(ours_x - half_w, ours_x + half_w,
                   color="#F5F5F5", zorder=1, linewidth=0)
        ax.axvline(ours_x - half_w, color="#CCCCCC", linewidth=0.5, zorder=2)
        ax.axvline(ours_x + half_w, color="#CCCCCC", linewidth=0.5, zorder=2)

        for mi, model in enumerate(MODEL_ORDER):
            for pi, p in enumerate(PIPELINE_ORDER):
                x_pos    = centers[pi] + offsets[mi]
                raw_val  = raw_rates.get(p, {}).get(model, {}).get("win_rate", 0.0)
                prod_val = prod_rates.get(p, {}).get(model, {}).get("win_rate", 0.0)
                color    = MODEL_COLORS[model]

                # Connecting stem
                ax.plot(
                    [x_pos, x_pos], [raw_val, prod_val],
                    color=color, linewidth=1.4, zorder=3,
                )
                # Raw endpoint — open circle
                ax.scatter(
                    x_pos, raw_val,
                    s=22, color=color,
                    facecolors="white", edgecolors=color,
                    linewidths=1.2, zorder=5,
                )
                # Production endpoint — filled circle
                ax.scatter(
                    x_pos, prod_val,
                    s=22, color=color,
                    zorder=6,
                )

        ax.set_xticks(centers)
        ax.set_xticklabels(
            [PIPELINE_LABELS[p] for p in PIPELINE_ORDER],
            rotation=25, ha="right",
        )
        ax.set_ylabel("Win Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(qt_label, pad=3)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Legend: model colours + raw/production marker key
    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="#555555", markersize=5,
               markerfacecolor="white", markeredgecolor="#555555",
               linewidth=0, label="Raw"),
        Line2D([0], [0], marker="o", color="#555555", markersize=5,
               markerfacecolor="#555555", markeredgecolor="#555555",
               linewidth=0, label="Production"),
    ]
    fig.legend(
        handles=model_handles + marker_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
    )
    fig.suptitle(
        "Raw vs. Production Win Rate",
        fontsize=9.5, fontweight="bold",
    )

    _save(fig, "appendix_delta")


# ── Bloom taxonomy constants ──────────────────────────────────────────────────
BLOOM_ORDER = ["knowledge", "understanding", "application", "analyze", "synthesis", "evaluation"]
BLOOM_LABELS = {
    "knowledge":     "Knowledge",
    "understanding": "Understanding",
    "application":   "Application",
    "analyze":       "Analyze",
    "synthesis":     "Synthesis",
    "evaluation":    "Evaluation",
}


# ── Additional data helpers ────────────────────────────────────────────────────
def build_win_matrix(results: Dict, mode: str) -> np.ndarray:
    """5x5 win-rate matrix; cell[i,j] = win rate of PIPELINE_ORDER[i] vs PIPELINE_ORDER[j]."""
    n = len(PIPELINE_ORDER)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.5)
    for entry in results[mode]["per_pair"].values():
        pa, pb = entry["pipeline_a"], entry["pipeline_b"]
        if pa not in PIPELINE_ORDER or pb not in PIPELINE_ORDER:
            continue
        i, j = PIPELINE_ORDER.index(pa), PIPELINE_ORDER.index(pb)
        wr = entry.get("win_rate", {})
        matrix[i, j] = wr.get(pa, np.nan)
        matrix[j, i] = wr.get(pb, np.nan)
    return matrix


def build_bloom_matrix(results: Dict, mode: str):
    """Pipeline x bloom win rates aggregated over all matchups.

    Returns (matrix, present_blooms) where matrix shape is
    (n_pipelines, n_present_blooms), ordered by BLOOM_ORDER.
    Cells with zero total are NaN.
    """
    bloom_wins:  Dict[str, Dict[str, int]]   = {p: {} for p in PIPELINE_ORDER}
    bloom_total: Dict[str, Dict[str, int]]   = {p: {} for p in PIPELINE_ORDER}
    for entry in results[mode]["per_pair"].values():
        pa, pb = entry["pipeline_a"], entry["pipeline_b"]
        for bloom, bv in entry.get("bloom_breakdown", {}).items():
            if pa in PIPELINE_ORDER:
                bloom_wins[pa][bloom]  = bloom_wins[pa].get(bloom, 0)  + bv["wins_a"]
                bloom_total[pa][bloom] = bloom_total[pa].get(bloom, 0) + bv["total"]
            if pb in PIPELINE_ORDER:
                bloom_wins[pb][bloom]  = bloom_wins[pb].get(bloom, 0)  + bv["wins_b"]
                bloom_total[pb][bloom] = bloom_total[pb].get(bloom, 0) + bv["total"]
    present = [b for b in BLOOM_ORDER if any(bloom_total[p].get(b, 0) > 0 for p in PIPELINE_ORDER)]
    n_p, n_b = len(PIPELINE_ORDER), len(present)
    matrix = np.full((n_p, n_b), np.nan)
    for pi, p in enumerate(PIPELINE_ORDER):
        for bi, b in enumerate(present):
            total = bloom_total[p].get(b, 0)
            if total > 0:
                matrix[pi, bi] = bloom_wins[p].get(b, 0) / total
    return matrix, present


def extract_ours_margins(results: Dict, mode: str) -> Dict[str, Dict[str, float]]:
    """Per-model score margins of proposed_pipeline over each baseline.

    Returns baseline → model → margin (positive = Ours scores higher on average).
    """
    margins: Dict[str, Dict[str, float]] = {}
    for entry in results[mode]["per_pair"].values():
        pa, pb = entry["pipeline_a"], entry["pipeline_b"]
        if pa == "proposed_pipeline":
            baseline, sign = pb, 1
        elif pb == "proposed_pipeline":
            baseline, sign = pa, -1
        else:
            continue
        margins[baseline] = {}
        for model, ms in entry.get("per_model", {}).items():
            margins[baseline][model] = sign * (
                ms.get("mean_score_a", 0.0) - ms.get("mean_score_b", 0.0)
            )
    return margins


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Pairwise win-rate matrix heatmap
#   Rows = pipeline A, Columns = pipeline B
#   Cell = win rate of A when facing B (diagonal = 0.5)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pairwise_matrix(saqs: Dict, sbqs: Dict) -> None:
    """2x2 pairwise win-rate heatmaps."""
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), constrained_layout=True)
    subplots = [
        (axes[0, 0], saqs, "raw",        "(a) SAQs (Raw)"),
        (axes[1, 0], saqs, "production", "(b) SAQs (Production)"),
        (axes[0, 1], sbqs, "raw",        "(c) SBQs (Raw)"),
        (axes[1, 1], sbqs, "production", "(d) SBQs (Production)"),
    ]
    labels = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER]
    norm   = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    ours_i = PIPELINE_ORDER.index("proposed_pipeline")
    n = len(PIPELINE_ORDER)

    for ax, data, mode, title in subplots:
        mat = build_win_matrix(data, mode)
        im  = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if (val < 0.18 or val > 0.82) else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color=text_color,
                    fontweight="bold" if i == ours_i else "normal",
                )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Opponent Pipeline", fontsize=7)
        ax.set_ylabel("Pipeline", fontsize=7)

        # Bold cell outlines for the "Ours" row
        for j in range(n):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, ours_i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1.2, zorder=4,
            ))

        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Win Rate", fontsize=6.5)

    fig.suptitle(
        "Head-to-Head Win Rate Matrix (cell = row pipeline's win rate vs. column pipeline)",
        fontsize=8.5, fontweight="bold",
    )
    _save(fig, "pairwise_win_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Bloom-level win-rate heatmap
#   Rows = pipelines, Columns = Bloom's taxonomy levels
#   Win rate aggregated across all matchups at that Bloom level
# ══════════════════════════════════════════════════════════════════════════════
def plot_bloom_heatmap(saqs: Dict, sbqs: Dict) -> None:
    """2x2 pipeline x Bloom-level win-rate heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), constrained_layout=True)
    subplots = [
        (axes[0, 0], saqs, "raw",        "(a) SAQs (Raw)"),
        (axes[1, 0], saqs, "production", "(b) SAQs (Production)"),
        (axes[0, 1], sbqs, "raw",        "(c) SBQs (Raw)"),
        (axes[1, 1], sbqs, "production", "(d) SBQs (Production)"),
    ]
    ours_i = PIPELINE_ORDER.index("proposed_pipeline")

    for ax, data, mode, title in subplots:
        mat, present_blooms = build_bloom_matrix(data, mode)
        bloom_xlabels = [BLOOM_LABELS.get(b, b.capitalize()) for b in present_blooms]

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        n_p, n_b = mat.shape
        for pi in range(n_p):
            for bi in range(n_b):
                val = mat[pi, bi]
                if np.isnan(val):
                    ax.text(bi, pi, "—", ha="center", va="center",
                            fontsize=6, color="#888888")
                    continue
                text_color = "white" if (val < 0.2 or val > 0.8) else "black"
                ax.text(
                    bi, pi, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color=text_color,
                    fontweight="bold" if pi == ours_i else "normal",
                )

        ax.set_xticks(range(n_b))
        ax.set_yticks(range(n_p))
        ax.set_xticklabels(bloom_xlabels, rotation=25, ha="right", fontsize=6.5)
        ax.set_yticklabels([PIPELINE_LABELS[p] for p in PIPELINE_ORDER], fontsize=6.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Bloom's Taxonomy Level", fontsize=7)
        ax.set_ylabel("Pipeline", fontsize=7)

        for bi in range(n_b):
            ax.add_patch(plt.Rectangle(
                (bi - 0.5, ours_i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1.2, zorder=4,
            ))

        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Win Rate", fontsize=6.5)

    fig.suptitle(
        "Pipeline Win Rate by Bloom's Taxonomy Level",
        fontsize=9.5, fontweight="bold",
    )
    _save(fig, "bloom_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Win margin: Ours vs each baseline
#   y = mean_score(Ours) − mean_score(baseline) per evaluation model
#   Bars show mean across models; dots show individual model values
# ══════════════════════════════════════════════════════════════════════════════
def plot_win_margin(saqs: Dict, sbqs: Dict) -> None:
    """2x2 score-margin plot: Ours vs each baseline, per evaluation model."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), constrained_layout=True)
    subplots = [
        (axes[0, 0], saqs, "raw",        "(a) SAQs (Raw)"),
        (axes[1, 0], saqs, "production", "(b) SAQs (Production)"),
        (axes[0, 1], sbqs, "raw",        "(c) SBQs (Raw)"),
        (axes[1, 1], sbqs, "production", "(d) SBQs (Production)"),
    ]
    baselines   = [p for p in PIPELINE_ORDER if p != "proposed_pipeline"]
    bl_labels   = [PIPELINE_LABELS[b] for b in baselines]
    x           = np.arange(len(baselines))
    jitter      = np.linspace(-0.13, 0.13, len(MODEL_ORDER))
    markers     = ["o", "^", "D"]

    for ax, data, mode, title in subplots:
        margins = extract_ours_margins(data, mode)

        # Background bar: mean margin across models
        for xi, b in enumerate(baselines):
            model_vals = [margins.get(b, {}).get(m, np.nan) for m in MODEL_ORDER]
            model_vals = [v for v in model_vals if not np.isnan(v)]
            if not model_vals:
                continue
            mm = float(np.mean(model_vals))
            ax.bar(
                xi, mm, width=0.40,
                color="#90CAF9" if mm >= 0 else "#EF9A9A",
                alpha=0.55, zorder=2,
            )
            ax.text(
                xi, mm + (0.003 if mm >= 0 else -0.003),
                f"{mm:+.2f}",
                ha="center", va="bottom" if mm >= 0 else "top",
                fontsize=5.0, color="#1565C0" if mm >= 0 else "#B71C1C",
                fontweight="bold",
            )

        # Per-model dots
        for mi, model in enumerate(MODEL_ORDER):
            ys = [margins.get(b, {}).get(model, np.nan) for b in baselines]
            ax.scatter(
                x + jitter[mi], ys,
                color=MODEL_COLORS[model],
                marker=markers[mi],
                s=28, zorder=4,
            )

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(bl_labels, rotation=20, ha="right")
        ax.set_ylabel("Score Margin (Ours − Baseline)")
        ax.set_title(title, pad=3)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS[m], marker=markers[mi],
            markersize=5, linewidth=0, label=MODEL_LABELS[m],
        )
        for mi, m in enumerate(MODEL_ORDER)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.suptitle(
        "Score Margin of Ours over Each Baseline (per evaluation model)",
        fontsize=9.5, fontweight="bold",
    )
    _save(fig, "win_margin")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading evaluation results …")
    saqs = load_results("saqs")
    sbqs = load_results("sbqs")

    print("\n[1/6] Generating main figure …")
    plot_main_figure(saqs, sbqs)

    print("\n[2/6] Generating appendix stacked figure …")
    plot_appendix_stacked(saqs, sbqs)

    print("\n[3/6] Generating appendix delta figure …")
    plot_appendix_delta(saqs, sbqs)

    print("\n[4/6] Generating pairwise win matrix …")
    plot_pairwise_matrix(saqs, sbqs)

    print("\n[5/6] Generating Bloom-level heatmap …")
    plot_bloom_heatmap(saqs, sbqs)

    print("\n[6/6] Generating win margin plot …")
    plot_win_margin(saqs, sbqs)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
