"""Matplotlib figures for the results report.

matplotlib is optional (pip install -r evaluation/report/requirements.txt):
without it the report still writes every table and lists the figures as skipped.

Colours (slots of a colour-vision-deficiency-checked categorical palette, each
with its own hatch and marker so no series relies on colour alone):
base zero-shot = blue, base few-shot = aqua, fine-tuned = orange. Text is ink
grey, never a series colour. Every figure is skipped, with a reason, when the
data it needs doesn't exist; nothing is drawn from placeholders.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.report.artifacts import CATEGORIES, CATEGORY_LABELS, Artifacts, EvalRun, merge, mode_label
from evaluation.report.render import search_groups, sorted_runs

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Search-mode bars: pass@1 (same prompts), best-of-4, + repair, then any other mode.
MODE_STYLES = [("#2a78d6", ""), ("#eb6834", "//"), ("#1baf7a", ".."), ("#eda100", "xx"), ("#e87ba4", "\\\\")]


def _series_style(run: EvalRun) -> tuple[str, str, str]:
    """(colour, hatch, marker), fixed per model + prompt mode so a colour always means the same series."""
    if run.model == "ft":
        return ("#eda100", "xx", "^") if run.prompt_mode == "few_shot" else ("#eb6834", "//", "D")
    return ("#1baf7a", "..", "s") if run.prompt_mode == "few_shot" else ("#2a78d6", "", "o")


def _pass1_runs(art: Artifacts) -> list[EvalRun]:
    return [r for r in sorted_runs(art) if not r.is_search]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # files only, no display needed
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "legend.frameon": False, "hatch.color": SURFACE,
    })
    return plt


def _pct_and_err(p):
    """Bar height and asymmetric error (in percent) from a Proportion; NaN height when n == 0."""
    if p.n == 0:
        return float("nan"), (0.0, 0.0)
    lo, hi = p.interval
    pct = p.fraction * 100
    return pct, (pct - lo * 100, hi * 100 - pct)


def correctness_by_category(art: Artifacts, path: Path) -> str:
    runs = _pass1_runs(art)
    plt = _pyplot()
    keys = [*CATEGORIES, "all"]
    fig, ax = plt.subplots(figsize=(8 + 1.5 * max(0, len(runs) - 2), 4.2))  # wider for >2 series
    width = 0.8 / len(runs)
    ns = {k: [] for k in keys}  # n per bar, shown on a second tick-label line
    for i, run in enumerate(runs):
        colour, hatch, _ = _series_style(run)
        props = [(run.overall() if k == "all" else run.categories.get(k)) for k in keys]
        heights, errs = zip(*[_pct_and_err(p.proportion) if p else (float("nan"), (0, 0)) for p in props])
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(keys))]
        ax.bar(xs, heights, width * 0.94, color=colour, hatch=hatch, label=run.label,
               yerr=list(zip(*errs)), error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
        for k, p in zip(keys, props):
            ns[k].append(str(p.n if p else 0))
    ax.set_xticks(range(len(keys)))
    sep = " | " if len(runs) <= 2 else "|"
    ax.set_xticklabels([f"{'All' if k == 'all' else CATEGORY_LABELS[k]}\nn = {sep.join(ns[k])}" for k in keys],
                       fontsize=9 if len(runs) <= 2 else 8)
    ax.set_ylim(0, 118)  # headroom above 100% so the one-row legend never covers a whisker
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct (whiskers: Wilson 95% CI)")
    if len(runs) > 1:
        ax.set_xlabel("n per bar, in legend order", color=INK_MUTED, fontsize=8)
    ax.set_title("Correctness by category on held-out templates (pass@1)", loc="left")
    ax.legend(loc="upper right", ncol=len(runs), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ("% correct per category for each pass@1 run (base zero-/few-shot kept separate, fine-tuned), "
            "with Wilson 95% intervals and n per bar.")


def speedup_distribution(art: Artifacts, path: Path) -> str:
    # Interpreter runs have no timings; leave them out rather than plot an empty series.
    runs = [r for r in _pass1_runs(art) if _has_speedups(r) and not r.from_summary]
    plt = _pyplot()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, ax = plt.subplots(figsize=(8, 4.2))
    step = 0.6 / max(len(runs), 1)
    for i, run in enumerate(runs):
        colour, _, marker = _series_style(run)
        offset = (i - (len(runs) - 1) / 2) * step
        xs, ys = [], []
        for j, cat in enumerate(CATEGORIES):
            values = run.categories[cat].speedups if cat in run.categories else []
            # Deterministic small jitter so overlapping points stay visible.
            xs += [j + offset + ((k % 5) - 2) * 0.02 for k in range(len(values))]
            ys += values
        ax.scatter(xs, ys, s=36, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1.2,
                   label=f"{run.label} (n={len(ys)} correct kernels)", zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.text(len(CATEGORIES) - 0.5, 1.0, " PyTorch reference (1×)", va="bottom", ha="right", fontsize=8,
            color=INK_MUTED)
    ax.set_yscale("log")
    ticks = [2.0 ** e for e in range(-6, 7)]  # 1/64x .. 64x; only those in range are drawn
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    ax.set_ylabel("Speedup vs PyTorch reference (log scale)")
    ax.set_title("Speedup of correct kernels (pass@1)", loc="left")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return "Speedup of each correct kernel vs the PyTorch reference, per category (log scale; one point per kernel)."


def search_gain(art: Artifacts, path: Path) -> str:
    """One cluster per search group: pass@1 on exactly the search prompts, then each search mode."""
    groups = [g for g in search_groups(art) if g.ids is not None]
    modes = [(1, 0)] + sorted({(r.num_candidates, r.repair_rounds) for g in groups for r in g.runs})
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(groups) + 2), 4.2))
    width = 0.8 / len(modes)
    ns = {i: [] for i in range(len(groups))}
    for m_i, mode in enumerate(modes):
        heights, errs = [], []
        for g_i, g in enumerate(groups):
            if mode == (1, 0):
                subset = g.pass1_same_prompts()
                p = merge(subset.values()).proportion if subset is not None else None
            else:
                run = next((r for r in g.runs if (r.num_candidates, r.repair_rounds) == mode), None)
                if run is None:
                    p = None
                elif g.incomplete_pass1:  # pass@1 missed some prompts: plot both on the shared ones only
                    p = merge(run.subset(g.shared_ids).values()).proportion
                else:
                    p = run.overall().proportion
            h, e = _pct_and_err(p) if p is not None else (float("nan"), (0.0, 0.0))
            heights.append(h)
            errs.append(e)
            ns[g_i].append(str(p.n) if p is not None else "–")
        colour, hatch = MODE_STYLES[m_i % len(MODE_STYLES)]
        label = "pass@1 (same prompts)" if mode == (1, 0) else mode_label(*mode)
        xs = [j + (m_i - (len(modes) - 1) / 2) * width for j in range(len(groups))]
        ax.bar(xs, heights, width * 0.94, color=colour, hatch=hatch, label=label,
               yerr=list(zip(*errs)), error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g.label}, {len(g.shared_ids if g.incomplete_pass1 else g.ids)} same prompts"
                        f"\nn = {' | '.join(ns[i])}"
                        for i, g in enumerate(groups)])
    ax.set_xlabel("n per bar, in legend order", color=INK_MUTED, fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct, all categories (Wilson 95% CI)")
    ax.set_title("What inference-time search buys (same prompts)", loc="left")
    ax.legend(loc="upper left", ncol=len(modes), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ("Overall % correct on the search prompts only: pass@1 recomputed on those prompts, best-of-N and "
            "best-of-N + repair, with Wilson 95% intervals.")


def _has_speedups(run: EvalRun) -> bool:
    return not run.interpreted and any(c.speedups for c in run.categories.values())


def make_figures(art: Artifacts, out_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Draw what the data supports. Returns ([(relative path, caption)], [(figure, reason skipped)])."""
    try:
        _pyplot()
    except ImportError:
        reason = "matplotlib is not installed (pip install -r evaluation/report/requirements.txt)"
        return [], [(n, reason) for n in ("correctness_by_category", "speedup_distribution", "search_gain")]

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    pass1 = _pass1_runs(art)
    plan = [
        ("correctness_by_category", correctness_by_category, bool(pass1), "no pass@1 evaluation results"),
        ("speedup_distribution", speedup_distribution,
         any(_has_speedups(r) and not r.from_summary for r in pass1),
         "no per-kernel speedups (no correct kernels, interpreter-only, or summary-only reports)"),
        ("search_gain", search_gain, any(g.ids is not None for g in search_groups(art)),
         "no search runs with per-entry results (nothing to compare on the same prompts)"),
    ]
    for name, draw, ok, why in plan:
        if not ok:
            skipped.append((name, why))
            continue
        path = fig_dir / f"{name}.png"
        caption = draw(art, path)
        made.append((f"figures/{name}.png", caption))
    return made, skipped
