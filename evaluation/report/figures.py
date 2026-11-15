"""Matplotlib figures for the results report.

matplotlib is optional (not in requirements.txt): without it the report still
writes every table and says the figures were skipped.

Colours: base = blue, fine-tuned = orange (slots 1-2 of a colour-vision-
deficiency-checked categorical palette), with a hatch on the fine-tuned series
so the two never rely on colour alone (greyscale print, CVD). Text is ink
grey, never the series colour. Every figure is skipped, with a reason, when
the data it needs doesn't exist; nothing is drawn from placeholders.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.report.artifacts import CATEGORIES, CATEGORY_LABELS, Artifacts, EvalRun, mode_label
from evaluation.report.render import SEARCH_MODES

SERIES = {"base": ("#2a78d6", ""), "ft": ("#eb6834", "//")}  # colour, hatch
MODE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # pass@1, best-of-4, + repair
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"base": "Base", "ft": "Fine-tuned"}


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
    runs = {m: art.run(m) for m in ("base", "ft") if art.run(m)}
    plt = _pyplot()
    keys = [*CATEGORIES, "all"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(runs)
    ns = {k: [] for k in keys}  # n per bar, shown on a second tick-label line
    for i, (model, run) in enumerate(runs.items()):
        colour, hatch = SERIES[model]
        props = [(run.overall() if k == "all" else run.categories.get(k)) for k in keys]
        heights, errs = zip(*[_pct_and_err(p.proportion) if p else (float("nan"), (0, 0)) for p in props])
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(keys))]
        ax.bar(xs, heights, width * 0.94, color=colour, hatch=hatch, label=f"{NAMES[model]} pass@1",
               yerr=list(zip(*errs)), error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
        for k, p in zip(keys, props):
            ns[k].append(str(p.n if p else 0))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{'All' if k == 'all' else CATEGORY_LABELS[k]}\nn = {' | '.join(ns[k])}" for k in keys])
    ax.set_ylim(0, 118)  # headroom above 100% so the one-row legend never covers a whisker
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct (whiskers: Wilson 95% CI)")
    if len(runs) == 2:
        ax.set_xlabel("n per bar: base | fine-tuned", color=INK_MUTED, fontsize=8)
    ax.set_title("Correctness by category on held-out templates", loc="left")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return "% correct per category, base vs fine-tuned (pass@1), with Wilson 95% intervals and n per bar."


def speedup_distribution(art: Artifacts, path: Path) -> str:
    # Interpreter runs have no timings; leave them out rather than plot an empty series.
    runs = {m: art.run(m) for m in ("base", "ft") if art.run(m) and _has_speedups(art.run(m))}
    plt = _pyplot()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, ax = plt.subplots(figsize=(8, 4.2))
    offsets = {"base": -0.15, "ft": 0.15} if len(runs) == 2 else {m: 0.0 for m in runs}
    markers = {"base": "o", "ft": "D"}  # shape as a second channel besides colour
    for model, run in runs.items():
        colour, _ = SERIES[model]
        xs, ys = [], []
        for j, cat in enumerate(CATEGORIES):
            values = run.categories.get(cat).speedups if cat in run.categories else []
            # Deterministic small jitter so overlapping points stay visible.
            xs += [j + offsets[model] + ((k % 5) - 2) * 0.025 for k in range(len(values))]
            ys += values
        ax.scatter(xs, ys, s=36, color=colour, marker=markers[model], edgecolors=SURFACE, linewidths=1.2,
                   label=f"{NAMES[model]} (n={len(ys)} correct kernels)", zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.text(len(CATEGORIES) - 0.5, 1.0, " PyTorch reference (1×)", va="bottom", ha="right", fontsize=8, color=INK_MUTED)
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
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return "Speedup of each correct kernel vs the PyTorch reference, per category (log scale; one point per kernel)."


def search_gain(art: Artifacts, path: Path) -> str:
    models = [m for m in ("base", "ft") if any(art.run(m, nc, rr) for nc, rr in SEARCH_MODES)]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.8 / len(SEARCH_MODES)
    ns = {m: [] for m in models}
    for i, (nc, rr) in enumerate(SEARCH_MODES):
        heights, errs = [], []
        for model in models:
            run = art.run(model, nc, rr)
            h, e = _pct_and_err(run.overall().proportion) if run else (float("nan"), (0.0, 0.0))
            heights.append(h)
            errs.append(e)
            ns[model].append(str(run.overall().n) if run else "not run")
        xs = [j + (i - 1) * width for j in range(len(models))]
        ax.bar(xs, heights, width * 0.94, color=MODE_COLOURS[i], hatch=["", "//", ".."][i], label=mode_label(nc, rr),
               yerr=list(zip(*errs)), error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{NAMES[m]}\nn = {' | '.join(ns[m])}" for m in models])
    ax.set_xlabel("n per bar, in legend order", color=INK_MUTED, fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct, all categories (Wilson 95% CI)")
    ax.set_title("What inference-time search buys", loc="left")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return "Overall % correct for pass@1, best-of-4 and best-of-4 + 1 repair, with Wilson 95% intervals."


def _has_speedups(run: EvalRun) -> bool:
    return not run.interpreted and any(c.speedups for c in run.categories.values())


def make_figures(art: Artifacts, out_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Draw what the data supports. Returns ([(relative path, caption)], [(figure, reason skipped)])."""
    try:
        _pyplot()
    except ImportError:
        reason = "matplotlib is not installed (pip install matplotlib)"
        return [], [(n, reason) for n in ("correctness_by_category", "speedup_distribution", "search_gain")]

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    pass1 = [r for r in (art.run("base"), art.run("ft")) if r]
    n_modes = sum(1 for m in ("base", "ft") for nc, rr in SEARCH_MODES if art.run(m, nc, rr))
    plan = [
        ("correctness_by_category", correctness_by_category, bool(pass1), "no pass@1 evaluation results"),
        ("speedup_distribution", speedup_distribution,
         any(_has_speedups(r) for r in pass1) and not any(r.from_summary for r in pass1),
         "no per-kernel speedups (no correct kernels, interpreter-only, or summary-only reports)"),
        ("search_gain", search_gain, n_modes >= 2, "fewer than two search modes have results"),
    ]
    for name, draw, ok, why in plan:
        if not ok:
            skipped.append((name, why))
            continue
        path = fig_dir / f"{name}.png"
        caption = draw(art, path)
        made.append((f"figures/{name}.png", caption))
    return made, skipped
