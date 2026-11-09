# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib==3.9.2"]
# ///

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "log.jsonl"
OUT = ROOT / "results" / "charts"

INK = "#0b0b0b"
TEXT = "#52514e"
FAINT = "#8b8a85"
GRID = "#e6e5e0"
BG = "#fcfcfb"
BLUE = "#2a78d6"
GRAY = "#a3a29c"

FIGSIZE = (8, 4.5)


def load_rows():
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no results in {LOG.relative_to(ROOT)}")
    return rows


def fmt_value(value):
    return f"{value:.2g}" if value < 1 else f"{value:.1f}"


def short_title(title):
    head = re.split(r"[:,]", title, maxsplit=1)[0].strip()
    return textwrap.shorten(head, width=28, placeholder="…")


def fit_size(text, start, width_in=7.25):
    size = start
    while size > 9:
        width = TextPath(
            (0, 0), text, size=size,
            prop=FontProperties(family="DejaVu Sans"),
        ).get_extents().width / 72
        if width <= width_in:
            break
        size -= 0.5
    return size


def label(row):
    text = f"{row['milestone']} · {short_title(row['title'])}"
    if row["token_match_rate"] < 1.0:
        text += " (int8)"
    return text


def new_figure():
    return plt.figure(figsize=FIGSIZE, dpi=200, facecolor=BG)


def finish(fig, path):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(path)


def header(fig, title, subtitle, headline=None, title_size=13):
    fig.text(0.055, 0.975, title, va="top", ha="left", fontsize=title_size,
             color=INK)
    y = 0.915
    if headline:
        fig.text(0.055, 0.885, headline, va="top", ha="left", fontsize=27,
                 fontweight="bold", color=INK)
        y = 0.79
    fig.text(0.055, y, subtitle, va="top", ha="left", fontsize=9, color=TEXT,
             linespacing=1.45)
    fig.text(0.055, 0.03, "Source: results/log.jsonl", va="bottom", ha="left",
             fontsize=8, color=FAINT)


def style_axis(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=TEXT, labelsize=8.5)


def progress():
    all_rows = load_rows()
    rows = [row for row in all_rows if "-threads" not in row["milestone"]]
    values = [row["decode_tok_s"] for row in rows]
    lo, hi = min(values) / 2, max(values) * 2.2

    fig = new_figure()
    labels = [label(row) for row in rows]
    tick_size = 8.5 if len(rows) <= 12 else 7.5
    gutter = max(len(text) for text in labels) * tick_size * 0.62 / 72 + 0.2
    left = min(gutter / 8, 0.42)
    ax = fig.add_axes((left, 0.115, 0.955 - left, 0.735))

    y = list(range(len(rows)))
    base = lo
    widths = [value - base for value in values]
    colors = [BLUE if row["milestone"].startswith("N") else GRAY for row in rows]
    bars = ax.barh(y, widths, left=base, height=0.6, color=colors,
                   edgecolor=TEXT, linewidth=0)
    for bar, row in zip(bars, rows):
        if row["token_match_rate"] < 1.0:
            bar.set_hatch("////")

    for yi, value in zip(y, values):
        ax.annotate(fmt_value(value), (value, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=TEXT)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(len(rows), 3) - 0.5, -0.5)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=tick_size)

    last = all_rows[-1]
    header(
        fig,
        "GPT-2 124M decode speed, tokens per second",
        f"{last['host']} · median of session medians",
    )
    finish(fig, OUT / "progress.png")


def previous_label(row):
    labels = {"R0-torch": "PyTorch", "R0-numpy": "NumPy"}
    return labels.get(row["milestone"], row["milestone"])


def step(name):
    rows = load_rows()
    current = next((row for row in rows if row["milestone"] == name), None)
    if current is None:
        raise SystemExit(f"milestone {name!r} not found in {LOG.relative_to(ROOT)}")

    index = rows.index(current)
    previous = next(
        (row for row in reversed(rows[:index])
         if row["milestone"].startswith("N") and "-threads" not in row["milestone"]),
        None,
    )
    if previous is None:
        previous = next(
            (row for row in rows[:index] if row["milestone"] == "R0-torch"), None
        )
    if previous is None:
        raise SystemExit(f"no earlier milestone to compare with {name}")

    ratio = current["decode_tok_s"] / previous["decode_tok_s"]
    if ratio >= 1:
        decimals = 1 if ratio < 100 else 0
        headline = f"{ratio:.{decimals}f}× faster decode"
    else:
        slower = 1 / ratio
        amount = f"{slower:.0f}" if slower >= 100 else f"{slower:.1f}"
        headline = f"{amount}× slower than {previous_label(previous)}"

    comparison = f"compared with {previous['milestone']}"
    title = f"{name}: {current['title']}"
    fig = new_figure()
    header(
        fig,
        title,
        f"{current['host']} · commit {current['commit']}\n{comparison}",
        headline=headline,
        title_size=fit_size(title, 13),
    )

    panels = [
        ("Decode, tokens/s", current["decode_tok_s"], previous["decode_tok_s"]),
        ("Time to first token, ms (lower is better)",
         current["ttft_ms"], previous["ttft_ms"]),
    ]
    grid = fig.add_gridspec(1, 2, left=0.17, right=0.94, bottom=0.16, top=0.66,
                            wspace=0.34)
    for column, (panel_title, now, before) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        ax.barh([0, 1], [before, now], height=0.55, color=[GRAY, BLUE],
                edgecolor=TEXT, linewidth=0)
        ax.set_yticks([0, 1], [previous["milestone"], name])
        ax.set_ylim(1.65, -0.65)
        ax.set_xlim(0, max(before, now) * 1.18)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(panel_title, loc="left", fontsize=9.5, color=INK, pad=9)
        ax.annotate(fmt_value(before), (before, 0), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=TEXT)
        ax.annotate(fmt_value(now), (now, 1), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=TEXT)
        style_axis(ax)

    finish(fig, OUT / f"step-{name}.png")


def main():
    parser = argparse.ArgumentParser(description="Chart GPT-2 decode results.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("progress", help="one bar per milestone")
    step_parser = sub.add_parser("step", help="before/after card for one milestone")
    step_parser.add_argument("milestone")
    args = parser.parse_args()

    if args.command == "progress":
        progress()
    else:
        step(args.milestone)


if __name__ == "__main__":
    main()
