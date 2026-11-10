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
from matplotlib.patches import Patch
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
ORANGE = "#eb6834"
GRAY = "#a3a29c"
DEFAULT_ENGINE = "gpt2-fast"
WEB_TARGETS = {"wasm", "wasm-simd", "wasm-threads", "webgpu"}

FIGSIZE = (8, 4.5)


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no results in {display_path(path)}")
    return rows


def engine(row):
    return row.get("engine") or DEFAULT_ENGINE


def model_name(row):
    return row.get("model") or "gpt2-124m"


def is_native(row):
    return (row.get("target") or "cpu-linux") == "cpu-linux"


def is_browser(row):
    return row.get("target") in WEB_TARGETS


def is_124m(row):
    return model_name(row) == "gpt2-124m"


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


def quantized(row):
    return row["milestone"].endswith("-int8") or row["token_match_rate"] < 1.0


def label(row):
    text = f"{row['milestone']} · {short_title(row['title'])}"
    if quantized(row) and "int8" not in text.lower():
        text += " (int8)"
    return text


def new_figure():
    return plt.figure(figsize=FIGSIZE, dpi=200, facecolor=BG)


def finish(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(path)


def header(fig, title, subtitle, headline=None, title_size=13,
           source="Source: results/log.jsonl"):
    fig.text(0.055, 0.975, title, va="top", ha="left", fontsize=title_size,
             color=INK)
    y = 0.915
    if headline:
        fig.text(0.055, 0.885, headline, va="top", ha="left", fontsize=27,
                 fontweight="bold", color=INK)
        y = 0.79
    fig.text(0.055, y, subtitle, va="top", ha="left", fontsize=9, color=TEXT,
             linespacing=1.45)
    fig.text(0.055, 0.03, source, va="bottom", ha="left",
             fontsize=8, color=FAINT)


def style_axis(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=TEXT, labelsize=8.5)


def progress(log_path, out_dir):
    all_rows = [row for row in load_rows(log_path) if is_native(row)]
    rows_124m = [row for row in all_rows if is_124m(row)]
    if not rows_124m:
        raise SystemExit(f"no gpt2-124m results in {display_path(log_path)}")
    rows = [
        row for row in rows_124m
        if "-threads" not in row["milestone"]
        and (engine(row) == DEFAULT_ENGINE or row["milestone"].startswith("R0"))
    ]
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
        if quantized(row):
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

    last = rows_124m[-1]
    header(
        fig,
        "GPT-2 124M decode speed, tokens per second",
        f"{last['host']} · median of session medians",
    )
    finish(fig, out_dir, "progress.png")


def previous_label(row):
    labels = {"R0-torch": "PyTorch", "R0-numpy": "NumPy"}
    return labels.get(row["milestone"], row["milestone"])


def step(name, log_path, out_dir):
    rows = load_rows(log_path)
    current = next((row for row in rows if row["milestone"] == name), None)
    if current is None:
        raise SystemExit(f"milestone {name!r} not found in {display_path(log_path)}")

    if "baseline" in current:
        baseline = current["baseline"]
        previous = next((row for row in rows if row["milestone"] == baseline), None)
        if previous is None:
            raise SystemExit(
                f"baseline {baseline!r} of {name} not found in "
                f"{display_path(log_path)}"
            )
    else:
        if not is_native(current):
            raise SystemExit(
                f"browser milestone {name!r} has no baseline in "
                f"{display_path(log_path)}"
            )
        native = [row for row in rows if is_native(row)]
        index = native.index(current)
        previous = next(
            (row for row in reversed(native[:index])
             if model_name(row) == model_name(current)
             and engine(row) == DEFAULT_ENGINE
             and row["milestone"].startswith("N")
             and "-threads" not in row["milestone"]),
            None,
        )
        if previous is None and is_124m(current):
            previous = next(
                (row for row in native[:index] if row["milestone"] == "R0-torch"),
                None,
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
    if is_native(current):
        subtitle = f"{current['host']} · commit {current['commit']}\n{comparison}"
    else:
        place = " · ".join(
            part for part in (current.get("browser"), current.get("host")) if part
        )
        subtitle = f"{place} · browser measurement\n{comparison}"
    title = f"{name}: {current['title']}"
    fig = new_figure()
    header(
        fig,
        title,
        subtitle,
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

    finish(fig, out_dir, f"step-{name}.png")


def race(log_path, out_dir, f32="N6", int8="N7", llamacpp="N9-llamacpp",
         name="race"):
    rows = [
        row for row in load_rows(log_path)
        if is_native(row) and is_124m(row)
    ]
    by_milestone = {row["milestone"]: row for row in rows}

    def pick(milestone):
        row = by_milestone.get(milestone)
        if row is None:
            raise SystemExit(
                f"milestone {milestone!r} not found in {display_path(log_path)}"
            )
        return row

    entries = []
    for row, caption, color in (
        (by_milestone.get("R0-torch"), "PyTorch", GRAY),
        (by_milestone.get("R0-numpy"), "NumPy", GRAY),
        (by_milestone.get(f"{llamacpp}-f32"), "llama.cpp f32", ORANGE),
        (pick(f32), "gpt2-fast f32", BLUE),
        (by_milestone.get(f"{llamacpp}-f16"), "llama.cpp f16", ORANGE),
        (by_milestone.get(f"{llamacpp}-q8_0"), "llama.cpp q8_0", ORANGE),
        (pick(int8), "gpt2-fast int8", BLUE),
    ):
        if row is None:
            continue
        display = (
            f"{caption} ({row['milestone']})" if caption.startswith("gpt2-fast")
            else caption
        )
        entries.append(
            {"label": display, "name": caption, "row": row, "color": color}
        )

    if not entries:
        raise SystemExit(f"no race results in {display_path(log_path)}")

    fig = new_figure()
    labels = [entry["label"] for entry in entries]
    gutter = max(len(text) for text in labels) * 8.5 * 0.62 / 72 + 0.2
    left = min(gutter / 8, 0.30)
    gap = 0.075
    width = (0.955 - left - gap) / 2
    y = list(range(len(entries)))

    for column, (panel_title, field) in enumerate((
        ("Decode, tokens/s", "decode_tok_s"),
        ("Prompt processing, tokens/s", "prefill_tok_s"),
    )):
        ax = fig.add_axes((left + column * (width + gap), 0.17, width, 0.595))
        values = [entry["row"][field] for entry in entries]
        bars = ax.barh(y, values, height=0.6,
                       color=[entry["color"] for entry in entries],
                       edgecolor=TEXT, linewidth=0)
        for bar, entry in zip(bars, entries):
            if entry["row"]["token_match_rate"] < 1.0:
                bar.set_hatch("////")
        for yi, value in zip(y, values):
            ax.annotate(fmt_value(value), (value, yi), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8.5, color=TEXT)
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_ylim(len(entries) - 0.5, -0.5)
        if column == 0:
            ax.set_yticks(y, labels)
        else:
            ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(panel_title, loc="left", fontsize=9.5, color=INK, pad=9)
        style_axis(ax)

    host = next((row["host"] for row in reversed(rows) if row.get("host")), "")
    source = "Source: results/log.jsonl"
    quantized = [
        entry for entry in entries if entry["row"]["token_match_rate"] < 1.0
    ]
    if quantized:
        agreement = ", ".join(
            f"{entry['name']} {entry['row']['token_match_rate'] * 100:.1f}%"
            for entry in quantized
        )
        source = f"{source} · token agreement: {agreement}"
    header(
        fig,
        "GPT-2 124M on 4 CPU cores",
        f"{host} ·\nexact-match rows first, then 16-bit and 8-bit rows",
        source=source,
    )
    finish(fig, out_dir, f"{name}.png")


WEB_STEPS = (
    ("W1", "WebAssembly"),
    ("W2", "+ SIMD"),
    ("W3", "+ threads"),
    ("W4", "WebGPU"),
)


def web(log_path, out_dir):
    by_milestone = {
        row["milestone"]: row
        for row in load_rows(log_path)
        if is_browser(row)
    }
    entries = []
    for milestone, caption in WEB_STEPS:
        row = by_milestone.get(milestone)
        if row is None or row.get("decode_tok_s") is None:
            continue
        if milestone == "W3" and row.get("threads") is not None:
            caption = f"+ threads ({row['threads']})"
        variant = by_milestone.get(f"{milestone}-f32")
        if variant is not None and variant.get("decode_tok_s") is None:
            variant = None
        entries.append((row, f"{milestone} · {caption}", variant))
    if not entries:
        raise SystemExit(f"no browser results in {display_path(log_path)}")

    values = [row["decode_tok_s"] for row, _, _ in entries]
    values += [variant["decode_tok_s"] for _, _, variant in entries if variant]

    fig = new_figure()
    ticks = []
    labels = []
    variant_ticks = []
    for index, (_, text, variant) in enumerate(entries):
        ticks.append(index)
        labels.append(text)
        variant_ticks.append(False)
        if variant is not None:
            ticks.append(index + 0.34)
            labels.append("f32")
            variant_ticks.append(True)
    tick_size = 8.5
    gutter = max(len(text) for text in labels) * tick_size * 0.62 / 72 + 0.2
    left = min(gutter / 8, 0.30)
    ax = fig.add_axes((left, 0.115, 0.955 - left, 0.735))

    for index, (row, _, variant) in enumerate(entries):
        value = row["decode_tok_s"]
        bars = ax.barh([index], [value], height=0.5, color=BLUE,
                       edgecolor=TEXT, linewidth=0)
        if row.get("weights") == "int8":
            bars[0].set_hatch("////")
        ax.annotate(fmt_value(value), (value, index), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=TEXT)
        if variant is not None:
            other = variant["decode_tok_s"]
            ax.barh([index + 0.34], [other], height=0.2, color=BLUE,
                    edgecolor=TEXT, linewidth=0)
            ax.annotate(fmt_value(other), (other, index + 0.34), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7.5, color=TEXT)

    if max(values) / min(values) > 20:
        ax.set_xscale("log")
        ax.set_xlim(min(values) / 2, max(values) * 2.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xlim(0, max(values) * 1.18)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(len(entries) - 0.45, -0.55)
    ax.set_yticks(ticks, labels)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=tick_size)
    for text, variant_tick in zip(ax.get_yticklabels(), variant_ticks):
        if variant_tick:
            text.set_color(FAINT)

    first = entries[0][0]
    place = " · ".join(
        part for part in (first.get("browser"), first.get("host")) if part
    )
    header(
        fig,
        "GPT-2 124M in the browser, decode tokens/s",
        f"{place} · median of session medians",
        source=(
            "Source: results/log.jsonl · browser measurements on the "
            "development laptop, not the 4-core container"
        ),
    )
    finish(fig, out_dir, "web.png")


def sizes(log_path, out_dir, stage="N8", name="sizes"):
    rows = [row for row in load_rows(log_path) if is_native(row)]
    by_milestone = {row["milestone"]: row for row in rows}
    fast_124m = ("N6", "N7") if stage == "N8" else (stage, f"{stage}-int8")
    groups = [
        ("124M", "124M", "R0-torch", fast_124m[0], fast_124m[1]),
        ("355M (medium)", "355M", "N8-gpt2-medium-torch",
         f"{stage}-gpt2-medium-f32", f"{stage}-gpt2-medium-int8"),
        ("774M (large)", "774M", "N8-gpt2-large-torch",
         f"{stage}-gpt2-large-f32", f"{stage}-gpt2-large-int8"),
        ("1.5B (xl)", "1.5B", "N8-gpt2-xl-torch",
         f"{stage}-gpt2-xl-f32", f"{stage}-gpt2-xl-int8"),
    ]
    engines = [
        ("PyTorch", GRAY, False),
        ("gpt2-fast f32", BLUE, False),
        ("gpt2-fast int8", BLUE, True),
    ]
    offsets = (-0.27, 0.0, 0.27)
    bar_width = 0.24

    values = []
    for _, _, torch_name, f32_name, int8_name in groups:
        for milestone in (torch_name, f32_name, int8_name):
            row = by_milestone.get(milestone)
            value = row.get("decode_tok_s") if row else None
            if value is not None:
                values.append(value)
    if not values:
        raise SystemExit(f"no size results in {display_path(log_path)}")

    fig = new_figure()
    ax = fig.add_axes((0.075, 0.13, 0.90, 0.62))
    top = max(values) * 1.18

    for index, (_, _, torch_name, f32_name, int8_name) in enumerate(groups):
        names = (torch_name, f32_name, int8_name)
        for offset, milestone, (_, color, hatched) in zip(offsets, names, engines):
            row = by_milestone.get(milestone)
            value = row.get("decode_tok_s") if row else None
            x = index + offset
            if value is None:
                text = "not measured"
                if row is not None and row.get("status") == "oom":
                    text = "did not fit in 8 GB"
                ax.text(x, top * 0.02, text, rotation=90, ha="center",
                        va="bottom", fontsize=6.5, color=FAINT)
                continue
            bars = ax.bar(x, value, width=bar_width, color=color,
                          edgecolor=TEXT, linewidth=0, zorder=3)
            if hatched:
                bars[0].set_hatch("////")
            ax.annotate(fmt_value(value), (x, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=TEXT)

    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, top)
    ax.set_xticks(
        range(len(groups)), [group[0] for group in groups]
    )
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.tick_params(axis="x", labelsize=9.5)

    handles = [
        Patch(facecolor=color, edgecolor=TEXT, linewidth=0, hatch="////"
              if hatched else None, label=label)
        for label, color, hatched in engines
    ]
    legend = ax.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.13),
        ncol=3, frameon=False, fontsize=8, handlelength=1.2,
        handleheight=0.9, columnspacing=1.6, handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    host = next((row["host"] for row in reversed(rows) if row.get("host")), "")
    subtitle = "median of session medians"
    if host:
        subtitle = f"{host} · {subtitle}"

    source = "Source: results/log.jsonl"
    perplexity = []
    for _, short, _, f32_name, int8_name in groups:
        f32_row = by_milestone.get(f32_name)
        int8_row = by_milestone.get(int8_name)
        if f32_row is None or int8_row is None:
            continue
        f32_ppl = f32_row.get("perplexity")
        int8_ppl = int8_row.get("perplexity")
        if f32_ppl is None or int8_ppl is None:
            continue
        perplexity.append(
            f"{short} {float(f32_ppl):.2f}→{float(int8_ppl):.2f}"
        )
    if perplexity:
        source = f"{source} · perplexity f32 → int8: " + ", ".join(perplexity)

    title = "GPT-2 at four sizes on 4 CPU cores, decode tokens/s"
    if stage != "N8":
        title = f"{title} ({stage})"
    header(
        fig,
        title,
        subtitle,
        source=source,
    )
    finish(fig, out_dir, f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description="Chart GPT-2 decode results.")
    parser.add_argument("--log", type=Path, default=LOG,
                        help="results log to read (default: results/log.jsonl)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="directory for the rendered charts (default: results/charts)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("progress", help="one bar per milestone")
    step_parser = sub.add_parser("step", help="before/after card for one milestone")
    step_parser.add_argument("milestone")
    race_parser = sub.add_parser(
        "race", help="gpt2-fast against llama.cpp and the references"
    )
    race_parser.add_argument("--f32", default="N6", metavar="MILESTONE",
                             help="gpt2-fast f32 row (default: N6)")
    race_parser.add_argument("--int8", default="N7", metavar="MILESTONE",
                             help="gpt2-fast int8 row (default: N7)")
    race_parser.add_argument("--llamacpp", default="N9-llamacpp",
                             metavar="PREFIX",
                             help="llama.cpp milestone prefix (default: N9-llamacpp)")
    race_parser.add_argument("--name", default="race", metavar="STEM",
                             help="output file stem (default: race)")
    sub.add_parser("web", help="browser decode speed for the W1 to W4 steps")
    sizes_parser = sub.add_parser("sizes",
                                  help="decode speed at four model sizes")
    sizes_parser.add_argument("--stage", default="N8", metavar="PREFIX",
                              help="milestone prefix for gpt2-fast rows "
                                   "(default: N8)")
    sizes_parser.add_argument("--name", default="sizes", metavar="STEM",
                              help="output file stem (default: sizes)")
    args = parser.parse_args()

    if args.command == "progress":
        progress(args.log, args.out)
    elif args.command == "step":
        step(args.milestone, args.log, args.out)
    elif args.command == "sizes":
        sizes(args.log, args.out, stage=args.stage, name=args.name)
    elif args.command == "web":
        web(args.log, args.out)
    else:
        race(args.log, args.out, f32=args.f32, int8=args.int8,
             llamacpp=args.llamacpp, name=args.name)


if __name__ == "__main__":
    main()
