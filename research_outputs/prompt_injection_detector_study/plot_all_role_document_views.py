"""Plot all five role-probe channels over full Tool results and injection zooms."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import torch
from analyze_probe_detector import load_docs, smooth
from plot_full_tool_probe import bins, polyline

OUT = Path(__file__).resolve().parent
PAIRS = ((6, 8), (12, 13))
ROLES = ("system", "user", "cot", "assistant", "tool")
COLORS = {
    "system": "#9467bd",
    "user": "#d62728",
    "cot": "#ff7f0e",
    "assistant": "#2ca02c",
    "tool": "#1f77b4",
}


def role_views(doc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = doc["logits"].softmax(-1)
    mean64 = torch.stack(
        [smooth(probabilities[:, index], 64) for index in range(5)], dim=1
    )
    mean512 = torch.stack(
        [smooth(probabilities[:, index], 512) for index in range(5)], dim=1
    )
    return mean64, mean64 - mean512


def shade(
    parts: list[str],
    region: torch.Tensor,
    range_start: int,
    range_stop: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    injection = torch.where(region == 2)[0]
    if not len(injection):
        return
    start = max(range_start, int(injection[0]))
    stop = min(range_stop, int(injection[-1]) + 1)
    if stop <= start:
        return
    denominator = max(1, range_stop - range_start)
    xx = x + (start - range_start) * width / denominator
    visible_width = max(3.0, (stop - start) * width / denominator)
    parts.append(
        f'<rect x="{xx:.1f}" y="{y:.1f}" width="{visible_width:.1f}" '
        f'height="{height:.1f}" fill="#ffe066" opacity="0.82"/>'
    )


def axes(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
    range_start: int,
    range_stop: int,
) -> None:
    ticks = (0.0, 0.5, 1.0) if low == 0.0 else (-0.5, 0.0, 0.5)
    for tick in ticks:
        if low <= tick <= high:
            yy = y + height - (tick - low) * height / (high - low)
            parts.append(
                f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" '
                'stroke="#dddddd"/>'
            )
            parts.append(
                f'<text x="{x - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                f'font-size="11">{tick:.1f}</text>'
            )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'fill="none" stroke="#555"/>'
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = x + fraction * width
        token = round(range_start + fraction * (range_stop - range_start))
        parts.append(
            f'<text x="{xx:.1f}" y="{y + height + 17:.1f}" '
            f'text-anchor="middle" font-size="10">{token:,}</text>'
        )


def legend(parts: list[str], y: float = 75) -> None:
    for index, role in enumerate(ROLES):
        x = 415 + index * 180
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" '
            f'stroke="{COLORS[role]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{x + 35}" y="{y + 5}" font-size="13">'
            f'{html.escape(role.title())}</text>'
        )


def add_role_lines(
    parts: list[str],
    values: torch.Tensor,
    start: int,
    stop: int,
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
    bin_count: int,
) -> None:
    for index, role in enumerate(ROLES):
        display = bins(values[start:stop, index], bin_count)
        parts.append(
            polyline(
                display,
                x,
                y,
                width,
                height,
                low,
                high,
                COLORS[role],
                opacity=0.9,
            )
        )


def full_pair(clean: dict[str, Any], attack: dict[str, Any], path: Path) -> None:
    width, height = 1420, 980
    left, plot_width, panel_height = 105, 1260, 155
    parts = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}</style>',
        (f'<text x="40" y="34" font-size="21" font-weight="bold">'
        f'All five roles over the full Tool result: clean {clean["case_id"]} '
        f'vs attack {attack["case_id"]}</text>'),
        ('<text x="40" y="58" font-size="13">Yellow is the exact simulator '
        'injection. Lower panels show mean-64 minus mean-512 for each role.</text>'),
    ]
    legend(parts)
    for row, doc in enumerate((clean, attack)):
        mean64, contrast = role_views(doc)
        base_y = 125 + row * 420
        label = "CLEAN" if row == 0 else "ATTACK"
        parts.append(
            f'<text x="40" y="{base_y - 12}" font-size="16" font-weight="bold">'
            f'{label}: case {doc["case_id"]}, {html.escape(doc["variant"])} '
            f'({len(doc["region"]):,} tokens)</text>'
        )
        for panel_y, values, panel_label, low, high in (
            (base_y, mean64, "mean-64 role probability", 0.0, 1.0),
            (base_y + 205, contrast, "local contrast by role", -0.5, 0.8),
        ):
            shade(
                parts,
                doc["region"],
                0,
                len(doc["region"]),
                left,
                panel_y,
                plot_width,
                panel_height,
            )
            axes(
                parts,
                left,
                panel_y,
                plot_width,
                panel_height,
                low,
                high,
                0,
                len(doc["region"]),
            )
            add_role_lines(
                parts,
                values,
                0,
                len(doc["region"]),
                left,
                panel_y,
                plot_width,
                panel_height,
                low,
                high,
                900,
            )
            parts.append(
                f'<text x="25" y="{panel_y + panel_height / 2:.1f}" '
                f'text-anchor="middle" font-size="12" '
                f'transform="rotate(-90 25 {panel_y + panel_height / 2:.1f})">'
                f'{html.escape(panel_label)}</text>'
            )
    parts.append(
        '<text x="735" y="965" text-anchor="middle" font-size="12">'
        'absolute token index in complete Tool result</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def injection_zoom(doc: dict[str, Any], path: Path, context: int = 384) -> None:
    injection = torch.where(doc["region"] == 2)[0]
    if not len(injection):
        raise ValueError(f'case {doc["case_id"]} has no injection')
    start = max(0, int(injection[0]) - context)
    stop = min(len(doc["region"]), int(injection[-1]) + context + 1)
    mean64, contrast = role_views(doc)

    width, height = 1420, 590
    left, plot_width, panel_height = 105, 1260, 175
    parts = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}</style>',
        (f'<text x="40" y="34" font-size="21" font-weight="bold">'
        f'Injection-centered all-role zoom: case {doc["case_id"]}, '
        f'{html.escape(doc["variant"])}</text>'),
        (f'<text x="40" y="58" font-size="13">Showing tokens {start:,}–{stop:,}; '
        'yellow is the exact injected span, with local context on both sides.</text>'),
    ]
    legend(parts)
    for row, (values, label, low, high) in enumerate(
        (
            (mean64, "mean-64 role probability", 0.0, 1.0),
            (contrast, "mean-64 minus mean-512", -0.5, 0.8),
        )
    ):
        y = 120 + row * 230
        shade(parts, doc["region"], start, stop, left, y, plot_width, panel_height)
        axes(parts, left, y, plot_width, panel_height, low, high, start, stop)
        add_role_lines(
            parts,
            values,
            start,
            stop,
            left,
            y,
            plot_width,
            panel_height,
            low,
            high,
            min(900, stop - start),
        )
        parts.append(
            f'<text x="25" y="{y + panel_height / 2:.1f}" text-anchor="middle" '
            f'font-size="12" transform="rotate(-90 25 {y + panel_height / 2:.1f})">'
            f'{html.escape(label)}</text>'
        )
    parts.append(
        '<text x="735" y="580" text-anchor="middle" font-size="12">'
        'absolute token index</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    docs = [doc for doc in load_docs(18) if doc["dataset"] == "layer"]
    by_id = {doc["case_id"]: doc for doc in docs}
    for clean_id, attack_id in PAIRS:
        full_pair(
            by_id[clean_id],
            by_id[attack_id],
            OUT / f"full-tool-all-roles-case-{attack_id:03d}-vs-clean-{clean_id:03d}.svg",
        )
        injection_zoom(
            by_id[attack_id],
            OUT / f"injection-all-roles-case-{attack_id:03d}.svg",
        )
    print(OUT / "full-tool-all-roles-case-013-vs-clean-012.svg")


if __name__ == "__main__":
    main()
