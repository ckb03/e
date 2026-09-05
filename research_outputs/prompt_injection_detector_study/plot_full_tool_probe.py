"""Plot full Tool-result role-probe traces for matched clean/attack pages."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import torch
from analyze_probe_detector import load_docs, smooth

OUT = Path(__file__).resolve().parent
PAIRS = [(6, 8), (12, 13)]
COLORS = {"user": "#d62728", "cot": "#ff7f0e", "tool": "#1f77b4"}


def bins(values: torch.Tensor, count: int, maximum: bool = False) -> list[float]:
    """Reduce a long token trace to equally spaced display bins."""
    output = []
    for index in range(count):
        start = index * len(values) // count
        stop = max(start + 1, (index + 1) * len(values) // count)
        chunk = values[start:stop]
        output.append(float(chunk.max() if maximum else chunk.mean()))
    return output


def polyline(
    values: list[float],
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
    color: str,
    opacity: float = 1.0,
    dash: str | None = None,
) -> str:
    scale = max(1e-9, high - low)
    points = " ".join(
        f"{x + i * width / max(1, len(values) - 1):.1f},"
        f"{y + height - (min(high, max(low, value)) - low) * height / scale:.1f}"
        for i, value in enumerate(values)
    )
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" opacity="{opacity}"{dashed}/>'
    )


def nms_peaks(
    score: torch.Tensor,
    region: torch.Tensor,
    count: int = 5,
    radius: int = 256,
) -> list[int]:
    """Return the strongest separated peaks outside the known injected span."""
    available = region != 2
    injection = torch.where(region == 2)[0]
    if len(injection):
        guard_start = max(0, int(injection[0]) - 512)
        guard_stop = min(len(region), int(injection[-1]) + 513)
        available[guard_start:guard_stop] = False
    selected = []
    for _ in range(count):
        masked = score.masked_fill(~available, float("-inf"))
        index = int(masked.argmax())
        if not torch.isfinite(masked[index]):
            break
        selected.append(index)
        available[max(0, index - radius) : min(len(score), index + radius + 1)] = False
    return selected


def shade_injection(
    parts: list[str],
    doc: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    injection = torch.where(doc["region"] == 2)[0]
    if not len(injection):
        return
    start = float(injection[0]) / max(1, len(doc["region"]) - 1)
    stop = float(injection[-1] + 1) / max(1, len(doc["region"]) - 1)
    visible_width = max(3.0, (stop - start) * width)
    parts.append(
        f'<rect x="{x + start * width:.1f}" y="{y:.1f}" width="{visible_width:.1f}" '
        f'height="{height:.1f}" fill="#ffe699" opacity="0.75"/>'
    )


def axes(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
    token_count: int,
) -> None:
    for fraction in (0.0, 0.5, 1.0):
        yy = y + height - (fraction - low) * height / (high - low)
        if y <= yy <= y + height:
            parts.append(
                f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" '
                'stroke="#dddddd"/>'
            )
            parts.append(
                f'<text x="{x - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">'
                f"{fraction:.1f}</text>"
            )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="#555"/>'
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = x + fraction * width
        parts.append(
            f'<line x1="{xx:.1f}" y1="{y + height}" x2="{xx:.1f}" '
            f'y2="{y + height + 4}" stroke="#555"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{y + height + 17}" text-anchor="middle" font-size="10">'
            f"{round(fraction * token_count):,}</text>"
        )


def plot_pair(
    clean: dict[str, Any], attack: dict[str, Any], path: Path
) -> list[dict[str, Any]]:
    width, height = 1400, 930
    left, plot_width, panel_height = 105, 1245, 145
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:system-ui,sans-serif;fill:#222}</style>",
        (
            f'<text x="40" y="34" font-size="21" font-weight="bold">Layer-18 probe over the full Tool result: '
            f"clean case {clean['case_id']} vs attack case {attack['case_id']}</text>"
        ),
        (
            '<text x="40" y="58" font-size="13">Every x-axis covers the complete serialized Tool result. '
            "Yellow is the exact injected span; numbered markers are high-scoring non-injection peaks.</text>"
        ),
    ]
    legend = [
        ("User probability, mean-64", COLORS["user"]),
        ("CoT probability, mean-64", COLORS["cot"]),
        ("Tool probability, mean-64", COLORS["tool"]),
        ("raw p(User)+p(CoT), max in display bin", "#a0a0a0"),
        ("p(User)+p(CoT), mean-64", "#7b3294"),
        ("local contrast: mean-64 minus mean-512", "#008837"),
    ]
    for i, (label, color) in enumerate(legend):
        row, column = divmod(i, 3)
        x = 105 + column * 410
        y = 78 + row * 20
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{x + 30}" y="{y + 4}" font-size="11">{html.escape(label)}</text>'
        )

    rows: list[dict[str, Any]] = []
    for doc_number, doc in enumerate((clean, attack)):
        logits = doc["logits"]
        probs = logits.softmax(-1)
        puc = probs[:, 1] + probs[:, 2]
        mean64 = smooth(puc, 64)
        contrast = mean64 - smooth(puc, 512)
        peaks = nms_peaks(contrast, doc["region"].clone())
        label = "CLEAN" if doc is clean else "ATTACK"
        base_y = 125 + doc_number * 395

        parts.append(
            f'<text x="40" y="{base_y - 9}" font-size="16" font-weight="bold">{label}: case '
            f"{doc['case_id']}, {html.escape(doc['variant'])} ({len(logits):,} tokens)</text>"
        )
        role_y = base_y
        detector_y = base_y + 190
        for panel_y, panel_label, low, high in (
            (role_y, "smoothed role probabilities", 0.0, 1.0),
            (detector_y, "detector views", -0.5, 1.0),
        ):
            shade_injection(parts, doc, left, panel_y, plot_width, panel_height)
            axes(parts, left, panel_y, plot_width, panel_height, low, high, len(logits))
            parts.append(
                f'<text x="26" y="{panel_y + panel_height / 2:.1f}" font-size="12" '
                f'text-anchor="middle" transform="rotate(-90 26 {panel_y + panel_height / 2:.1f})">'
                f"{panel_label}</text>"
            )

        bin_count = 800
        for role, index in (("user", 1), ("cot", 2), ("tool", 4)):
            values = bins(smooth(probs[:, index], 64), bin_count)
            parts.append(
                polyline(
                    values,
                    left,
                    role_y,
                    plot_width,
                    panel_height,
                    0.0,
                    1.0,
                    COLORS[role],
                )
            )
        parts.append(
            polyline(
                bins(puc, bin_count, maximum=True),
                left,
                detector_y,
                plot_width,
                panel_height,
                -0.5,
                1.0,
                "#a0a0a0",
                opacity=0.65,
            )
        )
        parts.append(
            polyline(
                bins(mean64, bin_count),
                left,
                detector_y,
                plot_width,
                panel_height,
                -0.5,
                1.0,
                "#7b3294",
            )
        )
        parts.append(
            polyline(
                bins(contrast, bin_count),
                left,
                detector_y,
                plot_width,
                panel_height,
                -0.5,
                1.0,
                "#008837",
            )
        )
        zero_y = detector_y + panel_height * (1.0 / 1.5)
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" '
            'stroke="#777" stroke-dasharray="3,5"/>'
        )

        for peak_number, index in enumerate(peaks, start=1):
            x = left + index * plot_width / max(1, len(logits) - 1)
            value = float(contrast[index])
            y = (
                detector_y
                + panel_height
                - (min(1.0, max(-0.5, value)) + 0.5) * panel_height / 1.5
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#fff" stroke="#111"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10">{peak_number}</text>'
            )
            token_text = "".join(
                doc["token_texts"][max(0, index - 28) : min(len(logits), index + 29)]
            )
            snippet = " ".join(token_text.split()).replace("|", "¦")
            role_values = probs[index]
            rows.append(
                {
                    "document": label.lower(),
                    "case_id": doc["case_id"],
                    "peak": peak_number,
                    "token_index": index,
                    "contrast_64_512": value,
                    "p_user": float(role_values[1]),
                    "p_cot": float(role_values[2]),
                    "p_tool": float(role_values[4]),
                    "snippet": snippet[:360],
                }
            )

    parts.append(
        '<text x="727" y="915" text-anchor="middle" font-size="12">token index in complete Tool result</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return rows


def write_report(all_rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in all_rows:
        table_rows.append(
            "| {document} {case_id} | {peak} | {token_index:,} | {contrast_64_512:.3f} | "
            "{p_user:.3f} | {p_cot:.3f} | {p_tool:.3f} | `{snippet}` |".format(**row)
        )
    report = (
        """# Full-document role-probe view

These plots answer why a detector can localize injected text reasonably well at the **token level** yet still false-positive on many **whole documents**. They use saved layer-18 logits; no new inference or steering is involved.

## All-five-role matched views

The enhanced full-document plots show all probe roles and explicitly shade the simulator's exact injected token span yellow. The clean member of each pair has no yellow region because it contains no injection.

![All-role CoT-forgery case 8 compared with its clean page](full-tool-all-roles-case-008-vs-clean-006.svg)

![All-role base-injection case 13 compared with its clean page](full-tool-all-roles-case-013-vs-clean-012.svg)

### Injection-centered zooms

Because an approximately 100-token injection becomes a very narrow stripe on a 25k–28k-token page, these views show the exact yellow span with 384 tokens of context on each side. Their upper panels show mean-64 probabilities; lower panels show each role's local contrast, `mean64(p_role) - mean512(p_role)`.

![All-role zoom around the CoT-forgery injection in case 8](injection-all-roles-case-008.svg)

![All-role zoom around the base injection in case 13](injection-all-roles-case-013.svg)

How to read the enhanced views:

- Purple is System.
- Red is User.
- Orange is CoT.
- Green is Assistant.
- Blue is Tool.
- Yellow is simulator ground truth used only for offline comparison; it is not available to the detector.

## Original User/CoT detector views

![CoT-forgery case 8 compared with its clean page](full-tool-case-008-vs-clean-006.svg)

![Base-injection case 13 compared with its clean page](full-tool-case-013-vs-clean-012.svg)

How to read them:

- Red/orange/blue are 64-token averages of User/CoT/Tool probe probabilities.
- Gray is the maximum raw `p(User)+p(CoT)` in each display bin. It repeatedly becomes large on ordinary page markup even in the clean document. That is why a rule based on the largest token in a long page false-positives badly.
- Purple is the 64-token mean. It removes isolated spikes, but benign structured passages can still remain elevated for dozens of tokens.
- Green is the local detector score `mean64(p(User)+p(CoT)) - mean512(p(User)+p(CoT))`. It asks whether a short span is more user/CoT-like than its local background. Numbered circles mark its five strongest **non-injection** peaks.
- Yellow is the simulator's exact injected token span. On these pages it is close to the end and occupies a tiny fraction of 25k–28k tokens, so the full-page view makes the multiple-comparisons problem apparent.

## Strong benign peaks shown in the plots

| document | marker | token | contrast | p(user) | p(CoT) | p(tool) | surrounding decoded text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
        + "\n".join(table_rows)
        + """

## Direct answer

There are not necessarily many false-positive **tokens**. There are many opportunities for at least one false-positive **span**: each page contains roughly 25,000–28,000 tokens, including HTML attributes, links, templates, citations, quoted prose, and script/metadata. Some benign spans genuinely look User-like or CoT-like to a role probe because the probe measures linguistic role resemblance, not malicious intent. Taking a maximum over the full page turns those uncommon local errors into a high document-level false-positive rate.

The plots support using the probe as a high-recall candidate generator, not as the final verdict. A practical selector should use sustained/segmented scores and local contrast, return a small top-K set with non-maximum suppression, then let a second-stage checker judge whether each candidate is an instruction that conflicts with the trusted task.
"""
    )
    (OUT / "FULL_DOCUMENT_PROBE_VIEW.md").write_text(report, encoding="utf-8")


def main() -> None:
    docs = [doc for doc in load_docs(18) if doc["dataset"] == "layer"]
    by_id = {doc["case_id"]: doc for doc in docs}
    all_rows = []
    for clean_id, attack_id in PAIRS:
        all_rows.extend(
            plot_pair(
                by_id[clean_id],
                by_id[attack_id],
                OUT / f"full-tool-case-{attack_id:03d}-vs-clean-{clean_id:03d}.svg",
            )
        )
    with (OUT / "full_tool_benign_peaks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_report(all_rows)
    print(OUT / "FULL_DOCUMENT_PROBE_VIEW.md")


if __name__ == "__main__":
    main()
