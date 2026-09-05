"""Offline prompt-injection detector study using saved role-probe activations."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ROLES = ["system", "user", "cot", "assistant", "tool"]
ROLE_COLORS = {
    "user": "#d62728",
    "cot": "#ff7f0e",
    "tool": "#1f77b4",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cache_layer8_logits() -> Path:
    out = OUT / "token_logits" / "layer-08"
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(
        ROOT
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    weight = state["probe_weight"][8].float()
    bias = state["probe_bias"][8].float()
    sources = [
        (
            "clean",
            ROOT
            / "research_outputs/phase3_steering_v2/tool_pre_mlp/clean/shards/layer-08",
        ),
        (
            "layer",
            ROOT
            / "research_outputs/phase3_steering_v2/tool_pre_mlp/layer/shards/layer-08",
        ),
    ]
    for prefix, directory in sources:
        for source in sorted(directory.glob("case-*.pt")):
            shard = torch.load(source, map_location="cpu", weights_only=False)
            target = (
                out / f"{prefix}-case-{int(shard['case_id']):03d}-{shard['variant']}.pt"
            )
            if target.exists():
                continue
            activations = shard["activations"]
            logits = torch.empty((len(activations), 5), dtype=torch.float32)
            for start in range(0, len(activations), 2048):
                stop = min(start + 2048, len(activations))
                logits[start:stop] = activations[start:stop].float() @ weight.T + bias
            payload = {
                "schema_version": 1,
                "source": str(source.relative_to(ROOT)),
                "case_id": shard["case_id"],
                "variant": shard["variant"],
                "dataset": prefix,
                "layer": 8,
                "token_ids": shard["token_ids"],
                "token_texts": shard["token_texts"],
                "region_codes": shard["region_codes"],
                "injection_token_range": shard.get("injection_token_range"),
                "logits": logits,
            }
            torch.save(payload, target)
    return out


def load_docs(layer: int) -> list[dict[str, Any]]:
    if layer == 8:
        directory = cache_layer8_logits()
    else:
        directory = (
            ROOT / "research_outputs/phase3_steering_v2/calibration/token_diagnostics"
        )
    docs = []
    for path in sorted(directory.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=False)
        if "logits" not in item or len(item["logits"]) < 100:
            continue
        name = path.name
        dataset = "clean" if name.startswith("clean-") else "layer"
        docs.append(
            {
                "path": path,
                "dataset": dataset,
                "case_id": int(item["case_id"]),
                "variant": item["variant"],
                "logits": item["logits"].float(),
                "token_texts": item.get("token_texts", []),
                "region": item["region_codes"].long(),
            }
        )
    return docs


def smooth(values: torch.Tensor, window: int) -> torch.Tensor:
    if window == 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = F.pad(values[None, None], (left, right), mode="replicate")
    return F.avg_pool1d(padded, window, stride=1)[0, 0]


def base_signal(logits: torch.Tensor, name: str) -> torch.Tensor:
    probs = logits.softmax(-1)
    margins = logits[:, :4] - logits[:, [4]]
    if name == "p_user+p_cot":
        return probs[:, 1] + probs[:, 2]
    if name == "max(user,cot)-tool":
        return margins[:, 1:3].max(-1).values
    if name == "max(wrong)-tool":
        return margins.max(-1).values
    if name == "document-robust user/cot z":
        pair = margins[:, 1:3]
        center = pair.median(0).values
        scale = (1.4826 * (pair - center).abs().median(0).values).clamp_min(0.1)
        return ((pair - center) / scale).max(-1).values
    raise ValueError(name)


CANDIDATES = [
    ("pUC token", "p_user+p_cot", 1, None),
    ("pUC mean-32", "p_user+p_cot", 32, None),
    ("pUC mean-64", "p_user+p_cot", 64, None),
    ("pUC mean-128", "p_user+p_cot", 128, None),
    ("UC-margin mean-64", "max(user,cot)-tool", 64, None),
    ("all-role-margin mean-64", "max(wrong)-tool", 64, None),
    ("robust-UC mean-32", "document-robust user/cot z", 32, None),
    ("robust-UC mean-64", "document-robust user/cot z", 64, None),
    ("pUC contrast 64-512", "p_user+p_cot", 64, 512),
    ("pUC contrast 128-1024", "p_user+p_cot", 128, 1024),
]


def score_map(
    logits: torch.Tensor, candidate: tuple[str, str, int, int | None]
) -> torch.Tensor:
    _, signal_name, short, long = candidate
    values = base_signal(logits, signal_name)
    score = smooth(values, short)
    return score if long is None else score - smooth(values, long)


def roc(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[float, list[tuple[float, float]]]:
    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    tpr = torch.cumsum(y, 0) / positives
    fpr = torch.cumsum(1 - y, 0) / negatives
    auc = float(
        torch.trapz(torch.cat([torch.zeros(1), tpr]), torch.cat([torch.zeros(1), fpr]))
    )
    indexes = torch.linspace(0, len(y) - 1, 201).long()
    points = [(0.0, 0.0)] + [(float(fpr[i]), float(tpr[i])) for i in indexes]
    return auc, points


def threshold_at_budget(scores: list[float], budget: float) -> float:
    allowed = max(1, round(budget * len(scores)))
    ordered = sorted(scores, reverse=True)
    return ordered[min(allowed, len(ordered) - 1)]


def evaluate_layer(
    layer: int,
    docs: list[dict[str, Any]],
    outcomes: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = [
        doc for doc in docs if doc["dataset"] == "clean" and doc["case_id"] < 20
    ]
    heldout_clean = [
        doc
        for doc in docs
        if (doc["dataset"] == "clean" and doc["case_id"] >= 20)
        or (doc["dataset"] == "layer" and doc["variant"] == "clean")
    ]
    attacks = [
        doc
        for doc in docs
        if doc["dataset"] == "layer"
        and doc["variant"] != "clean"
        and bool((doc["region"] == 2).any())
    ]
    results = []
    selected_detail: dict[str, Any] = {}
    for candidate in CANDIDATES:
        score_by_path = {
            doc["path"]: score_map(doc["logits"], candidate) for doc in docs
        }
        token_scores = []
        token_labels = []
        macro_auc = []
        localization_recall = []
        localization_precision = []
        top_hit = []
        for doc in attacks:
            region = doc["region"]
            keep = (region == 0) | (region == 2)
            scores = score_by_path[doc["path"]]
            labels = region[keep] == 2
            token_scores.append(scores[keep])
            token_labels.append(labels)
            macro_auc.append(roc(scores[keep], labels)[0])
            center = int(scores.argmax())
            start = max(0, center - 64)
            stop = min(len(scores), start + 128)
            start = max(0, stop - 128)
            overlap = int((region[start:stop] == 2).sum())
            injection_n = int((region == 2).sum())
            localization_recall.append(overlap / injection_n)
            localization_precision.append(overlap / max(1, stop - start))
            top_hit.append(int(overlap > 0))
        global_auc, roc_points = roc(torch.cat(token_scores), torch.cat(token_labels))
        cal_scores = [float(score_by_path[doc["path"]].max()) for doc in calibration]
        neg_scores = [float(score_by_path[doc["path"]].max()) for doc in heldout_clean]
        attack_scores = [float(score_by_path[doc["path"]].max()) for doc in attacks]
        tradeoffs = []
        for budget in (0.05, 0.10, 0.20):
            threshold = threshold_at_budget(cal_scores, budget)
            flagged = [score > threshold for score in attack_scores]
            negative_flagged = [score > threshold for score in neg_scores]
            base_mask = [doc["variant"] == "base-injection" for doc in attacks]
            cot_mask = [doc["variant"] == "cot-forgery-injection" for doc in attacks]
            success_mask = [
                outcomes[doc["case_id"]]["attack_success"] for doc in attacks
            ]

            def rate(values: list[bool], mask: list[bool]) -> float:
                chosen = [
                    value
                    for value, include in zip(values, mask, strict=True)
                    if include
                ]
                return sum(chosen) / len(chosen)

            tradeoffs.append(
                {
                    "calibration_budget": budget,
                    "threshold": threshold,
                    "calibration_fpr": sum(score > threshold for score in cal_scores)
                    / len(cal_scores),
                    "heldout_clean_fpr": sum(negative_flagged) / len(negative_flagged),
                    "attack_recall": sum(flagged) / len(flagged),
                    "base_recall": rate(flagged, base_mask),
                    "cot_recall": rate(flagged, cot_mask),
                    "baseline_success_recall": rate(flagged, success_mask),
                }
            )
        result = {
            "layer": layer,
            "candidate": candidate[0],
            "global_token_auroc": global_auc,
            "macro_case_token_auroc": sum(macro_auc) / len(macro_auc),
            "top128_any_overlap_rate": sum(top_hit) / len(top_hit),
            "top128_mean_injection_recall": sum(localization_recall)
            / len(localization_recall),
            "top128_mean_precision": sum(localization_precision)
            / len(localization_precision),
            "tradeoffs": tradeoffs,
        }
        results.append(result)
        selected_detail[candidate[0]] = {
            "roc": roc_points,
            "calibration": dict(
                zip([doc["case_id"] for doc in calibration], cal_scores, strict=True)
            ),
            "heldout_clean": neg_scores,
            "attacks": [
                {
                    "case_id": doc["case_id"],
                    "variant": doc["variant"],
                    "baseline_success": bool(
                        outcomes[doc["case_id"]]["attack_success"]
                    ),
                    "score": score,
                }
                for doc, score in zip(attacks, attack_scores, strict=True)
            ],
        }
    return results, {
        "calibration_docs": len(calibration),
        "heldout_clean_docs": len(heldout_clean),
        "attack_docs": len(attacks),
        "details": selected_detail,
    }


def svg_roc(results: list[dict[str, Any]], details: dict[str, Any], path: Path) -> None:
    chosen = ["pUC token", "pUC mean-64", "pUC mean-128", "robust-UC mean-64"]
    colors = ["#777", "#1f77b4", "#d62728", "#2ca02c"]
    width, height, pad = 720, 520, 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{pad}" stroke="#bbb" stroke-dasharray="5,5"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{pad}" y2="{pad}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle">False-positive rate: ordinary page tokens</text>',
        f'<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle">Recall: injection tokens</text>',
        '<text x="60" y="30" font-size="18" font-weight="bold">Layer-8 token ROC</text>',
    ]
    for offset, (name, color) in enumerate(zip(chosen, colors, strict=True)):
        points = details[name]["roc"]
        coords = " ".join(
            f"{pad + x * (width - 2 * pad):.1f},{height - pad - y * (height - 2 * pad):.1f}"
            for x, y in points
        )
        auc = next(
            row["global_token_auroc"] for row in results if row["candidate"] == name
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{width - 310}" y="{42 + offset * 22}" fill="{color}" font-size="13">{html.escape(name)} (AUROC {auc:.3f})</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_trace(doc: dict[str, Any], path: Path) -> None:
    region = doc["region"]
    injection = torch.where(region == 2)[0]
    if not len(injection):
        return
    start = max(0, int(injection[0]) - 96)
    stop = min(len(region), int(injection[-1]) + 97)
    probs = doc["logits"].softmax(-1)[start:stop]
    width, height, pad = 1000, 300, 50
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{pad + (int(injection[0]) - start) * plot_w / max(1, stop - start - 1):.1f}" y="{pad}" width="{(int(injection[-1]) - int(injection[0]) + 1) * plot_w / max(1, stop - start - 1):.1f}" height="{plot_h}" fill="#fff2cc"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{pad}" y2="{pad}" stroke="black"/>',
        f'<text x="{pad}" y="25" font-size="17" font-weight="bold">Case {doc["case_id"]}: {html.escape(doc["variant"])} (yellow = injection)</text>',
    ]
    for role, index in [("user", 1), ("cot", 2), ("tool", 4)]:
        values = smooth(probs[:, index], 8)
        coords = " ".join(
            f"{pad + i * plot_w / max(1, len(values) - 1):.1f},{height - pad - float(value) * plot_h:.1f}"
            for i, value in enumerate(values)
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{ROLE_COLORS[role]}" stroke-width="1.7" opacity="0.9"/>'
        )
    for i, role in enumerate(("user", "cot", "tool")):
        parts.append(
            f'<text x="{width - 220 + i * 68}" y="25" fill="{ROLE_COLORS[role]}" font-size="13">{role}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_token_views(
    docs: list[dict[str, Any]], outcomes: dict[int, dict[str, Any]]
) -> None:
    attacks = [
        doc for doc in docs if doc["dataset"] == "layer" and doc["variant"] != "clean"
    ]
    with (OUT / "token_logit_excerpts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "variant",
                "baseline_success",
                "token_index",
                "region",
                "token",
                *[f"logit_{r}" for r in ROLES],
                *[f"p_{r}" for r in ROLES],
            ]
        )
        for doc in attacks:
            inj = torch.where(doc["region"] == 2)[0]
            if not len(inj):
                continue
            start = max(0, int(inj[0]) - 64)
            stop = min(len(doc["region"]), int(inj[-1]) + 65)
            probs = doc["logits"].softmax(-1)
            for i in range(start, stop):
                region = {0: "page", 1: "context", 2: "injection"}.get(
                    int(doc["region"][i]), "?"
                )
                writer.writerow(
                    [
                        doc["case_id"],
                        doc["variant"],
                        int(outcomes[doc["case_id"]]["attack_success"]),
                        i,
                        region,
                        doc["token_texts"][i],
                        *[float(v) for v in doc["logits"][i]],
                        *[float(v) for v in probs[i]],
                    ]
                )
    sections = []
    for doc in attacks:
        inj = torch.where(doc["region"] == 2)[0]
        if not len(inj):
            continue
        start = max(0, int(inj[0]) - 32)
        stop = min(len(doc["region"]), int(inj[-1]) + 33)
        probs = doc["logits"].softmax(-1)
        spans = []
        for i in range(start, stop):
            p = probs[i]
            value = float(p[1] + p[2])
            red = int(255 - 55 * min(1.0, value))
            green = int(255 - 150 * min(1.0, value))
            title = " | ".join(
                f"{role}: logit={float(doc['logits'][i, j]):.2f}, p={float(p[j]):.3f}"
                for j, role in enumerate(ROLES)
            )
            border = "2px solid #cc9900" if int(doc["region"][i]) == 2 else "none"
            spans.append(
                f'<span title="{html.escape(title)}" style="background:rgb({red},{green},255);border-bottom:{border}">{html.escape(doc["token_texts"][i])}</span>'
            )
        sections.append(
            f'<h2>Case {doc["case_id"]}: {html.escape(doc["variant"])}; baseline success={int(outcomes[doc["case_id"]]["attack_success"])}</h2><div class="tokens">{"".join(spans)}</div>'
        )
    page = """<!doctype html><meta charset="utf-8"><title>Role-probe token heatmaps</title>
<style>body{font:14px system-ui;max-width:1200px;margin:30px auto}.tokens{font:14px/1.8 monospace;white-space:pre-wrap}.note{padding:12px;background:#eee}span{padding:1px}</style>
<h1>Layer-8 role-probe token heatmaps</h1><p class="note">Darker red/purple means larger p(user)+p(CoT). Gold underline marks the exact injected span. Hover any token for all five raw logits and probabilities.</p>""" + "\n".join(
        sections
    )
    (OUT / "token_heatmaps.html").write_text(page, encoding="utf-8")


def fmt(value: float) -> str:
    return f"{100 * value:.1f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(out)


def write_report(
    all_results: dict[int, list[dict[str, Any]]], meta: dict[int, Any]
) -> None:
    layer8 = all_results[8]
    rows = []
    for result in sorted(
        layer8, key=lambda item: item["global_token_auroc"], reverse=True
    ):
        operating = result["tradeoffs"][2]
        rows.append(
            [
                result["candidate"],
                f"{result['global_token_auroc']:.3f}",
                f"{result['macro_case_token_auroc']:.3f}",
                fmt(result["top128_any_overlap_rate"]),
                fmt(operating["attack_recall"]),
                fmt(operating["baseline_success_recall"]),
                fmt(operating["heldout_clean_fpr"]),
            ]
        )
    best = max(
        layer8,
        key=lambda item: (
            item["tradeoffs"][2]["attack_recall"],
            -item["tradeoffs"][2]["heldout_clean_fpr"],
            item["global_token_auroc"],
        ),
    )
    operating_rows = []
    for entry in best["tradeoffs"]:
        operating_rows.append(
            [
                fmt(entry["calibration_budget"]),
                f"{entry['threshold']:.4f}",
                fmt(entry["calibration_fpr"]),
                fmt(entry["heldout_clean_fpr"]),
                fmt(entry["attack_recall"]),
                fmt(entry["base_recall"]),
                fmt(entry["cot_recall"]),
                fmt(entry["baseline_success_recall"]),
            ]
        )
    report = f"""# Detecting prompt injection from the current role probe

## Answer in brief

Yes, useful token-level data remain. Raw five-role logits were already stored for layer 18, and this study exactly reconstructed layer-8 logits as `W h + b` from saved pre-MLP activations. This covers the 20-case attack-development suite and clean controls, **not the frozen 200-case final evaluation**. Two attack cases never returned a page, so injection detection is measurable on 18/20 attack cases.

The probe signal can localize injected text better than chance, but a single-token or single-threshold detector is not reliable enough. The existing layer-8 Mahalanobis/local score has injection-vs-page token AUROC **0.868**, yet its old threshold fires on **29.6%** of ordinary tokens in attacked pages. The raw-logit experiments below reach high token AUROC after smoothing, while clean-document false positives remain the limiting issue.

## Saved and generated artifacts

- `token_logits/layer-08/`: exact reconstructed logits for every page-bearing saved Tool result.
- `token_logit_excerpts.csv`: all five logits and probabilities for each injection plus 64 surrounding tokens.
- `token_heatmaps.html`: hoverable word/token view; gold underline is ground-truth injection.
- `token_roc.svg`: token-level localization ROC curves.
- `case-008-trace.svg`, `case-013-trace.svg`, `case-017-trace.svg`, and `case-020-trace.svg`: representative role-probability traces.
- `detector_metrics.json`: machine-readable results and operating points.

## What the logits look like

![Token ROC](token_roc.svg)

Smoothing matters because injected requests are contiguous spans, whereas ordinary HTML produces isolated role-logit spikes. The improvement from token score to 64–128-token averages supports a segment detector rather than a token threshold. However, maximizing a score anywhere in a 20k–40k-token page creates an extreme-value problem: even benign pages eventually contain a high-scoring region.

Representative traces (yellow is the exact injection):

![Case 8](case-008-trace.svg)

![Case 13](case-013-trace.svg)

![Case 17](case-017-trace.svg)

![Case 20](case-020-trace.svg)

For word-level inspection, open [the interactive heatmap](token_heatmaps.html) and hover tokens to see all logits.

## Layer-8 detector comparison

The final four columns use a threshold permitting at most a 20% false-positive budget on 20 clean calibration pages. “Held-out clean” contains the available clean-sanity and paired clean pages. These are development diagnostics, not final-test estimates.

{markdown_table(["Score", "Token AUROC", "Macro AUROC", "Top-128 overlaps injection", "Attack recall", "Baseline-success recall", "Held-out clean FPR"], rows)}

The best development configuration by attack recall at this operating point is **{best["candidate"]}**. Its threshold tradeoff is:

{markdown_table(["Clean calibration budget", "Threshold", "Calibration FPR", "Held-out clean FPR", "All-attack recall", "Base recall", "CoT recall", "Recall of baseline successes"], operating_rows)}

Small denominators matter: attack recall is over {meta[8]["attack_docs"]} observable attacks and held-out clean FPR over {meta[8]["heldout_clean_docs"]} pages. A single attack changes recall by 5.6 percentage points.

## Recommended algorithm: segmented multi-signal conformal detector

Do not deploy `max token Userness/CoTness > tau`. Use this pipeline:

1. **Segment the Tool result.** Parse markup where possible and split visible content into semantic blocks/sentences, preserving a token map. PromptLocate independently argues for semantic segmentation before instruction localization ([paper](https://arxiv.org/abs/2510.12252)).
2. **Compute role-confusion features at layers 8, 16, and 18.** For token `t` and layer `l`, retain the full vector `z_l,t`; use margins `z_user-z_tool`, `z_cot-z_tool`, `logsumexp(z_user,z_cot)-z_tool`, entropy, and cross-layer persistence. The role-confusion work specifically finds that style can override the architectural role and that Userness/CoTness tracks attacks ([project and paper](https://role-confusion.github.io/)).
3. **Aggregate within segments, not isolated tokens.** Use mean, 90th percentile, fraction above a benign quantile, maximum 16/32/64-token moving average, and a local-background contrast. Require a sustained run or high segment score.
4. **Add two signals the role probe does not contain.** (a) policy-conflict alignment learned from safe/unsafe contrastive pairs, following the motivation of PVDetector ([paper](https://arxiv.org/abs/2607.12624)); and (b) attention transfer from the trusted task to the candidate segment, motivated by Attention Tracker ([paper](https://aclanthology.org/2025.findings-naacl.123/)). RAP-ID likewise fuses directive likeness, attention takeover, and policy conflict instead of trusting one internal signal ([paper](https://aclanthology.org/2026.findings-acl.738/)).
5. **Train a small segment classifier with grouped splits.** Inputs should be only the aggregate internal features plus structural metadata, not attack-template keywords. Split by underlying page and attack template; hold out entire attack families. Optimize a high-recall objective or cost-sensitive loss.
6. **Calibrate at the document level.** Define the document score as the maximum segment score, then set the threshold from maximum scores on clean Tool documents. Use split-conformal p-values so document length and multiple scanning are included in calibration. Maintain two thresholds: quarantine/high-confidence and review/low-confidence.
7. **Fail safely before tool execution.** When flagged, prevent sensitive or external side-effect tool calls, but allow the model to continue the requested summarization on sanitized content. Detection alone should not be the sole security boundary.

Suggested score for the role-only first stage:

```text
r_l,t = logsumexp(z_l,t,user, z_l,t,cot) - z_l,t,tool
s(segment) = classifier([
  mean(r), q90(r), max_mean_16(r), max_mean_32(r), max_mean_64(r),
  fraction(r > benign_q99), local_background_contrast(r),
  same features at layers 8/16/18, cross_layer_agreement
])
document_score = max_segment s(segment)
```

For low false negatives, choose the review threshold to target ≥99% recall on a much larger attack-development corpus, then use the second-stage policy/attention signals to reduce false positives. The present 18 observable attacks are far too few to establish 99% recall.

## Why an ensemble is necessary

- Role logits capture **source/role impersonation**, not whether an instruction is harmful or conflicts with the task.
- Natural prose, quoted dialogue, documentation, and HTML/JavaScript can look user-like or reasoning-like, creating false positives.
- A semantically subtle injection can remain Tool-like and evade the role probe, creating false negatives.
- Activation detectors are adaptive attack surfaces: work on evasive injections demonstrates that linear activation probes can be deliberately bypassed ([paper](https://arxiv.org/abs/2602.00750)). Randomized or multi-layer ensembles and adversarial detector training are therefore important.
- Stronger architectural/training defenses remain complementary: structured-query training separates instructions from data ([StruQ](https://arxiv.org/abs/2402.06363)), while instruction-hierarchy training teaches models to ignore lower-privilege conflicts ([paper](https://arxiv.org/abs/2404.13208)).

## Limitations

- This is the fixed 20-case attack-development set with only two attack families and 18 observable Tool results.
- Ground-truth injection boundaries were used only for offline evaluation and plots, never as detector input.
- Candidate selection and evaluation share the same small development set, so the reported best method is optimistic.
- The logits precede the evaluated steering interventions; steered-run token logits were not logged.
- No claim is made about the frozen 200-case test set or adaptive attacks until logits are collected and thresholds are frozen before evaluation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes = {
        int(row["case_id"]): row
        for row in load_jsonl(
            ROOT
            / "research_outputs/phase3_steering/tool_activations/layer/results_task_quality.jsonl"
        )
        if row.get("variant") != "clean"
    }
    all_results = {}
    all_meta = {}
    docs_by_layer = {}
    for layer in (8, 18):
        docs = load_docs(layer)
        docs_by_layer[layer] = docs
        all_results[layer], all_meta[layer] = evaluate_layer(layer, docs, outcomes)
    payload = {
        "schema_version": 1,
        "layers": all_results,
        "dataset_counts": {
            layer: {k: v for k, v in meta.items() if k != "details"}
            for layer, meta in all_meta.items()
        },
    }
    (OUT / "detector_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    svg_roc(all_results[8], all_meta[8]["details"], OUT / "token_roc.svg")
    write_token_views(docs_by_layer[8], outcomes)
    for case_id in (8, 13, 17, 20):
        doc = next(
            (
                item
                for item in docs_by_layer[8]
                if item["dataset"] == "layer" and item["case_id"] == case_id
            ),
            None,
        )
        if doc is not None:
            svg_trace(doc, OUT / f"case-{case_id:03d}-trace.svg")
    write_report(all_results, all_meta)
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
