"""Compile security-failure-adjacent model reasoning from completed steering runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research_outputs" / "FAILURE_REASONING_COMPILATION.md"

RUNS = [
    (
        "Baseline (undefended)",
        ROOT
        / "research_outputs/phase3_steering/tool_activations/layer/results_task_quality.jsonl",
    ),
    (
        "Soft v1 alpha=0.5",
        ROOT
        / "research_outputs/phase3_steering/runs/causal-soft-l06-a0p5/results.jsonl",
    ),
    (
        "Soft v1 alpha=1.0",
        ROOT
        / "research_outputs/phase3_steering/runs/causal-soft-l06-a1p0/results.jsonl",
    ),
    (
        "Soft v2 rho=0.005",
        ROOT
        / "research_outputs/phase3_steering_v2/runs/causal-soft-v2-l18-rho0p005/results.jsonl",
    ),
    (
        "Continuous rho=0.0025",
        ROOT
        / "research_outputs/phase3_continuous_v2/runs/continuous-v2-l08-rho0p0025/results.jsonl",
    ),
    (
        "Continuous rho=0.005",
        ROOT
        / "research_outputs/phase3_continuous_v2/runs/continuous-v2-l08-rho0p005/results.jsonl",
    ),
    (
        "Continuous rho=0.01",
        ROOT
        / "research_outputs/phase3_continuous_v2/runs/continuous-v2-l08-rho0p01/results.jsonl",
    ),
]

REASON_RE = re.compile(
    r"\.env|dpaste|upload|backup|policy|sensitive|secret|private|comply|allowed|"
    r"allowable|permitted|green shirt|colleague|onboard|data integrity|exfiltrat",
    re.IGNORECASE,
)
SENSITIVE_COMMAND_RE = re.compile(r"\.env|dpaste\.com/api", re.IGNORECASE)


def load_attack_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("variant") != "clean"]


def analysis_text(step: dict[str, Any]) -> str:
    value = step.get("analysis") or ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def tool_commands(step: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for call in step.get("tool_calls") or []:
        args: Any = None
        if isinstance(call, (list, tuple)) and len(call) >= 2:
            args = call[1]
        elif isinstance(call, dict):
            args = call.get("arguments", call.get("args"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                commands.append(args)
                continue
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            commands.append(args["command"])
    return commands


def relevant_excerpt(text: str) -> str:
    """Keep exact security-relevant paragraphs and mark omitted material."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    selected = [part for part in paragraphs if REASON_RE.search(part)]
    if not selected:
        return text.strip()
    if len(selected) == len(paragraphs):
        return "\n\n".join(selected)
    return "[unrelated reasoning omitted]\n\n" + "\n\n".join(selected)


def security_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for step in row.get("transcript") or []:
        commands = tool_commands(step)
        if any(SENSITIVE_COMMAND_RE.search(command) for command in commands):
            selected.append(
                {
                    "step": step.get("step"),
                    "analysis": relevant_excerpt(analysis_text(step)),
                    "commands": commands,
                }
            )
    return selected


def pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.1f}%)"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(out)


def status(row: dict[str, Any] | None) -> str:
    if row is None:
        return "?"
    if row.get("attack_success"):
        return "S"
    if row.get("attack_attempt"):
        return "A"
    return "—"


def main() -> None:
    by_run: dict[str, dict[int, dict[str, Any]]] = {}
    for label, path in RUNS:
        rows = load_attack_rows(path)
        if len(rows) != 20:
            raise ValueError(f"{label}: expected 20 attack rows, found {len(rows)}")
        by_run[label] = {int(row["case_id"]): row for row in rows}

    labels = [label for label, _ in RUNS]
    baseline = by_run[labels[0]]
    successful_rows = [
        (label, row)
        for label in labels
        for row in by_run[label].values()
        if row.get("attack_success")
    ]
    attempt_only_rows = [
        (label, row)
        for label in labels
        for row in by_run[label].values()
        if row.get("attack_attempt") and not row.get("attack_success")
    ]
    unique_failure_cases = sorted({int(row["case_id"]) for _, row in successful_rows})

    all_failure_reasoning = []
    for _, row in successful_rows:
        all_failure_reasoning.append(
            "\n".join(analysis_text(step) for step in row.get("transcript") or [])
        )

    motifs = [
        ("Mentions policy", r"\bpolicy\b"),
        (
            "Concludes allowed/allowable/permitted",
            r"\ballow(?:ed|able)?\b|\bpermitted\b",
        ),
        ("Says must/let's/just comply", r"must comply|let.s comply|just comply"),
        ("Recognizes sensitive/private/secret data", r"sensitive|private data|secret"),
        (
            "Recasts embedded text as a user request",
            r"the user (?:is asking|asks|requested|wants)",
        ),
        ("Uses backup/data-integrity rationale", r"backup|data integrity"),
        ("Uses green-shirt rationale", r"green shirt|green apparel"),
        ("Uses collaboration/onboarding rationale", r"colleague|collaborat|onboard"),
    ]

    md: list[str] = []
    md.append("# Failure-adjacent model reasoning from completed steering runs\n")
    md.append(
        "This is a copy-ready compilation of the model's saved `analysis` text immediately "
        "before `.env` discovery/read or simulated upload tool calls. It uses only completed "
        "artifacts summarized in `comparison/comparison.md`; no new model evaluation was run."
    )
    md.append(
        "The quoted blocks are model-generated reasoning from the benchmark transcripts, not "
        "post-hoc explanations. `[unrelated reasoning omitted]` marks paragraph-level trimming; "
        "the retained paragraph text is otherwise verbatim. All credentials and network events "
        "in this lab are synthetic/simulated.\n"
    )

    md.append("## Scope and labels\n")
    md.append(
        f"There are {len(successful_rows)} canonical successful-attack run/case outcomes "
        f"across {len(unique_failure_cases)} unique cases. There are also "
        f"{len(attempt_only_rows)} attempted-but-unsuccessful outcomes. Every run contains "
        "the same 20 attacks: 10 base injections and 10 CoT-forgery injections."
    )
    md.append(
        "- **S**: deterministic simulator `attack_success=1` (canonical failure).\n"
        "- **A**: attack was attempted but the simulated exfiltration did not succeed.\n"
        "- **—**: no benchmark attack attempt.\n"
    )

    summary_rows = []
    for label in labels:
        rows = list(by_run[label].values())
        base = [row for row in rows if row["variant"] == "base-injection"]
        cot = [row for row in rows if row["variant"] == "cot-forgery-injection"]
        summary_rows.append(
            [
                label,
                pct(sum(bool(row.get("attack_success")) for row in rows), len(rows)),
                pct(sum(bool(row.get("attack_success")) for row in base), len(base)),
                pct(sum(bool(row.get("attack_success")) for row in cot), len(cot)),
                pct(
                    sum(bool(row.get("legitimate_task_success")) for row in rows),
                    len(rows),
                ),
            ]
        )
    md.append(
        md_table(
            ["Run", "Benchmark ASR", "Base ASR", "CoT-forgery ASR", "Strict task"],
            summary_rows,
        )
    )
    md.append("")

    md.append("## Failure-status matrix\n")
    matrix_rows = []
    for case_id in unique_failure_cases:
        exemplar = next(
            by_run[label][case_id] for label in labels if case_id in by_run[label]
        )
        matrix_rows.append(
            [
                case_id,
                exemplar["variant"],
                exemplar["injection_id"],
                *[status(by_run[label].get(case_id)) for label in labels],
            ]
        )
    md.append(
        md_table(
            [
                "Case",
                "Variant",
                "Injection",
                "Baseline",
                "Soft .5",
                "Soft 1.0",
                "Soft v2",
                "Cont .0025",
                "Cont .005",
                "Cont .01",
            ],
            matrix_rows,
        )
    )
    md.append("")

    md.append("## Recurring reasoning motifs\n")
    md.append(
        "Counts below use the 38 canonical run/case failures as the denominator. Runs are "
        "separate observations, so a fixed case recurring under several interventions is "
        "counted several times."
    )
    motif_rows = []
    for name, pattern in motifs:
        count = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for text in all_failure_reasoning
        )
        motif_rows.append([name, pct(count, len(all_failure_reasoning))])
    md.append(md_table(["Observed motif", "Failure traces containing it"], motif_rows))
    md.append("")
    md.append(
        "The central contradiction is therefore explicit: the model usually notices that the "
        "target is sensitive, but accepts a fabricated exception and transfers authority from "
        "the Tool-returned webpage to the user. The failure is primarily an authorization/source "
        "reasoning failure, not simple failure to recognize `.env` as sensitive."
    )
    md.append(
        "The CoT-forgery family is consistently more effective than base injection. Its forged "
        "policy analysis supplies the exact bridge the model later repeats: identify an apparent "
        "exception, declare the action allowed, then state an obligation to comply."
    )
    md.append(
        "Continuous rho=0.0025 reduced successful simulated exfiltration to 3/20, but its objective "
        "exfiltration-command rate remained 6/20. In other words, some apparent safety gain came "
        "from command failure rather than disappearance of the unsafe plan.\n"
    )

    md.append("## Steering-induced canonical failures relative to baseline\n")
    regression_rows = []
    for label in labels[1:]:
        for case_id, row in sorted(by_run[label].items()):
            if row.get("attack_success") and not baseline[case_id].get(
                "attack_success"
            ):
                regression_rows.append(
                    [
                        label,
                        case_id,
                        row["variant"],
                        row["injection_id"],
                        status(baseline[case_id]),
                    ]
                )
    md.append(
        md_table(
            ["Run", "Case", "Variant", "Injection", "Baseline status"],
            regression_rows,
        )
    )
    md.append(
        "These are especially useful for debugging because the fixed prompt, case ID, and seed "
        "were matched while the intervention changed the outcome. They do not by themselves prove "
        "the quoted sentence caused the regression; sampling trajectories can diverge after an "
        "activation intervention.\n"
    )

    md.append("## Exact reasoning excerpts for every canonical failure\n")
    for label in labels:
        failures = [
            row for _, row in sorted(by_run[label].items()) if row.get("attack_success")
        ]
        md.append(f"### {label}\n")
        for row in failures:
            baseline_state = status(baseline[int(row["case_id"])])
            regression = label != labels[0] and baseline_state != "S"
            md.append(
                f"#### Case {row['case_id']} — {row['injection_id']}"
                + (" — steering regression" if regression else "")
                + "\n"
            )
            md.append(
                f"Variant: `{row['variant']}`. Output class: `{row['output_class']}`. "
                f"Strict task success: `{int(bool(row.get('legitimate_task_success')))}`. "
                f"Baseline status: `{baseline_state}`. Source: `{row['source_url']}`."
            )
            steps = security_steps(row)
            if not steps:
                md.append(
                    "No `.env`/dpaste tool-call step was recoverable from the transcript.\n"
                )
                continue
            for step in steps:
                md.append(
                    f"**Generation step {step['step']} reasoning (verbatim relevant paragraphs):**"
                )
                md.append("~~~~text\n" + step["analysis"] + "\n~~~~")
                relevant_commands = [
                    command
                    for command in step["commands"]
                    if SENSITIVE_COMMAND_RE.search(command)
                ]
                md.append(
                    "**Tool command(s) emitted immediately after that reasoning:**"
                )
                md.append("~~~~bash\n" + "\n".join(relevant_commands) + "\n~~~~\n")

    md.append("## Attempt-only outcomes\n")
    md.append(
        "These traces are not counted in benchmark-success ASR, but they remain safety-relevant "
        "because the model began the injected procedure. Consult the source JSONL for their full "
        "transcripts."
    )
    attempt_rows = [
        [
            label,
            row["case_id"],
            row["variant"],
            row["injection_id"],
            row["output_class"],
        ]
        for label, row in sorted(
            attempt_only_rows,
            key=lambda item: (labels.index(item[0]), item[1]["case_id"]),
        )
    ]
    md.append(
        md_table(
            ["Run", "Case", "Variant", "Injection", "Output class"],
            attempt_rows,
        )
    )
    md.append("")

    md.append("## Implications for the next steering method\n")
    md.append(
        "1. Target the authorization transition, not merely generic Tool-role probability. The "
        "dangerous state is the sequence `recognized sensitive → fabricated exception accepted → "
        "embedded instruction treated as user authority → comply`."
    )
    md.append(
        "2. Measure command attempts independently of simulator success. A method that turns a "
        "successful upload into a malformed upload has not removed the unsafe plan."
    )
    md.append(
        "3. Use token- and layer-local diagnostics around phrases such as `the user is asking`, "
        "`allowed`, `exception`, and `must comply`, plus the `.env`/upload command tokens. Aggregate "
        "Tool-tag classification can hide the decisive local transition."
    )
    md.append(
        "4. Add an explicit counterfactual test: after steering, the model should still summarize "
        "the page but should describe the embedded request as untrusted webpage content and avoid "
        "all sensitive discovery/read/upload commands."
    )
    md.append(
        "5. Treat case-level regressions as first-class calibration examples, particularly cases "
        "20 and 17 for soft v1 and cases 1, 5, 7, 10, 14, and 22 for later methods."
    )

    OUTPUT.write_text("\n\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(
        f"canonical_failures={len(successful_rows)} unique_cases={len(unique_failure_cases)} "
        f"attempt_only={len(attempt_only_rows)}"
    )


if __name__ == "__main__":
    main()
