from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible GPT-OSS prompt-injection evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare", help="freeze the evaluation dataset"
    )
    prepare_parser.add_argument("--pages", type=int, default=100)
    prepare_parser.add_argument("--seed", type=int, default=1234)
    prepare_parser.add_argument("--max-kb", type=int, default=100)
    prepare_parser.add_argument("--overwrite", action="store_true")

    steering_parser = commands.add_parser(
        "prepare-steering", help="freeze disjoint steering datasets"
    )
    steering_parser.add_argument("--repr-seed", type=int, default=20260902)
    steering_parser.add_argument("--wikipedia-seed", type=int, default=20260903)
    steering_parser.add_argument("--max-kb", type=int, default=100)
    steering_parser.add_argument("--force", action="store_true")

    repr_collect_parser = commands.add_parser(
        "collect-repr-activations",
        help="collect aligned post-block residual activations for D_repr",
    )
    repr_collect_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    repr_collect_parser.add_argument("--max-tokens-per-base", type=int, default=64)
    repr_collect_parser.add_argument("--resume", action="store_true")

    repr_analyze_parser = commands.add_parser(
        "analyze-repr",
        help="fit layer probes, pair vectors, and continuous role bases",
    )
    repr_analyze_parser.add_argument("--seed", type=int, default=20260904)
    repr_analyze_parser.add_argument("--l2", type=float, default=1e-4)
    repr_analyze_parser.add_argument("--epochs", type=int, default=80)
    repr_analyze_parser.add_argument("--svd-rank", type=int, default=16)

    tool_collect_parser = commands.add_parser(
        "collect-tool-activations",
        help="run the undefended agent and collect unique Tool-result activations",
    )
    tool_collect_parser.add_argument(
        "--dataset", choices=("clean", "layer"), required=True
    )
    tool_collect_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    tool_collect_parser.add_argument("--max-tokens-per-message", type=int, default=512)
    tool_collect_parser.add_argument("--tail-tokens", type=int, default=128)
    tool_collect_parser.add_argument("--resume", action="store_true")

    commands.add_parser(
        "analyze-steering-layers",
        help="fit clean gates and screen layers using saved activations",
    )
    rescore_parser = commands.add_parser(
        "rescore-steering-baseline",
        help="apply the grounded-summary utility scorer to saved baselines",
    )
    rescore_parser.add_argument("--dataset", choices=("clean", "layer"), required=True)
    steering_run_parser = commands.add_parser(
        "run-steering", help="run a soft-pairwise or continuous intervention"
    )
    steering_run_parser.add_argument(
        "--dataset", choices=("layer", "tune", "devval"), required=True
    )
    steering_run_parser.add_argument(
        "--method", choices=("soft-pairwise", "continuous"), required=True
    )
    steering_run_parser.add_argument("--layer", type=int, required=True)
    steering_run_parser.add_argument("--alpha", type=float, required=True)
    steering_run_parser.add_argument("--rank", type=int, default=4)
    steering_run_parser.add_argument("--run-name", required=True)
    steering_run_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    steering_run_parser.add_argument("--attack-only", action="store_true")
    steering_run_parser.add_argument("--resume", action="store_true")

    steering_debug_parser = commands.add_parser(
        "debug-steering",
        help="replay adverse soft-pairwise cases and write a token-level report",
    )
    steering_debug_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    steering_debug_parser.add_argument("--output-dir", type=Path)

    v2_data_parser = commands.add_parser(
        "prepare-steering-v2-repr", help="freeze the paper-scale v2 role dataset"
    )
    v2_data_parser.add_argument("--seed", type=int, default=20260905)
    v2_data_parser.add_argument("--force", action="store_true")
    v2_collect_parser = commands.add_parser(
        "collect-steering-v2-repr", help="capture paper-aligned pre-MLP states"
    )
    v2_collect_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    v2_collect_parser.add_argument("--resume", action="store_true")
    v2_analyze_parser = commands.add_parser(
        "analyze-steering-v2-repr", help="fit v2 L2 role probes and directions"
    )
    v2_analyze_parser.add_argument("--c", type=float, default=5e-3)
    v2_analyze_parser.add_argument("--max-iter", type=int, default=100)
    v2_tool_parser = commands.add_parser(
        "collect-steering-v2-tool", help="replay Tool states at the pre-MLP site"
    )
    v2_tool_parser.add_argument("--dataset", choices=("clean", "layer"), required=True)
    v2_tool_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    v2_tool_parser.add_argument("--resume", action="store_true")
    v2_calibrate_parser = commands.add_parser(
        "calibrate-steering-v2", help="fit the robust joint gate and diagnostics"
    )
    v2_calibrate_parser.add_argument("--window", type=int, default=32)
    v2_calibrate_parser.add_argument("--quantile", type=float, default=0.99)
    v2_calibrate_parser.add_argument("--rho-max", type=float, default=0.005)
    v2_run_parser = commands.add_parser(
        "run-steering-v2", help="run the frozen soft-pairwise v2 D_layer evaluation"
    )
    v2_run_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    v2_run_parser.add_argument("--run-name", required=True)
    v2_run_parser.add_argument("--rho-max", type=float, default=0.005)
    v2_run_parser.add_argument("--resume", action="store_true")

    continuous_geometry_parser = commands.add_parser(
        "build-continuous-geometry",
        help="build the tokenwise paper-aligned continuous role basis",
    )
    continuous_geometry_parser.add_argument("--rank", type=int, default=4)
    continuous_geometry_parser.add_argument("--sample-per-base", type=int, default=256)
    continuous_geometry_parser.add_argument("--oversample", type=int, default=12)
    continuous_geometry_parser.add_argument("--power-iterations", type=int, default=2)
    continuous_geometry_parser.add_argument("--seed", type=int, default=20260906)
    continuous_clean_parser = commands.add_parser(
        "collect-continuous-clean",
        help="collect one projected canonical Tool prefill per clean page",
    )
    continuous_clean_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    continuous_clean_parser.add_argument("--resume", action="store_true")
    continuous_calibrate_parser = commands.add_parser(
        "calibrate-continuous",
        help="fit and diagnose the continuous clean Tool region",
    )
    continuous_calibrate_parser.add_argument("--window", type=int, default=32)
    continuous_calibrate_parser.add_argument("--quantile", type=float, default=0.99)
    continuous_calibrate_parser.add_argument(
        "--diagnostic-rho", type=float, default=0.005
    )
    continuous_run_parser = commands.add_parser(
        "run-continuous-v2",
        help="run frozen continuous role-region steering on D_layer",
    )
    continuous_run_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    continuous_run_parser.add_argument("--run-name", required=True)
    continuous_run_parser.add_argument("--rho-max", type=float, required=True)
    continuous_run_parser.add_argument("--resume", action="store_true")

    run_parser = commands.add_parser("run", help="run GPT-OSS evaluation")
    run_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument(
        "--case-ids",
        help="comma-separated manifest case IDs in the exact requested order",
    )
    run_parser.add_argument("--run-name")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted run after validating its metadata",
    )
    compare_parser = commands.add_parser(
        "compare", help="compare two run directories for exact behavior"
    )
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        from .dataset import prepare

        output = prepare(repo, args.pages, args.seed, args.max_kb, args.overwrite)
        print(output)
        return

    if args.command == "prepare-steering":
        from .steering_data import prepare_steering_data

        report = prepare_steering_data(
            repo,
            repr_seed=args.repr_seed,
            wikipedia_seed=args.wikipedia_seed,
            max_kb=args.max_kb,
            force=args.force,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "collect-repr-activations":
        from .steering_repr import collect_representation_activations

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = collect_representation_activations(
            repo,
            config_path,
            max_tokens_per_base=args.max_tokens_per_base,
            resume=args.resume,
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "analyze-repr":
        from .steering_repr import analyze_representation_activations

        output = analyze_representation_activations(
            repo,
            seed=args.seed,
            l2=args.l2,
            epochs=args.epochs,
            svd_rank=args.svd_rank,
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "collect-tool-activations":
        from .steering_agent import collect_tool_activations

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = collect_tool_activations(
            repo,
            dataset=args.dataset,
            config_path=config_path,
            resume=args.resume,
            max_tokens_per_message=args.max_tokens_per_message,
            tail_tokens=args.tail_tokens,
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "analyze-steering-layers":
        from .steering_diagnostics import analyze_layer_separation

        output = analyze_layer_separation(repo)
        print(f"Artifacts: {output}")
        return

    if args.command == "rescore-steering-baseline":
        from .steering_diagnostics import rescore_saved_tool_results

        output = rescore_saved_tool_results(repo, args.dataset)
        print((output / "summary_task_quality.json").read_text())
        print(f"Artifacts: {output}")
        return

    if args.command == "run-steering":
        from .steering_runtime import run_steering

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = run_steering(
            repo,
            dataset=args.dataset,
            config_path=config_path,
            method=args.method,
            layer=args.layer,
            alpha=args.alpha,
            rank=args.rank,
            run_name=args.run_name,
            attack_only=args.attack_only,
            resume=args.resume,
        )
        print((output / "summary.json").read_text())
        print(f"Artifacts: {output}")
        return

    if args.command == "debug-steering":
        from .steering_debug import build_steering_debug_report

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output_dir = args.output_dir
        if output_dir is not None and not output_dir.is_absolute():
            output_dir = repo / output_dir
        output = build_steering_debug_report(repo, config_path, output_dir)
        print(f"Report: {output}")
        return

    if args.command == "prepare-steering-v2-repr":
        from .steering_v2_data import prepare_v2_representation_data

        output = prepare_v2_representation_data(repo, args.seed, args.force)
        print(f"Manifest: {output}")
        return

    if args.command == "collect-steering-v2-repr":
        from .steering_v2_repr import collect_v2_representation_activations

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = collect_v2_representation_activations(
            repo, config_path, resume=args.resume
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "analyze-steering-v2-repr":
        from .steering_v2_repr import analyze_v2_representation_activations

        output = analyze_v2_representation_activations(
            repo, c_value=args.c, max_iter=args.max_iter
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "collect-steering-v2-tool":
        from .steering_v2_diagnostics import collect_v2_tool_activations

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = collect_v2_tool_activations(
            repo, args.dataset, config_path, resume=args.resume
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "calibrate-steering-v2":
        from .steering_v2_diagnostics import calibrate_and_diagnose_v2

        output = calibrate_and_diagnose_v2(
            repo,
            window=args.window,
            quantile=args.quantile,
            rho_max=args.rho_max,
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "run-steering-v2":
        from .steering_v2_runtime import run_v2_steering

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = run_v2_steering(
            repo,
            config_path,
            args.run_name,
            rho_max=args.rho_max,
            resume=args.resume,
        )
        print((output / "summary.json").read_text())
        print(f"Artifacts: {output}")
        return

    if args.command == "build-continuous-geometry":
        from .continuous_geometry import build_continuous_geometry

        output = build_continuous_geometry(
            repo,
            rank=args.rank,
            sample_per_base=args.sample_per_base,
            oversample=args.oversample,
            power_iterations=args.power_iterations,
            seed=args.seed,
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "collect-continuous-clean":
        from .continuous_diagnostics import collect_continuous_clean_projections

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = collect_continuous_clean_projections(
            repo, config_path, resume=args.resume
        )
        print(f"Artifacts: {output}")
        return

    if args.command == "calibrate-continuous":
        from .continuous_diagnostics import calibrate_continuous

        output = calibrate_continuous(
            repo,
            window=args.window,
            quantile=args.quantile,
            diagnostic_rho=args.diagnostic_rho,
        )
        print((output / "diagnostic_report.json").read_text())
        print(f"Artifacts: {output}")
        return

    if args.command == "run-continuous-v2":
        from .continuous_runtime import run_continuous_steering

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        output = run_continuous_steering(
            repo,
            config_path,
            args.run_name,
            rho_max=args.rho_max,
            resume=args.resume,
        )
        print((output / "summary.json").read_text())
        print(f"Artifacts: {output}")
        return

    if args.command == "compare":
        from .runner import compare_run_dirs

        report = compare_run_dirs(args.reference, args.candidate)
        payload = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        print(payload, end="")
        raise SystemExit(0 if report["all_equivalent"] else 1)

    from .runner import run

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    case_ids = None
    if args.case_ids:
        case_ids = [int(value) for value in args.case_ids.split(",")]
    output = run(
        Config.load(config_path),
        limit=args.limit,
        run_name=args.run_name,
        resume=args.resume,
        case_ids=case_ids,
    )
    print((output / "summary.json").read_text())
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
