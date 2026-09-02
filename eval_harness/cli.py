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
