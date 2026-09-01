from __future__ import annotations

import argparse
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

    run_parser = commands.add_parser("run", help="run GPT-OSS evaluation")
    run_parser.add_argument("--config", default="configs/gpt-oss-20b.yaml")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        from .dataset import prepare

        output = prepare(repo, args.pages, args.seed, args.max_kb, args.overwrite)
        print(output)
        return

    from .runner import run

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    output = run(Config.load(config_path), args.limit, args.run_name)
    print((output / "summary.json").read_text())
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
