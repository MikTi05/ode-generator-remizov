#!/usr/bin/env python3
"""Run full artifact validation for an all_methods_equations_v1 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from src.all_methods_dataset_v1 import (
    AllMethodsConfig,
    GenerationError,
    validate_dataset_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--output-dir",
        dest="input_dir",
        type=Path,
        default=Path("data"),
        help="directory containing the four versioned dataset artifacts",
    )
    parser.add_argument(
        "--expect-production",
        action="store_true",
        help="require seed 2488 and the frozen 200-family x 5-sample configuration",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected_config = AllMethodsConfig() if args.expect_production else None
        summary = validate_dataset_artifacts(
            args.input_dir,
            expected_config=expected_config,
        )
    except (GenerationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"valid": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
