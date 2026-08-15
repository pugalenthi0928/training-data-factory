#!/usr/bin/env python3
"""Measure human agreement, blind system preference, and judge alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation import analyse_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse a Forge blind evaluation release")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--judge-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-alpha", type=float, default=0.667)
    parser.add_argument("--minimum-judge-agreement", type=float, default=0.7)
    parser.add_argument("--minimum-position-consistency", type=float, default=0.8)
    parser.add_argument("--require-fixture-alpha", type=float)
    parser.add_argument("--require-fixture-position-consistency", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyse_evaluation(
        args.manifest,
        args.annotations,
        args.judge_predictions,
        args.output,
        minimum_alpha=args.minimum_alpha,
        minimum_judge_agreement=args.minimum_judge_agreement,
        minimum_position_consistency=args.minimum_position_consistency,
    )
    alpha = report["human_agreement"]["krippendorff_alpha_nominal"]
    position_consistency = report["judge_calibration"]["position_consistency"]
    if args.require_fixture_alpha is not None and (alpha is None or alpha < args.require_fixture_alpha):
        raise SystemExit(f"fixture alpha {alpha} is below {args.require_fixture_alpha}")
    if args.require_fixture_position_consistency is not None and (
        position_consistency is None or position_consistency < args.require_fixture_position_consistency
    ):
        raise SystemExit(
            f"fixture position consistency {position_consistency} is below {args.require_fixture_position_consistency}"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
