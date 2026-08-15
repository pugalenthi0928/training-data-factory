#!/usr/bin/env python3
"""Create or verify a content-addressed Forge dataset release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training_data_robo.releases import ReleaseValidationError, verify_release, write_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify a gated Forge dataset release.")
    parser.add_argument("--run-dir", required=True, help="Completed Forge run directory")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--version", help="Dataset version")
    parser.add_argument("--license", dest="dataset_license", help="Dataset license identifier or URL")
    parser.add_argument("--benchmark-origin", help="Who authored the benchmark and how it was isolated")
    parser.add_argument("--tier", choices=("smoke", "candidate"), default="smoke")
    parser.add_argument("--verify", action="store_true", help="Verify an existing release_manifest.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    try:
        if args.verify:
            result = verify_release(run_dir / "release_manifest.json")
        else:
            missing = [
                flag
                for flag, value in (
                    ("--name", args.name),
                    ("--version", args.version),
                    ("--license", args.dataset_license),
                    ("--benchmark-origin", args.benchmark_origin),
                )
                if not value
            ]
            if missing:
                parser.error(f"release creation requires: {', '.join(missing)}")
            manifest = write_release(
                run_dir,
                dataset_name=args.name,
                dataset_version=args.version,
                dataset_license=args.dataset_license,
                benchmark_origin=args.benchmark_origin,
                release_tier=args.tier,
            )
            result = {
                "created": True,
                "release_id": manifest["release_id"],
                "tier": manifest["release_tier"],
                "manifest": str(run_dir / "release_manifest.json"),
                "croissant": str(run_dir / "croissant.json"),
            }
    except ReleaseValidationError as exc:
        raise SystemExit(f"Release blocked: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
