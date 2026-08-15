from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a top-level mapping/object.")
    return data


def build_tdr_command(cfg: Dict[str, Any], profile_name: str) -> List[str]:
    defaults: Dict[str, Any] = cfg.get("defaults", {}) or {}
    profiles: Dict[str, Any] = cfg.get("profiles", {}) or {}

    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise SystemExit(f"Unknown profile '{profile_name}'. Available profiles: {available}")

    profile: Dict[str, Any] = profiles[profile_name] or {}

    def get(key: str, default: Any = None) -> Any:
        return profile.get(key, defaults.get(key, default))

    source = get("source")
    if not source:
        raise SystemExit(f"Profile '{profile_name}' is missing required field 'source'.")

    tasks = get("tasks", [])
    if isinstance(tasks, str):
        # Allow comma-separated string in YAML
        tasks_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        tasks_list = list(tasks)

    if not tasks_list:
        raise SystemExit(f"Profile '{profile_name}' has no tasks configured.")

    model = get("model", "gpt-4.1-mini")
    max_chars = int(get("max_chars", 900))
    overlap = int(get("overlap", 120))
    max_examples = int(get("max_examples", 300))
    output_dir = Path(get("output_dir", "output"))
    output_name = profile.get("output_name", profile_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}.jsonl"

    cmd: List[str] = [
        "tdr",
        "process",
        "--source",
        str(source),
        "--out",
        str(out_path),
        "--tasks",
        ",".join(tasks_list),
        "--max-examples",
        str(max_examples),
        "--max-chars",
        str(max_chars),
        "--overlap",
        str(overlap),
        "--model",
        model,
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Training Data Robo (tdr) using a YAML profile.")
    parser.add_argument(
        "--config",
        type=str,
        default="tdr_config.yaml",
        help="Path to YAML config file (default: tdr_config.yaml)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Profile name to run (must exist under profiles: in the YAML).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cmd = build_tdr_command(cfg, args.profile)

    print("Running command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
