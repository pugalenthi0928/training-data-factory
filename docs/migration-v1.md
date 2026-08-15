# Migrating orchestration code to `forge`

Stage 2 gives Forge one supported execution model. The historical
`training_data_robo` package remains installed so existing imports continue to
work, but new orchestration should use `forge`.

## Command line

The repository wrapper still works:

```bash
python scripts/run_forge.py --help
```

The installed command is now the shorter path:

```bash
forge --help
```

Both commands call `forge.cli.main` and construct the same typed stage graph.

## Python

```python
from pathlib import Path

from forge import ForgeConfig, run_forge

config = ForgeConfig.from_paths(
    sources=["./sample_docs"],
    benchmarks=["./sample_benchmarks/contamination_smoke_test.jsonl"],
    tasks=("qa", "summary"),
    max_examples=40,
    dry_run=True,
)

result = run_forge(Path("runs/example"), config)
print(result.release_id)
```

`run_forge` is the supported boundary for a CLI, test, worker, or future HTTP
service. It returns typed stage results and a verified release identity.

## Resume behavior

Calling `run_forge` again with the same run directory resumes by default. A
cache hit requires all of the following:

1. The stage name and implementation version match.
2. The typed configuration matches.
3. Every declared input has the same content fingerprint.
4. Every recorded output still exists and has the same fingerprint.

A changed configuration, changed input, missing output, or modified output
invalidates that stage. Downstream stages are reconsidered using their own
content keys.

Use `resume=False` in Python or `--no-resume` on the CLI to execute all stages
again.

## Evidence files

`pipeline_events.jsonl` is the detailed event stream. It records stage start,
completion, failure, and cache-hit events with input and output artifacts,
configuration, prompt identities, model identities, elapsed time, and error
text.

`pipeline_log.json` is a compact compatibility summary used by the release
gate. A cached stage is recorded as a passing stage with `execution: cached`.

## Training and evaluation

`TrainingConfig` and `EvaluationConfig` define backend-neutral contracts. The
current release workflow intentionally stops at a verified dataset. The older
MLX scripts remain available for local experiments, but they are not yet part
of the canonical release graph because they cannot run on every CI platform
and do not yet satisfy the independent evaluation gate.
