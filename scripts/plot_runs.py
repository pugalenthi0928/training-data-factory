from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

runs_csv = Path("runs/qa_runs.csv")
reports = Path("reports"); reports.mkdir(exist_ok=True)

if not runs_csv.exists():
    print("No runs/qa_runs.csv found. Run the pipeline once.")
    raise SystemExit(0)

df = pd.read_csv(runs_csv)
# Basic cleaning/sorting
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)

# Save a simple markdown summary
md = ["# QA Leaderboard (latest first)\n"]
keep_cols = [c for c in ["timestamp","model","num_eval_examples","exact_match","rougeL_f","rouge1_f","rouge2_f","dataset_size"] if c in df.columns]
md.append(df[keep_cols].to_markdown(index=False))
(Path("reports/qa_leaderboard.md")).write_text("\n\n".join(md), encoding="utf-8")

# Plot EM and ROUGE-L for top 10 rows
top = df.head(10)
plt.figure(figsize=(10,5))
x = range(len(top))
plt.bar([i-0.2 for i in x], top["exact_match"], width=0.4, label="Exact Match")
if "rougeL_f" in top.columns:
    plt.bar([i+0.2 for i in x], top["rougeL_f"], width=0.4, label="ROUGE-L F")
plt.xticks(list(x), [str(getattr(t, "date", t)) if hasattr(t, "date") else str(t) for t in top["timestamp"]], rotation=45, ha="right")
plt.title("QA Runs (Top 10)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/qa_leaderboard.png", dpi=150)
print("Wrote: reports/qa_leaderboard.md, reports/qa_leaderboard.png")
