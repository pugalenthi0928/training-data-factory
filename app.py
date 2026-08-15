"""Forge Dashboard for exploring pipeline runs, quality metrics,
training results, and experiment comparisons.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl_df(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records) if records else pd.DataFrame()


def altair_chart_compat(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def find_run_dirs() -> List[Path]:
    """Find all Forge run directories (contain config.json)."""
    runs_root = Path("runs")
    if not runs_root.exists():
        return []
    dirs = []
    for d in sorted(runs_root.iterdir(), reverse=True):
        if d.is_dir() and (d / "config.json").exists():
            dirs.append(d)
    return dirs


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_pipeline_log(run_dir: Path) -> List[Dict[str, Any]]:
    log_path = run_dir / "pipeline_log.json"
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Forge - Training Data Dashboard",
    page_icon="🔥",
    layout="wide",
)

# Session state for interactive curation
if "labels" not in st.session_state:
    st.session_state["labels"] = {}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🔥 Forge")
page = st.sidebar.radio(
    "Navigate",
    [
        "Pipeline Overview",
        "Quality Deep-Dive",
        "Training Results",
        "Experiment Comparison",
        "Dataset Explorer",
    ],
)

# =========================================================================
# PAGE: Pipeline Overview
# =========================================================================
if page == "Pipeline Overview":
    st.title("🔥 Forge Pipeline Overview")
    st.markdown(
        "View pipeline step execution status, timing, and configuration "
        "for each run."
    )

    run_dirs = find_run_dirs()
    if not run_dirs:
        st.info(
            "No runs found. Run `make forge` to execute the pipeline."
        )
        st.stop()

    run_labels = [d.name for d in run_dirs]
    selected_run_name = st.selectbox("Select run", run_labels)
    run_dir = Path("runs") / selected_run_name

    config = load_run_config(run_dir)
    pipeline_log = load_pipeline_log(run_dir)

    # Config summary
    st.subheader("Configuration")
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    cfg_col1.metric("Tasks", config.get("tasks", "N/A"))
    cfg_col2.metric("Model", config.get("model", "dry-run"))
    cfg_col3.metric("Max examples", config.get("max_examples", "N/A"))
    cfg_col4.metric("Status", config.get("status", config.get("completed_at", "unknown")))

    # Pipeline steps
    if pipeline_log:
        st.subheader("Pipeline Steps")

        step_data = []
        for entry in pipeline_log:
            status = entry.get("status", "unknown")
            emoji = {"ok": "✅", "failed": "❌", "skipped": "⏭️", "cached": "💾"}.get(
                status, "❓"
            )
            step_data.append({
                "Status": emoji,
                "Step": entry.get("step", ""),
                "Duration (s)": entry.get("elapsed_seconds", 0),
                "Time": entry.get("timestamp", ""),
            })

        st.dataframe(pd.DataFrame(step_data), use_container_width=True)

        # Timing chart
        timing_df = pd.DataFrame([
            {"step": e["step"], "seconds": e.get("elapsed_seconds", 0)}
            for e in pipeline_log
            if e.get("elapsed_seconds", 0) > 0
        ])
        if not timing_df.empty:
            chart = (
                alt.Chart(timing_df)
                .mark_bar()
                .encode(
                    x=alt.X("seconds:Q", title="Seconds"),
                    y=alt.Y("step:N", sort="-x", title="Step"),
                    color=alt.value("#ff6b35"),
                )
            )
            st.markdown("#### Step Timing")
            altair_chart_compat(chart)

        total_time = sum(e.get("elapsed_seconds", 0) for e in pipeline_log)
        st.metric("Total pipeline time", f"{total_time:.1f}s")
    else:
        st.info("No pipeline log found for this run.")

    # Output files
    st.subheader("Output Files")
    output_files = sorted(run_dir.glob("*.jsonl")) + sorted(run_dir.glob("*.json"))
    if output_files:
        file_info = []
        for f in output_files:
            size_kb = f.stat().st_size / 1024
            file_info.append({"File": f.name, "Size (KB)": f"{size_kb:.1f}"})
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    else:
        st.info("No output files found.")

# =========================================================================
# PAGE: Quality Deep-Dive
# =========================================================================
elif page == "Quality Deep-Dive":
    st.title("🔍 Quality Deep-Dive")
    st.markdown(
        "Explore judge scores, contamination results, difficulty distribution, "
        "and quality metrics."
    )

    run_dirs = find_run_dirs()
    if not run_dirs:
        st.info("No runs found.")
        st.stop()

    run_labels = [d.name for d in run_dirs]
    selected_run_name = st.selectbox("Select run", run_labels, key="quality_run")
    run_dir = Path("runs") / selected_run_name

    # Try to find judged data
    judged_path = run_dir / "judged.jsonl"
    quality_path = run_dir / "quality.jsonl"
    difficulty_path = run_dir / "difficulty.jsonl"
    data_path = None
    for candidate in [difficulty_path, judged_path, quality_path]:
        if candidate.exists():
            data_path = candidate
            break

    if data_path is None:
        # Fallback to any JSONL in the run
        jsonl_files = list(run_dir.glob("*.jsonl"))
        if jsonl_files:
            data_path = jsonl_files[0]

    if data_path is None:
        st.info("No dataset found in this run.")
        st.stop()

    df = load_jsonl_df(data_path)
    if df.empty:
        st.info("Dataset is empty.")
        st.stop()

    st.caption(f"Loaded {len(df)} examples from `{data_path.name}`")

    # --- Judge scores ---
    if "judge_avg_score" in df.columns:
        st.subheader("Judge Scores")
        j_col1, j_col2, j_col3 = st.columns(3)
        scores = pd.to_numeric(df["judge_avg_score"], errors="coerce").dropna()
        if not scores.empty:
            j_col1.metric("Mean judge score", f"{scores.mean():.2f}")
            j_col2.metric("Median", f"{scores.median():.2f}")
            j_col3.metric("Std dev", f"{scores.std():.2f}")

            # Distribution
            score_chart = (
                alt.Chart(pd.DataFrame({"score": scores}))
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Judge Score"),
                    y=alt.Y("count():Q", title="Count"),
                    color=alt.value("#4ecdc4"),
                )
            )
            altair_chart_compat(score_chart)

        # Per-dimension scores if available
        judge_dims = ["faithfulness", "helpfulness", "complexity", "coherence"]
        dim_cols = [c for c in df.columns if c.startswith("judge_") and c != "judge_avg_score"]
        if dim_cols:
            st.markdown("#### Per-Dimension Scores")
            dim_data = []
            for col in dim_cols:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty:
                    dim_name = col.replace("judge_", "").replace("_score", "")
                    dim_data.append({"dimension": dim_name, "mean_score": round(vals.mean(), 2)})

            if dim_data:
                dim_df = pd.DataFrame(dim_data)
                dim_chart = (
                    alt.Chart(dim_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("mean_score:Q", title="Mean Score", scale=alt.Scale(domain=[0, 5])),
                        y=alt.Y("dimension:N", sort="-x", title="Dimension"),
                        color=alt.value("#ff6b35"),
                    )
                )
                altair_chart_compat(dim_chart)

    # --- Quality score ---
    if "quality_score" in df.columns:
        st.subheader("Quality Scores")
        qs = pd.to_numeric(df["quality_score"], errors="coerce").dropna()
        if not qs.empty:
            q_col1, q_col2 = st.columns(2)
            q_col1.metric("Mean quality score", f"{qs.mean():.3f}")
            q_col2.metric("Examples with score > 0.7", f"{(qs > 0.7).sum()}")

            qs_chart = (
                alt.Chart(pd.DataFrame({"score": qs}))
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Quality Score"),
                    y=alt.Y("count():Q", title="Count"),
                    color=alt.value("#45b7d1"),
                )
            )
            altair_chart_compat(qs_chart)

    # --- Difficulty distribution ---
    if "difficulty" in df.columns:
        st.subheader("Difficulty Distribution")
        diff_counts = df["difficulty"].value_counts().reset_index()
        diff_counts.columns = ["difficulty", "count"]
        diff_chart = (
            alt.Chart(diff_counts)
            .mark_bar()
            .encode(
                x=alt.X("difficulty:N", sort=["easy", "medium", "hard"], title="Difficulty"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color(
                    "difficulty:N",
                    scale=alt.Scale(
                        domain=["easy", "medium", "hard"],
                        range=["#4ecdc4", "#ff6b35", "#e63946"],
                    ),
                ),
            )
        )
        altair_chart_compat(diff_chart)

    # --- Task distribution ---
    if "task_name" in df.columns:
        st.subheader("Task Distribution")
        task_counts = df["task_name"].value_counts().reset_index()
        task_counts.columns = ["task", "count"]
        task_chart = (
            alt.Chart(task_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("task:N", sort="-x", title="Task"),
                color=alt.value("#6c5ce7"),
            )
        )
        altair_chart_compat(task_chart)

    # --- Contamination ---
    contamination_path = run_dir / "contamination_report.json"
    if contamination_path.exists():
        st.subheader("Contamination Report")
        try:
            report = json.loads(contamination_path.read_text(encoding="utf-8"))
            c_col1, c_col2 = st.columns(2)
            c_col1.metric("Total checked", report.get("total_checked", "N/A"))
            c_col2.metric(
                "Contaminated",
                report.get("contaminated_count", report.get("flagged", "N/A")),
            )
            if "details" in report:
                with st.expander("Contamination details"):
                    st.json(report["details"])
        except (json.JSONDecodeError, OSError):
            st.warning("Could not parse contamination report.")


# =========================================================================
# PAGE: Training Results
# =========================================================================
elif page == "Training Results":
    st.title("🎯 Training Results")
    st.markdown(
        "Compare base vs fine-tuned model performance on held-out test data."
    )

    run_dirs = find_run_dirs()
    if not run_dirs:
        st.info("No runs found.")
        st.stop()

    run_labels = [d.name for d in run_dirs]
    selected_run_name = st.selectbox("Select run", run_labels, key="training_run")
    run_dir = Path("runs") / selected_run_name

    # Benchmark results
    benchmark_path = run_dir / "benchmark_results.json"
    if not benchmark_path.exists():
        st.info(
            "No benchmark results found for this run. "
            "Run the full pipeline with `make forge-live` to generate benchmarks."
        )
        st.stop()

    try:
        results = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        st.error("Could not parse benchmark results.")
        st.stop()

    st.metric("Test examples", results.get("test_examples", "N/A"))

    # --- Base model ---
    base = results.get("base", {})
    if base:
        st.subheader(f"Base Model: {base.get('model', 'N/A')}")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        rouge = base.get("rouge", {})
        b_col1.metric("ROUGE-1", f"{rouge.get('rouge1_f', 0):.4f}")
        b_col2.metric("ROUGE-2", f"{rouge.get('rouge2_f', 0):.4f}")
        b_col3.metric("ROUGE-L", f"{rouge.get('rougeL_f', 0):.4f}")
        b_col4.metric("Exact Match", f"{base.get('exact_match', 0):.4f}")

    # --- Fine-tuned model ---
    ft = results.get("finetuned", {})
    if ft:
        st.subheader(f"Fine-tuned: {ft.get('model', 'N/A')}")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        rouge = ft.get("rouge", {})
        f_col1.metric("ROUGE-1", f"{rouge.get('rouge1_f', 0):.4f}")
        f_col2.metric("ROUGE-2", f"{rouge.get('rouge2_f', 0):.4f}")
        f_col3.metric("ROUGE-L", f"{rouge.get('rougeL_f', 0):.4f}")
        f_col4.metric("Exact Match", f"{ft.get('exact_match', 0):.4f}")

    # --- Comparison ---
    comparison = results.get("comparison", {})
    delta = comparison.get("delta", {})
    if delta:
        st.subheader("Improvement (Fine-tuned vs Base)")

        metrics_list = []
        for metric_name, delta_val in delta.items():
            metrics_list.append({
                "metric": metric_name,
                "delta": delta_val,
                "direction": "+" if delta_val > 0 else "",
            })

        delta_df = pd.DataFrame(metrics_list)
        delta_chart = (
            alt.Chart(delta_df)
            .mark_bar()
            .encode(
                x=alt.X("delta:Q", title="Delta"),
                y=alt.Y("metric:N", sort="-x", title="Metric"),
                color=alt.condition(
                    alt.datum.delta > 0,
                    alt.value("#4ecdc4"),
                    alt.value("#e63946"),
                ),
            )
        )
        altair_chart_compat(delta_chart)

        # Significance
        significance = comparison.get("significance", {})
        if significance:
            sig_str = "Yes" if significance.get("significant") else "No"
            st.markdown(
                f"**Statistical significance (p<0.05):** {sig_str} "
                f"(p={significance.get('p_value', 'N/A')})"
            )

    # --- Side-by-side comparison chart ---
    if base and ft:
        st.subheader("Side-by-Side Comparison")
        compare_data = []
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f"]:
            compare_data.append({"metric": metric, "model": "Base", "score": base.get("rouge", {}).get(metric, 0)})
            compare_data.append({"metric": metric, "model": "Fine-tuned", "score": ft.get("rouge", {}).get(metric, 0)})
        compare_data.append({"metric": "exact_match", "model": "Base", "score": base.get("exact_match", 0)})
        compare_data.append({"metric": "exact_match", "model": "Fine-tuned", "score": ft.get("exact_match", 0)})

        compare_df = pd.DataFrame(compare_data)
        compare_chart = (
            alt.Chart(compare_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                y=alt.Y("score:Q", title="Score"),
                color=alt.Color("model:N", scale=alt.Scale(range=["#6c5ce7", "#ff6b35"])),
                xOffset="model:N",
            )
        )
        altair_chart_compat(compare_chart)

    # Finetune config
    ft_config_path = run_dir / "finetune" / "config.json"
    if ft_config_path.exists():
        with st.expander("Fine-tuning configuration"):
            st.json(json.loads(ft_config_path.read_text(encoding="utf-8")))


# =========================================================================
# PAGE: Experiment Comparison
# =========================================================================
elif page == "Experiment Comparison":
    st.title("🔬 Experiment Comparison")
    st.markdown("Compare metrics across multiple pipeline runs.")

    run_dirs = find_run_dirs()
    if not run_dirs:
        st.info("No runs found.")
        st.stop()

    run_labels = [d.name for d in run_dirs]
    selected_runs = st.multiselect(
        "Select runs to compare",
        run_labels,
        default=run_labels[:min(3, len(run_labels))],
    )

    if not selected_runs:
        st.info("Select at least one run.")
        st.stop()

    # Build comparison table
    rows = []
    for run_name in selected_runs:
        rdir = Path("runs") / run_name
        config = load_run_config(rdir)
        metrics = load_run_metrics(rdir)
        pipeline_log = load_pipeline_log(rdir)

        total_time = sum(e.get("elapsed_seconds", 0) for e in pipeline_log)
        num_steps_ok = sum(1 for e in pipeline_log if e.get("status") == "ok")

        # Count examples from the largest JSONL
        example_count = 0
        for jsonl_file in rdir.glob("*.jsonl"):
            try:
                count = sum(1 for line in jsonl_file.open() if line.strip())
                example_count = max(example_count, count)
            except OSError:
                pass

        row = {
            "Run": run_name,
            "Tasks": config.get("tasks", "N/A"),
            "Model": config.get("model", "dry-run"),
            "Examples": example_count,
            "Steps OK": num_steps_ok,
            "Total Time (s)": f"{total_time:.1f}",
            "Status": config.get("status", "unknown"),
        }
        row.update(metrics)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Benchmark comparison across runs ---
    benchmark_rows = []
    for run_name in selected_runs:
        rdir = Path("runs") / run_name
        bp = rdir / "benchmark_results.json"
        if bp.exists():
            try:
                bdata = json.loads(bp.read_text(encoding="utf-8"))
                base_rouge = bdata.get("base", {}).get("rouge", {})
                ft_rouge = bdata.get("finetuned", {}).get("rouge", {})
                delta = bdata.get("comparison", {}).get("delta", {})
                benchmark_rows.append({
                    "Run": run_name,
                    "Base ROUGE-L": base_rouge.get("rougeL_f", ""),
                    "FT ROUGE-L": ft_rouge.get("rougeL_f", ""),
                    "Delta ROUGE-L": delta.get("rougeL_f", ""),
                    "Delta EM": delta.get("exact_match", ""),
                })
            except (json.JSONDecodeError, OSError):
                pass

    if benchmark_rows:
        st.subheader("Benchmark Comparison")
        st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True)


# =========================================================================
# PAGE: Dataset Explorer (original dashboard functionality)
# =========================================================================
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown(
        "Browse and curate training data examples from any JSONL dataset."
    )

    # --- Dataset selection ---
    st.sidebar.header("Dataset Selection")

    # Check runs for datasets
    run_dirs = find_run_dirs()
    dataset_options: Dict[str, Path] = {}

    for rdir in run_dirs:
        for jsonl in sorted(rdir.glob("*.jsonl")):
            label = f"{rdir.name}/{jsonl.name}"
            dataset_options[label] = jsonl

    # Also check output/ for legacy datasets
    output_dir = Path("output")
    if output_dir.exists():
        for jsonl in sorted(output_dir.glob("*.jsonl")):
            dataset_options[f"output/{jsonl.name}"] = jsonl

    uploaded_file = st.sidebar.file_uploader(
        "Upload a JSONL file",
        type=["jsonl"],
    )

    df: pd.DataFrame

    if uploaded_file is not None:
        records: List[Dict[str, Any]] = []
        for raw_line in uploaded_file.readlines():
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        df = pd.DataFrame(records)
        dataset_label = f"Uploaded: {uploaded_file.name}"
    elif dataset_options:
        selected_key = st.sidebar.selectbox(
            "Select dataset",
            list(dataset_options.keys()),
        )
        path = dataset_options[selected_key]
        df = load_jsonl_df(path)
        dataset_label = selected_key
    else:
        custom_path = st.sidebar.text_input("Path to JSONL", "output/dataset.jsonl")
        path = Path(custom_path).expanduser()
        if not path.exists():
            st.error(f"File not found: {path}")
            st.stop()
        df = load_jsonl_df(path)
        dataset_label = str(path)

    st.sidebar.caption(f"Using: {dataset_label}")

    if df.empty:
        st.warning("No records found.")
        st.stop()

    # --- Normalise schema ---
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"].astype(str)
        elif "question" in df.columns:
            df["input_text"] = df["question"].astype(str)
        elif "context" in df.columns:
            df["input_text"] = df["context"].astype(str)
        else:
            df["input_text"] = ""

    if "output_text" not in df.columns:
        if "output" in df.columns:
            df["output_text"] = df["output"].astype(str)
        elif "answer" in df.columns:
            df["output_text"] = df["answer"].astype(str)
        else:
            df["output_text"] = ""

    for col in ["task_name", "task_type", "id"]:
        if col not in df.columns:
            df[col] = ""

    if df["id"].astype(str).str.strip().eq("").all():
        df["id"] = [f"ex-{i:05d}" for i in range(1, len(df) + 1)]

    df["input_length"] = df["input_text"].astype(str).str.len()
    df["output_length"] = df["output_text"].astype(str).str.len()

    # --- Overview ---
    st.subheader("Overview")

    num_examples = len(df)
    task_counts = Counter(df["task_name"].astype(str))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total examples", f"{num_examples:,}")
    col2.metric("Task types", len(task_counts))
    col3.metric(
        "Avg output length",
        f"{df['output_length'].mean():.0f} chars",
    )

    st.markdown("#### Examples per task")
    task_df = (
        pd.DataFrame([{"task": k, "count": v} for k, v in task_counts.items()])
        .sort_values("count", ascending=False)
    )
    if not task_df.empty:
        st.bar_chart(task_df.set_index("task")["count"])

    # Length distributions
    st.markdown("#### Length distributions")
    len_col1, len_col2 = st.columns(2)
    with len_col1:
        st.caption("Input length")
        chart_in = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("input_length:Q", bin=alt.Bin(maxbins=30), title="Input length"),
                alt.Y("count():Q", title="Count"),
            )
        )
        altair_chart_compat(chart_in)

    with len_col2:
        st.caption("Output length")
        chart_out = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("output_length:Q", bin=alt.Bin(maxbins=30), title="Output length"),
                alt.Y("count():Q", title="Count"),
            )
        )
        altair_chart_compat(chart_out)

    # --- Browse examples ---
    st.subheader("Browse examples")

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    task_options = ["(all)"] + sorted(task_counts.keys())
    with fcol1:
        selected_task = st.selectbox("Filter by task", task_options)
    with fcol2:
        if "difficulty" in df.columns:
            diff_options = ["(all)"] + sorted(df["difficulty"].dropna().unique())
            selected_diff = st.selectbox("Filter by difficulty", diff_options)
        else:
            selected_diff = "(all)"
    with fcol3:
        search_text = st.text_input("Search text", "")

    filtered = df.copy()
    if selected_task != "(all)":
        filtered = filtered[filtered["task_name"] == selected_task]
    if selected_diff != "(all)" and "difficulty" in filtered.columns:
        filtered = filtered[filtered["difficulty"] == selected_diff]
    if search_text:
        s = search_text.lower()
        mask = (
            filtered["input_text"].astype(str).str.lower().str.contains(s)
            | filtered["output_text"].astype(str).str.lower().str.contains(s)
        )
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered):,} of {num_examples:,} examples.")

    display_cols = [c for c in [
        "id", "task_name", "difficulty", "judge_avg_score",
        "quality_score", "input_text", "output_text",
    ] if c in filtered.columns]

    st.dataframe(filtered[display_cols], use_container_width=True)

    # Download
    if not filtered.empty:
        records = filtered.to_dict(orient="records")
        jsonl_str = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str) for r in records
        )
        st.download_button(
            "Download filtered as JSONL",
            data=jsonl_str,
            file_name="filtered_dataset.jsonl",
            mime="application/json",
        )

    # --- Single example inspector ---
    st.subheader("Example Inspector")
    example_ids = filtered["id"].astype(str).tolist()
    if not example_ids:
        st.info("No examples to inspect.")
    else:
        selected_id = st.selectbox("Select example", example_ids)
        row = filtered[filtered["id"].astype(str) == selected_id].iloc[0]

        meta_cols = [c for c in [
            "task_name", "task_type", "model_name", "difficulty",
            "judge_avg_score", "quality_score", "document_id",
        ] if c in row.index and str(row.get(c, "")).strip()]

        for col in meta_cols:
            st.markdown(f"**{col}:** {row[col]}")

        st.markdown("##### Input")
        st.code(str(row.get("input_text", "")), language="markdown")
        st.markdown("##### Output")
        st.code(str(row.get("output_text", "")), language="markdown")

        # Curation labels
        labels = st.session_state["labels"]
        existing = labels.get(selected_id, {})

        label_choice = st.radio(
            "Label",
            ["unlabeled", "keep", "drop"],
            index=["unlabeled", "keep", "drop"].index(
                existing.get("label", "unlabeled")
            ),
            horizontal=True,
        )

        if label_choice != "unlabeled":
            labels[selected_id] = {"label": label_choice}
        elif selected_id in labels:
            del labels[selected_id]

        st.session_state["labels"] = labels
