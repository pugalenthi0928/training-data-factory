from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import altair as alt
import pandas as pd
import streamlit as st


def load_jsonl(path: Path) -> pd.DataFrame:
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
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def find_default_dataset() -> Path | None:
    candidates = [
        Path("output/dataset_cli_rich_200_quality.jsonl"),
        Path("output/dataset_cli_rich.jsonl"),
        Path("output/dataset_cli_rich_200.jsonl"),
        Path("output/dataset_cli_summary_qa.jsonl"),
        Path("output/dataset_cli_summary.jsonl"),
        Path("output/dataset_real.jsonl"),
        Path("output/dataset.jsonl"),
        Path("output/papers_rich_summary_qa_300_quality.jsonl"),
        Path("output/papers_rich_summary_qa_300.jsonl"),
        Path("output/papers_qa_only_real_gpt4_quality.jsonl"),
        Path("output/papers_qa_only_real_gpt4.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def altair_chart_compat(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


st.set_page_config(
    page_title="Training Data Robo – Dataset Dashboard",
    layout="wide",
)

# Session state for interactive curation
if "labels" not in st.session_state:
    st.session_state["labels"] = {}

st.title("📊 Training Data Robo – Dataset Dashboard")
st.markdown(
    "Explore your generated training datasets (summaries, QA pairs, key points, titles, etc.) "
    "created by the `tdr` CLI."
)

st.sidebar.header("⚙️ Dataset selection")

default_path = find_default_dataset()
default_str = str(default_path) if default_path is not None else "output/dataset.jsonl"

dataset_path_str = st.sidebar.text_input(
    "Path to JSONL dataset",
    value=default_str,
    help="Relative path from the project root, e.g. `output/dataset_cli_summary_qa.jsonl`",
)

uploaded_file = st.sidebar.file_uploader(
    "…or upload a JSONL file",
    type=["jsonl"],
    help="You can either type a path above or upload a JSONL file here.",
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
    dataset_label = f"Uploaded file: {uploaded_file.name}"
else:
    path = Path(dataset_path_str).expanduser()
    if not path.exists():
        st.error(f"File not found: {path}")
        st.stop()
    df = load_jsonl(path)
    dataset_label = f"File: {path}"

st.sidebar.caption(f"Using dataset: {dataset_label}")

if df.empty:
    st.warning("No records found in this dataset.")
    st.stop()

# ---------- Normalise schema ----------

if "input_text" not in df.columns:
    if "input" in df.columns:
        df["input_text"] = df["input"].astype(str)
    elif "question" in df.columns or "context" in df.columns:
        if "context" in df.columns:
            df["input_text"] = df["context"].astype(str)
        else:
            df["input_text"] = df["question"].astype(str)
    else:
        df["input_text"] = ""

if "output_text" not in df.columns:
    if "output" in df.columns:
        df["output_text"] = df["output"].astype(str)
    elif "answer" in df.columns:
        df["output_text"] = df["answer"].astype(str)
    else:
        df["output_text"] = ""

if "task_name" not in df.columns:
    if "question" in df.columns and "answer" in df.columns:
        df["task_name"] = "rag_qa"
    else:
        df["task_name"] = ""

if "task_type" not in df.columns:
    if "question" in df.columns and "answer" in df.columns:
        df["task_type"] = "qa"
    else:
        df["task_type"] = ""

expected_cols = [
    "id",
    "task_name",
    "task_type",
    "input_text",
    "output_text",
    "document_id",
    "chunk_id",
    "model_name",
    "task_version",
    "temperature",
    "created_at",
    "metadata",
    "quality_flags",
    "quality_score",
]
for col in expected_cols:
    if col not in df.columns:
        df[col] = ""

if df["id"].astype(str).str.strip().eq("").all():
    df["id"] = [f"ex-{i:05d}" for i in range(1, len(df) + 1)]

df["input_length"] = df["input_text"].astype(str).str.len()
df["output_length"] = df["output_text"].astype(str).str.len()

df["created_at_parsed"] = pd.to_datetime(
    df["created_at"], errors="coerce"
)

# ---------- Overview ----------

st.subheader("Overview")

num_examples = len(df)
task_counts = Counter(df["task_name"].astype(str))
type_counts = Counter(df["task_type"].astype(str))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total examples", f"{num_examples:,}")
col2.metric("Distinct tasks", len(task_counts))
col3.metric("Distinct task types", len(type_counts))

num_docs = df["document_id"].astype(str).nunique() if "document_id" in df.columns else 0
col4.metric("Distinct documents", num_docs)

st.markdown("#### Examples per task")
task_df = (
    pd.DataFrame(
        [{"task_name": k, "count": v} for k, v in task_counts.items()]
    )
    .sort_values("count", ascending=False)
)
if not task_df.empty:
    st.bar_chart(task_df.set_index("task_name")["count"])
else:
    st.info("No task_name information available to plot.")

st.markdown("#### Length distribution (characters)")
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

if df["created_at_parsed"].notna().any():
    st.markdown("#### Examples over time")
    time_df = (
        df.dropna(subset=["created_at_parsed"])
        .assign(date=lambda d: d["created_at_parsed"].dt.date)
        .groupby("date")
        .size()
        .reset_index(name="count")
    )
    time_chart = (
        alt.Chart(time_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Examples"),
        )
    )
    altair_chart_compat(time_chart)

st.markdown("#### Per-task length & quality stats")
per_task_stats = (
    df.groupby("task_name")
    .agg(
        num_examples=("id", "count"),
        avg_input_len=("input_length", "mean"),
        avg_output_len=("output_length", "mean"),
        avg_quality_score=("quality_score", "mean"),
    )
    .reset_index()
)
st.dataframe(per_task_stats)

# ---------- Browse examples ----------

st.subheader("Browse examples")

fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 2])

task_options = ["(all)"] + sorted(task_counts.keys())
with fcol1:
    selected_task = st.selectbox("Filter by task_name", task_options)

doc_options = ["(all)"]
if "document_id" in df.columns:
    doc_options += sorted(df["document_id"].astype(str).unique())
with fcol2:
    selected_doc = st.selectbox("Filter by document_id", doc_options)

model_options = ["(all)"]
if "model_name" in df.columns:
    non_empty_models = sorted(
        m for m in df["model_name"].astype(str).unique() if m.strip()
    )
    model_options += non_empty_models

with fcol3:
    selected_model = st.selectbox("Filter by model_name", model_options)

with fcol4:
    search_text = st.text_input(
        "Search in input or output text",
        "",
        help="Case-insensitive substring search.",
    )

filtered = df.copy()

if selected_task != "(all)":
    filtered = filtered[filtered["task_name"] == selected_task]

if selected_doc != "(all)":
    filtered = filtered[filtered["document_id"].astype(str) == selected_doc]

if selected_model != "(all)":
    filtered = filtered[filtered["model_name"].astype(str) == selected_model]

if search_text:
    s = search_text.lower()
    mask = (
        filtered["input_text"].astype(str).str.lower().str.contains(s)
        | filtered["output_text"].astype(str).str.lower().str.contains(s)
    )
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered):,} of {num_examples:,} examples.")

display_cols = [
    "id",
    "task_name",
    "task_type",
    "model_name",
    "task_version",
    "document_id",
    "quality_score",
    "quality_flags",
    "input_text",
    "output_text",
]
existing_display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(filtered[existing_display_cols])

if not filtered.empty:
    records = filtered.to_dict(orient="records")
    jsonl_str = "\n".join(
        json.dumps(r, ensure_ascii=False, default=str) for r in records
    )
    csv_str = filtered.to_csv(index=False)

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇️ Download filtered as JSONL",
            data=jsonl_str,
            file_name="filtered_dataset.jsonl",
            mime="application/json",
        )
    with dcol2:
        st.download_button(
            "⬇️ Download filtered as CSV",
            data=csv_str,
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )

# ---------- Single example inspector ----------

st.subheader("Single example inspector")

example_ids = filtered["id"].astype(str).tolist()
if not example_ids:
    st.info("No examples available to inspect with current filters.")
else:
    selected_id = st.selectbox("Select example ID", example_ids)
    row = filtered[filtered["id"].astype(str) == selected_id].iloc[0]

    st.markdown("**Task name:** " + str(row.get("task_name", "")))
    st.markdown("**Task type:** " + str(row.get("task_type", "")))
    st.markdown("**Model name:** " + str(row.get("model_name", "")))
    st.markdown("**Task version:** " + str(row.get("task_version", "")))
    st.markdown("**Temperature:** " + str(row.get("temperature", "")))
    st.markdown("**Document ID:** " + str(row.get("document_id", "")))
    st.markdown("**Chunk ID:** " + str(row.get("chunk_id", "")))
    st.markdown("**Created at:** " + str(row.get("created_at", "")))

    qs = row.get("quality_score", "")
    if isinstance(qs, (int, float)):
        st.markdown(f"**Quality score:** {qs:.2f}")
    elif qs != "":
        st.markdown("**Quality score:** " + str(qs))

    flags_val = row.get("quality_flags", [])
    if isinstance(flags_val, str) and flags_val.strip():
        try:
            parsed = json.loads(flags_val)
            flags_val = parsed
        except json.JSONDecodeError:
            flags_val = [flags_val]
    if isinstance(flags_val, list) and flags_val:
        st.markdown("**Quality flags:** " + ", ".join(str(f) for f in flags_val))

    metadata_val = row.get("metadata", {})
    try:
        if isinstance(metadata_val, str) and metadata_val.strip():
            try:
                metadata_val = json.loads(metadata_val)
            except json.JSONDecodeError:
                pass
        pretty_meta = json.dumps(metadata_val, indent=2, ensure_ascii=False)
    except TypeError:
        pretty_meta = str(metadata_val)

    if pretty_meta and pretty_meta != "null":
        st.markdown("##### Metadata")
        st.code(pretty_meta, language="json")

    st.markdown("##### Input text")
    st.code(str(row.get("input_text", "")), language="markdown")

    st.markdown("##### Output text")
    st.code(str(row.get("output_text", "")), language="markdown")

    # Interactive curation controls
    labels = st.session_state["labels"]
    existing_info = labels.get(selected_id, {})
    existing_label = existing_info.get("label", "unlabeled")
    existing_note = existing_info.get("note", "")

    label_options = ["unlabeled", "keep", "drop"]
    try:
        default_index = label_options.index(existing_label)
    except ValueError:
        default_index = 0

    label_choice = st.radio(
        "Label this example",
        label_options,
        index=default_index,
        horizontal=True,
    )

    note_text = st.text_input(
        "Optional note/tag for this example",
        value=existing_note,
    )

    if label_choice != "unlabeled" or note_text:
        labels[selected_id] = {"label": label_choice, "note": note_text}
    elif selected_id in labels:
        # Remove label if switched back to unlabeled with no note
        del labels[selected_id]

    st.session_state["labels"] = labels

# ---------- Curation export ----------

st.subheader("Curation export")

labels = st.session_state["labels"]
if not labels:
    st.info("No labels yet. Use the 'Label this example' controls above.")
else:
    st.caption(f"Total labeled examples in this session: {len(labels)}")

    curated_df = filtered.copy()
    curated_df["id_str_for_label"] = curated_df["id"].astype(str)

    def _label_for_id(id_str: str):
        info = labels.get(id_str)
        if not info:
            return None
        return info.get("label")

    def _note_for_id(id_str: str):
        info = labels.get(id_str)
        if not info:
            return None
        return info.get("note")

    curated_df["label"] = curated_df["id_str_for_label"].map(_label_for_id)
    curated_df["label_note"] = curated_df["id_str_for_label"].map(_note_for_id)

    labeled_only_df = curated_df[curated_df["label"].notna()].drop(
        columns=["id_str_for_label"]
    )
    full_curated_df = curated_df.drop(columns=["id_str_for_label"])

    st.caption(
        f"Labeled rows in current filter: {len(labeled_only_df):,} "
        f"out of {len(curated_df):,}."
    )

    if labeled_only_df.empty:
        st.info("No labeled rows in the current filtered subset yet.")
    else:
        ccol1, ccol2 = st.columns(2)

        all_records = full_curated_df.to_dict(orient="records")
        labeled_records = labeled_only_df.to_dict(orient="records")

        all_jsonl = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str) for r in all_records
        )
        labeled_jsonl = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str) for r in labeled_records
        )

        all_csv = full_curated_df.to_csv(index=False)
        labeled_csv = labeled_only_df.to_csv(index=False)

        with ccol1:
            st.download_button(
                "⬇️ Download ALL filtered + labels (JSONL)",
                data=all_jsonl,
                file_name="filtered_with_labels_all.jsonl",
                mime="application/json",
            )
            st.download_button(
                "⬇️ Download ALL filtered + labels (CSV)",
                data=all_csv,
                file_name="filtered_with_labels_all.csv",
                mime="text/csv",
            )

        with ccol2:
            st.download_button(
                "⬇️ Download LABELED ONLY (JSONL)",
                data=labeled_jsonl,
                file_name="filtered_with_labels_labeled_only.jsonl",
                mime="application/json",
            )
            st.download_button(
                "⬇️ Download LABELED ONLY (CSV)",
                data=labeled_csv,
                file_name="filtered_with_labels_labeled_only.csv",
                mime="text/csv",
            )

# ---------- RAG QA inspector (optional) ----------

if {"question", "answer", "context"}.issubset(df.columns):
    st.subheader("RAG QA inspector (question / answer / context)")
    rag_df = df.dropna(subset=["question", "answer", "context"]).copy()
    if rag_df.empty:
        st.info("RAG QA columns are present but all rows are empty.")
    else:
        rag_questions = rag_df["question"].astype(str).tolist()
        selected_q = st.selectbox(
            "Select RAG question",
            rag_questions,
            key="rag_question_select",
        )
        rag_row = rag_df[rag_df["question"].astype(str) == selected_q].iloc[0]

        st.markdown("**Question:**")
        st.write(str(rag_row.get("question", "")))

        st.markdown("**Answer (ground truth):**")
        st.write(str(rag_row.get("answer", "")))

        st.markdown("**Context:**")
        st.code(str(rag_row.get("context", "")), language="markdown")

# ---------- Evaluation metrics explorer ----------

st.subheader("Evaluation metrics explorer")

metrics_root = Path("output")
metrics_files = sorted(metrics_root.glob("*metrics.json"))

if not metrics_files:
    st.info("No metrics JSON files found in 'output/'. Run scripts/evaluate_qa.py to generate some.")
else:
    label_to_path = {f.name: f for f in metrics_files}
    selected_metrics_name = st.selectbox(
        "Select metrics file",
        sorted(label_to_path.keys()),
        key="metrics_file_select",
    )
    metrics_path = label_to_path[selected_metrics_name]
    try:
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error(f"Could not parse metrics file: {metrics_path}")
    else:
        st.json(metrics_data)

# ---------- Leaderboard ----------
st.subheader("Leaderboard")
runs_csv = Path("runs/qa_runs.csv")
if runs_csv.exists():
    try:
        runs_df = pd.read_csv(runs_csv)
        if "timestamp" in runs_df.columns:
            runs_df["timestamp"] = pd.to_datetime(runs_df["timestamp"], errors="coerce")
            runs_df = runs_df.sort_values("timestamp", ascending=False)
        st.dataframe(runs_df, use_container_width=True)
        if Path("reports/qa_leaderboard.png").exists():
            st.image("reports/qa_leaderboard.png", caption="QA Runs (EM & ROUGE-L)", use_container_width=True)
    except Exception as e:
        st.error(f"Could not load leaderboard: {e}")
else:
    st.info("No leaderboard yet. Run ./scripts/run_all.sh to generate one.")
