from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List

from . import Settings, TaskTemplate, TaskType, TrainingDataBot


def build_task_templates_from_names(names: List[str]) -> List[TaskTemplate]:
    """
    Map simple names like 'summary' / 'qa' / 'key_points' / 'title'
    to concrete TaskTemplate objects.
    """
    templates: List[TaskTemplate] = []

    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue

        if name == "summary":
            templates.append(
                TaskTemplate(
                    name="summary_v1",
                    version="v1",
                    task_type=TaskType.SUMMARISATION,
                    system_prompt="You are a helpful assistant that summarises text.",
                    user_prompt_template="Summarise the following text:\n\n{text}",
                    max_output_tokens=256,
                    temperature=0.2,
                    top_p=1.0,
                )
            )
        elif name == "qa":
            templates.append(
                TaskTemplate(
                    name="qa_v1",
                    version="v1",
                    task_type=TaskType.QA,
                    system_prompt=("You are a helpful assistant that writes question-answer pairs from text."),
                    user_prompt_template=(
                        "Read the following passage and generate ONE useful question "
                        "and its answer.\n\nPassage:\n\n{text}"
                    ),
                    max_output_tokens=256,
                    temperature=0.3,
                    top_p=1.0,
                )
            )
        elif name == "key_points":
            templates.append(
                TaskTemplate(
                    name="key_points_v1",
                    version="v1",
                    task_type=TaskType.KEY_POINTS,
                    system_prompt=("You extract the most important bullet-point key ideas from the passage."),
                    user_prompt_template=(
                        "Read the following text and extract 3-5 concise bullet point key ideas.\n\n{text}"
                    ),
                    max_output_tokens=256,
                    temperature=0.3,
                    top_p=1.0,
                )
            )
        elif name == "title":
            templates.append(
                TaskTemplate(
                    name="title_v1",
                    version="v1",
                    task_type=TaskType.TITLE,
                    system_prompt="You write short, descriptive titles for texts.",
                    user_prompt_template=(
                        "Write a short (max 12 words) descriptive title for the following text:\n\n{text}"
                    ),
                    max_output_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                )
            )
        elif name in ("instruction", "instruction_following"):
            templates.append(
                TaskTemplate(
                    name="instruction_v1",
                    version="v1",
                    task_type=TaskType.INSTRUCTION_FOLLOWING,
                    system_prompt=(
                        "You are an expert at creating realistic user instructions "
                        "and ideal assistant responses. Given a passage, generate "
                        "a plausible user instruction that someone might ask about "
                        "the topic, and then provide a thorough, helpful response "
                        "grounded in the passage content."
                    ),
                    user_prompt_template=(
                        "Based on the following passage, generate a realistic user "
                        "instruction (a task or request someone might ask an AI "
                        "assistant) and then write the ideal response.\n\n"
                        "Format your output as:\n"
                        "INSTRUCTION: <the user instruction>\n"
                        "RESPONSE: <the ideal assistant response>\n\n"
                        "Passage:\n\n{text}"
                    ),
                    max_output_tokens=512,
                    temperature=0.4,
                    top_p=1.0,
                )
            )
        elif name in ("cot", "chain_of_thought"):
            templates.append(
                TaskTemplate(
                    name="cot_v1",
                    version="v1",
                    task_type=TaskType.CHAIN_OF_THOUGHT,
                    system_prompt=(
                        "You are an expert at creating multi-step reasoning "
                        "questions. Given a passage, write a question that requires "
                        "combining multiple facts or performing logical inference, "
                        "then provide a step-by-step chain-of-thought answer."
                    ),
                    user_prompt_template=(
                        "Based on the following passage, create a question that "
                        "requires multi-step reasoning to answer. Then provide "
                        "a detailed chain-of-thought answer.\n\n"
                        "Format your output as:\n"
                        "QUESTION: <a reasoning question>\n"
                        "REASONING:\n"
                        "Step 1: <first reasoning step>\n"
                        "Step 2: <second reasoning step>\n"
                        "...\n"
                        "ANSWER: <final answer>\n\n"
                        "Passage:\n\n{text}"
                    ),
                    max_output_tokens=512,
                    temperature=0.3,
                    top_p=1.0,
                )
            )
        else:
            # Ignore unknown names; we'll validate below.
            continue

    if not templates:
        raise SystemExit(f"No valid tasks in {names!r}. Supported: summary, qa, key_points, title, instruction, cot.")

    return templates


async def handle_process(args: argparse.Namespace) -> None:
    # Parse sources (allow multiple --source flags)
    sources = [Path(s) for s in args.source]

    # Build task templates
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_templates = build_task_templates_from_names(task_names)

    # Optional model override
    settings = Settings(default_model=args.model) if args.model else Settings()

    async with TrainingDataBot(settings=settings) as bot:
        dataset_path = await bot.run_pipeline(
            sources=sources,
            output_path=args.out,
            tasks=task_templates,
            max_examples=args.max_examples,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )

        # We just created one dataset
        dataset_id = next(iter(bot.datasets.keys()))
        stats = await bot.evaluate_dataset(dataset_id)

        print(f"Dataset saved to: {dataset_path}")
        print("Dataset stats:")
        print(json.dumps(stats, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tdr",
        description="Forge - generate source-grounded training datasets.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- process --------
    p_process = subparsers.add_parser(
        "process",
        help="Load sources, generate synthetic training examples, export JSONL.",
    )
    p_process.add_argument(
        "-s",
        "--source",
        required=True,
        action="append",
        help="Source folder or file. Use multiple -s flags for multiple sources.",
    )
    p_process.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output JSONL file path (e.g. output/dataset.jsonl).",
    )
    p_process.add_argument(
        "-t",
        "--tasks",
        default="summary",
        help=(
            "Comma-separated tasks to run. "
            "Supported: summary, qa, key_points, title, instruction, cot (default: summary)."
        ),
    )
    p_process.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Approximate maximum characters per chunk (default: 800).",
    )
    p_process.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Approximate overlap (in characters) between chunks (default: 100).",
    )
    p_process.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional hard cap on total generated examples.",
    )
    p_process.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override default model name (e.g. gpt-4.1-mini).",
    )

    # -------- export-finetune --------
    p_ft = subparsers.add_parser(
        "export-finetune",
        help="Convert a dataset JSONL into a fine-tuning JSONL.",
    )
    p_ft.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Input dataset JSONL produced by `tdr process`.",
    )
    p_ft.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Output JSONL path for fine-tuning examples.",
    )
    p_ft.add_argument(
        "--format",
        dest="format",
        choices=["chat", "io"],
        default="chat",
        help="Fine-tune format: 'chat' (OpenAI-style) or 'io' (input/output pairs).",
    )

    # -------- export-rag-qa --------
    p_rag = subparsers.add_parser(
        "export-rag-qa",
        help="Convert QA examples into RAG-friendly question/answer/context JSONL.",
    )
    p_rag.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Input dataset JSONL (should contain qa_v1 examples).",
    )
    p_rag.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Output JSONL path for RAG QA records.",
    )
    p_rag.add_argument(
        "--task-name",
        dest="task_name",
        default="qa_v1",
        help="Task name to treat as QA (default: qa_v1).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        asyncio.run(handle_process(args))
    elif args.command == "export-finetune":
        from .exporters import export_finetune

        export_finetune(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            fmt=args.format,
        )
    elif args.command == "export-rag-qa":
        from .exporters import export_rag_qa

        export_rag_qa(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            qa_task_name=args.task_name,
        )
    else:
        parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
