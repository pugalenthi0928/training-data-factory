.PHONY: install lint format typecheck test report curation-calibration evaluation-fixture forge forge-live dashboard clean

install:
	pip install -c constraints-dev.txt -e ".[dev]"

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

typecheck:
	mypy src/forge/ src/training_data_robo/ --ignore-missing-imports

test:
	python -m py_compile app.py scripts/*.py
	pytest

coverage:
	pytest --cov=forge --cov=training_data_robo --cov-report=term-missing

report:
	python scripts/plot_runs.py

curation-calibration:
	python scripts/evaluate_curation.py \
		--fixture sample_benchmarks/curation_calibration_pairs.jsonl \
		--output runs/curation_calibration.json \
		--min-fuzzy-precision 0.8 \
		--min-fuzzy-recall 0.5

evaluation-fixture:
	python scripts/prepare_evaluation.py \
		--items sample_evaluation/controlled_candidates.jsonl \
		--protocol docs/evaluation/annotation-protocol.md \
		--output-dir runs/evaluation_fixture \
		--author "repository controlled fixture" \
		--origin "Stage 4 mechanism test" \
		--independence-status controlled_fixture \
		--generator-family openai \
		--seed 42 \
		--target-items 200 \
		--minimum-overlap-items 4
	python scripts/analyse_evaluation.py \
		--manifest runs/evaluation_fixture/evaluation_manifest.json \
		--annotations sample_evaluation/controlled_annotations.jsonl \
		--judge-predictions sample_evaluation/controlled_judge_predictions.jsonl \
		--output runs/evaluation_fixture/calibration_report.json \
		--require-fixture-alpha 0.7 \
		--require-fixture-position-consistency 0.8

forge:
	python scripts/run_forge.py \
		--source ./sample_docs \
		--output-dir runs/forge_$$(date +%Y%m%d_%H%M%S) \
		--tasks qa,summary,instruction,cot \
		--benchmark-file sample_benchmarks/contamination_smoke_test.jsonl \
		--dry-run

forge-live:
	python scripts/run_forge.py \
		--source ./sample_docs \
		--output-dir runs/forge_$$(date +%Y%m%d_%H%M%S) \
		--tasks qa,summary,instruction,cot \
		--model gpt-4.1-mini \
		--max-examples 200 \
		--benchmark-file "$${FORGE_BENCHMARK_FILE}" \
		--source-manifest "$${FORGE_SOURCE_MANIFEST}" \
		--semantic-backend openai \
		--semantic-model text-embedding-3-small \
		--semantic-revision provider-managed \
		--skip-finetune

dashboard:
	streamlit run app.py

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ src/**/__pycache__ scripts/__pycache__ tests/__pycache__
