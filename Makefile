.PHONY: install lint format typecheck test report curation-calibration forge forge-live dashboard clean

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
