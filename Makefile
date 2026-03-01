.PHONY: install lint format typecheck test report forge forge-live dashboard clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

typecheck:
	mypy src/training_data_robo/ --ignore-missing-imports

test:
	python -m py_compile app.py scripts/*.py
	pytest

coverage:
	pytest --cov=training_data_robo --cov-report=term-missing

report:
	python scripts/plot_runs.py

forge:
	python scripts/run_forge.py \
		--source ./sample_docs \
		--output-dir runs/forge_$$(date +%Y%m%d_%H%M%S) \
		--tasks qa,summary,instruction,cot \
		--dry-run

forge-live:
	python scripts/run_forge.py \
		--source ./sample_docs \
		--output-dir runs/forge_$$(date +%Y%m%d_%H%M%S) \
		--tasks qa,summary,instruction,cot \
		--model gpt-4.1-mini \
		--max-examples 200

dashboard:
	streamlit run app.py

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ src/**/__pycache__ scripts/__pycache__ tests/__pycache__
