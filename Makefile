.PHONY: test run-dashboard run-sample eval-qa docker-build docker-run

test:
	python -m py_compile app.py scripts/run_profile.py scripts/evaluate_qa.py scripts/postprocess_quality.py
	pytest

run-dashboard:
	streamlit run app.py

run-sample:
	python scripts/run_profile.py --profile small_sample

eval-qa:
	python scripts/evaluate_qa.py --input output/papers_qa_only_real_gpt4.jsonl --output output/papers_qa_only_real_gpt4_metrics.json
	python scripts/evaluate_qa.py --input output/dataset_cli_rich_200.jsonl --output output/dataset_cli_rich_200_metrics.json

docker-build:
	docker build -t training-data-robo .

docker-run:
	docker run -p 8501:8501 training-data-robo
