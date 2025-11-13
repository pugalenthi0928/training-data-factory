.PHONY: test
test:
\tpython -m py_compile app.py \
\t\tscripts/run_profile.py scripts/evaluate_qa.py scripts/postprocess_quality.py \
\t\tscripts/export_finetune.py scripts/export_rag_qa.py scripts/run_qa_eval_model.py \
\t\tscripts/compute_dedupe.py scripts/log_metrics.py
\tpytest

.PHONY: report
report:
\tpython scripts/plot_runs.py
