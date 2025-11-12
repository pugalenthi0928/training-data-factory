.PHONY: test
test:
\tpython -m py_compile app.py scripts/run_profile.py scripts/evaluate_qa.py scripts/postprocess_quality.py scripts/export_finetune.py scripts/export_rag_qa.py scripts/run_qa_eval_model.py
\tpytest
