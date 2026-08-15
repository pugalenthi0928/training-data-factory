"""Stage 3 source-rights, schema, and privacy control tests."""

from __future__ import annotations

import json
from pathlib import Path

from forge.governance import govern_documents, govern_records, redact_pii, scan_pii


def test_structured_pii_is_detected_without_storing_raw_values() -> None:
    text = "Contact alex@example.com and use card 4111 1111 1111 1111 only in the test fixture."
    findings = scan_pii(text)

    assert {finding.entity_type for finding in findings} == {"email_address", "payment_card"}
    assert all("alex" not in finding.text_sha256 for finding in findings)
    redacted = redact_pii(text, findings)
    assert "alex@example.com" not in redacted
    assert "4111 1111 1111 1111" not in redacted


def test_candidate_source_rights_are_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("A clean source document", encoding="utf-8")
    document = {"id": "doc_1", "path": str(source), "content": source.read_text(), "metadata": {}}

    kept, rejected, report = govern_documents(
        [document],
        source_manifest=None,
        required_use="training",
        pii_action="reject",
        allow_unknown_rights=False,
    )

    assert kept == []
    assert len(rejected) == 1
    assert report["status"] == "failed"
    assert rejected[0]["forge_audit"]["decisions"][-1]["reason_codes"] == ["rights.unknown"]


def test_declared_permitted_source_is_kept_with_machine_readable_rights(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("A clean source document", encoding="utf-8")
    manifest = tmp_path / "rights.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "source_path": "source.txt",
                    "origin": "controlled fixture",
                    "license": "MIT",
                    "permitted_uses": ["training", "evaluation"],
                    "rights_holder": "fixture author",
                }
            ]
        ),
        encoding="utf-8",
    )
    document = {"id": "doc_1", "path": str(source), "content": source.read_text(), "metadata": {}}

    kept, rejected, report = govern_documents(
        [document],
        source_manifest=manifest,
        required_use="training",
        pii_action="reject",
        allow_unknown_rights=False,
    )

    assert rejected == []
    assert report["status"] == "passed"
    assert kept[0]["source_rights"]["license"] == "MIT"
    assert "training" in kept[0]["source_rights"]["permitted_uses"]


def test_exact_duplicate_documents_are_quarantined_with_reason(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same controlled source", encoding="utf-8")
    second.write_text("same controlled source", encoding="utf-8")
    rows = [
        {"id": "doc_same", "path": str(first), "content": first.read_text(), "metadata": {}},
        {"id": "doc_same", "path": str(second), "content": second.read_text(), "metadata": {}},
    ]

    kept, rejected, report = govern_documents(
        rows,
        source_manifest=None,
        required_use="training",
        pii_action="reject",
        allow_unknown_rights=True,
    )

    assert len(kept) == 1
    assert len(rejected) == 1
    reasons = rejected[0]["forge_audit"]["decisions"][-1]["reason_codes"]
    assert "duplicate.document_exact" in reasons
    assert report["status"] == "passed"


def test_record_governance_redacts_pii_and_rejects_invalid_schema() -> None:
    rows = [
        {
            "id": "valid",
            "document_id": "doc_1",
            "chunk_id": "chunk_1",
            "task_name": "qa",
            "input_text": "Email me at analyst@example.com",
            "output_text": "Acknowledged",
            "quality_score": 1.0,
        },
        {
            "id": "invalid",
            "document_id": "",
            "chunk_id": "chunk_2",
            "task_name": "qa",
            "input_text": "Question",
            "output_text": "Answer",
            "quality_score": 1.0,
        },
    ]

    kept, rejected, report = govern_records(rows, pii_action="redact")

    assert len(kept) == 1
    assert "analyst@example.com" not in kept[0]["input_text"]
    assert len(rejected) == 1
    reasons = rejected[0]["forge_audit"]["decisions"][-1]["reason_codes"]
    assert "schema.missing_document_id" in reasons
    assert report["remaining_pii_findings"] == 0


def test_rejected_record_quarantine_does_not_retain_raw_identifier() -> None:
    row = {
        "id": "pii",
        "document_id": "doc_1",
        "chunk_id": "chunk_1",
        "task_name": "qa",
        "input_text": "Contact private@example.com",
        "output_text": "Acknowledged",
        "quality_score": 1.0,
    }

    kept, rejected, report = govern_records([row], pii_action="reject")

    assert kept == []
    assert "private@example.com" not in rejected[0]["input_text"]
    assert report["quarantine_redactions"] == 1
