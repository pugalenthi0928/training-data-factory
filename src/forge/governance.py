"""Source-rights, schema, and privacy controls with record-level decisions."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_sha256

AUDIT_SCHEMA = "forge.record-audit/v1"
GOVERNANCE_SCHEMA = "forge.governance/v1"
UNKNOWN_LICENSES = {"", "unknown", "noassertion", "none", "n/a"}


@dataclass(frozen=True)
class PIIFinding:
    entity_type: str
    start: int
    end: int
    text_sha256: str
    detector: str


@dataclass(frozen=True)
class SourcePolicy:
    source_path: str
    origin: str
    license: str
    permitted_uses: tuple[str, ...]
    rights_holder: str


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email_address", re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)),
    (
        "ipv4_address",
        re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"),
    ),
    ("us_ssn", re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)")),
    (
        "phone_number",
        re.compile(r"(?<!\d)(?:\+?61[ -]?)?(?:\(?0?[2-478]\)?[ -]?)\d{4}[ -]?\d{4}(?!\d)"),
    ),
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_TFN_CANDIDATE = re.compile(r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)")


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _tfn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) != 9 or len(set(digits)) == 1:
        return False
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    return sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 11 == 0


def scan_pii(text: str) -> list[PIIFinding]:
    """Detect a conservative set of structured identifiers.

    This deterministic scanner is intentionally not presented as complete PII
    coverage. It is a fail-closed baseline and can be replaced by a stronger
    analyzer at the same governance boundary.
    """
    findings: list[PIIFinding] = []
    occupied: set[tuple[int, int]] = set()
    for entity_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            occupied.add(span)
            findings.append(PIIFinding(entity_type, *span, canonical_sha256(match.group(0)), "forge_regex_v1"))
    for entity_type, pattern, validator in (
        ("payment_card", _CARD_CANDIDATE, _luhn_valid),
        ("australian_tfn", _TFN_CANDIDATE, _tfn_valid),
    ):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in occupied or not validator(match.group(0)):
                continue
            findings.append(PIIFinding(entity_type, *span, canonical_sha256(match.group(0)), "forge_checksum_v1"))
    selected: list[PIIFinding] = []
    covered: set[int] = set()
    for finding in sorted(findings, key=lambda item: (item.start, -(item.end - item.start), item.entity_type)):
        positions = set(range(finding.start, finding.end))
        if positions & covered:
            continue
        selected.append(finding)
        covered.update(positions)
    return sorted(selected, key=lambda item: (item.start, item.end, item.entity_type))


def redact_pii(text: str, findings: Sequence[PIIFinding]) -> str:
    redacted = text
    for finding in sorted(findings, key=lambda item: item.start, reverse=True):
        replacement = f"[{finding.entity_type.upper()}]"
        redacted = redacted[: finding.start] + replacement + redacted[finding.end :]
    return redacted


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source policy JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Source policy line {line_number} must be an object")
            records.append(value)
        return records
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid source policy JSON: {path}") from exc
    if isinstance(value, dict):
        value = value.get("sources")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Source policy must be a JSON array or an object containing a sources array")
    return value


def load_source_policies(path: Path | None) -> dict[str, SourcePolicy]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Source policy manifest not found: {path}")
    policies: dict[str, SourcePolicy] = {}
    for index, record in enumerate(_load_json_records(path), start=1):
        source_path = str(record.get("source_path", "")).strip()
        if not source_path:
            raise ValueError(f"Source policy entry {index} has no source_path")
        resolved = Path(source_path).expanduser()
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        key = str(resolved.resolve())
        if key in policies:
            raise ValueError(f"Duplicate source policy for {source_path!r}")
        uses = record.get("permitted_uses")
        if not isinstance(uses, list) or not all(isinstance(item, str) for item in uses):
            raise ValueError(f"Source policy entry {index} must declare permitted_uses")
        policies[key] = SourcePolicy(
            source_path=source_path,
            origin=str(record.get("origin", "")).strip(),
            license=str(record.get("license", "NOASSERTION")).strip(),
            permitted_uses=tuple(sorted({item.strip().lower() for item in uses if item.strip()})),
            rights_holder=str(record.get("rights_holder", "")).strip(),
        )
    return policies


def audit_decision(
    row: dict[str, Any],
    *,
    control: str,
    outcome: str,
    reason_codes: Iterable[str],
    evidence: Mapping[str, Any] | None = None,
) -> None:
    audit = row.setdefault("forge_audit", {"schema_version": AUDIT_SCHEMA, "decisions": []})
    if not isinstance(audit, dict):
        raise ValueError("forge_audit must be an object")
    decisions = audit.setdefault("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("forge_audit.decisions must be an array")
    decisions.append(
        {
            "control": control,
            "outcome": outcome,
            "reason_codes": sorted(set(reason_codes)),
            "evidence": dict(evidence or {}),
        }
    )


def _policy_for_document(row: Mapping[str, Any], policies: Mapping[str, SourcePolicy]) -> SourcePolicy | None:
    path_value = row.get("path")
    if path_value:
        return policies.get(str(Path(str(path_value)).expanduser().resolve()))
    return None


def _finding_summary(findings: Sequence[PIIFinding]) -> dict[str, Any]:
    counts = Counter(finding.entity_type for finding in findings)
    return {
        "count": len(findings),
        "types": dict(sorted(counts.items())),
        "detectors": sorted({finding.detector for finding in findings}),
    }


def govern_documents(
    documents: Sequence[dict[str, Any]],
    *,
    source_manifest: Path | None,
    required_use: str,
    pii_action: str,
    allow_unknown_rights: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if pii_action not in {"reject", "redact"}:
        raise ValueError("pii_action must be 'reject' or 'redact'")
    policies = load_source_policies(source_manifest)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unknown_rights = 0
    disallowed_rights = 0
    pii_findings = 0
    seen_content: dict[str, str] = {}

    for original in documents:
        row = dict(original)
        row_id = str(row.get("id", ""))
        content = str(row.get("content", ""))
        signature = canonical_sha256({"content": " ".join(content.split())})
        reasons: list[str] = []
        policy = _policy_for_document(row, policies)
        if policy is None or policy.license.lower() in UNKNOWN_LICENSES:
            unknown_rights += 1
            reasons.append("rights.unknown")
        elif required_use.lower() not in policy.permitted_uses:
            disallowed_rights += 1
            reasons.append("rights.use_not_permitted")
        else:
            row["source_rights"] = {
                "origin": policy.origin,
                "license": policy.license,
                "permitted_uses": list(policy.permitted_uses),
                "rights_holder": policy.rights_holder,
                "policy_source": policy.source_path,
            }
        duplicate_of = seen_content.get(signature)
        if duplicate_of is not None:
            reasons.append("duplicate.document_exact")
            row["duplicate_of"] = duplicate_of
        else:
            seen_content[signature] = row_id

        findings = scan_pii(content)
        pii_findings += len(findings)
        if findings and pii_action == "reject":
            reasons.append("privacy.structured_identifier")
            row["content"] = redact_pii(content, findings)
            audit_decision(
                row,
                control="privacy.source",
                outcome="quarantined_redacted",
                reason_codes=("privacy.structured_identifier",),
                evidence=_finding_summary(findings),
            )
        elif findings:
            row["content"] = redact_pii(content, findings)
            audit_decision(
                row,
                control="privacy.source",
                outcome="redacted",
                reason_codes=("privacy.structured_identifier",),
                evidence=_finding_summary(findings),
            )

        blocking = any(
            reason in {"rights.use_not_permitted", "duplicate.document_exact", "privacy.structured_identifier"}
            or (reason == "rights.unknown" and not allow_unknown_rights)
            for reason in reasons
        )
        audit_decision(
            row,
            control="governance.source",
            outcome="rejected" if blocking else "kept",
            reason_codes=reasons or ("governance.passed",),
            evidence={"required_use": required_use, "content_sha256": signature},
        )
        (rejected if blocking else kept).append(row)

    report = {
        "schema_version": GOVERNANCE_SCHEMA,
        "control": "source_governance",
        "status": "passed"
        if kept and disallowed_rights == 0 and (allow_unknown_rights or unknown_rights == 0)
        else "failed",
        "input_documents": len(documents),
        "kept_documents": len(kept),
        "rejected_documents": len(rejected),
        "unknown_rights": unknown_rights,
        "disallowed_rights": disallowed_rights,
        "pii_findings": pii_findings,
        "pii_action": pii_action,
        "allow_unknown_rights": allow_unknown_rights,
        "required_use": required_use,
    }
    return kept, rejected, report


def _validate_example_schema(row: Mapping[str, Any]) -> list[str]:
    reasons = []
    for field in ("id", "document_id", "chunk_id", "task_name", "input_text", "output_text"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"schema.missing_{field}")
    if "quality_score" in row and not isinstance(row["quality_score"], (int, float)):
        reasons.append("schema.invalid_quality_score")
    return reasons


def govern_records(
    records: Sequence[dict[str, Any]],
    *,
    pii_action: str,
    text_fields: Sequence[str] = ("input_text", "output_text"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if pii_action not in {"reject", "redact"}:
        raise ValueError("pii_action must be 'reject' or 'redact'")
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    findings_total = 0
    redacted_records = 0
    quarantine_redactions = 0

    for original in records:
        row = dict(original)
        reasons = _validate_example_schema(row)
        row_findings: list[PIIFinding] = []
        for field in text_fields:
            value = str(row.get(field, ""))
            findings = scan_pii(value)
            row_findings.extend(findings)
            if findings:
                row[field] = redact_pii(value, findings)
        findings_total += len(row_findings)
        if row_findings:
            reasons.append("privacy.structured_identifier")
        if row_findings and pii_action == "redact" and not _validate_example_schema(row):
            redacted_records += 1
            audit_decision(
                row,
                control="privacy.record",
                outcome="redacted",
                reason_codes=("privacy.structured_identifier",),
                evidence=_finding_summary(row_findings),
            )
            reasons = [reason for reason in reasons if reason != "privacy.structured_identifier"]
        elif row_findings:
            quarantine_redactions += 1
            audit_decision(
                row,
                control="privacy.record",
                outcome="quarantined_redacted",
                reason_codes=("privacy.structured_identifier",),
                evidence=_finding_summary(row_findings),
            )
        blocking = bool(reasons)
        reason_counts.update(reasons)
        audit_decision(
            row,
            control="governance.record",
            outcome="rejected" if blocking else "kept",
            reason_codes=reasons or ("governance.passed",),
            evidence={"text_fields": list(text_fields)},
        )
        (rejected if blocking else kept).append(row)

    report = {
        "schema_version": GOVERNANCE_SCHEMA,
        "control": "record_governance",
        "status": "passed" if kept else "failed",
        "input_records": len(records),
        "kept_records": len(kept),
        "rejected_records": len(rejected),
        "redacted_records": redacted_records,
        "quarantine_redactions": quarantine_redactions,
        "pii_findings": findings_total,
        "remaining_pii_findings": 0,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "pii_action": pii_action,
        "detector_scope": "structured identifiers only",
    }
    return kept, rejected, report
