# Threat model

## Scope

This model covers the Forge CLI, Python workflow, hosted bounded demonstration, transient run storage and downloadable evidence bundle. It is a portfolio threat model, not a security certification.

## Assets and invariants

- Source content, declared usage rights and source provenance must remain attributable.
- Train and test partitions must not share a source document.
- Rejected records and release gates must not be silently bypassed.
- A resumed stage must not trust changed inputs or modified cached artifacts.
- Public callers must not gain file-system, network or model-key access.
- A release manifest must describe the exact artifacts it verifies.

## Trust boundaries

```text
public browser
  -> bounded FastAPI contract
  -> typed Forge workflow
  -> model / similarity adapters
  -> transient run directory
  -> allowlisted evidence API and ZIP
```

Source-rights declarations and supplied documents are untrusted operator input. External model and embedding providers, when enabled, are separate processors outside Forge's trust boundary. The hosted public mode does not accept URLs, file paths, uploads or model credentials.

## Threats, current controls and residual risk

| Threat | Current control | Residual risk |
| --- | --- | --- |
| Resource exhaustion through public runs | Bounded text inputs, controlled presets, in-memory rate limit, bounded retention and one worker process | In-memory controls do not coordinate across replicas and are not a production abuse-control system |
| Path or network access through user input | Public contract accepts text fields and allowlisted preset identifiers only | Application or dependency defects could still create an unexpected path; production needs isolation and egress policy |
| Script injection in submitted text or artifacts | Browser output is rendered as text and artifacts use declared media types | A future rich renderer must preserve escaping and content-type controls |
| Prompt injection in source documents | Deterministic public adapters; model adapters are explicit and stage outputs remain schema-checked | A live generator can still follow hostile document instructions or leak provider-visible content |
| Sensitive data reaching a model or release | Source and record governance, structured-identifier rejection/redaction and rejected-record artifacts | Pattern detection is incomplete and does not replace privacy review or data classification |
| False or unlawful usage-rights declaration | Candidate builds require a source-rights manifest and permitted-use fields | Forge verifies the declaration structure, not legal ownership or licence validity |
| Dataset or benchmark leakage | Whole-document split isolation and layered contamination gates | Semantic thresholds can miss paraphrases or flag legitimate related material |
| Duplicate or poisoned examples | Exact, MinHash, Jaccard and optional embedding controls with reason-coded quarantine | Detection quality depends on corpus-specific calibration |
| Stale or tampered cache reuse | Content-derived keys, stage contracts, input hashes and output-hash verification | Local run directories are not an authenticated multi-user artifact store |
| Manipulated evaluation claim | Independent benchmark requirement, blind packets, agreement thresholds and claim gates | Genuine human annotations and an independently run model evaluation are still missing |
| Supply-chain compromise | Locked project dependencies, CI checks and container build | No signed build provenance, SBOM enforcement or dependency allowlist is claimed |
| Evidence bundle disclosure | Allowlisted artifacts and short-lived transient jobs | Public run contents are visible to the caller and should never contain confidential data |

## Production gates

Before accepting confidential data or multi-user production traffic, add authenticated tenants, durable distributed rate limits, isolated workers, controlled egress, encrypted durable storage, retention enforcement, malware/content scanning, secrets management, dependency provenance, audit logging and an incident-response path.

## Security-reporting boundary

Do not submit secrets, personal data or proprietary documents to the public demonstration. Report suspected vulnerabilities through a private maintainer channel rather than placing exploit details in a public issue.
