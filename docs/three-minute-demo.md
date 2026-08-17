# Three-minute demonstration

Use the hosted deterministic smoke release. This walkthrough shows that the declared controls execute; it does not establish model quality, production safety or complete sensitive-data detection.

## 0:00–0:25 — Frame the problem

Open the [hosted Forge demonstration](https://training-data-factory-production.up.railway.app/).

> Training-data pipelines are easy to demo and hard to audit. Forge turns source documents into a content-addressed dataset release while preserving source rights, provenance, curation decisions, split isolation and artifact integrity.

Point to the 12-stage pipeline and seven release gates.

## 0:25–1:05 — Run the real workflow

Choose **AI release controls** and start the run.

> This hosted path calls the same typed Python workflow as the CLI and tests. Public mode uses deterministic adapters and bounded inputs, so anyone can reproduce the control flow without an API key.

As stages complete, point out source governance, record privacy, deduplication, contamination, source-safe splitting and release verification.

## 1:05–1:45 — Inspect one decision

Open the deduplication or contamination artifact.

> Records are not silently dropped. Kept and rejected rows carry reason codes, source identity and the configuration that produced the decision. Candidate releases also require a semantic similarity backend and an independent benchmark; this smoke run explicitly records that semantic checks are disabled.

Show the source split manifest and the zero-overlap result.

## 1:45–2:25 — Verify the release

Open `release_manifest.json`, then download the evidence bundle.

> The release ID is content-addressed. The manifest fingerprints the source inputs, configuration and output artifacts, and verification fails if a file changes, counts disagree, a stage failed, contamination crossed the threshold or a source appears in both train and test.

## 2:25–2:50 — Explain evaluation honesty

> The repository includes blind human-review packets, reviewer-agreement analysis and a reversed-order model-judge protocol. The included labels are authored fixtures that prove the machinery works. Genuine 200-item human annotation has not yet been collected, so Forge does not claim established model quality.

## 2:50–3:00 — Close

> The engineering contribution is a resumable, inspectable release system with explicit claim gates—not a screenshot of a fine-tuning result.

## Recording checklist

- Record between 2:45 and 3:15 and add captions.
- Show one live stage trace, one rejection artifact and the release manifest.
- Link the live system, source and technical controls in the description.
- Do not call fixture agreement scores human evidence.
- Do not say the privacy scanner is complete or the generated dataset is production-ready.
