# Forge blind evaluation protocol

Protocol version: `forge.annotation-protocol/v1`

## Purpose

This protocol measures whether a candidate training example is supported by its source, correct for the stated task, useful for learning, and preferable to a baseline. It does not ask reviewers to predict model performance.

## Blinding

- Reviewers receive Candidate A and Candidate B without system names.
- Reviewers must not receive `evaluation_blinding_key.json` until review is closed.
- Presentation order is deterministic and balanced from the frozen evaluation seed.
- Judge models receive both the primary and reversed order so position consistency can be measured.
- Reviewers must not discuss individual items until their submitted sheet is frozen.

## Review unit

For each presentation, inspect the source excerpt, task, prompt, reference answer, Candidate A, and Candidate B. The reference is supporting evidence, not an instruction to reward matching words.

## Preference label

Choose exactly one:

- `A`: Candidate A is materially better.
- `B`: Candidate B is materially better.
- `tie`: Both are acceptable and neither is materially better.
- `both_bad`: Neither candidate is acceptable.

Prefer an answer only when the difference matters for training usefulness. Stylistic preference alone is insufficient.

## Decision order

1. Source support: Does the source contain enough evidence for the candidate?
2. Correctness: Is the output factually and logically correct?
3. Instruction fit: Does it answer the stated task and prompt?
4. Usefulness: Would the example teach the intended behaviour without avoidable ambiguity?
5. Privacy and safety: Does it reproduce a sensitive identifier or introduce unsafe content?
6. Preference: Select A, B, tie, or both bad.

## Reason codes

Add one or more codes separated by `|` when they explain the decision:

- `unsupported_by_source`
- `incorrect`
- `incomplete`
- `instruction_mismatch`
- `ambiguous`
- `privacy_risk`
- `unsafe_content`
- `verbosity_without_value`
- `clearer_reasoning`
- `better_source_support`
- `both_acceptable`
- `both_unacceptable`

## Confidence

Record a value from 1 to 3:

- `1`: low confidence; the item may need adjudication.
- `2`: moderate confidence.
- `3`: high confidence.

## Reviewer identity

Use a pseudonymous ID such as `reviewer_01`. Do not put a name or email address in the annotation sheet. Real collection uses `reviewer_type=human`. Repository mechanism fixtures use `reviewer_type=fixture` and are never reported as human evidence.

## Coverage and agreement gate

A candidate evaluation requires at least 200 independently reviewed items. At least 50 items must receive overlapping reviews so nominal Krippendorff alpha can be calculated. The default minimum alpha is 0.667. Results below that boundary are treated as tentative and must not support a headline model-quality claim.

## Adjudication

Items with low confidence, `both_bad`, or reviewer disagreement go to an adjudicator who did not author either candidate. Original labels remain immutable. Adjudication is stored as a separate record rather than overwriting a reviewer.

## Judge calibration

The judge model, model family, prompt version, confidence, and response order are recorded. A judge from the same model family as a candidate generator cannot support a headline comparison. Human agreement is reported before judge agreement.
