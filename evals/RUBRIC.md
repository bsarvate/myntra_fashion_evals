# Evaluation protocol v0.1

Status: the included cases are synthetic engineering fixtures. The real benchmark is pending the dataset and human labels. Never report fixture results as Myntra/model accuracy.

## Data and splits

1. Audit product IDs, malformed rows, prices, category coverage, missing attributes and image joins.
2. Deduplicate catalog products and record the complete catalog hash. Preserve the original CSV separately.
3. Create query cases independently of system outputs. Use the proposed starting allocation: 40 text, 30 image, 30 conversations, 20 boundary/adversarial cases.
4. Assign related paraphrases, the same garment/photo variants and related conversations a shared group_id. Split groups, not individual variants, into development (approximately 70%) and reserved test (30%). These are query splits; retrieval can search the entire frozen catalog.
5. Freeze test cases and judgments before tuning. The notebook defaults to development; unlock the real test set only for a frozen configuration. No test set exists yet.
6. Preserve source image provenance. Report exact-catalog-image lookup separately from independent shopper photos, crops and transformed catalog images. Do not call transformed images independent photographs.

## Retrieval labels

Use two reviewers on at least 20% of cases, blinded to retrieval method and model. Record independent grades first; preserve disagreements and adjudication separately.

* 0: wrong garment or violates a hard requirement; irrelevant.
* 1: weak alternative that satisfies hard requirements but poorly matches the intent.
* 2: useful match with a minor preference mismatch.
* 3: highly relevant match to intent and all hard requirements.

Treat missing evidence for a required attribute as unverified, not confirmed. Fabric labels describe source claims, not independently measured composition.

For a manageable filtered candidate pool, judge every eligible product explicitly (including zeros), then set exhaustive=true. For a larger catalog, pool candidates from all comparison methods, review them and keep exhaustive=false. Never infer an unjudged item is irrelevant. Expand the pool when a new method retrieves unjudged items.

Recall@K is only computed for exhaustive cases with relevant products. Precision@K uses K as its denominator; precision_returned uses returned count. nDCG uses graded judgments and its ideal ranking comes from the judged pool. It is unavailable if a returned product is unjudged. No-match success requires exhaustive zero-relevance labels. An empty filtered pool can establish absence under those structured filters; low similarity alone cannot.

## Conversation and validation

Each conversation records initial confirmed settings, turns, expected status, required/forbidden IDs where applicable, and call limits. Do not require one exact outfit if several are valid. Run multiple repeats per model with fresh state for each conversation and retained state within it.

Measure task success, whole-dialogue success, tool/schema failures, constraint violations, retention/reinspection behavior, clarification quality and inappropriate refusal. Negative tests must include valid alternatives to catch over-refusal. Objective runner assertions alone do not establish whether the model understood every natural-language requirement.

Maintain a separate validator matrix: valid selections, boundary budget, over-budget, missing/zero/changed stock, wrong size, unknown IDs, duplicate IDs, missing inspection, unsupported unit and source constraint violations. Use deterministic fixtures for this layer.

## Outfit and explanation review

Score each dimension separately from 0 (poor) to 3 (strong): color compatibility, silhouette compatibility, occasion suitability and adherence to style preferences. These are subjective judgments, not facts. Show catalog photos to reviewers.

For factual grounding, break explanations into atomic claims. Record claim text, cited product/field and supported yes/no/uncertain. Report unsupported claims divided by assessed factual claims, with counts. Exclude subjective styling suggestions from factual accuracy; judge them under outfit quality. Do not turn absent claims into perfect grounding scores.

Use evals/templates/human_review.csv for independent reviews. Calculate inter-reviewer agreement before adjudication when sufficient labels exist. LLM judging is optional future work and must first be calibrated against these human reviews; no automatic LLM judge is included or claimed.

## Comparisons and reports

First compare BM25/dense/hybrid on identical text cases. Compare CLIP on image cases and multimodal fusion on image+text cases. Do not compare different query populations as if they were equivalent. Keep filters, candidate depth, K, catalog and embeddings fixed when comparing Nebius models.

Report denominators and failures per category, not only one average. Retrieval summaries use equal weight per independent group and bootstrap group means (seed 42) for 95% intervals. Small fixture intervals are not evidence of generalization. Test paired differences on a sufficiently sized real benchmark before declaring a winner.

Record package versions, code/catalog/benchmark hashes, model IDs, embedding revisions, settings, raw tool traces, token usage and latency. Cost must use current explicit input/output rates supplied by the user; unknown usage means unknown cost, never zero. Separate cold-start embedding/index work from request latency. Live integrations and rate limits need separate measurement.

Suggested release condition: no invalid outfit accepted in the validator suite; all regression tests pass; all benchmark outputs are scored or explicitly marked unjudged/error; retrieval and task success meet team-agreed thresholds defined before test evaluation. Do not invent a numerical accuracy target after viewing results.
