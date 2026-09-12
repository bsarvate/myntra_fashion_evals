# The 120-case benchmark

This is a generated, source-derived benchmark over a fixed 1,000-product catalog. It is ready for engineering evaluation; it is not a claim of 120 independently collected human shopping tasks.

| Family | Cases | Development | Reserved test | Label basis |
|---|---:|---:|---:|---|
| Text retrieval | 40 | 28 | 12 | Exhaustive exact source-attribute predicates, including no-match cases |
| Image retrieval | 30 | 21 | 9 | 10 known-image identities, 10 catalog-image crops, 10 visual alternatives pending human labels |
| Conversations | 30 | 21 | 9 | Budget/attribute/retention/replacement/stock expectations from task specifications |
| Validator boundaries | 20 | 14 | 6 | Ten valid controls and ten deliberate violations |
| Total | 120 | 84 | 36 | 50 related-case groups |

The searchable catalog is shared across splits. Queries/related variants are grouped; this is not a product-holdout split. Generated templates can occur in both splits, so the reserved set is not evidence of broad natural-language generalization. Use additional independently authored shopper queries before making that claim.

## Files and execution

* `all_cases.jsonl`: exactly 120 cases; no repeated counting of follow-up turns as separate cases.
* `case_index.csv`: compact case ID, family, group, split and evidence index.
* `text.jsonl`, `images.jsonl`, `conversations.jsonl`, `boundaries.jsonl`: runner-specific case definitions.
* `manifest.json`: catalog and benchmark hashes plus counts and provenance qualifications.
* `query_images/`: ten center crops generated from the identified catalog photographs. These are preprocessing fixtures, not new real-world photos.
* `image_relevance_review.csv`: blank relevance grades for eligible candidates in the ten visual-alternative cases. Names and grades must be supplied by actual reviewers.

Notebook 02 runs only development cases by default. It executes BM25 and validator cases locally; embeddings, CLIP, Pinecone and live Nebius comparisons have explicit optional steps. Preparing a case does not mean its model execution passed.

## What the metrics establish

Text predicates determine relevance from color/fabric/garment source fields. Because these filters are supplied to the retriever, success does not establish that the agent extracted constraints correctly from natural language. The paired conversational tests examine agent actions separately. Binary predicate relevance also does not measure subjective fashion compatibility.

Image identity tasks score retrieval of the known source photograph/product, including a simple crop variant. They are not equivalent to finding garments in independently photographed shopper outfits. The ten visual-alternative cases remain unscored until human relevance labels are imported. Do not count pending labels as zeros or successes.

Conversation controls use confirmed settings and explicitly simulated inventory. Valid outfit outcomes, retained/replaced IDs and call bounds are checkable; outfit appeal and unsupported free-text claims still require the human-review rubric in `../RUBRIC.md`.

Use separate reporting per family with counts. Never combine deterministic boundary checks, source-derived retrieval scores and subjective human ratings into one overall accuracy number.
