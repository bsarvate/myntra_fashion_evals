# Build verification — 12 September 2026

The active dataset is **1,000 products**, following the user's correction: 400 tops, 150 jeans, 100 shorts, 100 trousers, 50 skirts and 200 sets. The earlier 100-product draft is archived under artifacts/superseded_myntra100. The original data/raw files are unchanged.

Selection uses fixed SHA-256 ordering within garment strata after rejecting missing core data, unsupported structures, mixed-fabric sets and image URLs without the expected product ID. This is a deliberately scoped sample, not a representative random sample of the entire dataset.

The raw CSV index repeats by category: only 990 unique index values across 14,330 rows. It cannot identify image files uniquely. Each selected image was retrieved from its CSV source URL, checked for the product ID in that URL, decoded and hashed. The active subset has **1,000 readable photographs and 1,000 distinct image hashes**. This verifies listing-image provenance, not the correctness of every source attribute. Three representative listing-image pairs were visually inspected; this is not independent human benchmark labeling.

## Evaluation cases

**120 concrete cases:** 40 text, 30 image, 30 conversations and 20 validator boundaries. Related variants are grouped into **84 development / 36 reserved test cases**, with 50 groups. Objective labels are derived from source attributes or task specifications. Ten image-plus-text cases need attributed human visual relevance grades. Catalog-image crops are controlled stress checks, not independent shopper photographs.

## Verified locally

* **62 engineering tests passed**, covering catalog preparation, metrics, label provenance, split checks, pending-review handling, mixed-fabric sets, graph behavior, retention, validation and mocked integrations.
* **All 45 default cells in notebook 02 executed successfully** in a fresh Python 3.11 project kernel. Cloud/model-download stages explicitly skipped execution.
* **28 development BM25 text cases executed with zero execution errors**. These results measure source-constraint retrieval, not general fashion recommendation accuracy.
* **14/14 development validator cases passed**, including valid controls and deliberate violations.
* Reserved test cases were structurally audited but not executed against retrieval/models.
* The original 77-step synthetic notebook remains a separate introductory walkthrough.
* Reports and the executed real-data notebook are under artifacts/myntra1000/, with catalog/benchmark/code hashes.
* A separate approximately **137 MiB** prepared-data ZIP supports Colab without the full original dataset. The source bundle excludes raw data, user secrets and the virtual environment.

* Intel macOS compatibility verified with torch 2.2.2, NumPy 1.26.4, Transformers 4.57.2 and sentence-transformers 3.4.1. Actual safetensors model loading and inference passed: two text vectors (384 dimensions), two image vectors (512 dimensions), and CLIP text encoding (512 dimensions). Text and image vectors were finite and normalized. This smoke check does not constitute full-subset retrieval evaluation.

## Still pending

Full-subset Text/CLIP embedding generation and retrieval evaluation, Pinecone indexing and live Nebius comparisons remain optional next stages; no paid model calls or Pinecone writes were made during preparation. Ten visual-alternative cases need human relevance grades, and actual model outfits/explanations need human review. Independent shopper queries and photographs are needed for broader generalization claims. Hosted Colab execution has not been verified; local Jupyter execution has.

Start with notebooks/02_myntra1000_evaluation.ipynb. See evals/myntra1000_v1/README.md for interpretation limits.
