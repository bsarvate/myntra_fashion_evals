# Fashion assistant with reproducible evaluations

A new project separate from the original 20-product packet. Start with **notebooks/02_myntra1000_evaluation.ipynb** for the prepared real dataset: 1,000 products, 120 evaluation cases and 45 numbered steps. Each step explains its purpose and shows intermediate outputs. The default run executes local development checks without paid API calls.

**notebooks/01_build_and_evaluate.ipynb** remains the 77-step introductory walkthrough with eight fictional products. Its fixtures are separate from the real-catalog benchmark.

## Run locally

Requires Python 3.11 or newer (the system Python 3.8 is too old).

```bash
cd fashion-assistant-evals
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[notebook,test]'
.venv/bin/python -m ipykernel install --sys-prefix --name fashion-assistant --display-name 'Fashion Assistant (Python 3.11)'
.venv/bin/python -m jupyterlab notebooks/02_myntra1000_evaluation.ipynb
```

With an existing Python 3.11 environment, `python -m pip install -e '.[notebook,test]'` also works. Optional text/image encoders and Pinecone: `python -m pip install -e '.[search]'`. Their installation/model downloads are separated in the notebook.

The search dependencies select PyTorch 2.2.2 and NumPy 1.x on Intel macOS, because [official Intel Mac wheels end at PyTorch 2.2.x](https://pytorch.org/blog/pytorch2-2/). Other supported platforms retain PyTorch 2.6+. Both encoders require safetensors model weights; the default CLIP revision is an immutable safetensors conversion in the original model repository. Revision settings are commit/tag strings, not True/False toggles. After installing search dependencies into an already running notebook, restart its kernel before importing NumPy or the encoders.

Select **Fashion Assistant (Python 3.11)** from Jupyter's kernel selector. Colab manages its own Python kernel; use its default runtime.

## Local environment settings

Fill in the project-root `.env` file with `NEBIUS_API_KEY`, `NEBIUS_MODEL` and `PINECONE_API_KEY`. Optional `PINECONE_TEXT_INDEX` and `PINECONE_IMAGE_INDEX` set index names. `.env.example` provides the format. The actual `.env` is ignored by Git and excluded from the source ZIP.

Both notebooks load `.env` in their dependency/setup cell using `load_dotenv(PROJECT / '.env', override=False)`. Existing environment variables take precedence; values are not printed. After editing `.env` in a running session, restart the kernel and rerun setup to reload it. Live-service flags remain explicit notebook settings, so loading credentials alone does not make paid calls. Colab Secrets remain supported.

## Run on Google Colab

1. Use `dist/fashion-assistant-evals.zip`, or regenerate it with `python scripts/package_project.py`. The bundle excludes secrets, raw data and the virtual environment.
2. Upload notebooks/02_myntra1000_evaluation.ipynb to Colab.
3. Run the first code step and upload the project ZIP when prompted. It contains code and evaluation definitions, not the full raw dataset.
4. At the subset upload step, upload `dist/myntra1000-data.zip`. This separate bundle contains just the prepared 1,000 products and their photographs. Continue one cell at a time; the full 3.15 GB dataset is unnecessary for this notebook.
5. Add NEBIUS_API_KEY and PINECONE_API_KEY through Colab Secrets only when reaching those optional sections. Set the chosen NEBIUS_MODEL in its configuration step.

## What is implemented

* A reproducible 1,000-product subset: 400 tops, 150 jeans, 100 shorts, 100 trousers, 50 skirts and 200 sets. Source CSV index values repeat; subset images are retrieved from verified product-specific CSV URLs and named by product ID.
* 120 concrete cases: 40 text, 30 image, 30 conversation and 20 boundary cases. Related variants stay together: 84 development cases and 36 reserved test cases. Source-derived objective labels are distinguished from human judgments; ten visual-alternative cases remain pending review.
* Explicit CSV mapping, audit/rejection report, normalized catalog and image-file checks.
* BM25, exact local cosine baseline, reciprocal-rank fusion, local text/CLIP encoders and separate Pinecone text/image adapters.
* A bounded LangGraph using LangChain tools and a configurable ChatNebius model; source-based outfit validation and follow-up reinspection.
* A notebook interface with chat, budget/sizes, hard filters, image upload and product output. Images are queries for CLIP; the current LLM receives structured product evidence, not raw photos.
* Retrieval benchmark validation, leakage checks, graded metrics, group bootstrap intervals, per-category reports and immutable timestamped runs.
* A conversation evaluator, deterministic model replay, test suite and human-review templates/rubric.

## What is still pending

* Independent human shopping-query evaluation and outfit/claim judgments. The current real-catalog cases are generated from source attributes and task specifications, not independent shopper queries.
* Human visual relevance grades for ten image-plus-text cases. Exact-image lookup and catalog-derived crops do not establish independent shopper-image performance.
* Model selection and live Nebius/Pinecone checks with your credentials. These are opt-in notebook stages, not tested live by the build process.
* Evaluation of broader conversational constraints, vision-LLM inspection, real inventory and production deployment. The current fixed settings remain authoritative; text-only extra requirements depend on model filter choices and require evaluation.

No model is trained. Prices remain historical and inventory is explicitly simulated. The assistant validates a pair or set; it does not guarantee fit or subjective style quality. Call limits bound requests but are not an account billing cap. Reports with credentials or private query images should not be publicly shared.

## Files

* `notebooks/02_myntra1000_evaluation.ipynb`: real-data step-by-step evaluation and UI.
* `data/processed/myntra1000/`: canonical CSV/JSON, 1,000 product images, selection and image provenance, simulated inventory.
* `evals/myntra1000_v1/`: all 120 cases, group splits, source-derived labels, case index and pending human-review sheet.
* `notebooks/01_build_and_evaluate.ipynb`: granular tutorial and notebook UI.
* `src/fashion_assistant/`: importer, retrieval, embeddings, Pinecone, agent, validation, evaluation.
* `evals/RUBRIC.md`: labeling, split and measurement protocol.
* `evals/fixtures/`: explicitly synthetic cases.
* `evals/templates/`: blank human benchmark/review formats.
* `config/`: explicit mapping and experiment examples.
* `tests/`: objective checks and real LangGraph with scripted model responses.
* `artifacts/`: local generated catalog snapshots, traces and reports.

Run `python -m pytest -q`. Regenerate the notebook using `python scripts/build_notebook.py` if editing its source. External integrations are documented against [Nebius](https://docs.tokenfactory.nebius.com/integrations/frameworks/langchain), [LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api), [Pinecone](https://docs.pinecone.io/guides/index-data/create-an-index), and [CLIP](https://huggingface.co/docs/transformers/model_doc/clip). The model ID and capabilities must be verified before live evaluation.

To reproduce the real-data preparation, run `python scripts/prepare_myntra1000.py`, then `python scripts/build_benchmark120.py` and `python scripts/build_myntra_notebook.py`. The image downloader resumes from verified cached files. Dataset preparation contacts the CSV image URLs; it does not call an LLM. The benchmark builder refuses to overwrite completed human-review grades. Freeze a new benchmark version before making later data changes.

Regenerate the two Colab bundles with `python scripts/package_project.py` and `python scripts/package_myntra1000.py`. Raw files and API secrets are excluded.

### Observable conversation trials

Step 43 now defaults to one selected-split conversation. It prints each turn's start,
status and elapsed seconds, and atomically saves a checkpoint after each completed
turn. It stops after an API/execution error in trial mode. Keyboard interruption
returns a partial report; an in-flight turn is not recovered. Partial reports show
planned versus recorded turns and exclude incomplete dialogues from dialogue
success. These rates must not be compared as full-benchmark scores. Existing
checkpoint paths are rejected to prevent accidental overwriting. Provider calls
retain the existing 60-second request timeout and zero automatic retries; there is
no hard overall wall-clock deadline.

Use `run_conversations(..., checkpoint_path=path, progress=True,
stop_on_api_error=True)`. Reload `fashion_assistant.evaluation` in an existing
notebook kernel before importing the updated runner. Changing notebook source does
not change a running cell. Alternative models can be compared by setting
`MODELS_TO_COMPARE` to their exact Nebius IDs while keeping the case scope,
retrieval, repeats and call limits fixed. Model availability is account-dependent.

### Streamlit app

Install `.venv/bin/python -m pip install -e '.[ui]'`, then run
`.venv/bin/python -m streamlit run streamlit_app.py` from the project directory.
Open http://localhost:8503. Jupyter can continue independently.

The app loads your existing `.env`, uses a Nebius chat model entered in the sidebar,
and keeps each browser session's shopping agent separate. Start a new conversation
after changing confirmed settings or the model. Sending a message makes up to six
paid model calls; opening the app, browsing the catalog and viewing saved evaluation
reports do not. Chat currently uses BM25. Image search is a separate CLIP similarity
view; install the `search` extra to enable it. The first image query builds a local
1,000-image index with progress and caches it by catalog/model revision. Image
search results are not automatically passed into outfit chat. Conversation traces,
uploaded references and image vectors are saved under artifacts/myntra1000/streamlit.
The app displays historical prices in catalog units and simulated stock, not current
store prices or availability. The evaluation view reads existing notebook reports;
it does not start paid benchmarks.

### Clone from GitHub

```bash
git clone https://github.com/bsarvate/myntra_fashion_evals.git
cd myntra_fashion_evals
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[notebook,test,ui]'
cp .env.example .env
```

Fill in your API credentials locally. Raw/processed datasets, catalog photographs,
image-query crops, personal review outputs, recordings, generated reports, source/data
ZIPs and the virtual environment are excluded from Git. Synthetic fixtures and the
120 case definitions are included. Copy your Myntra CSV to
`data/raw/Fashion Dataset.csv`, then prepare the subset and regenerate image-query
assets with `scripts/prepare_myntra1000.py` and `scripts/build_benchmark120.py`.
See the preparation instructions above. Source-only tests skip checks that require
the prepared dataset. Notebook files published in Git are clean generated tutorials;
local working notebook outputs are not published.
