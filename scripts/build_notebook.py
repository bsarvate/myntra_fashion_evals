"""Generate the teaching notebook. Each operation has a purpose and visible output."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells, steps = [], []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()+"\n"})


def step(title, explanation, code):
    number = len(steps)+1
    steps.append(title)
    markdown(f"## Step {number:02d} — {title}\n\n{explanation}")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(code).strip()+"\n"})


markdown("""
# Build and evaluate a multimodal fashion assistant

Run **one cell at a time**. Every numbered step explains what it does and shows an intermediate result.

**Default run:** eight fictional products, deterministic retrieval/validation tests and a real LangGraph with scripted model responses. No API keys, model downloads or paid calls are needed for this route. Synthetic results are engineering checks, not model or Myntra accuracy.

**Real-data route:** supply your extracted dataset, map the CSV columns, review image joins and label evaluation cases. Optional sections then enable text embeddings, CLIP, Pinecone and a Nebius LLM.

**Jupyter:** launch this notebook from the project folder using the README commands. **Colab:** upload this notebook, then supply the project ZIP in Step 1. Dataset upload is separate.

You can inspect the Python functions in later cells; external work is never hidden in an import. Changing a configuration cell requires rerunning the dependent steps below it. Do not use Run All after enabling live services unless you intend all enabled calls to execute.
""")

step("Locate or upload the project", "Local Jupyter finds the project above this notebook. In Colab, upload the source ZIP when prompted; extraction checks paths and preserves the project directory.", '''
from pathlib import Path
import sys, json, os
candidates = [Path.cwd(), *Path.cwd().parents, Path('/content/fashion-assistant-evals')]
PROJECT = next((p for p in candidates if (p/'pyproject.toml').exists() and (p/'src/fashion_assistant').exists()), None)
if PROJECT is None:
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError('Open this notebook from inside the extracted project folder.')
    import io, zipfile
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Choose the single fashion-assistant-evals.zip project bundle.')
    destination = Path('/content').resolve()
    with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
        for name in archive.namelist():
            if not (destination/name).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive path')
        archive.extractall(destination)
    PROJECT = destination/'fashion-assistant-evals'
    assert (PROJECT/'pyproject.toml').is_file(), 'Upload the project bundle, not the dataset ZIP.'
print('Project:', PROJECT)
''')

step("Check Python", "Python 3.11 or newer is required. This prints the exact interpreter so environment problems are visible.", '''
print(sys.version)
print('Interpreter:', sys.executable)
assert sys.version_info >= (3, 11), 'Use Python 3.11+ or a current Colab runtime.'
''')

step("Install notebook dependencies", "This may download Python packages. It does not call Nebius or Pinecone. Existing installations are reused. Optional embedding libraries come later.", '''
import importlib.util, subprocess
required = ['fashion_assistant', 'langgraph', 'langchain_nebius', 'ipywidgets', 'dotenv']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', str(PROJECT)+'[notebook,test]'])
print('Core packages ready; no inference requests sent.')
from dotenv import load_dotenv
load_dotenv(PROJECT/'.env', override=False)
print('Project .env loaded; values hidden.' if (PROJECT/'.env').is_file() else 'No local .env; environment variables or Colab Secrets can be used.')
''')

step("Inspect installed versions", "Record the environment used by this run. requirements-tested.txt records the developer's tested environment; this cell records yours.", '''
from importlib.metadata import version
print({name: version(name) for name in ['fashion-assistant-evals', 'langchain-core', 'langchain-nebius', 'langgraph', 'numpy', 'pillow']})
''')

step("Choose synthetic or real data", "Leave USE_REAL_DATA=False for the initial run. Once your extracted Myntra files are available, set it to True and point DATASET_ROOT to that folder. Nothing downloads the Kaggle dataset automatically.", '''
USE_REAL_DATA = False
DATASET_ROOT = PROJECT/'data/raw'  # Change to your extracted dataset directory, including a Drive path if needed.
ARTIFACTS = PROJECT/'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
print('Data mode:', 'REAL DATA — requires mapping and review' if USE_REAL_DATA else 'SYNTHETIC ENGINEERING FIXTURE')
print('Dataset folder:', DATASET_ROOT)
''')

step("Optionally mount Google Drive", "For a large image dataset, place the extracted files in Drive and enable this cell. Set DATASET_ROOT in the preceding cell to the resulting folder. Local Jupyter skips this.", '''
MOUNT_DRIVE = False
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('Drive mount skipped.')
''')

step("Locate the CSV", "This lists possible CSV files without guessing which one is the catalog. Set CSV_PATH explicitly for the real dataset.", '''
csv_candidates = sorted(DATASET_ROOT.glob('**/*.csv')) if DATASET_ROOT.exists() else []
print('CSV candidates:', [str(p) for p in csv_candidates[:20]])
CSV_PATH = PROJECT/'data/sample/catalog.csv' if not USE_REAL_DATA else DATASET_ROOT/'Fashion Dataset.csv'
assert CSV_PATH.is_file(), f'Set CSV_PATH to your catalog file: {CSV_PATH}'
print('Selected:', CSV_PATH)
''')

step("Inspect raw columns and sample rows", "Inspect actual column names before mapping them. A malformed-row count makes CSV problems explicit.", '''
from fashion_assistant.catalog import inspect_csv
raw_audit = inspect_csv(CSV_PATH)
print(json.dumps(raw_audit, indent=2, ensure_ascii=False))
''')

step("Map CSV fields explicitly", "The sample uses canonical names. For Myntra, edit the JSON mapping file or this dictionary using the columns printed above. Map product_id, name and price at minimum; other absent facts stay unknown.", '''
from fashion_assistant.catalog import FIELDS
COLUMN_MAPPING = {field: field for field in FIELDS} if not USE_REAL_DATA else json.loads((PROJECT/'config/column_mapping.example.json').read_text())
print(json.dumps(COLUMN_MAPPING, indent=2))
''')

step("Define garment slots and image paths", "If no slot column exists, provide a reviewed category-to-slot map. Images require an explicit image_path column relative to IMAGE_ROOT; we never guess joins from directory order. Confirm currency from the source before replacing CATALOG_UNITS.", '''
CATEGORY_SLOTS = {}  # Example only: {'Tops': 'top', 'Jeans': 'bottom', 'Co-ords': 'set'}
IMAGE_ROOT = DATASET_ROOT if USE_REAL_DATA else PROJECT/'data/sample'
PRICE_UNIT = 'CATALOG_UNITS'
print('Category map:', CATEGORY_SLOTS)
print('Image root:', IMAGE_ROOT, '| fallback price unit:', PRICE_UNIT)
''')

step("Normalize the catalog", "Prices are validated, IDs deduplicated, slots checked and readable image files verified. Bad rows are rejected with reasons; missing images remain usable for text retrieval.", '''
from fashion_assistant.catalog import prepare_catalog
products, catalog_audit = prepare_catalog(CSV_PATH, COLUMN_MAPPING, category_slots=CATEGORY_SLOTS, image_root=IMAGE_ROOT, price_unit=PRICE_UNIT)
print('Accepted products:', len(products))
if not products:
    print(json.dumps(catalog_audit, indent=2))
assert products, 'No usable products. Inspect mappings and the rejection report.'
''')

step("Read the rejection and image report", "Review this before proceeding. A readable image does not prove it is mapped to the correct product; visually review joins in a separate step.", '''
print(json.dumps(catalog_audit, indent=2))
''')

step("Inspect a normalized product", "This is the source record used by retrieval and validation. No missing fabric/color is invented.", '''
print(json.dumps(products[0], indent=2, ensure_ascii=False))
''')

step("Save the prepared catalog snapshot", "Save the exact imported data and audit report. The catalog hash will travel with evaluation results.", '''
(ARTIFACTS/'catalog.json').write_text(json.dumps(products, indent=2))
(ARTIFACTS/'catalog_audit.json').write_text(json.dumps(catalog_audit, indent=2))
print('Catalog SHA-256:', catalog_audit['catalog_sha256'])
print('Saved:', ARTIFACTS/'catalog.json')
''')

step("Visually inspect product-image joins", "Display a few mapped images next to their source product names. The fictional sample intentionally has no photographs; it cannot be used to evaluate CLIP.", '''
from IPython.display import display, Image
image_products = [p for p in products if p['image_verified']]
print('Products with readable images:', len(image_products))
for product in image_products[:4]:
    print(product['product_id'], product['name'])
    display(Image(filename=str(IMAGE_ROOT/product['image_path']), width=200))
if not image_products:
    print('Image indexing/evaluation will be skipped until real images are supplied.')
''')

step("Inspect searchable text", "One searchable record represents one product. The text includes source attributes; prices and inventory are checked separately.", '''
from fashion_assistant.catalog import product_text
print(product_text(products[0]))
''')

step("Build the BM25 index", "BM25 scores keyword overlap with document-length normalization. Building this local index makes no external requests.", '''
from fashion_assistant.retrieval import BM25, HybridSearch, fuse, LocalVectors
bm25 = BM25(products)
search = HybridSearch(bm25)
print('Indexed documents:', len(bm25.docs), '| average tokens:', round(bm25.avgdl, 2))
''')

step("Choose a text query and hard filters", "Filters are exact source constraints. Edit these independently of the query to see the effect on the candidate pool.", '''
QUERY = 'black cotton top'
FILTERS = {'slot': 'top', 'color': 'black', 'fabric': 'pure cotton'}
print('Query:', QUERY)
print('Hard filters:', FILTERS)
''')

step("Show eligible products before ranking", "This separates filtering from relevance ranking, and exposes why a candidate is excluded.", '''
from fashion_assistant.catalog import matches
eligible = [p['product_id'] for p in products if matches(p, FILTERS)]
print('Eligible IDs:', eligible)
''')

step("Run and inspect BM25", "The score is a ranking signal, not confidence that an outfit is correct.", '''
bm25_results = bm25.search(QUERY, FILTERS, k=5)
print(json.dumps(bm25_results, indent=2))
''')

step("Load benchmark cases", "The default cases explicitly say synthetic. Real data requires a separate reviewed JSONL benchmark; do not reuse fictional product labels.", '''
from fashion_assistant.evaluation import load_cases, audit_cases
BENCHMARK_PATH = PROJECT/'evals/fixtures/retrieval.jsonl' if not USE_REAL_DATA else PROJECT/'evals/retrieval_reviewed.jsonl'
cases = load_cases(BENCHMARK_PATH) if BENCHMARK_PATH.exists() else []
print('Benchmark cases:', len(cases))
if cases:
    print(json.dumps(cases[0], indent=2))
else:
    print('No human benchmark yet. Retrieval evaluations will skip; use the labeling steps below.')
''')

step("Audit benchmark labels and splits", "Check IDs, relevance grades, hard-filter consistency, completeness claims and group leakage. Related queries must not cross dev/test splits.", '''
if cases:
    print(audit_cases(cases, products))
else:
    print('Pending independently reviewed cases.')
''')

step("Read the evaluation protocol", "Read how labels, no-match cases, image provenance and human style judgments should be collected before creating a real benchmark.", '''
from IPython.display import Markdown
display(Markdown((PROJECT/'evals/RUBRIC.md').read_text()))
''')

step("Choose evaluation settings", "Default to development cases. Reserve test cases until configuration and thresholds are frozen. K is fixed across methods.", '''
EVAL_SPLIT = 'dev'
K = 5
ALLOW_HELDOUT = False
assert EVAL_SPLIT != 'test' or ALLOW_HELDOUT, 'Freeze the experiment before enabling held-out evaluation.'
print('Split:', EVAL_SPLIT, '| K:', K)
''')

step("Run the BM25 evaluation", "Execute each eligible case. Tool errors remain recorded. No-match success, relevant-product recall and ranking quality have different denominators.", '''
from fashion_assistant.evaluation import run_retrieval
methods = {'bm25': lambda case, k: search.search(case['query'], case['filters'], k, mode='bm25')}
baseline_report = run_retrieval(cases, products, methods, split=EVAL_SPLIT, k=K) if cases else None
print('Completed cases:', len(baseline_report['rows']) if baseline_report else 0)
''')

step("Inspect each query result", "Review individual outputs before looking at averages. A null metric means not defined or not adequately judged, never zero error.", '''
if baseline_report:
    for row in baseline_report['rows']:
        print(json.dumps(row, indent=2))
else:
    print('Evaluation pending labels.')
''')

step("Inspect aggregate metrics and uncertainty", "Group-macro averages avoid overweighting paraphrases. Intervals bootstrap independent groups; six synthetic cases are only a test of the evaluator.", '''
print(json.dumps(baseline_report['summary'] if baseline_report else {}, indent=2))
''')

step("Save the baseline and run manifest", "Each run gets a new timestamped folder, code/package versions and catalog/benchmark hashes. Nothing overwrites the previous run.", '''
from fashion_assistant.evaluation import save_run
if baseline_report:
    baseline_path = save_run(baseline_report, ARTIFACTS/'retrieval', {'method': 'bm25', 'k': K, 'split': EVAL_SPLIT, 'seed': 42})
    print('Saved:', baseline_path)
else:
    print('No benchmark run to save.')
''')

step("Choose optional embedding downloads", "Set True to download pretrained text and CLIP weights. This uses network, disk and compute; it does not call the paid Nebius LLM. Keep False for the initial offline fixture run.", '''
ENABLE_MODEL_DOWNLOADS = False
TEXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
TEXT_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CLIP_REVISION = 'b33cedfd0df4e43b8238760678fcc89e1a0d38b3'  # Safetensors conversion; a commit ID, not a boolean.
print('Embedding downloads enabled:', ENABLE_MODEL_DOWNLOADS)
''')

step("Install optional search libraries", "This installs encoder and Pinecone dependencies only when enabled. The main notebook can run without them.", '''
if ENABLE_MODEL_DOWNLOADS:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', str(PROJECT)+'[search]'])
else:
    print('Optional libraries skipped.')
''')

step("Load the text embedding model", "Text embeddings support semantic matching. Loading is explicit so model-download time is separate from search latency.", '''
from fashion_assistant.embeddings import TextEncoder, CLIPEncoder
text_encoder = TextEncoder(TEXT_MODEL, TEXT_REVISION) if ENABLE_MODEL_DOWNLOADS else None
print('Text encoder:', TEXT_MODEL if text_encoder else 'not loaded')
''')

step("Encode catalog text", "Generate one vector per product. Save the model configuration with these vectors; a changed embedding model requires reindexing.", '''
text_ids = [p['product_id'] for p in products]
text_vectors = text_encoder.encode([product_text(p) for p in products]) if text_encoder else []
print('Text vectors:', len(text_vectors), '| dimension:', len(text_vectors[0]) if text_vectors else None)
''')

step("Build an exact local vector baseline", "Validate embedding retrieval locally before uploading vectors. This baseline helps distinguish embedding quality from remote-index behavior.", '''
text_store = LocalVectors(products, text_ids, text_vectors) if text_vectors else None
search.text_encoder, search.text_store = text_encoder, text_store
print('Local text index ready:', text_store is not None)
''')

step("Inspect dense text results", "Compare semantic ranking to the earlier BM25 result under identical filters.", '''
dense_results = search.search(QUERY, FILTERS, K, mode='dense') if text_store else []
print(json.dumps(dense_results, indent=2) if text_store else 'Dense search skipped.')
''')

step("Inspect fused rankings", "Reciprocal-rank fusion combines rank positions, not incompatible raw BM25/cosine scores. Review the actual product IDs that move upward.", '''
hybrid_results = fuse(bm25_results, dense_results, k=K) if text_store else []
print(json.dumps(hybrid_results, indent=2) if text_store else 'Fusion skipped: no dense ranking yet.')
''')

step("Compare BM25, dense and hybrid", "Use the same cases and K for all text methods. Adding CLIP is evaluated on the appropriate image-related cases later.", '''
if text_store and cases:
    methods.update({name: (lambda case, k, mode=name: search.search(case['query'], case['filters'], k, mode=mode)) for name in ['dense', 'hybrid']})
    comparison = run_retrieval(cases, products, methods, split=EVAL_SPLIT, k=K)
    print(json.dumps(comparison['summary'], indent=2))
    print(save_run(comparison, ARTIFACTS/'retrieval', {'text_model': TEXT_MODEL, 'revision': TEXT_REVISION, 'rrf_constant': 60, 'k': K}))
else:
    print('Comparison pending embeddings and benchmark labels.')
''')

step("Load CLIP for image retrieval", "CLIP finds visual neighbors; it does not verify fabric composition. No image model is downloaded for the synthetic sample without photographs.", '''
clip = CLIPEncoder(CLIP_MODEL, CLIP_REVISION) if ENABLE_MODEL_DOWNLOADS and image_products else None
print('CLIP ready:', clip is not None)
if clip:
    print('Resolved revision:', clip.resolved_revision)
''')

step("Encode verified catalog photographs", "Only products with readable image files participate in the image index. The same CLIP model must encode reference images.", '''
image_ids = [p['product_id'] for p in image_products]
image_vectors = clip.encode_images([IMAGE_ROOT/p['image_path'] for p in image_products]) if clip else []
print('Image vectors:', len(image_vectors), '| dimension:', len(image_vectors[0]) if image_vectors else None)
''')

step("Build the local image index", "Image vectors occupy their own embedding space, separate from the text embedding model.", '''
image_store = LocalVectors(products, image_ids, image_vectors) if image_vectors else None
search.clip, search.image_store = clip, image_store
print('Local image index ready:', image_store is not None)
''')

step("Choose a reference photograph", "Set a local path to a shopper's reference photograph. A catalog photograph may be used for a labeled self-match sanity check, not as proof of real-world image-search quality.", '''
REFERENCE_IMAGE = None  # Example: DATASET_ROOT/'query_images/reference.jpg'
if REFERENCE_IMAGE:
    assert Path(REFERENCE_IMAGE).is_file()
    display(Image(filename=str(REFERENCE_IMAGE), width=250))
else:
    print('Reference image pending. The UI also provides an image-upload control.')
''')

step("Run image similarity search", "This searches only CLIP image vectors. Text is not used in image mode; structured filters still apply.", '''
image_results = search.search('', {'slot': 'top'}, K, mode='image', image_path=REFERENCE_IMAGE) if image_store and REFERENCE_IMAGE else []
print(json.dumps(image_results, indent=2) if image_results else 'No image query executed.')
''')

step("Run image-plus-text search", "Combine visual reference ranking with text rankings. Explicit filters, rather than similarity alone, enforce a request such as 'but blue'.",'''
multimodal_results = search.search('blue top', {'slot': 'top', 'color': 'blue'}, K, mode='multimodal', image_path=REFERENCE_IMAGE) if image_store and text_store and REFERENCE_IMAGE else []
print(json.dumps(multimodal_results, indent=2) if multimodal_results else 'Multimodal search pending encoders/reference image, or no results.')
''')

step("Cache vectors with their provenance", "Keep the catalog hash, ordered IDs, model identifiers and revisions next to the vectors. This avoids accidentally attaching a vector to the wrong product.", '''
if text_vectors or image_vectors:
    import numpy as np
    np.savez_compressed(ARTIFACTS/'vectors.npz', text=np.asarray(text_vectors), image=np.asarray(image_vectors))
    (ARTIFACTS/'vectors_manifest.json').write_text(json.dumps({'catalog_sha256': catalog_audit['catalog_sha256'], 'text_ids': text_ids, 'image_ids': image_ids, 'text_model': TEXT_MODEL, 'text_revision': TEXT_REVISION, 'clip_model': CLIP_MODEL, 'clip_revision': clip.resolved_revision if clip else CLIP_REVISION}, indent=2))
    print('Saved vector cache and manifest.')
else:
    print('No vectors generated.')
''')

step("Choose whether to use Pinecone", "Enabling this section can create cloud indexes and send product vectors/metadata to Pinecone. Local vector baselines remain available. Keep False until you intend these external operations.", '''
ENABLE_PINECONE = False
TEXT_INDEX_NAME = os.environ.get('PINECONE_TEXT_INDEX') or 'fashion-text'
IMAGE_INDEX_NAME = os.environ.get('PINECONE_IMAGE_INDEX') or 'fashion-images'
PINECONE_CLOUD = 'aws'
PINECONE_REGION = 'us-east-1'
print('Pinecone enabled:', ENABLE_PINECONE)
''')

step("Load the Pinecone secret", "Read a shell variable or Colab Secret. The key is never printed, saved in notebook source or included in an evaluation manifest.", '''
def load_secret(name):
    if not os.environ.get(name):
        try:
            from google.colab import userdata
            os.environ[name] = userdata.get(name)
        except ImportError:
            from getpass import getpass
            os.environ[name] = getpass(name+': ')
    if not os.environ.get(name):
        raise ValueError('Missing secret: '+name)
    print(name, 'loaded without displaying it.')
if ENABLE_PINECONE:
    load_secret('PINECONE_API_KEY')
else:
    print('No Pinecone credentials accessed.')
''')

step("Configure separate Pinecone vector stores", "Namespaces bind a catalog snapshot to its embedding configuration. Pin model revisions for repeatable indexes. Creating the adapter itself does not create an index.", '''
from fashion_assistant.pinecone_store import PineconeVectors, index_namespace
remote_text = remote_images = None
if ENABLE_PINECONE:
    assert text_vectors and TEXT_REVISION, 'Generate text vectors and pin TEXT_REVISION before remote indexing.'
    namespace = index_namespace(catalog_audit['catalog_sha256'], TEXT_MODEL, TEXT_REVISION, len(text_vectors[0]))
    remote_text = PineconeVectors(TEXT_INDEX_NAME, namespace, len(text_vectors[0]))
    if image_vectors:
        namespace_images = index_namespace(catalog_audit['catalog_sha256'], CLIP_MODEL, clip.resolved_revision or CLIP_REVISION, len(image_vectors[0]))
        remote_images = PineconeVectors(IMAGE_INDEX_NAME, namespace_images, len(image_vectors[0]))
    print('Configured text/image adapters:', bool(remote_text), bool(remote_images))
else:
    print('Remote adapters skipped.')
''')

step("Create the text index if absent", "This is a remote write and may incur cloud charges. An existing index is retained; incompatible dimensions are rejected in the next step.", '''
if remote_text:
    print(remote_text.create_if_missing(PINECONE_CLOUD, PINECONE_REGION))
else:
    print('Text index creation skipped.')
''')

step("Check readiness and connect to text index", "A new index may take time to become ready. If this reports not ready, rerun this cell later; there is no hidden long polling loop.", '''
if remote_text:
    remote_text.connect()
    print('Connected:', TEXT_INDEX_NAME, '| namespace:', remote_text.namespace)
else:
    print('Text connection skipped.')
''')

step("Upload text vectors", "Upsert sends the actual text embeddings and filter metadata. Pinecone is eventually consistent; allow time before comparing search results.", '''
if remote_text:
    print('Uploaded text vectors:', remote_text.upsert(products, text_ids, text_vectors))
else:
    print('Text upload skipped.')
''')

step("Create the image index if absent", "CLIP uses its own vector dimension and index. It is never mixed with the separate text encoder's vectors.", '''
if remote_images:
    print(remote_images.create_if_missing(PINECONE_CLOUD, PINECONE_REGION))
else:
    print('Image index creation skipped.')
''')

step("Connect to the image index", "Check readiness and embedding dimensions before upload.", '''
if remote_images:
    remote_images.connect()
    print('Image index connected.')
else:
    print('Image connection skipped.')
''')

step("Upload image vectors", "Only verified-image product IDs are uploaded to the image namespace.", '''
if remote_images:
    print('Uploaded image vectors:', remote_images.upsert(products, image_ids, image_vectors))
else:
    print('Image upload skipped.')
''')

step("Compare remote and local retrieval", "Inspect IDs under the same query and filters. Differences may arise from indexing lag or approximate search; do not assume upload means data is already searchable.", '''
if remote_text:
    query_vector = text_encoder.encode([QUERY])[0]
    local_rows = text_store.search_vector(query_vector, FILTERS, K)
    remote_rows = remote_text.search_vector(query_vector, FILTERS, K)
    print('Local:', local_rows)
    print('Pinecone:', remote_rows)
    print('Shared IDs:', sorted({r['product_id'] for r in local_rows} & {r['product_id'] for r in remote_rows}))
else:
    print('Remote comparison skipped.')
''')

step("Choose active retrieval backends", "Switch explicitly after inspecting the remote result. Re-run benchmarks if the backend changes.", '''
USE_REMOTE_SEARCH = False
if USE_REMOTE_SEARCH:
    assert remote_text and remote_text.index, 'Connect and verify remote text retrieval first.'
    search.text_store = remote_text
    if remote_images:
        search.image_store = remote_images
print('Active backend:', 'Pinecone' if USE_REMOTE_SEARCH else 'local')
''')

step("Create a separate simulated inventory", "The dataset is not live retailer stock. This fixture assigns S/M stock and zero L stock, separately from catalog attributes.", '''
inventory = {'mode': 'SIMULATED', 'snapshot_id': 'notebook-demo-v1', 'products': {p['product_id']: {'S': 2, 'M': 2, 'L': 0} for p in products}}
print(json.dumps(dict(list(inventory['products'].items())[:3]), indent=2))
(ARTIFACTS/'inventory.json').write_text(json.dumps(inventory, indent=2))
''')

step("Confirm shopper settings", "These constraints belong to the shopper and cannot be changed by tool arguments. For another outfit type, use sizes={'set': 'M'} and corresponding filters.", '''
from fashion_assistant.validation import Settings, validate_outfit
settings = Settings(budget='2500', sizes={'top': 'M', 'bottom': 'M'}, price_unit=PRICE_UNIT,
                    filters={'top': {'color': 'black', 'fabric': 'pure cotton'}, 'bottom': {'category': 'jeans', 'color': 'blue'}})
from dataclasses import asdict
print(json.dumps(asdict(settings), indent=2))
''')

step("Inspect the validator source", "Read the exact checks rather than treating 'validation' as a model judgment. Budget totals use decimal arithmetic; missing stock is unknown, not zero.", '''
import inspect
print(inspect.getsource(validate_outfit))
''')

step("Validate a known synthetic pair", "This is a direct code check, not an agent or live model result. Real-data IDs must be independently selected after import.", '''
if not USE_REAL_DATA:
    valid_result = validate_outfit(['S01', 'S04'], products, inventory, settings, inspected={'S01', 'S04'})
    print(json.dumps(valid_result, indent=2))
    assert valid_result['valid'] and valid_result['subtotal'] == '2000'
else:
    print('Select reviewed real product IDs before running a direct positive validation case.')
''')

step("Exercise a stock failure", "Repeat the same pair with L sizes and inspect the deterministic rejection. No LLM is involved.", '''
if not USE_REAL_DATA:
    unavailable = Settings('2500', {'top': 'L', 'bottom': 'L'})
    invalid_result = validate_outfit(['S01', 'S04'], products, inventory, unavailable, {'S01', 'S04'})
    print(json.dumps(invalid_result, indent=2))
    assert not invalid_result['valid'] and 'OUT_OF_STOCK' in invalid_result['errors']
else:
    print('Stock regression is covered by the separate synthetic test suite.')
''')

step("Construct a real LangGraph with scripted responses", "The graph and Python tools are real. The model is a deterministic test double with fixed tool choices, so this tests integration, not intelligence.", '''
from fashion_assistant.agent import ShoppingAgent, scripted_pair, nebius_model
offline_agent = ShoppingAgent(products, inventory, settings, search, scripted_pair()) if not USE_REAL_DATA else None
print('Offline graph ready:', offline_agent is not None)
''')

step("Inspect the graph structure", "The model/tool loop ends on validation, clarification, no proposal, an error or a call limit. Mermaid source is printed without a rendering service.", '''
if offline_agent:
    print(offline_agent.graph.get_graph().draw_mermaid())
else:
    print('Graph structure is in src/fashion_assistant/agent.py.')
''')

step("Run the scripted shopping turn", "Watch the full search → inspection → proposal sequence execute locally. No Nebius credits are spent.", '''
offline_result = offline_agent.chat('Suggest a black pure cotton top with blue jeans.') if offline_agent else None
print(json.dumps(offline_result, indent=2) if offline_result else 'Synthetic replay skipped for real catalog.')
''')

step("Score and save the trajectory", "Objective assertions verify outcome, selected IDs and call bounds. Human claim/style review remains separate.", '''
from fashion_assistant.evaluation import score_turn, run_conversations
if offline_result:
    print(score_turn(offline_result, {'status': 'validated', 'required_ids': ['S01', 'S04'], 'max_calls': 6}))
    print(save_run(offline_result, ARTIFACTS/'scripted', {'model': 'SCRIPTED_TEST_DOUBLE', 'evidence': 'synthetic'}))
else:
    print('No scripted trace to score.')
''')

step("Run the conversation-evaluation harness", "This demonstrates a repeatable runner with a fresh graph per case/repeat. Replace fixtures with reviewed conversations for actual model comparisons.", '''
conversation_cases = load_cases(PROJECT/'evals/fixtures/conversations.jsonl') if not USE_REAL_DATA else []
if conversation_cases:
    def fixture_factory(case):
        return ShoppingAgent(products, inventory, Settings(**case['settings']), search, scripted_pair())
    conversation_report = run_conversations(conversation_cases, fixture_factory, repeats=2)
    print(json.dumps(conversation_report, indent=2))
else:
    print('Human conversation cases pending.')
''')

step("Choose the Nebius model and enable live calls", "Set the exact model ID available to your account. Verify tool support. This graph uses structured product facts; vision-LLM inspection is not yet implemented. CLIP still supports reference-image retrieval.", '''
ENABLE_LIVE_LLM = False
NEBIUS_MODEL = os.environ.get('NEBIUS_MODEL', '')
LIVE_RETRIEVAL_MODE = 'bm25'  # Change to hybrid/multimodal only after their indexes are ready.
print('Live LLM enabled:', ENABLE_LIVE_LLM, '| model:', NEBIUS_MODEL or 'not selected')
print('Up to six model calls per turn, 30 per agent instance. This is not a billing cap.')
''')

step("Load the Nebius secret", "Use Colab Secrets or a hidden local prompt. Credentials are excluded from saved traces.", '''
if ENABLE_LIVE_LLM:
    assert NEBIUS_MODEL, 'Choose your Nebius model ID first.'
    load_secret('NEBIUS_API_KEY')
else:
    print('No Nebius credentials accessed.')
''')

step("Create the live agent", "Construct the ChatNebius adapter and bind tools without sending an inference request. Changing the model requires creating a new agent.", '''
live_agent = ShoppingAgent(products, inventory, settings, search, nebius_model(NEBIUS_MODEL), mode=LIVE_RETRIEVAL_MODE) if ENABLE_LIVE_LLM else None
print('Live agent ready:', live_agent is not None)
''')

step("Send one live request", "This cell makes paid Nebius calls only when enabled. Image modes require REFERENCE_IMAGE. Do not rerun while an earlier request is still executing.", '''
live_result = live_agent.chat('Suggest a black pure cotton top with blue jeans.', image_path=REFERENCE_IMAGE) if live_agent else None
print(json.dumps(live_result, indent=2) if live_result else 'Live request skipped; no API call made.')
''')

step("Record usage and optional cost estimate", "Use currently verified rates for the selected model. Missing rates or token usage produce unknown cost, never zero. This is not an account bill.", '''
INPUT_USD_PER_MILLION = None
OUTPUT_USD_PER_MILLION = None
if live_result:
    usage = live_result['usage']
    print('Provider-reported usage:', usage)
    cost = None
    if usage and all('input_tokens' in row and 'output_tokens' in row for row in usage) and INPUT_USD_PER_MILLION is not None and OUTPUT_USD_PER_MILLION is not None:
        cost = sum(row['input_tokens']*INPUT_USD_PER_MILLION+row['output_tokens']*OUTPUT_USD_PER_MILLION for row in usage)/1_000_000
    print('Estimated USD:', cost if cost is not None else 'unknown')
    print(save_run(live_result, ARTIFACTS/'live', {'model': NEBIUS_MODEL, 'mode': LIVE_RETRIEVAL_MODE, 'estimated_usd': cost}))
else:
    print('No live usage to record.')
''')

step("Open the notebook shopping interface", "Controls include budget/sizes, explicit hard filters, chat and reference-image upload. Start conversation confirms settings; Send request makes paid calls when live mode is enabled. The interface is inert in offline mode.", '''
from fashion_assistant.notebook_ui import make_ui
ui = make_ui(products, inventory, search, IMAGE_ROOT, ARTIFACTS, live_enabled=ENABLE_LIVE_LLM, model_id=NEBIUS_MODEL)
display(ui)
''')

step("Prepare blank human relevance review sheets", "Pool candidates across methods, then let reviewers assign grades independently. The runner never invents labels. For real data without cases, copy the template and write query intents first.", '''
from fashion_assistant.evaluation import export_review_sheet
review_pool = {}
for case in cases:
    pooled = []
    for name, retrieve in methods.items():
        if name in case.get('methods', methods):
            pooled.extend(retrieve(case, K))
    review_pool[case['case_id']] = pooled
review_path = ARTIFACTS/'relevance_review_blank.csv'
export_review_sheet(cases, review_pool, review_path)
print('Blank review sheet:', review_path)
print('New case template:', PROJECT/'evals/templates/retrieval_case.json')
''')

step("Summarize completed human reviews", "Copy the human-review template, fill independent judgments, and save it at the path below. Missing reviews do not become perfect style or grounding scores. Agreement counts matched reviewer pairs before adjudication.", '''
from fashion_assistant.evaluation import summarize_human_reviews
HUMAN_REVIEWS_PATH = PROJECT/'evals/human_reviews_completed.csv'
if HUMAN_REVIEWS_PATH.exists():
    human_report = summarize_human_reviews(HUMAN_REVIEWS_PATH)
    print(json.dumps(human_report, indent=2))
    print(save_run(human_report, ARTIFACTS/'human_review', {'source': HUMAN_REVIEWS_PATH.name}))
else:
    print('Human ratings pending. Template:', PROJECT/'evals/templates/human_review.csv')
''')

step("Evaluate labeled image cases", "Image cases require human judgments, a query-image path and methods=['image'] or ['multimodal']. Do not silently evaluate image queries with the text-only methods.", '''
IMAGE_BENCHMARK_PATH = PROJECT/'evals/images_reviewed.jsonl'
if image_store and IMAGE_BENCHMARK_PATH.exists():
    image_cases = load_cases(IMAGE_BENCHMARK_PATH)
    def image_method(case, k):
        return search.search(case.get('query', ''), case['filters'], k, mode='image', image_path=DATASET_ROOT/case['image_path'])
    image_methods = {'image': image_method}
    if text_store:
        image_methods['multimodal'] = lambda case, k: search.search(case.get('query', ''), case['filters'], k, mode='multimodal', image_path=DATASET_ROOT/case['image_path'])
    image_report = run_retrieval(image_cases, products, image_methods, split=EVAL_SPLIT, k=K)
    print(json.dumps(image_report['summary'], indent=2))
    print(save_run(image_report, ARTIFACTS/'images', {'clip_model': CLIP_MODEL, 'revision': clip.resolved_revision}))
else:
    print('Image benchmark pending real images and reviewed labels; no image accuracy claimed.')
''')

step("Configure a live model comparison", "Use a fixed retriever and reviewed conversations. Each model/repeat incurs API calls. Keep this disabled until you intentionally run a benchmark; it is separate from the single-request toggle.", '''
RUN_LIVE_BENCHMARK = False
MODELS_TO_COMPARE = []  # Exact Nebius model IDs after checking tool support.
REPEATS = 3
LIVE_CASES_PATH = PROJECT/'evals/conversations_reviewed.jsonl'
print('Benchmark enabled:', RUN_LIVE_BENCHMARK, '| models:', MODELS_TO_COMPARE, '| repeats:', REPEATS)
''')

step("Execute reviewed conversation comparisons", "Each case starts a fresh agent. Outputs are saved per model; expected-status checks are only part of task quality. Use the human rubric to review explanations and outfits.", '''
if RUN_LIVE_BENCHMARK:
    assert ENABLE_LIVE_LLM and MODELS_TO_COMPARE and LIVE_CASES_PATH.exists()
    reviewed_dialogues = [c for c in load_cases(LIVE_CASES_PATH) if c['split'] == EVAL_SPLIT]
    assert reviewed_dialogues, 'No reviewed conversations for the selected split.'
    for model_id in MODELS_TO_COMPARE:
        def factory(case, selected=model_id):
            return ShoppingAgent(products, inventory, Settings(**case['settings']), search, nebius_model(selected), mode=LIVE_RETRIEVAL_MODE)
        report = run_conversations(reviewed_dialogues, factory, repeats=REPEATS)
        print(model_id, 'objective turn success:', report['turn_success_rate'])
        print(save_run(report, ARTIFACTS/'model_comparison', {'model': model_id, 'repeats': REPEATS, 'retrieval_mode': LIVE_RETRIEVAL_MODE}))
else:
    print('Live model benchmark skipped.')
''')

step("Run engineering regression tests", "This exercises catalog validation, retrieval metrics, filters, provenance, call limits and the real graph with scripted responses. It does not use your credentials or measure a real LLM.", '''
test_run = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT, capture_output=True, text=True)
print(test_run.stdout)
if test_run.returncode:
    print(test_run.stderr)
assert test_run.returncode == 0, 'Resolve regression failures before interpreting benchmark results.'
''')

step("Inspect generated artifacts and remaining work", "Keep evidence and configuration together. Real-data evaluation is complete only after data mapping, independent labels, frozen runs and human review are complete.", '''
print('Artifacts:', [str(p.relative_to(ARTIFACTS)) for p in sorted(ARTIFACTS.glob('*'))])
print('Current evidence:', 'real catalog; benchmark maturity depends on labels' if USE_REAL_DATA else 'synthetic engineering fixtures only')
print('Pending: real dataset audit, verified image joins, human labels, live integration checks and model comparison.')
''')

markdown("""
## References and interpretation

* [Nebius LangChain integration](https://docs.tokenfactory.nebius.com/integrations/frameworks/langchain)
* [Nebius tool calling](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling)
* [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
* [Pinecone index creation](https://docs.pinecone.io/guides/index-data/create-an-index)
* [Pinecone metadata filtering](https://docs.pinecone.io/guides/search/filter-by-metadata)
* [CLIP implementation](https://huggingface.co/docs/transformers/model_doc/clip)

Successful tests demonstrate only the tested behavior. Do not combine synthetic replays, local checks and live human-reviewed benchmark results into one accuracy percentage. No retailer integration, physical fit guarantee or training on the Myntra data is implemented.
""")

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Fashion Assistant (Python 3.11)", "language": "python", "name": "fashion-assistant"},
            "language_info": {"name": "python", "version": "3.11"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT/'notebooks/01_build_and_evaluate.ipynb'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Wrote {path}: {len(steps)} numbered steps, {len(cells)} cells")
