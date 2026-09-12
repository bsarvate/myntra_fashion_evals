"""Create the step-by-step notebook for the actual 1,000-product subset."""
import json
import textwrap
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cells, titles = [], []


def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip()+'\n'})


def step(title, reason, code):
    titles.append(title)
    md(f'## Step {len(titles):02d} — {title}\n\n{reason}')
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':textwrap.dedent(code).strip()+'\n'})


md('''
# Myntra: 1,000 products and 120 evaluation cases

This notebook uses your actual dataset subset: **400 tops, 400 bottoms, 200 sets**. The original data/raw files are preserved.

Run each numbered cell separately. Initial runs need no model API keys: they audit the data, run the BM25 development cases and exercise deterministic validation. Optional cells enable embeddings, CLIP, Pinecone and Nebius.

The benchmark has **40 text + 30 image + 30 conversation + 20 boundary cases**, grouped into **84 development / 36 reserved test cases**. Cases and objective labels are programmatically derived from catalog facts and task specifications. They are not independent human shopping queries. Ten image-plus-text cases need human visual relevance labels. Outfit and explanation quality require separate human review.

The image filenames in the original dataset are not a verified product join. The prepared subset uses photographs retrieved from each CSV image URL, with the product ID checked in that URL, decoded image validation and recorded hashes. The source association is verified; fashion attributes can still be wrong in the original listing.

For the full introductory explanation of every component, see **01_build_and_evaluate.ipynb**. This notebook applies the workflow to the prepared real subset.
''')

step('Locate the code project', 'Local Jupyter finds the enclosing project. In Colab, upload fashion-assistant-evals.zip. This is the code bundle, not the data bundle.', '''
from pathlib import Path
import json, sys, os, io, zipfile, subprocess
from IPython.display import display, Image, Markdown
PROJECT = next((p for p in [Path.cwd(), *Path.cwd().parents, Path('/content/fashion-assistant-evals')] if (p/'src/fashion_assistant').exists()), None)
def safe_extract(payload, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if not (destination/name).resolve().is_relative_to(destination):
                raise ValueError('Invalid archive path')
        archive.extractall(destination)
if PROJECT is None:
    from google.colab import files
    print('Upload fashion-assistant-evals.zip')
    uploaded = files.upload()
    assert len(uploaded) == 1
    safe_extract(next(iter(uploaded.values())), '/content')
    PROJECT = Path('/content/fashion-assistant-evals')
assert (PROJECT/'pyproject.toml').is_file()
print('Project:', PROJECT)
''')

step('Check Python and core dependencies', 'Python 3.11+ is required. Package installation may use the network; it does not make model calls.', '''
import importlib.util
assert sys.version_info >= (3,11)
print(sys.version)
if any(importlib.util.find_spec(name) is None for name in ['fashion_assistant','langgraph','ipywidgets','dotenv']):
    subprocess.check_call([sys.executable,'-m','pip','install','-e',str(PROJECT)+'[notebook,test]'])
print('Core dependencies ready.')
from dotenv import load_dotenv
load_dotenv(PROJECT/'.env', override=False)
print('Project .env loaded; values hidden.' if (PROJECT/'.env').is_file() else 'No local .env; environment variables or Colab Secrets can be used.')
''')

step('Locate or upload the prepared subset', 'Locally, the 1,000 products are already prepared. In Colab, upload myntra1000-data.zip, or set SUBSET to an extracted Drive directory. The large original dataset is not required to run these evaluations.', '''
SUBSET = PROJECT/'data/processed/myntra1000'
if not (SUBSET/'catalog.json').exists():
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError('Run scripts/prepare_myntra1000.py first or extract dist/myntra1000-data.zip under data/processed.')
    print('Upload myntra1000-data.zip')
    uploaded = files.upload()
    assert len(uploaded) == 1
    safe_extract(next(iter(uploaded.values())), PROJECT/'data/processed')
assert (SUBSET/'catalog.json').exists()
BENCHMARK = PROJECT/'evals/myntra1000_v1'
ARTIFACTS = PROJECT/'artifacts/myntra1000'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
print('Subset:', SUBSET)
''')

step('Load the 1,000 product records', 'The canonical JSON preserves source-row, attribute and image provenance. Product IDs are unique strings.', '''
products = json.loads((SUBSET/'catalog.json').read_text())
assert len(products) == 1000 and len({p['product_id'] for p in products}) == 1000
print('Products:', len(products))
print(json.dumps(products[0], indent=2, ensure_ascii=False))
''')

step('Check the frozen selection manifest', 'Verify that the loaded catalog matches the deterministic selection. Sampling used garment strata and a fixed seed, not retrieval/model scores.', '''
from fashion_assistant.catalog import fingerprint
selection = json.loads((SUBSET/'selection_manifest.json').read_text())
assert fingerprint(products) == selection['catalog_sha256']
print(json.dumps(selection, indent=2))
''')

step('Inspect category and attribute coverage', 'A 1,000-product catalog can support more than 1,000 query cases. Products are searchable records; evaluation cases are different requests and failure scenarios.', '''
from collections import Counter
print('Slots:', dict(Counter(p['slot'] for p in products)))
print('Categories:', dict(Counter(p['category'] for p in products)))
print('Colors:', Counter(p['color'] for p in products).most_common(12))
print('Fabrics:', Counter(p['fabric'] for p in products).most_common(12))
''')

step('Verify every prepared image hash', 'This detects missing or changed image files. It does not make network requests or infer clothing attributes from pixels.', '''
import hashlib
bad_images = []
for p in products:
    path = SUBSET/p['image_path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != p['image_sha256']:
        bad_images.append(p['product_id'])
assert not bad_images, f'Image verification failed: {bad_images[:10]}'
print('Verified image files:', len(products))
''')

step('Visually inspect source-image associations', 'Review representative source names and photographs across categories. This is an assistant/user inspection, not an independent human evaluation score.', '''
for index in [0,400,550,650,750,800]:
    p = products[index]
    print(p['product_id'], p['category'], p['name'])
    display(Image(filename=str(SUBSET/p['image_path']), width=180))
''')

step('Load and audit all 120 cases', 'Audit the full case structure and group split without executing reserved test queries.', '''
from fashion_assistant.evaluation import load_cases, audit_cases, run_retrieval, save_run
from fashion_assistant.benchmark120 import audit_suite, run_boundaries
all_cases = load_cases(BENCHMARK/'all_cases.jsonl')
manifest = json.loads((BENCHMARK/'manifest.json').read_text())
assert fingerprint(products) == manifest['catalog_sha256']
assert fingerprint(all_cases) == manifest['benchmark_sha256']
print(json.dumps(audit_suite(all_cases), indent=2))
''')

step('Inspect the case allocation', 'Related variants stay in one split. These are grouped generated cases, not independent real shopper observations.', '''
for kind in ['text','image','conversation','boundary']:
    subset_cases = [c for c in all_cases if c['kind']==kind]
    print(kind, len(subset_cases), dict(Counter(c['split'] for c in subset_cases)))
print('Case index:', BENCHMARK/'case_index.csv')
''')

step('Select development evaluation', 'Keep the 36 reserved test cases untouched while changing the application. Explicitly enable test evaluation only after freezing model/retrieval settings.', '''
SPLIT = 'dev'
ALLOW_RESERVED_TEST = False
K = 5
assert SPLIT != 'test' or ALLOW_RESERVED_TEST
print('Split:', SPLIT, '| K:', K)
''')

step('Inspect a text case and label provenance', 'Text labels check source-field constraints. Hard filters are supplied to retrieval, so this does not measure natural-language parsing. Conversational tests cover the agent separately.', '''
text_cases = load_cases(BENCHMARK/'text.jsonl')
print(audit_cases(text_cases, products))
example = next(c for c in text_cases if c['split']==SPLIT)
print({k:v for k,v in example.items() if k!='judgments'})
print('Relevant IDs:', [pid for pid,grade in example['judgments'].items() if grade])
''')

step('Build BM25 on the actual catalog', 'This local keyword index uses all 1,000 products.', '''
from fashion_assistant.retrieval import BM25, HybridSearch, LocalVectors
bm25 = BM25(products)
search = HybridSearch(bm25)
print('Indexed products:', len(bm25.docs))
''')

step('Run one development query', 'Inspect IDs and ranking scores before looking at aggregate results.', '''
print(json.dumps(search.search(example['query'], example['filters'], K, mode='bm25'), indent=2))
''')

step('Evaluate the text development cases', 'Only the 28 text development cases execute by default. Errors stay visible; no-match success has its own denominator.', '''
methods = {'bm25':lambda c,k:search.search(c['query'],c['filters'],k,mode='bm25')}
text_report = run_retrieval(text_cases, products, methods, split=SPLIT, k=K)
print('Executed:', len(text_report['rows']))
print(json.dumps(text_report['summary'], indent=2))
''')

step('Inspect individual text failures and save results', 'Source-derived scores describe the benchmark tasks, not general recommendation accuracy. Timestamped reports preserve catalog, benchmark and code hashes.', '''
for row in text_report['rows']:
    print(row['case_id'], row.get('returned_ids'), 'recall:', row.get('recall_at_k'), 'no-match:', row.get('no_match_success'), 'error:', row.get('error'))
print(save_run(text_report, ARTIFACTS/'text', {'method':'bm25','split':SPLIT,'k':K}))
''')

step('Load the simulated inventory', 'S and M have fixture stock; L has zero. These quantities are not from Myntra and do not establish retailer availability.', '''
inventory = json.loads((SUBSET/'inventory.json').read_text())
print(inventory['mode'], inventory['snapshot_id'])
print(dict(list(inventory['products'].items())[:3]))
''')

step('Run boundary and validation cases', 'By default this executes 14 development cases against the deterministic validator: valid controls and deliberate failures. No model requests are made.', '''
boundary_cases = load_cases(BENCHMARK/'boundaries.jsonl')
boundary_report = run_boundaries(boundary_cases, products, inventory, split=SPLIT)
for row in boundary_report['rows']:
    print(row['case_id'], 'passed:', row['passed'], 'errors:', row['result']['errors'])
assert boundary_report['passed'] == boundary_report['total']
print(save_run(boundary_report, ARTIFACTS/'boundary', {'split':SPLIT}))
''')

step('Inspect the image evaluation tasks', 'There are 10 exact-image lookups, 10 center crops from catalog photographs, and 10 image-plus-text alternatives. Exact lookup and crops are controlled stress checks, not independent shopper-image benchmarks.', '''
image_cases = load_cases(BENCHMARK/'images.jsonl')
print(audit_cases(image_cases, products))
print(Counter(c['category'] for c in image_cases))
def query_path(case):
    # Stored source paths are relative to the benchmark; support an alternate SUBSET location.
    if case['category'] != 'derived_center_crop':
        product = next(p for p in products if p['product_id']==case['source_product_id'])
        return SUBSET/product['image_path']
    return (BENCHMARK/case['image_path']).resolve()
for case in [c for c in image_cases if c['split']==SPLIT][:3]:
    assert hashlib.sha256(query_path(case).read_bytes()).hexdigest() == case['query_image_sha256']
    print(case['case_id'], case['category'], case['evidence'])
    display(Image(filename=str(query_path(case)), width=180))
''')

step('Choose whether to download embedding models', 'False keeps this notebook free of model downloads and inference. Set True to install libraries and run local text/CLIP embeddings; those use compute but no paid Nebius requests.', '''
ENABLE_EMBEDDINGS = False
TEXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
TEXT_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CLIP_REVISION = 'b33cedfd0df4e43b8238760678fcc89e1a0d38b3'  # Safetensors conversion; a commit ID, not a boolean.
print('Embedding models enabled:', ENABLE_EMBEDDINGS)
''')

step('Install optional embedding and vector dependencies', 'This step is explicit and may download large packages. Pin embedding revisions before frozen experiments.', '''
if ENABLE_EMBEDDINGS:
    subprocess.check_call([sys.executable,'-m','pip','install','-e',str(PROJECT)+'[search]'])
else:
    print('Embedding dependencies skipped.')
''')

step('Load the text embedding model', 'The selected text model has its own vector space. It is separate from CLIP.', '''
from fashion_assistant.embeddings import TextEncoder, CLIPEncoder
text_encoder = TextEncoder(TEXT_MODEL,TEXT_REVISION) if ENABLE_EMBEDDINGS else None
print('Text model ready:', bool(text_encoder))
''')

step('Encode all product text', 'Show the vector count and dimension before building an index.', '''
from fashion_assistant.catalog import product_text
ids = [p['product_id'] for p in products]
text_vectors = text_encoder.encode([product_text(p) for p in products]) if text_encoder else []
print('Text vectors:', len(text_vectors), 'dimension:', len(text_vectors[0]) if text_vectors else None)
''')

step('Build the local semantic index', 'An exact cosine index is a transparent baseline for later Pinecone comparisons.', '''
text_store = LocalVectors(products,ids,text_vectors) if text_vectors else None
search.text_encoder,search.text_store = text_encoder,text_store
print('Semantic index ready:', bool(text_store))
''')

step('Compare text retrieval methods', 'Compare BM25, dense and fusion on identical development cases and filters.', '''
if text_store:
    methods.update({name:(lambda c,k,m=name:search.search(c['query'],c['filters'],k,mode=m)) for name in ['dense','hybrid']})
    comparison = run_retrieval(text_cases,products,methods,split=SPLIT,k=K)
    print(json.dumps(comparison['summary'],indent=2))
    print(save_run(comparison,ARTIFACTS/'text_comparison',{'text_model':TEXT_MODEL,'revision':TEXT_REVISION,'split':SPLIT,'k':K}))
else:
    print('Semantic comparison pending embedding setup.')
''')

step('Load CLIP', 'CLIP encodes catalog and query photographs with the same model. It supports similarity retrieval, not fabric verification.', '''
clip = CLIPEncoder(CLIP_MODEL,CLIP_REVISION) if ENABLE_EMBEDDINGS else None
print('CLIP ready:', bool(clip))
''')

step('Encode the 1,000 product photographs', 'This can take several minutes on a CPU. Keep model/index preparation time separate from query latency.', '''
image_vectors = clip.encode_images([SUBSET/p['image_path'] for p in products]) if clip else []
print('Image vectors:',len(image_vectors),'dimension:',len(image_vectors[0]) if image_vectors else None)
''')

step('Build the separate image index', 'The image index uses CLIP dimensions, not the text encoder dimensions.', '''
image_store = LocalVectors(products,ids,image_vectors) if image_vectors else None
search.clip,search.image_store = clip,image_store
print('Image index ready:',bool(image_store))
''')

step('Run the labeled image development checks', 'Only known-image identity/crop labels are scored automatically. The pending visual-alternative cases are explicitly skipped.', '''
if image_store:
    image_report = run_retrieval(image_cases,products,{'image':lambda c,k:search.search('',c['filters'],k,mode='image',image_path=query_path(c))},split=SPLIT,k=K)
    print(json.dumps(image_report['summary'],indent=2))
    print('Pending review:',image_report['skipped'])
    print(save_run(image_report,ARTIFACTS/'image_identity',{'clip_model':CLIP_MODEL,'revision':clip.resolved_revision,'split':SPLIT}))
else:
    print('Image scoring pending CLIP; no image accuracy result claimed.')
''')

step('Review the image-and-text alternatives', 'The blank sheet lists candidates satisfying each case’s structured filters. Review their images against the reference and assign grades 0–3; do not infer visual relevance from color alone.', '''
review_path = BENCHMARK/'image_relevance_review.csv'
print('Fill reviewer names and relevance grades:',review_path)
pending = [c for c in image_cases if c['evidence']=='pending_review' and c['split']==SPLIT]
print('Pending development requests:',[(c['case_id'],c['query']) for c in pending])
display(Markdown((PROJECT/'evals/RUBRIC.md').read_text()))
''')

step('Import completed human relevance labels', 'Set the flag only after review. Empty rows stay unjudged; conflicting reviewer grades must be explicitly resolved. A new reviewed file preserves the original generated cases.', '''
IMPORT_HUMAN_REVIEWS = False
from fashion_assistant.benchmark120 import apply_image_reviews
reviewed_images = image_cases
if IMPORT_HUMAN_REVIEWS:
    reviewed_images = apply_image_reviews(image_cases,products,review_path)
    audit_cases(reviewed_images,products)
    output = BENCHMARK/'images_reviewed.jsonl'
    output.write_text(''.join(json.dumps(c)+'\\n' for c in reviewed_images))
    print('Saved:',output)
else:
    print('Human-label import skipped.')
''')

step('Evaluate image-and-text retrieval', 'Unreviewed cases remain skipped. Meaningful graded visual relevance scores require actual human labels.', '''
if image_store and text_store:
    result = run_retrieval(reviewed_images,products,{'multimodal':lambda c,k:search.search(c['query'],c['filters'],k,mode='multimodal',image_path=query_path(c))},split=SPLIT,k=K)
    print(json.dumps(result['summary'],indent=2))
    print('Skipped:',result['skipped'])
    print(save_run(result,ARTIFACTS/'multimodal',{'text_model':TEXT_MODEL,'clip_model':CLIP_MODEL,'split':SPLIT}))
else:
    print('Multimodal scoring pending embeddings and human labels.')
''')

step('Choose optional Pinecone execution', 'This enables remote index creation and vector uploads. Keep False for local evaluations. The detailed introductory notebook explains each Pinecone operation separately.', '''
ENABLE_PINECONE = False
print('Remote vector operations enabled:',ENABLE_PINECONE)
def load_secret(name):
    if not os.environ.get(name):
        try:
            from google.colab import userdata
            os.environ[name] = userdata.get(name)
        except ImportError:
            from getpass import getpass
            os.environ[name] = getpass(name+': ')
    assert os.environ.get(name), 'Missing secret: '+name
    print(name,'loaded without printing it.')
''')

step('Configure Pinecone indexes', 'Namespaces identify this catalog and each pinned embedding model. Keys never enter notebook source or saved reports.', '''
from fashion_assistant.pinecone_store import PineconeVectors,index_namespace
remote_text = remote_images = None
if ENABLE_PINECONE:
    assert text_vectors and image_vectors and TEXT_REVISION, 'Prepare embeddings and pin TEXT_REVISION first.'
    load_secret('PINECONE_API_KEY')
    remote_text = PineconeVectors(os.environ.get('PINECONE_TEXT_INDEX') or 'fashion-text',index_namespace(selection['catalog_sha256'],TEXT_MODEL,TEXT_REVISION,len(text_vectors[0])),len(text_vectors[0]))
    remote_images = PineconeVectors(os.environ.get('PINECONE_IMAGE_INDEX') or 'fashion-images',index_namespace(selection['catalog_sha256'],CLIP_MODEL,clip.resolved_revision,len(image_vectors[0])),len(image_vectors[0]))
print('Remote stores configured:',bool(remote_text),bool(remote_images))
''')

step('Create the text index', 'This creates a cloud resource only when enabled; it preserves existing compatible indexes.', '''
if remote_text:
    print(remote_text.create_if_missing())
else:
    print('Skipped.')
''')

step('Connect and upload text vectors', 'If the index is not yet ready, rerun this cell later. Index consistency can lag after upload.', '''
if remote_text:
    remote_text.connect()
    print('Text vectors uploaded:',remote_text.upsert(products,ids,text_vectors))
else:
    print('Skipped.')
''')

step('Create the image index', 'Image vectors live in their own index with their own dimension.', '''
if remote_images:
    print(remote_images.create_if_missing())
else:
    print('Skipped.')
''')

step('Connect and upload image vectors', 'Upload is optional and separate from the verified local evaluation route.', '''
if remote_images:
    remote_images.connect()
    print('Image vectors uploaded:',remote_images.upsert(products,ids,image_vectors))
else:
    print('Skipped.')
''')

step('Inspect remote search before switching', 'Compare the same query under the same filters. A populated remote index is not assumed ready merely because upload returned.', '''
USE_REMOTE_SEARCH = False
if remote_text:
    vector = text_encoder.encode([example['query']])[0]
    print('Local:',text_store.search_vector(vector,example['filters'],K))
    print('Remote:',remote_text.search_vector(vector,example['filters'],K))
if USE_REMOTE_SEARCH:
    assert remote_text and remote_images
    search.text_store,search.image_store = remote_text,remote_images
print('Active backend:','Pinecone' if USE_REMOTE_SEARCH else 'local')
''')

step('Inspect the 30 conversation cases', 'Each case has confirmed settings and one or two turns. Expectations cover valid selection, retained items, replacement and unavailable sizes; subjective style and factual prose need human review.', '''
conversation_cases = load_cases(BENCHMARK/'conversations.jsonl')
dev_dialogues = [c for c in conversation_cases if c['split']==SPLIT]
print('Conversations in selected split:',len(dev_dialogues))
print(json.dumps(dev_dialogues[0],indent=2))
''')

step('Choose Nebius models and benchmark repetition', 'Live comparison is disabled by default. Each turn can make six paid calls. A full repeated benchmark can make many requests; call bounds are not billing caps.', '''
ENABLE_NEBIUS = False
RUN_LIVE_BENCHMARK = False
NEBIUS_MODEL = os.environ.get('NEBIUS_MODEL','')
MODELS_TO_COMPARE = [NEBIUS_MODEL] if NEBIUS_MODEL else []
REPEATS = 1
RETRIEVAL_MODE = 'bm25'  # hybrid after text setup; image/multimodal via uploaded reference in the UI.
if ENABLE_NEBIUS:
    assert NEBIUS_MODEL
    load_secret('NEBIUS_API_KEY')
print('Live model enabled:',ENABLE_NEBIUS,'| benchmark enabled:',RUN_LIVE_BENCHMARK)
''')

step('Open the actual-catalog shopping UI', 'Start conversation confirms budget, sizes and filters. Send request makes paid calls only when live mode is enabled. Reference images require configured CLIP retrieval.', '''
from fashion_assistant.notebook_ui import make_ui
display(make_ui(products,inventory,search,SUBSET,ARTIFACTS,live_enabled=ENABLE_NEBIUS,model_id=NEBIUS_MODEL))
''')

step('Run the reviewed-scope conversation expectations', 'The expected objective outcomes are source-derived. This runner measures model behavior only when real models are enabled; it never presents scripted outputs as Nebius results.', '''
from fashion_assistant.agent import ShoppingAgent,nebius_model
from fashion_assistant.validation import Settings
from fashion_assistant.evaluation import run_conversations
if RUN_LIVE_BENCHMARK:
    assert ENABLE_NEBIUS and MODELS_TO_COMPARE
    for model_id in MODELS_TO_COMPARE:
        def factory(case,chosen=model_id):
            return ShoppingAgent(products,inventory,Settings(**case['settings']),search,nebius_model(chosen),mode=RETRIEVAL_MODE)
        # Trial scope; use None only when ready for all selected-split conversations.
        CASE_LIMIT = 1
        selected_dialogues = dev_dialogues[:CASE_LIMIT] if CASE_LIMIT is not None else dev_dialogues
        turn_count = sum(len(case['turns']) for case in selected_dialogues) * REPEATS
        print(f'Starting {len(selected_dialogues)} conversations, {turn_count} turns, up to {turn_count * 6} model calls.', flush=True)
        from datetime import datetime, timezone
        checkpoint = ARTIFACTS/'nebius'/('checkpoint_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))/'report.json'
        print('Checkpoint:',checkpoint,flush=True)
        report = run_conversations(selected_dialogues,factory,repeats=REPEATS,
                                   checkpoint_path=checkpoint,progress=True,stop_on_api_error=True)
        report['scope'] = 'trial' if len(selected_dialogues) < len(dev_dialogues) else 'selected_split'
        report['catalog_sha256'] = fingerprint(products)
        # Save completed calls before displaying diagnostics, including older in-memory reports.
        run_path = save_run(report,ARTIFACTS/'nebius',{'model':model_id,'repeats':REPEATS,'retrieval':RETRIEVAL_MODE,'split':SPLIT})
        print('Saved:',run_path)
        api_errors = sum(row.get('result',{}).get('status') == 'api_error' for row in report['rows'])
        execution_errors = sum(row.get('error') is not None for row in report['rows'])
        print('Run status:',report.get('run_status','legacy report; inspect error counts'),'| API error turns:',api_errors,'| execution error turns:',execution_errors)
        print(model_id,'turn success:',report['turn_success_rate'],'dialogue success:',report['dialogue_success_rate'])
else:
    print('No Nebius benchmark calls made. These 30 cases are prepared, not live-evaluated.')
''')

step('Run engineering tests', 'Tests check the implementation and dataset/benchmark structure. They do not execute the reserved model/retrieval cases or make paid API calls.', '''
result = subprocess.run([sys.executable,'-m','pytest','-q'],cwd=PROJECT,capture_output=True,text=True)
print(result.stdout)
if result.returncode:
    print(result.stderr)
assert result.returncode == 0
''')

step('Read the run status and sharing instructions', 'The 1,000-row subset and 120-case definitions are ready. Live model/image scores and human visual/style reviews are separate unfinished evaluation stages.', '''
print('Catalog:',SUBSET/'catalog.csv')
print('All 120 cases:',BENCHMARK/'case_index.csv')
print('Human image review:',BENCHMARK/'image_relevance_review.csv')
print('Generated results:',ARTIFACTS)
print('Colab uses dist/fashion-assistant-evals.zip plus dist/myntra1000-data.zip.')
print('No general recommendation accuracy claim follows from source-derived checks.')
''')

for i,cell in enumerate(cells):
    cell['id'] = f'myntra-{i:03d}'
notebook = {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
    'kernelspec':{'name':'fashion-assistant','display_name':'Fashion Assistant (Python 3.11)','language':'python'},
    'language_info':{'name':'python','version':'3.11'},'colab':{'provenance':[]}}}
path = root/'notebooks/02_myntra1000_evaluation.ipynb'
path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
print(f'Wrote {path.name}: {len(titles)} steps')
