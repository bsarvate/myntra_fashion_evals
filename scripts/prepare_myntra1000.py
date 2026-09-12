"""Prepare 1,000 source-grounded rows and URL-verified photographs, resumably."""
import concurrent.futures
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

from fashion_assistant.myntra import candidates, select_subset, write_catalog, QUOTAS, SEED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/raw/Fashion Dataset.csv'
OUTPUT = ROOT/'data/processed/myntra1000'
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT/'images').mkdir(exist_ok=True)
records, audit = candidates(SOURCE)
selected = select_subset(records)
cache_path = OUTPUT/'image_provenance.json'
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
hash_index_path = ROOT/'artifacts/raw_image_hash_index.json'
raw_hashes = json.loads(hash_index_path.read_text()) if hash_index_path.exists() else {}


def get_image(product):
    pid, url = product['product_id'], product['source_image_url']
    path = OUTPUT/product['image_path']
    previous = cache.get(pid)
    if path.exists() and previous and previous['source_url'] == url and hashlib.sha256(path.read_bytes()).hexdigest() == previous['sha256']:
        return pid, previous
    response = requests.get(url, timeout=(10, 40))
    response.raise_for_status()
    data = response.content
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    return pid, {'source_url': url, 'sha256': digest, 'bytes': len(data), 'width': width, 'height': height,
                 'verified_utc': datetime.now(timezone.utc).isoformat(),
                 'matching_original_files': raw_hashes.get(digest, []),
                 'method': 'downloaded exact CSV image URL containing product ID; JPEG decoded successfully',
                 'limitation': 'Verifies the listing-image association, not correctness of listing attributes.'}


errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(get_image, p): p['product_id'] for p in selected}
    for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
        try:
            pid, evidence = future.result()
            cache[pid] = evidence
        except Exception as exc:
            errors.append({'product_id': futures[future], 'error': type(exc).__name__})
        cache_path.write_text(json.dumps(cache, indent=2))
        if done % 50 == 0:
            print(f'Checked {done}/1000 images; failures: {len(errors)}', flush=True)
if errors:
    (OUTPUT/'download_errors.json').write_text(json.dumps(errors, indent=2))
    raise RuntimeError(f'{len(errors)} image requests failed; rerun to reuse successful downloads.')
for p in selected:
    p['image_verified'] = True
    p['image_sha256'] = cache[p['product_id']]['sha256']
    p['image_join_provenance'] = 'CSV source URL, product ID embedded in URL; no positional filename join'
selected_ids = {p['product_id'] for p in selected}
# Keep exactly the selected images in the active subset; retain earlier generated files separately.
for path in (OUTPUT/'images').glob('*.jpg'):
    if path.stem not in selected_ids:
        archive = ROOT/'artifacts/superseded_myntra_images'
        archive.mkdir(exist_ok=True)
        target = archive/(path.stem+'-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]+'.jpg')
        path.replace(target)
cache = {pid:value for pid,value in cache.items() if pid in selected_ids}
cache_path.write_text(json.dumps(cache,indent=2))
catalog_hash = write_catalog(selected, OUTPUT)
audit.update(selected=1000, quotas=QUOTAS, seed=SEED, source_csv_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             catalog_sha256=catalog_hash, verified_images=len(selected),
             selected_unique_image_hashes=len({p['image_sha256'] for p in selected}),
             price_unit='CATALOG_UNITS; currency not independently confirmed',
             selection='Deterministic SHA-256 order within garment strata; no model/retrieval outcomes used.')
(OUTPUT/'selection_manifest.json').write_text(json.dumps(audit, indent=2))
(OUTPUT/'inventory.json').write_text(json.dumps({'mode':'SIMULATED','snapshot_id':'myntra1000-demo-v1',
    'products':{p['product_id']:{'S':2,'M':2,'L':0} for p in selected}}, indent=2))
print(json.dumps({'output':str(OUTPUT), 'products':1000, 'slots':dict(Counter(p['slot'] for p in selected)),
                  'verified_images':1000, 'unique_image_hashes':audit['selected_unique_image_hashes']}, indent=2))
