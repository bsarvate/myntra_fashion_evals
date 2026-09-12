"""120 concrete, provenance-labeled cases; never fabricate human judgments."""
import copy
import csv
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .catalog import fingerprint, matches
from .validation import Settings, validate_outfit


def make_split(family, count):
    groups = [f'{family}-{i:02d}' for i in range(count)]
    random.Random('benchmark120-v1-'+family).shuffle(groups)
    test = set(groups[:round(count*.3)])
    return {group: 'test' if group in test else 'dev' for group in groups}


def base(case_id, group, splits, kind, category, evidence='source_derived'):
    return {'case_id':case_id, 'group_id':group, 'split':splits[group], 'kind':kind,
            'category':category, 'evidence':evidence, 'version':'myntra1000-benchmark120-v1'}


def text_cases(products):
    splits, cases = make_split('text',20), []
    # Evenly spaced selection spans all six strata in the prepared catalog.
    targets = [products[i*len(products)//20] for i in range(20)]
    for i, p in enumerate(targets):
        group = f'text-{i:02d}'
        filters = {k:p[k] for k in ('slot','category','color','fabric')}
        descriptor = f"{p['color']} {p['fabric']} {p['category']}"
        for variant in range(2):
            constraints = dict(filters)
            no_match = variant == 1 and i >= 10
            if no_match:
                for fabric in ['Pure Silk','Pure Linen','Pure Wool','Cashmere','Pure Cashmere']:
                    constraints['fabric'] = fabric
                    if not any(matches(product, constraints) for product in products):
                        break
                else:
                    raise ValueError('Cannot construct the intended no-match case')
                query = f"Find a {p['color']} {p['category']} made of {fabric}. Do not substitute another fabric."
            elif variant:
                query = f"I'm looking for a {p['category']} in {p['color']}, made from {p['fabric']}."
            else:
                query = 'Find a '+descriptor+'.'
            case = base(f'T{i*2+variant+1:03d}', group, splits, 'text', 'catalog_no_match' if no_match else ('wording_variant' if variant else 'exact_attributes'))
            case.update(query=query, image_path=None, filters=constraints, methods=['bm25','dense','hybrid'], exhaustive=True,
                        expected_no_match=no_match, judgments={x['product_id']: (3 if matches(x,constraints) else 0) for x in products},
                        label_basis='Binary relevance derived from exact catalog fields, not human style relevance.',
                        measures='Retrieval under supplied filters; does not evaluate natural-language filter extraction.')
            cases.append(case)
    return cases


def image_cases(products, subset, output):
    from PIL import Image
    splits, cases = make_split('image',10), []
    query_dir = output/'query_images'
    query_dir.mkdir(parents=True, exist_ok=True)
    targets, hashes = [], set()
    for p in products[::max(len(products)//10,1)]:
        if p['image_sha256'] not in hashes:
            targets.append(p)
            hashes.add(p['image_sha256'])
        if len(targets) == 10:
            break
    if len(targets) != 10:
        raise ValueError('Need ten distinct source images')
    for i, p in enumerate(targets):
        group = f'image-{i:02d}'
        original = subset/p['image_path']
        for variant in range(3):
            kind = ['exact_image_lookup','derived_center_crop','image_plus_text_alternative'][variant]
            case = base(f'I{i*3+variant+1:03d}', group, splits, 'image', kind,
                        'pending_review' if variant == 2 else 'source_derived')
            if variant == 1:
                query_path = query_dir/(p['product_id']+'-center-crop.jpg')
                with Image.open(original) as image:
                    width,height = image.size
                    image.crop((int(width*.15),int(height*.15),int(width*.85),int(height*.85))).convert('RGB').save(query_path,quality=95)
                rel_path = str(Path('query_images')/query_path.name)
            else:
                query_path = original
                rel_path = str(Path('../../data/processed/myntra1000')/p['image_path'])
            filters = {'slot':p['slot'], 'category':p['category']}
            query = ''
            if variant == 2:
                alternatives = [x for x in products if x['category']==p['category'] and x['color'] != p['color']]
                target_color = alternatives[i%len(alternatives)]['color']
                filters['color'] = target_color
                query = f"Find a {p['category']} similar to the reference, but in {target_color}."
            case.update(query=query, image_path=rel_path, source_product_id=p['product_id'],
                        source_image_sha256=p['image_sha256'], query_image_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),
                        filters=filters, methods=['multimodal'] if variant == 2 else ['image'],
                        exhaustive=variant != 2, expected_no_match=False,
                        judgments={} if variant == 2 else {x['product_id']:3 if x['image_sha256']==p['image_sha256'] else 0 for x in products},
                        label_basis='Human visual relevance pending' if variant == 2 else 'Known source-image identity; identical-source-image catalog records all relevant.',
                        image_provenance='Catalog-derived query; NOT an independent shopper photograph.',
                        review_status='pending' if variant == 2 else 'automatic identity labels')
            cases.append(case)
    return cases


def conversation_cases(products):
    splits, cases = make_split('conversation',10), []
    tops = [p for p in products if p['slot']=='top']
    bottoms = [p for p in products if p['slot']=='bottom']
    budget = str(max(Decimal(p['price']) for p in tops)+max(Decimal(p['price']) for p in bottoms)+100)
    for i in range(10):
        p = tops[i*len(tops)//10]
        top_filters = {k:p[k] for k in ('color','fabric','category')}
        allowed = { 'top':[x['product_id'] for x in tops if matches(x,top_filters)], 'bottom':[x['product_id'] for x in bottoms] }
        request = f"Suggest a {p['color']} {p['fabric']} top with a bottom, within my confirmed budget."
        settings = {'budget':budget,'sizes':{'top':'M','bottom':'M'},'price_unit':'CATALOG_UNITS','filters':{'top':top_filters}}
        for variant in range(3):
            category = ['retain_outfit','replace_bottom_keep_top','unavailable_size'][variant]
            case = base(f'C{i*3+variant+1:03d}', f'conversation-{i:02d}', splits, 'conversation', category)
            config = copy.deepcopy(settings)
            if variant == 2:
                config['sizes'] = {'top':'L','bottom':'L'}
                turns = [{'request':request+' Keep my confirmed L sizes.',
                          'expected':{'status_any_of':['no_validated_outfit','clarification'], 'max_calls':6}}]
            else:
                turns = [{'request':request, 'expected':{'status':'validated','allowed_ids_by_slot':allowed,'max_calls':6}}]
                expected = {'status':'validated','allowed_ids_by_slot':allowed,'max_calls':6,'retain_slots':['top']}
                if variant == 0:
                    expected['retain_slots'].append('bottom')
                    followup = 'Keep both items from your last outfit and check them again.'
                else:
                    expected['replace_slots'] = ['bottom']
                    followup = 'Keep that exact top, but replace the bottom with a different product. Keep my sizes and budget.'
                turns.append({'request':followup,'expected':expected})
            case.update(settings=config, turns=turns, inventory_snapshot='myntra1000-demo-v1',
                        label_basis='Programmatic budget/attribute/identity/stock expectations. Human wording, style and claim review pending.')
            cases.append(case)
    return cases


def boundary_cases(products):
    splits, cases = make_split('boundary',10), []
    tops = [p for p in products if p['slot']=='top']
    bottoms = [p for p in products if p['slot']=='bottom']
    negative = ['over_budget','unknown_id','missing_inspection','zero_stock','unknown_stock',
                'duplicate_id','missing_bottom','unit_mismatch','fabric_conflict','wrong_outfit_structure']
    error = ['OVER_BUDGET','UNKNOWN_ID','INSPECTION_REQUIRED','OUT_OF_STOCK','STOCK_UNKNOWN',
             'DUPLICATE_ID','OUTFIT_STRUCTURE','PRICE_UNIT_MISMATCH','CONSTRAINT_VIOLATION','OUTFIT_STRUCTURE']
    for i in range(10):
        top,bottom = tops[i],bottoms[i]
        ids = [top['product_id'],bottom['product_id']]
        total = Decimal(top['price'])+Decimal(bottom['price'])
        for variant in range(2):
            case = base(f'B{i*2+variant+1:03d}',f'boundary-{i:02d}',splits,'boundary','exact_budget_valid' if not variant else negative[i])
            case.update(settings={'budget':str(total),'sizes':{'top':'M','bottom':'M'},'price_unit':'CATALOG_UNITS'},
                        product_ids=ids.copy(), inspected=ids.copy(), inventory_changes=[],
                        expected_valid=not variant, expected_errors=[] if not variant else [error[i]],
                        label_basis='Deterministic validator specification; no model or human judgment.')
            if variant:
                if i == 0: case['settings']['budget'] = str(total-Decimal('.01'))
                elif i == 1: case['product_ids'][1] = 'NONEXISTENT-TEST-ID'
                elif i == 2: case['inspected'] = []
                elif i in (3,4): case['inventory_changes'] = [{'product_id':ids[0],'size':'M','quantity':0 if i == 3 else None}]
                elif i == 5: case['product_ids'] = [ids[0],ids[0]]
                elif i == 6: case['product_ids'] = [ids[0]]
                elif i == 7: case['settings']['price_unit'] = 'UNCONFIRMED_OTHER_UNIT'
                elif i == 8: case['settings']['filters'] = {'top':{'fabric':'Deliberately nonmatching test fabric'}}
                elif i == 9: case['settings']['sizes'] = {'set':'M'}
            cases.append(case)
    return cases


def run_boundaries(cases, products, inventory, split='dev'):
    rows = []
    for case in cases:
        if case['split'] != split:
            continue
        stock = copy.deepcopy(inventory)
        for change in case['inventory_changes']:
            stock['products'][change['product_id']][change['size']] = change['quantity']
        result = validate_outfit(case['product_ids'], products, stock, Settings(**case['settings']), set(case['inspected']))
        passed = result['valid'] == case['expected_valid'] and set(case['expected_errors']) <= set(result['errors'])
        rows.append({'case_id':case['case_id'],'passed':passed,'result':result})
    return {'rows':rows,'passed':sum(r['passed'] for r in rows),'total':len(rows),'split':split,
            'evidence':'source-derived validator cases; no LLM requests', 'catalog_sha256':fingerprint(products)}


def audit_suite(cases):
    if len(cases) != 120 or len({c['case_id'] for c in cases}) != 120:
        raise ValueError('Require exactly 120 unique cases')
    if Counter(c['kind'] for c in cases) != {'text':40,'image':30,'conversation':30,'boundary':20}:
        raise ValueError('Wrong case allocation')
    groups = {}
    for case in cases:
        group = case['group_id']
        if group in groups and groups[group] != case['split']:
            raise ValueError('Related-case split leakage')
        groups[group] = case['split']
    return {'cases':120, 'by_kind':dict(Counter(c['kind'] for c in cases)),
            'by_split':dict(Counter(c['split'] for c in cases)), 'groups':len(groups),
            'pending_human_relevance':sum(c['evidence']=='pending_review' for c in cases),
            'benchmark_sha256':fingerprint(cases)}


def build_suite(subset, output):
    subset,output = Path(subset),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    review_file = output/'image_relevance_review.csv'
    if review_file.exists():
        with review_file.open(newline='') as handle:
            if any(row.get('relevance_0_3','').strip() for row in csv.DictReader(handle)):
                raise ValueError('Completed human labels exist. Create a new benchmark version rather than overwriting them.')
    products = json.loads((subset/'catalog.json').read_text())
    if len(products) != 1000:
        raise ValueError('Benchmark expects the frozen 1,000-product subset')
    parts = {'text':text_cases(products),'images':image_cases(products,subset,output),
             'conversations':conversation_cases(products),'boundaries':boundary_cases(products)}
    expected_crops = {Path(c['image_path']).name for c in parts['images'] if c['category']=='derived_center_crop'}
    for path in (output/'query_images').glob('*.jpg'):
        if path.name not in expected_crops:
            archive = output.parents[1]/'artifacts/superseded_query_images'
            archive.mkdir(parents=True,exist_ok=True)
            path.replace(archive/path.name)
    all_cases = [case for cases in parts.values() for case in cases]
    manifest = audit_suite(all_cases)
    manifest.update(catalog_sha256=fingerprint(products), version='myntra1000-benchmark120-v1',
                    scope='Generated development benchmark with a reserved test split, not independent human shopping-query evidence.',
                    split_policy='Related variants grouped; 84 dev / 36 reserved test. No test model/retrieval runs made during preparation.',
                    image_policy='10 exact-image lookups + 10 catalog-derived crops + 10 image/text requests pending human relevance judgments.')
    for name,cases in {**parts,'all_cases':all_cases}.items():
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (output/'case_index.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['case_id','group_id','split','kind','category','evidence'])
        writer.writeheader()
        writer.writerows({k:c[k] for k in writer.fieldnames} for c in all_cases)
    with (output/'image_relevance_review.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['case_id','product_id','reviewer','relevance_0_3','notes'])
        writer.writeheader()
        for case in parts['images']:
            if case['evidence']=='pending_review':
                for p in products:
                    if matches(p,case['filters']):
                        writer.writerow({'case_id':case['case_id'],'product_id':p['product_id']})
    return manifest


def apply_image_reviews(cases, products, review_csv):
    """Import explicit grades; preserve pending cases and reject disagreements."""
    result = copy.deepcopy(cases)
    by_id = {c['case_id']:c for c in result}
    catalog = {p['product_id']:p for p in products}
    labels, reviewers = {}, {}
    with Path(review_csv).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if not row['relevance_0_3'].strip():
                continue
            cid,pid = row['case_id'],row['product_id']
            if cid not in by_id or pid not in catalog or not row['reviewer'].strip():
                raise ValueError('Known case/product IDs and reviewer attribution required')
            if by_id[cid]['category'] != 'image_plus_text_alternative':
                raise ValueError('This importer only updates the pending visual relevance tasks')
            grade = int(row['relevance_0_3'])
            if grade not in (0,1,2,3):
                raise ValueError('Grade must be 0–3')
            if grade and not matches(catalog[pid],by_id[cid]['filters']):
                raise ValueError('Positive grade violates an explicit hard requirement')
            previous = labels.setdefault(cid,{}).get(pid)
            if previous is not None and previous != grade:
                raise ValueError('Resolve reviewer disagreement explicitly before importing final labels')
            labels[cid][pid] = grade
            reviewers.setdefault(cid,set()).add(row['reviewer'].strip())
    for cid,judgments in labels.items():
        case = by_id[cid]
        case.update(judgments=judgments, evidence='human_reviewed', review_status='reviewed',
                    reviewers=sorted(reviewers[cid]), label_basis='Attributed human visual relevance judgments')
        eligible = {pid for pid,p in catalog.items() if matches(p,case['filters'])}
        case['exhaustive'] = eligible <= judgments.keys()
        case['expected_no_match'] = case['exhaustive'] and not any(judgments.values())
    return result
