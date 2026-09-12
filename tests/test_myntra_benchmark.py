import copy
import json

import pytest

from fashion_assistant.myntra import classify, candidates, select_subset, QUOTAS
from fashion_assistant.benchmark120 import audit_suite, apply_image_reviews, run_boundaries
from fashion_assistant.evaluation import load_cases, audit_cases, run_retrieval, score_turn


def test_source_declared_set_precedes_title():
    assert classify('A top with trousers',{'Top Type':'Top','Bottom Type':'Trousers'}) == ('set','set')
    assert classify('Women Blue Jeans',{}) == ('bottom','jeans')
    assert classify('Women Jumpsuit with shorts',{}) is None


def test_mixed_dupatta_fabric_not_flattened(tmp_path):
    import csv
    path = tmp_path/'catalog.csv'
    attrs = {'Top Type':'Kurta','Bottom Type':'Trousers','Top Fabric':'Cotton','Bottom Fabric':'Cotton','Dupatta':'With Dupatta','Dupatta Fabric':'Silk'}
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['','p_id','price','p_attributes','name','colour','img','description','brand'])
        writer.writeheader()
        writer.writerow({'':'0','p_id':'123','price':'20','p_attributes':repr(attrs),'name':'Top with trousers','colour':'Blue','img':'https://assets.myntassets.com/assets/images/123/x.jpg','description':'Set','brand':'Fixture'})
    records,report = candidates(path)
    assert records == [] and report['rejected_counts']['missing_or_mixed_required_attributes'] == 1


def test_sampling_reproducible_and_stratified():
    from collections import Counter
    records = [{'product_id':f'{category}-{i}','category':category} for category,count in QUOTAS.items() for i in range(count+5)]
    chosen=select_subset(records)
    assert chosen == select_subset(list(reversed(records)))
    assert len(chosen)==1000 and Counter(p['category'] for p in chosen)==QUOTAS


def test_pending_reviews_are_not_scored(root,products):
    case=load_cases(root/'evals/fixtures/retrieval.jsonl')[0]
    case.update(evidence='pending_review',exhaustive=False,judgments={})
    def should_not_run(case,k):
        raise AssertionError('Pending case must not execute')
    result=run_retrieval([case],products,{'bm25':should_not_run})
    assert not result['rows'] and len(result['skipped'])==1


def test_source_derived_label_type(root,products):
    cases=load_cases(root/'evals/fixtures/retrieval.jsonl')
    cases[0]['evidence']='source_derived'
    assert audit_cases(cases,products)['cases']==6


def test_followup_scoring_detects_wrong_retained_product():
    previous={'proposal':{'valid':True,'items':[{'slot':'top','product_id':'A'},{'slot':'bottom','product_id':'B'}]}}
    result={'status':'validated','proposal':{'valid':True,'items':[{'slot':'top','product_id':'C'},{'slot':'bottom','product_id':'D'}]},'calls':1}
    score=score_turn(result,{'status':'validated','retain_slots':['top'],'replace_slots':['bottom']},previous)
    assert score['checks']['replace_bottom'] and not score['checks']['retain_top'] and not score['success']


@pytest.fixture
def prepared(root):
    path=root/'data/processed/myntra1000/catalog.json'
    if not path.exists():
        pytest.skip('Prepared user dataset is not in the source-only bundle')
    return json.loads(path.read_text())


def test_prepared_catalog_and_manifest(root,prepared):
    from fashion_assistant.catalog import fingerprint
    manifest=json.loads((root/'data/processed/myntra1000/selection_manifest.json').read_text())
    assert len(prepared)==1000 and manifest['catalog_sha256']==fingerprint(prepared)
    assert len({p['image_sha256'] for p in prepared})==1000
    for p in prepared:
        assert f"/images/{p['product_id']}/" in p['source_image_url']


def test_real_suite_structure_without_running_reserved_cases(root,prepared):
    folder=root/'evals/myntra1000_v1'
    cases=load_cases(folder/'all_cases.jsonl')
    report=audit_suite(cases)
    assert report['by_split']=={'dev':84,'test':36}
    assert report['pending_human_relevance']==10
    for filename in ['text','images']:
        audit_cases(load_cases(folder/(filename+'.jsonl')),prepared)


def test_real_development_validator_controls(root,prepared):
    stock=json.loads((root/'data/processed/myntra1000/inventory.json').read_text())
    report=run_boundaries(load_cases(root/'evals/myntra1000_v1/boundaries.jsonl'),prepared,stock,split='dev')
    assert report['passed']==report['total']==14


def test_human_review_import_keeps_missing_labels_pending(root,prepared,tmp_path):
    cases=load_cases(root/'evals/myntra1000_v1/images.jsonl')
    path=tmp_path/'reviews.csv'
    path.write_text('case_id,product_id,reviewer,relevance_0_3,notes\n')
    result=apply_image_reviews(cases,prepared,path)
    assert sum(c['evidence']=='pending_review' for c in result)==10


def test_second_notebook_compiles(root):
    path=root/'notebooks/02_myntra1000_evaluation.ipynb'
    if not path.exists():
        pytest.skip('Notebook has not yet been generated')
    notebook=json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),cell['id'],'exec')
