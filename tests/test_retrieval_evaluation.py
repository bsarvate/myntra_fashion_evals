import copy
import json

import pytest

from fashion_assistant.retrieval import BM25, LocalVectors, fuse, HybridSearch
from fashion_assistant.evaluation import audit_cases, load_cases, retrieval_metrics, run_retrieval, save_run, bootstrap_mean


def test_bm25_constraints_and_no_match(products):
    bm25 = BM25(products)
    assert bm25.search('blue jeans', {'category':'jeans'})[0]['product_id'] == 'S04'
    assert not bm25.search('pure silk jeans', {'category':'jeans','fabric':'pure silk'})
    assert not bm25.search('nonexistenttoken')


def test_fusion_rewards_agreement():
    rankings = [[{'product_id': p} for p in ids] for ids in [['A','B'], ['B','C']]]
    assert fuse(*rankings)[0]['product_id'] == 'B'
    with pytest.raises(ValueError):
        fuse([{'product_id':'A'}, {'product_id':'A'}])


def test_cosine_and_filters(products):
    store = LocalVectors(products, ['S01','S04'], [[1,0],[0,1]])
    assert store.search_vector([0, 2])[0]['product_id'] == 'S04'
    assert store.search_vector([0, 2], {'slot':'top'})[0]['product_id'] == 'S01'
    with pytest.raises(ValueError):
        store.search_vector([0,0])
    with pytest.raises(ValueError):
        store.search_vector([1,0,0])


def test_multimodal_dispatch_with_fake_encoders(products):
    class Encoder:
        def encode(self, texts): return [[1,0]]
        def encode_images(self, paths): return [[0,1]]
    store = LocalVectors(products, ['S01','S04'], [[1,0],[0,1]])
    search = HybridSearch(BM25(products), Encoder(), store, Encoder(), store)
    assert search.search('', mode='image', image_path='fixture')[0]['product_id'] == 'S04'
    assert search.search('top', mode='multimodal', image_path='fixture')
    with pytest.raises(ValueError):
        search.search('top', mode='image')


def test_group_leakage_detected(root, products):
    cases = load_cases(root/'evals/fixtures/retrieval.jsonl')
    duplicate = copy.deepcopy(cases[0])
    duplicate.update(case_id='other', split='test')
    with pytest.raises(ValueError, match='Leakage'):
        audit_cases(cases+[duplicate], products)


def test_false_exhaustive_claim_rejected(root, products):
    case = load_cases(root/'evals/fixtures/retrieval.jsonl')[0]
    case['filters'] = {}
    case['judgments'] = {'S01':3}
    with pytest.raises(ValueError, match='Exhaustive'):
        audit_cases([case], products)


def test_metrics_known_values(products):
    case = {'judgments': {'S01':3, 'S04':2, 'S02':0}, 'exhaustive':True, 'filters':{}}
    result = retrieval_metrics([{'product_id':'S01'}], case, products, 5)
    assert result['precision_at_k'] == .2
    assert result['precision_returned'] == 1
    assert result['recall_at_k'] == .5
    assert 0 < result['ndcg_at_k_judged_pool'] < 1


def test_unjudged_is_not_irrelevant(products):
    case = {'judgments': {'S01':3}, 'exhaustive':False, 'filters':{}}
    result = retrieval_metrics([{'product_id':'S04'}], case, products, 5)
    assert result['precision_at_k'] is None
    assert result['ndcg_at_k_judged_pool'] is None
    assert result['recall_at_k'] is None and result['unjudged_count'] == 1


def test_no_match_and_empty_ranking(products):
    case = {'judgments': {p['product_id']:0 for p in products}, 'exhaustive':True, 'expected_no_match':True, 'filters':{}}
    result = retrieval_metrics([], case, products, 5)
    assert result['no_match_success'] == 1
    assert result['recall_at_k'] is None and result['ndcg_at_k_judged_pool'] is None
    assert result['precision_returned'] is None


def test_errors_remain_in_report(root, products, tmp_path):
    cases = load_cases(root/'evals/fixtures/retrieval.jsonl')
    def broken(case, k): raise RuntimeError('not an evaluated zero')
    report = run_retrieval(cases, products, {'bm25':broken})
    assert report['summary']['bm25']['errors'] == len(cases)
    assert report['summary']['bm25']['recall_at_k']['n'] == 0
    path = save_run(report, tmp_path, {'model':'fixture'})
    assert json.loads((path/'manifest.json').read_text())['source_sha256']
    assert save_run(report, tmp_path, {}) != path


def test_bootstrap_reproducible():
    assert bootstrap_mean([0,1,.5]) == bootstrap_mean([0,1,.5])
    assert bootstrap_mean([1])['ci95'] is None


def test_human_reviews_missing_and_disagreement(tmp_path):
    from fashion_assistant.evaluation import summarize_human_reviews
    path = tmp_path/'reviews.csv'
    header = 'case_id,run_id,reviewer,dimension,score_0_3,claim,supported_yes_no\n'
    path.write_text(header)
    report = summarize_human_reviews(path)
    assert report['unsupported_claim_rating_rate'] is None
    assert report['pairwise_exact_agreement'] is None
    path.write_text(header+'C1,R1,A,grounding,,It is cotton,yes\nC1,R1,B,grounding,,It is cotton,no\nC1,R1,A,style,3,,\nC1,R1,B,style,3,,\n')
    report = summarize_human_reviews(path)
    assert report['unsupported_claim_rating_rate'] == .5
    assert report['pairwise_exact_agreement'] == .5
    assert report['agreement_pairs'] == 2


def test_conversation_runner_repeats_and_dialogue_success(root, products, inventory):
    from fashion_assistant.evaluation import run_conversations
    from fashion_assistant.agent import ShoppingAgent, scripted_pair
    from fashion_assistant.validation import Settings
    cases = load_cases(root/'evals/fixtures/conversations.jsonl')
    def factory(case):
        return ShoppingAgent(products, inventory, Settings(**case['settings']), HybridSearch(BM25(products)), scripted_pair())
    report = run_conversations(cases, factory, repeats=2)
    assert report['turn_success_rate'] == 1 and report['dialogue_success_rate'] == 1
    assert report['turns'] == 2 and report['dialogues'] == 2
