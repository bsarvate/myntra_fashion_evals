import json
from types import SimpleNamespace

import pytest

from fashion_assistant.pinecone_store import PineconeVectors, index_namespace, pinecone_filter


def test_namespaces_separate_models_and_snapshots():
    assert index_namespace('a','model','v1',2) != index_namespace('b','model','v1',2)
    assert index_namespace('a','model','v1',2) != index_namespace('a','other','v1',2)
    assert pinecone_filter({'color':'BLUE'}) == {'color':{'$eq':'blue'}}


def test_pinecone_adapter_with_fake_client(products):
    class Index:
        def upsert(self, **kwargs): self.upload = kwargs
        def query(self, **kwargs):
            self.query_args = kwargs
            return SimpleNamespace(matches=[SimpleNamespace(id='S01',score=.9)])
    index = Index()
    class Client:
        def describe_index(self, name): return SimpleNamespace(dimension=2,metric='cosine',status={'ready':True},host='fixture')
        def Index(self, host): return index
    store = PineconeVectors('fixture','namespace',2,Client())
    store.connect()
    assert store.upsert(products, ['S01'], [[1,0]]) == 1
    assert store.search_vector([1,0], {'color':'BLACK'})[0]['product_id'] == 'S01'
    assert index.query_args['filter'] == {'color':{'$eq':'black'}}
    with pytest.raises(ValueError): store.upsert(products,['S01'],[[1,0,0]])


def test_notebook_cells_have_valid_python(root):
    notebook = json.loads((root/'notebooks/01_build_and_evaluate.ipynb').read_text())
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(code) >= 60
    for index, cell in enumerate(code):
        compile(''.join(cell['source']), f'cell-{index}', 'exec')
