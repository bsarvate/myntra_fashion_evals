from langchain_core.messages import AIMessage

from fashion_assistant.agent import ShoppingAgent, ScriptedModel, scripted_pair
from fashion_assistant.retrieval import BM25, HybridSearch
from fashion_assistant.validation import Settings


def call(name, args, cid='x'):
    return AIMessage(content='', tool_calls=[{'name':name,'args':args,'id':cid,'type':'tool_call'}])


def agent(products, inventory, model, **kwargs):
    return ShoppingAgent(products, inventory, Settings('3000', {'top':'M','bottom':'M'}), HybridSearch(BM25(products)), model, **kwargs)


def test_real_graph_valid_pair(products, inventory):
    result = agent(products, inventory, scripted_pair()).chat('Black cotton top and blue jeans')
    assert result['status'] == 'validated'
    assert result['proposal']['subtotal'] == '2000'
    assert [row['tool'] for row in result['trace']] == ['search_products','search_products','inspect_product','inspect_product','propose_outfit']


def test_invalid_bottom_category_returns_actionable_feedback(products, inventory):
    shopper = agent(products, inventory, ScriptedModel([]))
    search = shopper.tools['search_products']
    result = search.invoke({'query': 'blue jeans', 'slot': 'bottom',
                            'filters': {'category': 'bottom', 'color': 'blue'}})
    assert result['error'] == 'INVALID_CATEGORY'
    assert 'jeans' in result['allowed_categories']
    assert not shopper.seen
    corrected = search.invoke({'query': 'blue jeans', 'slot': 'bottom',
                               'filters': {'category': 'jeans', 'color': 'blue'}})
    assert corrected['results']
    assert corrected['effective_filters']['color'] == 'blue'
    assert 'Catalog categories by slot:' in search.description


def test_any_bottom_search_can_omit_category(products, inventory):
    shopper = agent(products, inventory, ScriptedModel([]))
    result = shopper.tools['search_products'].invoke(
        {'query': 'bottom', 'slot': 'bottom', 'filters': {}})
    assert 'error' not in result
    assert result['effective_filters'] == {'slot': 'bottom'}


def test_confirmed_invalid_category_is_not_silently_removed(products, inventory):
    shopper = agent(products, inventory, ScriptedModel([]))
    shopper.settings.filters['bottom'] = {'category': 'bottom'}
    result = shopper.tools['search_products'].invoke(
        {'query': 'jeans', 'slot': 'bottom', 'filters': {}})
    assert result['error'] == 'INVALID_CATEGORY'
    assert shopper.settings.filters['bottom']['category'] == 'bottom'


def test_uninspected_outfit_and_free_prose_blocked(products, inventory):
    model = ScriptedModel([call('propose_outfit', {'product_ids':['S01','S04'],'style_suggestion':'Buy these'}), AIMessage(content='Buy S01 and S04, valid outfit!')])
    result = agent(products, inventory, model).chat('Give me an outfit')
    assert result['status'] == 'no_validated_outfit' and not result['proposal']
    assert 'Buy' not in result['reply']


def test_unknown_tool_and_bad_schema_rejected(products, inventory):
    model = ScriptedModel([call('delete_catalog', {}), call('inspect_product', {}), AIMessage(content='Done')])
    result = agent(products, inventory, model).chat('Hello')
    assert all(row['result']['error'] == 'TOOL_REJECTED' for row in result['trace'])


def test_call_limit(products, inventory):
    model = ScriptedModel([call('inspect_product', {'product_id':'FAKE'}, str(i)) for i in range(10)])
    result = agent(products, inventory, model, max_calls=2).chat('Hello')
    assert result['status'] == 'call_limit' and result['calls'] == 2


def test_followup_requires_fresh_inspection(products, inventory):
    shopper = agent(products, inventory, scripted_pair())
    assert shopper.chat('Pair')['status'] == 'validated'
    shopper.model = ScriptedModel([call('propose_outfit', {'product_ids':['S01','S04'],'style_suggestion':'Retain'}), AIMessage(content='Stop')])
    result = shopper.chat('Keep both')
    assert not result['proposal'] and 'INSPECTION_REQUIRED' in result['trace'][0]['result']['errors']
    shopper.model = ScriptedModel([call('inspect_product', {'product_id':'S01'}, 'a'), call('inspect_product', {'product_id':'S04'}, 'b'), call('propose_outfit', {'product_ids':['S01','S04'],'style_suggestion':'Retain'}, 'c')])
    assert shopper.chat('Keep both')['status'] == 'validated'


def test_stock_change_after_inspection_is_rechecked(products, inventory):
    shopper = agent(products, inventory, scripted_pair())
    assert shopper.chat('Pair')['status'] == 'validated'
    shopper.tools['inspect_product'].invoke({'product_id':'S01'})
    shopper.inventory['products']['S01']['M'] = 0
    result = shopper.tools['propose_outfit'].invoke({'product_ids':['S01','S04'],'style_suggestion':'Reuse'})
    assert not result['valid'] and 'OUT_OF_STOCK' in result['errors']


def test_confirmed_constraints_cannot_be_overridden(products, inventory):
    settings = Settings('3000', {'top':'M','bottom':'M'}, filters={'top':{'fabric':'pure silk'}})
    model = ScriptedModel([call('search_products', {'query':'cotton top','slot':'top','filters':{'fabric':'pure cotton'}}), AIMessage(content='No')])
    result = ShoppingAgent(products, inventory, settings, HybridSearch(BM25(products)), model).chat('Cotton')
    assert result['trace'][0]['result']['error'] == 'TOOL_REJECTED'


def test_clarification_ends_turn(products, inventory):
    result = agent(products, inventory, ScriptedModel([call('ask_clarification', {'question':'Which fabric do you require?'})])).chat('Something nice')
    assert result['status'] == 'clarification' and not result['proposal']


def test_api_error_does_not_expose_exception_body(products, inventory):
    class Broken:
        def bind_tools(self, tools): return self
        def invoke(self, messages): raise RuntimeError('secret-provider-body')
    result = agent(products, inventory, Broken()).chat('Hello')
    assert result['status'] == 'api_error'
    assert 'secret-provider-body' not in str(result)


def test_compact_context_keeps_validation_and_reports_progress(products, inventory):
    shopper = agent(products, inventory, scripted_pair(), compact_context=True)
    progress = []
    shopper.on_progress = progress.append
    result = shopper.chat('Black cotton top and blue jeans')
    assert result['status'] == 'validated'
    assert result['proposal']['subtotal'] == '2000'
    assert result['trace'][0]['result']['previews']
    assert 'stock_for_confirmed_size' in result['trace'][0]['result']['previews'][0]
    inspected = result['trace'][2]['result']['product']
    assert 'description' in inspected and 'price' in inspected
    assert len(progress) == result['calls'] * 2


def test_filter_only_search_returns_matches_without_losing_requirements(products, inventory):
    shopper = agent(products, inventory, ScriptedModel([]), compact_context=True)
    tool = shopper.tools['search_products']
    tops = tool.invoke({'query':'','slot':'top','filters':{'color':'black'}})
    bottoms = tool.invoke({'query':'','slot':'bottom','filters':{'category':'jeans','color':'blue'}})
    assert tops['results'] and bottoms['results']
    for row in bottoms['results']:
        p = shopper.catalog[row['product_id']]
        assert p['slot']=='bottom' and p['category']=='jeans' and p['color']=='blue'
    assert tool.invoke({'query':'','slot':'top','filters':{'color':'nonexistent'}})['results']==[]


def test_over_budget_proposal_exposes_affordable_retrieved_alternative(products, inventory):
    from copy import deepcopy
    ps=deepcopy(products)
    # Add a more expensive version of the known eligible bottom.
    expensive=deepcopy(next(p for p in ps if p['product_id']=='S04'))
    expensive.update(product_id='EXPENSIVE',price='2900')
    ps.append(expensive)
    inv=deepcopy(inventory);inv['products']['EXPENSIVE']=deepcopy(inv['products']['S04'])
    shopper=agent(ps,inv,ScriptedModel([]),compact_context=True)
    for slot,query in [('top','black'),('bottom','jeans')]:
        shopper.tools['search_products'].invoke({'slot':slot,'query':query,'filters':{}})
    for pid in ['S01','EXPENSIVE']:
        shopper.tools['inspect_product'].invoke({'product_id':pid})
    r=shopper.tools['propose_outfit'].invoke({'product_ids':['S01','EXPENSIVE'],'style_suggestion':'Casual'})
    assert not r['valid'] and 'OVER_BUDGET' in r['errors']
    alt=r['affordable_alternative']['product_ids']
    assert 'EXPENSIVE' not in alt
    blocked=shopper.tools['ask_clarification'].invoke({'question':'Can you increase your budget?'})
    assert blocked['error']=='AFFORDABLE_ALTERNATIVE_AVAILABLE'
    for pid in alt:shopper.tools['inspect_product'].invoke({'product_id':pid})
    assert shopper.tools['propose_outfit'].invoke({'product_ids':alt,'style_suggestion':'Casual'})['valid']


def test_initial_budget_repair_finishes_without_seventh_model_call(products, inventory):
    from copy import deepcopy
    ps=deepcopy(products)
    expensive=deepcopy(next(p for p in ps if p['product_id']=='S04'))
    expensive.update(product_id='EXPENSIVE',price='2900')
    ps.append(expensive)
    inv=deepcopy(inventory); inv['products']['EXPENSIVE']=deepcopy(inv['products']['S04'])
    model=ScriptedModel([
        call('search_products',{'query':'black','slot':'top','filters':{'color':'black','fabric':'pure cotton'}},'1'),
        call('search_products',{'query':'jeans','slot':'bottom','filters':{'category':'jeans','color':'blue'}},'2'),
        call('inspect_product',{'product_id':'S01'},'3'),
        call('inspect_product',{'product_id':'EXPENSIVE'},'4'),
        call('propose_outfit',{'product_ids':['S01','EXPENSIVE'],'style_suggestion':'Casual'},'5'),
        call('inspect_product',{'product_id':'S04'},'6'),
    ])
    shopper=agent(ps,inv,model,compact_context=True)
    r=shopper.chat('Black pure cotton top with blue jeans')
    assert r['calls']==6 and r['status']=='validated'
    assert r['proposal']['subtotal']=='2000'
    assert r['trace'][-1]['origin']=='application_validator'
    assert r['trace'][-1]['event']=='deterministic_budget_repair'
