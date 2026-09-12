import csv

import pytest

from fashion_assistant.catalog import prepare_catalog, positive_amount, matches
from fashion_assistant.validation import Settings, validate_outfit


@pytest.mark.parametrize('value', [True, 'nan', 'Infinity', '-1', '0', 'unknown'])
def test_invalid_money(value):
    with pytest.raises(ValueError):
        positive_amount(value)


def test_csv_rejections_are_explicit(tmp_path):
    path = tmp_path/'input.csv'
    path.write_text('id,title,cost,slot\nA,Top,10,top\nA,Duplicate,20,top\nB,Bad price,nan,top\nC,No slot,10,\nD,Broken,10,top,extra\n')
    products, report = prepare_catalog(path, {'product_id': 'id', 'name': 'title', 'price': 'cost', 'slot': 'slot'})
    assert len(products) == 1 and len(report['rejected']) == 4
    assert products[0]['fabric'] == '' and not products[0]['image_verified']


def test_invalid_image_path_does_not_become_verified(tmp_path):
    path = tmp_path/'catalog.csv'
    path.write_text('id,name,price,slot,image\nX,Top,5,top,../outside.jpg\n')
    products, report = prepare_catalog(path, {'product_id':'id','name':'name','price':'price','slot':'slot','image_path':'image'}, image_root=tmp_path)
    assert not products[0]['image_verified']
    assert 'relative' in report['image_issues'][0]['reason']


def test_filter_is_exact(products):
    assert matches(products[0], {'fabric': 'PURE COTTON'})
    assert not matches(products[0], {'fabric': 'cotton'})
    with pytest.raises(ValueError):
        matches(products[0], {'invented_filter': 'x'})


def test_valid_pair_and_budget_boundary(products, inventory):
    result = validate_outfit(['S01', 'S04'], products, inventory, Settings('2000', {'top':'M','bottom':'M'}), {'S01','S04'})
    assert result['valid'] and result['subtotal'] == '2000'
    result = validate_outfit(['S01', 'S04'], products, inventory, Settings('1999.99', {'top':'M','bottom':'M'}), {'S01','S04'})
    assert result['errors'] == ['OVER_BUDGET']


@pytest.mark.parametrize('ids,inspected,error', [
    (['S01','S04'], set(), 'INSPECTION_REQUIRED'),
    (['S01','FAKE'], {'S01','FAKE'}, 'UNKNOWN_ID'),
    (['S01','S01'], {'S01'}, 'DUPLICATE_ID'),
    (['S01'], {'S01'}, 'OUTFIT_STRUCTURE')])
def test_invalid_selections(products, inventory, ids, inspected, error):
    result = validate_outfit(ids, products, inventory, Settings('3000', {'top':'M','bottom':'M'}), inspected)
    assert not result['valid'] and error in result['errors']


@pytest.mark.parametrize('quantity,error', [(0,'OUT_OF_STOCK'), (None,'STOCK_UNKNOWN'), (True,'STOCK_UNKNOWN'), (-1,'STOCK_UNKNOWN')])
def test_unknown_and_zero_stock_distinct(products, inventory, quantity, error):
    inventory['products']['S01']['M'] = quantity
    result = validate_outfit(['S01','S04'], products, inventory, Settings('3000', {'top':'M','bottom':'M'}), {'S01','S04'})
    assert not result['valid'] and error in result['errors']


def test_set_and_constraint_checks(products, inventory):
    assert validate_outfit(['S07'], products, inventory, Settings('1600', {'set':'M'}), {'S07'})['valid']
    settings = Settings('3000', {'top':'M','bottom':'M'}, filters={'top': {'fabric':'pure silk'}})
    result = validate_outfit(['S01','S04'], products, inventory, settings, {'S01','S04'})
    assert 'CONSTRAINT_VIOLATION' in result['errors']


def test_unit_mismatch(products, inventory):
    result = validate_outfit(['S01','S04'], products, inventory, Settings('3000', {'top':'M','bottom':'M'}, 'USD'), {'S01','S04'})
    assert not result['valid'] and 'PRICE_UNIT_MISMATCH' in result['errors']
