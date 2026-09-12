import json
from pathlib import Path

import pytest

from fashion_assistant.catalog import FIELDS, prepare_catalog


@pytest.fixture
def root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def products(root):
    return prepare_catalog(root/'data/sample/catalog.csv', {key: key for key in FIELDS})[0]


@pytest.fixture
def inventory(root):
    return json.loads((root/'data/sample/inventory.json').read_text())
