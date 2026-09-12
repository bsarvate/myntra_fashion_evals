from pathlib import Path
from unittest.mock import patch
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
APP=Path(__file__).resolve().parents[1]/'streamlit_app.py'
pytestmark = pytest.mark.skipif(not (APP.parent/'data/processed/myntra1000/catalog.json').exists(), reason='Streamlit integration tests require the locally prepared dataset')


def test_app_loads_and_browses_without_provider_calls():
    with patch('fashion_assistant.agent.nebius_model') as provider:
        app=AppTest.from_file(str(APP),default_timeout=60).run()
        assert not app.exception
        assert app.chat_input[0].disabled
        next(w for w in app.text_input if w.label=='Search products').set_value('blue jeans').run()
        assert not app.exception
        provider.assert_not_called()


def test_new_conversation_and_settings_change(monkeypatch):
    class FakeAgent:
        def __init__(self,*args,**kwargs): pass
        def chat(self,prompt):
            return {'status':'clarification','reply':'Which fabric would you prefer?',
                    'proposal':{},'calls':1,'trace':[],'request':prompt}
    with patch('fashion_assistant.agent.nebius_model',return_value=object()),patch('fashion_assistant.agent.ShoppingAgent',FakeAgent),patch('fashion_assistant.evaluation.save_run'):
        app=AppTest.from_file(str(APP),default_timeout=60).run()
        app.sidebar.button[0].click().run()
        assert not app.exception and not app.chat_input[0].disabled
        app.chat_input[0].set_value('Suggest an outfit').run()
        assert not app.exception
        assert len(app.session_state['messages'])==1
        app.sidebar.number_input[0].set_value(3000).run()
        assert app.chat_input[0].disabled
