import pytest

from fashion_assistant.agent import nebius_model
from fashion_assistant.evaluation import load_cases, run_conversations


@pytest.mark.parametrize('model', ['Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B'])
def test_retrieval_model_rejected_before_provider_creation(monkeypatch, model):
    monkeypatch.delenv('NEBIUS_API_KEY', raising=False)
    with pytest.raises(ValueError, match='chat model'):
        nebius_model(model)


def test_api_failure_report_is_identified(root):
    class FailedAgent:
        def chat(self, *args, **kwargs):
            return {'status': 'api_error', 'calls': 1, 'proposal': {}}

    cases = load_cases(root / 'evals/fixtures/conversations.jsonl')
    report = run_conversations(cases, lambda case: FailedAgent())
    assert report['api_error_turns'] == report['turns']
    assert report['execution_error_turns'] == 0
    assert report['run_status'] == 'all_turns_failed_to_execute'
    assert report['turn_success_rate'] == 0
