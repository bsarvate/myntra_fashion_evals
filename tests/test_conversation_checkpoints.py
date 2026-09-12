import json
import pytest
from fashion_assistant.evaluation import run_conversations


def cases():
    return [{'case_id':'C1','group_id':'G1','split':'dev','evidence':'synthetic',
             'turns':[{'request':'one','expected':{}},{'request':'two','expected':{}}]}]


def test_interrupt_preserves_completed_turn_and_excludes_partial_dialogue(tmp_path):
    path=tmp_path/'report.json'
    class Agent:
        calls=0
        def chat(self,*args,**kwargs):
            self.calls+=1
            if self.calls==2:
                assert len(json.loads(path.read_text())['rows'])==1
                raise KeyboardInterrupt
            return {'status':'validated','calls':1,'proposal':{}}
    r=run_conversations(cases(),lambda c:Agent(),checkpoint_path=path,progress=True)
    assert r['run_status']=='interrupted'
    assert r['turns']==1 and r['planned_turns']==2
    assert r['dialogue_success_rate'] is None and r['incomplete_dialogues']==1
    assert json.loads(path.read_text())==r
    with pytest.raises(ValueError,match='already exists'):
        run_conversations(cases(),lambda c:Agent(),checkpoint_path=path)


def test_stop_after_api_error_preserves_failed_turn(tmp_path):
    class Agent:
        def chat(self,*args,**kwargs):
            return {'status':'api_error','calls':1,'proposal':{}}
    r=run_conversations(cases(),lambda c:Agent(),checkpoint_path=tmp_path/'r.json',stop_on_api_error=True)
    assert r['run_status']=='stopped_on_error'
    assert r['turns']==1 and r['api_error_turns']==1
    assert r['dialogues']==0
