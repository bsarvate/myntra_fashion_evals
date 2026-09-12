import pytest
from packaging.requirements import Requirement

from fashion_assistant.embeddings import checked_revision, CLIP_MODEL_REVISION


@pytest.mark.parametrize('revision',[True,False,17,''])
def test_boolean_or_empty_revision_rejected(revision):
    with pytest.raises(ValueError,match='commit/tag'):
        checked_revision('model',revision,'model','commit')


def test_default_revision_is_pinned_without_changing_custom_models():
    assert checked_revision('openai/clip-vit-base-patch32',None,'openai/clip-vit-base-patch32',CLIP_MODEL_REVISION)==CLIP_MODEL_REVISION
    assert checked_revision('custom/model',None,'openai/clip-vit-base-patch32',CLIP_MODEL_REVISION) is None


def test_platform_dependencies_have_installable_torch_branch(root):
    import tomllib
    requirements=[Requirement(value) for value in tomllib.loads((root/'pyproject.toml').read_text())['project']['optional-dependencies']['search']]
    for system,machine,version in [('Darwin','x86_64','2.2.2'),('Darwin','arm64','2.6.0'),('Linux','x86_64','2.6.0')]:
        environment={'platform_system':system,'platform_machine':machine}
        torch=[r for r in requirements if r.name=='torch' and (r.marker is None or r.marker.evaluate(environment))]
        assert len(torch)==1 and version in torch[0].specifier
