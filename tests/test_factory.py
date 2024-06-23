import pytest
from pipeliner.factory import PipelineFactory


def test_pipeline_factory():
    with pytest.raises(TypeError):
        PipelineFactory()
