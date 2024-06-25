import pytest
from pipeliner.sagemaker.pipeline import PipelineFactory


def test_pipeline_factory():
    with pytest.raises(TypeError):
        PipelineFactory()
