import pytest
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from pipeliner.exceptions import SagemakerSessionException
from pipeliner.sagemaker.session import create_pipeline_session


def test_create_pipeline_session(fx_boto3_session):
    session = create_pipeline_session("test-bucket")

    assert session.__class__ == PipelineSession
    fx_boto3_session.return_value.client.assert_called()


@pytest.mark.parametrize(
    "local, local_code, expected",
    [
        (True, False, LocalPipelineSession),
        (True, True, LocalPipelineSession),
    ],
)
def test_create_pipeline_session_local(fx_boto3_session, local, local_code, expected):
    local_args = {} if local is None else {"local": local}
    local_code_args = {} if local_code is None else {"local_code": local_code}
    args = {**local_args, **local_code_args}

    session = create_pipeline_session(**args)

    assert session.__class__ == expected


def test_create_pipeline_session_error(fx_boto3_session):
    fx_boto3_session.return_value.client.side_effect = Exception("error")

    with pytest.raises(SagemakerSessionException):
        create_pipeline_session("test-bucket")
