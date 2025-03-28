from sagemaker import Session
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
import boto3

from pipeliner.exceptions import SagemakerSessionException


def create_pipeline_session(
    bucket: str | None = None, region="eu-west-1", local=False, local_code=False
) -> Session:
    """Creates a SageMaker pipeline session.

    Args:
        bucket (str | None): S3 bucket to use for pipeline artifacts
        region (str): AWS region to use for SageMaker resources
        local (bool): Whether to create a local pipeline session
        local_code (bool): Whether to use local code in local mode

    Returns:
        Session: A SageMaker pipeline session

    Raises:
        SagemakerSessionException: If there is an error creating the session
    """
    try:
        if local:
            session = LocalPipelineSession()
            session.config = {"local": {"local_code": local_code}}
        else:
            boto_session = boto3.Session(region_name=region)
            sagemaker_client = boto_session.client("sagemaker")

            session = PipelineSession(
                boto_session=boto_session,
                sagemaker_client=sagemaker_client,
                default_bucket=bucket,
            )
    except Exception as e:
        raise SagemakerSessionException from e

    return session
