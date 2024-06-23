from sagemaker import Session
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
import boto3

from pipeliner.exceptions import SagemakerSessionException


def create_pipeline_session(
    bucket: str, region="eu-west-1", local=False, local_code=False
) -> Session:
    try:
        if local:
            session = LocalPipelineSession()
            # session = LocalPipelineSession(sagemaker_config={'LOCAL': {'LOCAL_CODE': local_code}})
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
