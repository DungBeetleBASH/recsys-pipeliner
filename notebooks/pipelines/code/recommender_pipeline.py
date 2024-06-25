import sagemaker
from sagemaker import ScriptProcessor
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

from pipeliner.factory import SagemakerPipelineFactory


class RecommenderPipeline(SagemakerPipelineFactory):
    local: bool

    def create(
        self,
        role: str,
        name: str,
        session: sagemaker.Session,
    ) -> Pipeline:
        self.local = isinstance(session, LocalPipelineSession)

        instance_type = ParameterString(
            name="InstanceType",
            default_value="local" if self.local else "ml.m5.large",
        )

        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=session.boto_region_name,
            version="1.2-1",
        )

        # Create a ScriptProcessor and add code / run parameters
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type,
            instance_count=1,
            role=role,
            sagemaker_session=session,
        )

        processing_step = ProcessingStep(
            name="processing-example",
            step_args=processor.run(
                code="pipelines/sources/example_pipeline/evaluate.py",
            ),
        )

        return Pipeline(
            name=name,
            steps=[processing_step],
            sagemaker_session=session,
            parameters=[instance_type],
        )
