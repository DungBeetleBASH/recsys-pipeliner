import sagemaker
from sagemaker import ScriptProcessor
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from pipeliner.sagemaker.pipeline import PipelineFactory


class RecommenderPipeline(PipelineFactory):
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

        cache_config = CacheConfig(
            enable_caching=True,
            expire_after="P30d",  # 30 days
        )

        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type,
            instance_count=1,
            role=role,
            sagemaker_session=session,
        )

        user_item_matrix_step = ProcessingStep(
            name="user_item_matrix_transformer",
            step_args=processor.run(
                code="pipelines/code/user_item_matrix_transformer.py",
            ),
        )

        item_similarity_matrix_step = ProcessingStep(
            name="similarity_matrix_transformer",
            step_args=processor.run(
                code="pipelines/code/user_similarity_matrix_transformer.py",
            ),
            job_arguments=["--kind", "item"],
        )

        sklearn_estimator = SKLearn(
            entry_point="pipelines/code/item_recommender_train.py",
            role=role,
            image_uri=image_uri,
            instance_type=instance_type,
            sagemaker_session=session,
            base_job_name="training_job",
            # hyperparameters=hyperparameters,
            enable_sagemaker_metrics=True,
        )

        training_step = TrainingStep(
            name="Train", estimator=sklearn_estimator, cache_config=cache_config
        )

        return Pipeline(
            name=name,
            steps=[user_item_matrix_step, item_similarity_matrix_step, training_step],
            sagemaker_session=session,
            parameters=[instance_type],
        )
