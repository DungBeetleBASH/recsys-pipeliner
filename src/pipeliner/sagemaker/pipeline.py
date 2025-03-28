from abc import abstractmethod
from sagemaker import Session
from pydantic import BaseModel
from sagemaker.workflow.pipeline import Pipeline


class PipelineFactory(BaseModel):
    """Base class for all pipeline factories.

    Attributes:
        local (bool): Whether to run the pipeline locally or in SageMaker
    """

    local: bool = False

    @abstractmethod
    def create(
        self,
        role: str,
        name: str,
        session: Session,
    ) -> Pipeline:
        """Creates a SageMaker pipeline.

        Args:
            role (str): IAM role to use for the pipeline
            name (str): Name of the pipeline
            session (Session): SageMaker session to use

        Returns:
            Pipeline: The created SageMaker pipeline

        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
