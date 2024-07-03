from abc import abstractmethod
from sagemaker import Session
from pydantic import BaseModel
from sagemaker.workflow.pipeline import Pipeline


class PipelineFactory(BaseModel):
    """Base class for all pipeline factories."""

    local: bool = False

    @abstractmethod
    def create(
        self,
        role: str,
        name: str,
        session: Session,
    ) -> Pipeline:
        """Abstract create method"""
