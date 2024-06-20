from abc import abstractmethod
from sagemaker import Session
from pydantic import BaseModel
from sagemaker.workflow.pipeline import Pipeline

class PipelineFactory(BaseModel):  
    """Base class for all pipeline factories."""  
    @abstractmethod  
    def create(  
        self,  
        role: str,  
        name: str,  
        session: Session,  
    ) -> Pipeline:  
        raise NotImplementedError