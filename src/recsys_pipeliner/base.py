from abc import abstractmethod
from typing import Any, Sequence, Self
from sklearn.base import BaseEstimator



class BaseRecommender(BaseEstimator):
    """Base class for all recommenders.

    Args:
        n: Number of recommendations to generate.
    """
    _n: int

    def __init__(self, n: int = 5):
        self._n = n

    @abstractmethod
    def fit(self, X: Sequence[Any], y: Sequence[Any] | None) -> Self:
        """Fit the recommender to the data."""

    @abstractmethod
    def predict(self, X: Sequence[Any], y: Sequence[Any] | None) -> Sequence[Any]:
        """Predicts item ratings."""

    @abstractmethod
    def recommend(self, X: Sequence[Any]) -> Sequence[Any]:
        """Recommends items."""
