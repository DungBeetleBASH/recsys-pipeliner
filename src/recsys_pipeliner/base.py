from abc import abstractmethod
from typing import Any, Sequence
from sklearn.base import BaseEstimator
import numpy as np


class BaseRecommender(BaseEstimator):
    """Base class for all recommenders.

    Args:
        n: Number of recommendations to generate.
    """

    _n: int

    def __init__(self, n: int = 5):
        self._n = n

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None):
        """Fit the recommender to the data."""

    @abstractmethod
    def predict(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """Predicts item ratings."""

    @abstractmethod
    def recommend(self, X: np.ndarray) -> Sequence[Any]:
        """Recommends items."""
