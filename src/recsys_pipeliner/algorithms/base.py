from abc import abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
from typing import Self


class BaseRecommender(BaseEstimator):
    """Base class for all recommenders.

    Args:
        n: Number of recommendations to generate.
    """

    _n: int

    def __init__(self, n: int = 5):
        self._n = n

    @abstractmethod
    def fit(self, X, y=None) -> Self:
        """Fit the recommender to the data."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts item ratings."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def recommend(self, X: np.ndarray) -> np.ndarray:
        """Recommends items."""
        raise NotImplementedError("Subclasses must implement this method")
