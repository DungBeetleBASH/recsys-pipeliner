from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    def __init__(self, n=5, k=5):
        self.n = n  # number of recommendations
        self.k = k  # number of neighbours

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

    @abstractmethod
    def recommend(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass
