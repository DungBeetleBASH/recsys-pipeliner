import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
)
from collections.abc import Sequence


class ItemRecommender(BaseEstimator):
    """Item-based collaborative filtering recommender."""

    n: int
    threshold: float
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5, threshold=0.1):
        self.n = n
        self.threshold = threshold

    def fit(self, X: pd.DataFrame | Sequence[pd.DataFrame], y=None):
        """Fits the recommender to the given data.

        Args:
          X (pd.DataFrame | Sequence[pd.DataFrame]):
            Single DataFrame with similarity matrix
            or Sequence of (similarity matrix, user/item matrix)
        """
        if isinstance(X, pd.DataFrame):
            self.similarity_matrix = X
        elif isinstance(X, Sequence) and not isinstance(X, str):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be a DataFrame or a sequence of DataFrames")

        return self

    def _recommend(self, item: str | Sequence[str]):
        if isinstance(item, str):
            item_id = item
            exclusions = [item]
        elif isinstance(item, Sequence):
            user_id, item_id = item[0], item[1]

            single_user_matrix = self.user_item_matrix.loc[user_id]
            user_rated_items = single_user_matrix[single_user_matrix > 0]

            exclusions = [item_id] + user_rated_items.index.to_list()
        else:
            return np.array([])

        item_recommendations = (
            self.similarity_matrix[self.similarity_matrix[item_id] > self.threshold][
                item_id
            ]
            .drop(exclusions, errors="ignore")
            .sort_values(ascending=False)
        )
        return np.array(item_recommendations.head(self.n).index)

    def predict(self, X: Sequence[str] | Sequence[Sequence[str]]):
        """Predicts n recommendations

        Args:
          X (Sequence): List of item_ids or list of (user_id, item_id) pairs

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._recommend(item) for item in X])

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba not implemented yet")
