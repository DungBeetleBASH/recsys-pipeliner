import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from collections.abc import Sequence


class ItemBasedRecommender(BaseEstimator):
    """Item-based collaborative filtering recommender."""

    n: int
    threshold: float
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5, threshold=0.1):
        self.n = n
        self.threshold = threshold

    def fit(self, X: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame], y=None):
        """Fits the recommender to the given data.

        Args:
          X (pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]):
            Single DataFrame with similarity matrix
            or tuple of (similarity matrix, user/item matrix)
        """
        if isinstance(X, pd.DataFrame):
            self.similarity_matrix = X
        elif isinstance(X, tuple):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be DataFrame or (DataFrame, DataFrame)")

        return self

    def _get_exclusions(self, item_id: str, user_id: str | None) -> list[str]:
        if user_id is None:
            return [item_id]
        single_user_matrix = self.user_item_matrix.loc[user_id]
        user_rated_items = single_user_matrix[single_user_matrix > 0]
        return [item_id] + user_rated_items.index.to_list()

    def _get_recommendations(self, item: str | tuple[str, str]) -> np.array:
        if isinstance(item, str):
            item_id, user_id = item, None
        elif isinstance(item, tuple):
            item_id, user_id = item[0], item[1]
        else:
            raise ValueError("Input items should be str or (str, str)")

        exclusions = self._get_exclusions(item_id, user_id)

        item_recommendations = (
            self.similarity_matrix[self.similarity_matrix[item_id] > self.threshold][
                item_id
            ]
            .drop(exclusions, errors="ignore")
            .sort_values(ascending=False)
        )
        return np.array(item_recommendations.head(self.n).index)

    def predict(self, X: Sequence[str] | Sequence[tuple[str, str]]) -> np.array:
        """Predicts n item recommendations for each item_id provided
        If tuples of (user_id, item_id) are provided, items previously
        rated by the user will be excluded from the recommendations.

        Args:
          X (Sequence): List of item_id or (user_id, item_id)

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._get_recommendations(item) for item in X])

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")
