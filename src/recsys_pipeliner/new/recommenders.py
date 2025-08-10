from typing import Any, Sequence
import numpy as np
import scipy as sp
from recsys_pipeliner.new.base import BaseRecommender
from recsys_pipeliner.recommendations.transformer import SimilarityTransformer


class ItemBasedCFRecommender(BaseRecommender):
    """Item-based collaborative filtering recommender.

    Args:
        n: Number of recommendations to generate.
        k: Number of similar items to consider.
        exp: Regularization parameter.
    """

    _n: int
    _k: int
    _exp: float

    def __init__(self, n: int = 5, k: int = 5, exp: float = 1e-6):
        super().__init__(n)
        self._k = k
        self._exp = exp

    def fit(self, X: sp.sparse.sparray, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparse.sparray:
                user/item matrix of shape (n_users, n_items)

        Returns:
            self

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparse.sparray):
            self._user_item_matrix = X
            self._item_similarity_matrix = SimilarityTransformer().transform(
                self._user_item_matrix.T
            )
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _predict(self, X: np.ndarray) -> np.float32:
        user_idx, item_idx = X[0], X[1]
        
        _, single_users_rated_items, single_users_ratings = sp.sparse.find(
            self._user_item_matrix[user_idx, :]
        )

        # exclude item if already rated
        users_rated_items = single_users_rated_items[single_users_rated_items != item_idx]
        users_ratings = single_users_ratings[single_users_rated_items != item_idx]

        # get the item similarities to item_idx
        item_similarities = (
            self._item_similarity_matrix[:, users_rated_items][item_idx]
            .toarray()
            .astype(np.float32)
            .round(6)
        )

        # sort by similarity (desc) and get top k
        top_k_mask = np.argsort(1 - item_similarities)[:self._k]
        top_k_user_ratings = users_ratings[top_k_mask]
        top_k_rated_item_similarities = item_similarities[top_k_mask]

        # weighted average rating
        return np.average(
            top_k_user_ratings, axis=0, weights=top_k_rated_item_similarities
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the ratings for the given data.

        Args:
            X: np.ndarray of shape (n, 2)

        Returns:
            np.ndarray: predicted ratings of shape (n,)
        """
        return (
            np.apply_along_axis(self._predict, 1, X)
            .astype(np.float32)
            .round(6)
        )