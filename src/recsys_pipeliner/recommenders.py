from typing import Any, Sequence
import numpy as np
import scipy as sp
from recsys_pipeliner.base import BaseRecommender
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

    def _predict_for_user_item_pair(self, X: np.ndarray) -> np.float32:
        u, i = X[0], X[1]
        _, users_rated_items, users_ratings = sp.sparse.find(
            self._user_item_matrix[u, :]
        )

        # get the similarities to item_id
        item_similarities = (
            self._item_similarity_matrix[:, users_rated_items][i]
            .toarray()
            .astype(np.float32)
            .round(6)
        )

        # sort by similarity (desc) and get top k
        top_k_mask = np.argsort(1 - item_similarities)[1 : self._k + 1]
        top_k_user_ratings = users_ratings[top_k_mask]
        top_k_rated_item_similarities = item_similarities[top_k_mask]

        # weighted average rating
        predicted_score = (
            np.average(
                top_k_user_ratings, axis=0, weights=top_k_rated_item_similarities
            )
            .astype(np.float32)
            .round(6)
        )
        return predicted_score

    def _predict_for_item(self, i) -> np.float32:
        # get the similarities to item_id
        item_similarities = self._item_similarity_matrix[i].mean(axis=0)
        item_similarities2 = self._item_similarity_matrix[i].mean(axis=1)

        print("item_similarities", item_similarities.shape)
        print("item_similarities2", item_similarities2.shape)

        return np.float32(1.0)

        # # sort by similarity (desc) and get top k
        # top_k_mask = np.argsort(1 - item_similarities)[1 : self._k + 1]
        # top_k_ratings = self._user_item_matrix[:, top_k_mask]
        # top_k_item_similarities = item_similarities[top_k_mask]

        # # weighted average rating
        # predicted_score = (
        #     np.average(top_k_item_similarities, axis=0).astype(np.float32).round(6)
        # )
        # return predicted_score

    def predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Predicts the ratings for the given data.

        Args:
            X: np.ndarray of shape (n,) or (n, 2)

        Returns:
            np.ndarray: predicted ratings of shape (n,)
        """
        if X.ndim == 1:
            return np.vectorize(self._predict_for_item)(X)
        elif X.ndim == 2:
            return np.apply_along_axis(self._predict_for_user_item_pair, 1, X)
        else:
            raise ValueError("X should be a 1D or 2D array")
