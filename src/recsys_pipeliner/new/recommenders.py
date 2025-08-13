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
        users_rated_items = single_users_rated_items[
            single_users_rated_items != item_idx
        ]
        users_ratings = single_users_ratings[single_users_rated_items != item_idx]

        # get the item similarities to item_idx
        item_similarities = (
            self._item_similarity_matrix[:, users_rated_items][item_idx]
            .toarray()
            .astype(np.float32)
            .round(6)
        )

        # sort by similarity (desc) and get top k
        top_k_mask = np.argsort(1 - item_similarities)[: self._k]
        top_k_user_ratings = users_ratings[top_k_mask]
        top_k_rated_item_similarities = item_similarities[top_k_mask]

        # weighted average rating
        return np.average(
            top_k_user_ratings, axis=0, weights=top_k_rated_item_similarities
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the rating for each user/item pair

        Args:
            X: np.ndarray of user/item pairs

        Returns:
            np.ndarray: predicted ratings
        """
        return np.apply_along_axis(self._predict, 1, X).astype(np.float32).round(6)

    def _recommend(self, X: np.ndarray) -> np.array:
        user_idx, item_idx = X[0], X[1]

        _, users_rated_items, _ = sp.sparse.find(self._user_item_matrix[user_idx, :])

        # exclude items already rated and the item itself
        candidates = np.setdiff1d(
            np.arange(self._item_similarity_matrix.shape[0]),
            np.concatenate([[item_idx], users_rated_items]),
        )
        candidate_similarity_matrix = self._item_similarity_matrix[item_idx, candidates]

        _, item_indices, item_similarities = sp.sparse.find(candidate_similarity_matrix)
        similar_items = candidates[item_indices]

        sorter = np.argsort(1 - item_similarities, kind="stable")
        recommendations = similar_items[sorter][: self._n]
        defaults = np.full(self._n - recommendations.shape[0], -1)
        return np.concatenate([recommendations, defaults])

    def recommend(self, X: np.ndarray) -> np.ndarray:
        """Recommend n items for each user/item pair

        Args:
            X: np.ndarray of user/item pairs

        Returns:
          np.ndarray
        """
        return np.apply_along_axis(self._recommend, 1, X)
