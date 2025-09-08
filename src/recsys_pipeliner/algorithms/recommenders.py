import numpy as np
import scipy as sp
from recsys_pipeliner.algorithms.base import BaseRecommender
from recsys_pipeliner.recommendations.transformer import SimilarityTransformer


class RandomRecommender(BaseRecommender):
    """Random recommender.

    Args:
        n: Number of recommendations to generate.
    """

    _n: int

    def __init__(self, n: int = 5, random_seed: int | None = None):
        super().__init__(n)
        self._random_seed = random_seed

    def fit(self, X: np.ndarray, y=None):
        """Fits the recommender to the given data.

        Args:
            X np.ndarray:
                user/item matrix of shape (n_users, n_items)

        Returns:
            self
        """
        self._user_item_matrix = X

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Random predictions

        Args:
            X: np.ndarray of user/item pairs

        Returns:
            np.ndarray: random ratings
        """
        if self._random_seed is not None:
            rand = np.random.RandomState(self._random_seed)
            return rand.rand(X.shape[0]).astype(np.float32).round(6)
        else:
            return np.random.rand(X.shape[0]).astype(np.float32).round(6)

    def recommend(self, X: np.ndarray) -> np.ndarray:
        """Recommend n random items

        Args:
            X: np.ndarray

        Returns:
          np.ndarray
        """
        if self._random_seed is not None:
            rand = np.random.RandomState(self._random_seed)

            return rand.choice(
                np.arange(self._user_item_matrix.shape[1]), size=(X.shape[0], self._n)
            )
        else:
            return np.random.choice(
                np.arange(self._user_item_matrix.shape[1]), size=(X.shape[0], self._n)
            )


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
        # should this be:
        # top_k_rated_item_similarities = np.where(
        #     item_similarities[top_k_mask] > 0,
        #     item_similarities[top_k_mask],
        #     item_similarities[top_k_mask] + self._exp,
        # )

        # weighted average rating
        return (
            np.average(
                top_k_user_ratings, axis=0, weights=top_k_rated_item_similarities
            )
            .astype(np.float32)
            .round(6)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the rating for each user/item pair

        Args:
            X: np.ndarray of user/item pairs

        Returns:
            np.ndarray: predicted ratings
        """
        user_item_pairs = X[:,[0,1]].astype(np.int32)
        return np.apply_along_axis(self._predict, 1, user_item_pairs).astype(np.float32).round(6)

    # TODO: refactor this to reuse code
    def _recommend_similar_items(self, X: np.ndarray) -> np.ndarray:
        item_idx = X[0]
        _, item_indices, item_similarities = sp.sparse.find(
            self._item_similarity_matrix[item_idx, :]
        )
        sorter = np.argsort(1 - item_similarities, kind="stable")
        sorted_item_indices = item_indices[sorter]
        sorted_item_similarities = item_similarities[sorter]
        sorted_mask = (sorted_item_similarities > 0) * (sorted_item_indices != item_idx)
        recommendations = sorted_item_indices[sorted_mask][: self._n]
        defaults = np.full(self._n - recommendations.shape[0], -1)
        return np.concatenate([recommendations, defaults])

    def _recommend_personalised(self, X: np.ndarray) -> np.ndarray:
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
        """Recommend n items

        If X is a 2D array of user/item ids, the recommender will recommend n personalised similar items for each user.
        If X is a 1D array of item ids, the recommender will recommend n similar items for each item.

        Args:
            X: np.ndarray

        Returns:
          np.ndarray
        """
        if X.ndim == 1:
            return np.apply_along_axis(
                self._recommend_similar_items, 1, X[np.newaxis, :]
            )
        elif X.ndim == 2 and X.shape[1] == 2:
            return np.apply_along_axis(self._recommend_personalised, 1, X)
        else:
            raise ValueError("X must be a 1D or 2D array")
