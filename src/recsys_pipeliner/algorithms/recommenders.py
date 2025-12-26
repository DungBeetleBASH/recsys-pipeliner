import numpy as np
import scipy as sp
from recsys_pipeliner.algorithms.base import BaseRecommender
from recsys_pipeliner.recommendations.transformer import SimilarityTransformer


class RandomRecommender(BaseRecommender):
    """Random recommender.

    Args:
        n: Number of recommendations to generate.
        random_seed: Random seed for reproducibility.
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

        # get the user's rated items
        items = np.setdiff1d(np.arange(self._user_item_matrix.shape[1]), item_idx)
        _, rated_item_indices, user_ratings = sp.sparse.find(
            self._user_item_matrix[user_idx, items]
        )
        rated_items = items[rated_item_indices]

        # get the user's rated items' similarities to item_idx
        item_similarities = (
            self._item_similarity_matrix[item_idx, rated_items]
            .toarray()
            .astype(np.float32)
            .round(6)
        )

        # sort by similarity (desc) and get top k
        top_k_mask = np.argsort(1 - item_similarities, kind="stable")[: self._k]
        top_k_user_ratings = user_ratings[top_k_mask]
        top_k_item_similarities = item_similarities[top_k_mask]

        # weighted average rating
        return (
            np.average(
                top_k_user_ratings, axis=0, weights=top_k_item_similarities
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
        return np.apply_along_axis(self._predict, 1, X).astype(np.float32).round(6)


    def _recommend(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 2:
            user_idx, item_idx = X[0], X[1]
        else:
            user_idx, item_idx = None, X[0]

        if user_idx is None:
            items = np.arange(self._item_similarity_matrix.shape[0])
            candidates = items[items != item_idx]
        else:
            _, target_user_rated_items, _ = sp.sparse.find(self._user_item_matrix[user_idx, :])
            candidates = np.setdiff1d(
                np.arange(self._item_similarity_matrix.shape[0]),
                np.concatenate([[item_idx], target_user_rated_items]),
            )

        candidate_similarity_matrix = self._item_similarity_matrix[item_idx, candidates]
        _, item_indices, item_similarities = sp.sparse.find(candidate_similarity_matrix)
        similar_items = candidates[item_indices]

        top_n_mask = np.argsort(1 - item_similarities, kind="stable")[: self._n]
        recommendations = similar_items[top_n_mask]

        if recommendations.shape[0] < self._n:
            defaults = np.full(self._n - recommendations.shape[0], -1)
            return np.concatenate([recommendations, defaults])
        else:
            return recommendations


    def recommend(self, X: np.ndarray) -> np.ndarray:
        """Recommend n items

        If X is a 2D array of user/item ids, recommend n personalised items for each user/item pair.
        If X is a 2D array of item ids, recommend n items for each item.

        Args:
            X: np.ndarray

        Returns:
          np.ndarray
        """
        if X.ndim == 2 and X.shape[1] in (1, 2):
            return np.apply_along_axis(self._recommend, 1, X)
        else:
            raise ValueError("X must be a 2D array with 1 or 2 columns")


class ItemBasedHybridRecommender(ItemBasedCFRecommender):
    """Item-based hybrid recommender.

    Args:
        n: Number of recommendations to generate.
        k: Number of similar items to consider.
        exp: Regularization parameter.
    """

    _n: int
    _k: int
    _exp: float

    def __init__(self, n: int = 5, k: int = 5, exp: float = 1e-6):
        super().__init__(n, k, exp)

    def fit(self, X: tuple[sp.sparse.sparray, sp.sparse.sparray], y=None):
        """Fits the recommender to the given data.

        Args:
            X: tuple[sp.sparse.sparray, sp.sparse.sparray]

        Returns:
            self

        Raises:
            ValueError: If input is not a tuple[sp.sparse.sparray, scipy.sparse.sparray]
        """
        if (
            isinstance(X, tuple) and
            len(X) == 2 and
            isinstance(X[0], sp.sparse.sparray) and
            isinstance(X[1], sp.sparse.sparray)
        ):
            self._item_similarity_matrix = SimilarityTransformer().transform(
                X[0]
            )
            self._user_item_matrix = X[1]
        else:
            raise ValueError("Input should be tuple[sp.sparse.sparray, scipy.sparse.sparray]")

        return self