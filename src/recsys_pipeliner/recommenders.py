from typing import Any, Sequence, Self
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

    def fit(self, X: sp.sparse.sparray, y=None) -> Self:
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
    
    def predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Predicts the ratings for the given data.

        Args:
            X: np.ndarray of shape (n,) or (n, 2)

        Returns:
            np.ndarray: predicted ratings of shape (n,)
        """
        if X.ndim == 1:
            user_ids, item_ids = None, X
        elif X.ndim == 2:
            user_ids, item_ids = X[:, 0], X[:, 1]
            _, users_rated_items, users_ratings = sp.sparse.find(
                self._user_item_matrix[user_ids, :]
            )
            item_similarities = (
                self._item_similarity_matrix[:, users_rated_items][item_ids]
                .toarray()
                .astype(np.float32)
                .round(6)
            )
            # TODO: Implement the prediction logic
        else:
            raise ValueError("X should be a 1D or 2D array")
        
        pass