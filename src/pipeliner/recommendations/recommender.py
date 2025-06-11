import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from pipeliner.recommendations.transformer import (
    SimilarityTransformer,
)


class UserBasedRecommender(BaseEstimator):
    """User-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each user
        k (int): Number of similar users to consider for recommendations
    """

    n: int
    k: int

    def __init__(self, n=5, k=5):
        self.n = n
        self.k = k
        self._user_transformer = SimilarityTransformer()

    def fit(self, X: sp.sparse.sparray, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparse.sparray:
                user/item matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparse.sparray):
            self._user_item_matrix = X
            self._user_similarity_matrix = self._user_transformer.transform(X)
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _get_similar_users(self, id: int) -> np.array:
        matrix = self._user_similarity_matrix[[id]]
        user_mask = matrix > 0
        user_mask[[0], [id]] = False
        user_sorter = np.argsort(1 - matrix.toarray()[0], kind="stable")
        sorted_mask = user_mask.toarray()[0][user_sorter]
        similar_users = user_sorter[sorted_mask][: self.k]

        return similar_users

    def _get_exclusions(self, id: int) -> np.array:
        single_user_ratings = self._user_item_matrix[[id]]
        rated = (single_user_ratings > 0).nonzero()[1]
        return rated

    def _get_recommendations(self, id: int) -> np.array:
        excluded_items = self._get_exclusions(id)
        similar_users = self._get_similar_users(id)

        matrix = self._user_item_matrix[similar_users]

        any_ratings = np.nonzero(matrix.sum(axis=0))[0]
        items_to_use = np.setdiff1d(any_ratings, excluded_items)

        filtered_matrix = matrix[:, items_to_use]

        mean_ratings = filtered_matrix.toarray().T.mean(axis=1)
        item_sorter = np.argsort(1 - mean_ratings, kind="stable")

        return items_to_use[item_sorter][: self.n]

    def recommend(self, X) -> list[np.array]:
        """Predicts n recommendations for each id provided

        Args:
          X (Sequence): List of id

        Returns:
          list of np.array
        """
        return [self._get_recommendations(id) for id in X]


class SimilarityRecommender(BaseEstimator):
    """Similarity recommender.

    Args:
        n (int): Number of recommendations to generate.
    """

    n: int
    similarity_matrix: sp.sparse.sparray

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparse.sparray:
                similarity matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparse.sparray):
            self.similarity_matrix = X
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _get_recommendations(self, id) -> np.array:
        item_similarity = self.similarity_matrix[[id], :].toarray()
        mask = (item_similarity > 0) * (np.arange(item_similarity.size) != id)
        sorter = np.argsort(1 - item_similarity, kind="stable")
        sorted_mask = mask[0, sorter]
        return sorter[sorted_mask][: self.n]

    def recommend(self, X) -> list[np.array]:
        """Predicts n recommendations for each id provided

        Args:
          X (Sequence): List of id

        Returns:
          list of np.array
        """
        return [self._get_recommendations(id) for id in X]

    def predict_proba(self, X):
        return self.similarity_matrix[X]


class ItemBasedRecommender(BaseEstimator):
    """Item-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate
        k (int): Number of similar items to consider for recommendations
    """

    n: int
    k: int

    def __init__(self, n=5, k=5, debias=False):
        self.n = n
        self.k = k
        self.debias = debias
        self._item_transformer = SimilarityTransformer()

    def fit(self, X: sp.sparse.sparray, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparse.sparray:
                user/item matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparse.sparray):
            self._user_item_matrix = X
            self._item_similarity_matrix = self._item_transformer.transform(X.T)
        else:
            raise ValueError("Input should be scipy.sparse.sparray")
        
        if self.debias:
            bias = self._user_item_matrix.mean(axis=0)[np.newaxis, :]
            self._user_item_matrix -= bias

        return self


# class Recommender:
#     def __init__(
#         self,
#         n,
#         user_item_matrix,
#         k=40,
#         debias=False,
#         eps=1e-9
#     ):
#         self.n = n
#         self.user_item_matrix = user_item_matrix
#         self.item_similarity_matrix = SimilarityTransformer().transform(user_item_matrix.T)
#         self.debias = debias
#         self.k = k
#         self.eps = eps
#         self.predictions = self._predict_all()

#     def fit():
#         pass
    
#     def _predict_all(self):
#         pred = np.empty_like(self.user_item_matrix)
        
#         # Computes the new interaction matrix if needed.
#         user_item_matrix = self.user_item_matrix
#         if self.debias:
#             item_bias = self.user_item_matrix.mean(axis=0)[np.newaxis, :]
#             user_item_matrix -= item_bias
#         # An item has the higher similarity score with itself,
#         # so we skip the first element.
#         sorted_ids = np.argsort(-self.item_similarity_matrix)[:, 1:self.k+1]
#         for item_id, k_items in enumerate(sorted_ids):
#             pred[:, item_id] = self.item_similarity_matrix[item_id, k_items].dot(user_item_matrix[:, k_items].T)
#             pred[:, item_id] /= np.abs(self.item_similarity_matrix[item_id, k_items]).sum() + self.eps
#         if self.debias:
#             pred += item_bias
                
#         return pred.clip(0, 5)
    
#     def get_top_recomendations(self, item_id, n=6):
#         sim_row = self.item_similarity_matrix[item_id - 1, :]
#         # once again, we skip the first item for obviouos reasons.
#         items_idxs = np.argsort(-sim_row)[1:n+1]
#         similarities = sim_row[items_idxs]
#         return items_idxs + 1, similarities