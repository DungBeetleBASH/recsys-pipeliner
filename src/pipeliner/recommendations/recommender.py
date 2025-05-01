import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from pipeliner.recommendations.transformer import (
    SimilarityTransformer,
)


class ItemBasedRecommenderPandas(BaseEstimator):
    """Item-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each item.
    """

    n: int
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X (pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]):
                Single DataFrame with similarity matrix
                or tuple of (similarity matrix, user/item matrix)

        Returns:
            self: Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            self.similarity_matrix = X
        elif isinstance(X, tuple):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be DataFrame or (DataFrame, DataFrame)")

        return self

    def _get_exclusions(self, item_id: str, user_id: str | None):
        if user_id is None:
            return [item_id]
        single_user_matrix = self.user_item_matrix.loc[user_id]
        user_rated_items = single_user_matrix[single_user_matrix > 0]
        return [item_id] + user_rated_items.index.to_list()

    def _get_recommendations(self, item) -> np.array:
        if isinstance(item, str):
            item_id, user_id = item, None
        elif isinstance(item, tuple):
            item_id, user_id = item[0], item[1]
        else:
            raise ValueError("Input items should be str or (str, str)")

        exclusions = self._get_exclusions(item_id, user_id)

        item_recommendations = (
            self.similarity_matrix[item_id]
            .drop(exclusions, errors="ignore")
            .sort_values(ascending=False)
        )
        return np.array(item_recommendations.head(self.n).index)

    def predict(self, X) -> np.array:
        """Predicts n item recommendations for each item_id provided
        If tuples of (user_id, item_id) are provided, items previously
        rated by the user will be excluded from the recommendations.

        Args:
          X (Sequence): List of item_id or (item_id, user_id)

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._get_recommendations(item) for item in X])

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")


class UserBasedRecommender(BaseEstimator):
    """User-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each user
        n_users (int): Number of similar users to consider for recommendations
    """

    n: int
    n_users: int
    similarity_matrix: sp.sparray
    user_item_matrix: sp.sparray

    def __init__(self, n=5, n_users=5):
        self.n = n
        self.n_users = n_users
        self._user_transformer = SimilarityTransformer()
        self._item_transformer = SimilarityTransformer()

    def fit(self, X: sp.sparray, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparray:
                user/item matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparray):
            self._user_item_matrix = X
            self._user_indices = np.arange(X.shape[0])
            self._item_indices = np.arange(X.shape[1])
            self._user_similarity_matrix = self._user_transformer.transform(X)
            # self._item_similarity_matrix = self._item_transformer.transform(X.T)
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _get_similar_users(self, id: int) -> np.array:
        matrix = self._user_similarity_matrix[[id]]
        user_mask = matrix > 0
        user_mask[[0], [id]] = False
        user_sorter = np.argsort(1 - matrix.toarray()[0], kind="stable")
        sorted_mask = user_mask.toarray()[0][user_sorter]
        similar_users = user_sorter[sorted_mask][:self.n]

        return similar_users

    def _get_exclusions(self, id: int) -> np.array:
        single_user_ratings = self._user_item_matrix[
            [id]
        ]
        rated = (single_user_ratings > 0).nonzero()[1]
        return rated

    def _get_recommendations(self, id: int) -> np.array:
        excluded_items = self._get_exclusions(id)
        similar_users = self._get_similar_users(id)
        # matrix = self.user_item_matrix.T[similar_users.index]

        return np.array([])

    # def predict(self, X) -> np.array:
    #     """Predicts n item recommendations for each user_id provided.

    #     Args:
    #       X (Sequence): List of user_id

    #     Returns:
    #       np.array of shape (X.shape[0], n)
    #     """
    #     return np.array([self._get_recommendations(user_id) for user_id in X])

    # def score(self, y_preds, y_test):
    #     """Calculates the accuracy score of the recommender.

    #     Args:
    #         y_preds: Predicted recommendations
    #         y_test: Ground truth recommendations

    #     Returns:
    #         float: Accuracy score between 0 and 1
    #     """
    #     scores = np.array([1.0 if t in p else 0.0 for t, p in zip(y_test, y_preds)])
    #     if len(scores) == 0:
    #         return np.nan
    #     accuracy = np.mean(scores)
    #     return accuracy

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")


class SimilarityRecommender(BaseEstimator):
    """Similarity recommender.

    Args:
        n (int): Number of recommendations to generate.
    """

    n: int
    similarity_matrix: sp.sparray

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparray:
                similarity matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparray):
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

    def predict(self, X) -> list[np.array]:
        """Predicts n recommendations for each id provided

        Args:
          X (Sequence): List of id

        Returns:
          list of np.array
        """
        return [self._get_recommendations(id) for id in X]

    def predict_proba(self, X):
        return self.similarity_matrix[X]
