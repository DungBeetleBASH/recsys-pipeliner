import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


class UserItemMatrixTransformer(TransformerMixin, BaseEstimator):
    """
    This class is a custom scikit-learn transformer
    that accepts a pandas dataframe of user/item interactions
    and returns a user/item matrix.

    :param user (str): Column name for user id
    :param item (str): Column name for item id
    :param rating (float): Column name for user/item rating
    :param agg (str): Panadas aggregation function to use when combining duplicate user/item interactions
    :param binary (bool): If True, user/item interactions are converted to binary values in the user/item output matrix
    """

    _parameter_constraints = {
        "user": [str],
        "item": [str],
        "rating": [str],
        "agg": [str],
        "normalise": [bool],
    }

    def __init__(
        self, user="user_id", item="item_id", rating="rating", agg="max", binary=False
    ):
        self.user = user
        self.item = item
        self.rating = rating
        self.agg = agg
        self.binary = binary

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        matrix = X.groupby([self.user, self.item])[self.rating].agg(self.agg).unstack()
        if self.binary:
            return matrix.notnull().astype(int)
        else:
            return matrix.fillna(0)


class SimilarityTransformer(TransformerMixin, BaseEstimator):
    """
    This class is a custom scikit-learn transformer
    that accepts a user/item matrix where user ids are
    the index and item ids are the columns and returns
    a similarity matrix. It can be used to calculate
    user-user or item-item similarity.
    """

    def __init__(self, kind="user", metric="cosine", normalise=False):
        if kind not in ["user", "item"]:
            raise ValueError("kind must be 'user' or 'item'")
        if metric not in ["cosine", "dot", "euclidean"]:
            raise ValueError("metric must be 'cosine', 'dot', or 'euclidean'")
        self.kind = kind
        self.metric = metric
        self.normalise = normalise

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        matrix = X
        if self.kind == "item":
            matrix = X.T

        if self.metric == "cosine":
            df = pd.DataFrame(
                cosine_similarity(matrix), index=matrix.index, columns=matrix.index
            )
        else:
            raise NotImplementedError("Only cosine similarity is currently supported")

        if self.normalise:
            df = (df - df.min()) / (df.max() - df.min())

        return df
