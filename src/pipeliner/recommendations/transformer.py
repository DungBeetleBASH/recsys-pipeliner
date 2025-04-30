import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


class UserItemMatrixTransformer(TransformerMixin, BaseEstimator):
    """A custom scikit-learn transformer that accepts a numpy ndarray of user/item ratings
    with 3 columns, user, item, and rating, and returns a sparse user/item matrix.

    The input array should not include duplicates user/item pairs.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> sp.sparray:
        users, user_pos = np.unique(X[:, 0], return_inverse=True)
        items, item_pos = np.unique(X[:, 1], return_inverse=True)
        matrix_shape = (len(users), len(items))

        assert len(X) == len(
            np.unique(X[:, 0:2], axis=0)
        ), "Duplicate user/item pairs found"

        matrix = sp.csr_array(
            (X[:, 2], (user_pos, item_pos)), shape=matrix_shape, dtype=np.float32
        )

        return matrix.astype(np.float32)


class SimilarityTransformerPandas(TransformerMixin, BaseEstimator):
    """A custom scikit-learn transformer that accepts a user/item matrix where user ids are
    the index and item ids are the columns and returns a similarity matrix.

    It can be used to calculate user-user or item-item similarity.

    Args:
        kind (str): Either 'user' or 'item' to specify similarity type
        metric (str): Similarity metric to use ('cosine', 'dot', or 'euclidean')
        round (int): Number of decimal places to round results
        normalise (bool): Whether to normalize the similarity scores
    """

    def __init__(self, kind="user", metric="cosine", round=6, normalise=False):
        if kind not in ["user", "item"]:
            raise ValueError("kind must be 'user' or 'item'")
        if metric not in ["cosine", "dot", "euclidean"]:
            raise ValueError("metric must be 'cosine', 'dot', or 'euclidean'")
        self.kind = kind
        self.metric = metric
        self.normalise = normalise
        self.round = round

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        matrix = X
        if self.kind == "item":
            matrix = X.T

        if self.metric == "cosine":
            df = pd.DataFrame(
                cosine_similarity(matrix),
                index=matrix.index,
                columns=matrix.index,
            )
        else:
            raise NotImplementedError("Only cosine similarity is currently supported")

        if self.normalise:
            df = (df - df.min()) / (df.max() - df.min()).round(self.round)

        return df.astype(np.float32)


class SimilarityTransformer(TransformerMixin, BaseEstimator):
    """A custom scikit-learn transformer that accepts a sparse user/item matrix.
    It can be used to calculate user-user or item-item similarity.

    Args:
        metric (str): Similarity metric to use ('cosine', 'dot', or 'euclidean')
        round (int): Number of decimal places to round results
    """

    def __init__(self, metric="cosine", round=6):
        if metric not in ["cosine", "dot", "euclidean"]:
            raise ValueError("metric must be 'cosine', 'dot', or 'euclidean'")
        self.metric = metric
        self.round = round

    def fit(self, X, y=None):
        return self

    def transform(self, X: sp.sparray) -> sp.sparray:
        if not isinstance(X, sp.sparray):
            raise ValueError("Input must be a scipy.sparse.sparray")

        if self.metric == "cosine":
            matrix = cosine_similarity(X, dense_output=False).astype(np.float32)
        else:
            raise NotImplementedError("Only cosine similarity is currently supported")

        return np.round(matrix, self.round)
