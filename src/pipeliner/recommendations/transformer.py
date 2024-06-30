import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
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

    def transform(self, X):
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


# WIP: taken from: https://ploomber.io/blog/sklearn-custom/
class TemplateTransformer(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, accept_sparse=True)

        # Return the transformer
        return self

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return np.sqrt(X)

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}
