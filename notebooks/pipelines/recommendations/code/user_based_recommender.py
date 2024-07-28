import argparse
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import logging

logging.basicConfig(level=logging.INFO)


class UserBasedRecommender(BaseEstimator):
    """User-based collaborative filtering recommender."""

    n: int
    n_users: int
    threshold: float
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5, n_users=5, threshold=0.1):
        self.n = n
        self.n_users = n_users
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
          X (tuple[pd.DataFrame, pd.DataFrame]):
            tuple of (similarity matrix, user/item matrix)
        """
        if isinstance(X, tuple):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be tuple of (DataFrame, DataFrame)")

        return self

    def _get_similar_users(self, user_id: str):
        return (
            self.similarity_matrix[self.similarity_matrix[user_id] > self.threshold][
                user_id
            ]
            .drop(user_id, errors="ignore")
            .sort_values(ascending=False)
        )

    def _get_exclusions(self, user_id):
        single_user_matrix = self.user_item_matrix.loc[user_id]
        user_rated_items = single_user_matrix[single_user_matrix > 0]
        return user_rated_items.index.to_list()

    def _get_recommendations(self, user_id):
        if not isinstance(user_id, str):
            raise ValueError("Input items should be str")
        exclusions = self._get_exclusions(user_id)
        similar_users = self._get_similar_users(user_id)
        matrix = self.user_item_matrix.T[similar_users.head(self.n_users).index]

        user_recommendations = (
            matrix[~matrix.index.isin(exclusions) & (matrix > 0).any(axis="columns")]
            .max(axis=1)
            .sort_values(ascending=False)
        )

        return np.array(user_recommendations.head(self.n).index)

    def predict(self, X):
        """Predicts n item recommendations for each user_id provided.

        Args:
          X (Sequence): List of user_id

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._get_recommendations(item) for item in X])

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))

    args = parser.parse_args()

    logging.info(f"SM_MODEL_DIR: {args.model_dir}")
    logging.info(f"SM_INPUT_DIR: {args.input}")

    base_dir = "/opt/ml"

    user_item_matrix = pd.read_csv(
        f"{args.input}/data/user_item_matrix/user_item_matrix.csv", dtype=np.float64
    )
    similarity_matrix = pd.read_csv(
        f"{args.input}/data/similarity_matrix/user_similarity_matrix.csv",
        dtype=np.float64,
    )

    rec = UserBasedRecommender(5, 5, 0.1).fit((similarity_matrix, user_item_matrix))

    joblib.dump(rec, os.path.join(args.model_dir, "rec.joblib"))
