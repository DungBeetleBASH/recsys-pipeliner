import pandas as pd
import numpy as np
import scipy as sp
from sklearn.preprocessing import LabelEncoder


class RatingsDataset:
    """
    Dataset for recommender systems.

    Args:
        ratings_df: DataFrame with columns "user_id", "item_id", "rating"

    Attributes:
        dataset: numpy array of shape (n_ratings, 3)
    """

    _ratings_df: pd.DataFrame
    _dataset: np.ndarray
    _user_encoder: LabelEncoder
    _item_encoder: LabelEncoder

    def __init__(self, ratings_df: pd.DataFrame):
        self._ratings_df = ratings_df
        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        ratings = self._ratings_df[["user_id", "item_id", "rating"]].copy()
        ratings.loc[:, "user_id"] = self._user_encoder.fit_transform(ratings["user_id"])
        ratings.loc[:, "item_id"] = self._item_encoder.fit_transform(ratings["item_id"])
        self._dataset = ratings.to_numpy()
        self._anti_testset = None

    @property
    def dataset(self) -> np.ndarray:
        return self._dataset

    @property
    def anti_testset(self) -> np.ndarray:
        if self._anti_testset is None:
            users, items = self._dataset[:, 0], self._dataset[:, 1]
            unique_users = np.unique(users)
            unique_items = np.unique(items)
            anti_testset = []

            for user in unique_users:
                rated_items = items[users == user]
                unrated_items = np.setdiff1d(unique_items, rated_items)[np.newaxis, :].T
                anti_testset.append(np.insert(unrated_items, 0, user, axis=1))

            self._anti_testset = np.concatenate(anti_testset, axis=0).astype(np.int32)
        return self._anti_testset

    def leave_one_out(self, **kwargs):
        return LeaveOneOutDataset(self, **kwargs)


class LeaveOneOutDataset:
    """
    Dataset for leave-one-out cross-validation.

    Args:
        dataset: RatingsDataset
        n_splits: Number of cross-validation splits.
        min_ratings: Minimum number of ratings per user.
        random_seed: Random seed.
    """

    def __init__(
        self,
        dataset: RatingsDataset,
        n_splits: int = 1,
        min_ratings: int = 5,
        random_seed: int | None = None,
    ):
        self._dataset = dataset.dataset
        self._n_splits = n_splits
        self._min_ratings = min_ratings
        self._current_split = 0
        self._random_seed = random_seed

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_split < self._n_splits:
            self._current_split += 1
            return self._split()
        else:
            self._current_split = 0
            raise StopIteration

    def _split(self):
        users = self._dataset[:, 0]
        unique_users = np.unique(users)
        trainset = []
        testset = []

        if self._random_seed is not None:
            rand = np.random.RandomState(self._random_seed)
        else:
            rand = np.random

        for user in unique_users:
            user_ratings = self._dataset[users == user]
            if len(user_ratings) < self._min_ratings:
                continue
            test_rating_idx = rand.randint(0, len(user_ratings))
            test_rating = user_ratings[test_rating_idx]
            testset.append(test_rating)
            train_ratings = np.delete(user_ratings, test_rating_idx, axis=0)
            trainset.append(train_ratings)

        if len(trainset) == 0 or len(testset) == 0:
            raise ValueError("No users with enough ratings to split")

        return np.vstack(trainset), np.vstack(testset)
