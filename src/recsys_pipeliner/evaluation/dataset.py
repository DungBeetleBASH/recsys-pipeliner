import numpy as np


class EvaluationDataset:
    """
    Dataset for recommender systems.

    Args:
        ratings: np.ndarray with columns "user_id", "item_id", "rating"
        min_user_ratings: Minimum number of ratings per user.
        min_item_ratings: Minimum number of ratings per item.
    """

    _ratings: np.ndarray

    def __init__(
        self,
        ratings: np.ndarray,
        min_user_ratings: int = 5,
        min_item_ratings: int = 5,
        random_seed: int | None = None,
    ):
        self._ratings = ratings
        self._random_seed = random_seed

        self._create_usable_ratings(min_user_ratings, min_item_ratings)
        self._create_train_test_split()
        self._create_anti_testset()

    def _create_train_test_split(self):
        self._trainset, self._testset = next(
            self.leave_one_out(random_seed=self._random_seed)
        )

    def _create_usable_ratings(self, min_user_ratings, min_item_ratings):
        users, items = self._ratings[:, 0], self._ratings[:, 1]
        unique_users, unique_user_counts = np.unique(users, return_counts=True)
        unique_items, unique_item_counts = np.unique(items, return_counts=True)
        usable_users = unique_users[unique_user_counts >= min_user_ratings]
        usable_items = unique_items[unique_item_counts >= min_item_ratings]
        self._usable_ratings = self._ratings[
            np.isin(users, usable_users) & np.isin(items, usable_items)
        ]

    def _create_anti_testset(self):
        users, items = self._ratings[:, 0], self._ratings[:, 1]
        unique_users = np.unique(users)
        unique_items = np.unique(items)
        anti_testset = []

        for user in unique_users:
            rated_items = items[users == user]
            unrated_items = np.setdiff1d(unique_items, rated_items)[np.newaxis, :].T
            anti_testset.append(np.insert(unrated_items, 0, user, axis=1))

        self._anti_testset = np.concatenate(anti_testset, axis=0).astype(np.int32)

    @property
    def full(self) -> np.ndarray:
        return self._ratings

    @property
    def usable(self) -> np.ndarray:
        return self._usable_ratings

    @property
    def trainset(self) -> np.ndarray:
        return self._trainset

    @property
    def testset(self) -> np.ndarray:
        return self._testset

    @property
    def anti_testset(self) -> np.ndarray:
        return self._anti_testset

    def leave_one_out(self, **kwargs):
        return LeaveOneOutIterator(self, **kwargs)


class LeaveOneOutIterator:
    """
    Dataset for leave-one-out cross-validation.

    Args:
        dataset: EvaluationDataset
        n_splits: Number of cross-validation splits.
        random_seed: Random seed.
    """

    def __init__(
        self,
        dataset: EvaluationDataset,
        n_splits: int = 1,
        random_seed: int | None = None,
    ):
        self._ratings = dataset.usable
        self._n_splits = n_splits
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
        users = self._ratings[:, 0]
        unique_users = np.unique(users)
        trainset = []
        testset = []

        if self._random_seed is not None:
            rand = np.random.RandomState(self._random_seed)
        else:
            rand = np.random

        for user in unique_users:
            user_ratings = self._ratings[users == user]
            test_rating_idx = rand.randint(0, len(user_ratings))
            test_rating = user_ratings[test_rating_idx]
            testset.append(test_rating)
            train_ratings = np.delete(user_ratings, test_rating_idx, axis=0)
            trainset.append(train_ratings)

        if len(trainset) == 0 or len(testset) == 0:
            raise ValueError("No users with enough ratings to split")

        return np.vstack(trainset), np.vstack(testset)
