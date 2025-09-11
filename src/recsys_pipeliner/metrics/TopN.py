import numpy as np

class TopN:
    def __init__(self, testset: np.ndarray, predictions: np.ndarray, anti_testset: np.ndarray, anti_test_predictions: np.ndarray, n: int = 10, minimum_rating: float = 1e-5):
        self._testset = testset
        self._predictions = predictions
        self._anti_testset = anti_testset
        self._anti_test_predictions = anti_test_predictions
        self._dataset = np.hstack([testset, predictions[:, np.newaxis]])
        self._n = n
        self._minimum_rating = minimum_rating

        self._create_top_n_predictions()
        self._calculate_hit_rate_metrics()

    def _create_top_n_predictions(self):
        # Build candidate lists per user using anti-test items scored by the model,
        # and include the held-out test item scored by the model as well.
        users = np.concatenate([self._anti_testset[:, 0], self._testset[:, 0]])
        items = np.concatenate([self._anti_testset[:, 1], self._testset[:, 1]])
        predictions = np.concatenate([self._anti_test_predictions, self._predictions])
        data = np.stack([users, items, predictions], axis=1)
        
        unique_users = np.unique(users)

        candidates_by_user = {uid: [] for uid in unique_users}

        # Candidates
        for (uid, iid, score) in data:
            candidates_by_user[uid].append((iid, score))

        # Compute Top-N item lists per user by predicted score (descending)
        top_n_items_by_user: dict = {}
        for uid, candidates in candidates_by_user.items():
            candidates_sorted = sorted(candidates, key=lambda t: t[1], reverse=True)
            top_n_items = [iid for iid, _ in candidates_sorted[: self._n]]
            top_n_items_by_user[uid] = top_n_items

        self._top_n_items_by_user = top_n_items_by_user

    def _calculate_hit_rate_metrics(self):
        hits = 0
        cumulative_hits = 0
        reciprocal_hits = 0.0
        total = len(self._testset)

        if total == 0:
            self._hit_rate = 0.0
            self._cumulative_hit_rate = 0.0
            self._average_reciprocal_hit_rank = 0.0
            return

        for row in self._testset:
            uid, iid, true_rating = row[0], row[1], row[2]

            # Is it in the predicted Top-N for this user?
            topn_items = self._top_n_items_by_user.get(uid, [])
            if len(topn_items) == 0:
                continue

            if iid in topn_items:
                rank = topn_items.index(iid) + 1  # 1-based rank
                hits += 1
                reciprocal_hits += 1.0 / rank
                if true_rating >= self._minimum_rating:
                    cumulative_hits += 1

        hit_rate = hits / total
        cumulative_hit_rate = cumulative_hits / total
        average_reciprocal_hit_rank = reciprocal_hits / total

        self._hit_rate = np.round(hit_rate, 6)
        self._cumulative_hit_rate = np.round(cumulative_hit_rate, 6)
        self._average_reciprocal_hit_rank = np.round(average_reciprocal_hit_rank, 6)

    @property
    def hit_rate(self):
        return self._hit_rate
    
    @property
    def cumulative_hit_rate(self):
        return self._cumulative_hit_rate
    
    @property
    def average_reciprocal_hit_rank(self):
        return self._average_reciprocal_hit_rank