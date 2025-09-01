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
        unique_users = np.unique(self._dataset[:, 0])
        top_n_predictions = []
        for user in unique_users:
            user_data = self._dataset[(self._dataset[:, 0] == user) & (self._dataset[:, 2] >= self._minimum_rating)]
            user_data = user_data[user_data[:, 2].argsort()[::-1]]
            top_n_predictions.append(user_data[:self._n])
        self._top_n_predictions = np.concatenate(top_n_predictions, axis=0)

    def _calculate_hit_rate_metrics(self):
        
        hits = 0
        cumulative_hits = 0
        reciprocal_hits = 0
        total = len(self._testset)

        for row in self._testset:
            uid, iid, true_rating = row[0], row[1], row[2]

            # Is it in the predicted top-N for this user?
            # # TODO: confirm this is correct

            user_top_n_predictions = self._top_n_predictions[self._top_n_predictions[:, 0] == uid]
            print("user_top_n_predictions", user_top_n_predictions)
            print("iid", iid)
            print(user_top_n_predictions[:, 1] == iid)
            rank = self._n - np.argmin(user_top_n_predictions[:, 1] == iid)

            print("rank", rank)

            if rank is not None:
                hits += 1
                reciprocal_hits += 1.0 / (rank + 1)
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