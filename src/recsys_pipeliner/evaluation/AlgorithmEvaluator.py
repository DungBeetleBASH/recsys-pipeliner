from recsys_pipeliner.metrics import Accuracy, TopN
import logging
from collections import namedtuple
import numpy as np

AccuracyMetrics = namedtuple("AccuracyMetrics", ["rmse", "mae"])
TopNMetrics = namedtuple("TopNMetrics", ["HR", "cHR", "ARHR"])


class AlgorithmEvaluator:
    def __init__(self, algorithm, name=None, minimum_rating=1e-5, coverage_threshold=1e-5, verbose=False):
        self._algorithm = algorithm
        self._name = name if name else algorithm.__class__.__name__
        self._minimum_rating = minimum_rating
        self._coverage_threshold = coverage_threshold

        self._logger = logging.getLogger(
            f"{self.__class__.__name__}({algorithm.__class__.__name__}, {name})"
        )
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def evaluate(
        self, evaluation_dataset, top_n: int | None = None
    ) -> AccuracyMetrics | tuple[AccuracyMetrics, TopNMetrics]:
        self._logger.info(f"Evaluating: {self._name}")

        self._algorithm.fit(evaluation_dataset.trainset)

        testset = evaluation_dataset.testset
        predictions = self._algorithm.predict(testset)
        y_true = testset[:, 2]

        rmse = np.round(Accuracy.rmse(predictions, y_true), 6)
        mae = np.round(Accuracy.mae(predictions, y_true), 6)

        accuracy_metrics = AccuracyMetrics(rmse, mae)

        if not top_n:
            return accuracy_metrics
        
        anti_testset = evaluation_dataset.anti_testset
        anti_test_predictions = self._algorithm.predict(
            anti_testset
        )
        
        top_n = TopN(
            testset=testset,
            predictions=predictions,
            anti_testset=anti_testset,
            anti_test_predictions=anti_test_predictions,
            n=top_n, 
            minimum_rating=self._minimum_rating
        )

        top_n_metrics = TopNMetrics(
            HR=top_n.hit_rate,
            cHR=top_n.cumulative_hit_rate,
            ARHR=top_n.average_reciprocal_hit_rank
        )

        return accuracy_metrics, top_n_metrics


    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
