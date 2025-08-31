from recsys_pipeliner.metrics import Accuracy, TopN
import logging
from collections import namedtuple

AccuracyMetrics = namedtuple("AccuracyMetrics", ["rmse", "mae"])
TopNMetrics = namedtuple("TopNMetrics", ["hit_rate"])


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
        self, evaluation_dataset, n=10, top_n_metrics=True
    ) -> AccuracyMetrics | tuple[AccuracyMetrics, TopNMetrics]:
        self._logger.info(f"Evaluating: {self._name}")
        
        self._algorithm.fit(evaluation_dataset.trainset)
        predictions = self._algorithm.test(evaluation_dataset.testset)
        rmse = Accuracy.rmse(predictions)
        mae = Accuracy.mae(predictions)

        accuracy = AccuracyMetrics(rmse, mae)

        if not top_n_metrics:
            return accuracy

        hit_rate = TopN.hit_rate(predictions)
        top_n = TopNMetrics(hit_rate)

        return accuracy, top_n


    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
