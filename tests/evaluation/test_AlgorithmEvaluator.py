import pytest
from recsys_pipeliner.evaluation import (
    AccuracyMetrics,
    AlgorithmEvaluator,
    EvaluationDataset,
    TopNMetrics,
)
from recsys_pipeliner.algorithms.recommenders import (
    RandomRecommender,
    ItemBasedCFRecommender,
)


@pytest.mark.parametrize(
    "name, expected", [(None, "RandomRecommender"), ("Random", "Random")]
)
def test_AlgorithmEvaluator(name, expected):
    rec = RandomRecommender(random_seed=42)

    if name is None:
        evaluator = AlgorithmEvaluator(rec)
    else:
        evaluator = AlgorithmEvaluator(rec, name)

    assert evaluator.algorithm == rec
    assert evaluator.name == expected


@pytest.mark.parametrize(
    "rec, top_n, expected",
    [
        (
            RandomRecommender(random_seed=42),
            None,
            AccuracyMetrics(rmse=0.437034, mae=0.299027),
        ),
        (
            RandomRecommender(random_seed=42),
            5,
            (
                AccuracyMetrics(rmse=0.437034, mae=0.299027),
                TopNMetrics(HR=1.0, cHR=1.0, ARHR=0.166667),
            ),
        ),
        # (
        #     ItemBasedCFRecommender(),
        #     None,
        #     AccuracyMetrics(rmse=0.437034, mae=0.299027),
        # ),
        # (
        #     ItemBasedCFRecommender(),
        #     5,
        #     (
        #         AccuracyMetrics(rmse=0.437034, mae=0.299027),
        #         TopNMetrics(HR=1.0, cHR=1.0, ARHR=0.166667),
        #     ),
        # ),
    ],
)
def test_AlgorithmEvaluator_evaluate(fx_user_item_ratings_toy_np, rec, top_n, expected):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np, random_seed=42)
    evaluator = AlgorithmEvaluator(rec)

    result = evaluator.evaluate(dataset, top_n=top_n)

    assert result == expected
