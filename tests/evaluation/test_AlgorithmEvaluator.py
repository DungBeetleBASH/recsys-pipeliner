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
from recsys_pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
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
            AccuracyMetrics(rmse=0.342165, mae=0.252982),
        ),
        (
            RandomRecommender(random_seed=42),
            5,
            (
                AccuracyMetrics(rmse=0.342165, mae=0.252982),
                TopNMetrics(HR=0.5, cHR=0.5, ARHR=0.233333),
            ),
        ),
        (
            ItemBasedCFRecommender(),
            None,
            AccuracyMetrics(rmse=0.3305, mae=0.2694),
        ),
        (
            ItemBasedCFRecommender(),
            5,
            (
                AccuracyMetrics(rmse=0.3305, mae=0.2694),
                TopNMetrics(HR=0.25, cHR=0.25, ARHR=0.083333),
            ),
        ),
    ],
)
def test_AlgorithmEvaluator_evaluate(fx_user_item_ratings_toy_np, rec, top_n, expected):
    dataset = EvaluationDataset(
        fx_user_item_ratings_toy_np,
        min_user_ratings=2,
        min_item_ratings=2,
        random_seed=42,
    )
    evaluator = AlgorithmEvaluator(rec)

    user_item_matrix_transformer = UserItemMatrixTransformer()

    user_item_matrix = user_item_matrix_transformer.transform(
        dataset.trainset,
    )

    result = evaluator.evaluate(
        user_item_matrix, dataset.testset, dataset.anti_testset, top_n=top_n
    )

    assert result == expected
