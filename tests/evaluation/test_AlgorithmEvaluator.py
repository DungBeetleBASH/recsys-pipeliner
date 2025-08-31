import pytest
from recsys_pipeliner.evaluation import AlgorithmEvaluator
from recsys_pipeliner.algorithms.recommenders import (
    RandomRecommender
)
from recsys_pipeliner.evaluation import EvaluationDataset


@pytest.mark.parametrize("name, expected", [
    (None, "RandomRecommender"),
    ("Random", "Random")
])
def test_AlgorithmEvaluator(name, expected):
    rec = RandomRecommender()
    
    if name is None:
        evaluator = AlgorithmEvaluator(rec)
    else:
        evaluator = AlgorithmEvaluator(rec, name)

    assert evaluator.algorithm == rec
    assert evaluator.name == expected

# def test_AlgorithmEvaluator_evaluate(fx_user_item_ratings_toy_np):
#     dataset = EvaluationDataset(fx_user_item_ratings_toy_np)
#     rec = RandomRecommender()
#     evaluator = AlgorithmEvaluator(rec)
#     result = evaluator.evaluate(dataset, n=10, top_n_metrics=False)
#     assert result == (0.0, 0.0)