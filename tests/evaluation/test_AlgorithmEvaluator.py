import pytest
from recsys_pipeliner.evaluation import AlgorithmEvaluator
from recsys_pipeliner.algorithms.recommenders import (
    RandomRecommender
)

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