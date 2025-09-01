import pytest
from recsys_pipeliner.evaluation import AccuracyMetrics, AlgorithmEvaluator, EvaluationDataset, TopNMetrics
from recsys_pipeliner.algorithms.recommenders import RandomRecommender


@pytest.mark.parametrize("name, expected", [
    (None, "RandomRecommender"),
    ("Random", "Random")
])
def test_AlgorithmEvaluator(name, expected):
    rec = RandomRecommender(random_seed=42)
    
    if name is None:
        evaluator = AlgorithmEvaluator(rec)
    else:
        evaluator = AlgorithmEvaluator(rec, name)

    assert evaluator.algorithm == rec
    assert evaluator.name == expected

def test_AlgorithmEvaluator_evaluate(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np, random_seed=42)
    rec = RandomRecommender(random_seed=42)
    evaluator = AlgorithmEvaluator(rec)

    result = evaluator.evaluate(dataset, top_n=None)
    
    assert result == AccuracyMetrics(rmse=0.437034, mae=0.299027)

def test_AlgorithmEvaluator_evaluate_top_n_metrics(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np, random_seed=42)
    rec = RandomRecommender(random_seed=42)
    evaluator = AlgorithmEvaluator(rec)

    accuracy_metrics, top_n_metrics = evaluator.evaluate(dataset, top_n=5)

    assert accuracy_metrics == AccuracyMetrics(rmse=0.437034, mae=0.299027)
    assert top_n_metrics == TopNMetrics(HR=1.0, cHR=1.0, ARHR=0.166667)