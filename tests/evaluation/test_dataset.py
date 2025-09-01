import pytest
from recsys_pipeliner.evaluation import EvaluationDataset, LeaveOneOutIterator


def test_EvaluationDataset(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np)
    assert dataset.full.shape == fx_user_item_ratings_toy_np.shape

def test_EvaluationDataset_anti_testset(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np)

    anti_testset = dataset.anti_testset

    assert anti_testset.shape[1] == 2

def test_EvaluationDataset_anti_testset_cached(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np)

    anti_testset_1 = dataset.anti_testset
    anti_testset_2 = dataset.anti_testset

    assert anti_testset_1 is anti_testset_2

def test_EvaluationDataset_leave_one_out(fx_user_item_ratings_toy_np):
    dataset = EvaluationDataset(fx_user_item_ratings_toy_np)
    leave_one_out = dataset.leave_one_out()

    assert isinstance(leave_one_out, LeaveOneOutIterator)

    for trainset, testset in leave_one_out:
        assert trainset.shape[1] == 3
        assert testset.shape[1] == 3

def test_EvaluationDataset_leave_one_out_random_seed(mocker, fx_user_item_ratings_toy_np):
    mock_RandomState = mocker.patch("recsys_pipeliner.evaluation.dataset.np.random.RandomState")
    mock_randint = mock_RandomState.return_value.randint
    mock_randint.return_value = 0

    dataset = EvaluationDataset(fx_user_item_ratings_toy_np)
    leave_one_out = dataset.leave_one_out(random_seed=42)

    assert isinstance(leave_one_out, LeaveOneOutIterator)

    next(leave_one_out)
    mock_randint.assert_called()

def test_EvaluationDataset_leave_one_out_min_ratings_error(fx_user_item_ratings_toy_np):
    with pytest.raises(ValueError, match="No users with enough ratings to split"):
        EvaluationDataset(fx_user_item_ratings_toy_np, min_user_ratings=20, min_item_ratings=20)