import pytest
from recsys_pipeliner.dataset import RatingsDataset, LeaveOneOutDataset


def test_RatingsDataset(fx_user_item_ratings_toy):
    dataset = RatingsDataset(fx_user_item_ratings_toy)
    assert dataset.dataset.shape == fx_user_item_ratings_toy.shape

def test_RatingsDataset_anti_testset(fx_user_item_ratings_toy):
    dataset = RatingsDataset(fx_user_item_ratings_toy)

    anti_testset = dataset.anti_testset

    assert anti_testset.shape[1] == 2

def test_RatingsDataset_anti_testset_cached(fx_user_item_ratings_toy):
    dataset = RatingsDataset(fx_user_item_ratings_toy)

    anti_testset_1 = dataset.anti_testset
    anti_testset_2 = dataset.anti_testset

    assert anti_testset_1 is anti_testset_2

def test_RatingsDataset_leave_one_out(fx_user_item_ratings_toy):
    dataset = RatingsDataset(fx_user_item_ratings_toy)
    leave_one_out = dataset.leave_one_out()

    assert isinstance(leave_one_out, LeaveOneOutDataset)

    for trainset, testset in leave_one_out:
        assert trainset.shape[1] == 3
        assert testset.shape[1] == 3

def test_RatingsDataset_leave_one_out_random_seed(mocker, fx_user_item_ratings_toy):
    mock_RandomState = mocker.patch("recsys_pipeliner.dataset.np.random.RandomState")
    mock_randint = mock_RandomState.return_value.randint
    mock_randint.return_value = 0

    dataset = RatingsDataset(fx_user_item_ratings_toy)
    leave_one_out = dataset.leave_one_out(random_seed=42)

    assert isinstance(leave_one_out, LeaveOneOutDataset)

    next(leave_one_out)
    mock_randint.assert_called()

def test_RatingsDataset_leave_one_out_min_ratings_error(fx_user_item_ratings_toy):
    dataset = RatingsDataset(fx_user_item_ratings_toy)
    leave_one_out = dataset.leave_one_out(min_ratings=20)

    with pytest.raises(ValueError, match="No users with enough ratings to split"):
        next(leave_one_out)