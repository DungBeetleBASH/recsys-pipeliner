import pytest
import numpy as np
from pipeliner.recommendations.recommender import (
    ItemBasedRecommender,
    UserBasedRecommender,
    SimilarityRecommender,
)


@pytest.mark.parametrize(
    "input, output_shape",
    [
        (["I1069"], (1, 5)),
        (["I1069", "I1013"], (2, 5)),
        ([("I1069", "U1003")], (1, 5)),
        ([("I1069", "U1003"), ("I1013", "U1003")], (2, 5)),
        ([], (0,)),
    ],
)
def test_ItemBasedRecommender(
    fx_item_similarity_matrix, fx_user_item_matrix, input, output_shape
):
    X = (
        fx_item_similarity_matrix
        if isinstance(input, str)
        else (fx_item_similarity_matrix, fx_user_item_matrix)
    )
    rec = ItemBasedRecommender().fit(X)
    predictions = rec.predict(input)
    assert predictions.shape == output_shape


def test_ItemBasedRecommender_fit_error():
    with pytest.raises(ValueError):
        ItemBasedRecommender().fit("cat")


def test_ItemBasedRecommender_predict_error(fx_item_similarity_matrix):
    rec = ItemBasedRecommender().fit(fx_item_similarity_matrix)
    with pytest.raises(ValueError):
        rec.predict([1.3])


@pytest.mark.parametrize(
    "input, output_shape",
    [
        (["U1002"], (1, 5)),
        (["U1002", "U1003"], (2, 5)),
        (["U1003", "U1003", "U1004"], (3, 5)),
        ([], (0,)),
    ],
)
def test_UserBasedRecommender(
    fx_user_similarity_matrix, fx_user_item_matrix, input, output_shape
):
    X = (fx_user_similarity_matrix, fx_user_item_matrix)
    rec = UserBasedRecommender().fit(X)
    predictions = rec.predict(input)
    assert predictions.shape == output_shape


def test_UserBasedRecommender_fit_error():
    with pytest.raises(ValueError):
        UserBasedRecommender().fit("cat")


def test_UserBasedRecommender_predict_error(
    fx_user_similarity_matrix, fx_user_item_matrix
):
    X = (fx_user_similarity_matrix, fx_user_item_matrix)
    rec = UserBasedRecommender().fit(X)
    with pytest.raises(ValueError):
        rec.predict([1.3])



@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], ['I00002', 'I00006', 'I00003', 'I00005']),
        (["I00002"], ['I00001', 'I00003', 'I00004', 'I00006']),
        (["I00003"], ['I00002', 'I00004', 'I00001', 'I00005']),
        (["I00004"], ['I00003', 'I00005', 'I00002', 'I00006']),
        (["I00005"], ['I00006', 'I00004', 'I00001', 'I00003']),
        (["I00006"], ['I00001', 'I00005', 'I00002', 'I00004']),
    ],
)
def test_SimilarityRecommender(
    fx_item_similarity_matrix_toy, input, expected
):
    rec = SimilarityRecommender(5).fit(fx_item_similarity_matrix_toy)
    predictions = rec.predict(input)[0]
    np.testing.assert_array_equal(predictions, np.array(expected))  


def test_SimilarityRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be DataFrame"):
        SimilarityRecommender().fit("cat")