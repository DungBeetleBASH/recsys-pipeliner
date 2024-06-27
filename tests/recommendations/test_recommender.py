import pytest
from pipeliner.recommendations.recommender import ItemBasedRecommender


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
    item_recommender = ItemBasedRecommender().fit(X)
    predictions = item_recommender.predict(input)
    assert predictions.shape == output_shape


def test_ItemBasedRecommender_fit_error():
    with pytest.raises(ValueError):
        ItemBasedRecommender().fit("cat")


def test_ItemBasedRecommender_predict_error(fx_item_similarity_matrix):
    item_recommender = ItemBasedRecommender().fit(fx_item_similarity_matrix)
    with pytest.raises(ValueError):
        item_recommender.predict([1.3])
