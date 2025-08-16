import pytest
import numpy as np
import scipy as sp
from recsys_pipeliner.algorithms.recommenders import (
    ItemBasedCFRecommender
)


def test_ItemBasedCFRecommender_fit(fx_user_item_matrix_toy_np):
    rec = ItemBasedCFRecommender()
    assert rec == rec.fit(fx_user_item_matrix_toy_np)


def test_ItemBasedCFRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        ItemBasedCFRecommender().fit("cat")


@pytest.mark.parametrize(
    "input, expected",
    [
        (["U00001", "I00001"], ['I00016', 'I00002', 'I00004']),
        (["U00001", "I00002"], ['I00009', 'I00001', 'I00019']),
        (["U00001", "I00003"], ['I00014', 'I00006', 'I00022']),
        (["U00001", "I00004"], ['I00006', 'I00016', 'I00003']),
        (["U00001", "I00005"], ['I00001', 'I00002', 'I00016']),
        (["U00001", "I00006"], ['I00004', 'I00003', 'I00016']),
    ],
)
def test_ItemBasedCFRecommender_recommend(fx_user_item_matrix_toy, fx_user_item_matrix_toy_encoders, input, expected):
    item_encoder, user_encoder = fx_user_item_matrix_toy_encoders
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedCFRecommender(n=3).fit(matrix)

    user, item = [input[0]], [input[1]]

    input_encoded = np.array([[
        user_encoder.transform(user)[0],
        item_encoder.transform(item)[0]
    ]])

    predictions = rec.recommend(input_encoded)
    predictions_decoded = item_encoder.inverse_transform(predictions[0])
    np.testing.assert_array_equal(predictions_decoded, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], ["I00016", "I00024", "I00005", "I00002", "I00013"]),
        (["I00002"], ["I00009", "I00015", "I00001", "I00023", "I00019"]),
        (["I00003"], ["I00011", "I00014", "I00006", "I00022", "I00018"]),
        (["I00004"], ["I00006", "I00016", "I00011", "I00024", "I00003"]),
        (["I00005"], ["I00024", "I00023", "I00001", "I00013", "I00002"]),
        (["I00006"], ["I00004", "I00011", "I00003", "I00016", "I00014"]),
        (["I00007"], ["I00015", "I00009", "I00008", "I00021", "I00024"]),
        (["I00008"], ["I00019", "I00009", "I00015", "I00007", "I00013"]),
        (["I00009"], ["I00019", "I00008", "I00002", "I00007", "I00018"]),
        (["I00010"], ["I00012", "I00003", "I00020", "I00002", "I00011"]),
    ],
)
def test_ItemBasedCFRecommender_recommend_item_only(fx_user_item_matrix_toy, fx_user_item_matrix_toy_encoders, input, expected):
    item_encoder, _ = fx_user_item_matrix_toy_encoders
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedCFRecommender().fit(matrix)

    input_encoded = item_encoder.transform(input)

    predictions = rec.recommend(input_encoded)
    predictions_decoded = item_encoder.inverse_transform(predictions[0])
    np.testing.assert_array_equal(predictions_decoded, expected)


def test_ItemBasedCFRecommender_predict(
    fx_user_item_matrix_toy, fx_user_item_matrix_toy_encoders
): 
    item_encoder, user_encoder = fx_user_item_matrix_toy_encoders

    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedCFRecommender().fit(matrix)

    users = np.full(10, "U00003")
    items = np.array(["I00007", "I00008", "I00009", "I00010", "I00011", "I00012", "I00013", "I00014", "I00015", "I00016"])

    input = np.hstack([
        user_encoder.transform(users)[np.newaxis, :].T,
        item_encoder.transform(items)[np.newaxis, :].T
    ])

    expected = [
         0.694382,
         0.738779,
         0.644923,
         0.852358,
         0.891748,
         0.739046,
         0.782644,
         0.817382,
         0.775919,
         0.812887,
    ]

    predictions = rec.predict(input)

    np.testing.assert_almost_equal(predictions, expected)
