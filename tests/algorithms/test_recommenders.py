import pytest
import numpy as np
import scipy as sp
from recsys_pipeliner.algorithms.recommenders import (
    ItemBasedCFRecommender,
    RandomRecommender

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


def test_ItemBasedCFRecommender_recommend_error(fx_user_item_matrix_toy):
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedCFRecommender().fit(matrix)
    
    input = np.array([[1, 1, 1]])

    with pytest.raises(ValueError, match="X must be a 1D or 2D array"):
        rec.recommend(input)


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


def test_RandomRecommender_fit(fx_user_item_matrix_toy_np):
    rec = RandomRecommender()
    assert rec == rec.fit(fx_user_item_matrix_toy_np)

def test_RandomRecommender_predict(fx_user_item_matrix_toy_np):
    rec = RandomRecommender(n=10, random_seed=42).fit(fx_user_item_matrix_toy_np.toarray())
    users, items = np.arange(10), np.arange(10)
    input = np.vstack([users, items]).T

    predictions = rec.predict(input)

    expected = [
        0.37454,
        0.950714,
        0.731994,
        0.598658,
        0.156019,
        0.155995,
        0.058084,
        0.866176,
        0.601115,
        0.708073
    ]

    np.testing.assert_almost_equal(predictions, expected)

def test_RandomRecommender_recommend(fx_user_item_matrix_toy_np):
    rec = RandomRecommender(n=10, random_seed=42).fit(fx_user_item_matrix_toy_np.toarray())
    users, items = np.arange(10), np.arange(10)
    input = np.vstack([users, items]).T

    recommendations = rec.recommend(input)

    expected = np.array([
        [ 6, 19, 14, 10,  7, 20,  6, 18, 22, 10],
        [10, 23, 20,  3,  7, 23,  2, 21, 20,  1],
        [23, 11,  5,  1, 20,  0, 11, 21, 11, 16],
        [ 9, 15, 14, 14, 18, 11, 22, 19,  2,  4],
        [18,  6, 20,  8,  6, 17,  3, 13, 17,  8],
        [20,  1, 19, 14,  6, 11,  7, 14,  2, 13],
        [16,  3, 17,  7,  3,  1,  5, 21,  9,  3],
        [21, 17, 11,  1,  9,  3, 13, 15, 14,  7],
        [13, 22,  7, 20, 15, 12, 17, 14, 20, 23],
        [12,  8, 14, 12,  0,  6,  8, 23,  0, 11]
    ]).astype(np.int32)

    np.testing.assert_array_equal(recommendations, expected)
