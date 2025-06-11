import pytest
import numpy as np
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from pipeliner.recommendations.recommender import (
    UserBasedRecommender,
    ItemBasedRecommender,
    SimilarityRecommender,
)


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], ['I00012', 'I00002', 'I00011', 'I00003', 'I00004']),
        (["I00002"], ['I00001', 'I00003', 'I00012', 'I00004', 'I00011']),
        (["I00003"], ['I00002', 'I00004', 'I00001', 'I00005', 'I00012']),
        (["I00004"], ['I00003', 'I00005', 'I00002', 'I00006', 'I00012']),
        (["I00005"], ['I00004', 'I00006', 'I00003', 'I00007', 'I00001']),
        (["I00006"], ['I00007', 'I00005', 'I00008', 'I00004', 'I00010']),
    ],
)
def test_SimilarityRecommender(fx_item_similarity_matrix_toy, input, expected):
    item_ids = fx_item_similarity_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    item_similarity_matrix_np = fx_item_similarity_matrix_toy.to_numpy()
    item_similarity_matrix_np_sparse = sp.sparse.csr_array(item_similarity_matrix_np)

    rec = SimilarityRecommender(5).fit(item_similarity_matrix_np_sparse)
    predictions = rec.recommend(item_encoder.transform(input))[0]
    predictions_decoded = item_encoder.inverse_transform(predictions)
    np.testing.assert_array_equal(predictions_decoded, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], [1.     , 0.64285, 0.50527, 0.4116 , 0.39642, 0.20552, 0.     , 0.23835, 0.40195, 0.40799, 0.51602, 0.65564]),
        (["I00002"], [0.64285, 1.     , 0.63202, 0.49563, 0.38417, 0.3655 , 0.21015, 0.     , 0.24326, 0.40828, 0.41853, 0.52868]),
        (["I00003"], [0.50527, 0.63202, 1.     , 0.62321, 0.47361, 0.35156, 0.37471, 0.21657, 0.     , 0.25021, 0.418  , 0.43262]),
    ],
)
def test_SimilarityRecommender_predict_proba(
    fx_item_similarity_matrix_toy, input, expected
):
    item_ids = fx_item_similarity_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    item_similarity_matrix_np = fx_item_similarity_matrix_toy.to_numpy()
    item_similarity_matrix_np_sparse = sp.sparse.csr_array(item_similarity_matrix_np)

    rec = SimilarityRecommender(5).fit(item_similarity_matrix_np_sparse)
    probs = rec.predict_proba(item_encoder.transform(input)).toarray()[0].round(5)
    np.testing.assert_array_equal(probs, np.array(expected).astype(np.float32).round(5))


def test_SimilarityRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        SimilarityRecommender().fit("cat")


def test_SimilarityRecommender_omit_input():
    similarity_matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    similarity_matrix_sparse = sp.sparse.csr_array(similarity_matrix)
    rec = SimilarityRecommender(5)
    rec.fit(similarity_matrix_sparse)
    predictions = rec.recommend([0, 1, 2])

    for pred, expected in zip(predictions, [[2], [], [0]]):
        np.testing.assert_array_equal(pred, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (["U00001"], ['I00007', 'I00012', 'I00008', 'I00009', 'I00011']),
        (["U00002"], ['I00008', 'I00001', 'I00009', 'I00010', 'I00012']),
        (["U00003"], ['I00009', 'I00002', 'I00010', 'I00011', 'I00012']),
        (["U00004"], ['I00010', 'I00003', 'I00011', 'I00012', 'I00001']),
        (["U00005"], ['I00011', 'I00004', 'I00003', 'I00012', 'I00001']),
        (["U00006"], ['I00012', 'I00005', 'I00004', 'I00001', 'I00002']),
    ],
)
def test_UserBasedRecommender_predict(fx_user_item_matrix_toy, input, expected):
    item_ids = fx_user_item_matrix_toy.columns.to_numpy()
    user_ids = fx_user_item_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    user_encoder = LabelEncoder().fit(user_ids)
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = UserBasedRecommender().fit(matrix)

    input_encoded = user_encoder.transform(input)
    predictions = rec.recommend(input_encoded)
    predictions_decoded = item_encoder.inverse_transform(predictions[0])
    np.testing.assert_array_equal(predictions_decoded, expected)


def test_UserBasedRecommender_fit(fx_user_item_matrix_toy_np):
    rec = UserBasedRecommender()
    assert rec == rec.fit(fx_user_item_matrix_toy_np)


def test_UserBasedRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        UserBasedRecommender().fit("cat")


def test_ItemBasedRecommender_fit(fx_user_item_matrix_toy_np):
    rec = ItemBasedRecommender()
    assert rec == rec.fit(fx_user_item_matrix_toy_np)


def test_ItemBasedRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        ItemBasedRecommender().fit("cat")
