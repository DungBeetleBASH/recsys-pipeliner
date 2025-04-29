import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from pipeliner.recommendations.recommender import (
    ItemBasedRecommenderPandas,
    UserBasedRecommenderPandas,
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
def test_ItemBasedRecommenderPandas(
    fx_item_similarity_matrix, fx_user_item_matrix, input, output_shape
):
    X = (
        fx_item_similarity_matrix
        if isinstance(input, str)
        else (fx_item_similarity_matrix, fx_user_item_matrix)
    )
    rec = ItemBasedRecommenderPandas().fit(X)
    predictions = rec.predict(input)
    assert predictions.shape == output_shape


def test_ItemBasedRecommenderPandas_fit_error():
    with pytest.raises(ValueError):
        ItemBasedRecommenderPandas().fit("cat")


def test_ItemBasedRecommenderPandas_predict_error(fx_item_similarity_matrix):
    rec = ItemBasedRecommenderPandas().fit(fx_item_similarity_matrix)
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
def test_UserBasedRecommenderPandas(
    fx_user_similarity_matrix, fx_user_item_matrix, input, output_shape
):
    X = (fx_user_similarity_matrix, fx_user_item_matrix)
    rec = UserBasedRecommenderPandas().fit(X)
    predictions = rec.predict(input)
    assert predictions.shape == output_shape


@pytest.mark.parametrize(
    "input, predictions, expected",
    [
        (["U1002"], ["U1002"], 1.0),
        (["U1002", "U1003"], ["U1002", "U1005"], 0.5),
        (
            ["U1002", "U1003", "U1003", "U1004"],
            ["U1001", "U1003", "U1003", "U1004"],
            0.75,
        ),
        ([], [], np.nan),
    ],
)
def test_UserBasedRecommenderPandas_score(
    fx_user_similarity_matrix, fx_user_item_matrix, input, predictions, expected
):
    X = (fx_user_similarity_matrix, fx_user_item_matrix)
    rec = UserBasedRecommenderPandas().fit(X)
    score = rec.score(predictions, input)
    np.testing.assert_equal(score, expected)


def test_UserBasedRecommenderPandas_fit_error():
    with pytest.raises(ValueError):
        UserBasedRecommenderPandas().fit("cat")


def test_UserBasedRecommenderPandas_predict_error(
    fx_user_similarity_matrix, fx_user_item_matrix
):
    X = (fx_user_similarity_matrix, fx_user_item_matrix)
    rec = UserBasedRecommenderPandas().fit(X)
    with pytest.raises(ValueError):
        rec.predict([1.3])


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], ["I00002", "I00012", "I00003", "I00011", "I00004"]),
        (["I00002"], ["I00001", "I00003", "I00004", "I00012", "I00005"]),
        (["I00003"], ["I00002", "I00004", "I00001", "I00005", "I00006"]),
        (["I00004"], ["I00003", "I00005", "I00002", "I00006", "I00001"]),
        (["I00005"], ["I00004", "I00006", "I00003", "I00007", "I00002"]),
        (["I00006"], ["I00005", "I00007", "I00004", "I00008", "I00003"]),
    ],
)
def test_SimilarityRecommender(fx_item_similarity_matrix_toy, input, expected):
    item_ids = fx_item_similarity_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder()
    item_ids_encoded = item_encoder.fit_transform(item_ids)
    item_similarity_matrix_np = fx_item_similarity_matrix_toy.to_numpy()
    item_similarity_matrix_np_sparse = sp.csr_array(item_similarity_matrix_np)

    rec = SimilarityRecommender(5).fit(item_similarity_matrix_np_sparse)
    predictions = rec.predict(item_encoder.transform(input))[0]
    predictions_decoded = item_encoder.inverse_transform(predictions)
    np.testing.assert_array_equal(predictions_decoded, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], [1.0, 0.56, 0.4, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16, 0.4, 0.56]),
        (["I00002"], [0.56, 1.0, 0.56, 0.4, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16, 0.4]),
        (["I00003"], [0.4, 0.56, 1.0, 0.56, 0.4, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16]),
    ],
)
def test_SimilarityRecommender_predict_proba(
    fx_item_similarity_matrix_toy, input, expected
):
    item_ids = fx_item_similarity_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder()
    item_ids_encoded = item_encoder.fit_transform(item_ids)
    item_similarity_matrix_np = fx_item_similarity_matrix_toy.to_numpy()
    item_similarity_matrix_np_sparse = sp.csr_array(item_similarity_matrix_np)

    rec = SimilarityRecommender(5).fit(item_similarity_matrix_np_sparse)
    probs = rec.predict_proba(item_encoder.transform(input)).toarray()[0].round(5)
    np.testing.assert_array_equal(probs, np.array(expected).astype(np.float32).round(5))


def test_SimilarityRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        SimilarityRecommender().fit("cat")
