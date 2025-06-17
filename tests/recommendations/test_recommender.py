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
        (["I00001"], ["I00002", "I00023", "I00003", "I00004", "I00006"]),
        (["I00002"], ["I00001", "I00023", "I00021", "I00022", "I00024"]),
        (["I00003"], ["I00004", "I00001", "I00005", "I00006", "I00008"]),
        (["I00004"], ["I00003", "I00001", "I00023", "I00024", "I00008"]),
        (["I00005"], ["I00006", "I00007", "I00003", "I00008", "I00010"]),
        (["I00006"], ["I00005", "I00003", "I00001", "I00010", "I00007"]),
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
        (
            ["I00001"],
            [
                1.0,
                0.95251,
                0.57236,
                0.54081,
                0.3821,
                0.47118,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.40651,
                0.32432,
                0.60379,
                0.42309,
            ],
        ),
        (
            ["I00002"],
            [
                0.95251,
                1.0,
                0.37515,
                0.34452,
                0.27094,
                0.3341,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.51885,
                0.41395,
                0.5802,
                0.38795,
            ],
        ),
        (
            ["I00003"],
            [
                0.57236,
                0.37515,
                1.0,
                0.94239,
                0.5389,
                0.49357,
                0.39618,
                0.48996,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.42094,
                0.33608,
            ],
        ),
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
        (["U00001"], ["I00007", "I00023", "I00008", "I00009", "I00021"]),
        (["U00002"], ["I00009", "I00001", "I00010", "I00011", "I00023"]),
        (["U00003"], ["I00011", "I00003", "I00012", "I00013", "I00014"]),
        (["U00004"], ["I00013", "I00005", "I00014", "I00015", "I00016"]),
        (["U00005"], ["I00015", "I00007", "I00016", "I00017", "I00018"]),
        (["U00006"], ["I00017", "I00009", "I00018", "I00019", "I00020"]),
    ],
)
def test_UserBasedRecommender_recommend(fx_user_item_matrix_toy, input, expected):
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


@pytest.mark.parametrize(
    "user_id, item_id, expected",
    [
        ("U00003", "I00007", 0.914590),
        ("U00003", "I00008", 0.619084),
        ("U00003", "I00009", 0.725423),
        ("U00003", "I00010", 0.325423),
        ("U00003", "I00011", 0.640000),
        ("U00003", "I00012", 0.240000),
    ],
)
def test_UserBasedRecommender_predict(
    fx_user_item_matrix_toy, user_id, item_id, expected
):
    item_ids = fx_user_item_matrix_toy.columns.to_numpy()
    user_ids = fx_user_item_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    user_encoder = LabelEncoder().fit(user_ids)
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = UserBasedRecommender().fit(matrix)

    item_idx = item_encoder.transform([item_id])[0]
    user_idx = user_encoder.transform([user_id])[0]

    prediction = rec.predict(user_idx, item_idx)

    np.testing.assert_almost_equal(prediction, expected)


@pytest.mark.parametrize(
    "user_id, item_id",
    [
        ("U00003", "I00013"),
        ("U00003", "I00014"),
        ("U00003", "I00015"),
        ("U00003", "I00016"),
    ],
)
def test_UserBasedRecommender_predict_none(fx_user_item_matrix_toy, user_id, item_id):
    item_ids = fx_user_item_matrix_toy.columns.to_numpy()
    user_ids = fx_user_item_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    user_encoder = LabelEncoder().fit(user_ids)
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = UserBasedRecommender().fit(matrix)

    item_idx = item_encoder.transform([item_id])[0]
    user_idx = user_encoder.transform([user_id])[0]

    prediction = rec.predict(user_idx, item_idx)

    assert prediction is None


def test_UserBasedRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        UserBasedRecommender().fit("cat")


def test_ItemBasedRecommender_fit(fx_user_item_matrix_toy_np):
    rec = ItemBasedRecommender()
    assert rec == rec.fit(fx_user_item_matrix_toy_np)


def test_ItemBasedRecommender_fit_error():
    with pytest.raises(ValueError, match="Input should be scipy.sparse.sparray"):
        ItemBasedRecommender().fit("cat")


@pytest.mark.parametrize(
    "input, expected",
    [
        (["I00001"], ["I00002", "I00023", "I00003", "I00004", "I00006"]),
        (["I00002"], ["I00001", "I00023", "I00021", "I00022", "I00024"]),
        (["I00003"], ["I00004", "I00001", "I00005", "I00006", "I00008"]),
        (["I00004"], ["I00003", "I00001", "I00023", "I00024", "I00008"]),
        (["I00005"], ["I00006", "I00007", "I00003", "I00008", "I00010"]),
        (["I00006"], ["I00005", "I00003", "I00001", "I00010", "I00007"]),
        (["I00007"], ["I00008", "I00009", "I00005", "I00010", "I00012"]),
        (["I00008"], ["I00007", "I00005", "I00003", "I00012", "I00009"]),
        (["I00009"], ["I00010", "I00011", "I00007", "I00012", "I00014"]),
        (["I00010"], ["I00009", "I00007", "I00005", "I00014", "I00006"]),
    ],
)
def test_ItemBasedRecommender_recommend(fx_user_item_matrix_toy, input, expected):
    item_ids = fx_user_item_matrix_toy.columns.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedRecommender().fit(matrix)

    input_encoded = item_encoder.transform(input)
    predictions = rec.recommend(input_encoded)
    predictions_decoded = item_encoder.inverse_transform(predictions[0])
    np.testing.assert_array_equal(predictions_decoded, expected)


@pytest.mark.parametrize(
    "user_id, item_id, expected",
    [
        ("U00003", "I00007", 0.607503),
        ("U00003", "I00008", 0.718121),
        ("U00003", "I00009", 0.645280),
        ("U00003", "I00010", 0.718702),
        ("U00003", "I00011", 0.571700),
        ("U00003", "I00012", 0.543242),
        ("U00003", "I00013", 0.819996),
        ("U00003", "I00014", 0.819997),
        ("U00003", "I00015", 0.616000),
        ("U00003", "I00016", 0.616000),
    ],
)
def test_ItemBasedRecommender_predict(
    fx_user_item_matrix_toy, user_id, item_id, expected
):
    item_ids = fx_user_item_matrix_toy.columns.to_numpy()
    user_ids = fx_user_item_matrix_toy.index.to_numpy()
    item_encoder = LabelEncoder().fit(item_ids)
    user_encoder = LabelEncoder().fit(user_ids)
    matrix = sp.sparse.csr_array(fx_user_item_matrix_toy.to_numpy())
    rec = ItemBasedRecommender().fit(matrix)

    item_idx = item_encoder.transform([item_id])[0]
    user_idx = user_encoder.transform([user_id])[0]

    prediction = rec.predict(user_idx, item_idx)

    np.testing.assert_almost_equal(prediction, expected)
