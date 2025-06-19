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
                0.51367,
                0.0,
                0.22995,
                0.53716,
                0.0,
                0.0,
                0.22674,
                0.0,
                0.2145,
                0.17409,
                0.0,
                0.47624,
                0.0,
                0.1406,
                0.67963,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.20723,
                0.60541,
            ],
        ),
        (
            ["I00002"],
            [
                0.51367,
                1.0,
                0.0,
                0.0,
                0.41734,
                0.0,
                0.38825,
                0.44999,
                0.66082,
                0.33426,
                0.0,
                0.03453,
                0.14172,
                0.29601,
                0.56991,
                0.42668,
                0.11033,
                0.35573,
                0.4573,
                0.10986,
                0.07654,
                0.21246,
                0.49644,
                0.31098,
            ],
        ),
        (
            ["I00003"],
            [
                0.0,
                0.0,
                1.0,
                0.44982,
                0.0,
                0.599,
                0.0,
                0.0,
                0.0,
                0.34548,
                0.79341,
                0.48373,
                0.0,
                0.64103,
                0.22006,
                0.41281,
                0.4786,
                0.52415,
                0.27733,
                0.0,
                0.42391,
                0.58545,
                0.34879,
                0.16982,
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
        (["U00001"], ["I00016", "I00008", "I00002", "I00019", "I00001"]),
        (["U00002"], ["I00021", "I00016", "I00011", "I00008", "I00010"]),
        (["U00003"], ["I00015", "I00022", "I00012", "I00018", "I00010"]),
        (["U00004"], ["I00005", "I00014", "I00002", "I00023", "I00021"]),
        (["U00005"], ["I00003", "I00024", "I00012", "I00011", "I00023"]),
        (["U00006"], ["I00018", "I00015", "I00024", "I00011", "I00002"]),
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
        ("U00003", "I00007", 0.67),
        ("U00003", "I00008", 0.713238),
        ("U00003", "I00009", 0.87),
        ("U00003", "I00010", 0.524629),
        ("U00003", "I00011", 0.591182),
        ("U00003", "I00012", 0.634922),
        ("U00003", "I00013", 0.519954),
        ("U00003", "I00014", 0.642335),
        ("U00003", "I00015", 0.648502),
        ("U00003", "I00016", 0.72047),
    ],
)
def test_UserBasedRecommender_predict(fx_user_item_matrix_toy, user_id, item_id, expected):
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
        ("U00003", "I00007", 0.756353),
        ("U00003", "I00008", 0.836384),
        ("U00003", "I00009", 0.820002),
        ("U00003", "I00010", 0.836961),
        ("U00003", "I00011", 0.891748),
        ("U00003", "I00012", 0.721764),
        ("U00003", "I00013", 0.849971),
        ("U00003", "I00014", 0.817382),
        ("U00003", "I00015", 0.761754),
        ("U00003", "I00016", 0.812887),
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
