import pytest
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)
import scipy.sparse as sp


def test_empty_fit(
    fx_user_item_ratings_toy_np,
    fx_user_item_matrix_toy_np,
):
    user_item_matrix_tf_np = UserItemMatrixTransformer()
    similarity_matrix_tf_np = SimilarityTransformer()

    assert (
        user_item_matrix_tf_np.fit(fx_user_item_ratings_toy_np)
        == user_item_matrix_tf_np
    )
    assert (
        similarity_matrix_tf_np.fit(fx_user_item_matrix_toy_np)
        == similarity_matrix_tf_np
    )


def test_UserItemMatrixTransformer(fx_user_item_ratings_toy_np):
    tf = UserItemMatrixTransformer()
    user_item_matrix = tf.transform(fx_user_item_ratings_toy_np)

    assert user_item_matrix.shape == (12, 12)
    assert isinstance(user_item_matrix, sp.csr_array)


def test_SimilarityTransformer_error():
    with pytest.raises(ValueError):
        SimilarityTransformer(metric=None)


def test_SimilarityTransformer_input_error():
    transformer = SimilarityTransformer()
    with pytest.raises(ValueError, match="Input must be a scipy.sparse.sparray"):
        transformer.transform([])


@pytest.mark.parametrize(
    "metric",
    [("euclidean"), ("dot")],
)
def test_SimilarityTransformer_not_implemented_error(
    fx_user_item_matrix_toy_np, metric
):
    transformer = SimilarityTransformer(metric=metric)

    with pytest.raises(NotImplementedError):
        transformer.transform(fx_user_item_matrix_toy_np)


@pytest.mark.parametrize(
    "kind, metric, expected_shape",
    [
        ("user", "cosine", (10, 10)),
        ("item", "cosine", (44, 44)),
        ("user", "cosine", (10, 10)),
    ],
)
def test_SimilarityTransformer(
    fx_user_item_matrix_np,
    kind,
    metric,
    expected_shape,
):
    # Select appropriate input matrix based on sparse parameter
    input_matrix = fx_user_item_matrix_np
    if kind == "item":
        input_matrix = input_matrix.T

    transformer = SimilarityTransformer(metric=metric)
    similarity_matrix = transformer.transform(input_matrix)

    assert similarity_matrix.shape == expected_shape
