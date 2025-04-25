import pytest
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformerPandas,
    SimilarityTransformerPandas,
    UserItemMatrixTransformer,
    SimilarityTransformer,
)
import numpy as np
import scipy.sparse as sp


def test_empty_fit(
    fx_user_item_ratings_toy,
    fx_user_item_matrix_toy,
    fx_user_item_ratings_toy_np,
    fx_user_item_matrix_toy_np,
):
    user_item_matrix_tf = UserItemMatrixTransformerPandas()
    similarity_matrix_tf = SimilarityTransformerPandas()
    user_item_matrix_tf_np = UserItemMatrixTransformer()
    similarity_matrix_tf_np = SimilarityTransformer()

    assert user_item_matrix_tf.fit(fx_user_item_ratings_toy) == user_item_matrix_tf
    assert similarity_matrix_tf.fit(fx_user_item_matrix_toy) == similarity_matrix_tf
    assert (
        user_item_matrix_tf_np.fit(fx_user_item_ratings_toy_np)
        == user_item_matrix_tf_np
    )
    assert (
        similarity_matrix_tf_np.fit(fx_user_item_matrix_toy_np)
        == similarity_matrix_tf_np
    )


@pytest.mark.parametrize(
    "binary",
    [True, False],
)
def test_UserItemMatrixTransformerPandas(
    fx_user_item_ratings_toy, fx_user_item_matrix_toy, binary
):
    expected_matrix = fx_user_item_matrix_toy.copy()
    if binary is True:
        expected_matrix = (expected_matrix >= 0.5).astype(np.int32)
    else:
        expected_matrix = expected_matrix.fillna(0.0).astype(np.float32)

    tf = UserItemMatrixTransformerPandas(binary=binary)
    user_item_matrix = tf.transform(fx_user_item_ratings_toy)

    # Assert matrices are equal
    np.testing.assert_array_equal(
        user_item_matrix.to_numpy(), expected_matrix.to_numpy()
    )


def test_UserItemMatrixTransformer(fx_user_item_ratings_toy_np):
    tf = UserItemMatrixTransformer()
    user_item_matrix = tf.transform(fx_user_item_ratings_toy_np)

    assert user_item_matrix.shape == (6, 6)
    assert isinstance(user_item_matrix, sp.csr_matrix)


def test_UserItemMatrixTransformerPandas_equality(
    fx_user_item_ratings_toy, fx_user_item_ratings_toy_np
):
    # Create matrix using pandas transformer
    tf_pandas = UserItemMatrixTransformerPandas()
    matrix_pandas = tf_pandas.transform(fx_user_item_ratings_toy).to_numpy()

    # Create matrix using numpy transformer (sparse)
    tf_numpy = UserItemMatrixTransformer()
    matrix_numpy = tf_numpy.transform(fx_user_item_ratings_toy_np).toarray()

    # Assert matrices are equal
    np.testing.assert_array_equal(matrix_pandas, matrix_numpy)


@pytest.mark.parametrize(
    "kind, metric",
    [(None, "cosine"), ("user", None)],
)
def test_SimilarityTransformerPandas_error(kind, metric):
    with pytest.raises(ValueError):
        SimilarityTransformerPandas(kind=kind, metric=metric)


@pytest.mark.parametrize(
    "kind, metric",
    [("user", "euclidean"), ("user", "dot")],
)
def test_SimilarityTransformerPandas_not_implemented_error(
    fx_user_item_matrix_toy, kind, metric
):
    transformer = SimilarityTransformerPandas(kind=kind, metric=metric)

    with pytest.raises(NotImplementedError):
        transformer.transform(fx_user_item_matrix_toy)


@pytest.mark.parametrize(
    "kind, metric, normalise, expected_shape",
    [
        ("user", "cosine", False, (6, 6)),
        ("user", "cosine", True, (6, 6)),
        ("item", "cosine", False, (6, 6)),
    ],
)
def test_SimilarityTransformerPandas(
    fx_user_item_matrix_toy, kind, metric, normalise, expected_shape
):
    transformer = SimilarityTransformerPandas(
        kind=kind, metric=metric, normalise=normalise
    )
    similarity_matrix = transformer.transform(fx_user_item_matrix_toy)

    assert similarity_matrix.shape == expected_shape


def test_SimilarityTransformer_error():
    with pytest.raises(ValueError):
        SimilarityTransformer(metric=None)


def test_SimilarityTransformer_input_error():
    transformer = SimilarityTransformer()
    with pytest.raises(ValueError, match="Input must be a scipy.sparse.spmatrix"):
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


@pytest.mark.parametrize(
    "kind",
    ["user", "item"],
)
def test_SimilarityTransformerPandas_equality(
    fx_user_item_matrix,
    fx_user_item_matrix_np,
    kind,
):
    """Test that SimilarityTransformerPandas and SimilarityTransformer produce the same results."""

    user_item_matrix = fx_user_item_matrix
    user_item_matrix_np = fx_user_item_matrix_np
    if kind == "item":
        user_item_matrix_np = user_item_matrix_np.T

    # Create transformers
    transformer = SimilarityTransformerPandas(kind=kind)
    transformer_np = SimilarityTransformer()

    # Transform data
    result = transformer.transform(user_item_matrix).to_numpy()
    result_np = transformer_np.transform(user_item_matrix_np).toarray()

    # Compare results
    np.testing.assert_array_almost_equal(result, result_np, decimal=6)
