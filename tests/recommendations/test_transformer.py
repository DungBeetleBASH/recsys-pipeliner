import pytest
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
    UserItemMatrixTransformerNP,
    SimilarityTransformerNP,
)
import numpy as np


@pytest.mark.parametrize(
    "transformer",
    [UserItemMatrixTransformer, SimilarityTransformer],
)
def test_empty_fit(fx_user_item_ratings, transformer):
    tf = transformer()

    assert tf.fit(fx_user_item_ratings) == tf


@pytest.mark.parametrize(
    "transformer",
    [UserItemMatrixTransformerNP],
)
def test_empty_fit_np(fx_user_item_ratings_np, transformer):
    user_item_ratings = fx_user_item_ratings_np["ratings"]
    tf = transformer()

    assert tf.fit(user_item_ratings) == tf


@pytest.mark.parametrize(
    "binary",
    [True, False],
)
def test_UserItemMatrixTransformer(fx_user_item_ratings, binary):
    tf = UserItemMatrixTransformer(binary=binary)
    user_item_matrix = tf.transform(fx_user_item_ratings)

    assert user_item_matrix.shape == (10, 44)


@pytest.mark.parametrize(
    "sparse",
    [True, False],
)
def test_UserItemMatrixTransformerNP(fx_user_item_ratings_np, sparse):
    user_item_ratings = fx_user_item_ratings_np["ratings"]
    tf = UserItemMatrixTransformerNP(sparse=sparse)
    user_item_matrix = tf.transform(user_item_ratings)

    assert user_item_matrix.shape == (10, 44)


def test_UserItemMatrixTransformer_equality(
    fx_user_item_ratings, fx_user_item_ratings_np
):
    # Create matrix using pandas transformer
    tf_pandas = UserItemMatrixTransformer()
    matrix_pandas = tf_pandas.transform(fx_user_item_ratings).to_numpy()

    # Create matrix using numpy transformer (dense)
    user_item_ratings = fx_user_item_ratings_np["ratings"]
    tf_numpy = UserItemMatrixTransformerNP(sparse=False)
    matrix_numpy = tf_numpy.transform(user_item_ratings)

    # Create matrix using numpy transformer (sparse)
    tf_numpy_sparse = UserItemMatrixTransformerNP(sparse=True)
    matrix_numpy_sparse = tf_numpy_sparse.transform(user_item_ratings).toarray()

    # Assert matrices are equal
    np.testing.assert_array_equal(matrix_pandas, matrix_numpy)
    np.testing.assert_array_equal(matrix_pandas, matrix_numpy_sparse)
    np.testing.assert_array_equal(matrix_numpy, matrix_numpy_sparse)


@pytest.mark.parametrize(
    "kind, metric",
    [(None, "cosine"), ("user", None)],
)
def test_SimilarityTransformer_error(kind, metric):
    with pytest.raises(ValueError):
        SimilarityTransformer(kind=kind, metric=metric)


@pytest.mark.parametrize(
    "kind, metric",
    [("user", "euclidean"), ("user", "dot")],
)
def test_SimilarityTransformer_not_implemented_error(fx_user_item_matrix, kind, metric):
    transformer = SimilarityTransformer(kind=kind, metric=metric)

    with pytest.raises(NotImplementedError):
        transformer.transform(fx_user_item_matrix)


@pytest.mark.parametrize(
    "kind, metric, normalise, expected_shape",
    [
        ("user", "cosine", False, (10, 10)),
        ("user", "cosine", True, (10, 10)),
        ("item", "cosine", False, (44, 44)),
    ],
)
def test_SimilarityTransformer(
    fx_user_item_matrix, kind, metric, normalise, expected_shape
):
    transformer = SimilarityTransformer(kind=kind, metric=metric, normalise=normalise)
    similarity_matrix = transformer.transform(fx_user_item_matrix)

    assert similarity_matrix.shape == expected_shape


@pytest.mark.parametrize(
    "kind, metric, normalise, expected_shape, sparse",
    [
        ("user", "cosine", False, (10, 10), False),
        ("user", "cosine", True, (10, 10), False),
        ("item", "cosine", False, (44, 44), False),
        ("user", "cosine", False, (10, 10), True),
        ("user", "cosine", True, (10, 10), True),
    ],
)
def test_SimilarityTransformerNP(
    fx_user_item_matrix_np,
    fx_user_item_matrix_sp,
    kind,
    metric,
    normalise,
    expected_shape,
    sparse,
):
    # Select appropriate input matrix based on sparse parameter
    input_matrix = fx_user_item_matrix_sp if sparse else fx_user_item_matrix_np
    if kind == "item":
        input_matrix = input_matrix.T

    transformer = SimilarityTransformerNP(metric=metric, normalise=normalise)
    similarity_matrix = transformer.transform(input_matrix)

    assert similarity_matrix.shape == expected_shape


@pytest.mark.parametrize(
    "kind, normalise, sparse_input, sparse_output",
    [
        ("user", False, False, False),
        ("user", True, False, False),
        ("item", False, False, False),
        ("item", True, False, False),
        ("user", False, True, False),
        ("user", True, True, False),
        ("item", False, True, False),
        ("item", True, True, False),
        ("user", False, False, True),
        ("user", True, False, True),
        ("item", False, False, True),
        ("item", True, False, True),
        ("user", False, True, True),
        ("user", True, True, True),
        ("item", False, True, True),
    ],
)
def test_SimilarityTransformer_equality(
    fx_user_item_matrix,
    fx_user_item_matrix_np,
    fx_user_item_matrix_sp,
    kind,
    normalise,
    sparse_input,
    sparse_output,
):
    user_item_matrix_pd = fx_user_item_matrix
    if sparse_input:
        user_item_matrix_np = fx_user_item_matrix_sp
    else:
        user_item_matrix_np = fx_user_item_matrix_np
    if kind == "item":
        user_item_matrix_np = user_item_matrix_np.T

    # Create transformers
    tf_pd = SimilarityTransformer(kind=kind, metric="cosine", normalise=normalise)
    tf_np = SimilarityTransformerNP(
        metric="cosine", normalise=normalise, sparse=sparse_output
    )

    # Create similarity matrix using pandas transformer
    similarity_matrix_pd = tf_pd.transform(user_item_matrix_pd).to_numpy()

    # Create similarity matrix using numpy transformer
    similarity_matrix_np = tf_np.transform(user_item_matrix_np)

    if sparse_output:
        similarity_matrix_np = similarity_matrix_np.toarray()

    # Assert matrices are equal
    np.testing.assert_array_almost_equal(
        similarity_matrix_pd, similarity_matrix_np, decimal=6
    )
