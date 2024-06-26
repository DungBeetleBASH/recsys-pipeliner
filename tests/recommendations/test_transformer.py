import pytest
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize(
    "transformer",
    [UserItemMatrixTransformer, SimilarityTransformer],
)
def test_empty_fit(fx_user_item_ratings, transformer):
    tf = transformer()

    assert tf.fit(fx_user_item_ratings) == tf


@pytest.mark.parametrize(
    "binary",
    [True, False],
)
def test_UserItemMatrixTransformer(fx_user_item_ratings, binary):
    tf = UserItemMatrixTransformer(binary=binary)
    user_item_matrix = tf.transform(fx_user_item_ratings)

    assert user_item_matrix.shape == (10, 44)


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
def test_SimilarityTransformer_not_implemented_error(
    fx_user_item_ratings, kind, metric
):
    pipe = Pipeline(
        [
            ("user_item", UserItemMatrixTransformer()),
            ("similarity", SimilarityTransformer(kind=kind, metric=metric)),
        ]
    )
    with pytest.raises(NotImplementedError):
        pipe.transform(fx_user_item_ratings)


@pytest.mark.parametrize(
    "kind, metric, normalise, expected_shape",
    [
        ("user", "cosine", False, (10, 10)),
        ("user", "cosine", True, (10, 10)),
        ("item", "cosine", False, (44, 44)),
    ],
)
def test_SimilarityTransformer(
    fx_user_item_ratings, kind, metric, normalise, expected_shape
):
    pipe = Pipeline(
        [
            ("user_item", UserItemMatrixTransformer()),
            (
                "similarity",
                SimilarityTransformer(kind=kind, metric=metric, normalise=normalise),
            ),
        ]
    )

    similarity_matrix = pipe.transform(fx_user_item_ratings)

    assert similarity_matrix.shape == expected_shape
