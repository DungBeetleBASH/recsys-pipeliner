import pytest
import pandas as pd
import numpy as np

from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)


@pytest.fixture
def fx_user_item_ratings():
    yield pd.read_csv(
        "tests/test_data/user_item_ratings.csv",
        dtype={"user_id": str, "item_id": str},
    )


@pytest.fixture
def fx_user_item_ratings_np():
    yield np.load(
        "tests/test_data/user_item_ratings.npz",
    )


@pytest.fixture
def fx_user_item_matrix(fx_user_item_ratings):
    yield UserItemMatrixTransformer().transform(fx_user_item_ratings)


@pytest.fixture
def fx_user_similarity_matrix(fx_user_item_matrix):
    yield SimilarityTransformer(kind="user", metric="cosine", normalise=True).transform(
        fx_user_item_matrix
    )


@pytest.fixture
def fx_item_similarity_matrix(fx_user_item_matrix):
    yield SimilarityTransformer(kind="item", metric="cosine", normalise=True).transform(
        fx_user_item_matrix
    )
