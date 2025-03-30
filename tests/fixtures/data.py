import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    UserItemMatrixTransformerNP,
    SimilarityTransformer,
    UserItemMatrixTransformerNP,
    SimilarityTransformerNP,
)


@pytest.fixture
def fx_user_item_ratings():
    yield pd.read_csv(
        "tests/test_data/user_item_ratings.csv",
        dtype={"user_id": str, "item_id": str, "rating": np.float32},
    )


@pytest.fixture
def fx_user_item_ratings_toy():
    yield pd.read_csv(
        "tests/test_data/user_item_ratings_toy.csv",
        dtype={"user_id": str, "item_id": str, "rating": np.float32},
        header=0,
    )


@pytest.fixture
def fx_user_item_ratings_toy_np(fx_user_item_ratings_toy):
    user_item_ratings = fx_user_item_ratings_toy.copy()
    user_item_ratings["user_id"] = LabelEncoder().fit_transform(user_item_ratings["user_id"])
    user_item_ratings["item_id"] = LabelEncoder().fit_transform(user_item_ratings["item_id"])
    yield user_item_ratings.to_numpy()


@pytest.fixture
def fx_user_item_matrix_toy():
    yield pd.read_csv(
        "tests/test_data/user_item_matrix_toy.csv",
        header=0,
        index_col=['user_id'],
    ).astype(np.float32)


@pytest.fixture
def fx_user_item_matrix_toy_np(fx_user_item_ratings_toy_np):
    yield UserItemMatrixTransformerNP().transform(fx_user_item_ratings_toy_np)


@pytest.fixture
def fx_item_similarity_matrix_toy():
    yield pd.read_csv(
        "tests/test_data/item_similarity_matrix_toy.csv",
        header=0,
        index_col=['item_id'],
    ).astype(np.float32)


@pytest.fixture
def fx_user_similarity_matrix_toy():
    yield pd.read_csv(
        "tests/test_data/user_similarity_matrix_toy.csv",
        header=0,
        index_col=['user_id'],
    ).astype(np.float32)


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


@pytest.fixture
def fx_user_item_matrix_np(fx_user_item_ratings_np):
    user_item_ratings = fx_user_item_ratings_np["ratings"]
    yield UserItemMatrixTransformerNP().transform(user_item_ratings)


@pytest.fixture
def fx_user_similarity_matrix_np(fx_user_item_matrix_np):
    yield SimilarityTransformerNP().transform(fx_user_item_matrix_np)


@pytest.fixture
def fx_item_similarity_matrix_np(fx_user_item_matrix_np):
    yield SimilarityTransformerNP().transform(fx_user_item_matrix_np)
