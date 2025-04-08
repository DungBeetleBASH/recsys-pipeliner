import numpy as np
import pandas as pd
import logging
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)

logging.basicConfig(level=logging.INFO)


def train_test_split(df):
    MIN_USER_RATINGS = 5

    user_id_value_counts = df.user_id.value_counts()

    excluded_users = (
        user_id_value_counts[user_id_value_counts < MIN_USER_RATINGS]
        .index.to_series()
        .reset_index(drop=True)
    )

    excluded_data_df = df[
        df.user_id.isin(excluded_users)
    ]

    test_train_data_df = df[
        ~df.user_id.isin(excluded_users)
    ].sort_values(by="date", ascending=True)

    test_data_df = (
        test_train_data_df.reset_index()
        .groupby(["user_id"], as_index=False)
        .last()
        .set_index("index")
        .sort_index()
    )[["user_id", "item_id"]]
    test_data_df.index.names = [None]

    train_data_df = (
        test_train_data_df[~test_train_data_df.index.isin(test_data_df.index)]
        .groupby(["user_id", "item_id"])
        .agg({"count": "sum"})
        .reset_index()
        .rename(columns={"count": "rating"})
    )

    train_data_df["rating"] = 1 + np.log10(train_data_df["rating"])
    train_data_df["rating"] = train_data_df["rating"] / train_data_df["rating"].max()

    return train_data_df, test_data_df, excluded_data_df


def create_user_item_matrix(df):
    transformer = UserItemMatrixTransformer()
    return transformer.transform(df)


def create_similarity_matrix(df, kind="user", metric="cosine"):
    transformer = SimilarityTransformer(kind=kind, metric=metric)
    return transformer.transform(df)


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    data_types = {"user_id": str, "item_id": str, "rating": np.float32}

    user_item_interactions_df = pd.read_csv(
        f"{base_dir}/data/user_item_interactions.csv.gz",
        compression="gzip",
        dtype=data_types,
    )

    train_data, test_data = train_test_split(
        user_item_interactions_df
    )

    train_user_item_matrix = create_user_item_matrix(train_data)

    train_user_item_matrix.to_csv(
        f"{base_dir}/output/user_item_matrix/user_item_matrix.csv",
        header=True,
        index=True,
    )

    train_item_similarity_matrix = create_similarity_matrix(
        train_user_item_matrix, kind="item", metric="cosine"
    )

    train_item_similarity_matrix.to_csv(
        f"{base_dir}/output/item_similarity_matrix/item_similarity_matrix.csv",
        header=True,
        index=True,
    )

    test_data.to_csv(
        f"{base_dir}/output/test/test.csv",
        header=True,
        index=True,
    )
