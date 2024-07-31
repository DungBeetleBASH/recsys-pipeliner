import numpy as np
import pandas as pd
import logging
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)

logging.basicConfig(level=logging.INFO)


def train_test_split(df):
    user_id_value_counts = df.user_id.value_counts()

    user_id_multiple_interaction = (
        user_id_value_counts[user_id_value_counts > 1]
        .index.to_series()
        .reset_index(drop=True)
    )
    train_test_users = set(user_id_multiple_interaction)

    train_test_interactions_df = df[df.user_id.isin(train_test_users)].sort_values(
        by=["user_id", "date"]
    )

    test_data_df = (
        train_test_interactions_df.reset_index()
        .groupby(["user_id"], as_index=False)
        .last()
        .set_index("index")
    )[["user_id", "item_id"]]
    test_data_df.index.names = [None]

    train_data_df = train_test_interactions_df[
        (
            train_test_interactions_df[["user_id", "item_id"]]
            .merge(
                test_data_df.drop_duplicates(),
                on=["user_id", "item_id"],
                how="left",
                indicator=True,
            )
            ._merge
            == "left_only"
        ).values
    ]

    train_test_indices = test_data_df.index.union(train_data_df.index)
    excluded_data_df = train_test_interactions_df.loc[
        ~train_test_interactions_df.index.isin(train_test_indices)
    ]

    train_data_ratings_df = (
        train_data_df[["user_id", "item_id", "count"]]
        .groupby(["user_id", "item_id"])
        .sum()
        .reset_index()
        .rename(columns={"count": "rating"})
    )
    train_data_ratings_df["rating"] = 1 + np.log10(train_data_ratings_df["rating"])
    train_data_ratings_df["rating"] = (
        train_data_ratings_df["rating"] / train_data_ratings_df["rating"].max()
    ).round(2)

    all_data_ratings_df = (
        df[["user_id", "item_id", "count"]]
        .groupby(["user_id", "item_id"])
        .sum()
        .reset_index()
        .rename(columns={"count": "rating"})
    )
    all_data_ratings_df["rating"] = 1 + np.log10(all_data_ratings_df["rating"])
    all_data_ratings_df["rating"] = (
        all_data_ratings_df["rating"] / all_data_ratings_df["rating"].max()
    ).round(2)

    return train_data_ratings_df, test_data_df, excluded_data_df, all_data_ratings_df


def create_user_item_matrix(df):
    transformer = UserItemMatrixTransformer()
    return transformer.transform(df)


def create_similarity_matrix(df, kind="user", metric="cosine"):
    transformer = SimilarityTransformer(kind=kind, metric=metric)
    return transformer.transform(df)


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    data_types = {"user_id": str, "item_id": str, "rating": np.float64}

    user_item_interactions_df = pd.read_csv(
        f"{base_dir}/input/data/user_item_interactions.csv",
        dtype=data_types,
        parse_dates=["date"],
        engine="python",
    )

    train_ratings, test_data, excluded_data, all_ratings = train_test_split(
        user_item_interactions_df
    )

    user_item_matrix = create_user_item_matrix(all_ratings)
    train_user_item_matrix = create_user_item_matrix(train_ratings)

    user_item_matrix.to_csv(
        f"{base_dir}/output/user_item_matrix/user_item_matrix.csv",
        header=True,
        index=True,
    )

    train_user_item_matrix.to_csv(
        f"{base_dir}/output/train_user_item_matrix/user_item_matrix.csv",
        header=True,
        index=True,
    )

    user_similarity_matrix = create_similarity_matrix(
        user_item_matrix, kind="user", metric="cosine"
    )

    train_user_similarity_matrix = create_similarity_matrix(
        train_user_item_matrix, kind="user", metric="cosine"
    )

    user_similarity_matrix.to_csv(
        f"{base_dir}/output/user_similarity_matrix/user_similarity_matrix.csv",
        header=True,
        index=True,
    )

    train_user_similarity_matrix.to_csv(
        f"{base_dir}/output/train_user_similarity_matrix/user_similarity_matrix.csv",
        header=True,
        index=True,
    )

    item_similarity_matrix = create_similarity_matrix(
        user_item_matrix, kind="item", metric="cosine"
    )

    train_item_similarity_matrix = create_similarity_matrix(
        train_user_item_matrix, kind="item", metric="cosine"
    )

    item_similarity_matrix.to_csv(
        f"{base_dir}/output/item_similarity_matrix/item_similarity_matrix.csv",
        header=True,
        index=True,
    )

    train_item_similarity_matrix.to_csv(
        f"{base_dir}/output/train_item_similarity_matrix/item_similarity_matrix.csv",
        header=True,
        index=True,
    )

    test_data.to_csv(
        f"{base_dir}/output/test/test.csv",
        header=True,
        index=True,
    )

    excluded_data.to_csv(
        f"{base_dir}/output/excluded/excluded.csv",
        header=True,
        index=True,
    )
