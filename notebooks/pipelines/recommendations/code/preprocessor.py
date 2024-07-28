import numpy as np
import pandas as pd
import logging
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    data_types = {"user_id": str, "item_id": str, "rating": np.float64}

    user_item_ratings = pd.read_csv(
        f"{base_dir}/input/data/user_item_ratings.csv",
        dtype=data_types,
        engine="python",
    )

    # TODO train/test split
    user_item_ratings.to_csv(
        f"{base_dir}/output/train/train.csv",
        header=True,
        index=False,
    )
    user_item_ratings.to_csv(
        f"{base_dir}/output/test/test.csv",
        header=True,
        index=False,
    )
    #

    user_item_matrix_transformer = UserItemMatrixTransformer()
    user_item_matrix = user_item_matrix_transformer.transform(user_item_ratings)

    user_item_matrix.to_csv(
        f"{base_dir}/output/user_item_matrix/user_item_matrix.csv",
        header=True,
        index=False,
    )

    user_similarity_matrix_transformer = SimilarityTransformer(
        kind="user", metric="cosine"
    )
    user_similarity_matrix = user_similarity_matrix_transformer.transform(
        user_item_matrix
    )

    user_similarity_matrix.to_csv(
        f"{base_dir}/output/user_similarity_matrix/user_similarity_matrix.csv",
        header=True,
        index=False,
    )

    item_similarity_matrix_transformer = SimilarityTransformer(
        kind="item", metric="cosine"
    )
    item_similarity_matrix = item_similarity_matrix_transformer.transform(
        user_item_matrix
    )

    item_similarity_matrix.to_csv(
        f"{base_dir}/output/item_similarity_matrix/item_similarity_matrix.csv",
        header=True,
        index=False,
    )
