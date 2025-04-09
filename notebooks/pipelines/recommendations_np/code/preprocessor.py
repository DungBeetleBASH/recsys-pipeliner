import numpy as np
import pandas as pd
import scipy as sp
import logging
import joblib
from pipeliner.recommendations.transformer import (
    UserItemMatrixTransformerNP,
    SimilarityTransformerNP,
)
from pipeliner.recommendations.encoder import encode_labels
from pipeliner.recommendations.utils import train_test_split

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    data_types = {"user_id": str, "item_id": str, "rating": np.float32}

    df = pd.read_csv(
        f"{base_dir}/input/data/user_item_interactions.csv.gz",
        compression="gzip",
        dtype=data_types,
        parse_dates=["date"],
    )

    df, user_encoder, item_encoder = encode_labels(df)

    train_data, test_data = train_test_split(df)

    user_item_ratings = train_data.to_numpy()

    user_item_matrix_transformer = UserItemMatrixTransformerNP()
    user_item_matrix = user_item_matrix_transformer.transform(user_item_ratings)

    item_similarity_transformer = SimilarityTransformerNP()
    item_similarity_matrix = item_similarity_transformer.transform(user_item_matrix.T)

    joblib.dump(user_encoder, f"{base_dir}/output/user_encoder/user_encoder.joblib")
    joblib.dump(item_encoder, f"{base_dir}/output/item_encoder/item_encoder.joblib")

    np.savez(
        f"{base_dir}/output/test_data/test_data.npz",
        test_data=test_data.to_numpy(),
    )

    sp.sparse.save_npz(
        f"{base_dir}/output/user_item_matrix/user_item_matrix.npz",
        user_item_matrix,
        compressed=True,
    )

    sp.sparse.save_npz(
        f"{base_dir}/output/item_similarity_matrix/item_similarity_matrix.npz",
        item_similarity_matrix,
        compressed=True,
    )
