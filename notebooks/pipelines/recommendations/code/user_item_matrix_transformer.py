import numpy as np
import pandas as pd
import logging
from pipeliner.recommendations.transformer import UserItemMatrixTransformer

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    data_types = {"user_id": str, "item_id": str, "rating": np.float64}

    user_item_ratings = pd.read_csv(
        f"{base_dir}/input/data/user_item_ratings.csv",
        dtype=data_types,
        engine="python",
    )

    transformer = UserItemMatrixTransformer()
    user_item_matrix = transformer.transform(user_item_ratings)

    user_item_matrix.to_csv(
        f"{base_dir}/output/data/user_item_matrix.csv", header=True, index=False
    )
