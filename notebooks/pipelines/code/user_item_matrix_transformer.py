
import numpy as np
import pandas as pd

from pipeliner.recommendations.transformer import UserItemMatrixTransformer

data_types = {"user_id": str, "item_id": str, "rating": np.float64}

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    input_file = "user_item_ratings.csv"
    output_file = "user_item_matrix.csv"

    user_item_ratings = pd.read_csv(f"{base_dir}/{input_file}", dtype=data_types)
    transformer = UserItemMatrixTransformer()
    user_item_matrix = transformer.transform(user_item_ratings)

    user_item_matrix.to_csv(f"{base_dir}/{output_file}", header=True, index=False)
