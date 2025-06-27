import argparse
import os
import joblib
import logging
import numpy as np
from recsys_pipeliner.recommendations.recommender import SimilarityRecommender
from recsys_pipeliner.recommendations.transformer import (
    UserItemMatrixTransformer,
    SimilarityTransformer,
)

logging.basicConfig(level=logging.INFO)

user_item_ratings = np.array(
    [
        (0, 0, 5),
        (0, 1, 4),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 5),
        (1, 0, 3),
        (1, 1, 5),
        (1, 2, 4),
        (1, 3, 1),
        (1, 4, 1),
        (1, 5, 5),
        (2, 1, 3),
        (2, 2, 1),
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))

    args = parser.parse_args()

    logging.info(f"SM_MODEL_DIR: {args.model_dir}")
    logging.info(f"SM_INPUT_DIR: {args.input}")

    user_item_matrix_transformer = UserItemMatrixTransformer()
    user_item_matrix = user_item_matrix_transformer.transform(user_item_ratings)

    item_similarity_transformer = SimilarityTransformer()
    item_similarity_matrix = item_similarity_transformer.transform(user_item_matrix.T)

    rec = SimilarityRecommender(5).fit(item_similarity_matrix)

    joblib.dump(rec, os.path.join(args.model_dir, "model.joblib"))
