import argparse
import os
import pandas as pd
import scipy as sp
import joblib
import logging
from pipeliner.recommendations.recommender import SimilarityRecommenderNP

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))

    args = parser.parse_args()

    logging.info(f"SM_MODEL_DIR: {args.model_dir}")
    logging.info(f"SM_INPUT_DIR: {args.input}")

    item_similarity_matrix = sp.sparse.load_npz(
        f"{args.input}/data/item_similarity_matrix/item_similarity_matrix.npz"
    )

    rec = SimilarityRecommenderNP(5).fit(item_similarity_matrix)

    joblib.dump(rec, os.path.join(args.model_dir, "rec.joblib"))
