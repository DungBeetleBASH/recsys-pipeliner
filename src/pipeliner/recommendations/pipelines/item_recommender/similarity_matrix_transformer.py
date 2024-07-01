
import numpy as np
import pandas as pd

from pipeliner.recommendations.transformer import SimilarityTransformer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kind", type=str, default="user")
parser.add_argument("--metric", type=str, default="cosine")
args = parser.parse_args()

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    input_file = "user_item_matrix.csv"
    output_file = "item_similarity_matrix.csv"

    user_item_matrix = pd.read_csv(f"{base_dir}/{input_file}", dtype=np.float64)
    transformer = SimilarityTransformer(kind=args.kind, metric=args.metric)
    similarity_matrix = transformer.transform(user_item_matrix)

    similarity_matrix.to_csv(f"{base_dir}/{output_file}", header=True, index=False)
