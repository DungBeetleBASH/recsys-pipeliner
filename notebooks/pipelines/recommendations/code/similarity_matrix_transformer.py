
import numpy as np
import pandas as pd
import argparse
from pipeliner.recommendations.transformer import SimilarityTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str, default="user")
    parser.add_argument("--metric", type=str, default="cosine")
    args = parser.parse_args()
    
    base_dir = "/opt/ml/processing"

    user_item_matrix = pd.read_csv(f"{base_dir}/input/data/user_item_matrix.csv", dtype=np.float64)
    
    transformer = SimilarityTransformer(kind=args.kind, metric=args.metric)
    similarity_matrix = transformer.transform(user_item_matrix)

    similarity_matrix.to_csv(f"{base_dir}/output/data/{args.kind}_similarity_matrix.csv", header=True, index=False)
