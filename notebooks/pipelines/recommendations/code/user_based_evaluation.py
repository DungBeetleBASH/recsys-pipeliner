import json
import pathlib
import tarfile
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import logging


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
)


def score_predictions(y_true, predictions):
    return np.array([1.0 if t in p else 0.0 for t, p in zip(y_true, predictions)])


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SM_INPUT_MODEL"),
    )

    args = parser.parse_args()

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/models")

    try:
        model = joblib.load("/opt/ml/models/rec.joblib")
    except Exception as e:
        logging.info(e)

    test_path = "/opt/ml/processing/test/test.csv"
    test_df = pd.read_csv(
        test_path, dtype={"user_id": str, "item_id": str}, engine="python"
    )

    y_true = test_df.item_id.to_numpy()
    predictions = model.predict(test_df.user_id)

    print("y_true", y_true.shape)
    print("predictions", predictions.shape)

    mse = model.score(predictions, y_true)

    print(f"mse: {mse}")

    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
