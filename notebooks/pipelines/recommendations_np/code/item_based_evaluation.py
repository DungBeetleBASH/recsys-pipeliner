import json
import pathlib
import tarfile
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def accuracy_score(predictions, y_true):
    results = (y_true[..., None] == predictions).any(1)
    return results.astype(np.float32).mean().round(5)


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    accuracy = np.nan

    try:
        with tarfile.open(model_path) as tar:
            tar.extract("rec.joblib")

        model = joblib.load("rec.joblib")

        test_data = np.load(
            "/opt/ml/processing/test_data/test_data.npz"
        )["test_data"]

    except Exception as e:
        logging.info(e)

    print("test_data", test_data.shape)

    items = test_data[:100, 0]
    y_true = test_data[:100, 1]

    print("items", items.shape)
    print("y_true", y_true.shape)

    print("items[0]", items[0])
    print("y_true[0]", y_true[0])

    predictions = model.predict(items)

    print("predictions", predictions.shape)

    accuracy = accuracy_score(predictions, y_true)

    print(f"accuracy: {accuracy}")

    report_dict = {
        "regression_metrics": {
            "accuracy": {"value": accuracy},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
