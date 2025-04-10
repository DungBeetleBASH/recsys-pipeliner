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
    parser.add_argument("--model", type=str, default=os.environ.get("SM_INPUT_MODEL"))
    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))

    args = parser.parse_args()

    logging.info(f"args: {args}")

    for p in os.listdir("/opt/ml/processing"):
        print(f"/opt/ml/processing/{p}")
        if os.path.isdir(f"/opt/ml/processing/{p}"):
            for p2 in os.listdir(f"/opt/ml/processing/{p}"):
                print(f"/opt/ml/processing/{p}/{p2}")
        print("")


    try:
        model = joblib.load("/opt/ml/processing/models/rec.joblib")
    except Exception as e:
        logging.info(e)

    test_data = np.load(
        f"{args.input}/data/test_data/test_data.npz"
    )
    print("test_data", test_data.shape)

    items = test_data[:, 0]
    y_true = test_data[:, 1]

    predictions = model.predict(items)

    print("y_true", y_true.shape)
    print("predictions", predictions.shape)

    accuracy = model.score(predictions, y_true)

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
