import os
import joblib
import logging
from pipeliner.recommendations.recommender import SimilarityRecommender # noqa: F401

logging.basicConfig(level=logging.INFO)


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def predict_fn(input_data, model):
    logging.info(f"### predict_fn called ###")
    logging.info(f"input_data: {input_data}")
    logging.info(f"input_data type: {type(input_data)}")
    
    recommendations = model.recommend(input_data[0])
    
    logging.info(f"recommendations: {recommendations}")
    logging.info(f"### predict_fn end ###")
    return recommendations