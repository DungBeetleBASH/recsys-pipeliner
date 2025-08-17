import pytest
from recsys_pipeliner.algorithms.base import BaseRecommender


def test_BaseRecommender_fit():
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        BaseRecommender().fit([])


def test_BaseRecommender_predict():
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        BaseRecommender().predict([])


def test_BaseRecommender_recommend():
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        BaseRecommender().recommend([])