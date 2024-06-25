import pytest
import pandas as pd

@pytest.fixture
def fx_user_item_ratings():
    yield pd.read_csv("tests/test_data/test_user_item_ratings.csv", dtype={"user_id": str, "item_id": str})