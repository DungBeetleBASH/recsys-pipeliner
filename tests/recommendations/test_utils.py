import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from recsys_pipeliner.recommendations.utils import train_test_split


def test_train_test_split():
    # Create a sample DataFrame with dates
    base_date = datetime(2023, 1, 1)
    df = pd.DataFrame(
        {
            "user_id": [
                "user1",
                "user1",
                "user1",
                "user1",
                "user1",  # 5 interactions
                "user2",
                "user2",
                "user2",
                "user2",
                "user2",  # 5 interactions
                "user3",
                "user3",
                "user3",
                "user3",
                "user3",  # 5 interactions
                "user4",
                "user4",
                "user4",
                "user4",
            ],  # 4 interactions (should be excluded)
            "item_id": [
                "item1",
                "item2",
                "item3",
                "item4",
                "item5",
                "item1",
                "item2",
                "item3",
                "item4",
                "item5",
                "item1",
                "item2",
                "item3",
                "item4",
                "item5",
                "item1",
                "item2",
                "item3",
                "item4",
            ],
            "interactions": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
            "date": [base_date + timedelta(days=i) for i in range(19)],
        }
    )

    # Test with default parameters
    train_df, test_df = train_test_split(df)

    # Check that user4 is excluded (less than 5 interactions)
    np.testing.assert_array_equal(
        train_df["user_id"].unique(), ["user1", "user2", "user3"]
    )
    np.testing.assert_array_equal(
        test_df["user_id"].unique(), ["user1", "user2", "user3"]
    )

    # Check that ratings are normalized between 0 and 1
    assert train_df["rating"].min() >= 0
    assert train_df["rating"].max() <= 1

    # Test with custom parameters
    train_df_custom, test_df_custom = train_test_split(
        df, min_user_ratings=3, interaction_cap=3
    )

    # Check that user4 is now included (has 4 interactions)
    assert "user4" in train_df_custom["user_id"].unique()
