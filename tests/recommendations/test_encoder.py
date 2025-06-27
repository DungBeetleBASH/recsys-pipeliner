import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from recsys_pipeliner.recommendations.encoder import encode_labels


def test_encode_labels():
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "user_id": ["user1", "user2", "user1", "user3"],
            "item_id": ["item1", "item2", "item1", "item3"],
            "rating": [1.0, 0.5, 0.8, 0.3],
        }
    )

    # Test with default column names
    encoded_df, user_encoder, item_encoder = encode_labels(df)

    # Check that the encoders are LabelEncoder instances
    assert isinstance(user_encoder, LabelEncoder)
    assert isinstance(item_encoder, LabelEncoder)

    # Check that the encoded values are integers
    assert encoded_df["user_id"].dtype == np.int64
    assert encoded_df["item_id"].dtype == np.int64

    # Check that the encoding is consistent
    assert encoded_df["user_id"].nunique() == 3  # 3 unique users
    assert encoded_df["item_id"].nunique() == 3  # 3 unique items

    # Check that the original values can be recovered
    assert set(user_encoder.inverse_transform(encoded_df["user_id"])) == {
        "user1",
        "user2",
        "user3",
    }
    assert set(item_encoder.inverse_transform(encoded_df["item_id"])) == {
        "item1",
        "item2",
        "item3",
    }

    # Test with custom column names
    df_custom = pd.DataFrame(
        {
            "user": ["user1", "user2", "user1", "user3"],
            "item": ["item1", "item2", "item1", "item3"],
            "rating": [1.0, 0.5, 0.8, 0.3],
        }
    )

    encoded_df_custom, user_encoder_custom, item_encoder_custom = encode_labels(
        df_custom, user="user", item="item"
    )

    # Check that the custom column names are used
    assert "user" in encoded_df_custom.columns
    assert "item" in encoded_df_custom.columns
    assert "user_id" not in encoded_df_custom.columns
    assert "item_id" not in encoded_df_custom.columns

    # Check that the encoding is consistent with custom names
    assert encoded_df_custom["user"].nunique() == 3
    assert encoded_df_custom["item"].nunique() == 3
